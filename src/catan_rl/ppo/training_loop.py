"""End-to-end PPO training loop for the v2 codebase (Phase 10).

Wires together every Phase 0-9 module into one resumable loop:

* :class:`~catan_rl.policy.network.CatanPolicy` + ``set_board_geometry``
  (Phase 2 architecture).
* :class:`~catan_rl.ppo.vec_env.SerialVecEnv` (Phase 5) seeded once at
  construction; per-env auto-reset Generators captured by the
  checkpoint (Phase 8).
* :class:`~catan_rl.ppo.buffer.CompositeRolloutBuffer` (Phase 3) sized
  to ``rollout.n_envs * rollout.n_steps``.
* :class:`~catan_rl.ppo.game_manager.RolloutCollector` (Phase 5)
  drives one rollout's worth of transitions.
* :class:`~catan_rl.ppo.trainer.PPOTrainer` (Phase 4) does the SGD.
* :class:`~catan_rl.selfplay.league.League` (Phase 6) appends a
  snapshot every ``add_snapshot_every_n_updates`` updates; the
  opponent mix is locked at construction time (vec env can't swap
  opponents mid-rollout in Phase 6).
* :class:`~catan_rl.checkpoint.CheckpointManager` (Phase 8) saves +
  prunes every ``checkpoint.save_every_updates`` updates; resume
  picks up from ``mgr.latest()`` with the canonical apply order
  (policy → optimizer → league → vec_env → RNG).
* :class:`~catan_rl.eval.harness.EvalHarness` (Phase 7) runs
  symmetric-seat WR every ``eval.eval_every_updates`` updates.
* TensorBoard scalars are written to ``run_dir/tb/`` so the operator
  can monitor live.

piKL (Phase 9) primitives are intentionally NOT wired here — the
``PiKLConfig`` is not yet a field on :class:`TrainConfig`, and the
trainer's ``_sgd_step`` doesn't accept an anchor argument. The wiring
ships in a follow-up once the BC anchor checkpoint exists.

The loop is constructed by :func:`build_training_state` (returns a
:class:`TrainingState` carrying every owned object — testable in
isolation) and driven by :func:`run_training_loop` (the actual
update loop). The script entry point in ``scripts/train.py`` calls
:func:`run_training` which composes both.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    load_checkpoint,
)
from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.harness import EvalHarness
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.policy.obs_schema import N_DEV_TYPES
from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.buffer import CompositeRolloutBuffer
from catan_rl.ppo.game_manager import RolloutCollector
from catan_rl.ppo.trainer import PPOTrainer
from catan_rl.ppo.vec_env import (
    SerialVecEnv,
    mask_spec_from_env,
    obs_spec_from_env,
)
from catan_rl.selfplay.league import League

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Bundle of every owned object the training loop touches.

    Constructed by :func:`build_training_state`. Holding everything in
    one place makes the resume path testable without reaching into the
    script entry point.
    """

    cfg: TrainConfig
    device: torch.device
    policy: CatanPolicy
    optimizer: torch.optim.Optimizer
    vec_env: SerialVecEnv
    buffer: CompositeRolloutBuffer
    collector: RolloutCollector
    trainer: PPOTrainer
    league: League
    ckpt_mgr: CheckpointManager
    eval_harness: EvalHarness
    rng: np.random.Generator
    """RNG used for minibatch shuffle ordering inside the PPO update."""

    update_idx: int = 0
    """Next update to execute (i.e., the loop runs
    ``range(update_idx, n_updates_total)``)."""

    global_step: int = 0
    """Total env-step count consumed by rollouts so far."""

    n_updates_total: int = 0
    """Total updates the loop is configured to run. Cached from cfg
    so the integer doesn't have to be re-derived inside the loop."""

    # Trailing rollout state — carried across :meth:`RolloutCollector.collect`
    # calls. ``None`` before the first ``reset_all``.
    obs: dict[str, np.ndarray] | None = None
    masks: dict[str, np.ndarray] | None = None

    # TB writer is created lazily and closed by the loop on exit.
    tb_writer: Any = None

    # Metrics JSONL writer — append-only file handle under
    # ``run_dir/metrics.jsonl``. One line per PPO update with
    # ``kind="train"``, plus one line per opponent per eval round with
    # ``kind="eval"``. Open lazily and closed by the loop on exit.
    metrics_fh: io.TextIOBase | None = None
    metrics_path: Path | None = None

    # Resume-attribution metadata.
    resumed_from: Path | None = None
    metadata_extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _resolve_device_to_torch(label: str) -> torch.device:
    """Convert the resolved-device label (``"cpu"`` / ``"mps"`` / ``"cuda"``)
    to a ``torch.device``. Centralised so callers don't repeat the
    string-vs-device dance."""
    return torch.device(label)


def _build_env_kwargs_list(
    cfg: TrainConfig, *, opponent_mix: Iterable[str]
) -> list[dict[str, Any]]:
    """One CatanEnv-kwargs dict per env, threaded with the league's
    opponent mix.

    Phase 6 still raises ``NotImplementedError`` when the league
    samples ``"snapshot"`` against a non-empty pool, so the loop's
    first rollout only sees ``"random"`` / ``"heuristic"`` opponents.
    The training_loop module passes the mix through unchanged so a
    future Phase that wires snapshot opponents (Phase 11+) doesn't
    need to touch the loop.
    """
    return [{"opponent_type": opp, "max_turns": cfg.rollout.max_turns} for opp in opponent_mix]


def build_training_state(
    cfg: TrainConfig,
    *,
    run_dir: Path,
    device_label: str,
    logger: logging.Logger | None = None,
) -> TrainingState:
    """Construct every object needed to run the loop.

    Idempotent except for filesystem side-effects (creates the
    checkpoint directory under ``run_dir``). Safe to call once per
    process; calling twice with the same ``run_dir`` reuses any
    existing checkpoints.

    ``device_label`` is the resolved label from
    :func:`catan_rl.ppo.arguments.resolve_device` — pass the already-
    resolved value rather than ``cfg.device`` so a ``"auto"`` request
    doesn't get re-resolved here against a different host's
    capabilities.
    """
    log = logger or logging.getLogger("catan_rl.train")
    device = _resolve_device_to_torch(device_label)

    # Policy + board geometry. ``set_board_geometry`` must run BEFORE
    # any forward pass — the axial PE + GNN adjacency tables are zero
    # placeholders at construction time.
    policy = CatanPolicy()
    geom = build_geometry()
    policy.set_board_geometry(geom.as_dict_of_tensors())
    policy = policy.to(device)
    log.info("policy: %d parameters", policy.num_parameters())

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.optimizer.lr_start,
        weight_decay=cfg.optimizer.weight_decay,
    )

    league = League(cfg.league)
    # Initial opponent mix at construction. When self-play is on
    # (snapshot_weight>0) the training loop re-draws the per-env assignment
    # each rollout via vec_env.set_opponents (US3 / T026), so this is just the
    # first rollout's opponents; otherwise it stands for the whole run.
    opp_rng = np.random.default_rng(cfg.seed)
    opp_mix = league.build_env_opponent_mix(n_envs=cfg.rollout.n_envs, rng=opp_rng)
    vec_env = SerialVecEnv(
        env_kwargs_list=_build_env_kwargs_list(cfg, opponent_mix=opp_mix),
        seed=cfg.seed,
    )

    # Use a throwaway env exclusively for spec discovery so the live
    # vec_env's internal state isn't perturbed by ``env.reset(seed=0)``
    # inside ``mask_spec_from_env``. Reviewer-caught HIGH: spec
    # discovery should not have side effects on a rollout-bound env.
    spec_env = CatanEnv(opponent_type="random", max_turns=cfg.rollout.max_turns)
    try:
        obs_spec = obs_spec_from_env(spec_env)
        mask_spec = mask_spec_from_env(spec_env)
    finally:
        spec_env.close()
    # Allocate belief-target storage only when the belief head exists AND its
    # loss is active (coef > 0). Once allocated, game_manager MUST supply a
    # target on every add() (the buffer enforces this), so gate it tightly.
    belief_target_dim = (
        N_DEV_TYPES if (policy.belief_head is not None and cfg.loss.belief_coef > 0.0) else None
    )
    buffer = CompositeRolloutBuffer(
        n_steps=cfg.rollout.n_steps,
        n_envs=cfg.rollout.n_envs,
        obs_spec=obs_spec,
        mask_spec=mask_spec,
        belief_target_dim=belief_target_dim,
    )
    # Wire the league into the collector only when PFSP is on, so it attributes
    # game outcomes to opponents (the win-rate signal). Off → no attribution,
    # byte-identical rollout.
    collector = RolloutCollector(
        vec_env=vec_env,
        policy=policy,
        buffer=buffer,
        device=device,
        league=league if cfg.league.pfsp_enabled else None,
    )
    trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=optimizer, device=device)

    # Checkpoint manager pins the directory; pruning is keep_last_n
    # under the embedded update_idx so NFS mtime drift doesn't bite.
    # Eagerly create the dir so an operator inspecting ``run_dir``
    # mid-launch sees the empty checkpoints/ directory (rather than
    # waiting for the first save_every_updates to materialise it).
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        keep_last_n=cfg.checkpoint.keep_last_n,
        save_optimizer_state=cfg.checkpoint.save_optimizer_state,
    )

    # Eval harness owns its own envs (one per opponent) — never shared
    # with the rollout vec env. ``n_games_per_seat = eval_games // 2``
    # because the harness plays both seats for each game_idx.
    n_per_seat = max(1, cfg.eval.eval_games // 2)
    eval_harness = EvalHarness(
        opponent_types=cfg.eval.eval_opponents,
        n_games_per_seat=n_per_seat,
        seed=cfg.seed,
        # Pin eval to CPU even when the learner trains on MPS/CUDA. Eval
        # runs the policy at batch=1 (sequential games), the one regime
        # where MPS is ~7-8x SLOWER than CPU. EvalHarness.run() moves the
        # policy to this device for the eval round and restores it after,
        # so training resumes on `device` untouched. (wall-clock audit)
        device=torch.device("cpu"),
        max_turns=cfg.rollout.max_turns,
    )

    n_updates_total = cfg.total_steps // (cfg.rollout.n_envs * cfg.rollout.n_steps)

    state = TrainingState(
        cfg=cfg,
        device=device,
        policy=policy,
        optimizer=optimizer,
        vec_env=vec_env,
        buffer=buffer,
        collector=collector,
        trainer=trainer,
        league=league,
        ckpt_mgr=ckpt_mgr,
        eval_harness=eval_harness,
        rng=np.random.default_rng(cfg.seed + 1),
        n_updates_total=n_updates_total,
    )
    return state


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


_RESUME_CRITICAL_FIELDS: tuple[tuple[str, ...], ...] = (
    ("rollout", "n_envs"),
    ("rollout", "n_steps"),
    ("rollout", "max_turns"),
    ("ppo", "batch_size"),
    ("gae", "gamma"),
    ("gae", "gae_lambda"),
)
"""Config fields where a resume-time mismatch silently corrupts
training. Architecture fields (``trunk_dim``, ``use_graph_encoder``,
...) live on the policy and torch raises a clean ``RuntimeError`` from
``load_state_dict`` for those. The fields here are the SLOW
silent-corruption cases — a different ``n_envs`` would only blow up
inside ``apply_to_vec_env`` (clear error there); a different ``gamma``
or ``gae_lambda`` would silently change the value targets going
forward."""


def _diff_resume_config(saved: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Return ``[<dotted-path>: saved=... live=...]`` lines for any
    :data:`_RESUME_CRITICAL_FIELDS` that differ between the saved and
    live configs. Empty list = clean resume."""
    out: list[str] = []
    for path in _RESUME_CRITICAL_FIELDS:
        s: Any = saved
        v: Any = live
        for key in path:
            s = s.get(key, None) if isinstance(s, dict) else None
            v = v.get(key, None) if isinstance(v, dict) else None
        if s != v:
            dotted = ".".join(path)
            out.append(f"{dotted}: saved={s!r} live={v!r}")
    return out


def maybe_resume_from_checkpoint(
    state: TrainingState,
    *,
    logger: logging.Logger | None = None,
) -> Path | None:
    """If ``state.ckpt_mgr.latest()`` exists, restore everything into
    ``state`` and return the checkpoint path. Otherwise ``None``.

    Apply order matches the
    :meth:`CheckpointPayload.apply_all` contract: policy → optimizer →
    league → vec_env → RNG. ``state.update_idx`` advances to the
    saved update + 1 so the loop picks up at the NEXT update.

    Before apply, compares the saved config against the live ``cfg``
    on a small set of resume-critical fields and emits a WARNING for
    each mismatch. Architecture / state-dict mismatches still fail
    loudly inside torch's ``load_state_dict``; this diff catches the
    silent-corruption cases (e.g., resumed with a different ``gamma``
    that quietly changes the value-target distribution).
    """
    log = logger or logging.getLogger("catan_rl.train")
    latest = state.ckpt_mgr.latest()
    if latest is None:
        return None
    payload: CheckpointPayload = load_checkpoint(latest, map_location=state.device)

    diffs = _diff_resume_config(payload.config, state.cfg.to_dict())
    for d in diffs:
        log.warning("resume config diff %s", d)

    payload.apply_all(
        policy=state.policy,
        optimizer=state.optimizer,
        league=state.league,
        vec_env=state.vec_env,
    )
    state.update_idx = payload.update_idx + 1
    state.global_step = payload.global_step
    state.resumed_from = latest
    log.info(
        "resumed from %s at update %d, global_step %d",
        latest,
        state.update_idx,
        state.global_step,
    )
    return latest


# ---------------------------------------------------------------------------
# TB
# ---------------------------------------------------------------------------


def _open_tb_writer(run_dir: Path) -> Any:
    """Open a tensorboard ``SummaryWriter`` under ``run_dir/tb/``.

    Imported lazily because ``tensorboard`` ships as part of ``torch``
    but pulls in protobuf which lights up Modal cold-start telemetry
    we don't need.
    """
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(run_dir / "tb"))


def _log_update_metrics(writer: Any, metrics: Any, *, global_step: int) -> None:
    """Write every scalar field on the ``UpdateMetrics`` dataclass to TB
    under ``train/<field>``. Unknown / non-scalar fields are skipped."""
    if writer is None:
        return
    for k, v in vars(metrics).items():
        if isinstance(v, int | float):
            writer.add_scalar(f"train/{k}", v, global_step)


def _log_eval_report(writer: Any, report: Any, *, global_step: int) -> None:
    """Log per-opponent WR + Wilson CI bounds + n_truncated."""
    if writer is None:
        return
    for res in report.results:
        opp = res.opponent_type
        writer.add_scalar(f"eval/wr_{opp}", res.wr, global_step)
        writer.add_scalar(f"eval/wr_{opp}_ci_low", res.ci.lower, global_step)
        writer.add_scalar(f"eval/wr_{opp}_ci_high", res.ci.upper, global_step)
        writer.add_scalar(f"eval/n_truncated_{opp}", res.n_truncated, global_step)


# ---------------------------------------------------------------------------
# Metrics JSONL (run_dir/metrics.jsonl)
# ---------------------------------------------------------------------------

#: Path component relative to ``run_dir``. Kept as a constant so
#: downstream tooling (notebooks, the dashboard) can import it.
METRICS_FILENAME = "metrics.jsonl"


def _open_metrics_writer(run_dir: Path) -> tuple[io.TextIOBase, Path]:
    """Open ``run_dir/metrics.jsonl`` in append mode and return
    ``(file_handle, path)``.

    Append-only so a resumed run extends the same log instead of
    clobbering it. The operator can ``cat metrics.jsonl | jq`` or
    ``pandas.read_json(..., lines=True)`` to inspect training progress
    live without waiting on TB to load.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / METRICS_FILENAME
    # ``"a"`` keeps prior lines when a run resumes. Line-buffered
    # writes (``buffering=1``) flush after every line so a kill -9
    # never loses a finalised line. The handle has a multi-call
    # lifetime (one open at loop entry, close in the ``finally``
    # block of ``run_training``); a context manager would scope it
    # too tightly. Suppress SIM115.
    fh = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    return fh, path


def _write_jsonl(fh: io.TextIOBase | None, record: dict[str, Any]) -> None:
    """Serialise ``record`` to JSON and append one line to ``fh``.

    Numpy scalar types are silently cast to Python primitives so
    ``json.dumps`` doesn't choke. Missing-handle is a no-op (callers
    that disabled the writer just pass ``None``).
    """
    if fh is None:
        return
    coerced: dict[str, Any] = {}
    for k, v in record.items():
        if hasattr(v, "item") and not isinstance(v, str | bytes):
            # numpy scalar / 0-dim tensor → Python primitive.
            try:
                coerced[k] = v.item()
                continue
            except (ValueError, TypeError):
                pass
        coerced[k] = v
    fh.write(json.dumps(coerced, separators=(",", ":")) + "\n")


def _wall_time_iso() -> str:
    """UTC timestamp as ISO-8601 with second precision. Cheap to call
    and stable enough to correlate metrics with external systems."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log_update_metrics_jsonl(
    fh: io.TextIOBase | None,
    metrics: Any,
    *,
    update_idx: int,
    global_step: int,
) -> None:
    """Append one ``kind=train`` line per PPO update."""
    if fh is None:
        return
    record: dict[str, Any] = {
        "kind": "train",
        "wall_time": _wall_time_iso(),
        "update_idx": int(update_idx),
        "global_step": int(global_step),
    }
    for k, v in vars(metrics).items():
        if isinstance(v, int | float):
            record[k] = float(v)
    _write_jsonl(fh, record)


def _log_eval_report_jsonl(
    fh: io.TextIOBase | None,
    report: Any,
    *,
    update_idx: int,
    global_step: int,
) -> None:
    """Append one ``kind=eval`` line per (opponent, eval round)."""
    if fh is None:
        return
    when = _wall_time_iso()
    for res in report.results:
        record = {
            "kind": "eval",
            "wall_time": when,
            "update_idx": int(update_idx),
            "global_step": int(global_step),
            "opponent_type": str(res.opponent_type),
            "wr": float(res.wr),
            "ci_low": float(res.ci.lower),
            "ci_high": float(res.ci.upper),
            "n_games": int(res.n),
            "n_wins": int(res.wins),
            "n_truncated": int(res.n_truncated),
            "wr_seat0": float(res.wr_seat0),
            "wr_seat1": float(res.wr_seat1),
        }
        _write_jsonl(fh, record)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def run_training_loop(
    state: TrainingState,
    *,
    run_dir: Path,
    max_updates: int | None = None,
    logger: logging.Logger | None = None,
    open_tb: bool = True,
    open_metrics_jsonl: bool = True,
) -> TrainingState:
    """Drive the PPO loop from ``state.update_idx`` to ``n_updates_total``.

    Honours the per-section cadences in ``cfg``:

    * League snapshots every ``league.add_snapshot_every_n_updates``.
    * Eval runs every ``eval.eval_every_updates``.
    * Checkpoints saved every ``checkpoint.save_every_updates``.

    Args:
        state: As returned by :func:`build_training_state`, with
            ``maybe_resume_from_checkpoint`` already applied.
        run_dir: For the TB writer + checkpoint dir (the latter is
            already set up by ``build_training_state``; passed through
            for consistency).
        max_updates: Optional hard cap on how many updates to execute
            in THIS call. Lets the smoke-test exit after N updates
            instead of running the full schedule. ``None`` = run until
            ``state.n_updates_total``.
        logger: defaults to ``catan_rl.train`` logger.
        open_tb: ``True`` opens a TB ``SummaryWriter``. The smoke tests
            pass ``False`` to avoid the dependency.
        open_metrics_jsonl: ``True`` (default) opens
            ``run_dir/metrics.jsonl`` and appends a line per PPO update
            (``kind=train``) plus a line per opponent per eval round
            (``kind=eval``). The file is the operator's primary
            non-TB inspection surface — ``pandas.read_json(p,
            lines=True)`` loads the whole training history.

    Returns the same ``state`` object (mutated in place) so callers
    can introspect after the loop exits.
    """
    log = logger or logging.getLogger("catan_rl.train")
    cfg = state.cfg

    if state.obs is None or state.masks is None:
        # First call. Reset envs to draw the initial obs. Explicit
        # per-env seeds are derived from cfg.seed + the env index so
        # the loop never touches global ``np.random`` here
        # (Reviewer HIGH: ``reset_all(seeds=None)`` would otherwise
        # draw from ``np.random.randint`` and silently desynchronise
        # any downstream consumer of the global PRNG).
        base = (cfg.seed * 1_000_003) & 0x7FFFFFFF
        seeds = [(base + i) & 0x7FFFFFFF for i in range(cfg.rollout.n_envs)]
        state.obs, state.masks = state.vec_env.reset_all(seeds=seeds)

    if open_tb and state.tb_writer is None:
        state.tb_writer = _open_tb_writer(run_dir)

    if open_metrics_jsonl and state.metrics_fh is None:
        state.metrics_fh, state.metrics_path = _open_metrics_writer(run_dir)

    end_update = state.n_updates_total
    if max_updates is not None:
        end_update = min(end_update, state.update_idx + max_updates)

    # Self-play opponent resolver (US3). Cache the heavy CatanPolicy per
    # snapshot id (D6), pruned to the live pool — but hand back a SEPARATE
    # RNG-bearing wrapper per (snapshot id, env) so envs sharing an id never
    # clobber each other's per-game seed (review BLOCKER). Geometry is built
    # lazily so the default (no self-play) run allocates nothing extra.
    _snap_policy_cache: dict[int, Any] = {}
    _snap_geom: dict[str, Any] = {}

    def _resolve_snapshot(snapshot_id: int, env_idx: int) -> Any:
        from catan_rl.selfplay.snapshot_opponent import (
            FrozenSnapshotOpponent,
            load_frozen_policy,
        )

        policy = _snap_policy_cache.get(snapshot_id)
        if policy is None:
            snap = state.league.peek_by_id(snapshot_id)
            if snap is None:
                return None  # evicted -> env falls back to heuristic body (FR-011)
            if not _snap_geom:
                _snap_geom.update(build_geometry().as_dict_of_tensors())
            policy = load_frozen_policy(snap.state_dict, geometry=_snap_geom)
            _snap_policy_cache[snapshot_id] = policy
        # Pin the opponent to CPU regardless of the learner device: opponent
        # inference is per-env BATCH-1, the regime where MPS/CUDA are ~7-8x
        # SLOWER than CPU (the same reason eval is CPU-pinned). The agent's
        # rollout forward stays batched on the learner device. (T019 perf fix —
        # ~3-4x faster self-play updates than running the opponent on MPS.)
        seed = (cfg.seed ^ (snapshot_id * 0x9E3779B1) ^ (env_idx * 0x85EBCA6B)) & 0x7FFFFFFF
        return FrozenSnapshotOpponent(policy, device=torch.device("cpu"), seed=seed)

    try:
        while state.update_idx < end_update:
            update_idx = state.update_idx

            # ---- self-play opponent refresh (US3) -------------------
            # Re-draw the per-env opponent assignment each rollout so freshly
            # added snapshots (and the frozen anchor) enter play. Guarded by
            # snapshot_weight>0 OR anchor_weight>0 so the default (heuristic-only)
            # run is unchanged. Per-update-seeded RNG keeps the assignment
            # sequence resume-reproducible.
            if cfg.league.snapshot_weight > 0 or cfg.league.anchor_weight > 0:
                live_ids = state.league.snapshot_ids()
                for stale in [k for k in _snap_policy_cache if k not in live_ids]:
                    del _snap_policy_cache[stale]
                # SeedSequence-style derivation is collision-free across
                # (seed, update_idx) and keeps the assignment resume-reproducible.
                opp_rng = np.random.default_rng([cfg.seed, update_idx])
                assignments = state.league.build_env_opponent_assignments(
                    n_envs=cfg.rollout.n_envs, rng=opp_rng
                )
                state.vec_env.set_opponents(assignments, snapshot_resolver=_resolve_snapshot)

            # ---- rollout ---------------------------------------------
            state.obs, state.masks = state.collector.collect(state.obs, state.masks)
            state.buffer.compute_returns_and_advantages(
                last_values=state.collector.last_values,
                gamma=cfg.gae.gamma,
                gae_lambda=cfg.gae.gae_lambda,
                advantage_norm=cfg.ppo.advantage_norm,
            )

            # ---- SGD -------------------------------------------------
            metrics = state.trainer.update(
                state.buffer,
                update_idx=update_idx,
                rng=state.rng,
            )

            # ---- non-finite loss guard ------------------------------
            # If the PPO total loss explodes (NaN / inf), bail before
            # the next checkpoint save writes poisoned weights.
            # Reviewer-caught MEDIUM. Last-good ckpt stays intact.
            if not (
                math.isfinite(metrics.policy_loss)
                and math.isfinite(metrics.value_loss)
                and math.isfinite(metrics.total_loss)
            ):
                raise RuntimeError(
                    f"non-finite loss at update {update_idx}: "
                    f"policy={metrics.policy_loss} value={metrics.value_loss} "
                    f"total={metrics.total_loss}"
                )

            # ---- bookkeeping ----------------------------------------
            transitions = cfg.rollout.n_envs * cfg.rollout.n_steps
            state.global_step += transitions
            log.info(
                "update %d/%d global_step=%d "
                "policy_loss=%.4f value_loss=%.4f entropy=%.4f kl=%.4f lr=%.2e",
                update_idx,
                state.n_updates_total,
                state.global_step,
                metrics.policy_loss,
                metrics.value_loss,
                metrics.entropy_bonus,
                metrics.approx_kl,
                metrics.lr,
            )
            _log_update_metrics(state.tb_writer, metrics, global_step=state.global_step)
            _log_update_metrics_jsonl(
                state.metrics_fh,
                metrics,
                update_idx=update_idx,
                global_step=state.global_step,
            )

            # ---- league snapshot ------------------------------------
            if state.league.should_snapshot_this_update(update_idx):
                snap_id = state.league.add_snapshot(
                    state.policy.state_dict(),
                    update_idx=update_idx,
                )
                log.info(
                    "league: added snapshot id=%d at update %d",
                    snap_id,
                    update_idx,
                )

            # ---- eval -----------------------------------------------
            if (
                cfg.eval.eval_every_updates > 0
                and (update_idx + 1) % cfg.eval.eval_every_updates == 0
            ):
                report = state.eval_harness.run(state.policy)
                _log_eval_report(state.tb_writer, report, global_step=state.global_step)
                _log_eval_report_jsonl(
                    state.metrics_fh,
                    report,
                    update_idx=update_idx,
                    global_step=state.global_step,
                )
                for res in report.results:
                    log.info(
                        "eval %s WR %.3f (CI %.3f-%.3f) n=%d",
                        res.opponent_type,
                        res.wr,
                        res.ci.lower,
                        res.ci.upper,
                        res.n,
                    )

            # ---- checkpoint -----------------------------------------
            if (
                cfg.checkpoint.save_every_updates > 0
                and (update_idx + 1) % cfg.checkpoint.save_every_updates == 0
            ):
                path = state.ckpt_mgr.save(
                    config=cfg.to_dict(),
                    policy=state.policy,
                    optimizer=state.optimizer,
                    update_idx=update_idx,
                    global_step=state.global_step,
                    league=state.league,
                    vec_env=state.vec_env,
                )
                log.info("checkpoint saved: %s", path)

            state.update_idx += 1
    finally:
        if state.tb_writer is not None:
            import contextlib

            with contextlib.suppress(Exception):  # pragma: no cover
                state.tb_writer.flush()
        if state.metrics_fh is not None:
            import contextlib

            with contextlib.suppress(Exception):
                state.metrics_fh.flush()

    return state


def run_training(
    cfg: TrainConfig,
    *,
    run_dir: Path,
    device_label: str,
    logger: logging.Logger | None = None,
    max_updates: int | None = None,
    open_tb: bool = True,
    open_metrics_jsonl: bool = True,
) -> TrainingState:
    """Compose :func:`build_training_state`, the optional resume, and
    :func:`run_training_loop` into a single call.

    The script entry point in ``scripts/train.py`` calls this; tests
    that need finer-grained control (e.g., to inspect state mid-loop)
    can call the three sub-functions directly.
    """
    log = logger or logging.getLogger("catan_rl.train")
    state = build_training_state(cfg, run_dir=run_dir, device_label=device_label, logger=log)
    maybe_resume_from_checkpoint(state, logger=log)
    try:
        run_training_loop(
            state,
            run_dir=run_dir,
            max_updates=max_updates,
            logger=log,
            open_tb=open_tb,
            open_metrics_jsonl=open_metrics_jsonl,
        )
    finally:
        state.vec_env.close()
        if state.tb_writer is not None:
            import contextlib

            with contextlib.suppress(Exception):  # pragma: no cover
                state.tb_writer.close()
        if state.metrics_fh is not None:
            import contextlib

            with contextlib.suppress(Exception):
                state.metrics_fh.close()
    return state
