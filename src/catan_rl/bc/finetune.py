"""Champion fine-tune with self-distillation anchoring (spec D5).

Installs the owner's hand-labeled openings into ``runs/anchors/ptr_v1_u500.pt``
without damaging its midgame.

The objective
-------------
``L = L_BC(human rows) + kl_coef · KL_anchor + anchor_value_coef · MSE_value``

* ``L_BC`` is the ordinary BC loss over the human shard, with the **value loss
  zeroed per row** (``bc_loss(..., value_row_mask=...)``): a hand-labeled
  opening carries no outcome, so its ``z_disc`` is a filler zero and regressing
  toward it would teach the champion that every labeled opening is a dead draw.

* ``KL_anchor`` is the ONLINE self-distillation term the owner chose over
  freezing heads. The fine-tuned policy and a FROZEN copy of ``u500`` are both
  evaluated on freshly sampled NON-setup states, and the divergence between them
  is penalised. Online rather than offline because the offline form needs a
  stored per-head-distribution dataset that does not exist anywhere in ``bc/``
  today — building one is the largest hidden cost in this slice, and its absence
  is exactly what invites the silent simplification to hard-label self-BC that
  the spec calls a violation.

  **What the term actually measures.** The six heads are autoregressive: corner
  is conditioned on the action type, resource2 on the type and the chosen
  resource1. There is no tractable joint distribution to diverge, so this is the
  relevance-weighted sum of the six per-head CONDITIONAL KLs **at a single
  reference action context** — the legal action the anchor sampler walked the
  game with. That is an upper-bound-flavoured surrogate for the joint KL, not
  the joint KL, and it is named as such here and in
  :meth:`catan_rl.policy.heads.CatanActionHeads.masked_log_dists` so no later
  reader mistakes it for one.

  **Where the states come from** is not a detail either: they are rolled out
  from games whose setups are FORCED to the owner's labeled openings AND walked
  forward by the FROZEN CHAMPION ITSELF (see :mod:`catan_rl.bc.anchor_states`),
  so the anchor covers the distribution the fine-tune moves TOWARD rather than
  the one it leaves — and covers it on-policy rather than a few random plies off
  it.

* ``MSE_value`` is the frozen-VALUE guard, on the same states and out of the
  same pair of forward passes (:func:`anchor_terms`). The six action heads are
  not the whole champion: the deployed search reads the value head through
  ``sigma(3.22V - 1.14)``, so a value head free to drift while every action head
  is pinned is a deployment-visible regression the KL term structurally cannot
  see.

Held-out gating (D7)
--------------------
:func:`finetune` REFUSES a shard whose manifest withheld no ``game_seed``
(:class:`UngatedShardError`) unless ``FinetuneConfig.allow_ungated`` is set. The
gate CLI already refuses such a shard — but it refuses after the run, when the
labels are spent on a candidate no held-out measurement can read.

Epoch (D6)
----------
``ptr_v1_u500.pt`` carries no ruleset stamp, so it is **R0**. The candidate is
stamped R0 explicitly. Any R1 transition is the successor slice's explicit
decision, never a side effect of a default config.

Deliverables (D8)
-----------------
:func:`finetune` writes the candidate checkpoint AND a frozen copy designated
the **human-opening prior**. The successor self-play slice is contractually
required to carry a ready-to-enable KL anchor to that prior at setup nodes —
without it, the installed openings are expected to decay on contact with PPO.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from catan_rl.bc.anchor_states import sample_anchor_states
from catan_rl.bc.loader import BcDataset, bc_collate
from catan_rl.bc.loss import bc_loss

# The torch-RNG snapshot/restore pair lives in ``eval.harness`` and is IMPORTED
# rather than re-derived (``eval.cross_arch`` already does the same):
# ``torch.manual_seed`` reseeds every backend, so a second copy of the
# cpu+cuda+mps list would be a second place for one to be forgotten.
from catan_rl.eval.harness import _restore_torch_rng, _snapshot_torch_rng
from catan_rl.policy import CatanPolicy
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.obs_schema import AUTOREGRESSIVE_HEAD, MASK_KEY_FOR

_HEAD_NAMES: tuple[str, ...] = ("type", "corner", "edge", "tile", "resource1", "resource2")


@dataclass
class FinetuneConfig:
    """Everything the fine-tune needs. No hidden defaults that change behaviour."""

    ckpt_path: Path
    """The champion to fine-tune. Production value: ``runs/anchors/ptr_v1_u500.pt``."""

    shard_dir: Path
    """Human-label BC shard, from ``catan_rl.labeling.to_shard.convert``."""

    labels_path: Path
    """The label JSONL the anchor states' FORCED openings are drawn from."""

    out_dir: Path
    """Receives ``candidate.pt``, ``human_opening_prior.pt`` and ``history.json``."""

    steps: int = 200
    batch_size: int = 32
    lr: float = 1e-5
    """Deliberately small. This is a surgical edit to a banked champion, not a
    training run; the anchor term bounds drift but does not license a large LR."""

    kl_coef: float = 1.0
    """Keep this MODERATE. Measured while building this module: raising it to
    10x-200x does not tighten the anchor, it destabilises the optimisation —
    with ``clip_grad_norm_(0.5)`` a huge KL coefficient consumes the whole
    gradient budget and held-out drift comes out WORSE than an unanchored run.
    At 1.0 the same setup cut held-out drift to ~0.19x the unanchored control.
    ``tests/unit/bc/test_finetune.py`` pins the effect at this default."""

    anchor_value_coef: float = 10.0
    """Weight on the frozen-VALUE guard, ``MSE(V_ft, V_frozen)`` over the same
    anchor states. The six action heads are not the whole champion: the deployed
    search reads the value head through ``sigma(3.22V - 1.14)``
    (:mod:`catan_rl.search`), so a value head that drifts while the policy heads
    are pinned is a deployment-visible regression the KL term cannot see.

    NOT the same order as ``kl_coef``, and the difference is measured, not
    guessed. Squared-error on a scalar in roughly ``[-1, 1]`` is an order of
    magnitude smaller than the summed six-head KL, so at ``1.0`` the term
    carries too little gradient to matter. MEASURED on the mock corpus while
    building this module (60 steps, lr 1e-3, refresh on, three seeds; held-out
    value MSE, mean over seeds):

    ===========  ==================
    coefficient  held-out value MSE
    ===========  ==================
    0.0          0.090
    10.0         0.0060
    50.0         0.0078
    100.0        0.0039
    ===========  ==================

    ``10.0`` is the smallest weight that cut drift on EVERY seed (~0.07x the
    unguarded control) while leaving the action-head KL where the unguarded run
    left it. Beyond it the value number stops improving and the KL starts
    swinging seed to seed — the ``clip_grad_norm_(0.5)`` budget hazard the
    ``kl_coef`` note describes, arriving here too.
    ``tests/unit/bc/test_finetune.py::test_the_anchor_bounds_value_drift_too``
    pins the effect at this default."""

    anchor_batch: int = 16
    n_anchor_states: int = 256
    anchor_refresh_every: int = 50
    """Anchor states are re-sampled this often, so the term is evaluated on
    FRESH states rather than a fixed set the fine-tune can overfit around.

    Not optional in practice: with refresh OFF and a small pool, the fine-tune
    drives the training KL down on those exact states while HELD-OUT drift grows
    — measured at ~3.2x an unanchored run. Setting this to 0 disables the
    refresh and reintroduces that failure."""

    belief_weight: float = 0.05
    seed: int = 0
    device: str = "cpu"
    """CPU by default per the repo device policy (eval is pinned to CPU; this is
    a few hundred small-batch steps, not a training run)."""

    allow_ungated: bool = False
    """Train on a shard that withheld NOTHING. Off by default, and refusing is
    the point: a candidate fine-tuned on every label cannot be read by D7 gate 1
    at all, and by the time ``scripts/eval_setup_agreement.py`` notices, the
    training run is already spent. Setting this True is legitimate (the labels
    may deliberately all be in-sample for a final build), but it is a DECISION —
    recorded in ``history.json`` as ``allow_ungated`` / ``held_out_gated``."""


def build_policy(ckpt_path: Path, device: torch.device) -> CatanPolicy:
    """Load ``ckpt_path`` into a fresh :class:`CatanPolicy` on ``device``.

    Public because the gate CLIs (``scripts/eval_setup_agreement.py``,
    ``scripts/eval_wr_non_inferiority.py``) load the same checkpoints the
    fine-tune does, and a second loader would be a second place for the
    geometry-before-state-dict ordering below to be got wrong.
    """
    from catan_rl.checkpoint import load_checkpoint

    policy = CatanPolicy()
    # Geometry BEFORE the device move + state-dict apply, matching
    # ``replay.player_factory``: the setter writes registered buffers that the
    # checkpoint then overwrites.
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy = policy.to(device)
    payload = load_checkpoint(ckpt_path, map_location=device)
    payload.apply_to_policy(policy, strict=True)
    return policy


@dataclass(frozen=True)
class AnchorTerms:
    """Both halves of the anchor, from ONE pair of forward passes."""

    kl: torch.Tensor
    """Relevance-weighted sum of the six per-head conditional KLs."""

    value_mse: torch.Tensor
    """``MSE(V_fine-tuned, V_frozen)`` on the same states."""

    head_relevance: torch.Tensor
    """(6,) — the per-head DENOMINATORS :attr:`kl` was normalised by, in
    :data:`_HEAD_NAMES` order.

    Reported because the sum alone cannot distinguish "this head did not drift"
    from "this head had no rows". A batch of pure ``END_TURN`` / ``ROLL_DICE``
    nodes constrains nothing but the type head, and its anchor KL is small for
    that reason rather than because the placement heads are pinned."""


def _head_coverage(masks: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """(B, 6) float — 1.0 where a head's own mask offers at least one legal index.

    An all-False head mask is not a constrained head: ``masked_log_softmax``
    hands BOTH policies the same uniform placeholder, so that row's KL is
    identically zero. ``PLAY_KNIGHT``'s ``tile`` head is the live case — the
    ``tile`` mask is only ever written in the env's robber-placement branch
    (``env/masks.py``), which is a different node, so it is all-False at every
    node offering ``PLAY_KNIGHT``. Counting those rows in the tile denominator
    divides real drift by a larger number and reports the anchor as tighter than
    it is.

    Head 0 (``type``) is always covered: ``compute_action_masks`` falls back to
    ``END_TURN`` rather than emitting an empty type mask.
    """
    any_legal = {
        key: mask.reshape(mask.shape[0], -1).any(dim=-1).cpu().numpy()
        for key, mask in masks.items()
    }
    types = action[:, 0].cpu().numpy()
    cover = np.ones((len(types), len(_HEAD_NAMES)), dtype=np.float32)
    for row, action_type in enumerate(types):
        for head in range(1, len(_HEAD_NAMES)):
            key = MASK_KEY_FOR[action_type][head]
            if key is None:  # head irrelevant for this type; relevance zeroes it
                continue
            legal = bool(any_legal[key][row])
            if head == AUTOREGRESSIVE_HEAD and key == "resource2_yop":
                # ``MASK_KEY_FOR[..][5]`` names only the BASE key; a doubled YoP
                # pick is routed to ``resource2_yop_same`` instead, so the head
                # is covered if EITHER offers an index.
                legal = legal or bool(any_legal["resource2_yop_same"][row])
            cover[row, head] = float(legal)
    return torch.as_tensor(cover, device=action.device)


def anchor_terms(
    trainable: CatanPolicy,
    frozen: CatanPolicy,
    batch: dict[str, Any],
) -> AnchorTerms:
    """Anchor KL **and** value drift, computed together.

    ``KL(fine-tuned ‖ frozen)`` — the fine-tuned policy is the one being held in
    place, so it is the distribution the expectation is taken under. Heads the
    reference action type does not use contribute nothing (their masks are
    all-False and ``masked_log_softmax`` would hand back a uniform placeholder —
    noise wearing the shape of an opinion).

    The value term rides along because both policies' ``forward`` already emits
    ``"value"``: a separate function would double the forward passes to
    re-derive a number this one already has in hand. It is the guard the six
    action heads do not provide — the deployed search squashes this head through
    ``sigma(3.22V - 1.14)``, so unconstrained value drift changes deployed
    behaviour even with every action head pinned.
    """
    out_ft = trainable.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    with torch.no_grad():
        out_fz = frozen.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    # Relevance says the head CONTRIBUTES to this action type's joint log-prob;
    # coverage says its mask actually offered a choice. A head that is relevant
    # but entirely masked contributes an identically-zero KL, so leaving it in
    # the denominator would report the anchor as tighter than it is.
    relevance = out_ft["relevance"] * _head_coverage(batch["mask"], batch["action"])

    total = relevance.new_zeros(())
    for h_idx, name in enumerate(_HEAD_NAMES):
        log_p = out_ft[f"log_dist/{name}"]
        log_q = out_fz[f"log_dist/{name}"]
        # p*(log p - log q). Masked slots carry a large finite negative logit
        # (``heads._LOGIT_NEG_INF``), so p ≈ 0 there and the product vanishes
        # without any -inf arithmetic.
        per_row = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        rel = relevance[:, h_idx]
        total = total + (per_row * rel).sum() / rel.sum().clamp_min(1.0)

    value_mse = torch.nn.functional.mse_loss(
        out_ft["value"].reshape(-1), out_fz["value"].reshape(-1)
    )
    return AnchorTerms(kl=total, value_mse=value_mse, head_relevance=relevance.detach().sum(dim=0))


def _anchor_batch_tensors(
    states: list[dict[str, Any]], idx: np.ndarray, device: torch.device
) -> dict[str, Any]:
    obs_keys = states[0]["obs"].keys()
    mask_keys = states[0]["mask"].keys()
    obs: dict[str, torch.Tensor] = {}
    for key in obs_keys:
        arr = np.stack([np.asarray(states[i]["obs"][key]) for i in idx])
        obs[key] = torch.as_tensor(arr, device=device)
        if obs[key].dtype not in (torch.int64,):
            obs[key] = obs[key].float()
    mask = {
        key: torch.as_tensor(
            np.stack([np.asarray(states[i]["mask"][key], dtype=bool) for i in idx]),
            device=device,
        )
        for key in mask_keys
    }
    action = torch.as_tensor(
        np.stack([states[i]["action"] for i in idx]).astype(np.int64), device=device
    )
    return {"obs": obs, "mask": mask, "action": action}


def _head_relevance_dict(counts: torch.Tensor) -> dict[str, float]:
    """The (6,) coverage vector as a NAMED mapping, for the JSON artefacts."""
    return dict(zip(_HEAD_NAMES, (float(x) for x in counts.tolist()), strict=True))


class UngatedShardError(RuntimeError):
    """The shard withheld nothing, so the candidate it produces is ungateable."""


def _shard_held_out_seeds(shard_dir: Path) -> tuple[int, ...]:
    """The ``game_seed`` s the converter withheld from ``shard_dir``.

    Empty for a shard built with ``held_out_frac=0`` (and for the pre-split
    manifests), which is the honest reading: nothing was withheld, so nothing
    needs excluding. :func:`_require_gated_shard` is what decides whether that is
    acceptable.
    """
    manifest = Path(shard_dir) / "manifest.json"
    if not manifest.is_file():
        return ()
    payload = json.loads(manifest.read_text())
    return tuple(int(s) for s in payload.get("held_out_game_seeds", ()))


def _require_gated_shard(shard_dir: Path, *, allow_ungated: bool) -> tuple[int, ...]:
    """Refuse a shard that withheld nothing, unless the caller said so.

    ``scripts/eval_setup_agreement.py`` already refuses such a shard — but it
    refuses at GATE time, when the fine-tune has run and every label is spent on
    a candidate no held-out measurement can read. The same refusal here costs a
    re-run of the converter and nothing else.

    Missing-manifest and empty-set are reported separately: the first is a
    malformed shard directory, the second a deliberate ``--held-out-frac 0``.
    """
    manifest = Path(shard_dir) / "manifest.json"
    if not manifest.is_file():
        if allow_ungated:
            return ()
        raise UngatedShardError(
            f"{shard_dir} has no manifest.json, so nothing can say which game_seeds "
            f"were withheld for the D7 gate-1 measurement. Rebuild it with "
            f"scripts/convert_labels_to_bc_shard.py, or pass allow_ungated=True to "
            f"train a deliberately ungateable candidate."
        )
    seeds = _shard_held_out_seeds(shard_dir)
    if not seeds and not allow_ungated:
        payload = json.loads(manifest.read_text())
        how = (
            "carries no 'held_out_game_seeds' key"
            if "held_out_game_seeds" not in payload
            else "has an EMPTY held-out set (--held-out-frac 0)"
        )
        raise UngatedShardError(
            f"{manifest} {how}, so every label is in-sample and D7 gate 1 would "
            f"report memorisation rather than generalisation. Re-run "
            f"scripts/convert_labels_to_bc_shard.py with --held-out-frac > 0, or "
            f"pass allow_ungated=True to make that trade-off explicitly."
        )
    return seeds


def _human_row_value_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Every row of the human shard is value-less. Explicit, not implied."""
    return torch.zeros_like(batch["z_disc"])


def finetune(config: FinetuneConfig) -> dict[str, Any]:
    """Run the fine-tune and write the D8 deliverables. Returns the history dict.

    Raises:
        UngatedShardError: if ``config.shard_dir`` withheld nothing and
            ``config.allow_ungated`` is False.
    """
    # The shard's manifest names the game_seeds the converter WITHHELD for the
    # D7 gate-1 measurement. Checked BEFORE anything expensive: a shard that
    # withheld nothing produces a candidate no held-out gate can read, and the
    # cheap moment to say so is now.
    #
    # The same seeds are excluded from the anchor rollouts below: forcing a
    # held-out opening into the anchor states would let it shape the candidate
    # through the KL term, and the gate would again be measuring a position the
    # fine-tune had seen.
    held_out_seeds = _require_gated_shard(config.shard_dir, allow_ungated=config.allow_ungated)
    with _global_rng_restored():
        return _finetune(config, held_out_seeds)


@contextmanager
def _global_rng_restored() -> Iterator[None]:
    """Snapshot and restore the process-global RNGs around the run.

    :func:`_finetune` SEEDS numpy's global, the stdlib ``random`` and torch (the
    engine's ``StackedDice`` and the heuristic opponent read the first two),
    which is what makes a candidate checkpoint reproducible — but leaving those
    streams re-seeded on exit would make whatever runs next depend on whether a
    fine-tune happened first.

    The torch half goes through ``eval.harness``'s ``_snapshot_torch_rng`` /
    ``_restore_torch_rng``, which cover **cpu + cuda + mps**. That matters even
    though this routine runs on CPU by default: ``torch.manual_seed`` reseeds
    EVERY backend's generator, so restoring the CPU state alone would hand back
    one stream and leave a caller's MPS/CUDA stream clobbered.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = _snapshot_torch_rng()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        _restore_torch_rng(torch_state)


def _finetune(config: FinetuneConfig, held_out_seeds: tuple[int, ...]) -> dict[str, Any]:
    from catan_rl.checkpoint import save_policy_only

    device = torch.device(config.device)
    # Seed EVERY generator the run transitively touches, not just torch: the
    # anchor rollouts go through the engine, whose ``StackedDice`` draws its seed
    # from the stdlib ``random`` and whose heuristic opponent uses the numpy
    # GLOBAL. Leaving those to ambient process state makes a checkpoint-producing
    # routine irreproducible depending on what ran before it.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    trainable = build_policy(config.ckpt_path, device)
    trainable.train()
    frozen = build_policy(config.ckpt_path, device)
    frozen.eval()
    frozen.requires_grad_(False)

    dataset = BcDataset(config.shard_dir, aug_prob=0.0, seed=config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=bc_collate,
        num_workers=0,
        drop_last=False,
    )
    optimizer = AdamW(trainable.parameters(), lr=config.lr, weight_decay=1e-4)

    # The walker is the FROZEN copy (D5): a trainable walker would make the
    # anchor's state distribution drift with the very update the term bounds.
    anchor = sample_anchor_states(
        config.labels_path,
        policy=frozen,
        n_states=config.n_anchor_states,
        rng=rng,
        device=device,
        exclude_game_seeds=held_out_seeds,
    )
    history: list[dict[str, Any]] = []
    step = 0
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            if config.anchor_refresh_every and step and step % config.anchor_refresh_every == 0:
                anchor = sample_anchor_states(
                    config.labels_path,
                    policy=frozen,
                    n_states=config.n_anchor_states,
                    rng=rng,
                    device=device,
                    exclude_game_seeds=held_out_seeds,
                )
            batch = _to_device(batch, device)
            out = trainable.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
            losses = bc_loss(
                policy_out=out,
                batch=batch,
                belief_weight=config.belief_weight,
                value_row_mask=_human_row_value_mask(batch),
            )
            idx = rng.choice(len(anchor), size=min(config.anchor_batch, len(anchor)), replace=False)
            terms = anchor_terms(trainable, frozen, _anchor_batch_tensors(anchor, idx, device))
            total = (
                losses["total"]
                + config.kl_coef * terms.kl
                + config.anchor_value_coef * terms.value_mse
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), 0.5)
            optimizer.step()

            history.append(
                {
                    "step": float(step),
                    "total": float(total.item()),
                    "policy": float(losses["policy"].item()),
                    "value": float(losses["value"].item()),
                    "anchor_kl": float(terms.kl.item()),
                    "anchor_value_mse": float(terms.value_mse.item()),
                    # The denominators ``anchor_kl`` was normalised by. Without
                    # them a small KL is unreadable: it could mean the heads were
                    # held in place, or that this batch offered them no rows.
                    "anchor_head_relevance": _head_relevance_dict(terms.head_relevance),
                }
            )
            step += 1

    config.out_dir.mkdir(parents=True, exist_ok=True)
    trainable.eval()
    # D6: stamp the epoch EXPLICITLY. ``checkpoint_ruleset`` reads this back, and
    # an absent stamp would be read as R0 by accident rather than by decision.
    stamped_config = {"rollout": {"ruleset": "R0"}, "source": "bc.finetune"}
    candidate = config.out_dir / "candidate.pt"
    save_policy_only(
        candidate,
        config=stamped_config,
        policy=trainable,
        update_idx=0,
        global_step=config.steps,
        metadata={
            "kind": "policy_only",
            "finetune": "human_openings",
            "base_ckpt": str(config.ckpt_path),
            "kl_coef": config.kl_coef,
            "anchor_value_coef": config.anchor_value_coef,
        },
    )
    # D8 deliverable (iii): a FROZEN copy designated the human-opening prior. It
    # is a separate file on purpose — the candidate may be superseded, and the
    # successor self-play slice's KL anchor must keep pointing at these weights.
    prior = config.out_dir / "human_opening_prior.pt"
    save_policy_only(
        prior,
        config=stamped_config,
        policy=trainable,
        update_idx=0,
        global_step=config.steps,
        metadata={
            "kind": "policy_only",
            "role": "human_opening_prior",
            "base_ckpt": str(config.ckpt_path),
        },
    )
    out_history = {
        "steps": history,
        "candidate": str(candidate),
        "human_opening_prior": str(prior),
        "n_rows": len(dataset),
        "ruleset": "R0",
        "held_out_game_seeds": sorted(held_out_seeds),
        # Whether this candidate is gateable at all, recorded in the artefact
        # rather than inferable from an empty list: an empty ``held_out_game_seeds``
        # with ``allow_ungated`` False is impossible, and with it True is a
        # decision someone made.
        "held_out_gated": bool(held_out_seeds),
        "allow_ungated": bool(config.allow_ungated),
    }
    (config.out_dir / "history.json").write_text(json.dumps(out_history, indent=2))
    return out_history


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        else:
            out[key] = value.to(device)
    return out


@dataclass(frozen=True)
class AnchorDrift:
    """A held-out drift measurement. Both terms, plus what it was measured on."""

    kl: float
    value_mse: float
    n_states: int

    head_relevance: dict[str, float]
    """Per-head denominators behind :attr:`kl`, keyed by head name.

    ``n_states`` alone does not say WHICH heads the measurement could see. A
    drift number whose ``corner`` count is 0 says nothing about the placement
    behaviour the anchor exists to protect, and the caller can only tell by
    reading this."""


def held_out_anchor_drift(
    candidate_path: Path,
    base_path: Path,
    labels_path: Path,
    *,
    n_states: int = 64,
    seed: int = 1234,
    device: str = "cpu",
    exclude_game_seeds: Sequence[int] = (),
) -> AnchorDrift:
    """Held-out anchor drift between a candidate and the base champion.

    ``kl`` is the SAME quantity :func:`anchor_terms` optimises — a
    relevance-normalised SUM of the six per-head conditional KLs at one reference
    action context per state, averaged over states within each head. It is not a
    mean per-state joint KL; there is no tractable joint over six autoregressive
    heads to take a mean of. ``value_mse`` is the frozen-value guard's term.

    Read ``head_relevance`` before reading ``kl``. The sum is normalised per
    head, so a head that no sampled state exercised contributes nothing and is
    indistinguishable from a head that did not drift. On a short walk the corner
    head in particular is reached on a fraction of a percent of states, so a
    small ``kl`` is NOT by itself a statement about settlement placement.

    Evaluated on anchor states sampled with a DIFFERENT seed than training used,
    so the number is a held-out drift measurement rather than the training term
    read back. Pass the shard's ``held_out_game_seeds`` to keep the same
    openings out of this measurement that the fine-tune kept out of training.

    The walker is the BASE champion — the same frozen policy the fine-tune
    anchored against, so the measurement is taken on the distribution the anchor
    was defined over rather than on the candidate's own.

    The process-global RNGs are seeded and restored around the call: this now
    consumes the torch stream through ``policy.sample``, so without that the
    number would depend on what ran before it and the measurement would perturb
    whatever runs after.
    """
    torch_device = torch.device(device)
    with _global_rng_restored():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        candidate = build_policy(candidate_path, torch_device)
        candidate.eval()
        base = build_policy(base_path, torch_device)
        base.eval()
        rng = np.random.default_rng(seed)
        states = sample_anchor_states(
            labels_path,
            policy=base,
            n_states=n_states,
            rng=rng,
            device=torch_device,
            exclude_game_seeds=exclude_game_seeds,
        )
        with torch.no_grad():
            batch = _anchor_batch_tensors(states, np.arange(len(states)), torch_device)
            terms = anchor_terms(candidate, base, batch)
        return AnchorDrift(
            kl=float(terms.kl.item()),
            value_mse=float(terms.value_mse.item()),
            n_states=len(states),
            head_relevance=_head_relevance_dict(terms.head_relevance),
        )


def setup_agreement(
    ckpt_path: Path,
    labels: list[dict[str, Any]],
    *,
    device: str = "cpu",
    opponent_kind: int | None = None,
) -> list[bool]:
    """Per-label top-1 settlement agreement for the policy at ``ckpt_path``.

    The position is rebuilt through the SAME converter path the training shard
    uses (:func:`catan_rl.labeling.to_shard.rows_for_label`), so an agreement
    number can never be measured on a differently-encoded state than the one the
    fine-tune trained on.

    **Opponent identity is stamped here, explicitly.** ``rows_for_label``
    returns PRE-duplication rows, whose ``opponent_kind`` is whatever
    :class:`~catan_rl.policy.obs_encoder.EnvObsState` defaults to —
    ``OPP_KIND_UNKNOWN``, which
    :data:`catan_rl.labeling.to_shard.DEFAULT_OPPONENT_KINDS` deliberately
    EXCLUDES from the shard. Reading the gate in that slice would measure the
    one opponent-id conditional the fine-tune never trains, and D4's own
    reasoning (a single-kind stamp "can leave the learned openings unexpressed
    under the other kind") says that number need not match. ``None`` therefore
    means ``OPP_KIND_HEURISTIC`` — a kind the shard DOES carry, and the one the
    D7 gate-2 full-game eval plays.
    """
    from catan_rl.labeling.to_shard import UNKNOWN_POLICY_ID, rows_for_label
    from catan_rl.policy.obs_schema import OPP_KIND_HEURISTIC

    kind = OPP_KIND_HEURISTIC if opponent_kind is None else int(opponent_kind)

    torch_device = torch.device(device)
    policy = build_policy(ckpt_path, torch_device)
    policy.eval()

    out: list[bool] = []
    for label in labels:
        settle_row, _road_row = rows_for_label(label, game_id=0)
        obs = {
            key: torch.as_tensor(np.asarray(value), device=torch_device).unsqueeze(0)
            for key, value in settle_row.obs.items()
        }
        for key, value in obs.items():
            if value.dtype not in (torch.int64,):
                obs[key] = value.float()
            else:
                obs[key] = value.reshape(-1)
        obs["opponent_kind"] = torch.full((1,), kind, dtype=torch.int64, device=torch_device)
        obs["opponent_policy_id"] = torch.full(
            (1,), UNKNOWN_POLICY_ID, dtype=torch.int64, device=torch_device
        )
        mask = {
            key: torch.as_tensor(np.asarray(value, dtype=bool), device=torch_device).unsqueeze(0)
            for key, value in settle_row.mask.items()
        }
        action = torch.as_tensor(settle_row.action, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            head_out = policy.evaluate_actions(obs, action, mask)
        top1 = head_out["log_dist/corner"].argmax(dim=-1).item()
        out.append(top1 == int(label["settlement_vertex"]))
    return out
