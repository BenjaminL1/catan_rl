"""Root-headroom probe (spec 008 STAGE-A kill-gate, FR-004).

Standalone, additive diagnostic. Decides whether the deployed determinized
PUCT-MCTS leaves meaningful win-probability ON THE TABLE AT THE ROOT — i.e.
whether a better root *decision rule* (Gumbel root + Sequential Halving, STAGE-B)
is worth building (GO), or PUCT already picks near-best root moves so v8 + search
is the practical ceiling (NO-GO).

What it measures: PUCT's ROOT-DECISION REGRET against a rollout oracle. For a
representative mix of in-game positions (early / mid / late), it (a) runs the
deployed PUCT search and records its chosen root action + the depth-0 visit
concentration, and (b) estimates the TRUE win-prob ``q(a)`` of each candidate
root action by Monte-Carlo rolling each one out to terminal under v8 on both
seats. ``regret = max_a q(a) - q(PUCT's action)`` is how much PUCT gives up at
the root vs the rollout oracle.

Candidate root action set (DOCUMENTED CHOICE)
---------------------------------------------
PUCT expands ONE representative child per legal TYPE (the policy's modal/argmax
sub-action for that type — see ``search/priors.py``), and that representative set
is exactly the keys of the search's root ``visit_counts`` diagnostic. The legal
TYPE head is the real branching factor (~2-6 mid-game). We therefore take the
candidate root action set = those root children PUCT actually expanded (the
modal sub-action per legal type), matching the spec's "restrict to the legal
TYPE-level actions and pick the modal sub-action per type". This is the SAME
action set PUCT chose from, so the regret is a faithful root-decision regret (it
does NOT credit the oracle with where-to-build sub-actions PUCT never considered;
that is a separate sub-action-search question, out of scope for this gate).

Method
------
1. Load v8 (``runs/anchors/v8_promobar_u243.pt``) via ``build_actor`` (same load
   path as the irreducible-ceiling probe / the search bake-off). Build a
   ``SearchAgent`` on it with the DEPLOYED default ``SearchConfig``
   (``fpu_mode='zero'``, ``n_determinizations=1``, ``c_puct=1.5``, ...) at
   ``sims_per_move=--sims`` (default 100).
2. **Anchor games.** Play v8-vs-v8 RAW games (no search; the irreducible-ceiling
   probe's ``_play_anchor_game`` pattern), seat-symmetric, fixed seed. At each
   NON-FORCED agent decision (``masks["type"].sum() > 1``) — across ALL phases,
   not just early/late — reservoir-sample positions, recording a FAITHFUL deep
   clone of the env (Rust dice bag copied byte-for-byte, via ``mcts.clone_env``).
3. **Memory safety (critical).** Hold only the CURRENT anchor game's reservoir of
   snapshots (capped) plus the transient clones for ONE position's evaluation at
   a time. When the anchor game ENDS, process its snapshots and DISCARD them
   before the next game (same bounded-memory streaming as the ceiling probe —
   NEVER a growing list of envs).
4. **Per position** (env clone held at a non-forced decision):
   (a) Run the deployed PUCT ``SearchAgent`` on a CLONE -> its chosen root action
       + the depth-0 visit concentration ``max(child_N)/sum(child_N)`` from the
       search diagnostics.
   (b) Candidate root actions = the root children PUCT expanded (above). For EACH
       candidate: deep-clone the position, APPLY the action, then run
       ``K=--rollouts`` INDEPENDENT MC rollouts to terminal under v8 on both seats
       (the FRESH-seeded ``StackedDice`` fix — deep-copy alone replays an
       identical dice future). ``q(a) = mean(agent win)``.
   (c) Record ``q_best = max_a q(a)``, ``q_puct = q(PUCT's action)``,
       ``regret = q_best - q_puct (>=0)``, ``agreement = [PUCT == argmax_a q(a)]``,
       ``n_candidates``, ``phase``, ``depth0_concentration``.
5. **Metrics** (1000-resample bootstrap 95% CIs): mean regret (overall +
   per-phase), agreement rate, mean depth-0 visit-concentration, and the regret
   distribution (median, p90).
6. **Verdict + JSON** to ``runs/probe_root_headroom.json`` + a printed table.
   **Pre-registered GO rule (FR-004 root-headroom analogue, stated + computed):**
     * GO    iff mean-regret bootstrap CI LOWER bound > 0.01 win-prob (PUCT
              demonstrably leaves >~1pp/move at the root) AND mean depth-0
              concentration > 0.7 (the documented visit collapse).
     * NO-GO  iff mean-regret CI UPPER bound < 0.01 (PUCT already near-best at the
              root) -> v8 + search is the practical ceiling.
     * INCONCLUSIVE otherwise (report the numbers).
7. **Sanity.** Report agreement rate + mean q_puct vs mean q_best; WARN if
   agreement is implausibly low (<0.3) or the q values are degenerate (all ~0/1,
   i.e. rollouts give no spread) — either signals a rollout-independence or
   action-set bug.

Constraints: CPU only, fixed seeds everywhere, additive (new file), no GUI
import, ruff + ``mypy --strict`` clean. Does NOT modify any existing code.

Usage (smoke — run here)::

    python scripts/dev/probe_root_headroom.py --n-states 4 --rollouts 4 --sims 25 \
        --out runs/probe_root_headroom_smoke.json

Usage (full — run by the human, NOT here; per-position search + many rollouts to
terminal is hours)::

    python scripts/dev/probe_root_headroom.py --n-states 120 --rollouts 24 --sims 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.priors import ActionTuple
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"
N_BOOTSTRAP = 1000

#: Pre-registered GO rule thresholds (FR-004).
REGRET_GO_LB = 0.01  # mean-regret CI lower bound above this (win-prob) => GO side
REGRET_NOGO_UB = 0.01  # mean-regret CI upper bound below this => NO-GO side
CONCENTRATION_HIGH = 0.70  # mean depth-0 visit concentration above this => collapse

#: Phase split by the acting agent's VP (whole-game mix: early/mid/late).
PHASE_EARLY_MAX_VP = 4  # early = agent vp <= 4
PHASE_LATE_MIN_VP = 10  # late  = agent vp >= 10  (mid is in between)
PHASES: tuple[str, ...] = ("early", "mid", "late")

#: Sanity thresholds (warnings, not gate inputs).
AGREEMENT_IMPLAUSIBLE_LOW = 0.30
Q_SPREAD_DEGENERATE = 1e-6  # std of all q values below this => no spread (bug)

#: Stride keeping per-rollout / per-game seed ranges disjoint.
_SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Snapshot:
    """A held mid-game position: the faithful env clone + its phase.

    Lives ONLY from capture until the owning anchor game ends (then it is
    searched + rolled out and discarded)."""

    phase: str
    env: CatanEnv  # faithful deep-clone (Rust dice copied); discarded after eval


@dataclass(slots=True)
class _StateResult:
    """One evaluated position's record (env already discarded)."""

    phase: str
    q_best: float  # max over candidate root actions of MC win-prob
    q_puct: float  # MC win-prob of PUCT's chosen root action
    regret: float  # CROSS-FIT debiased: q_est[argmax q_sel] - q_est[puct] (>= 0)
    regret_naive: float  # biased-high pooled-max regret (winner's-curse), for transparency
    agreement: int  # 1 iff (select-half argmax) == PUCT's action
    n_candidates: int
    depth0_concentration: float  # max child visit share at the root
    mc_n: int  # rollouts per candidate


def _phase_for_vp(vp: int) -> str:
    """early/mid/late phase label by the acting agent's VP (whole-game mix)."""
    if vp <= PHASE_EARLY_MAX_VP:
        return "early"
    if vp >= PHASE_LATE_MIN_VP:
        return "late"
    return "mid"


# ---------------------------------------------------------------------------
# Rollout — independent future to terminal under v8 on both seats
# ---------------------------------------------------------------------------


def _rollout_agent_win(
    start_env: CatanEnv,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    seed: int,
    safety_cap: int,
) -> float:
    """Play ONE independent future from ``start_env`` to terminal; 1.0/0.0.

    Independence (open-loop determinization, mirroring ``mcts._reseed`` extended
    to the dice — the MANDATORY fix): deep-clone via ``mcts.clone_env`` (so the
    live env is never mutated and the opponent net is not cloned), INSTALL A
    FRESH-seeded Rust ``StackedDice`` on the clone (deep-copying alone yields an
    IDENTICAL dice future, so we replace it for a genuinely independent draw), and
    reseed the global ``np.random`` / ``random`` streams + the opponent model.
    Both seats are v8: the agent acts via ``policy.sample`` and the opponent's
    whole turn is folded into the env's ``EndTurn`` via ``opponent``.
    """
    from catan_rl.engine.dice import StackedDice
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env
    from catan_rl.search.node import agent_outcome

    s32 = seed & 0x7FFF_FFFF
    env = clone_env(start_env, opponent)
    # Independent dice future: deepcopy yields an identical Rust bag, so swap in a
    # fresh-seeded StackedDice. The Karma state (last_player_to_roll_7) is part of
    # the cloned game state and is preserved; only the bag/RNG is re-randomized.
    assert env.game is not None
    env.game.dice = StackedDice(seed=s32)
    np.random.seed(s32)
    random.seed(s32)
    opponent.reset_rng(seed=s32)

    obs = env._get_obs()
    masks = env.get_action_masks()
    terminated = False
    truncated = False
    n_env_steps = 0
    while not terminated and not truncated:
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        with torch.no_grad():
            sample_out = policy.sample(obs_t, masks_t)
        action = sample_out["action"][0].cpu().numpy().astype(np.int64)
        obs, _, terminated, truncated, _ = env.step(action)
        masks = env.get_action_masks()
        n_env_steps += 1
        if n_env_steps > safety_cap:
            truncated = True
            break

    win = agent_outcome(env)  # 1.0 iff agent VP >= maxPoints and > opponent
    env.close()
    del env
    return win


def _q_for_root_action(
    position_env: CatanEnv,
    action: ActionTuple,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    base_seed: int,
    rollouts: int,
    safety_cap: int,
) -> tuple[float, float]:
    """MC win-prob of applying ``action`` at ``position_env`` then rolling out.

    Returns ``(q_select, q_estimate)`` — the means of two DISJOINT rollout halves
    (CROSS-FIT). Selecting the best action on ``q_select`` and reading its value
    off the independent ``q_estimate`` removes the max-selection (winner's-curse)
    upward bias that would otherwise dominate the small regret threshold.

    Deep-clone the position, APPLY the candidate root action once, then average
    ``rollouts`` INDEPENDENT futures. At most one transient rollout env is
    resident at a time (plus the post-action base clone, discarded after)."""
    from catan_rl.search.mcts import clone_env

    base = clone_env(position_env, opponent)
    base.step(np.asarray(action, dtype=np.int64))
    half = max(1, rollouts // 2)
    wins_sel = 0.0
    wins_est = 0.0
    for k in range(rollouts):
        w = _rollout_agent_win(
            base,
            policy,
            opponent,
            device=device,
            seed=(base_seed + k) & 0x7FFF_FFFF,
            safety_cap=safety_cap,
        )
        if k < half:
            wins_sel += w
        else:
            wins_est += w
    base.close()
    del base
    return (wins_sel / half, wins_est / max(1, rollouts - half))


# ---------------------------------------------------------------------------
# Per-position evaluation: PUCT decision vs the rollout oracle
# ---------------------------------------------------------------------------


def _evaluate_position(
    snap: _Snapshot,
    agent: SearchAgent,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    base_seed: int,
    rollouts: int,
    safety_cap: int,
) -> _StateResult | None:
    """Run PUCT + the rollout oracle on one held position; ``None`` if forced.

    (a) PUCT on a clone -> chosen root action + depth-0 visit concentration.
    (b) For each root candidate PUCT expanded, MC ``q(a)``.
    (c) regret / agreement / concentration.
    """
    from catan_rl.search.mcts import clone_env

    # (a) Deployed PUCT search on a faithful clone (SearchAgent clones internally
    # too, but we never want to risk mutating the held snapshot env).
    search_env = clone_env(snap.env, opponent)
    puct_action_arr = agent.choose_action(search_env)
    diag = agent.last_diagnostics
    search_env.close()
    del search_env

    if diag.get("forced", False):
        return None  # forced root: no decision to make, not a regret sample
    visit_counts: dict[ActionTuple, int] = diag.get("visit_counts", {})
    if not visit_counts:
        return None  # no sims landed (degenerate) — skip

    total_visits = sum(visit_counts.values())
    if total_visits <= 0:
        return None
    depth0_conc = max(visit_counts.values()) / total_visits

    puct_action: ActionTuple = cast("ActionTuple", tuple(int(x) for x in puct_action_arr.tolist()))
    # Candidate root action set = the root children PUCT expanded (documented).
    candidates: list[ActionTuple] = list(visit_counts.keys())
    if puct_action not in candidates:
        # Should not happen (best is chosen from this very set); be robust.
        candidates.append(puct_action)

    # (b) Rollout oracle q(a) for each candidate. Disjoint per-candidate seed
    # ranges so the K rollouts of distinct candidates are independent draws.
    q_sel: dict[ActionTuple, float] = {}
    q_est: dict[ActionTuple, float] = {}
    for ci, action in enumerate(candidates):
        cand_seed = (base_seed + ci * _SEED_STRIDE) & 0x7FFF_FFFF
        qs, qe = _q_for_root_action(
            snap.env,
            action,
            policy,
            opponent,
            device=device,
            base_seed=cand_seed,
            rollouts=rollouts,
            safety_cap=safety_cap,
        )
        q_sel[action] = qs
        q_est[action] = qe

    # CROSS-FIT (debiased): pick the oracle's best action on the SELECT half, read
    # its value (and PUCT's) off the disjoint ESTIMATE half -> no winner's-curse.
    oracle_action = max(q_sel, key=lambda a: q_sel[a])
    q_best = q_est[oracle_action]
    q_puct = q_est[puct_action]
    regret = max(0.0, q_best - q_puct)
    agreement = 1 if oracle_action == puct_action else 0
    # Naive (biased-high) regret over pooled means — transparency / upper bound.
    q_pool = {a: 0.5 * (q_sel[a] + q_est[a]) for a in candidates}
    regret_naive = max(0.0, max(q_pool.values()) - q_pool[puct_action])
    return _StateResult(
        phase=snap.phase,
        q_best=q_best,
        q_puct=q_puct,
        regret=regret,
        regret_naive=regret_naive,
        agreement=agreement,
        n_candidates=len(candidates),
        depth0_concentration=depth0_conc,
        mc_n=rollouts,
    )


# ---------------------------------------------------------------------------
# Anchor game — reservoir-sample non-forced decisions across all phases
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GameSnapshots:
    """Per-game reservoir of sampled non-forced positions + the game outcome."""

    snaps: list[_Snapshot] = field(default_factory=list)


def _play_anchor_game(
    env: CatanEnv,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    seed: int,
    agent_seat: int,
    per_game_cap: int,
) -> _GameSnapshots:
    """Play one v8-vs-v8 RAW anchor game; reservoir-sample non-forced positions.

    Mirrors the ceiling probe's ``_play_anchor_game`` loop. At each NON-FORCED
    agent decision (``masks["type"].sum() > 1``, ANY phase) it reservoir-samples
    up to ``per_game_cap`` positions uniformly over the game's eligible decisions
    (seeded). Snapshots are FAITHFUL env clones taken BEFORE the agent's action is
    applied (the state the root decision is made at)."""
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env

    obs, _ = env.reset(seed=seed, options={"agent_seat": agent_seat})
    masks = env.get_action_masks()
    held = _GameSnapshots()
    # Local reservoir RNG (does not perturb the engine's global streams).
    res_rng = random.Random(seed ^ 0x5DEECE66D)
    seen = 0

    terminated = False
    truncated = False
    n_env_steps = 0
    safety_cap = env.max_turns * 50
    while not terminated and not truncated:
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        with torch.no_grad():
            sample_out = policy.sample(obs_t, masks_t)
        action = sample_out["action"][0].cpu().numpy().astype(np.int64)

        if int(masks["type"].sum()) > 1:
            seen += 1
            vp = int(getattr(env.agent_player, "victoryPoints", 0))
            phase = _phase_for_vp(vp)
            snap = _Snapshot(phase=phase, env=clone_env(env, opponent))
            # Reservoir sampling of size ``per_game_cap`` over the game's eligible
            # decisions (uniform, seeded, reproducible).
            if len(held.snaps) < per_game_cap:
                held.snaps.append(snap)
            else:
                j = res_rng.randint(0, seen - 1)
                if j < per_game_cap:
                    held.snaps[j].env.close()
                    held.snaps[j] = snap
                else:
                    snap.env.close()  # not kept — free the clone immediately

        obs, _, terminated, truncated, _ = env.step(action)
        masks = env.get_action_masks()
        n_env_steps += 1
        if n_env_steps > safety_cap:
            truncated = True
            break

    return held


# ---------------------------------------------------------------------------
# Driver — generate state results with bounded memory
# ---------------------------------------------------------------------------


def generate_state_results(
    *,
    n_states: int,
    rollouts: int,
    sims: int,
    seed: int,
    device: torch.device,
) -> list[_StateResult]:
    """Play anchor games until ~``n_states`` positions are evaluated.

    Memory bound: only the current game's reservoir of <=``per_game_cap``
    snapshots is resident, plus the transient clones of ONE position's evaluation.
    After each game we evaluate its snapshots, append the records, and ``del`` the
    snapshot envs before the next game (no growing env list)."""
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = cast("CatanPolicy", actor.policy)
    opponent = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(opponent)
    rollout_safety_cap = env.max_turns * 50

    # DEPLOYED default search config: fpu_mode='zero', n_determinizations=1,
    # c_puct=1.5, ... at the requested sim budget (matches scripts/elo_ladder.py
    # and cli/search_eval.py's deployed-search construction).
    cfg = SearchConfig(sims_per_move=sims, seed=seed)
    search_agent = SearchAgent(policy, cfg, device=device)

    # A handful of positions per game keeps memory bounded while sampling the
    # whole-game phase mix (a single position/game would over-sample the same
    # opening states across games).
    per_game_cap = max(1, min(4, n_states))

    results: list[_StateResult] = []
    t0 = time.time()
    game_id = 0
    while len(results) < n_states:
        game_seed = (seed * _SEED_STRIDE + game_id) % (2**31 - 1)
        agent_seat = game_id % 2
        opponent.reset_rng(seed=game_seed)
        held = _play_anchor_game(
            env,
            policy,
            opponent,
            device=device,
            seed=game_seed,
            agent_seat=agent_seat,
            per_game_cap=per_game_cap,
        )
        for si, snap in enumerate(held.snaps):
            if len(results) >= n_states:
                snap.env.close()
                continue
            eval_base = (game_seed * 7919 + si * 104_729) & 0x7FFF_FFFF
            res = _evaluate_position(
                snap,
                search_agent,
                policy,
                opponent,
                device=device,
                base_seed=eval_base,
                rollouts=rollouts,
                safety_cap=rollout_safety_cap,
            )
            if res is not None:
                results.append(res)
            snap.env.close()
        # Bounded memory: drop all references to this game's snapshot envs.
        held.snaps.clear()
        del held
        game_id += 1
        pc = {ph: sum(1 for r in results if r.phase == ph) for ph in PHASES}
        print(
            f"[probe] game {game_id}: results={len(results)} phase={pc} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if game_id > max(50, n_states * 4):
            print(
                f"[probe] WARNING: stopping after {game_id} games with only {len(results)} states.",
                file=sys.stderr,
                flush=True,
            )
            break

    env.close()
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(x: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    """95% bootstrap CI on the mean of ``x``."""
    n = x.size
    if n < 2:
        return (float(x.mean()) if n else 0.0, float(x.mean()) if n else 0.0)
    means = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        means[i] = float(x[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _subset_metrics(rows: list[_StateResult], *, seed: int) -> dict[str, Any]:
    """Regret / agreement / concentration summary + CIs for a subset."""
    rng = np.random.default_rng(seed)
    regret = np.array([r.regret for r in rows], dtype=np.float64)
    regret_naive = np.array([r.regret_naive for r in rows], dtype=np.float64)
    agree = np.array([float(r.agreement) for r in rows], dtype=np.float64)
    conc = np.array([r.depth0_concentration for r in rows], dtype=np.float64)
    q_best = np.array([r.q_best for r in rows], dtype=np.float64)
    q_puct = np.array([r.q_puct for r in rows], dtype=np.float64)

    regret_lo, regret_hi = _bootstrap_mean_ci(regret, rng=rng)
    agree_lo, agree_hi = _bootstrap_mean_ci(agree, rng=rng)
    conc_lo, conc_hi = _bootstrap_mean_ci(conc, rng=rng)
    return {
        "n": int(regret.size),
        "mean_regret": float(regret.mean()),
        "mean_regret_ci": [regret_lo, regret_hi],
        "mean_regret_naive_biased": float(regret_naive.mean()),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
        "agreement_rate": float(agree.mean()),
        "agreement_rate_ci": [agree_lo, agree_hi],
        "mean_depth0_concentration": float(conc.mean()),
        "mean_depth0_concentration_ci": [conc_lo, conc_hi],
        "mean_q_best": float(q_best.mean()),
        "mean_q_puct": float(q_puct.mean()),
    }


def _verdict(overall: dict[str, Any]) -> tuple[str, str]:
    """Pre-registered GO / NO-GO / INCONCLUSIVE from the overall metrics."""
    regret_lo, regret_hi = overall["mean_regret_ci"]
    conc = overall["mean_depth0_concentration"]
    if regret_lo > REGRET_GO_LB and conc > CONCENTRATION_HIGH:
        return (
            "GO",
            f"mean-regret CI lower bound {regret_lo:+.4f} > {REGRET_GO_LB} "
            f"(PUCT leaves >~1pp/move at the root) AND mean depth-0 concentration "
            f"{conc:.3f} > {CONCENTRATION_HIGH} (documented visit collapse) -> a "
            f"better root decision rule (Gumbel, STAGE-B) is worth building.",
        )
    if regret_hi < REGRET_NOGO_UB:
        return (
            "NO-GO",
            f"mean-regret CI upper bound {regret_hi:+.4f} < {REGRET_NOGO_UB} "
            f"(PUCT is already near-best at the root) -> v8 + search is the "
            f"practical ceiling; do NOT build Gumbel.",
        )
    bits = []
    if not (regret_lo > REGRET_GO_LB):
        bits.append(f"regret CI lower bound {regret_lo:+.4f} <= {REGRET_GO_LB}")
    if not (conc > CONCENTRATION_HIGH):
        bits.append(f"depth-0 concentration {conc:.3f} <= {CONCENTRATION_HIGH}")
    if not (regret_hi < REGRET_NOGO_UB):
        bits.append(f"regret CI upper bound {regret_hi:+.4f} >= {REGRET_NOGO_UB}")
    return (
        "INCONCLUSIVE",
        "neither GO nor NO-GO cleanly met (" + "; ".join(bits) + ") -> report the "
        "numbers; consider more states/rollouts.",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_probe(
    *, n_states: int, rollouts: int, sims: int, seed: int, out_path: Path
) -> dict[str, Any]:
    """Full probe: generate -> per-phase + overall metrics -> verdict -> JSON."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    results = generate_state_results(
        n_states=n_states, rollouts=rollouts, sims=sims, seed=seed, device=device
    )
    if not results:
        raise RuntimeError("no states evaluated — increase --n-states / check config")

    overall = _subset_metrics(results, seed=seed + 17)
    ov_verdict, ov_reason = _verdict(overall)
    overall["verdict"] = ov_verdict
    overall["reason"] = ov_reason

    per_phase: dict[str, dict[str, Any]] = {}
    for ph in PHASES:
        rows = [r for r in results if r.phase == ph]
        if not rows:
            per_phase[ph] = {"n": 0}
            continue
        per_phase[ph] = _subset_metrics(rows, seed=seed + 13 + hash(ph) % 97)

    # --- sanity: agreement + q spread (a rollout-independence / action-set check) ---
    all_q = np.array([r.q_best for r in results] + [r.q_puct for r in results], dtype=np.float64)
    q_std = float(all_q.std())
    sanity: dict[str, Any] = {
        "agreement_rate": overall["agreement_rate"],
        "mean_q_best": overall["mean_q_best"],
        "mean_q_puct": overall["mean_q_puct"],
        "q_std": q_std,
        "warnings": [],
    }
    if overall["agreement_rate"] < AGREEMENT_IMPLAUSIBLE_LOW:
        sanity["warnings"].append(
            f"agreement {overall['agreement_rate']:.3f} < {AGREEMENT_IMPLAUSIBLE_LOW} "
            "(implausibly low — likely a rollout-independence or action-set bug)"
        )
    if q_std < Q_SPREAD_DEGENERATE:
        sanity["warnings"].append(
            f"q std {q_std:.2e} ~ 0 (rollouts give NO spread / all 0 or 1 — likely a "
            "rollout-independence bug; rollouts must produce a spread of win-probs)"
        )
    sanity["ok"] = not sanity["warnings"]

    result: dict[str, Any] = {
        "probe": "root-headroom probe (spec 008 STAGE-A kill-gate, FR-004)",
        "ckpt": V8_CKPT,
        "config": {
            "n_states": n_states,
            "rollouts": rollouts,
            "sims": sims,
            "seed": seed,
            "n_bootstrap": N_BOOTSTRAP,
            "search_config": "DEPLOYED default (fpu_mode=zero, n_det=1, c_puct=1.5)",
            "candidate_set": "root children PUCT expanded (modal sub-action per legal type)",
            "phase_boundaries": {
                "early_max_vp": PHASE_EARLY_MAX_VP,
                "late_min_vp": PHASE_LATE_MIN_VP,
            },
            "go_rule": {
                "regret_go_lb": REGRET_GO_LB,
                "regret_nogo_ub": REGRET_NOGO_UB,
                "concentration_high": CONCENTRATION_HIGH,
            },
        },
        "data": {
            "n_states": len(results),
            "phase_counts": {ph: sum(1 for r in results if r.phase == ph) for ph in PHASES},
            "mean_n_candidates": float(np.mean([r.n_candidates for r in results])),
        },
        "per_phase": per_phase,
        "overall": overall,
        "sanity": sanity,
        "verdict": ov_verdict,
        "reason": ov_reason,
        "conclusion": _plain_conclusion(ov_verdict),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _print_report(result)
    print(f"\n[probe] wrote {out_path}", flush=True)
    return result


def _plain_conclusion(verdict: str) -> str:
    if verdict == "GO":
        return (
            "GO => the deployed PUCT search demonstrably leaves win-probability on "
            "the table at the root (regret CI clears ~1pp AND visits collapse onto "
            "one action) -> build the Gumbel root decision rule (STAGE-B / US1)."
        )
    if verdict == "NO-GO":
        return (
            "NO-GO => the deployed PUCT search already picks near-best root moves "
            "(regret CI below ~1pp) -> there is no root-decision headroom; v8 + "
            "search is at the practical ceiling. Do NOT build Gumbel; the bottleneck "
            "is elsewhere (chance/belief or a true ceiling)."
        )
    return (
        "INCONCLUSIVE => the regret CI does not cleanly clear the GO or NO-GO bar. "
        "Report the numbers; rerun with more --n-states / --rollouts to tighten the CI."
    )


def _print_report(result: dict[str, Any]) -> None:
    """Human-readable summary to stdout."""
    print("\n" + "=" * 86)
    print("ROOT-HEADROOM PROBE — RESULT (spec 008 STAGE-A kill-gate)")
    print("=" * 86)
    data = result["data"]
    print(
        f"states={data['n_states']}  phase_counts={data['phase_counts']}  "
        f"mean_candidates={data['mean_n_candidates']:.2f}"
    )
    san = result["sanity"]
    flag = "OK" if san["ok"] else "WARNING"
    print(
        f"sanity [{flag}]: agreement={san['agreement_rate']:.3f}  "
        f"q_best={san['mean_q_best']:.3f}  q_puct={san['mean_q_puct']:.3f}  "
        f"q_std={san['q_std']:.4f}"
    )
    for w in san["warnings"]:
        print(f"  *** WARNING: {w} ***", file=sys.stderr, flush=True)
    print("-" * 86)
    hdr = (
        f"{'subset':>8} {'n':>4} {'regret':>8} {'regret_CI':>17} "
        f"{'median':>7} {'p90':>7} {'agree':>6} {'conc':>6}"
    )
    print(hdr)
    rows: list[tuple[str, dict[str, Any]]] = [
        *((ph, result["per_phase"][ph]) for ph in PHASES),
        ("overall", result["overall"]),
    ]
    for name, m in rows:
        if m.get("n", 0) == 0:
            print(f"{name:>8} {0:>4}  (no data)")
            continue
        ci = m["mean_regret_ci"]
        print(
            f"{name:>8} {m['n']:>4} {m['mean_regret']:>8.4f} "
            f"[{ci[0]:>6.4f},{ci[1]:>6.4f}] {m['median_regret']:>7.4f} "
            f"{m['p90_regret']:>7.4f} {m['agreement_rate']:>6.3f} "
            f"{m['mean_depth0_concentration']:>6.3f}"
        )
    print("-" * 86)
    print(f"VERDICT: {result['verdict']}")
    print(result["reason"])
    print(result["conclusion"])
    print("=" * 86)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-states",
        type=int,
        default=120,
        help="target number of positions to evaluate (default 120; use 4 for smoke)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=40,
        help="MC rollouts/candidate (default 40 = 20 select + 20 est cross-fit; 4 smoke)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=100,
        help="PUCT simulation budget per move (deployed default 100; 25 for smoke)",
    )
    parser.add_argument("--seed", type=int, default=0, help="master seed (default 0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/probe_root_headroom.json"),
        help="output JSON path (default runs/probe_root_headroom.json)",
    )
    args = parser.parse_args(argv)
    run_probe(
        n_states=args.n_states,
        rollouts=args.rollouts,
        sims=args.sims,
        seed=args.seed,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
