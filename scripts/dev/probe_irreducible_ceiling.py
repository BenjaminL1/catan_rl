"""Irreducible-ceiling probe (spec 007 follow-up to US0).

Standalone, additive diagnostic. Decides WHY v8's value head ranks EARLY
(agent ``vp<=4``) and LATE (agent ``vp>=10``) states poorly against the realized
game outcome (US0: Spearman ~0.47 early / ~0.50 late vs ~0.72 mid).

Two hypotheses for the low realized-rank on early/late states:

(a) **CAPACITY-HEADROOM** — the frozen trunk UNDER-REPRESENTS these states, so a
    better evaluator *could* rank them. Then a stronger estimate of each state's
    TRUE win-prob would rank the realized outcome much better than the net does
    (large headroom), and/or the net would disagree with that better estimate.

(b) **IRREDUCIBLE** — early/late states are INHERENTLY high-variance (the dice
    bag + opponent draws dominate the eventual result), so NO evaluator can rank
    a *single* realized outcome any better than the net already does. Then the
    best possible estimator (the Monte-Carlo win-prob) ranks the single realized
    outcome about as poorly as the net does (small headroom), AND the net already
    matches that MC win-prob closely (high net_accuracy). The low realized-rank
    is then coin-flip noise, not a representation gap — value work is dead and
    v8 + search is the ceiling.

Method
------
1. Load v8 (``runs/anchors/v8_promobar_u243.pt``) via the recorder factory
   (same load path as US0 / the search bake-off).
2. **Anchor games.** Play v8-vs-v8 RAW games (no search), seat-symmetric, fixed
   seed (US0's ``_play_one_game`` pattern: ``policy.sample`` agent +
   ``FrozenSnapshotOpponent`` on the other seat, folded into the env's
   ``EndTurn``). During a game, at NON-FORCED agent decisions
   (``masks["type"].sum() > 1``) that are EARLY (``vp<=4``) or LATE (``vp>=10``),
   reservoir-cap at most ONE early + ONE late snapshot per game (seeded), so the
   total ~= ``--n-states``. For each kept decision record (i) the net's squashed
   win-prob at that state and (ii) a FAITHFUL deep-clone of the env (Rust dice
   bag copied byte-for-byte, via ``mcts.clone_env``).
3. **Memory safety (critical).** Hold only the CURRENT anchor game's handful of
   snapshots plus ``K`` transient rollout clones at a time. When the anchor game
   ENDS, record the realized agent outcome (1.0 win / 0.0 loss); then for each
   held snapshot do ``K = --rollouts`` INDEPENDENT rollouts to terminal under v8
   on BOTH seats (each rollout: deep-clone the snapshot, install a FRESH-seeded
   Rust dice bag for an independent dice future, reseed the global RNG + opponent
   — the open-loop determinization of ``mcts._reseed`` extended to the dice).
   ``mc_winprob = mean(rollout agent-wins)``. Then DISCARD all snapshots before
   the next anchor game (``del`` + no growing env lists).
4. **Metrics** per phase (early, late) + overall, 1000-resample bootstrap 95% CI:
   * ``ceiling_rank`` = Spearman(mc_winprob, realized_outcome)  — the BEST
     estimator vs a single realized outcome = the irreducible ceiling.
   * ``net_rank``     = Spearman(net_value,  realized_outcome)  — should
     reproduce US0's ~0.47 early / ~0.50 late (SANITY ANCHOR).
   * ``net_accuracy`` = Spearman(net_value,  mc_winprob)        — net vs the
     near-ground-truth MC win-prob (no single-realization noise).
   * ``headroom`` = ceiling_rank - net_rank (paired bootstrap CI).
5. **Verdict** (printed + JSON):
   * IRREDUCIBLE iff (on early/late) the headroom CI is small / includes ~0 AND
     net_accuracy is high -> coin-flip noise, capacity won't help, value work dead.
   * CAPACITY-HEADROOM iff the headroom CI lower bound clears ~0.05 AND/OR
     net_accuracy is low -> the trunk under-represents, capacity worth trying.

Constraints: CPU only, fixed seeds everywhere, additive (new file), no GUI
import, ruff + ``mypy --strict`` clean.

Usage (smoke — run here)::

    python scripts/dev/probe_irreducible_ceiling.py --n-states 6 --rollouts 4 \
        --out runs/probe_ceiling_smoke.json

Usage (full — run by the human, NOT here; early-state rollouts are near-full
games, so this is hours)::

    python scripts/dev/probe_irreducible_ceiling.py --n-states 160 --rollouts 24
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
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"
N_BOOTSTRAP = 1000
#: US0's per-phase net-vs-outcome Spearman (the sanity anchor for ``net_rank``).
US0_NET_RANK = {"early": 0.47, "late": 0.50}
SANITY_TOL = 0.20  # |net_rank(phase) - US0 value| must be < this (loose: small-n)
#: Headroom CI lower bound above this => CAPACITY-HEADROOM on that phase.
HEADROOM_CLEAR = 0.05
#: net_accuracy at/above this counts as "net already matches MC" (IRREDUCIBLE-side).
NET_ACCURACY_HIGH = 0.70

PHASE_EARLY_MAX_VP = 4  # early = agent vp <= 4
PHASE_LATE_MIN_VP = 10  # late  = agent vp >= 10
PHASES: tuple[str, ...] = ("early", "late")

#: Stride keeping per-rollout seed ranges disjoint across snapshots/games.
_SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Snapshot:
    """A held mid-game snapshot: the faithful env clone + its (phase, net_value).

    Lives ONLY from capture until the owning anchor game ends (then it is rolled
    out and discarded). ``net_value`` is the squashed [0,1] win-prob the net
    assigns at this exact state."""

    phase: str
    net_value: float
    env: CatanEnv  # faithful deep-clone (Rust dice copied); discarded after rollout


@dataclass(slots=True)
class _StateResult:
    """One finished snapshot's record (env already discarded)."""

    phase: str
    realized_outcome: float  # 0/1 outcome of the ANCHOR game (agent-POV)
    net_value: float  # squashed [0,1] net win-prob at the snapshot state
    mc_winprob: float  # mean agent-win over K independent rollouts
    mc_n: int


def _phase_for_vp(vp: int) -> str | None:
    """early/late phase label, or ``None`` for mid states (not sampled)."""
    if vp <= PHASE_EARLY_MAX_VP:
        return "early"
    if vp >= PHASE_LATE_MIN_VP:
        return "late"
    return None


# ---------------------------------------------------------------------------
# Rollout — independent future to terminal under v8 on both seats
# ---------------------------------------------------------------------------


def _rollout_agent_win(
    snapshot_env: CatanEnv,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    seed: int,
    safety_cap: int,
) -> float:
    """Play ONE independent future from ``snapshot_env`` to terminal; 1.0/0.0.

    Independence (open-loop determinization, mirroring ``mcts._reseed`` and
    extending it to the dice): deep-clone the snapshot via ``mcts.clone_env`` (so
    the live snapshot env is never mutated and the opponent net is not cloned),
    INSTALL A FRESH-seeded Rust dice bag on the clone (deep-copying alone yields
    an IDENTICAL dice future, so we replace it for a genuinely independent draw),
    and reseed the global ``np.random`` / ``random`` streams + the opponent model.
    Both seats are v8: the agent acts via ``policy.sample`` and the opponent's
    whole turn is folded into the env's ``EndTurn`` via ``opponent``.
    """
    from catan_rl.engine.dice import StackedDice
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env
    from catan_rl.search.node import agent_outcome

    s32 = seed & 0x7FFF_FFFF
    env = clone_env(snapshot_env, opponent)
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


def _mc_winprob_for_snapshot(
    snap: _Snapshot,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    base_seed: int,
    rollouts: int,
    safety_cap: int,
) -> float:
    """Mean agent-win over ``rollouts`` INDEPENDENT futures from the snapshot.

    Each rollout clone is created, played, and freed in turn — at most one
    transient rollout env is resident at a time (plus the held snapshot)."""
    wins = 0.0
    for k in range(rollouts):
        wins += _rollout_agent_win(
            snap.env,
            policy,
            opponent,
            device=device,
            seed=(base_seed + k) & 0x7FFF_FFFF,
            safety_cap=safety_cap,
        )
    return wins / max(1, rollouts)


# ---------------------------------------------------------------------------
# Anchor game — record early/late snapshots, capped per game
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GameSnapshots:
    """Per-game reservoir: at most one early + one late snapshot, plus outcome."""

    by_phase: dict[str, _Snapshot] = field(default_factory=dict)


def _play_anchor_game(
    env: CatanEnv,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    seed: int,
    agent_seat: int,
) -> tuple[_GameSnapshots, float]:
    """Play one v8-vs-v8 RAW anchor game; return (>=1 phase snapshots, outcome).

    Mirrors US0's ``_play_one_game`` loop. At each NON-FORCED agent decision in an
    early/late phase, reservoir-keep at most one snapshot per phase per game
    (seeded coin flip on collision so the kept state isn't always the first).
    Snapshots are FAITHFUL env clones taken BEFORE the agent's action is applied
    (the state the net/decision is evaluated at)."""
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env
    from catan_rl.search.node import agent_outcome
    from catan_rl.search.value import value_from_obs

    obs, _ = env.reset(seed=seed, options={"agent_seat": agent_seat})
    masks = env.get_action_masks()
    held = _GameSnapshots()
    # Local reservoir RNG (does not perturb the engine's global streams).
    res_rng = random.Random(seed ^ 0x5DEECE66D)
    seen_count: dict[str, int] = {"early": 0, "late": 0}

    terminated = False
    truncated = False
    n_env_steps = 0
    safety_cap = env.max_turns * 50
    hit_safety = False
    while not terminated and not truncated:
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        with torch.no_grad():
            sample_out = policy.sample(obs_t, masks_t)
        action = sample_out["action"][0].cpu().numpy().astype(np.int64)

        if int(masks["type"].sum()) > 1:
            vp = int(getattr(env.agent_player, "victoryPoints", 0))
            phase = _phase_for_vp(vp)
            if phase is not None:
                seen_count[phase] += 1
                # Reservoir sampling of size 1 per phase: keep the n-th eligible
                # state with probability 1/n -> uniform over the game's eligible
                # states for that phase (seeded, reproducible).
                keep = (phase not in held.by_phase) or (res_rng.random() < 1.0 / seen_count[phase])
                if keep:
                    net_value = value_from_obs(policy, obs, device=device)
                    held.by_phase[phase] = _Snapshot(
                        phase=phase,
                        net_value=net_value,
                        env=clone_env(env, opponent),  # faithful clone (Rust dice copied)
                    )

        obs, _, terminated, truncated, _ = env.step(action)
        masks = env.get_action_masks()
        n_env_steps += 1
        if n_env_steps > safety_cap:
            truncated = True
            hit_safety = True
            break

    outcome = 0.0 if hit_safety else agent_outcome(env)
    return held, outcome


# ---------------------------------------------------------------------------
# Driver — generate state results with bounded memory
# ---------------------------------------------------------------------------


def generate_state_results(
    *,
    n_states: int,
    rollouts: int,
    seed: int,
    device: torch.device,
) -> list[_StateResult]:
    """Play anchor games until ~``n_states`` early/late snapshots are rolled out.

    Memory bound: only the current game's <=2 snapshots are resident, plus a
    single transient rollout clone. After each game we roll out its snapshots,
    append the finished records, and ``del`` the snapshot envs before the next
    game (no growing env list)."""
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
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

    results: list[_StateResult] = []
    t0 = time.time()
    game_id = 0
    while len(results) < n_states:
        game_seed = (seed * _SEED_STRIDE + game_id) % (2**31 - 1)
        agent_seat = game_id % 2
        opponent.reset_rng(seed=game_seed)
        held, outcome = _play_anchor_game(
            env,
            policy,
            opponent,
            device=device,
            seed=game_seed,
            agent_seat=agent_seat,
        )
        # Roll out each held snapshot to its MC win-prob, then discard the envs.
        for phase, snap in held.by_phase.items():
            roll_base = (game_seed * 7919 + (0 if phase == "early" else 1) * 104_729) & 0x7FFF_FFFF
            mc = _mc_winprob_for_snapshot(
                snap,
                policy,
                opponent,
                device=device,
                base_seed=roll_base,
                rollouts=rollouts,
                safety_cap=rollout_safety_cap,
            )
            results.append(
                _StateResult(
                    phase=phase,
                    realized_outcome=outcome,
                    net_value=snap.net_value,
                    mc_winprob=mc,
                    mc_n=rollouts,
                )
            )
            snap.env.close()
        # Bounded memory: drop all references to this game's snapshot envs.
        held.by_phase.clear()
        del held
        game_id += 1
        n_early = sum(1 for r in results if r.phase == "early")
        n_late = sum(1 for r in results if r.phase == "late")
        print(
            f"[probe] game {game_id}: results={len(results)} "
            f"(early={n_early} late={n_late}) outcome={outcome:.0f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        # Safety: avoid an unbounded loop if a phase is never reached.
        if game_id > max(50, n_states * 8):
            print(
                f"[probe] WARNING: stopping after {game_id} games with only "
                f"{len(results)} states (phase may be rare at this scale).",
                file=sys.stderr,
                flush=True,
            )
            break

    env.close()
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank-correlation; NaN-safe (degenerate / single-value -> 0.0)."""
    if a.size < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return 0.0
    rho = spearmanr(a, b).statistic
    return 0.0 if (rho is None or np.isnan(rho)) else float(rho)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    """95% bootstrap CI on Spearman(a, b)."""
    n = a.size
    if n < 2:
        return (0.0, 0.0)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        samples[i] = _spearman(a[idx], b[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))


def _headroom_ci(
    mc: np.ndarray,
    net: np.ndarray,
    outcome: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on headroom = Spearman(mc,outcome) - Spearman(net,outcome).

    Same resampled rows for both terms (paired) so the CI reflects the ceiling-vs-net
    gap, not the sum of two independent sampling variances. Returns (point, lo, hi)."""
    n = outcome.size
    point = _spearman(mc, outcome) - _spearman(net, outcome)
    if n < 2:
        return (point, 0.0, 0.0)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        oc = outcome[idx]
        samples[i] = _spearman(mc[idx], oc) - _spearman(net[idx], oc)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _metrics_for_subset(
    mc: np.ndarray,
    net: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """All three ranks + CIs + headroom for one subset (a phase or overall)."""
    rng = np.random.default_rng(seed)
    ceiling_rank = _spearman(mc, outcome)
    net_rank = _spearman(net, outcome)
    net_accuracy = _spearman(net, mc)
    ceiling_ci = _bootstrap_ci(mc, outcome, rng=rng)
    net_rank_ci = _bootstrap_ci(net, outcome, rng=rng)
    net_acc_ci = _bootstrap_ci(net, mc, rng=rng)
    hr_pt, hr_lo, hr_hi = _headroom_ci(mc, net, outcome, rng=rng)
    return {
        "n": int(mc.size),
        "ceiling_rank": ceiling_rank,
        "ceiling_rank_ci": [ceiling_ci[0], ceiling_ci[1]],
        "net_rank": net_rank,
        "net_rank_ci": [net_rank_ci[0], net_rank_ci[1]],
        "net_accuracy": net_accuracy,
        "net_accuracy_ci": [net_acc_ci[0], net_acc_ci[1]],
        "headroom": hr_pt,
        "headroom_ci": [hr_lo, hr_hi],
    }


def _verdict_for_phase(m: dict[str, Any]) -> tuple[str, str]:
    """Per-phase verdict from the headroom CI + net_accuracy."""
    headroom_lo = m["headroom_ci"][0]
    net_acc = m["net_accuracy"]
    capacity = (headroom_lo > HEADROOM_CLEAR) or (net_acc < NET_ACCURACY_HIGH)
    if capacity:
        why = (
            f"headroom CI lower bound {headroom_lo:+.3f} clears {HEADROOM_CLEAR} "
            if headroom_lo > HEADROOM_CLEAR
            else ""
        ) + (
            f"net_accuracy {net_acc:.3f} below {NET_ACCURACY_HIGH}"
            if net_acc < NET_ACCURACY_HIGH
            else ""
        )
        return "CAPACITY-HEADROOM", why.strip()
    return (
        "IRREDUCIBLE",
        f"headroom CI lower bound {headroom_lo:+.3f} <= {HEADROOM_CLEAR} (best estimator "
        f"barely out-ranks the net) AND net_accuracy {net_acc:.3f} >= {NET_ACCURACY_HIGH} "
        f"(net already matches the MC win-prob)",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_probe(*, n_states: int, rollouts: int, seed: int, out_path: Path) -> dict[str, Any]:
    """Full probe: generate -> per-phase + overall metrics -> verdict -> JSON."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    results = generate_state_results(n_states=n_states, rollouts=rollouts, seed=seed, device=device)
    if not results:
        raise RuntimeError("no states recorded — increase --n-states / check phases")

    phase_arr = np.array([r.phase for r in results])
    net = np.array([r.net_value for r in results], dtype=np.float64)
    mc = np.array([r.mc_winprob for r in results], dtype=np.float64)
    outcome = np.array([r.realized_outcome for r in results], dtype=np.float64)

    phase_counts = {ph: int(np.sum(phase_arr == ph)) for ph in PHASES}
    print(
        f"[probe] {len(results)} states | phase counts {phase_counts} | "
        f"anchor outcome rate {float(np.mean(outcome)):.3f}",
        flush=True,
    )

    per_phase: dict[str, dict[str, Any]] = {}
    for ph in PHASES:
        mask = phase_arr == ph
        if int(mask.sum()) == 0:
            per_phase[ph] = {"n": 0, "verdict": "NO-DATA", "reason": "no states in phase"}
            continue
        m = _metrics_for_subset(mc[mask], net[mask], outcome[mask], seed=seed + 13)
        v, why = _verdict_for_phase(m)
        m["verdict"] = v
        m["reason"] = why
        per_phase[ph] = m

    overall = _metrics_for_subset(mc, net, outcome, seed=seed + 17)
    ov_verdict, ov_why = _verdict_for_phase(overall)
    overall["verdict"] = ov_verdict
    overall["reason"] = ov_why

    # --- sanity anchor: net_rank on early/late should land near US0's values ---
    sanity: dict[str, Any] = {"tol": SANITY_TOL, "us0_net_rank": US0_NET_RANK, "phases": {}}
    sanity_ok = True
    for ph in PHASES:
        if per_phase[ph].get("n", 0) == 0:
            continue
        measured = per_phase[ph]["net_rank"]
        expected = US0_NET_RANK[ph]
        within = abs(measured - expected) < SANITY_TOL
        sanity_ok = sanity_ok and within
        sanity["phases"][ph] = {
            "measured_net_rank": measured,
            "us0_expected": expected,
            "within_tol": within,
        }
    sanity["all_within_tol"] = sanity_ok
    if not sanity_ok:
        print(
            "\n*** WARNING: net_rank on early/late is NOT within "
            f"{SANITY_TOL} of US0's (~0.47 early / ~0.50 late) — the probe may be "
            "buggy (small-n noise IS expected at tiny --n-states). ***\n",
            file=sys.stderr,
            flush=True,
        )

    # --- overall study verdict: drive off the early/late phases (the question) ---
    phase_verdicts = [per_phase[ph]["verdict"] for ph in PHASES if per_phase[ph].get("n", 0) > 0]
    any_capacity = any(v == "CAPACITY-HEADROOM" for v in phase_verdicts)
    study_verdict = "CAPACITY-HEADROOM" if any_capacity else "IRREDUCIBLE"
    if not phase_verdicts:
        study_verdict = "NO-DATA"
    if study_verdict == "IRREDUCIBLE":
        conclusion = (
            "On early/late states the best possible estimator (MC win-prob) barely "
            "out-ranks the net against the single realized outcome (small headroom) "
            "and the net already matches the MC win-prob (high net_accuracy). The low "
            "realized-rank is dice/opponent coin-flip noise, NOT a representation gap "
            "-> value work is dead; v8 + search is the ceiling. Do NOT pursue capacity."
        )
    elif study_verdict == "CAPACITY-HEADROOM":
        conclusion = (
            "On at least one of early/late, the MC win-prob out-ranks the net by a "
            "CI-clean headroom AND/OR the net poorly matches the MC win-prob -> the "
            "frozen trunk under-represents these states -> a capacity/representation "
            "attempt is worth it."
        )
    else:
        conclusion = "No early/late states recorded — increase --n-states."

    result: dict[str, Any] = {
        "probe": "irreducible-ceiling probe (spec 007 follow-up to US0)",
        "ckpt": V8_CKPT,
        "config": {
            "n_states": n_states,
            "rollouts": rollouts,
            "seed": seed,
            "n_bootstrap": N_BOOTSTRAP,
            "phase_boundaries": {
                "early_max_vp": PHASE_EARLY_MAX_VP,
                "late_min_vp": PHASE_LATE_MIN_VP,
            },
            "headroom_clear": HEADROOM_CLEAR,
            "net_accuracy_high": NET_ACCURACY_HIGH,
        },
        "data": {
            "n_states": len(results),
            "phase_counts": phase_counts,
            "anchor_outcome_rate": float(np.mean(outcome)),
        },
        "per_phase": per_phase,
        "overall": overall,
        "sanity_anchor": sanity,
        "verdict": study_verdict,
        "conclusion": conclusion,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _print_report(result)
    print(f"\n[probe] wrote {out_path}", flush=True)
    return result


def _print_report(result: dict[str, Any]) -> None:
    """Human-readable summary to stdout."""
    print("\n" + "=" * 78)
    print("IRREDUCIBLE-CEILING PROBE — RESULT")
    print("=" * 78)
    data = result["data"]
    print(
        f"states={data['n_states']}  phase_counts={data['phase_counts']}  "
        f"anchor_outcome_rate={data['anchor_outcome_rate']:.3f}"
    )
    san = result["sanity_anchor"]
    flag = "OK" if san["all_within_tol"] else "WARNING (off-anchor)"
    san_bits = "  ".join(
        f"{ph}: net_rank={d['measured_net_rank']:.3f} vs US0 {d['us0_expected']}"
        for ph, d in san["phases"].items()
    )
    print(f"sanity [{flag}]: {san_bits}")
    print("-" * 78)
    hdr = (
        f"{'subset':>8} {'n':>4} {'ceiling':>9} {'net_rank':>9} {'headroom':>9} "
        f"{'hr_CI':>17} {'net_acc':>8}  verdict"
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
        hr = m["headroom_ci"]
        print(
            f"{name:>8} {m['n']:>4} {m['ceiling_rank']:>9.3f} {m['net_rank']:>9.3f} "
            f"{m['headroom']:>9.3f} [{hr[0]:>6.3f},{hr[1]:>6.3f}] "
            f"{m['net_accuracy']:>8.3f}  {m['verdict']}"
        )
    print("-" * 78)
    print(f"VERDICT: {result['verdict']}")
    print(result["conclusion"])
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-states",
        type=int,
        default=160,
        help="target number of early/late snapshots to roll out (default 160; use 6 for smoke)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=24,
        help="independent MC rollouts per snapshot (default 24; use 4 for smoke)",
    )
    parser.add_argument("--seed", type=int, default=0, help="master seed (default 0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/probe_irreducible_ceiling.json"),
        help="output JSON path (default runs/probe_irreducible_ceiling.json)",
    )
    args = parser.parse_args(argv)
    run_probe(
        n_states=args.n_states,
        rollouts=args.rollouts,
        seed=args.seed,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
