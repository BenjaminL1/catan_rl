"""LCB decision-QUALITY probe (spec 008 US0(a) follow-up to the disagreement probe).

The disagreement probe showed the LCB final move differs from the deployed
max-visit move ~18.5% of the time (so LCB is active despite the visit collapse).
This probe answers the next question: when they DIFFER, is the LCB move actually
BETTER? It rolls out BOTH moves at each disagreement position and compares their
Monte-Carlo win-probabilities (a PAIRED delta, far more powered for LCB's small
effect than a full win-rate tournament).

Method
------
1. Generate v8-vs-v8 positions (reuse ``probe_root_headroom._play_anchor_game`` —
   bounded-memory anchor games). At each non-forced position run the deployed PUCT
   search ONCE; the max-visit move = ``SearchAgent.choose_action``; the LCB move =
   ``lcb_disagreement._lcb_pick_from_diagnostics`` over the SAME tree (no 2nd search).
2. On positions where the two DIFFER, roll BOTH moves out to terminal under v8 on
   both seats (``probe_root_headroom._q_for_root_action`` — the fresh-StackedDice
   independent-dice fix), ``q = mean win-prob``. ``delta = q(LCB) - q(max_visit)``.
   This is paired (same position) so it cancels position variance; no winner's-curse
   bias (both moves are pre-chosen, not max-selected over noisy estimates).
3. Aggregate mean delta + bootstrap CI over the disagreement positions.

Verdict: LCB HELPS iff mean-delta bootstrap CI lower bound > 0 (the LCB move is
reliably higher win-prob); NEUTRAL/HURTS otherwise. Combined with the ~18.5%
disagreement rate this gives the overall expected WR effect of switching to LCB
(rate x mean-delta) — the cheap decisive read before any full SPRT/WR eval.

Constraints: CPU only, fixed seeds, additive (new file), no GUI import, ruff +
``mypy --strict`` clean. Memory-safe streaming (one game's positions, one move's
rollouts at a time). Does NOT modify existing code.

Usage (smoke)::

    python scripts/dev/probe_lcb_value.py --n-disagree 3 --rollouts 6 --sims 25 \
        --out runs/probe_lcb_value_smoke.json

Usage (full — run alone, ~1h)::

    python scripts/dev/probe_lcb_value.py --n-disagree 90 --rollouts 40 --sims 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.lcb_disagreement import _lcb_pick_from_diagnostics
from scripts.dev.probe_root_headroom import (
    V8_CKPT,
    _phase_for_vp,
    _play_anchor_game,
    _q_for_root_action,
)

if TYPE_CHECKING:
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.priors import ActionTuple

_SEED_STRIDE = 1_000_003
N_BOOTSTRAP = 1000
PHASES: tuple[str, ...] = ("early", "mid", "late")


@dataclass(slots=True)
class _DisagreeResult:
    """One disagreement position: rollout win-prob of each move + their delta."""

    phase: str
    q_lcb: float
    q_max_visit: float
    delta: float  # q_lcb - q_max_visit (paired)


def _pooled_q(
    position_env: Any,
    action: ActionTuple,
    policy: Any,
    opponent: Any,
    *,
    device: torch.device,
    base_seed: int,
    rollouts: int,
    safety_cap: int,
) -> float:
    """K-rollout win-prob of a FIXED move (pooled both cross-fit halves; unbiased
    for a pre-chosen move — no max-selection here, so no winner's-curse)."""
    q_sel, q_est = _q_for_root_action(
        position_env,
        action,
        policy,
        opponent,
        device=device,
        base_seed=base_seed,
        rollouts=rollouts,
        safety_cap=safety_cap,
    )
    return 0.5 * (q_sel + q_est)


def generate_results(
    *, n_disagree: int, rollouts: int, sims: int, lcb_z: float, seed: int, device: torch.device
) -> list[_DisagreeResult]:
    """Play anchor games; on LCB-vs-max-visit DISAGREEMENT positions, roll out both
    moves and record the paired delta. Stops once ~``n_disagree`` are collected."""
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.search.mcts import clone_env
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = actor.policy
    opponent = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(opponent)
    rollout_safety_cap = env.max_turns * 50

    cfg = SearchConfig(sims_per_move=sims, seed=seed)  # deployed default (max_visit)
    agent: SearchAgent = SearchAgent(policy, cfg, device=device)
    per_game_cap = 4

    results: list[_DisagreeResult] = []
    n_seen = 0
    t0 = time.time()
    game_id = 0
    while len(results) < n_disagree:
        game_seed = (seed * _SEED_STRIDE + game_id) % (2**31 - 1)
        opponent.reset_rng(seed=game_seed)
        held = _play_anchor_game(
            env,
            policy,
            opponent,
            device=device,
            seed=game_seed,
            agent_seat=game_id % 2,
            per_game_cap=per_game_cap,
        )
        for snap in held.snaps:
            if len(results) >= n_disagree:
                snap.env.close()
                continue
            n_seen += 1
            search_env = clone_env(snap.env, opponent)
            mv_arr = agent.choose_action(search_env)
            diag = agent.last_diagnostics
            search_env.close()
            vc: dict[ActionTuple, int] = diag.get("visit_counts", {})
            if diag.get("forced", False) or sum(vc.values()) <= 0:
                snap.env.close()
                continue
            lcb_action = _lcb_pick_from_diagnostics(diag, lcb_z)
            mv_action: ActionTuple = cast("ActionTuple", tuple(int(x) for x in mv_arr.tolist()))
            if lcb_action is None or lcb_action == mv_action:
                snap.env.close()
                continue  # only score disagreements
            phase = _phase_for_vp(int(getattr(snap.env.agent_player, "victoryPoints", 0)))
            rs = (game_seed * 7919 + len(results) * 104_729) & 0x7FFF_FFFF
            q_mv = _pooled_q(
                snap.env,
                mv_action,
                policy,
                opponent,
                device=device,
                base_seed=rs,
                rollouts=rollouts,
                safety_cap=rollout_safety_cap,
            )
            q_lcb = _pooled_q(
                snap.env,
                lcb_action,
                policy,
                opponent,
                device=device,
                base_seed=rs ^ 0x5DEECE66D,
                rollouts=rollouts,
                safety_cap=rollout_safety_cap,
            )
            results.append(
                _DisagreeResult(phase=phase, q_lcb=q_lcb, q_max_visit=q_mv, delta=q_lcb - q_mv)
            )
            snap.env.close()
        held.snaps.clear()
        del held
        game_id += 1
        print(
            f"[lcbq] game {game_id}: disagreements={len(results)}/{n_disagree} "
            f"(seen {n_seen}, {time.time() - t0:.0f}s)",
            flush=True,
        )
        if game_id > max(80, n_disagree * 30):
            print(
                f"[lcbq] WARNING: stop after {game_id} games, {len(results)} disagreements.",
                file=sys.stderr,
                flush=True,
            )
            break
    env.close()
    return results


def _bootstrap_mean_ci(x: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    if x.size < 2:
        return (0.0, 0.0)
    s = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        s[i] = x[rng.integers(0, x.size, size=x.size)].mean()
    lo, hi = np.percentile(s, [2.5, 97.5])
    return (float(lo), float(hi))


def run_probe(
    *, n_disagree: int, rollouts: int, sims: int, lcb_z: float, seed: int, out_path: Path
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    results = generate_results(
        n_disagree=n_disagree, rollouts=rollouts, sims=sims, lcb_z=lcb_z, seed=seed, device=device
    )
    if not results:
        raise RuntimeError("no disagreement positions found — increase --n-disagree budget")

    delta = np.array([r.delta for r in results], dtype=np.float64)
    rng = np.random.default_rng(seed + 17)
    lo, hi = _bootstrap_mean_ci(delta, rng=rng)
    mean_delta = float(delta.mean())
    helps = lo > 0.0
    hurts = hi < 0.0
    verdict = "LCB-HELPS" if helps else ("LCB-HURTS" if hurts else "LCB-NEUTRAL")

    per_phase = {
        ph: {
            "n": int(sum(1 for r in results if r.phase == ph)),
            "mean_delta": (
                float(np.mean([r.delta for r in results if r.phase == ph]))
                if any(r.phase == ph for r in results)
                else 0.0
            ),
        }
        for ph in PHASES
    }
    result: dict[str, Any] = {
        "probe": "LCB decision-quality (spec 008 US0a follow-up)",
        "ckpt": V8_CKPT,
        "config": {
            "n_disagree": n_disagree,
            "rollouts": rollouts,
            "sims": sims,
            "lcb_z": lcb_z,
            "seed": seed,
        },
        "n_disagreements": len(results),
        "mean_delta": mean_delta,
        "mean_delta_ci": [lo, hi],
        "mean_q_lcb": float(np.mean([r.q_lcb for r in results])),
        "mean_q_max_visit": float(np.mean([r.q_max_visit for r in results])),
        "per_phase": per_phase,
        "verdict": verdict,
        "conclusion": (
            "mean_delta = mean(q_LCB - q_max_visit) over DISAGREEMENT positions. "
            "LCB-HELPS (CI lower > 0) => the LCB move is reliably higher win-prob; the "
            "overall WR effect of switching is ~ disagreement_rate (0.185) x mean_delta. "
            "LCB-NEUTRAL/HURTS => keep max-visit; combined with the value ceiling, "
            "v8+search is at the practical ceiling."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 72)
    print("LCB DECISION-QUALITY — RESULT (spec 008 US0a)")
    print("=" * 72)
    print(f"disagreement positions scored: {len(results)}  (rollouts/move={rollouts})")
    print(f"mean delta q(LCB) - q(max_visit) = {mean_delta:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"mean q_LCB={result['mean_q_lcb']:.3f}  q_max_visit={result['mean_q_max_visit']:.3f}")
    print("per-phase mean delta:", {ph: round(per_phase[ph]["mean_delta"], 4) for ph in PHASES})
    print(f"\nVERDICT: {verdict}")
    print(result["conclusion"])
    print(f"\n[lcbq] wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-disagree", type=int, default=90, help="disagreement positions to score")
    p.add_argument("--rollouts", type=int, default=40, help="MC rollouts per move (default 40)")
    p.add_argument("--sims", type=int, default=100, help="PUCT sims/move (deployed 100; 25 smoke)")
    p.add_argument("--lcb-z", type=float, default=1.96, help="LCB z-multiplier (default 1.96)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/probe_lcb_value.json"))
    a = p.parse_args(argv)
    run_probe(
        n_disagree=a.n_disagree,
        rollouts=a.rollouts,
        sims=a.sims,
        lcb_z=a.lcb_z,
        seed=a.seed,
        out_path=a.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
