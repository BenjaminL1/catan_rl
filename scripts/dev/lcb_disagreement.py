"""LCB-vs-max-visit final-move disagreement probe (spec 008 US0(a) / FR-002).

Standalone, additive diagnostic. For a sample of v8 self-play positions it runs
the determinized PUCT search ONCE per position and reports how often the LCB
final move (``argmax mean_Q - z*stderr``) DIFFERS from the deployed max-visit
final move (the "disagreement rate").

Why: the deployed search has a documented depth-0 visit collapse (~0.92 of visits
land on a single root action, ``fpu_mode='zero'``). If LCB almost never disagrees
with max-visit, it CANNOT change decisions and a full WR / SPRT eval would be
wasted budget. This probe is the cheap pre-check: one search per position, no
rollouts to terminal, so it is orders of magnitude cheaper than the root-headroom
probe (which rolls every candidate out) — yet it answers "can LCB even move?".

How it reuses existing machinery
--------------------------------
* Position generation: ``probe_root_headroom._play_anchor_game`` (the same
  reservoir-sampled, bounded-memory v8-vs-v8 anchor games used by the root-headroom
  kill-gate). Imported, not re-implemented.
* Search: a ``SearchAgent`` at the DEPLOYED default ``SearchConfig`` (so the
  collapse this probe checks against is the real one).
* LCB pick: reconstructed from the SINGLE search's diagnostics — ``visit_counts``
  (N), ``action_q`` (mean_Q = W/N), ``action_q2`` (Σ value²) — via the production
  ``mcts._lcb_select`` selection rule. No second search pass.

The max-visit pick is the SearchAgent's chosen action (deployed default
``final_move_mode='max_visit'``); we never re-run search for the LCB side.

Constraints: CPU only, fixed seeds everywhere, additive (new file), no GUI
import, ruff + ``mypy --strict`` clean. Does NOT modify any existing code.

Usage (smoke — run here)::

    python scripts/dev/lcb_disagreement.py --n-states 4 --sims 25 \
        --out runs/lcb_disagreement_smoke.json

Usage (full — run by the human; many positions x deployed sims)::

    python scripts/dev/lcb_disagreement.py --n-states 200 --sims 100 --lcb-z 1.96
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

# Make ``scripts.dev`` importable when this file is run directly
# (``python scripts/dev/lcb_disagreement.py`` puts ``scripts/dev`` on sys.path[0],
# not the repo root). Adding the repo root lets the ``scripts.dev`` package import
# resolve; it is a no-op when imported as a module (the path is already present).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the root-headroom probe's vetted, bounded-memory position generator.
from scripts.dev.probe_root_headroom import (
    V8_CKPT,
    _phase_for_vp,
    _play_anchor_game,
)

if TYPE_CHECKING:
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.priors import ActionTuple

#: Stride keeping per-game seed ranges disjoint (matches the root-headroom probe).
_SEED_STRIDE = 1_000_003
PHASES: tuple[str, ...] = ("early", "mid", "late")


@dataclass(slots=True)
class _PositionResult:
    """One position's LCB-vs-max-visit comparison."""

    phase: str
    disagree: int  # 1 iff the LCB pick != the max-visit pick
    max_visit_action: ActionTuple
    lcb_action: ActionTuple
    depth0_concentration: float  # max child visit share at the root
    n_candidates: int  # number of visited root children


def _lcb_pick_from_diagnostics(diag: dict[str, Any], z: float) -> ActionTuple | None:
    """Reconstruct the LCB final move from ONE search's diagnostics.

    Uses the production selection rule ``mcts._lcb_select`` over the aggregated root
    child stats the search already recorded: N (visit_counts), W = mean_Q*N
    (action_q), Σvalue² (action_q2). Returns ``None`` if no visits landed.
    """
    from catan_rl.search.mcts import _lcb_select

    visit_counts: dict[ActionTuple, int] = diag.get("visit_counts", {})
    action_q: dict[ActionTuple, float] = diag.get("action_q", {})
    action_q2: dict[ActionTuple, float] = diag.get("action_q2", {})
    priors: dict[ActionTuple, float] = diag.get("priors", {})
    if not visit_counts:
        return None
    agg_w = {a: action_q.get(a, 0.0) * n for a, n in visit_counts.items()}
    return _lcb_select(list(visit_counts.keys()), visit_counts, agg_w, action_q2, priors, z)


def _evaluate_position(
    env: Any, agent: SearchAgent, z: float, phase: str
) -> _PositionResult | None:
    """Run search ONCE on a clone of ``env``; compare LCB vs max-visit picks."""
    from catan_rl.search.mcts import clone_env

    search_env = clone_env(env, agent.opponent)
    max_visit_arr = agent.choose_action(search_env)  # deployed default = max_visit
    diag = agent.last_diagnostics
    search_env.close()

    if diag.get("forced", False):
        return None  # forced root: no choice, not a disagreement sample
    visit_counts: dict[ActionTuple, int] = diag.get("visit_counts", {})
    total = sum(visit_counts.values())
    if total <= 0:
        return None

    lcb_action = _lcb_pick_from_diagnostics(diag, z)
    if lcb_action is None:
        return None
    max_visit_action: ActionTuple = cast(
        "ActionTuple", tuple(int(x) for x in max_visit_arr.tolist())
    )
    return _PositionResult(
        phase=phase,
        disagree=1 if lcb_action != max_visit_action else 0,
        max_visit_action=max_visit_action,
        lcb_action=lcb_action,
        depth0_concentration=max(visit_counts.values()) / total,
        n_candidates=sum(1 for n in visit_counts.values() if n > 0),
    )


def generate_results(
    *, n_states: int, sims: int, lcb_z: float, seed: int, device: torch.device
) -> list[_PositionResult]:
    """Play anchor games (bounded memory) until ~``n_states`` positions are scored."""
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = actor.policy
    opponent = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(opponent)

    cfg = SearchConfig(sims_per_move=sims, seed=seed)  # DEPLOYED default (max_visit)
    agent = SearchAgent(policy, cfg, device=device)
    per_game_cap = max(1, min(4, n_states))

    results: list[_PositionResult] = []
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
        for snap in held.snaps:
            if len(results) < n_states:
                vp = int(getattr(snap.env.agent_player, "victoryPoints", 0))
                res = _evaluate_position(snap.env, agent, lcb_z, _phase_for_vp(vp))
                if res is not None:
                    results.append(res)
            snap.env.close()
        held.snaps.clear()
        del held
        game_id += 1
        print(
            f"[lcb] game {game_id}: results={len(results)} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if game_id > max(50, n_states * 4):
            print(
                f"[lcb] WARNING: stopping after {game_id} games with only {len(results)} states.",
                file=sys.stderr,
                flush=True,
            )
            break

    env.close()
    return results


def run_probe(
    *, n_states: int, sims: int, lcb_z: float, seed: int, out_path: Path
) -> dict[str, Any]:
    """Generate -> disagreement rate (overall + per-phase) -> JSON + table."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    results = generate_results(n_states=n_states, sims=sims, lcb_z=lcb_z, seed=seed, device=device)
    if not results:
        raise RuntimeError("no positions evaluated — increase --n-states / check config")

    def _summary(rows: list[_PositionResult]) -> dict[str, Any]:
        n = len(rows)
        disagree = sum(r.disagree for r in rows)
        return {
            "n": n,
            "disagreements": disagree,
            "disagreement_rate": disagree / n if n else 0.0,
            "mean_depth0_concentration": (
                float(np.mean([r.depth0_concentration for r in rows])) if n else 0.0
            ),
            "mean_n_candidates": (float(np.mean([r.n_candidates for r in rows])) if n else 0.0),
        }

    overall = _summary(results)
    per_phase = {ph: _summary([r for r in results if r.phase == ph]) for ph in PHASES}

    result: dict[str, Any] = {
        "probe": "LCB-vs-max-visit disagreement (spec 008 US0(a) / FR-002)",
        "ckpt": V8_CKPT,
        "config": {"n_states": n_states, "sims": sims, "lcb_z": lcb_z, "seed": seed},
        "overall": overall,
        "per_phase": per_phase,
        "conclusion": (
            "disagreement_rate is the fraction of non-forced positions where the LCB "
            "final move differs from the deployed max-visit move. ~0 => LCB cannot "
            "change decisions under the visit collapse (a WR/SPRT eval would be wasted "
            "budget); a meaningful rate => LCB is worth a matched-budget SPRT eval."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 72)
    print("LCB-vs-MAX-VISIT DISAGREEMENT — RESULT (spec 008 US0(a) / FR-002)")
    print("=" * 72)
    print(f"{'subset':>8} {'n':>4} {'disagree':>9} {'rate':>7} {'conc':>6} {'cands':>6}")
    for name, m in [*((ph, per_phase[ph]) for ph in PHASES), ("overall", overall)]:
        print(
            f"{name:>8} {m['n']:>4} {m['disagreements']:>9} "
            f"{m['disagreement_rate']:>7.3f} {m['mean_depth0_concentration']:>6.3f} "
            f"{m['mean_n_candidates']:>6.2f}"
        )
    print("-" * 72)
    print(result["conclusion"])
    print(f"\n[lcb] wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=200, help="positions to score (4 smoke)")
    parser.add_argument(
        "--sims", type=int, default=100, help="PUCT sims/move (deployed 100; 25 smoke)"
    )
    parser.add_argument("--lcb-z", type=float, default=1.96, help="LCB z-multiplier (default 1.96)")
    parser.add_argument("--seed", type=int, default=0, help="master seed (default 0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/lcb_disagreement.json"),
        help="output JSON path",
    )
    args = parser.parse_args(argv)
    run_probe(
        n_states=args.n_states,
        sims=args.sims,
        lcb_z=args.lcb_z,
        seed=args.seed,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
