"""Opening-isolated win-prob value-ranking probe (Tier-0 diagnostic).

Follows up the irreducible-ceiling probe (probe_irreducible_ceiling.py) to answer
the question raised by the ThePhantom playtest: is v8's weak OPENING a VALUE
problem (the value head mis-ranks openings -> retarget/representation worth it,
research direction #1) or a POLICY problem (the value ranks openings fine by v8's
own standard, but v8 PLAYS the opening wrong -> the lever is an out-of-lineage
opening prior #2 + an external human-signal gate #3, NOT value retargeting)?

What it measures (per phase, INCLUDING the opening which the old probe lumped into
"early" vp<=4):
  * ``net_value``  = the net's squashed [0,1] value at a state (rank-equivalent to
    the raw value head, since Spearman is invariant to the monotonic squash -- so
    this also answers "would a win-prob TARGET rank better?": no, not via the
    target alone; only WHERE the ranking fails matters, hence the opening slice).
  * ``mc_winprob`` = mean agent-win over K independent rollouts to terminal (1/0).
  * ``realized_outcome`` = the anchor game's agent win (1/0).
  * ``ceiling_rank`` = Spearman(mc_winprob, realized_outcome)  (best estimator).
  * ``net_rank``     = Spearman(net_value,  realized_outcome).
  * ``net_accuracy`` = Spearman(net_value,  mc_winprob)        (net vs near-truth).
  * ``headroom``     = ceiling_rank - net_rank (paired bootstrap CI).

Verdict is driven by the OPENING slice:
  * VALUE-RECOVERABLE iff opening headroom CI lower bound clears ~0.05 OR opening
    net_accuracy < ~0.70  -> a better estimator out-ranks the net on openings
    (even by v8's own standard) -> value retarget/representation (research #1)
    has headroom. PROCEED to #1.
  * VALUE-CONSISTENT otherwise -> the net already ranks openings by v8's own MC
    win-prob; there is no value-side headroom. The opening weakness lives in the
    POLICY, so the lever is the out-of-lineage opening prior (#2) + the external
    human-signal gate (#3), and value retargeting (#1) is likely a DEAD END.

CONFOUND (load-bearing, stated in the JSON): the MC rollouts play BOTH seats with
v8, so this measures value<->policy CONSISTENCY, not strategic correctness vs a
stronger player. A high net_accuracy does NOT clear v8 of the opening blind spot
(v8 grades its own homework); it only says no value-SIDE fix helps within v8's own
policy. The blind spot itself can only be confirmed by an EXTERNAL signal (the
ThePhantom corpus / a stronger opening oracle) -- which is exactly research #3.

Reuses the vetted rollout + metrics from probe_irreducible_ceiling (fresh-dice
open-loop rollout, bounded memory, cross-fit-free paired-bootstrap CIs).

Constraints: CPU only, fixed seeds, additive (new file), no GUI import, ruff +
mypy --strict clean.

Usage (smoke -- run here)::

    python scripts/dev/probe_value_winprob.py --n-states 8 --rollouts 4 \
        --out runs/probe_winprob_smoke.json

Usage (full -- hours; opening rollouts are full games)::

    python scripts/dev/probe_value_winprob.py --n-states 200 --rollouts 24
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

# Make ``scripts.dev`` importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the vetted rollout + metrics machinery (do not re-implement).
from scripts.dev.probe_irreducible_ceiling import (
    _SEED_STRIDE,
    V8_CKPT,
    _mc_winprob_for_snapshot,
    _metrics_for_subset,
    _Snapshot,
    _StateResult,
)

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

# Opening isolated as its own slice (the old probe lumped vp<=2 into "early").
PHASES: tuple[str, ...] = ("opening", "early", "mid", "late")
HEADROOM_CLEAR = 0.05
NET_ACCURACY_HIGH = 0.70


def _phase_for_vp(vp: int) -> str:
    if vp <= 2:
        return "opening"  # post-setup, before any city/expansion -> the opening
    if vp <= 4:
        return "early"
    if vp <= 9:
        return "mid"
    return "late"


@dataclass(slots=True)
class _GameSnapshots:
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
    """Play one v8-vs-v8 RAW anchor game; reservoir-keep <=1 snapshot per phase."""
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env
    from catan_rl.search.node import agent_outcome
    from catan_rl.search.value import value_from_obs

    obs, _ = env.reset(seed=seed, options={"agent_seat": agent_seat})
    masks = env.get_action_masks()
    held = _GameSnapshots()
    res_rng = random.Random(seed ^ 0x5DEECE66D)
    seen: dict[str, int] = {ph: 0 for ph in PHASES}

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

        if int(masks["type"].sum()) > 1:  # non-forced decision
            vp = int(getattr(env.agent_player, "victoryPoints", 0))
            phase = _phase_for_vp(vp)
            seen[phase] += 1
            keep = (phase not in held.by_phase) or (res_rng.random() < 1.0 / seen[phase])
            if keep:
                net_value = value_from_obs(policy, obs, device=device)
                held.by_phase[phase] = _Snapshot(
                    phase=phase,
                    net_value=net_value,
                    env=clone_env(env, opponent),
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


def generate_state_results(
    *, n_states: int, rollouts: int, seed: int, device: torch.device
) -> list[_StateResult]:
    """Play anchor games until ~``n_states`` snapshots (across phases) are rolled out."""
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
            env, policy, opponent, device=device, seed=game_seed, agent_seat=agent_seat
        )
        for phase, snap in held.by_phase.items():
            roll_base = (game_seed * 7919 + PHASES.index(phase) * 104_729) & 0x7FFF_FFFF
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
        held.by_phase.clear()
        del held
        game_id += 1
        counts = {ph: sum(1 for r in results if r.phase == ph) for ph in PHASES}
        print(
            f"[winprob] game {game_id}: results={len(results)} {counts} "
            f"outcome={outcome:.0f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if game_id > max(50, n_states * 6):
            print(
                f"[winprob] WARNING: stopping after {game_id} games with {len(results)} states.",
                file=sys.stderr,
                flush=True,
            )
            break

    env.close()
    return results


def _verdict_from_opening(opening: dict[str, Any]) -> tuple[str, str]:
    """Drive the study verdict off the OPENING slice metrics."""
    if opening.get("n", 0) == 0:
        return "NO-DATA", "no opening states recorded — increase --n-states"
    hr_lo = opening["headroom_ci"][0]
    net_acc = opening["net_accuracy"]
    recoverable = (hr_lo > HEADROOM_CLEAR) or (net_acc < NET_ACCURACY_HIGH)
    if recoverable:
        return (
            "VALUE-RECOVERABLE",
            f"opening headroom CI lower bound {hr_lo:+.3f} clears {HEADROOM_CLEAR} "
            f"and/or net_accuracy {net_acc:.3f} < {NET_ACCURACY_HIGH}: a better value "
            "estimator out-ranks v8's value on openings -> value retarget / "
            "representation (research #1) has headroom; PROCEED to #1.",
        )
    return (
        "VALUE-CONSISTENT",
        f"opening headroom CI lower bound {hr_lo:+.3f} <= {HEADROOM_CLEAR} AND "
        f"net_accuracy {net_acc:.3f} >= {NET_ACCURACY_HIGH}: the net already ranks "
        "openings by v8's own MC win-prob, so there is NO value-side headroom. The "
        "opening weakness is in the POLICY, not the value ranking -> the lever is "
        "the out-of-lineage opening prior (#2) + external human-signal gate (#3); "
        "value retargeting (#1) is likely a dead end. (Confound: MC is under v8's "
        "own play, so this is value<->policy consistency, NOT strategic correctness "
        "vs a stronger player — the blind spot itself needs an external signal.)",
    )


def run_probe(*, n_states: int, rollouts: int, seed: int, out_path: Path) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    results = generate_state_results(n_states=n_states, rollouts=rollouts, seed=seed, device=device)
    if not results:
        raise RuntimeError("no states recorded — increase --n-states")

    phase_arr = np.array([r.phase for r in results])
    net = np.array([r.net_value for r in results], dtype=np.float64)
    mc = np.array([r.mc_winprob for r in results], dtype=np.float64)
    outcome = np.array([r.realized_outcome for r in results], dtype=np.float64)
    phase_counts = {ph: int(np.sum(phase_arr == ph)) for ph in PHASES}

    per_phase: dict[str, dict[str, Any]] = {}
    for i, ph in enumerate(PHASES):
        mask = phase_arr == ph
        if int(mask.sum()) == 0:
            per_phase[ph] = {"n": 0}
            continue
        per_phase[ph] = _metrics_for_subset(mc[mask], net[mask], outcome[mask], seed=seed + i)
    overall = _metrics_for_subset(mc, net, outcome, seed=seed + 99)

    verdict, why = _verdict_from_opening(per_phase["opening"])

    result: dict[str, Any] = {
        "probe": "opening-isolated win-prob value-ranking probe (Tier-0)",
        "ckpt": V8_CKPT,
        "config": {"n_states": n_states, "rollouts": rollouts, "seed": seed},
        "data": {
            "n_states": len(results),
            "phase_counts": phase_counts,
            "anchor_outcome_rate": float(np.mean(outcome)),
        },
        "per_phase": per_phase,
        "overall": overall,
        "verdict": verdict,
        "reason": why,
        "confound": (
            "MC win-prob is computed under v8 playing BOTH seats, so this measures "
            "value<->policy CONSISTENCY, not strategic correctness vs a stronger "
            "player. High net_accuracy does NOT clear v8 of the opening blind spot; "
            "it only rules OUT a value-side fix within v8's own policy. The blind "
            "spot can only be confirmed by an external signal (ThePhantom corpus)."
        ),
        "note_spearman_invariance": (
            "net_value is the squashed value; Spearman is invariant to the monotonic "
            "squash, so these ranks equal the raw value head's ranks. A win-prob "
            "TARGET therefore cannot change the ranking by itself — only WHERE the "
            "ranking fails (the opening slice) is informative."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _print_report(result)
    print(f"\n[winprob] wrote {out_path}", flush=True)
    return result


def _print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 84)
    print("OPENING-ISOLATED WIN-PROB VALUE-RANKING PROBE — RESULT")
    print("=" * 84)
    d = result["data"]
    print(
        f"states={d['n_states']}  counts={d['phase_counts']}  "
        f"anchor_outcome_rate={d['anchor_outcome_rate']:.3f}"
    )
    print("-" * 84)
    print(
        f"{'subset':>8} {'n':>4} {'ceiling':>8} {'net_rank':>8} {'headroom':>8} "
        f"{'hr_CI':>16} {'net_acc':>8}"
    )
    rows = [*((ph, result["per_phase"][ph]) for ph in PHASES), ("overall", result["overall"])]
    for name, m in rows:
        if m.get("n", 0) == 0:
            print(f"{name:>8} {0:>4}  (no data)")
            continue
        hr = m["headroom_ci"]
        print(
            f"{name:>8} {m['n']:>4} {m['ceiling_rank']:>8.3f} {m['net_rank']:>8.3f} "
            f"{m['headroom']:>8.3f} [{hr[0]:>6.3f},{hr[1]:>6.3f}] {m['net_accuracy']:>8.3f}"
        )
    print("-" * 84)
    print(f"VERDICT (opening-driven): {result['verdict']}")
    print(result["reason"])
    print(f"\nCONFOUND: {result['confound']}")
    print("=" * 84)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=200, help="target snapshots (8 smoke)")
    parser.add_argument("--rollouts", type=int, default=24, help="MC rollouts/snapshot (4 smoke)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/probe_value_winprob.json"))
    args = parser.parse_args(argv)
    run_probe(n_states=args.n_states, rollouts=args.rollouts, seed=args.seed, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
