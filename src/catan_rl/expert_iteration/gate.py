"""Pilot gate (contract C3, encodes SC-001).

SEARCH-FREE evaluation of the distilled policy vs raw v6 (single forward pass, no
MCTS — the whole point is a fast stronger policy). PASS iff the seat-symmetrized
win-rate's Wilson lower bound > 0.50 at n>=200, re-confirmed on a disjoint n>=500.
Also reports win-rate vs the heuristic as a catastrophic-forgetting guard (the
distilled policy must beat raw v6 WITHOUT collapsing against earlier opponents).
"""

from __future__ import annotations

from typing import Any, cast


def run_gate(
    distilled_ckpt: str,
    v6_ckpt: str,
    *,
    n_quick: int = 200,
    n_confirm: int = 500,
    seed: int = 0,
    device: str = "cpu",
    heuristic_floor: float = 0.60,
) -> dict[str, Any]:
    """Return the PASS/FAIL verdict + WRs + CIs (search-free distilled-vs-raw-v6)."""
    from catan_rl.eval.harness import EvalHarness, evaluate_policy_vs_policy
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor

    champ = cast(
        _PolicyActor,
        build_actor(PlayerSpec(kind="policy", ckpt_path=distilled_ckpt), seed=seed, device=device),
    ).policy

    quick = evaluate_policy_vs_policy(champ, v6_ckpt, n_games=n_quick, seed=seed, device=device)
    verdict: dict[str, Any] = {
        "distilled_ckpt": str(distilled_ckpt),
        "v6_ckpt": str(v6_ckpt),
        "n_quick": quick.n,
        "wr_quick_vs_v6": quick.wr,
        "ci_quick": [quick.ci.lower, quick.ci.upper],
        "rules_violations_quick": len(quick.rules_violations),
    }
    if quick.ci.lower <= 0.5:
        verdict["passed"] = False
        verdict["failure_mode"] = (
            f"quick Wilson LB {quick.ci.lower:.3f} <= 0.50 at n={quick.n}: "
            "distillation did not beat raw v6"
        )
        return verdict

    confirm = evaluate_policy_vs_policy(
        champ, v6_ckpt, n_games=n_confirm, seed=seed + n_quick, device=device
    )
    # Forgetting guard: the distilled policy must still crush the heuristic.
    heur = (
        EvalHarness(
            opponent_types=("heuristic",),
            n_games_per_seat=max(1, n_quick // 2),
            seed=seed,
            device=device,
        )
        .run(champ)
        .results[0]
    )

    verdict.update(
        {
            "n_confirm": confirm.n,
            "wr_confirm_vs_v6": confirm.wr,
            "ci_confirm": [confirm.ci.lower, confirm.ci.upper],
            "rules_violations_confirm": len(confirm.rules_violations),
            "wr_vs_heuristic": heur.wr,
            "rules_violations_heuristic": len(heur.rules_violations),
        }
    )
    # PASS requires BOTH: significantly beats raw v6 AND no catastrophic forgetting
    # (still crushes the heuristic — v6 itself is ~0.9, so a healthy distill stays
    # well above the floor; a distill that overfit to v6 and collapsed elsewhere fails).
    beats_v6 = confirm.ci.lower > 0.5
    no_forgetting = heur.wr >= heuristic_floor
    verdict["passed"] = bool(beats_v6 and no_forgetting)
    if beats_v6 and no_forgetting:
        verdict["failure_mode"] = None
    elif not beats_v6:
        verdict["failure_mode"] = (
            f"confirm Wilson LB {confirm.ci.lower:.3f} <= 0.50 at n={confirm.n}"
        )
    else:
        verdict["failure_mode"] = (
            f"catastrophic forgetting: WR vs heuristic {heur.wr:.3f} < floor {heuristic_floor:.2f} "
            "(beat v6 but collapsed vs the heuristic)"
        )
    return verdict
