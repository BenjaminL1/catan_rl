"""Bake-off gate (contract C6, encodes SC-001) — the go/no-go for the feature.

Minimal determinized search vs the RAW policy it wraps. PASS iff the seat-
symmetrized win-rate's Wilson lower bound is strictly > 0.50 at n>=200, then
re-confirmed at n>=500. A WR ~ 0.50 (or LB <= 0.50) is a documented FAIL: the
value-head leaf is too off-distribution for lookahead to help, and the build
should stop / pivot (priors-weighted search, bounded rollout to a better-
calibrated late state, or fix the leaf) before any US2/US3 work.

The time-budget ladder (SC-002, monotone WR vs sims) is added in US2 (T021); the
gate itself only needs the fixed-budget Wilson test.
"""

from __future__ import annotations

from typing import Any

from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import evaluate_search_vs_policy


def _summary(tag: str, result: Any) -> dict[str, Any]:
    return {
        f"n_{tag}": result.n,
        f"wr_{tag}": result.wr,
        f"ci_{tag}": [result.ci.lower, result.ci.upper],
        f"rules_violations_{tag}": len(result.rules_violations),
    }


def run_bakeoff(
    ckpt: str,
    *,
    sims: int = 50,
    n_quick: int = 200,
    n_confirm: int = 500,
    seed: int = 0,
    device: str = "cpu",
    max_turns: int = 400,
) -> dict[str, Any]:
    """Run the bake-off gate; return a JSON-serialisable verdict dict.

    Keys: ``passed`` (bool), ``failure_mode`` (str | None), the wrapped ckpt /
    sims / seed, and the quick (n>=200) + confirm (n>=500) WR + Wilson CI +
    rules-violation counts. ``confirm`` is only run if ``quick`` clears the gate.
    """
    cfg = SearchConfig(sims_per_move=sims, seed=seed)

    quick = evaluate_search_vs_policy(
        cfg, ckpt, ckpt, n_games=n_quick, seed=seed, device=device, max_turns=max_turns
    )
    verdict: dict[str, Any] = {
        "ckpt": str(ckpt),
        "sims": sims,
        "seed": seed,
        **_summary("quick", quick),
    }
    if quick.ci.lower <= 0.5:
        verdict["passed"] = False
        verdict["failure_mode"] = (
            f"quick Wilson LB {quick.ci.lower:.3f} <= 0.50 at n={quick.n}: "
            "search does not beat the raw policy"
        )
        return verdict

    # Disjoint games from the quick screen (offset the eval seed) so the n>=500
    # confirmation is independent evidence, not a superset of the quick sample.
    confirm = evaluate_search_vs_policy(
        cfg, ckpt, ckpt, n_games=n_confirm, seed=seed + n_quick, device=device, max_turns=max_turns
    )
    verdict.update(_summary("confirm", confirm))
    passed = confirm.ci.lower > 0.5
    verdict["passed"] = passed
    verdict["failure_mode"] = (
        None if passed else f"confirm Wilson LB {confirm.ci.lower:.3f} <= 0.50 at n={confirm.n}"
    )
    return verdict
