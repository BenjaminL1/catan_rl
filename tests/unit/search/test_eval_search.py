"""Search-aware eval loop structure (T012, contract C4).

Uses the random-init fixture policy + the heuristic opponent at a tiny budget +
short ``max_turns`` — this validates the LOOP (seat symmetry, Wilson CI,
reproducibility, legality), not strategic strength (that is the bake-off T016).
"""

from __future__ import annotations

import math

from catan_rl.eval.harness import EvalMatchupResult
from catan_rl.eval.wilson import WilsonInterval
from catan_rl.search.agent import SearchAgent
from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import run_search_matchup


def _run(policy, *, seed: int):  # type: ignore[no-untyped-def]
    agent = SearchAgent(policy, SearchConfig(sims_per_move=2, seed=0))
    return run_search_matchup(
        agent,
        opponent_type="heuristic",
        opponent=None,
        n_games=2,
        seed=seed,
        max_turns=30,
        opponent_ref="heuristic",
    )


def test_returns_matchup_result_with_wilson_ci(policy) -> None:  # type: ignore[no-untyped-def]
    res = _run(policy, seed=0)
    assert isinstance(res, EvalMatchupResult)
    assert isinstance(res.ci, WilsonInterval)
    assert res.n == 2
    assert res.ci.n == 2
    assert 0.0 <= res.wr <= 1.0
    assert math.isclose(res.wr, res.wins / res.n)


def test_is_seat_symmetrized(policy) -> None:  # type: ignore[no-untyped-def]
    res = _run(policy, seed=0)
    # n_games=2 -> one game from each seat.
    assert res.n_seat0 == 1
    assert res.n_seat1 == 1


def test_is_reproducible_at_fixed_seed(policy) -> None:  # type: ignore[no-untyped-def]
    a = _run(policy, seed=1)
    b = _run(policy, seed=1)
    assert a.wins == b.wins
    assert [g.final_vp_agent for g in a.games] == [g.final_vp_agent for g in b.games]
    assert [g.n_turns for g in a.games] == [g.n_turns for g in b.games]


def test_no_rules_violations(policy) -> None:  # type: ignore[no-untyped-def]
    res = _run(policy, seed=0)
    assert res.rules_violations == ()
