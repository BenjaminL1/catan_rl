"""Integration smoke (T014): a tiny-budget search agent vs the heuristic plays
full games to completion with ZERO ruleset violations (SC-006).

A random-init policy + short ``max_turns`` keeps it fast; the point is that the
search-aware loop drives the engine end-to-end producing only legal actions
(asserted via ``eval/rules_invariants``), not that it wins.
"""

from __future__ import annotations

from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.search.agent import SearchAgent
from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import run_search_matchup


def _policy() -> CatanPolicy:
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    return policy


def test_search_vs_heuristic_completes_with_zero_violations() -> None:
    agent = SearchAgent(_policy(), SearchConfig(sims_per_move=4, seed=0))
    res = run_search_matchup(
        agent,
        opponent_type="heuristic",
        opponent=None,
        n_games=2,
        seed=0,
        max_turns=60,
        opponent_ref="heuristic",
        audit_rules=True,
    )
    assert res.n == 2
    assert res.rules_violations == ()
    assert all(g.n_turns > 0 for g in res.games)
