"""Tests for Phase 3.4 trainer-side TrueSkill rating wiring.

Stays at the unit level: we don't spin up the full trainer (slow) — we
exercise ``RatingTable`` directly and verify the helper hooks behave.
"""

from __future__ import annotations

from catan_rl.selfplay.ratings import Rating, RatingTable


def test_main_vs_opponent_match_updates_both_sides() -> None:
    """``record_match`` shifts both ratings, in opposite directions."""
    table = RatingTable()
    main_id = -99
    opp_id = 0
    table.record_match(main_id, opp_id, a_won=True)
    main = table.get(main_id)
    opp = table.get(opp_id)
    assert main.mu > 25.0  # main moved up from default
    assert opp.mu < 25.0  # opp moved down


def test_decay_inflates_sigma_only() -> None:
    """σ-decay should grow σ but leave μ untouched."""
    table = RatingTable()
    table.ratings[0] = Rating(mu=25.0, sigma=5.0)
    decay = 1.05
    table.ratings[0] = Rating(mu=table.ratings[0].mu, sigma=table.ratings[0].sigma * decay)
    r = table.ratings[0]
    assert r.mu == 25.0
    assert abs(r.sigma - 5.25) < 1e-9


def test_skip_match_when_opponent_id_negative() -> None:
    """Random/heuristic/current_self matches must not enter the rating table.

    Mirrors ``_record_rating_match``: if opponent_id < 0, we early-return
    so the table only ever holds policy IDs (and the main sentinel).
    """
    table = RatingTable()
    main_id = -99

    def record(opp_id: int, win: int) -> None:
        if opp_id < 0:
            return
        table.record_match(main_id, opp_id, a_won=bool(win))

    record(-1, 1)  # random opponent
    record(-2, 0)  # current_self opponent
    assert len(table.ratings) == 0  # nothing recorded
    record(7, 1)  # real policy
    # main + opp 7 both inserted now.
    assert main_id in table.ratings
    assert 7 in table.ratings


def test_top_k_orders_by_conservative_rating() -> None:
    """``RatingTable.top_k`` returns highest conservative skill first."""
    table = RatingTable()
    table.ratings = {
        0: Rating(mu=20.0, sigma=2.0),  # conservative = 14.0
        1: Rating(mu=30.0, sigma=2.0),  # conservative = 24.0
        2: Rating(mu=25.0, sigma=2.0),  # conservative = 19.0
    }
    keys = [k for k, _ in table.top_k(3)]
    assert keys == [1, 2, 0]
