"""Unit tests for the skill rating wrapper (Phase 0/3)."""

from __future__ import annotations

import pytest

from catan_rl.selfplay.ratings import Rating, RatingSystem, RatingTable


def test_rating_system_creates_default_rating() -> None:
    sys = RatingSystem()
    r = sys.create()
    assert r.mu == pytest.approx(25.0)
    assert r.sigma == pytest.approx(25.0 / 3.0)
    assert r.conservative == pytest.approx(25.0 - 3.0 * 25.0 / 3.0)


def test_a_winning_increases_a_mu_and_decreases_b_mu() -> None:
    """A single win moves the winner's mu up and the loser's down."""
    sys = RatingSystem()
    a, b = sys.create(), sys.create()
    new_a, new_b = sys.update_match(a, b, a_won=True)
    assert new_a.mu > a.mu
    assert new_b.mu < b.mu


def test_expected_win_prob_symmetric_at_equal_skill() -> None:
    sys = RatingSystem()
    a, b = sys.create(), sys.create()
    p = sys.expected_win_prob(a, b)
    assert p == pytest.approx(0.5)


def test_expected_win_prob_higher_for_higher_mu() -> None:
    sys = RatingSystem()
    strong = Rating(mu=40.0, sigma=4.0)
    weak = Rating(mu=10.0, sigma=4.0)
    assert sys.expected_win_prob(strong, weak) > 0.95
    assert sys.expected_win_prob(weak, strong) < 0.05


def test_rating_table_records_match_and_top_k() -> None:
    table = RatingTable()
    for _ in range(20):
        table.record_match("strong", "weak", a_won=True)
    top = table.top_k(2)
    assert top[0][0] == "strong"
    assert top[1][0] == "weak"
    assert top[0][1].conservative > top[1][1].conservative


def test_rating_table_serialization() -> None:
    table = RatingTable()
    table.record_match("a", "b", a_won=True)
    d = table.to_dict()
    assert "a" in d
    assert "b" in d
    assert set(d["a"].keys()) == {"mu", "sigma", "conservative"}
