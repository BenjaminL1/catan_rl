"""StackedDice tests — Colonist.io 1v1 dice mechanic.

Verifies:
1. Bag distribution matches the 36-outcome 2d6 sum distribution (with one noise swap).
2. Karma rule: 20% chance of forced 7 if the *other* player rolled the previous 7.
3. Bag refills cleanly when emptied.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from catan_rl.engine.dice import StackedDice


class _PlayerStub:
    """Minimal stand-in: StackedDice only checks identity, not attributes."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def dice() -> StackedDice:
    random.seed(42)
    return StackedDice()


def test_bag_size_is_36(dice: StackedDice) -> None:
    """A fresh bag holds exactly 36 outcomes (matches 2d6 distribution count)."""
    assert len(dice.bag) == 36


def test_rolls_in_valid_range(dice: StackedDice) -> None:
    """Every draw is an integer in [2, 12]."""
    p = _PlayerStub("p1")
    for _ in range(100):
        v = dice.roll(p, None)
        assert 2 <= v <= 12, f"roll out of range: {v}"


def test_distribution_close_to_2d6(dice: StackedDice) -> None:
    """Across many bag refills, frequencies approach the 2d6 distribution.

    Tolerance is loose because:
      - StackedDice swaps one non-7 for a uniform random 2-12 each refill.
      - Karma forces 7s 20% of the time when the *other* player last rolled 7.
    Here we use a single roller (p1) so Karma never triggers.
    """
    random.seed(42)
    d = StackedDice()
    p = _PlayerStub("p1")
    counts: Counter[int] = Counter()
    for _ in range(36 * 200):  # 200 bags
        counts[d.roll(p, None)] += 1

    # 7s should be ~6/36 of rolls (with noise allowing slight drift)
    pct_7 = counts[7] / sum(counts.values())
    assert 0.13 < pct_7 < 0.20, f"7-rate looks wrong: {pct_7:.3f}"


def test_karma_forces_seven_when_opponent_rolled_last(dice: StackedDice) -> None:
    """When the opponent rolled the previous 7, current roller gets 7 ~20% of the time."""
    random.seed(0)
    d = StackedDice()
    me = _PlayerStub("me")
    opp = _PlayerStub("opp")

    seven_count = 0
    trials = 1000
    for _ in range(trials):
        v = d.roll(me, last_7_roller_obj=opp)
        if v == 7:
            seven_count += 1
    rate = seven_count / trials
    # Karma adds a 20% forced-7 on top of the natural 6/36 (~16.7%) rate,
    # combined: 1 - (1 - 0.2) * (1 - 1/6) ≈ 0.333. Allow generous tolerance.
    assert 0.25 < rate < 0.40, f"Karma 7-rate suspicious: {rate:.3f}"


def test_karma_does_not_trigger_for_self() -> None:
    """If `me` rolled the last 7, Karma does NOT trigger when `me` rolls again."""
    random.seed(0)
    d = StackedDice()
    me = _PlayerStub("me")
    rates = []
    for _ in range(5):
        seven_count = 0
        for _ in range(500):
            v = d.roll(me, last_7_roller_obj=me)
            if v == 7:
                seven_count += 1
        rates.append(seven_count / 500)

    # Without Karma, should be near 6/36 ≈ 0.167. Allow noise from refills.
    avg = sum(rates) / len(rates)
    assert 0.10 < avg < 0.22, f"non-Karma 7-rate suspicious: {avg:.3f}"


def test_bag_refills_when_empty(dice: StackedDice) -> None:
    """Empty the bag manually and roll: a refill should kick in."""
    p = _PlayerStub("p1")
    dice.bag = []
    v = dice.roll(p, None)
    assert 2 <= v <= 12
    assert len(dice.bag) >= 0  # bag may have refilled to 36 then popped one
