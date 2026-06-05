"""Behavioral tests for the Rust ``catan_engine.StackedDice``.

Per the Q1 decision (statistical equivalence, not byte parity with
the Python StackedDice), these tests verify the Rust impl follows
the documented Colonist.io 1v1 dice rules:

* Bag holds 36 outcomes per the 2d6 distribution + 1 noise swap.
* Karma fires (20%) only when ``last_seven_roller != current_player``
  and ``last_seven_roller is not None``.
* Distribution over many games is close to the standard 2d6 PMF.
* Determinism: same seed → same sequence.

The byte-parity test against the Python ChaCha8 reference lives in
``test_rng_parity.py``.
"""

from __future__ import annotations

from collections import Counter

import pytest

catan_engine = pytest.importorskip("catan_engine")


def test_construction_with_seed_is_deterministic() -> None:
    a = catan_engine.StackedDice(42)
    b = catan_engine.StackedDice(42)
    rolls_a = [a.roll(0, None) for _ in range(100)]
    rolls_b = [b.roll(0, None) for _ in range(100)]
    assert rolls_a == rolls_b


def test_different_seeds_diverge() -> None:
    a = catan_engine.StackedDice(1)
    b = catan_engine.StackedDice(2)
    rolls_a = [a.roll(0, None) for _ in range(100)]
    rolls_b = [b.roll(0, None) for _ in range(100)]
    assert rolls_a != rolls_b


def test_initial_bag_has_36_outcomes() -> None:
    dice = catan_engine.StackedDice(1)
    assert dice.bag_remaining() == 36
    assert all(2 <= v <= 12 for v in dice.bag_view())


def test_all_rolls_are_in_range() -> None:
    dice = catan_engine.StackedDice(7)
    for _ in range(1000):
        v = dice.roll(0, None)
        assert 2 <= v <= 12


def test_distribution_is_close_to_2d6_pmf() -> None:
    """36k rolls — each bucket should fall within 0.01 of the
    expected 2d6 frequency. Reviewer-tightened (M1): the old 4%
    tolerance was ~14× looser than empirical worst-case (~0.003)
    and would silently mask a bag-shape bug (e.g., missing one 7
    landing at ~0.139 vs the expected 0.167)."""
    dice = catan_engine.StackedDice(99)
    counts: Counter = Counter()
    n = 36_000
    for _ in range(n):
        counts[dice.roll(0, None)] += 1
    expected_pmf = {
        2: 1 / 36,
        3: 2 / 36,
        4: 3 / 36,
        5: 4 / 36,
        6: 5 / 36,
        7: 6 / 36,
        8: 5 / 36,
        9: 4 / 36,
        10: 3 / 36,
        11: 2 / 36,
        12: 1 / 36,
    }
    for face, expected in expected_pmf.items():
        observed = counts[face] / n
        err = abs(observed - expected)
        assert err < 0.01, f"face={face} observed={observed:.4f} expected={expected:.4f}"


def test_initial_bag_shape_matches_distribution_minus_one_swap() -> None:
    """Belt-and-suspenders for M1: assert the first bag's composition
    is exactly the standard 2d6 multiset with one non-7 swapped for a
    uniform random value in [2, 12]. Catches noise-swap bugs that the
    empirical PMF test can't (e.g., swap removing a 7 instead of a
    non-7, or off-by-one on the count table)."""
    dice = catan_engine.StackedDice(123)
    bag = sorted(dice.bag_view())
    # Standard 2d6 multiset: {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}
    canonical = sorted(
        [
            2,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            10,
            10,
            10,
            11,
            11,
            12,
        ]
    )
    # Both lists have 36 elements.
    assert len(bag) == 36
    assert len(canonical) == 36
    # The noise swap removes one non-7 and appends a uniform [2, 12].
    # The bag's multiset differs from canonical by 0 elements (swap
    # picked the same value back, P=1/11) or by 2 elements (one
    # removal + one different addition, P=10/11). Anything else is
    # a bug.
    bag_counter = Counter(bag)
    canon_counter = Counter(canonical)
    # The 7-count must NOT change (noise swap excludes 7s).
    assert bag_counter[7] == canon_counter[7] == 6
    # Total absolute count delta is 0 (same-value swap) or 2 (one
    # removal + one different addition). No other diff is valid.
    diff = sum(abs(bag_counter[k] - canon_counter[k]) for k in range(2, 13))
    assert diff in (0, 2), f"expected diff in (0, 2), got {diff}"
    # No single face's count can have shifted by more than 1.
    for face in range(2, 13):
        delta = abs(bag_counter[face] - canon_counter[face])
        assert delta <= 1, f"face {face} shifted by {delta}, expected ≤ 1"


def test_karma_fires_when_buffed_player_rolls() -> None:
    """When player 1 rolls and player 0 was the last 7-roller,
    the Karma 20% chance of an instant 7 fires. Empirical 7-rate
    should be measurably higher than the bag's natural rate."""
    dice = catan_engine.StackedDice(42)
    sevens = 0
    n = 1000
    for _ in range(n):
        if dice.roll(1, 0) == 7:
            sevens += 1
    rate = sevens / n
    # Expected ~33% (20% Karma + 80% * ~17% bag). Accept > 25%.
    assert rate > 0.25, f"Karma seven rate {rate} too low"


def test_karma_does_not_fire_for_same_player() -> None:
    """When the buff is on player 0 and player 0 rolls,
    Karma is inactive — only the bag matters."""
    dice = catan_engine.StackedDice(7)
    sevens = 0
    n = 1000
    for _ in range(n):
        if dice.roll(0, 0) == 7:
            sevens += 1
    rate = sevens / n
    assert 0.10 < rate < 0.25


def test_karma_does_not_fire_when_no_prior_seven() -> None:
    """``last_seven_roller=None`` means Karma is inactive — covers
    the first part of every game before anyone rolls a 7."""
    dice = catan_engine.StackedDice(100)
    sevens = 0
    n = 1000
    for _ in range(n):
        if dice.roll(0, None) == 7:
            sevens += 1
    rate = sevens / n
    assert 0.10 < rate < 0.25


def test_bag_refills_after_exhaustion() -> None:
    dice = catan_engine.StackedDice(1)
    initial = dice.bag_remaining()
    # Consume the entire initial bag.
    for _ in range(initial):
        dice.roll(0, None)
    # Next roll must still produce a valid sum (i.e., bag refilled).
    v = dice.roll(0, None)
    assert 2 <= v <= 12
