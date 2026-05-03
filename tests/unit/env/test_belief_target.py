"""Tests for the env-side belief-head target (Phase 2.5b)."""

from __future__ import annotations

import numpy as np

from catan_rl.env.catan_env import DEV_CARD_ORDER, CatanEnv


class _FakePlayer:
    """Stand-in for engine.player.player — only the fields we need."""

    def __init__(self, devCards: dict, newDevCards: list) -> None:
        self.devCards = devCards
        self.newDevCards = newDevCards


def _empty_env() -> CatanEnv:
    """Construct an env without booting a game (we only call `_belief_target`)."""
    env = CatanEnv(use_belief_head=True)
    return env


def test_belief_target_uniform_when_no_hidden_cards() -> None:
    """Empty hand → uniform distribution (avoid log(0) in soft-CE)."""
    env = _empty_env()
    p = _FakePlayer(devCards={}, newDevCards=[])
    target = env._belief_target(p)
    np.testing.assert_array_almost_equal(target, np.full(5, 0.2))


def test_belief_target_normalized_count_distribution() -> None:
    """Counts sum to 1 across the 5 types."""
    env = _empty_env()
    p = _FakePlayer(devCards={"KNIGHT": 2, "VP": 1}, newDevCards=["MONOPOLY"])
    target = env._belief_target(p)
    assert abs(target.sum() - 1.0) < 1e-6
    # KNIGHT (idx 0) gets 2/4 = 0.5; VP (idx 1) gets 1/4 = 0.25;
    # MONOPOLY (idx 4) gets 1/4 = 0.25.
    assert abs(target[DEV_CARD_ORDER.index("KNIGHT")] - 0.5) < 1e-6
    assert abs(target[DEV_CARD_ORDER.index("VP")] - 0.25) < 1e-6
    assert abs(target[DEV_CARD_ORDER.index("MONOPOLY")] - 0.25) < 1e-6


def test_belief_target_includes_new_dev_cards() -> None:
    """``newDevCards`` (just-bought, not yet playable) counts toward the target."""
    env = _empty_env()
    p_with_new = _FakePlayer(devCards={"KNIGHT": 1}, newDevCards=["KNIGHT", "VP"])
    target = env._belief_target(p_with_new)
    # 2 KNIGHTs total + 1 VP = 3 cards, KNIGHT proportion = 2/3.
    assert abs(target[DEV_CARD_ORDER.index("KNIGHT")] - 2.0 / 3) < 1e-6


def test_belief_target_dim_matches_dev_card_types() -> None:
    """Output has exactly N_DEV_CARD_TYPES = 5 entries (KNIGHT/VP/RB/YOP/MONO)."""
    env = _empty_env()
    p = _FakePlayer(devCards={"YEAROFPLENTY": 3}, newDevCards=[])
    target = env._belief_target(p)
    assert target.shape == (5,)
    assert target[DEV_CARD_ORDER.index("YEAROFPLENTY")] == 1.0


def test_belief_target_ignores_unknown_card_strings() -> None:
    """Unknown dev-card strings in newDevCards are silently dropped."""
    env = _empty_env()
    p = _FakePlayer(devCards={}, newDevCards=["GARBAGE", "VP"])
    target = env._belief_target(p)
    # Only VP counted; full mass on VP.
    assert target[DEV_CARD_ORDER.index("VP")] == 1.0
