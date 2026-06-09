"""Belief-head target: the opponent's hidden dev-card posterior.

Pins (a) the target is the normalized hidden-dev composition, (b) it is masked
(all-zeros) when the opponent holds none, and (c) the hidden dev-card *types*
NEVER leak into the agent's observation — the aux task is only meaningful if its
answer is not already an input.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.env.catan_env import DEV_CARD_ORDER, CatanEnv
from catan_rl.policy.obs_schema import N_DEV_TYPES

_VP_IDX = 5  # next/current_player_main layout: [0:5]=resources, [5]=VP/15


def _fresh_env() -> CatanEnv:
    env = CatanEnv(opponent_type="random", max_turns=200)
    env.reset(seed=0)
    return env


def test_belief_target_is_normalized_composition() -> None:
    env = _fresh_env()
    env.opponent_player.devCards = {"KNIGHT": 2, "MONOPOLY": 1}
    env.opponent_player.newDevCards = []
    t = env.belief_target()
    assert t.shape == (N_DEV_TYPES,)
    assert t.dtype == np.float32
    assert t.sum() == 1.0  # normalized -> a probability vector
    expected = np.zeros(N_DEV_TYPES, dtype=np.float32)
    expected[DEV_CARD_ORDER.index("KNIGHT")] = 2 / 3
    expected[DEV_CARD_ORDER.index("MONOPOLY")] = 1 / 3
    np.testing.assert_allclose(t, expected, atol=1e-6)


def test_belief_target_counts_just_bought_cards() -> None:
    # newDevCards (bought this turn, not yet playable) are still hidden info.
    env = _fresh_env()
    env.opponent_player.devCards = {"KNIGHT": 1}
    env.opponent_player.newDevCards = ["MONOPOLY"]
    t = env.belief_target()
    assert t.sum() == 1.0
    assert t[DEV_CARD_ORDER.index("KNIGHT")] == 0.5
    assert t[DEV_CARD_ORDER.index("MONOPOLY")] == 0.5


def test_belief_target_excludes_observable_vp() -> None:
    # VP cards are observable in 1v1 (the victoryPoints vs visibleVictoryPoints
    # gap reveals the hidden VP count), so they are EXCLUDED from the hidden
    # posterior: the head predicts only the genuinely-secret buyable types.
    env = _fresh_env()
    env.opponent_player.devCards = {"KNIGHT": 1, "VP": 3}
    env.opponent_player.newDevCards = []
    t = env.belief_target()
    assert t[DEV_CARD_ORDER.index("VP")] == 0.0  # VP never carries target mass
    assert t[DEV_CARD_ORDER.index("KNIGHT")] == 1.0  # normalized over non-VP only
    assert t.sum() == 1.0


def test_belief_target_vp_only_is_masked() -> None:
    # An opponent holding ONLY VP cards has no hidden-TYPE signal -> all-zeros.
    env = _fresh_env()
    env.opponent_player.devCards = {"VP": 2}
    env.opponent_player.newDevCards = []
    assert env.belief_target().sum() == 0.0


def test_belief_target_empty_is_all_zeros_masked() -> None:
    env = _fresh_env()
    env.opponent_player.devCards = {}
    env.opponent_player.newDevCards = []
    t = env.belief_target()
    assert t.sum() == 0.0  # all-zeros -> masked out of the belief loss


def test_hidden_dev_types_do_not_leak_into_observation() -> None:
    """Same hidden-dev COUNT, different TYPES -> identical agent obs.

    The obs may carry the opponent's hidden-dev *count* (a 6-bin one-hot) and
    *played* cards, but never the hidden *type composition* — otherwise the
    belief head would just be reading its own answer off the input.
    """
    env = _fresh_env()
    env.opponent_player.newDevCards = []

    env.opponent_player.devCards = {"KNIGHT": 3}
    obs_a = env._get_obs()
    target_a = env.belief_target()

    env.opponent_player.devCards = {"MONOPOLY": 3}
    obs_b = env._get_obs()
    target_b = env.belief_target()

    # Obs identical (no leak of hidden types) ...
    assert set(obs_a) == set(obs_b)
    for k in obs_a:
        np.testing.assert_array_equal(
            obs_a[k], obs_b[k], err_msg=f"hidden dev types leaked into obs[{k!r}]"
        )
    # ... but the belief target distinguishes them (it IS the GT).
    assert not np.array_equal(target_a, target_b)


def test_opponent_vp_obs_uses_visible_not_total() -> None:
    # The opponent's VP obs feature must be visibleVictoryPoints (observable),
    # never total victoryPoints — otherwise the hidden-VP count leaks.
    env = _fresh_env()
    env.opponent_player.victoryPoints = 7
    env.opponent_player.visibleVictoryPoints = 5  # 2 hidden VP dev cards
    obs = env._get_obs()
    assert obs["next_player_main"][_VP_IDX] == pytest.approx(5 / 15)


def test_agent_vp_obs_uses_own_total() -> None:
    # The agent DOES see its own true total VP (it knows its own hidden cards).
    env = _fresh_env()
    env.agent_player.victoryPoints = 6
    env.agent_player.visibleVictoryPoints = 4
    obs = env._get_obs()
    assert obs["current_player_main"][_VP_IDX] == pytest.approx(6 / 15)
