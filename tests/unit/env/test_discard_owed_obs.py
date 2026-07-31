"""D3 — the remaining-owed discard count must be correct at EVERY sub-step.

The 7-roll discard is decomposed into ``floor(H0 / 2)`` independent
``env.step`` calls and the count is fixed at roll time then decremented
invisibly, so at a 9-card hand the policy could not distinguish "owes 3,
started at 12" from "owes 4, started at 9". Self-play was solving a sequential
problem blind to its position in the sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.env.catan_env import CatanEnv
from catan_rl.policy import obs_schema as S

_SLOT = -S.CURR_RESERVED_SLOTS - 1


def _owed_scalar(obs: dict[str, np.ndarray]) -> float:
    return float(obs["current_player_main"][_SLOT])


def test_owed_scalar_decrements_once_per_discarded_card() -> None:
    env = CatanEnv()
    env.reset(seed=7)
    agent = env.agent_player
    assert agent is not None
    agent.resources = {"WOOD": 3, "BRICK": 3, "WHEAT": 3, "ORE": 1, "SHEEP": 2}  # 12 cards
    env.initial_placement_phase = False
    env.discard_pending = True
    env.roll_pending = False
    env._cards_to_discard = sum(agent.resources.values()) // 2  # 6

    seen = []
    for _ in range(6):
        obs = env._get_obs()
        seen.append(_owed_scalar(obs))
        masks = env.get_action_masks()
        legal = np.flatnonzero(masks["resource1_discard"])
        action = np.zeros(6, dtype=np.int64)
        action[0] = S.ActionType.DISCARD
        action[4] = int(legal[0])
        env.step(action)
        if not env.discard_pending:
            break

    assert seen == [pytest.approx(min(1.0, n / 8.0)) for n in (6, 5, 4, 3, 2, 1)]
    assert env._cards_to_discard <= 0
    assert not env.discard_pending
    # Once the sub-phase is over the slot returns to strict 0.0.
    assert _owed_scalar(env._get_obs()) == 0.0


def test_owed_scalar_is_zero_outside_the_discard_subphase() -> None:
    env = CatanEnv()
    obs, _ = env.reset(seed=3)
    assert _owed_scalar(obs) == 0.0


def test_owed_scalar_saturates_at_eight() -> None:
    env = CatanEnv()
    env.reset(seed=5)
    env.initial_placement_phase = False
    env.discard_pending = True
    env._cards_to_discard = 12
    assert _owed_scalar(env._get_obs()) == 1.0


def test_opponent_seat_sees_its_own_owed_count() -> None:
    """Both seats or neither: threading only the agent trains an asymmetry."""
    env = CatanEnv()
    env.reset(seed=11)
    for owed in (1, 4, 8, 20):
        state = env._opponent_env_state(discard_pending=True, cards_to_discard=owed)
        obs = env._build_obs_for(env.opponent_player, env.agent_player, state)
        assert _owed_scalar(obs) == pytest.approx(min(1.0, owed / 8.0))


def test_masks_are_computed_with_the_owed_count_threaded() -> None:
    """``_compute_masks`` builds its own EnvObsState; it must carry the field
    too, or the mask and obs would describe different states."""
    env = CatanEnv()
    env.reset(seed=13)
    env.initial_placement_phase = False
    env.discard_pending = True
    env._cards_to_discard = 2
    # Smoke: computing masks in the discard sub-phase must not raise and must
    # offer DISCARD only.
    masks = env.get_action_masks()
    assert masks["type"][S.ActionType.DISCARD]
    assert int(masks["type"].sum()) == 1
