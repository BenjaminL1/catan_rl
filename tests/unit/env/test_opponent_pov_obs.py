"""T006 — opponent-POV obs must not leak the agent's hidden information (FR-012).

The single most dangerous bug in the self-play keystone: if the snapshot
opponent's policy input is built from the agent's perspective (or reuses the
agent's obs), the opponent sees the learner's hidden hand and self-play
strength becomes meaningless. These tests pin that ``_build_obs_for(opp, agent,
...)`` yields the OPPONENT's view — its own hidden dev cards, only the agent's
*played* cards — and that the agent's hidden cards appear nowhere.
"""

from __future__ import annotations

import numpy as np

from catan_rl.env.catan_env import CatanEnv
from catan_rl.policy.obs_encoder import EnvObsState, _dev_counts


def _env() -> CatanEnv:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    return env


def _main_phase_state(env: CatanEnv) -> EnvObsState:
    return EnvObsState(
        initial_placement_phase=False,
        setup_step=0,
        roll_pending=False,
        discard_pending=False,
        robber_placement_pending=False,
        road_building_roads_left=0,
        last_dice_roll=env.last_dice_roll,
    )


def test_opponent_pov_uses_own_hidden_and_agent_played_only() -> None:
    env = _env()
    agent = env.agent_player
    opp = env.opponent_player
    # Agent holds hidden (unplayed) dev cards the opponent must NOT see.
    agent.devCards["KNIGHT"] = 2
    # Opponent holds its own hidden dev card.
    opp.devCards["MONOPOLY"] = 1

    opp_pov = env._build_obs_for(opp, agent, _main_phase_state(env))

    # current_dev_counts == the OPPONENT's own hidden hand (oracle).
    np.testing.assert_array_equal(opp_pov["current_dev_counts"], _dev_counts(opp, hidden=True))
    # next_played_dev_counts == only the agent's PLAYED dev cards (oracle).
    np.testing.assert_array_equal(
        opp_pov["next_played_dev_counts"], _dev_counts(agent, hidden=False)
    )


def test_agent_hidden_cards_appear_nowhere_in_opponent_obs() -> None:
    env = _env()
    agent = env.agent_player
    opp = env.opponent_player
    agent.devCards["KNIGHT"] = 2  # hidden, unplayed
    opp.devCards["MONOPOLY"] = 1

    opp_pov = env._build_obs_for(opp, agent, _main_phase_state(env))
    agent_hidden = _dev_counts(agent, hidden=True)

    # The agent's hidden vector must not surface as either dev-count field.
    assert not np.array_equal(opp_pov["current_dev_counts"], agent_hidden)
    # Agent played nothing -> the field the opponent CAN see is all zeros.
    assert int(np.asarray(opp_pov["next_played_dev_counts"]).sum()) == 0


def test_agent_and_opponent_pov_are_distinct() -> None:
    env = _env()
    agent = env.agent_player
    opp = env.opponent_player
    agent.devCards["KNIGHT"] = 2
    opp.devCards["MONOPOLY"] = 1
    state = _main_phase_state(env)

    agent_pov = env._build_obs_for(agent, opp, state)
    opp_pov = env._build_obs_for(opp, agent, state)

    np.testing.assert_array_equal(agent_pov["current_dev_counts"], _dev_counts(agent, hidden=True))
    np.testing.assert_array_equal(opp_pov["current_dev_counts"], _dev_counts(opp, hidden=True))
    assert not np.array_equal(agent_pov["current_dev_counts"], opp_pov["current_dev_counts"])
