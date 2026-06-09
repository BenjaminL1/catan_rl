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
from catan_rl.policy.obs_schema import OPP_KIND_SELF_LATEST


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


def test_resource_channel_follows_the_pov_swap() -> None:
    """RL-review SHOULD-FIX 1: the resource channel (current/next_player_main)
    must swap with POV, not stay agent-centric. With asymmetric hands the two
    POVs must differ on both player-main vectors."""
    env = _env()
    agent = env.agent_player
    opp = env.opponent_player
    agent.resources = {"WOOD": 5, "BRICK": 4, "WHEAT": 0, "ORE": 0, "SHEEP": 0}
    opp.resources = {"WOOD": 0, "BRICK": 0, "WHEAT": 3, "ORE": 2, "SHEEP": 1}
    state = _main_phase_state(env)

    agent_pov = env._build_obs_for(agent, opp, state)
    opp_pov = env._build_obs_for(opp, agent, state)

    # The acting player's OWN resources (current_player_main, read directly from
    # ``acting_player.resources``) must follow the POV swap — the opponent sees
    # its own hand here, never the agent's.
    assert not np.array_equal(agent_pov["current_player_main"], opp_pov["current_player_main"])


def test_opponent_pov_opp_id_comes_from_env_state() -> None:
    """RL-review SHOULD-FIX 2: from the snapshot opponent's POV, ITS opponent is
    the learner — the opp-id fields are whatever the opponent-LOCAL env_state
    carries (Phase 3's driver MUST set these to the agent's kind/id, e.g.
    OPP_KIND_SELF_LATEST), NOT reused from the agent's env_state."""
    env = _env()
    agent = env.agent_player
    opp = env.opponent_player
    opp_state = EnvObsState(
        initial_placement_phase=False,
        setup_step=0,
        roll_pending=False,
        discard_pending=False,
        robber_placement_pending=False,
        road_building_roads_left=0,
        last_dice_roll=env.last_dice_roll,
        opp_kind=OPP_KIND_SELF_LATEST,
        opp_policy_id=7,
    )
    opp_pov = env._build_obs_for(opp, agent, opp_state)
    assert int(opp_pov["opponent_kind"]) == OPP_KIND_SELF_LATEST
    assert int(opp_pov["opponent_policy_id"]) == 7


def test_compute_masks_honours_opponent_local_env_state() -> None:
    """CONSIDER 3: the masks env_state param (the Phase-3 opponent path) must
    reflect ITS sub-turn phase, not the agent's. An opponent-local state with
    roll_pending=True yields a different mask than the agent default."""
    env = _env()
    opp = env.opponent_player
    roll_state = EnvObsState(
        initial_placement_phase=False,
        setup_step=0,
        roll_pending=True,
        discard_pending=False,
        robber_placement_pending=False,
        road_building_roads_left=0,
        last_dice_roll=env.last_dice_roll,
    )
    main_masks = env._compute_masks(opp, _main_phase_state(env))
    roll_masks = env._compute_masks(opp, roll_state)
    # roll_pending forces RollDice-only -> the type mask differs from main phase.
    assert not np.array_equal(main_masks["type"], roll_masks["type"])
