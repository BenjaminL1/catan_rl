"""T004 — regression coverage for the shared ``_apply_main_action`` extraction.

The broad behavior-identity guarantee is provided by the full existing suite
(1891 tests exercising ``step``). These tests pin the property that motivated
the extraction: a single action's follow-on sub-turn state (robber after a
knight, free roads after a road-builder) is written to the *player-local*
``_TurnState`` and NEVER to the env's agent-centric ``self.*`` flags — so the
snapshot opponent's turn-driver cannot clobber the agent's pending state.
"""

from __future__ import annotations

from catan_rl.env.catan_env import ActionType, CatanEnv, _TurnState


def _fresh_env() -> CatanEnv:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    return env


def test_play_knight_sets_turnstate_not_self() -> None:
    env = _fresh_env()
    agent = env.agent_player
    agent.devCards["KNIGHT"] = 1
    agent.devCardPlayedThisTurn = False
    env.robber_placement_pending = False  # sentinel: must stay untouched

    ts = _TurnState()
    env._apply_main_action(
        agent,
        action_type=ActionType.PLAY_KNIGHT,
        corner_idx=0,
        edge_idx=0,
        tile_idx=0,
        res1_idx=0,
        res2_idx=0,
        ts=ts,
    )

    # Follow-on lands in the player-local turn-state...
    assert ts.robber_placement_pending is True
    # ...and the env's agent-centric flag is NOT touched by the shared helper.
    assert env.robber_placement_pending is False
    # Engine effects still applied.
    assert agent.devCards["KNIGHT"] == 0
    assert agent.knightsPlayed == 1
    assert agent.devCardPlayedThisTurn is True


def test_play_road_builder_sets_turnstate_not_self() -> None:
    env = _fresh_env()
    agent = env.agent_player
    agent.devCards["ROADBUILDER"] = 1
    agent.devCardPlayedThisTurn = False
    env.road_building_roads_left = 0  # sentinel

    ts = _TurnState()
    env._apply_main_action(
        agent,
        action_type=ActionType.PLAY_ROAD_BUILDER,
        corner_idx=0,
        edge_idx=0,
        tile_idx=0,
        res1_idx=0,
        res2_idx=0,
        ts=ts,
    )

    assert ts.road_building_roads_left == 2
    assert env.road_building_roads_left == 0
    assert agent.devCards["ROADBUILDER"] == 0
    assert agent.roadBuilderPlayed == 1


def test_buy_dev_card_no_turnstate_change() -> None:
    """A non-follow-on action leaves _TurnState at its defaults."""
    env = _fresh_env()
    agent = env.agent_player
    for res in ("WHEAT", "ORE", "SHEEP"):
        agent.resources[res] = agent.resources.get(res, 0) + 1

    ts = _TurnState()
    env._apply_main_action(
        agent,
        action_type=ActionType.BUY_DEV_CARD,
        corner_idx=0,
        edge_idx=0,
        tile_idx=0,
        res1_idx=0,
        res2_idx=0,
        ts=ts,
    )

    assert ts.robber_placement_pending is False
    assert ts.road_building_roads_left == 0
