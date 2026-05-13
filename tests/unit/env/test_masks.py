"""Direct tests for compute_action_masks.

The function is exercised heavily through the env's smoke test, but
those tests don't pin per-phase mask shapes. These do.
"""

from __future__ import annotations

import numpy as np

from catan_rl.engine.game import catanGame
from catan_rl.env.masks import compute_action_masks
from catan_rl.policy.obs_encoder import EnvObsState
from catan_rl.policy.obs_schema import (
    N_ACTION_TYPES,
    N_EDGES,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    ActionType,
)


def _build_index_maps(board) -> tuple[dict, dict]:
    vertex_to_idx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
    seen: set[tuple[str, str]] = set()
    edge_to_idx: dict[tuple[str, str], int] = {}
    for v_pt, v_obj in board.boardGraph.items():
        for nb_pt in v_obj.neighbors:
            s1, s2 = str(v_pt), str(nb_pt)
            key = (s1, s2) if s1 < s2 else (s2, s1)
            if key not in seen:
                seen.add(key)
                edge_to_idx[key] = len(edge_to_idx)
    return vertex_to_idx, edge_to_idx


def test_mask_shapes() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=True), vmap, emap
    )
    assert masks["type"].shape == (N_ACTION_TYPES,)
    assert masks["corner_settlement"].shape == (N_VERTICES,)
    assert masks["corner_city"].shape == (N_VERTICES,)
    assert masks["edge"].shape == (N_EDGES,)
    assert masks["tile"].shape == (N_TILES,)
    assert masks["resource1_trade"].shape == (N_RESOURCES,)
    assert masks["resource1_discard"].shape == (N_RESOURCES,)
    assert masks["resource1_default"].shape == (N_RESOURCES,)
    assert masks["resource2_default"].shape == (N_RESOURCES,)
    for v in masks.values():
        assert v.dtype == bool


def test_setup_step_0_allows_only_settlement() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=True, setup_step=0), vmap, emap
    )
    legal = np.flatnonzero(masks["type"])
    assert legal.tolist() == [ActionType.BUILD_SETTLEMENT]
    assert masks["corner_settlement"].any()


def test_setup_step_1_allows_only_road() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    # Player needs an existing settlement before any road is "legal" in setup.
    settle_v = next(iter(game.board.get_setup_settlements(player)))
    player.build_settlement(settle_v, game.board, is_free=True)
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=True, setup_step=1), vmap, emap
    )
    legal = np.flatnonzero(masks["type"])
    assert legal.tolist() == [ActionType.BUILD_ROAD]
    assert masks["edge"].any()


def test_roll_pending_allows_only_roll_dice() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game,
        player,
        EnvObsState(initial_placement_phase=False, roll_pending=True),
        vmap,
        emap,
    )
    legal = np.flatnonzero(masks["type"])
    assert legal.tolist() == [ActionType.ROLL_DICE]


def test_robber_placement_allows_move_robber() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game,
        player,
        EnvObsState(initial_placement_phase=False, robber_placement_pending=True),
        vmap,
        emap,
    )
    assert masks["type"][ActionType.MOVE_ROBBER]
    assert masks["tile"].any()


def test_discard_pending_allows_discard_with_resource_mask() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    player.resources = {"WOOD": 2, "BRICK": 0, "WHEAT": 1, "ORE": 0, "SHEEP": 0}
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game,
        player,
        EnvObsState(initial_placement_phase=False, discard_pending=True),
        vmap,
        emap,
    )
    legal = np.flatnonzero(masks["type"])
    assert legal.tolist() == [ActionType.DISCARD]
    # Only WOOD (idx 0) and WHEAT (idx 2) are in the player's hand.
    assert masks["resource1_discard"][0]
    assert masks["resource1_discard"][2]
    assert not masks["resource1_discard"][1]


def test_main_turn_always_allows_end_turn() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=False), vmap, emap
    )
    assert masks["type"][ActionType.END_TURN]


def test_main_turn_with_settlement_resources_enables_settlement() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    # Place an initial settlement so road->settlement adjacency works.
    settle_v = next(iter(game.board.get_setup_settlements(player)))
    player.build_settlement(settle_v, game.board, is_free=True)
    # Place a road from that settlement so the player has a road network.
    road = next(iter(game.board.get_setup_roads(player)))
    player.build_road(road[0], road[1], game.board, is_free=True)
    # Give the player the resources for another settlement.
    player.resources = {"WOOD": 1, "BRICK": 1, "WHEAT": 1, "ORE": 0, "SHEEP": 1}
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=False), vmap, emap
    )
    # Whether or not BUILD_SETTLEMENT fires depends on whether there's a
    # legal vertex 2-away from existing settlements. The road we built
    # creates such a vertex on at least one side; test that the type
    # head allows it under those conditions.
    if game.board.get_potential_settlements(player):
        assert masks["type"][ActionType.BUILD_SETTLEMENT]
        assert masks["corner_settlement"].any()


def test_main_turn_with_no_resources_only_end_turn() -> None:
    game = catanGame(render_mode=None)
    player = next(iter(game.playerQueue.queue))
    player.resources = {"WOOD": 0, "BRICK": 0, "WHEAT": 0, "ORE": 0, "SHEEP": 0}
    player.devCards = {"KNIGHT": 0, "VP": 0, "ROADBUILDER": 0, "YEAROFPLENTY": 0, "MONOPOLY": 0}
    vmap, emap = _build_index_maps(game.board)
    masks = compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=False), vmap, emap
    )
    legal_types = set(np.flatnonzero(masks["type"]).tolist())
    # Should only have END_TURN with empty hand.
    assert legal_types == {ActionType.END_TURN}


def test_env_compute_masks_delegates_correctly() -> None:
    """Compute masks via the env wrapper and via the standalone function.
    Results must be identical when the env_state matches.
    """
    from catan_rl.env.catan_env import CatanEnv

    env = CatanEnv(opponent_type="random", max_turns=200)
    env.reset(seed=0)
    env_masks = env.get_action_masks()

    vmap, emap = _build_index_maps(env.game.board)
    standalone_masks = compute_action_masks(
        env.game,
        env.agent_player,
        EnvObsState(
            initial_placement_phase=env.initial_placement_phase,
            setup_step=env._setup_step,
            roll_pending=env.roll_pending,
            discard_pending=env.discard_pending,
            robber_placement_pending=env.robber_placement_pending,
            road_building_roads_left=env.road_building_roads_left,
            last_dice_roll=env.last_dice_roll,
        ),
        vmap,
        emap,
    )
    for key in env_masks:
        assert np.array_equal(env_masks[key], standalone_masks[key]), (
            f"env vs standalone disagree on {key}"
        )
