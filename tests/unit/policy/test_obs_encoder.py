"""Unit tests for ObsEncoder.

The obs encoder is the highest-risk module for silent bugs: shapes and
dtypes can match while resource ordering is wrong, mask interpretation
is flipped, or per-tile dims are off by one. Tests are deliberately
exhaustive on shape/dtype contracts and use synthetic game state to pin
the semantics of each obs field.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.engine.game import catanGame
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.policy.obs_encoder import (
    EDGE_FEATURE_DIM,
    HEX_FEATURE_DIM,
    VERTEX_FEATURE_DIM,
    EnvObsState,
    ObsEncoder,
)
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    DEV_CARD_ORDER,
    N_DEV_TYPES,
    N_EDGES,
    N_OPP_POLICY_SLOTS,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    OPP_KIND_HEURISTIC,
    OPP_KIND_UNKNOWN,
    RESOURCES_CW,
    TILE_DIM,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def game_with_players() -> tuple[catanGame, object, object]:
    game = catanGame(render_mode=None)
    players = list(game.playerQueue.queue)
    return game, players[0], players[1]


@pytest.fixture
def encoder(game_with_players: tuple[catanGame, object, object]) -> ObsEncoder:
    return ObsEncoder(game_with_players[0].board)


@pytest.fixture
def initial_env_state() -> EnvObsState:
    return EnvObsState(initial_placement_phase=True)


# ---------------------------------------------------------------------------
# Shape + dtype contract
# ---------------------------------------------------------------------------


def test_all_keys_present(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    expected = {
        "tile_representations",
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
        "hex_features",
        "vertex_features",
        "edge_features",
        "global_features",
        "is_setup",
        "opponent_kind",
        "opponent_policy_id",
    }
    assert set(obs.keys()) == expected


def test_shapes_match_schema(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    assert obs["tile_representations"].shape == (N_TILES, TILE_DIM)
    assert obs["current_player_main"].shape == (CURR_PLAYER_DIM,)
    assert obs["next_player_main"].shape == (NEXT_PLAYER_DIM,)
    assert obs["current_dev_counts"].shape == (N_DEV_TYPES,)
    assert obs["next_played_dev_counts"].shape == (N_DEV_TYPES,)
    assert obs["hex_features"].shape == (N_TILES, HEX_FEATURE_DIM)
    assert obs["vertex_features"].shape == (N_VERTICES, VERTEX_FEATURE_DIM)
    assert obs["edge_features"].shape == (N_EDGES, EDGE_FEATURE_DIM)


def test_dtypes(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    for k in (
        "tile_representations",
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
        "hex_features",
        "vertex_features",
        "edge_features",
    ):
        assert obs[k].dtype == np.float32, f"{k} dtype mismatch"
    assert obs["opponent_kind"].dtype == np.int64
    assert obs["opponent_policy_id"].dtype == np.int64


# ---------------------------------------------------------------------------
# Resource ordering — the highest-risk footgun (CLAUDE.md rule #5)
# ---------------------------------------------------------------------------


def test_resource_ordering_charlesworth(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """current_player_main's first 5 dims must be in Charlesworth order
    (WOOD, BRICK, WHEAT, ORE, SHEEP), NOT engine alphabetical."""
    game, agent, opp = game_with_players
    # Set distinctive resource counts that disambiguate the ordering.
    agent.resources = {"WOOD": 1, "BRICK": 2, "WHEAT": 3, "ORE": 4, "SHEEP": 5}
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    # The first 5 dims are resources / 8.0.
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / 8.0
    np.testing.assert_allclose(obs["current_player_main"][:5], expected, atol=1e-6)
    # Spot-check WOOD is index 0 and SHEEP is index 4 (not the alphabetical
    # BRICK=0 / WOOD=4 layout).
    assert RESOURCES_CW[0] == "WOOD"
    assert RESOURCES_CW[4] == "SHEEP"


def test_opponent_vp_is_live_not_stale(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """FN3 regression: the opponent VP obs feature (next_player_main[5]) must be
    computed LIVE (victoryPoints - hidden VP cards), not read from the cached
    player.visibleVictoryPoints, which the engine only refreshes at init +
    VP-card buy — so it goes stale after every settlement/city (the dominant
    VP source in 1v1)."""
    game, agent, opp = game_with_players
    # Engine state after the opponent has built to 4 VP, but with the cached
    # attribute left at its stale post-init value (what the engine actually does
    # — build_settlement/build_city never refresh visibleVictoryPoints).
    opp.victoryPoints = 4
    opp.devCards["VP"] = 0
    opp.visibleVictoryPoints = 0  # stale cache
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    assert obs["next_player_main"][5] == pytest.approx(4.0 / 15.0)  # live, not stale 0

    # Hidden VP cards stay hidden: a VP card raises victoryPoints but not visible.
    opp.victoryPoints = 6
    opp.devCards["VP"] = 2
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    assert obs["next_player_main"][5] == pytest.approx(4.0 / 15.0)  # 6 - 2 hidden


def test_next_player_resources_in_charlesworth_order(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    opp.resources = {"WOOD": 6, "BRICK": 5, "WHEAT": 4, "ORE": 3, "SHEEP": 2}
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    expected = np.array([6.0, 5.0, 4.0, 3.0, 2.0]) / 8.0
    np.testing.assert_allclose(obs["next_player_main"][:5], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Per-tile features — static dims
# ---------------------------------------------------------------------------


def test_tile_static_intrinsic_dims_match_engine(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    # For each tile, the resource one-hot (dims 0..5) must agree with the engine.
    res_layout = ("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD", "DESERT")
    for h_idx in range(N_TILES):
        actual_res = game.board.hexTileDict[h_idx].resource_type
        oh = obs["tile_representations"][h_idx, :6]
        assert oh.sum() == 1.0, f"tile {h_idx}: resource one-hot not exclusive: {oh}"
        chosen = res_layout[int(np.argmax(oh))]
        assert chosen == actual_res, (
            f"tile {h_idx}: encoder says {chosen!r}, engine says {actual_res!r}"
        )


def test_has_robber_is_dynamic(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """dim 17 (has_robber) reflects current board state, not cached state."""
    game, agent, opp = game_with_players
    obs_before = encoder.build_obs(game, agent, opp, initial_env_state)
    # Robber is on the desert at start; flip it.
    for h_idx in range(N_TILES):
        game.board.hexTileDict[h_idx].has_robber = False
    game.board.hexTileDict[5].has_robber = True
    obs_after = encoder.build_obs(game, agent, opp, initial_env_state)
    assert obs_after["tile_representations"][5, 17] == 1.0
    assert obs_before["tile_representations"][5, 17] != obs_after["tile_representations"][5, 17]


def test_per_tile_dim_breakdown_sums_to_seventy_nine() -> None:
    """Sanity check that 6 + 11 + 1 + 1 + 36 + 24 = 79."""
    assert 6 + 11 + 1 + 1 + 36 + 24 == 79
    assert TILE_DIM == 79


# ---------------------------------------------------------------------------
# Per-vertex / per-edge GNN features
# ---------------------------------------------------------------------------


def test_vertex_features_owner_is_one_hot(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """At game start no vertex has an owner; the 'empty' slot must be 1.0."""
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    owner_block = obs["vertex_features"][:, :3]
    # Every row's owner block sums to 1 (one-hot).
    assert np.allclose(owner_block.sum(axis=1), 1.0)
    # All vertices empty → empty column = 1.
    assert np.allclose(owner_block[:, 0], 1.0)


def test_vertex_features_port_block_sums_to_one(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """Port one-hot (dims 6..13) is exclusive — every vertex has exactly
    one port slot set (most are 'NONE')."""
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    port_block = obs["vertex_features"][:, 6:13]
    assert np.allclose(port_block.sum(axis=1), 1.0)


def test_edge_features_no_road_at_game_start(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    # Owner one-hot: dim 0 (no-road) must be 1 everywhere.
    assert np.allclose(obs["edge_features"][:, 0], 1.0)
    # has_road flag (dim 3) is 0 everywhere.
    assert np.allclose(obs["edge_features"][:, 3], 0.0)


# ---------------------------------------------------------------------------
# Dev card counts
# ---------------------------------------------------------------------------


def test_current_dev_counts_zero_at_start(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    assert obs["current_dev_counts"].sum() == 0.0
    assert obs["next_played_dev_counts"].sum() == 0.0


def test_current_dev_counts_after_buying(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    # Mock: agent has 2 KNIGHT and 1 YOP in hand; opp has played 3 KNIGHT.
    agent.devCards = {"KNIGHT": 2, "VP": 0, "ROADBUILDER": 0, "YEAROFPLENTY": 1, "MONOPOLY": 0}
    opp.knightsPlayed = 3
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    cur = obs["current_dev_counts"]
    assert cur[DEV_CARD_ORDER.index("KNIGHT")] == 2.0
    assert cur[DEV_CARD_ORDER.index("YEAROFPLENTY")] == 1.0
    next_played = obs["next_played_dev_counts"]
    assert next_played[DEV_CARD_ORDER.index("KNIGHT")] == 3.0


# ---------------------------------------------------------------------------
# Opp-id embedding inputs
# ---------------------------------------------------------------------------


def test_opp_id_passes_through_when_unmasked(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
) -> None:
    game, agent, opp = game_with_players
    state = EnvObsState(
        initial_placement_phase=True,
        opp_kind=OPP_KIND_HEURISTIC,
        opp_policy_id=7,
        opp_id_masked=False,
    )
    obs = encoder.build_obs(game, agent, opp, state)
    assert int(obs["opponent_kind"]) == OPP_KIND_HEURISTIC
    assert int(obs["opponent_policy_id"]) == 7


def test_opp_id_masked_overrides_to_unknown(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
) -> None:
    game, agent, opp = game_with_players
    state = EnvObsState(
        initial_placement_phase=True,
        opp_kind=OPP_KIND_HEURISTIC,
        opp_policy_id=7,
        opp_id_masked=True,
    )
    obs = encoder.build_obs(game, agent, opp, state)
    assert int(obs["opponent_kind"]) == OPP_KIND_UNKNOWN
    assert int(obs["opponent_policy_id"]) == N_OPP_POLICY_SLOTS - 1


def test_opp_id_out_of_range_clamps_to_unknown(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
) -> None:
    game, agent, opp = game_with_players
    state = EnvObsState(
        initial_placement_phase=True,
        opp_kind=999,
        opp_policy_id=-1,
    )
    obs = encoder.build_obs(game, agent, opp, state)
    assert int(obs["opponent_kind"]) == OPP_KIND_UNKNOWN
    assert 0 <= int(obs["opponent_policy_id"]) < N_OPP_POLICY_SLOTS


# ---------------------------------------------------------------------------
# Phase-flag interaction
# ---------------------------------------------------------------------------


def test_phase_flags_exclusive(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
) -> None:
    """Exactly one of {setup, roll, main, robber, discard} fires per state.

    Layout in current_player_main: after 5 res + 1 VP + 5 income + 5 trade
    + 6 port + 2 LR/LA + 1 road_len + 1 knights + 3 building_left + 5 devs
    = 34 dims, then 5 phase flags at dims 34..38.
    """
    game, agent, opp = game_with_players
    flag_offset = 34
    for state_kwargs in (
        {"initial_placement_phase": True},
        {"initial_placement_phase": False, "roll_pending": True},
        {"initial_placement_phase": False, "robber_placement_pending": True},
        {"initial_placement_phase": False, "discard_pending": True},
        {"initial_placement_phase": False},  # main phase
    ):
        state = EnvObsState(**state_kwargs)
        obs = encoder.build_obs(game, agent, opp, state)
        flags = obs["current_player_main"][flag_offset : flag_offset + 5]
        assert flags.sum() == 1.0, f"phase flags not exclusive for {state_kwargs}: {flags}"


# ---------------------------------------------------------------------------
# Hand tracker integration
# ---------------------------------------------------------------------------


def test_hand_tracker_overrides_opp_resources(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    """When a hand_tracker is provided, opp resources come from the tracker
    (the broadcast-derived source), not opp.resources directly."""
    game, agent, opp = game_with_players
    # opp.resources says 0/0/0/0/0, but the tracker says something different
    # (a scenario only possible if the tracker missed an event — useful for
    # testing the integration without engaging the broadcast bus).
    opp.resources = dict.fromkeys(("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"), 0)
    tracker = BroadcastHandTracker([agent.name, opp.name])
    tracker._hands[opp.name] = {"WOOD": 3, "BRICK": 2, "WHEAT": 1, "ORE": 0, "SHEEP": 0}
    obs_with = encoder.build_obs(game, agent, opp, initial_env_state, hand_tracker=tracker)
    np.testing.assert_allclose(
        obs_with["next_player_main"][:5],
        np.array([3.0, 2.0, 1.0, 0.0, 0.0]) / 8.0,
        atol=1e-6,
    )
    # Without the tracker, falls back to opp.resources (zeros here).
    obs_no = encoder.build_obs(game, agent, opp, initial_env_state)
    np.testing.assert_allclose(obs_no["next_player_main"][:5], np.zeros(5), atol=1e-6)


# ---------------------------------------------------------------------------
# Validity / NaN / range
# ---------------------------------------------------------------------------


def test_no_nans_or_infs(
    encoder: ObsEncoder,
    game_with_players: tuple[catanGame, object, object],
    initial_env_state: EnvObsState,
) -> None:
    game, agent, opp = game_with_players
    obs = encoder.build_obs(game, agent, opp, initial_env_state)
    for k, v in obs.items():
        if v.dtype.kind == "f":
            assert np.isfinite(v).all(), f"{k} contains NaN/inf"


def test_construction_rejects_undersized_vertex_or_edge_dim(
    game_with_players: tuple[catanGame, object, object],
) -> None:
    game, _, _ = game_with_players
    with pytest.raises(ValueError):
        ObsEncoder(game.board, vertex_dim=4)
    with pytest.raises(ValueError):
        ObsEncoder(game.board, edge_dim=4)
