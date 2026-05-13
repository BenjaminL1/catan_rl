"""Standalone action-mask computation for 1v1 Catan.

Extracted from ``CatanEnv._compute_masks`` so the BC dataset generator
(Step 3) can produce the *same* masks the env emits at training time,
without instantiating an env. The env's ``_compute_masks`` delegates
here.

The function takes:

  - ``game``: a live ``catanGame`` (needs ``board``, ``board.devCardStack``).
  - ``acting_player``: the player whose action-set we're computing.
  - ``env_state``: subset of env state-machine flags
    (``EnvObsState`` from :mod:`catan_rl.policy.obs_encoder`, shared
    with the obs encoder so callers only build one).
  - ``vertex_to_idx`` / ``edge_to_idx``: precomputed index maps
    (the env builds these once in ``_build_index_maps``; the BC
    dataset can derive them via ``catan_rl.policy.board_geometry``).

Returns the same 9-key mask dict the env produces. The dict layout
matches the v2 6-head autoregressive action contract.

Design note: phase flags in ``env_state`` (``initial_placement_phase``,
``roll_pending``, ``discard_pending``, ``robber_placement_pending``,
``road_building_roads_left``) are *per-player* in BC use — when P1 is
the acting player and waiting to roll, P1's roll_pending is True; when
P2 is the acting player, P2's roll_pending is whatever P2's state
machine says. The env historically only ran the mask for the agent
player (P1) and treated the opponent's turn as internal, so the v1 code
gated the phase branches with ``and is_agent``; that gate is dropped
here because the function is now called for both players in BC.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan_rl.policy.obs_schema import (
    ActionType,
    N_ACTION_TYPES,
    N_EDGES,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    RESOURCES_CW,
)


def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def compute_action_masks(
    game: Any,
    acting_player: Any,
    env_state: Any,
    vertex_to_idx: dict[Any, int],
    edge_to_idx: dict[tuple[str, str], int],
) -> dict[str, np.ndarray]:
    """Compute the 9-key mask dict for ``acting_player`` in this game state.

    Args:
        game: a ``catanGame`` instance.
        acting_player: the player whose decision we're masking.
        env_state: an :class:`~catan_rl.policy.obs_encoder.EnvObsState`
            with the phase flags reflecting the acting player.
        vertex_to_idx: maps board vertex pixel coords to vertex-index
            (matches :class:`CatanEnv._vertex_to_idx`).
        edge_to_idx: maps the canonical edge key
            (``(min(str(v1), str(v2)), max(...))``) to edge-index.

    Returns:
        The 9-key mask dict expected by the v2 action heads:
            ``type``, ``corner_settlement``, ``corner_city``, ``edge``,
            ``tile``, ``resource1_trade``, ``resource1_discard``,
            ``resource1_default``, ``resource2_default``.
    """
    board = game.board
    p = acting_player

    type_mask = np.zeros(N_ACTION_TYPES, dtype=bool)
    corner_set = np.zeros(N_VERTICES, dtype=bool)
    corner_city = np.zeros(N_VERTICES, dtype=bool)
    edge_mask = np.zeros(N_EDGES, dtype=bool)
    tile_mask = np.zeros(N_TILES, dtype=bool)
    res1_trade = np.zeros(N_RESOURCES, dtype=bool)
    res1_disc = np.zeros(N_RESOURCES, dtype=bool)
    res1_def = np.zeros(N_RESOURCES, dtype=bool)
    res2_def = np.zeros(N_RESOURCES, dtype=bool)

    def _pack() -> dict[str, np.ndarray]:
        return {
            "type": type_mask,
            "corner_settlement": corner_set,
            "corner_city": corner_city,
            "edge": edge_mask,
            "tile": tile_mask,
            "resource1_trade": res1_trade,
            "resource1_discard": res1_disc,
            "resource1_default": res1_def,
            "resource2_default": res2_def,
        }

    # ---- Setup phase ----
    if env_state.initial_placement_phase:
        step = env_state.setup_step
        if step in (0, 2):  # settle
            type_mask[ActionType.BUILD_SETTLEMENT] = True
            for v_px in board.get_setup_settlements(p):
                idx = vertex_to_idx.get(v_px)
                if idx is not None:
                    corner_set[idx] = True
        elif step in (1, 3):  # road
            type_mask[ActionType.BUILD_ROAD] = True
            for v1, v2 in board.get_setup_roads(p):
                idx = edge_to_idx.get(_edge_key(v1, v2))
                if idx is not None:
                    edge_mask[idx] = True
        return _pack()

    # ---- Discard ----
    if env_state.discard_pending:
        type_mask[ActionType.DISCARD] = True
        for i, r in enumerate(RESOURCES_CW):
            if p.resources.get(r, 0) > 0:
                res1_disc[i] = True
        if not res1_disc.any():
            res1_disc[:] = True
        return _pack()

    # ---- Robber placement ----
    if env_state.robber_placement_pending:
        type_mask[ActionType.MOVE_ROBBER] = True
        for hex_idx in board.get_robber_spots():
            tile_mask[hex_idx] = True
        if not tile_mask.any():
            tile_mask[:] = True
        return _pack()

    # ---- Roll dice ----
    if env_state.roll_pending:
        type_mask[ActionType.ROLL_DICE] = True
        return _pack()

    # ---- Road builder cleanup ----
    if env_state.road_building_roads_left > 0:
        type_mask[ActionType.BUILD_ROAD] = True
        for v1, v2 in board.get_potential_roads(p):
            idx = edge_to_idx.get(_edge_key(v1, v2))
            if idx is not None:
                edge_mask[idx] = True
        if not edge_mask.any():
            type_mask[ActionType.BUILD_ROAD] = False
            type_mask[ActionType.END_TURN] = True
        return _pack()

    # ---- Main turn ----
    res = p.resources

    type_mask[ActionType.END_TURN] = True

    pot_settle = board.get_potential_settlements(p)
    if (
        res.get("BRICK", 0) >= 1
        and res.get("WOOD", 0) >= 1
        and res.get("SHEEP", 0) >= 1
        and res.get("WHEAT", 0) >= 1
        and p.settlementsLeft > 0
        and pot_settle
    ):
        type_mask[ActionType.BUILD_SETTLEMENT] = True
        for v_px in pot_settle:
            idx = vertex_to_idx.get(v_px)
            if idx is not None:
                corner_set[idx] = True

    pot_city = board.get_potential_cities(p)
    if res.get("ORE", 0) >= 3 and res.get("WHEAT", 0) >= 2 and p.citiesLeft > 0 and pot_city:
        type_mask[ActionType.BUILD_CITY] = True
        for v_px in pot_city:
            idx = vertex_to_idx.get(v_px)
            if idx is not None:
                corner_city[idx] = True

    pot_roads = board.get_potential_roads(p)
    if res.get("BRICK", 0) >= 1 and res.get("WOOD", 0) >= 1 and p.roadsLeft > 0 and pot_roads:
        type_mask[ActionType.BUILD_ROAD] = True
        for v1, v2 in pot_roads:
            idx = edge_to_idx.get(_edge_key(v1, v2))
            if idx is not None:
                edge_mask[idx] = True

    deck_total = sum(board.devCardStack.values())
    if (
        res.get("ORE", 0) >= 1
        and res.get("WHEAT", 0) >= 1
        and res.get("SHEEP", 0) >= 1
        and deck_total > 0
    ):
        type_mask[ActionType.BUY_DEV_CARD] = True

    if p.devCards.get("KNIGHT", 0) > 0 and not p.devCardPlayedThisTurn:
        type_mask[ActionType.PLAY_KNIGHT] = True

    if p.devCards.get("YEAROFPLENTY", 0) > 0 and not p.devCardPlayedThisTurn:
        type_mask[ActionType.PLAY_YOP] = True
        res1_def[:] = True
        res2_def[:] = True

    if p.devCards.get("MONOPOLY", 0) > 0 and not p.devCardPlayedThisTurn:
        type_mask[ActionType.PLAY_MONOPOLY] = True
        res1_def[:] = True

    if p.devCards.get("ROADBUILDER", 0) > 0 and not p.devCardPlayedThisTurn:
        type_mask[ActionType.PLAY_ROAD_BUILDER] = True

    for i, r in enumerate(RESOURCES_CW):
        r_port = f"2:1 {r}"
        if r_port in p.portList and res.get(r, 0) >= 2:
            res1_trade[i] = True
        elif "3:1 PORT" in p.portList and res.get(r, 0) >= 3:
            res1_trade[i] = True
        elif res.get(r, 0) >= 4:
            res1_trade[i] = True
    if res1_trade.any():
        type_mask[ActionType.BANK_TRADE] = True
        res2_def[:] = True

    if not type_mask.any():
        type_mask[ActionType.END_TURN] = True

    return _pack()
