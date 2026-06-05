"""V1 — pygame board renderer.

Takes a :class:`BoardLayout` (precomputed pixel positions for hexes,
vertices, edges, ports) plus the current :class:`StepStateSnapshot`
and draws:

* 19 hex tiles filled with resource colors + number tokens.
* Robber marker on the current robber hex.
* 54 vertex markers — empty by default; settlements / cities are
  rendered as colored polygons for the player who owns them.
* 72 edge markers — empty by default; built roads rendered as
  colored strokes between the two vertex pixels.
* 9 ports — small rim badges at the port-vertex midpoints showing the
  trade ratio + resource glyph.

Drawing is pure pygame primitives. No external image assets.
"""

from __future__ import annotations

import os
from typing import Any

# Suppress pygame's import banner — see event_loop.py for the why.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import pygame

from catan_rl.replay.schema import BoardStatic, StepStateSnapshot
from catan_rl.replay.viewer.board_layout import BoardLayout

# -- color palette -----------------------------------------------------------

#: Resource → fill color (R, G, B). Visually distinct + colorblind-safe.
_RESOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "WOOD": (52, 109, 60),  # forest green
    "BRICK": (181, 78, 56),  # terracotta
    "WHEAT": (228, 191, 86),  # wheat-gold
    "ORE": (130, 138, 156),  # slate
    "SHEEP": (149, 196, 124),  # pasture-green
    "DESERT": (210, 198, 156),  # sand
}

#: Per-actor color (R, G, B). Matches v1 GUI's "black" vs "darkslateblue"
#: roughly, but biased for accessibility against the dark background.
_ACTOR_COLORS: dict[str, tuple[int, int, int]] = {
    "player_a": (228, 96, 96),  # warm red
    "player_b": (96, 168, 232),  # blue
}

_ROBBER_COLOR = (28, 28, 28)
_NUMBER_TOKEN_BG = (240, 234, 218)
_NUMBER_TOKEN_FG = (38, 30, 24)
_HIGH_PROB_TOKEN_FG = (170, 32, 32)  # 6 and 8 — high-probability tokens
_PORT_BG = (50, 78, 110)
_PORT_FG = (228, 234, 244)
_HEX_OUTLINE = (220, 224, 232)

#: Unique single-char glyphs for the 5 resources. WOOD and WHEAT both
#: start with 'W' so we use L for Lumber and G for Grain — the
#: standard Catan shorthand. Used on port badges.
_RESOURCE_GLYPHS: dict[str, str] = {
    "WOOD": "L",
    "BRICK": "B",
    "WHEAT": "G",
    "ORE": "O",
    "SHEEP": "S",
}


def render_board(
    screen: Any,
    board_static: BoardStatic,
    layout: BoardLayout,
    state: StepStateSnapshot,
    *,
    font: Any,
    small_font: Any,
    hex_size: float,
) -> None:
    """Draw the board state into ``screen``. Called once per frame.

    ``state`` is the snapshot at the current step's end (i.e.,
    ``ReplayStep.state_after``); ownership maps + resources + the
    robber position come from there.

    ``hex_size`` is the pixel radius used to lay out the board (see
    :func:`compute_board_layout`). The renderer scales token / robber /
    port / settlement marker sizes proportionally so the board reads
    correctly at any window size.
    """
    _draw_hex_tiles(screen, board_static, layout, state.robber_hex, font, hex_size)
    _draw_ports(screen, board_static, layout, small_font, hex_size)
    _draw_roads(screen, board_static, layout, state, hex_size)
    _draw_settlements_and_cities(screen, layout, state, hex_size)


# ---------------------------------------------------------------------------
# Hex tiles
# ---------------------------------------------------------------------------


def _draw_hex_tiles(
    screen: Any,
    board_static: BoardStatic,
    layout: BoardLayout,
    robber_hex: int,
    font: Any,
    hex_size: float,
) -> None:
    for hex_obj, center, corners in zip(
        board_static.hexes,
        layout.hex_centers,
        layout.hex_corners_by_idx,
        strict=True,
    ):
        color = _RESOURCE_COLORS.get(hex_obj.resource, (64, 64, 64))
        pygame.draw.polygon(screen, color, corners)
        pygame.draw.polygon(screen, _HEX_OUTLINE, corners, width=2)

        if hex_obj.number_token is not None:
            _draw_number_token(screen, center, hex_obj.number_token, font, hex_size)

        if hex_obj.hex_idx == robber_hex:
            _draw_robber(screen, center, hex_size)


def _draw_number_token(
    screen: Any, center: tuple[float, float], token: int, font: Any, hex_size: float
) -> None:
    cx, cy = int(center[0]), int(center[1])
    # Token radius scales with hex_size, clamped to a readable floor.
    radius = max(10, int(hex_size * 0.22))
    pygame.draw.circle(screen, _NUMBER_TOKEN_BG, (cx, cy), radius)
    pygame.draw.circle(screen, _NUMBER_TOKEN_FG, (cx, cy), radius, width=2)
    fg = _HIGH_PROB_TOKEN_FG if token in (6, 8) else _NUMBER_TOKEN_FG
    surf = font.render(str(token), True, fg)
    rect = surf.get_rect(center=(cx, cy))
    screen.blit(surf, rect)


def _draw_robber(screen: Any, center: tuple[float, float], hex_size: float) -> None:
    # Robber sits above the number token, offset proportional to
    # hex_size so it doesn't overlap at small windows or float off at
    # large ones.
    offset = max(16, int(hex_size * 0.38))
    radius = max(5, int(hex_size * 0.12))
    cx, cy = int(center[0]), int(center[1] - offset)
    pygame.draw.circle(screen, _ROBBER_COLOR, (cx, cy), radius)
    pygame.draw.circle(screen, _HEX_OUTLINE, (cx, cy), radius, width=1)


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def _draw_roads(
    screen: Any,
    board_static: BoardStatic,
    layout: BoardLayout,
    state: StepStateSnapshot,
    hex_size: float,
) -> None:
    # Map edge_idx → owning actor (absent if no road on this edge).
    owner: dict[int, str] = {}
    for actor_name, road_indices in state.roads.items():
        for edge_idx in road_indices:
            owner[int(edge_idx)] = actor_name

    width = max(3, int(hex_size * 0.07))
    for edge in board_static.edges:
        owning = owner.get(edge.edge_idx)
        if owning is None:
            continue
        v1 = layout.vertex_pixels[edge.v1_idx]
        v2 = layout.vertex_pixels[edge.v2_idx]
        color = _ACTOR_COLORS.get(owning, (200, 200, 200))
        pygame.draw.line(screen, color, v1, v2, width=width)


# ---------------------------------------------------------------------------
# Settlements + cities
# ---------------------------------------------------------------------------


def _draw_settlements_and_cities(
    screen: Any, layout: BoardLayout, state: StepStateSnapshot, hex_size: float
) -> None:
    # Maps vertex_idx → (kind, actor). Cities overwrite settlements
    # because a city upgrade can replace the original settlement at
    # the same vertex.
    placements: dict[int, tuple[str, str]] = {}
    for actor, v_indices in state.settlements.items():
        for v_idx in v_indices:
            placements[int(v_idx)] = ("settlement", actor)
    for actor, v_indices in state.cities.items():
        for v_idx in v_indices:
            placements[int(v_idx)] = ("city", actor)

    settle_size = max(6, int(hex_size * 0.13))
    city_size = max(8, int(hex_size * 0.17))
    for v_idx, (kind, actor) in placements.items():
        cx, cy = layout.vertex_pixels[v_idx]
        color = _ACTOR_COLORS.get(actor, (200, 200, 200))
        if kind == "settlement":
            _draw_settlement_polygon(screen, (cx, cy), color, settle_size)
        else:
            _draw_city_polygon(screen, (cx, cy), color, city_size)


def _draw_settlement_polygon(
    screen: Any, center: tuple[float, float], color: tuple[int, int, int], s: int
) -> None:
    cx, cy = center
    # Pentagon "house" shape.
    points = [
        (cx - s, cy + s),
        (cx + s, cy + s),
        (cx + s, cy - s * 0.3),
        (cx, cy - s),
        (cx - s, cy - s * 0.3),
    ]
    pygame.draw.polygon(screen, color, points)
    pygame.draw.polygon(screen, _HEX_OUTLINE, points, width=1)


def _draw_city_polygon(
    screen: Any, center: tuple[float, float], color: tuple[int, int, int], s: int
) -> None:
    cx, cy = center
    # Stepped "city" shape: a taller right block + a shorter left block.
    points = [
        (cx - s, cy + s),
        (cx + s, cy + s),
        (cx + s, cy - s * 0.5),
        (cx, cy - s * 0.5),
        (cx, cy - s),
        (cx - s, cy - s),
    ]
    pygame.draw.polygon(screen, color, points)
    pygame.draw.polygon(screen, _HEX_OUTLINE, points, width=1)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


def _draw_ports(
    screen: Any, board_static: BoardStatic, layout: BoardLayout, font: Any, hex_size: float
) -> None:
    radius = max(10, int(hex_size * 0.20))
    for port, anchor in zip(board_static.ports, layout.port_anchors, strict=True):
        cx, cy = int(anchor[0]), int(anchor[1])
        pygame.draw.circle(screen, _PORT_BG, (cx, cy), radius)
        pygame.draw.circle(screen, _HEX_OUTLINE, (cx, cy), radius, width=2)
        if port.resource is None:
            label: str = port.ratio
        else:
            # Resource first-letter glyph table avoids the WOOD/WHEAT
            # 'W' collision. See ``_RESOURCE_GLYPHS``.
            label = f"{port.ratio[0]}/{_RESOURCE_GLYPHS.get(port.resource, '?')}"
        surf = font.render(label, True, _PORT_FG)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)
