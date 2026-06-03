"""Shared pygame rendering primitives (plan §A).

Pure functions that take a ``pygame.Surface`` + scene data and draw.
No global state except for the per-process symbol cache (built lazily on
first use; never mutated after).

Both consumers — the setup labeling tool (``catan_rl.labeling.ui``) and
the live human-vs-AI GUI (``catan_rl.gui.view``) — import from this
module so their visuals stay in sync.

Layer order is fixed (drawn back to front by the caller):

    water  → island  → hex tiles (with bevel)  → number tokens
          → resource symbols  → ports  → vertex markers
          → prior picks  → robber  → top-bar overlay

Per CLAUDE.md rule 8, this module sits under ``gui/`` so RL paths
(``bc/``, ``policy/``, ``env/``, ``ppo/``, ``selfplay/``, ``eval/``)
cannot import it. ``labeling/`` keeps the explicit carve-out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pygame

from catan_rl.gui import render_constants as C

# ---------------------------------------------------------------------------
# Symbol-rendering cache (built on first use, never mutated after)
# ---------------------------------------------------------------------------


@dataclass
class _ResourceSymbolEntry:
    """Pre-rendered symbol surface for one resource.

    ``emoji_surface`` is non-None if we successfully rendered the unicode
    emoji glyph at module-probe time. Otherwise ``fallback`` carries a
    pre-drawn polygon icon. ``size`` is the bounding box used by callers
    to compute placement.
    """

    surface: pygame.Surface
    used_fallback: bool


_SYMBOL_CACHE: dict[str, _ResourceSymbolEntry] = {}
"""Per-process cache of pre-rendered hex-center resource symbols
(font size ``C.RESOURCE_SYMBOL_FONT_SIZE``)."""

_PORT_ICON_CACHE: dict[str, pygame.Surface] = {}
"""Per-process cache of small resource icons used inside port label
badges (font size ``C.PORT_LABEL_ICON_SIZE``). The ``"GENERIC"`` key
holds the ``?`` glyph for 3:1 generic ports."""

_CACHE_INITIALISED: bool = False


def _build_symbol_cache() -> None:
    """Populate ``_SYMBOL_CACHE`` once. Idempotent."""
    global _CACHE_INITIALISED
    if _CACHE_INITIALISED:
        return

    if not pygame.font.get_init():
        pygame.font.init()

    for resource, emoji in C.RESOURCE_EMOJI.items():
        surf = _try_render_emoji(emoji)
        if surf is not None:
            _SYMBOL_CACHE[resource] = _ResourceSymbolEntry(surf, used_fallback=False)
        else:
            _SYMBOL_CACHE[resource] = _ResourceSymbolEntry(
                _draw_fallback_icon(resource), used_fallback=True
            )

    # Build the small port-badge variant alongside the main cache.
    for resource, emoji in C.RESOURCE_EMOJI.items():
        surf = _try_render_emoji(emoji, font_size=C.PORT_LABEL_ICON_SIZE)
        if surf is None:
            surf = _draw_fallback_icon(resource, size_hint=C.PORT_LABEL_ICON_SIZE)
        _PORT_ICON_CACHE[resource] = surf
    # Generic 3:1 port icon: a "?" glyph rendered at the same size.
    _PORT_ICON_CACHE["GENERIC"] = _draw_generic_port_glyph()

    _CACHE_INITIALISED = True


def _draw_generic_port_glyph() -> pygame.Surface:
    """Render the ``?`` mark used on 3:1 generic port labels."""
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont(None, C.PORT_LABEL_ICON_SIZE + 6, bold=True)
    return font.render(C.PORT_LABEL_GENERIC_GLYPH, True, C.PORT_LABEL_COLOR)


def _try_render_emoji(emoji: str, font_size: int | None = None) -> pygame.Surface | None:
    """Probe candidate emoji-capable fonts; return a rendered surface if
    any of them produces a visually non-blank glyph wider than the
    minimum threshold, else None. ``font_size`` defaults to the main
    hex-center size; pass ``C.PORT_LABEL_ICON_SIZE`` for the small
    badge variant."""
    size = font_size if font_size is not None else C.RESOURCE_SYMBOL_FONT_SIZE
    for font_name in C.EMOJI_FONT_CANDIDATES:
        try:
            font = pygame.font.SysFont(font_name, size)
        except Exception:
            continue
        try:
            surf = font.render(emoji, True, (0, 0, 0))
        except Exception:
            continue
        if _surface_has_visible_glyph(surf):
            return surf
    # Final fallback: try the default font.
    try:
        font = pygame.font.SysFont(None, size)
        surf = font.render(emoji, True, (0, 0, 0))
        if _surface_has_visible_glyph(surf):
            return surf
    except Exception:
        pass
    return None


def _surface_has_visible_glyph(surf: pygame.Surface) -> bool:
    """Return True if ``surf`` contains a glyph wider than the threshold.

    Definition: the bounding box of non-transparent / non-background
    pixels must be at least ``EMOJI_PROBE_MIN_BBOX_WIDTH`` px wide AND
    the surface itself must be wider than that threshold (otherwise the
    glyph was rendered as a single-character placeholder).
    """
    if surf.get_width() < C.EMOJI_PROBE_MIN_BBOX_WIDTH:
        return False
    # Use pygame's bounding-rect calculation, which masks against the
    # surface's alpha (or against the colorkey if alpha isn't set).
    try:
        rect = surf.get_bounding_rect()
    except Exception:
        return False
    return rect.width >= C.EMOJI_PROBE_MIN_BBOX_WIDTH


def _draw_fallback_icon(resource: str, size_hint: int | None = None) -> pygame.Surface:
    """Render a drawn-polygon icon for the given resource onto a small
    transparent surface. One-time cost at module-load.

    ``size_hint`` overrides the default sizing — used by the port-badge
    cache to produce smaller icons that fit alongside the ratio text.
    """
    base = size_hint if size_hint is not None else C.RESOURCE_SYMBOL_FONT_SIZE
    size = base + 6
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    color = C.ICON_FALLBACK_COLORS.get(resource, (60, 60, 60))
    cx = cy = size // 2

    if resource == "WOOD":
        # Three stacked triangles → pine tree.
        for i, offset in enumerate((-8, -3, 2)):
            tip = (cx, cy + offset - 4)
            base_l = (cx - 7 + i, cy + offset + 2)
            base_r = (cx + 7 - i, cy + offset + 2)
            pygame.draw.polygon(surf, color, [tip, base_l, base_r])
        # Trunk.
        pygame.draw.rect(surf, (90, 60, 30), (cx - 1, cy + 4, 3, 5))
    elif resource == "BRICK":
        # 2-row brick wall.
        for row in range(2):
            y = cy - 6 + row * 6
            offset = 3 if row == 1 else 0
            for col in range(2):
                rect = (cx - 8 + col * 8 + offset, y, 7, 5)
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (40, 20, 10), rect, 1)
    elif resource == "SHEEP":
        # Three overlapping circles → wool body.
        pygame.draw.circle(surf, color, (cx, cy), 8)
        pygame.draw.circle(surf, color, (cx - 6, cy + 1), 6)
        pygame.draw.circle(surf, color, (cx + 6, cy + 1), 6)
        # Two dark dots for eyes.
        pygame.draw.circle(surf, (40, 40, 40), (cx - 3, cy - 1), 1)
        pygame.draw.circle(surf, (40, 40, 40), (cx + 3, cy - 1), 1)
    elif resource == "WHEAT":
        # Fan of 5 lines → wheat sheaf.
        for i, angle_deg in enumerate((-60, -30, 0, 30, 60)):
            angle = math.radians(angle_deg - 90)  # -90 so 0 points up
            x2 = cx + int(math.cos(angle) * 11)
            y2 = cy + int(math.sin(angle) * 11)
            pygame.draw.line(surf, color, (cx, cy + 4), (x2, y2), 2)
    elif resource == "ORE":
        # Three stacked stone shapes.
        pygame.draw.polygon(surf, color, [(cx - 9, cy + 6), (cx - 4, cy - 4), (cx + 5, cy + 6)])
        pygame.draw.polygon(surf, color, [(cx - 2, cy + 6), (cx + 4, cy - 2), (cx + 9, cy + 6)])
        pygame.draw.polygon(
            surf, (160, 160, 180), [(cx - 6, cy + 8), (cx, cy + 4), (cx + 6, cy + 8)]
        )
    elif resource == "DESERT":
        # Cactus: vertical body with two arms.
        pygame.draw.rect(surf, color, (cx - 2, cy - 8, 5, 16))
        pygame.draw.rect(surf, color, (cx - 8, cy - 2, 6, 3))
        pygame.draw.rect(surf, color, (cx - 8, cy - 6, 3, 5))
        pygame.draw.rect(surf, color, (cx + 3, cy + 2, 6, 3))
        pygame.draw.rect(surf, color, (cx + 6, cy - 2, 3, 5))
    else:
        # Unknown resource: a question mark in a circle.
        pygame.draw.circle(surf, color, (cx, cy), 9, 2)
        font = pygame.font.SysFont(None, 18)
        text = font.render("?", True, color)
        rect = text.get_rect(center=(cx, cy))
        surf.blit(text, rect)
    return surf


# ---------------------------------------------------------------------------
# 1. draw_water
# ---------------------------------------------------------------------------


def draw_water(screen: pygame.Surface, size: tuple[int, int]) -> None:
    """Fill the entire canvas with the water color."""
    screen.fill(C.WATER_COLOR)
    # ``size`` is honored for callers that maintain their own bookkeeping;
    # ``screen.fill`` operates on the whole surface regardless.
    del size


# ---------------------------------------------------------------------------
# 2. draw_island_outline
# ---------------------------------------------------------------------------


def draw_island_outline(screen: pygame.Surface, hex_centers: list[Any]) -> None:
    """Draw a sandy island polygon behind the hexes.

    Strategy: compute the average hex center; for each hex center, push
    radially outward by ``ISLAND_OUTLINE_BUFFER`` plus a small jitter.
    Sort the resulting points by angle from the centroid and connect
    into a filled polygon. The result looks like a coastline without
    needing convex hull math.
    """
    if not hex_centers:
        return
    cx = sum(float(p.x) for p in hex_centers) / len(hex_centers)
    cy = sum(float(p.y) for p in hex_centers) / len(hex_centers)

    # Use only the outermost ring of hexes to define the polygon —
    # inner hexes are well covered by the buffer and would warp the
    # outline. Sort by distance from centroid, keep the top 60%.
    distances: list[tuple[float, Any]] = []
    for p in hex_centers:
        dx = float(p.x) - cx
        dy = float(p.y) - cy
        distances.append((math.hypot(dx, dy), p))
    distances.sort(key=lambda t: -t[0])  # farthest first
    n_outer = max(12, len(hex_centers) * 6 // 10)
    outer = [p for _, p in distances[:n_outer]]

    pts: list[tuple[float, float, float]] = []  # (angle, x, y)
    for p in outer:
        dx = float(p.x) - cx
        dy = float(p.y) - cy
        d = math.hypot(dx, dy) or 1.0
        # Push outward.
        out_x = float(p.x) + dx / d * C.ISLAND_OUTLINE_BUFFER
        out_y = float(p.y) + dy / d * C.ISLAND_OUTLINE_BUFFER
        # Add a small perpendicular jitter for "coastline".
        # Deterministic per-vertex via hash of position so the outline
        # is stable across renders.
        h = (int(p.x) * 73856093) ^ (int(p.y) * 19349663)
        jitter = (h & 0xFF) / 255.0 * 2 - 1  # -1..1
        perp_x = -dy / d
        perp_y = dx / d
        out_x += perp_x * jitter * C.ISLAND_OUTLINE_JITTER
        out_y += perp_y * jitter * C.ISLAND_OUTLINE_JITTER
        ang = math.atan2(out_y - cy, out_x - cx)
        pts.append((ang, out_x, out_y))

    pts.sort(key=lambda t: t[0])
    polygon = [(int(x), int(y)) for _, x, y in pts]
    pygame.draw.polygon(screen, C.SAND_COLOR, polygon)
    pygame.draw.polygon(screen, C.ISLAND_OUTLINE_BORDER_COLOR, polygon, 2)


# ---------------------------------------------------------------------------
# 3. draw_hex_tile
# ---------------------------------------------------------------------------


def draw_hex_tile(
    screen: pygame.Surface,
    hex_tile: Any,
    board: Any,
    with_bevel: bool = True,
) -> None:
    """Draw a single hex tile with optional top/bottom bevel shading."""
    corners = hex_tile.get_corners(board.flat)
    pts = [(int(c.x), int(c.y)) for c in corners]
    base = C.TILE_COLORS.get(hex_tile.resource_type, C.TILE_FALLBACK_COLOR)
    pygame.draw.polygon(screen, base, pts)
    if with_bevel:
        _draw_hex_bevel(screen, hex_tile, board, base)
    pygame.draw.polygon(screen, C.HEX_OUTLINE_COLOR, pts, 2)


def _draw_hex_bevel(
    screen: pygame.Surface,
    hex_tile: Any,
    board: Any,
    base_color: tuple[int, int, int],
) -> None:
    """Add a lighter polygon on the top half + darker on the bottom half."""
    corners = hex_tile.get_corners(board.flat)
    cy = sum(float(c.y) for c in corners) / len(corners)
    light = _shift_color(base_color, C.HEX_BEVEL_LIGHTEN, lighten=True)
    dark = _shift_color(base_color, C.HEX_BEVEL_DARKEN, lighten=False)

    top_pts: list[tuple[int, int]] = []
    bot_pts: list[tuple[int, int]] = []
    for c in corners:
        if float(c.y) <= cy:
            top_pts.append((int(c.x), int(c.y)))
        else:
            bot_pts.append((int(c.x), int(c.y)))

    # Close each half by adding centerline points.
    if top_pts:
        sorted_top = sorted(top_pts, key=lambda p: p[0])
        if sorted_top:
            top_polygon = sorted_top + [
                (sorted_top[-1][0], int(cy)),
                (sorted_top[0][0], int(cy)),
            ]
            if len(top_polygon) >= 3:
                pygame.draw.polygon(screen, light, top_polygon)
    if bot_pts:
        sorted_bot = sorted(bot_pts, key=lambda p: p[0])
        if sorted_bot:
            bot_polygon = sorted_bot + [
                (sorted_bot[-1][0], int(cy)),
                (sorted_bot[0][0], int(cy)),
            ]
            if len(bot_polygon) >= 3:
                pygame.draw.polygon(screen, dark, bot_polygon)


def _shift_color(rgb: tuple[int, int, int], frac: float, lighten: bool) -> tuple[int, int, int]:
    """Shift each channel toward white (lighten) or black (darken) by
    ``frac`` of the gap. ``frac`` is in [0, 1]."""
    if lighten:
        return tuple(min(255, int(c + (255 - c) * frac)) for c in rgb)  # type: ignore[return-value]
    return tuple(max(0, int(c * (1.0 - frac))) for c in rgb)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 4. draw_number_token
# ---------------------------------------------------------------------------


def draw_number_token(screen: pygame.Surface, center: tuple[int, int], number: int) -> None:
    """Draw the Colonist-style square white token at ``center``.

    Red text for 6 and 8. Pip dots underneath showing 2d6 probability.
    7 has no token (and no entry in ``PIP_COUNTS``); the function is a
    no-op for 7.
    """
    if number not in C.PIP_COUNTS:
        return

    cx, cy = center
    size = C.NUMBER_TOKEN_SIZE
    rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
    pygame.draw.rect(screen, C.NUMBER_TOKEN_BG, rect, border_radius=C.NUMBER_TOKEN_CORNER_RADIUS)
    pygame.draw.rect(
        screen,
        C.NUMBER_TOKEN_BORDER,
        rect,
        width=1,
        border_radius=C.NUMBER_TOKEN_CORNER_RADIUS,
    )

    if not pygame.font.get_init():
        pygame.font.init()
    is_red = number in (6, 8)
    text_color = C.NUMBER_TEXT_RED if is_red else C.NUMBER_TEXT_DEFAULT
    font = pygame.font.SysFont(None, 26, bold=True)
    text = font.render(str(number), True, text_color)
    text_rect = text.get_rect(center=(cx, cy - 4))
    screen.blit(text, text_rect)

    # Pip dots underneath.
    n_dots = C.PIP_COUNTS[number]
    dot_color = C.PIP_DOT_COLOR_RED if is_red else C.PIP_DOT_COLOR
    dot_y = cy + size // 2 - 6
    total_width = (n_dots - 1) * C.PIP_DOT_GAP
    dot_x_start = cx - total_width // 2
    for i in range(n_dots):
        dot_x = dot_x_start + i * C.PIP_DOT_GAP
        pygame.draw.circle(screen, dot_color, (dot_x, dot_y), C.PIP_DOT_RADIUS)


# ---------------------------------------------------------------------------
# 5. draw_resource_symbol
# ---------------------------------------------------------------------------


def draw_resource_symbol(
    screen: pygame.Surface, center: tuple[int, int], resource_type: str
) -> None:
    """Blit the resource glyph (emoji or fallback polygon) at ``center``."""
    if not _CACHE_INITIALISED:
        _build_symbol_cache()
    entry = _SYMBOL_CACHE.get(resource_type)
    if entry is None:
        # Build a fallback on the fly for unknown resources.
        entry = _ResourceSymbolEntry(_draw_fallback_icon(resource_type), used_fallback=True)
        _SYMBOL_CACHE[resource_type] = entry
    surf = entry.surface
    rect = surf.get_rect(center=center)
    screen.blit(surf, rect)


# ---------------------------------------------------------------------------
# 6. draw_port_ship
# ---------------------------------------------------------------------------


def draw_port_ship(
    screen: pygame.Surface,
    ratio: str,
    resource_type: str | None,
    anchor: tuple[int, int],
) -> None:
    """Draw a small ship + sail at ``anchor`` with a separate
    cream-colored label badge below the ship.

    The badge shows ``ratio`` (e.g. ``"2:1"`` / ``"3:1"``) followed by
    a small resource icon. For 3:1 generic ports, pass
    ``resource_type=None`` and the badge renders ``?`` instead.
    """
    ax, ay = anchor
    w = C.PORT_SHIP_WIDTH
    h = C.PORT_SHIP_HEIGHT

    # Hull (trapezoid).
    hull = [
        (ax - w // 2, ay + 4),
        (ax + w // 2, ay + 4),
        (ax + w // 2 - 4, ay + h // 2),
        (ax - w // 2 + 4, ay + h // 2),
    ]
    pygame.draw.polygon(screen, C.PORT_HULL_COLOR, hull)
    pygame.draw.polygon(screen, (40, 25, 10), hull, 1)

    # Mast.
    pygame.draw.rect(screen, C.PORT_MAST_COLOR, (ax - 1, ay - h // 2 + 2, 2, h // 2 + 2))

    # Sail — symmetric triangle for better visual weight.
    sail = [
        (ax, ay - h // 2 + 2),
        (ax + w // 2 - 2, ay + 3),
        (ax - w // 2 + 2, ay + 3),
    ]
    pygame.draw.polygon(screen, C.PORT_SAIL_COLOR, sail)
    pygame.draw.polygon(screen, (60, 45, 20), sail, 1)

    # Label badge below the ship.
    _draw_port_label_badge(screen, ratio, resource_type, ax, ay + C.PORT_LABEL_VERTICAL_OFFSET)


def _draw_port_label_badge(
    screen: pygame.Surface,
    ratio: str,
    resource_type: str | None,
    cx: int,
    cy: int,
) -> None:
    """Cream badge: ratio text on the left, resource icon (or ``?``) on right."""
    if not _CACHE_INITIALISED:
        _build_symbol_cache()
    if not pygame.font.get_init():
        pygame.font.init()

    font = pygame.font.SysFont(None, 18, bold=True)
    text = font.render(ratio, True, C.PORT_LABEL_COLOR)
    tw, th = text.get_size()

    icon_key = resource_type if (resource_type and resource_type in _PORT_ICON_CACHE) else "GENERIC"
    icon = _PORT_ICON_CACHE[icon_key]
    iw, ih = icon.get_size()

    gap = 3
    badge_w = tw + gap + iw
    badge_h = max(th, ih)
    pad = C.PORT_LABEL_PADDING

    rect = pygame.Rect(
        cx - badge_w // 2 - pad,
        cy - badge_h // 2 - pad,
        badge_w + 2 * pad,
        badge_h + 2 * pad,
    )
    pygame.draw.rect(screen, C.PORT_LABEL_BG_COLOR, rect, border_radius=3)
    pygame.draw.rect(screen, C.PORT_LABEL_BG_BORDER_COLOR, rect, width=1, border_radius=3)

    text_x = cx - badge_w // 2
    text_y = cy - th // 2
    screen.blit(text, (text_x, text_y))

    icon_x = cx - badge_w // 2 + tw + gap
    icon_y = cy - ih // 2
    screen.blit(icon, (icon_x, icon_y))


# ---------------------------------------------------------------------------
# 7. draw_port_planks
# ---------------------------------------------------------------------------


def draw_port_planks(
    screen: pygame.Surface,
    anchor: tuple[int, int],
    v1_pixel: tuple[int, int],
    v2_pixel: tuple[int, int],
) -> None:
    """Draw two wooden-plank lines from ``anchor`` to each vertex."""
    for vx in (v1_pixel, v2_pixel):
        pygame.draw.line(
            screen,
            C.PORT_PLANK_COLOR,
            (int(vx[0]), int(vx[1])),
            (int(anchor[0]), int(anchor[1])),
            C.PORT_PLANK_WIDTH,
        )


# ---------------------------------------------------------------------------
# 8. draw_vertex_marker
# ---------------------------------------------------------------------------


def draw_vertex_marker(screen: pygame.Surface, pixel: tuple[int, int], state: str) -> None:
    """Draw a state-keyed circular marker at ``pixel``."""
    color = C.VERTEX_COLORS.get(state)
    radius = C.VERTEX_RADII.get(state, 8)
    if color is None:
        return
    px, py = int(pixel[0]), int(pixel[1])
    pygame.draw.circle(screen, color, (px, py), radius)
    pygame.draw.circle(screen, C.VERTEX_BORDER_COLOR, (px, py), radius, 1)


# ---------------------------------------------------------------------------
# 9. draw_robber_pawn
# ---------------------------------------------------------------------------


def draw_robber_pawn(screen: pygame.Surface, hex_tile: Any, board: Any) -> None:
    """Draw the robber pawn on the hex iff ``hex_tile.has_robber``."""
    if not getattr(hex_tile, "has_robber", False):
        return
    center = hex_tile.to_pixel(board.flat)
    cx, cy = int(center.x), int(center.y)
    w = C.ROBBER_BASE_WIDTH
    h = C.ROBBER_HEIGHT
    # Pyramid/triangle body.
    pts = [
        (cx, cy - h // 2 + 4),  # apex
        (cx - w // 2, cy + h // 2 - 2),  # bottom-left
        (cx + w // 2, cy + h // 2 - 2),  # bottom-right
    ]
    pygame.draw.polygon(screen, C.ROBBER_COLOR, pts)
    # Small "head" circle on top for character.
    pygame.draw.circle(screen, C.ROBBER_COLOR, (cx, cy - h // 2 + 1), 4)
    # Outline.
    pygame.draw.polygon(screen, (15, 15, 15), pts, 1)


# ---------------------------------------------------------------------------
# 10. Port-edge collection (board iteration helper)
# ---------------------------------------------------------------------------


def collect_port_edges(
    board: Any,
) -> list[tuple[int, int, str, str | None]]:
    """Walk the board and return every port edge as
    ``(v1_idx, v2_idx, ratio, resource_type_or_None)``.

    A port edge is a pair of adjacent vertices that both carry the same
    ``port`` attribute. The engine sets the same string label on both
    vertices of each port. We deduplicate so each port appears once,
    keyed on the canonical (lower-idx, higher-idx) pair.

    Output shape matches ``draw_port_ship``: ``ratio`` is ``"2:1"`` or
    ``"3:1"``; ``resource_type`` is the long-form resource name
    (``"BRICK"`` etc.) for resource-specific ports or ``None`` for
    generic 3:1 ports.
    """
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, str, str | None]] = []
    for v_obj in board.boardGraph.values():
        port = getattr(v_obj, "port", None)
        if not port:
            continue
        v_idx = v_obj.vertex_index
        for nb_pt in v_obj.neighbors:
            nb_obj = board.boardGraph[nb_pt]
            if getattr(nb_obj, "port", None) != port:
                continue
            nb_idx = nb_obj.vertex_index
            key = (min(v_idx, nb_idx), max(v_idx, nb_idx))
            if key in seen:
                continue
            seen.add(key)
            ratio, resource = _parse_port(port)
            out.append((key[0], key[1], ratio, resource))
    return out


def _parse_port(port: str) -> tuple[str, str | None]:
    """Split an engine port string into ``(ratio, resource_or_None)``.

    ``"2:1 BRICK"`` → ``("2:1", "BRICK")``;
    ``"3:1 PORT"``  → ``("3:1", None)``.
    """
    if "3:1" in port:
        return ("3:1", None)
    for full in ("BRICK", "WOOD", "WHEAT", "SHEEP", "ORE"):
        if full in port:
            return ("2:1", full)
    return ("2:1", None)
