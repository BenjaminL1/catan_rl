"""V2 — player-side panels.

Two side panels (one per actor) showing:

* Name + kind (random / heuristic / policy) + a color swatch.
* Victory points (cap = 15 for 1v1 Colonist).
* Resource counts (5 columns: wood, brick, wheat, ore, sheep).
* Dev cards — hand (omniscient) + played, in 5 columns matching the
  schema's ``STATE_DEV_CARD_ORDER``.

All draws are pure pygame primitives; no external assets. Panel
geometry is computed once per layout build and threaded into
``render_panels`` along with the current step's
:class:`StepStateSnapshot`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import pygame

from catan_rl.replay.schema import Metadata, PlayerStateSnapshot, StepStateSnapshot

# Match board_renderer's actor colors so the panel and the on-board
# placements read as a single team. Kept in sync by copying the
# RGB triples; importing from board_renderer would couple this
# module's "actor-as-a-team" semantics to the board's rendering
# layer, which we want to keep optional.
_ACTOR_COLORS: dict[str, tuple[int, int, int]] = {
    "player_a": (228, 96, 96),
    "player_b": (96, 168, 232),
}

#: Resource columns. Order matches ``STATE_RESOURCE_ORDER`` from the
#: schema — Charlesworth's wood/brick/wheat/ore/sheep convention.
_RESOURCE_COLUMNS: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Dev card columns. Order matches ``STATE_DEV_CARD_ORDER``.
_DEV_COLUMNS: tuple[str, ...] = ("KNIGHT", "ROAD_BUILDER", "YEAR_OF_PLENTY", "MONOPOLY", "VP")

#: Short labels for tight column widths.
_RES_LABELS: dict[str, str] = {
    "WOOD": "L",
    "BRICK": "B",
    "WHEAT": "G",
    "ORE": "O",
    "SHEEP": "S",
}
_DEV_LABELS: dict[str, str] = {
    "KNIGHT": "K",
    "ROAD_BUILDER": "R",
    "YEAR_OF_PLENTY": "Y",
    "MONOPOLY": "M",
    "VP": "V",
}

_PANEL_BG = (40, 46, 60)
_PANEL_FG = (220, 224, 232)
_PANEL_DIM = (148, 156, 172)
_PANEL_BORDER = (88, 100, 124)
_CELL_BG = (28, 32, 44)
_CELL_FG = (236, 238, 244)


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    """Pixel rectangles for each panel. ``board_rect`` is what the
    board renderer should use for its own layout — the available
    horizontal slice between the two side panels."""

    panel_a_rect: pygame.Rect
    panel_b_rect: pygame.Rect
    board_rect: pygame.Rect


def compute_panel_geometry(
    window_size: tuple[int, int],
    *,
    panel_width: int = 240,
    top_margin: int = 64,
    bottom_margin: int = 24,
) -> PanelGeometry:
    """Carve the window into ``[panel_a | board | panel_b]`` strips.

    ``top_margin`` reserves space above the panels for the existing
    metadata header (rendered by :mod:`event_loop`); ``bottom_margin``
    keeps a clean edge below.

    The function clamps ``panel_width`` so the central board strip is
    at least 360 px wide (the smallest size that produces a
    recognizable 19-hex layout).
    """
    w, h = window_size
    max_panel_w = max(120, (w - 360) // 2)
    panel_w = min(panel_width, max_panel_w)
    panel_a_rect = pygame.Rect(0, top_margin, panel_w, h - top_margin - bottom_margin)
    panel_b_rect = pygame.Rect(w - panel_w, top_margin, panel_w, h - top_margin - bottom_margin)
    board_rect = pygame.Rect(panel_w, top_margin, w - 2 * panel_w, h - top_margin - bottom_margin)
    return PanelGeometry(
        panel_a_rect=panel_a_rect,
        panel_b_rect=panel_b_rect,
        board_rect=board_rect,
    )


def render_panels(
    screen: Any,
    metadata: Metadata,
    state: StepStateSnapshot,
    geom: PanelGeometry,
    *,
    font: Any,
    small_font: Any,
) -> None:
    """Draw both player panels for ``state``."""
    _draw_panel(
        screen,
        rect=geom.panel_a_rect,
        actor="player_a",
        kind=metadata.player_a.kind,
        snap=state.players["player_a"],
        is_longest_road=state.longest_road_holder == "player_a",
        is_largest_army=state.largest_army_holder == "player_a",
        font=font,
        small_font=small_font,
    )
    _draw_panel(
        screen,
        rect=geom.panel_b_rect,
        actor="player_b",
        kind=metadata.player_b.kind,
        snap=state.players["player_b"],
        is_longest_road=state.longest_road_holder == "player_b",
        is_largest_army=state.largest_army_holder == "player_b",
        font=font,
        small_font=small_font,
    )


def _draw_panel(
    screen: Any,
    *,
    rect: pygame.Rect,
    actor: str,
    kind: str,
    snap: PlayerStateSnapshot,
    is_longest_road: bool,
    is_largest_army: bool,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, _PANEL_BG, rect)
    pygame.draw.rect(screen, _PANEL_BORDER, rect, width=2)
    pad = 12
    inner_left = rect.left + pad
    inner_right = rect.right - pad
    inner_top = rect.top + pad

    # ---- header row: swatch + actor name + kind ----
    swatch_size = 14
    color = _ACTOR_COLORS.get(actor, (200, 200, 200))
    pygame.draw.rect(
        screen,
        color,
        pygame.Rect(inner_left, inner_top + 4, swatch_size, swatch_size),
    )
    name_surf = font.render(f"{actor}  ({kind})", True, _PANEL_FG)
    screen.blit(name_surf, (inner_left + swatch_size + 8, inner_top))

    y = inner_top + 28

    # ---- VP row ----
    vp_surf = font.render(f"VP {snap.vp:>2} / 15", True, _PANEL_FG)
    screen.blit(vp_surf, (inner_left, y))

    # Trophy markers.
    trophy_x = inner_left + 130
    if is_longest_road:
        lr_surf = small_font.render("LR", True, (240, 196, 96))
        screen.blit(lr_surf, (trophy_x, y + 4))
        trophy_x += 26
    if is_largest_army:
        la_surf = small_font.render("LA", True, (240, 196, 96))
        screen.blit(la_surf, (trophy_x, y + 4))

    y += 30

    # ---- resources row ----
    _section_header(screen, "Resources", inner_left, y, small_font)
    y += 18
    y = _draw_count_row(
        screen,
        keys=_RESOURCE_COLUMNS,
        counts=snap.resources,
        labels=_RES_LABELS,
        left=inner_left,
        right=inner_right,
        y=y,
        font=small_font,
    )
    y += 12

    # ---- dev cards hand (omniscient) ----
    _section_header(screen, "Dev cards (hand)", inner_left, y, small_font)
    y += 18
    y = _draw_count_row(
        screen,
        keys=_DEV_COLUMNS,
        counts=snap.dev_cards_hand,
        labels=_DEV_LABELS,
        left=inner_left,
        right=inner_right,
        y=y,
        font=small_font,
    )
    y += 12

    # ---- dev cards played ----
    _section_header(screen, "Dev cards (played)", inner_left, y, small_font)
    y += 18
    _draw_count_row(
        screen,
        keys=_DEV_COLUMNS,
        counts=snap.dev_cards_played,
        labels=_DEV_LABELS,
        left=inner_left,
        right=inner_right,
        y=y,
        font=small_font,
    )


def _section_header(screen: Any, label: str, x: int, y: int, font: Any) -> None:
    surf = font.render(label, True, _PANEL_DIM)
    screen.blit(surf, (x, y))


def _draw_count_row(
    screen: Any,
    *,
    keys: tuple[str, ...],
    counts: dict[str, int],
    labels: dict[str, str],
    left: int,
    right: int,
    y: int,
    font: Any,
) -> int:
    """Draw a horizontal row of count cells. Returns the y-coordinate
    just below the row, for chained sections."""
    n = len(keys)
    available = right - left
    cell_w = max(28, available // n - 4)
    spacing = (available - cell_w * n) // max(1, n - 1) if n > 1 else 0
    cell_h = 32
    x = left
    for key in keys:
        count = int(counts.get(key, 0))
        cell_rect = pygame.Rect(x, y, cell_w, cell_h)
        pygame.draw.rect(screen, _CELL_BG, cell_rect)
        pygame.draw.rect(screen, _PANEL_BORDER, cell_rect, width=1)
        # Letter glyph on top, count below.
        glyph_surf = font.render(labels.get(key, key[:1]), True, _PANEL_DIM)
        glyph_rect = glyph_surf.get_rect(center=(cell_rect.centerx, y + 9))
        screen.blit(glyph_surf, glyph_rect)
        count_surf = font.render(str(count), True, _CELL_FG)
        count_rect = count_surf.get_rect(center=(cell_rect.centerx, y + 23))
        screen.blit(count_surf, count_rect)
        x += cell_w + spacing
    return y + cell_h
