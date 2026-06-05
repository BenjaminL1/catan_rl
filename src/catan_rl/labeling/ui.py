"""Pygame UI for the setup labeling tool (plan §C).

Two layers:

1. :class:`LabelingUIState` — pure-Python state machine. Holds the
   currently-selected settlement + road, validates against legal masks,
   exposes ``submit()`` / ``skip()`` / ``undo()`` / ``quit()``. No
   pygame deps; trivially headless-testable.

2. :class:`LabelingUI` — the pygame renderer + event loop. Wraps
   :class:`LabelingUIState`, renders the hex board, handles mouse +
   keyboard events. Headless via ``SDL_VIDEODRIVER=dummy``.

The plan's §0.2 preflight (vertex-centroid click mapping) is enforced
by :func:`nearest_vertex` which uses Euclidean distance against the
engine's vertex pixel coords — single source of truth for the
coordinate transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from catan_rl.gui import render
from catan_rl.gui import render_constants as RC
from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.session import LabelingSession

PHASE_SETTLEMENT_PICK = "settlement_pick"
PHASE_ROAD_PICK = "road_pick"

# Archetype shortcut letters (first letter of the value, with 'h' for
# OWS_HYBRID since 'o' is taken by OWS).
_ARCHETYPE_BY_KEY: dict[str, Archetype] = {
    "b": Archetype.BALANCED,
    "o": Archetype.OWS,
    "h": Archetype.OWS_HYBRID,
    "r": Archetype.ROAD_BUILDER,
    "x": Archetype.OTHER,
}


def archetype_from_key(key: str) -> Archetype | None:
    """Map a single-letter keypress to an Archetype, or None if unknown."""
    return _ARCHETYPE_BY_KEY.get(key.lower())


def nearest_vertex(
    click_x: float,
    click_y: float,
    vertex_pixels: dict[int, tuple[float, float]],
    max_radius: float = 25.0,
    legal: set[int] | None = None,
) -> int | None:
    """Return the vertex index closest to a click, or None if too far.

    Args:
        click_x, click_y: click coordinates.
        vertex_pixels: dict {vertex_idx → (x, y)}.
        max_radius: maximum click-to-centroid distance to accept.
        legal: optional set of legal vertex indices. If provided, only
            legal vertices are considered.

    Ties broken by lowest index.
    """
    best_idx: int | None = None
    best_d2 = max_radius * max_radius
    for idx, (vx, vy) in vertex_pixels.items():
        if legal is not None and idx not in legal:
            continue
        dx = vx - click_x
        dy = vy - click_y
        d2 = dx * dx + dy * dy
        if d2 < best_d2 or (d2 == best_d2 and best_idx is not None and idx < best_idx):
            best_d2 = d2
            best_idx = idx
    return best_idx


# Re-export from the shared render module so existing test imports
# (``from catan_rl.labeling.ui import collect_port_edges``) keep working.
collect_port_edges = render.collect_port_edges
_parse_port = render._parse_port


def nearest_edge(
    click_x: float,
    click_y: float,
    edge_midpoints: dict[int, tuple[float, float]],
    max_radius: float = 20.0,
    legal: set[int] | None = None,
) -> int | None:
    """Same shape as :func:`nearest_vertex` but for edges (clicks near
    the midpoint between two vertices)."""
    return nearest_vertex(click_x, click_y, edge_midpoints, max_radius, legal)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass
class LabelingUIState:
    """Pure-Python state machine for the labeling UI.

    No pygame dependencies — exists separately so it can be unit-tested
    without booting a display.
    """

    session: LabelingSession
    phase: str = PHASE_SETTLEMENT_PICK
    selected_settlement: int | None = None
    selected_road: int | None = None
    last_click_rejected: bool = False
    last_click_rejected_time_ms: int = 0
    # Cache of the legal-roads mask after a settlement is selected.
    _cached_legal_roads: np.ndarray | None = field(default=None)

    def reset_for_new_scenario(self) -> None:
        self.phase = PHASE_SETTLEMENT_PICK
        self.selected_settlement = None
        self.selected_road = None
        self.last_click_rejected = False
        self._cached_legal_roads = None

    def current_legal_settlements(self) -> set[int]:
        scenario = self.session.current_scenario()
        if scenario is None:
            return set()
        return {int(i) for i in np.where(scenario.legal_settlement_corners)[0]}

    def current_legal_roads(self) -> set[int]:
        """Set of legal road edges given the currently selected settlement."""
        if self.selected_settlement is None or self._cached_legal_roads is None:
            return set()
        return {int(i) for i in np.where(self._cached_legal_roads)[0]}

    def select_settlement(self, vertex_idx: int) -> bool:
        """Accept a settlement pick. Returns True on success, False if
        the pick was illegal (in which case ``last_click_rejected``
        flips to True for a frame so the UI can flash a denied cue)."""
        if self.phase != PHASE_SETTLEMENT_PICK:
            return False
        scenario = self.session.current_scenario()
        if scenario is None:
            return False
        if not (0 <= vertex_idx < 54):
            self.last_click_rejected = True
            return False
        if not bool(scenario.legal_settlement_corners[vertex_idx]):
            self.last_click_rejected = True
            return False
        self.selected_settlement = int(vertex_idx)
        self._cached_legal_roads = scenario.compute_legal_road_edges(vertex_idx)
        self.phase = PHASE_ROAD_PICK
        self.last_click_rejected = False
        return True

    def select_road(self, edge_idx: int) -> bool:
        if self.phase != PHASE_ROAD_PICK:
            return False
        if self._cached_legal_roads is None:
            return False
        if not (0 <= edge_idx < 72):
            self.last_click_rejected = True
            return False
        if not bool(self._cached_legal_roads[edge_idx]):
            self.last_click_rejected = True
            return False
        self.selected_road = int(edge_idx)
        self.last_click_rejected = False
        return True

    def is_ready_to_submit(self) -> bool:
        return (
            self.phase == PHASE_ROAD_PICK
            and self.selected_settlement is not None
            and self.selected_road is not None
        )

    def undo(self) -> None:
        """Revert the most recent pick within the current scenario.

        Cannot undo a submitted scenario (use skip on the next one).
        """
        if self.selected_road is not None:
            self.selected_road = None
        elif self.selected_settlement is not None:
            self.selected_settlement = None
            self._cached_legal_roads = None
            self.phase = PHASE_SETTLEMENT_PICK
        # else: nothing to undo.

    def submit(
        self,
        archetype: Archetype,
        notes: str = "",
        decision_time_ms: int = 0,
    ) -> None:
        if not self.is_ready_to_submit():
            raise RuntimeError("not ready to submit (settlement + road both required)")
        assert self.selected_settlement is not None
        assert self.selected_road is not None
        self.session.submit(
            settlement_vertex=self.selected_settlement,
            road_edge=self.selected_road,
            archetype=archetype,
            notes=notes,
            decision_time_ms=decision_time_ms,
        )
        self.reset_for_new_scenario()

    def skip(self) -> None:
        """Abandon the current draft, advance to a fresh board."""
        self.session.skip()
        self.reset_for_new_scenario()


# ---------------------------------------------------------------------------
# Pygame renderer + event loop
# ---------------------------------------------------------------------------

# Constants kept in this module: things specific to the labeling tool's
# overlays that don't map onto the shared `render.py` primitives.
# Tile / vertex / port colors moved to `gui.render_constants`.
_VERTEX_REJECTED_COLOR = (240, 60, 60)
"""Red flash border drawn after a rejected click."""

_ROAD_LEGAL_COLOR = (90, 200, 250)
"""Cyan-ish color for legal road edge candidates during the road-pick phase."""

_ROAD_SELECTED_COLOR = (250, 230, 60)
"""Bright yellow for the just-clicked road."""

_PRIOR_PICK_ROAD_WIDTH = 6
"""Line thickness (px) for already-placed roads from earlier draft picks."""

_TEXT_COLOR = (235, 235, 235)
"""Top-bar / bottom-bar text on the dark canvas overlay."""


class LabelingUI:
    """Pygame UI driver. Call :meth:`run` to enter the event loop."""

    def __init__(
        self,
        session: LabelingSession,
        screen_size: tuple[int, int] = (1100, 900),
        click_radius: float = 22.0,
    ) -> None:
        import pygame  # Local import: pygame is an optional dependency.

        pygame.init()
        pygame.display.set_caption("Catan Setup Labeling")
        self._pygame = pygame
        self.screen = pygame.display.set_mode(screen_size)
        self.screen_size = screen_size
        self.click_radius = click_radius
        self.session = session
        self.state = LabelingUIState(session)
        # Fonts.
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)
        self.font_large = pygame.font.SysFont(None, 32)
        # Selected archetype defaults to BALANCED; user can change with key.
        self.current_archetype = Archetype.BALANCED
        # Wall-clock per scenario.
        self._scenario_start_ms = self._now_ms()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main event loop. Returns when the user quits."""
        clock = self._pygame.time.Clock()
        running = True
        while running:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    running = False
                elif event.type == self._pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == self._pygame.KEYDOWN:
                    running = self._handle_keydown(event)
            self._render()
            self._pygame.display.flip()
            clock.tick(30)
        self.session.quit()
        self._pygame.quit()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_click(self, pos: tuple[int, int]) -> None:
        click_x, click_y = pos
        scenario = self.session.current_scenario()
        if scenario is None:
            return
        if self.state.phase == PHASE_SETTLEMENT_PICK:
            legal = self.state.current_legal_settlements()
            vidx = nearest_vertex(
                click_x,
                click_y,
                self._vertex_pixels(scenario),
                max_radius=self.click_radius,
            )
            if vidx is None:
                return
            ok = self.state.select_settlement(vidx)
            if not ok:
                self.state.last_click_rejected_time_ms = self._now_ms()
            else:
                # Treat off-legal-vertex clicks as rejected even if a
                # nearby legal vertex existed within radius.
                if vidx not in legal:
                    self.state.last_click_rejected = True
                    self.state.last_click_rejected_time_ms = self._now_ms()
        elif self.state.phase == PHASE_ROAD_PICK:
            legal_roads = self.state.current_legal_roads()
            eidx = nearest_edge(
                click_x,
                click_y,
                self._edge_midpoints(scenario),
                max_radius=self.click_radius,
                legal=legal_roads,
            )
            if eidx is None:
                return
            ok = self.state.select_road(eidx)
            if not ok:
                self.state.last_click_rejected_time_ms = self._now_ms()

    def _handle_keydown(self, event: Any) -> bool:
        """Return False to stop the run loop."""
        K = self._pygame.K_q
        key = event.key
        unicode = getattr(event, "unicode", "").lower()

        if key == K:
            return False  # quit
        if unicode == "s":
            self._try_submit()
        elif unicode == "k":
            self._skip()
        elif unicode == "u":
            self.state.undo()
        else:
            # Archetype shortcut.
            arch = archetype_from_key(unicode)
            if arch is not None:
                self.current_archetype = arch
        return True

    def _try_submit(self) -> None:
        if not self.state.is_ready_to_submit():
            return
        elapsed_ms = self._now_ms() - self._scenario_start_ms
        self.state.submit(
            archetype=self.current_archetype,
            decision_time_ms=elapsed_ms,
        )
        self._scenario_start_ms = self._now_ms()

    def _skip(self) -> None:
        self.state.skip()
        self._scenario_start_ms = self._now_ms()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        scenario = self.session.current_scenario()
        if scenario is None:
            # Done-state still gets water + cream message for visual continuity.
            render.draw_water(self.screen, self.screen_size)
            self._render_done_message()
            return
        # _render_board paints the water + island backdrop itself, so no
        # initial screen.fill is needed.
        self._render_board(scenario)
        self._render_top_bar(scenario)
        self._render_bottom_bar(scenario)

    def _render_board(self, scenario: Any) -> None:
        """Render the board via the shared `gui.render` primitives.

        Layer order (back to front): water → island → hex tiles
        (with bevel) → resource symbols → number tokens → ports →
        prior-pick roads → prior-pick settlements (vertex markers) →
        legal vertices / selected vertex / legal roads / selected road
        → robber pawn → rejected-click flash.
        """
        pygame = self._pygame
        board = scenario.game.board

        # ---- Water + island backdrop -----------------------------
        render.draw_water(self.screen, self.screen_size)
        centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]
        render.draw_island_outline(self.screen, centers)

        # ---- Tiles + per-tile symbol + number token --------------
        for h_idx in range(19):
            hex_tile = board.hexTileDict[h_idx]
            render.draw_hex_tile(self.screen, hex_tile, board, with_bevel=True)
            center = hex_tile.to_pixel(board.flat)
            cx_int, cy_int = int(center.x), int(center.y)
            # Resource symbol — upper half of the hex.
            render.draw_resource_symbol(
                self.screen,
                (cx_int, cy_int + RC.RESOURCE_SYMBOL_VERTICAL_OFFSET),
                hex_tile.resource_type,
            )
            # Number token — lower half of the hex.
            num = getattr(hex_tile, "number_token", None)
            if num is not None and num != 0:
                render.draw_number_token(
                    self.screen,
                    (cx_int, cy_int + RC.NUMBER_TOKEN_VERTICAL_OFFSET),
                    num,
                )

        # ---- Ports -----------------------------------------------
        board_cx = self.screen_size[0] / 2.0
        board_cy = self.screen_size[1] / 2.0
        for v1_idx, v2_idx, ratio, resource in collect_port_edges(board):
            v1_px = scenario._idx_to_vertex_pixel[v1_idx]
            v2_px = scenario._idx_to_vertex_pixel[v2_idx]
            mid_x = (float(v1_px.x) + float(v2_px.x)) / 2.0
            mid_y = (float(v1_px.y) + float(v2_px.y)) / 2.0
            dx = mid_x - board_cx
            dy = mid_y - board_cy
            d = math.hypot(dx, dy) or 1.0
            ax = int(mid_x + dx * RC.PORT_PUSH_DISTANCE / d)
            ay = int(mid_y + dy * RC.PORT_PUSH_DISTANCE / d)
            render.draw_port_planks(
                self.screen,
                (ax, ay),
                (int(v1_px.x), int(v1_px.y)),
                (int(v2_px.x), int(v2_px.y)),
            )
            render.draw_port_ship(self.screen, ratio, resource, (ax, ay))

        # ---- Prior picks: roads first, then settlements ----------
        for pick in scenario.prior_picks:
            color = RC.PLAYER_COLORS.get(pick.player, (180, 180, 180))
            (rv1, rv2) = scenario._idx_to_edge_pixel_pair[pick.road_edge]
            pygame.draw.line(
                self.screen,
                color,
                (int(rv1.x), int(rv1.y)),
                (int(rv2.x), int(rv2.y)),
                _PRIOR_PICK_ROAD_WIDTH,
            )
        for pick in scenario.prior_picks:
            vpx = scenario._idx_to_vertex_pixel[pick.settlement_vertex]
            state_key = (
                RC.VERTEX_STATE_SETTLED_P1 if pick.player == 0 else RC.VERTEX_STATE_SETTLED_P2
            )
            render.draw_vertex_marker(self.screen, (int(vpx.x), int(vpx.y)), state_key)

        # ---- Legal / selected vertex + road overlays -------------
        vertex_pixels = self._vertex_pixels(scenario)
        if self.state.phase == PHASE_SETTLEMENT_PICK:
            for idx in self.state.current_legal_settlements():
                vx, vy = vertex_pixels[idx]
                render.draw_vertex_marker(self.screen, (int(vx), int(vy)), RC.VERTEX_STATE_LEGAL)
        else:  # PHASE_ROAD_PICK
            if self.state.selected_settlement is not None:
                sx, sy = vertex_pixels[self.state.selected_settlement]
                render.draw_vertex_marker(self.screen, (int(sx), int(sy)), RC.VERTEX_STATE_SELECTED)
            edge_mid = self._edge_midpoints(scenario)
            for idx in self.state.current_legal_roads():
                (v1, v2) = scenario._idx_to_edge_pixel_pair[idx]
                pygame.draw.line(
                    self.screen,
                    _ROAD_LEGAL_COLOR,
                    (int(v1.x), int(v1.y)),
                    (int(v2.x), int(v2.y)),
                    4,
                )
                ex, ey = edge_mid[idx]
                pygame.draw.circle(self.screen, _ROAD_LEGAL_COLOR, (int(ex), int(ey)), 5)
            if self.state.selected_road is not None:
                (v1, v2) = scenario._idx_to_edge_pixel_pair[self.state.selected_road]
                pygame.draw.line(
                    self.screen,
                    _ROAD_SELECTED_COLOR,
                    (int(v1.x), int(v1.y)),
                    (int(v2.x), int(v2.y)),
                    8,
                )

        # ---- Robber pawn on the desert ---------------------------
        for hex_tile in board.hexTileDict.values():
            if getattr(hex_tile, "has_robber", False):
                render.draw_robber_pawn(self.screen, hex_tile, board)
                break

        # ---- Rejected-click flash --------------------------------
        if (
            self.state.last_click_rejected
            and self._now_ms() - self.state.last_click_rejected_time_ms < 400
        ):
            pygame.draw.rect(self.screen, _VERTEX_REJECTED_COLOR, (0, 0, *self.screen_size), 6)

    def _render_top_bar(self, scenario: Any) -> None:
        n_total = self.session.total_scenarios_in_dataset()
        pick = scenario.draft_position
        acting = "P1" if scenario.acting_player_idx == 0 else "P2"
        elapsed = int(self.session.elapsed_seconds())
        text = (
            f"Scenario #{n_total + 1}    Pick {pick}/4 (you are {acting})    "
            f"Session: {elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        surf = self.font.render(text, True, _TEXT_COLOR)
        self.screen.blit(surf, (16, 10))

    def _render_bottom_bar(self, scenario: Any) -> None:
        del scenario  # not yet used in bottom bar; kept for symmetry.
        y0 = self.screen_size[1] - 80
        # Archetype + shortcut hints.
        text1 = f"Archetype: {self.current_archetype.value}    [B/O/H/R/X to change]"
        text2 = "[click vertex → click edge → S submit]    [K skip]    [U undo]    [Q quit]"
        s1 = self.font.render(text1, True, _TEXT_COLOR)
        s2 = self.font_small.render(text2, True, _TEXT_COLOR)
        self.screen.blit(s1, (16, y0))
        self.screen.blit(s2, (16, y0 + 30))

    def _render_done_message(self) -> None:
        surf = self.font_large.render(
            "Session quit. Reopen scripts/label_setup.py to continue.",
            True,
            _TEXT_COLOR,
        )
        rect = surf.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2))
        self.screen.blit(surf, rect)

    # ------------------------------------------------------------------
    # Pixel coordinate helpers
    # ------------------------------------------------------------------

    def _vertex_pixels(self, scenario: Any) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for idx, px in scenario._idx_to_vertex_pixel.items():
            out[idx] = (float(px.x), float(px.y))
        return out

    def _edge_midpoints(self, scenario: Any) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for idx, (v1, v2) in scenario._idx_to_edge_pixel_pair.items():
            out[idx] = ((float(v1.x) + float(v2.x)) / 2.0, (float(v1.y) + float(v2.y)) / 2.0)
        return out

    def _now_ms(self) -> int:
        return self._pygame.time.get_ticks()


# Small geometric helper retained for tests + introspection.
def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
