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
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import (
    PICK_CLARITY_CLEAR,
    PICK_CLARITY_CLOSE,
    REVEAL_MODE_REVEAL,
)
from catan_rl.setup_phase.scorer import (
    SetupScorer,
    probabilities,
    rank_of,
    top_k,
)

PHASE_SETTLEMENT_PICK = "settlement_pick"
PHASE_ROAD_PICK = "road_pick"
PHASE_REVEAL = "reveal"
"""Post-submit overlay showing the scorer's opinion beside the owner's pick.

Reached ONLY from :meth:`LabelingUIState.submit`, and only after
``session.submit`` has returned — i.e. after the row is durably on disk. There
is no other assignment to ``LabelingUIState.reveal`` anywhere, which is what
makes "the reveal never precedes the submit" a property of the code rather than
a discipline (spec ``setup-scorer-and-blind-reveal`` D3)."""


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
    #: The fitted scorer whose opinion is revealed after a submit. ``None``
    #: disables the overlay entirely (and writes no scorer fields).
    scorer: SetupScorer | None = None
    #: The reveal payload for the just-submitted pick, or ``None``. Assigned in
    #: exactly one place: after ``session.submit`` returns.
    reveal: dict[str, Any] | None = None
    #: The scenario the reveal payload GRADES — captured before the submit, so
    #: the overlay is painted on the board that was labeled. ``session.submit``
    #: advances the snake draft, so ``session.current_scenario()`` is already
    #: the NEXT decision point by the time the overlay is drawn; rendering
    #: against it would put the scorer's rings on the board the owner is about
    #: to label, which is not a display bug but an anchoring leak of exactly the
    #: kind D3's control exists to detect.
    reveal_scenario: Any | None = None
    # Cache of the legal-roads mask after a settlement is selected.
    _cached_legal_roads: np.ndarray | None = field(default=None)

    def reset_for_new_scenario(self) -> None:
        self.phase = PHASE_SETTLEMENT_PICK
        self.selected_settlement = None
        self.selected_road = None
        self.last_click_rejected = False
        self.reveal = None
        self.reveal_scenario = None
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

        Cannot undo a submitted scenario (use skip on the next one), and is a
        NO-OP once the reveal is up: the row is already on disk, so an "undo"
        there could only mean "re-pick after seeing the answer" — which is the
        exact contamination the blind-first design exists to prevent.
        """
        if self.phase == PHASE_REVEAL:
            return
        if self.selected_road is not None:
            self.selected_road = None
        elif self.selected_settlement is not None:
            self.selected_settlement = None
            self._cached_legal_roads = None
            self.phase = PHASE_SETTLEMENT_PICK
        # else: nothing to undo.

    def submit(
        self,
        notes: str = "",
        decision_time_ms: int = 0,
        pick_clarity: str = PICK_CLARITY_CLOSE,
    ) -> None:
        """Persist the pick, THEN (and only then) show the scorer's opinion.

        The payload is computed into a LOCAL before the write — it has to be, it
        goes into the row — but it is not visible anywhere until
        ``session.submit`` has returned. If the append raises, ``self.reveal``
        stays ``None`` and the phase is unchanged, so a failed write cannot leak
        the answer for a pick that was never recorded.

        ``pick_clarity`` is the owner's own D3 tag ("clear best" vs "close
        call"), bound to two submit keys in :class:`LabelingUI`. It is written
        in BOTH arms — it is the owner's statement about the position, not the
        scorer's about the pick.
        """
        if not self.is_ready_to_submit():
            raise RuntimeError("not ready to submit (settlement + road both required)")
        assert self.selected_settlement is not None
        assert self.selected_road is not None
        scenario = self.session.current_scenario()
        payload = self._scorer_payload(scenario, self.selected_settlement, self.selected_road)
        row_fields = None if payload is None else payload["row_fields"]
        self.session.submit(
            settlement_vertex=self.selected_settlement,
            road_edge=self.selected_road,
            notes=notes,
            decision_time_ms=decision_time_ms,
            scorer_fields=row_fields,
            pick_clarity=pick_clarity,
        )
        if payload is None or self.session.reveal_mode != REVEAL_MODE_REVEAL:
            self.reset_for_new_scenario()
            return
        self.reveal = payload["display"]
        # Pin the board the reveal describes. ``session.submit`` has already
        # advanced the draft, so this is the ONLY handle on the graded position.
        self.reveal_scenario = scenario
        self.phase = PHASE_REVEAL
        self.selected_settlement = None
        self.selected_road = None
        self._cached_legal_roads = None

    def dismiss_reveal(self) -> None:
        """Close the overlay and move on to the next decision point."""
        if self.phase != PHASE_REVEAL:
            return
        self.reset_for_new_scenario()

    def _scorer_payload(self, scenario: Any, vertex: int, edge: int) -> dict[str, Any] | None:
        """Grade the just-made pick.

        Returns ``None`` — no overlay, and no scorer fields on the row — when no
        scorer is loaded, and also on a REPLAYED board (D0). The second case is
        why the exclusive ``--replay-session`` mode refuses ``--scorer-weights``
        outright: a reveal mid-replay anchors the owner on the scorer during the
        one owner-vs-owner measurement every later bar is read against. Folding
        replay boards into a scorer session (``--replay-boards``) must not
        smuggle that back in, so the suppression is per BOARD rather than per
        session, and it is the session that says which board is which.
        """
        if self.scorer is None or scenario is None:
            return None
        if self.session.current_is_replay:
            return None
        board = scenario.game.board
        v_scores = self.scorer.score_vertices(
            board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
        )
        v_top = top_k(v_scores, 3)
        legal_roads = scenario.compute_legal_road_edges(vertex)
        e_scores = self.scorer.score_edges(
            board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
            vertex,
            legal_roads,
        )
        e_top = top_k(e_scores, 3)
        v_probs = probabilities(v_scores)
        e_probs = probabilities(e_scores)
        row_fields = {
            "scorer_version": self.scorer.version,
            "scorer_top1": int(v_top[0]),
            "scorer_rank_of_pick": rank_of(v_scores, vertex),
            "agree": bool(v_top[0] == vertex),
            "reveal_mode": self.session.reveal_mode,
        }
        # D3 (amended): the reveal shows the scorer's PROBABILITIES, not bare
        # picks. A bare top-1 reads as an assertion; a distribution that says
        # "0.31 / 0.28 / 0.25" reads as the near-tie D0 measured, which is the
        # thing the owner needs to see to judge whether the scorer is wrong or
        # merely undecided.
        return {
            "row_fields": row_fields,
            "display": {
                **row_fields,
                "settlement_top3": [int(v) for v in v_top],
                "settlement_top3_probs": [float(v_probs[int(v)]) for v in v_top],
                "owner_settlement_prob": float(v_probs[int(vertex)]),
                "road_top1": int(e_top[0]) if e_top else None,
                "road_top3": [int(e) for e in e_top],
                "road_top3_probs": [float(e_probs[int(e)]) for e in e_top],
                "owner_road_prob": float(e_probs[int(edge)]),
                "owner_settlement": int(vertex),
                "owner_road": int(edge),
            },
        }

    def skip(self) -> None:
        """Abandon the current draft, advance to a fresh board.

        Never sets ``reveal``: a skipped position was never submitted, so there
        is nothing to grade and nothing the owner is owed an answer to.
        """
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

_REVEAL_TOP1_COLOR = (255, 140, 0)
"""Orange ring / line for the scorer's top-1 in the post-submit reveal."""

_REVEAL_TOPK_COLOR = (150, 90, 30)
"""Dimmed ring for the scorer's ranks 2-3."""

_REVEAL_OWNER_COLOR = (60, 200, 255)
"""Wide ring around the owner's own pick, so the reveal is a COMPARISON."""


class LabelingUI:
    """Pygame UI driver. Call :meth:`run` to enter the event loop."""

    def __init__(
        self,
        session: LabelingSession,
        screen_size: tuple[int, int] = (1100, 900),
        click_radius: float = 22.0,
        scorer: SetupScorer | None = None,
    ) -> None:
        import pygame  # Local import: pygame is an optional dependency.

        pygame.init()
        pygame.display.set_caption("Catan Setup Labeling")
        self._pygame = pygame
        self.screen = pygame.display.set_mode(screen_size)
        self.screen_size = screen_size
        self.click_radius = click_radius
        self.session = session
        self.state = LabelingUIState(session, scorer=scorer)
        # Fonts.
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)
        self.font_large = pygame.font.SysFont(None, 32)
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
        if self.state.phase == PHASE_REVEAL:
            # Clicks never dismiss the reveal — only SPACE does (see
            # _handle_key). A screenshot drag or focus click must not advance.
            return
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
        if self.state.phase == PHASE_REVEAL:
            # SPACE alone dismisses (owner request 2026-08-22): a deliberate key
            # means an accidental keystroke — or the modifier chords of a
            # screenshot — can never blow past a reveal the owner wants to
            # capture. Undo stays deliberately inert here.
            if key == self._pygame.K_SPACE:
                self.state.dismiss_reveal()
                self._scenario_start_ms = self._now_ms()
            return True
        if unicode == "s":
            self._try_submit(PICK_CLARITY_CLOSE)
        elif unicode == "b":
            self._try_submit(PICK_CLARITY_CLEAR)
        elif unicode == "k":
            self._skip()
        elif unicode == "u":
            self.state.undo()
        return True

    def _try_submit(self, pick_clarity: str = PICK_CLARITY_CLOSE) -> None:
        """Submit with the owner's clarity tag (D3's two submit keys).

        ``S`` — the pre-existing submit key — means "close call"; ``B`` means
        "clear BEST". Two keys rather than a prompt because the tag has to cost
        nothing: a modal question after every pick would be paid 150+ times and
        would be answered carelessly, which is worse than not asking.

        **``S`` deliberately carries the CONSERVATIVE tag.** ``close`` is what
        the store reads for an untagged row (``store._V3_DEFAULTS``) and what
        every default in the stack spells, because only ``clear`` picks are held
        to D4's >=70% top-1 bar. ``S`` has meant plain "submit" for the whole
        292-label corpus, so binding it to ``clear`` would let reflex populate
        the strict subset with close calls — contaminating the one bar the spec
        says is "revisable only upward". Asserting "clear best" is worth one
        deliberate, unfamiliar keystroke; not asserting it must stay free.
        """
        if not self.state.is_ready_to_submit():
            return
        elapsed_ms = self._now_ms() - self._scenario_start_ms
        self.state.submit(decision_time_ms=elapsed_ms, pick_clarity=pick_clarity)
        self._scenario_start_ms = self._now_ms()

    def _skip(self) -> None:
        self.state.skip()
        self._scenario_start_ms = self._now_ms()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self.state.phase == PHASE_REVEAL and self.state.reveal is not None:
            # Paint the GRADED board, not the next one. ``session.submit`` has
            # already advanced the draft, so ``current_scenario()`` here is the
            # position the owner has not seen yet.
            graded = self.state.reveal_scenario
            if graded is not None:
                self._render_board(graded)
                self._render_top_bar(graded)
                self._render_bottom_bar(graded)
                self._render_reveal(graded, self.state.reveal)
                return
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
        # Shortcut hints.
        text = (
            "[click vertex → click edge → S submit (CLOSE call) / B submit CLEAR best]"
            "    [K skip]    [U undo]    [Q quit]"
        )
        surf = self.font_small.render(text, True, _TEXT_COLOR)
        self.screen.blit(surf, (16, y0))

    def _render_reveal(self, scenario: Any, reveal: dict[str, Any]) -> None:
        """Draw the post-submit overlay: the scorer's picks beside the owner's.

        Drawn from ``state.reveal`` only, which is populated exactly once, after
        the durable write, and against ``state.reveal_scenario`` — the position
        that was GRADED, not whatever the session advanced to. Nothing here
        reads the scorer directly."""
        pygame = self._pygame
        vertex_pixels = self._vertex_pixels(scenario)
        top3 = reveal.get("settlement_top3") or []
        for rank, vidx in enumerate(top3):
            if vidx not in vertex_pixels:
                continue
            vx, vy = vertex_pixels[vidx]
            colour = _REVEAL_TOP1_COLOR if rank == 0 else _REVEAL_TOPK_COLOR
            pygame.draw.circle(self.screen, colour, (int(vx), int(vy)), 13, 3)
        road_top1 = reveal.get("road_top1")
        if road_top1 is not None:
            v1, v2 = scenario._idx_to_edge_pixel_pair[int(road_top1)]
            pygame.draw.line(
                self.screen,
                _REVEAL_TOP1_COLOR,
                (int(v1.x), int(v1.y)),
                (int(v2.x), int(v2.y)),
                4,
            )
        owner_vertex = reveal.get("owner_settlement")
        if owner_vertex is not None and int(owner_vertex) in vertex_pixels:
            ox, oy = vertex_pixels[int(owner_vertex)]
            pygame.draw.circle(self.screen, _REVEAL_OWNER_COLOR, (int(ox), int(oy)), 17, 3)
        verdict = "AGREE" if reveal.get("agree") else "DIFFER"
        probs = reveal.get("settlement_top3_probs") or []
        ranked = ", ".join(f"{v} {p:.0%}" for v, p in zip(top3, probs, strict=False))
        own_prob = reveal.get("owner_settlement_prob")
        own_text = "—" if own_prob is None else f"{float(own_prob):.0%}"
        lines = [
            f"scorer {reveal.get('scorer_version')} — {verdict}   [SPACE to continue]",
            f"settlement p: {ranked}",
            (f"your pick {owner_vertex}: p={own_text}  (rank {reveal.get('scorer_rank_of_pick')})"),
        ]
        road_probs = reveal.get("road_top3_probs") or []
        if road_probs:
            own_road_prob = reveal.get("owner_road_prob")
            own_road_text = "—" if own_road_prob is None else f"{float(own_road_prob):.0%}"
            lines.append(
                f"road p: {reveal.get('road_top1')} {road_probs[0]:.0%}  "
                f"your road {reveal.get('owner_road')}: p={own_road_text}"
            )
        surfaces = [self.font.render(line, True, _TEXT_COLOR) for line in lines]
        pad = 8
        width = max(surf.get_width() for surf in surfaces)
        height = sum(surf.get_height() for surf in surfaces)
        pygame.draw.rect(
            self.screen,
            (20, 20, 20),
            (16 - pad, 40 - pad, width + 2 * pad, height + 2 * pad),
        )
        y = 40
        for surf in surfaces:
            self.screen.blit(surf, (16, y))
            y += surf.get_height()

    def _done_message(self) -> str:
        """The end-of-session banner.

        ``current_scenario()`` returns ``None`` for two very different reasons,
        and a replay session hits the one the banner used to name wrongly: an
        EXHAUSTED replay plan is the measurement finishing, not the owner
        quitting, and telling them to "reopen to continue" invites a second
        sitting that would re-label boards already re-labeled.
        """
        if self.session.exhausted:
            n = self.session.replay_boards_presented
            return f"Replay complete — {n} board{'' if n == 1 else 's'} re-presented. Q to close."
        return "Session quit. Reopen scripts/label_setup.py to continue."

    def _render_done_message(self) -> None:
        surf = self.font_large.render(
            self._done_message(),
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
