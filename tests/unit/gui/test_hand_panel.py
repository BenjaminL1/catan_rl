"""Leak pins for the opponent hand panel (``gui/view.hand_panel_lines``) and
for the on-screen broadcast banner (``gui/view.broadcast_message``).

Every human-vs-policy game played while the bot's FULL hand was rendered on
screen is uninterpretable as a strength read, so the blind default is pinned
here rather than left to the harness.

``hand_panel_lines`` and ``broadcast_message`` are deliberately pygame-free and
duck-typed, so most of these tests need neither a display nor a real engine
``player``. ``TestPanelRendering`` is the exception: it boots a headless dummy
surface to exercise the box-sizing path in ``_draw_hand_panel``, which the pure
line tests cannot reach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from catan_rl.gui import view as view_module
from catan_rl.gui.view import (
    HAND_PANEL_LINE_HEIGHT,
    MOVE_LOG_LINE_HEIGHT,
    MOVE_LOG_LINES,
    MOVE_LOG_RECT,
    broadcast_message,
    catanGameView,
    hand_panel_lines,
)

_RESOURCE_NAMES = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")
_DEV_TYPE_LABELS = ("Knight", "VP", "Mono", "RB", "YOP")


@dataclass
class _FakePlayer:
    """The attributes ``hand_panel_lines`` reads."""

    resources: dict[str, int] = field(
        default_factory=lambda: {"WOOD": 2, "BRICK": 1, "WHEAT": 0, "ORE": 3, "SHEEP": 1}
    )
    # 9 total VP of which 2 come from hidden VP cards -> 7 visible.
    victoryPoints: int = 9
    devCards: dict[str, int] = field(
        default_factory=lambda: {
            "KNIGHT": 1,
            "VP": 2,
            "MONOPOLY": 0,
            "ROADBUILDER": 0,
            "YEAROFPLENTY": 1,
        }
    )
    newDevCards: list[str] = field(default_factory=lambda: ["KNIGHT"])
    # PUBLIC progress facts: knights are played face up and roads sit on the
    # board, so both panels carry them.
    knightsPlayed: int = 2
    #: The cache. Read only when no board is supplied.
    maxRoadLength: int = 4

    def get_road_length(self, board) -> int:  # type: ignore[no-untyped-def]
        """Live recompute; the stub board carries the answer."""
        return int(board["road_length"])


class TestBlindPanel:
    def test_no_resource_type_is_reachable(self) -> None:
        blob = "\n".join(hand_panel_lines(_FakePlayer(), reveal=False))
        for name in _RESOURCE_NAMES:
            assert name not in blob, f"blind panel leaks resource type {name}"

    def test_no_dev_card_type_is_reachable(self) -> None:
        blob = "\n".join(hand_panel_lines(_FakePlayer(), reveal=False))
        for label in _DEV_TYPE_LABELS:
            assert f"{label}:" not in blob, f"blind panel leaks dev-card type {label}"

    def test_shows_hand_size_and_unplayed_dev_count(self) -> None:
        lines = hand_panel_lines(_FakePlayer(), reveal=False)
        assert "Cards: 7" in lines  # 2+1+0+3+1
        assert "Dev Cards: 5" in lines  # sum(devCards)=4 + 1 newDevCard

    def test_vp_shown_is_visible_vp_not_the_total(self) -> None:
        # 9 total - 2 VP cards = 7 visible. The VP-inclusive total must NOT
        # appear anywhere in the blind panel.
        lines = hand_panel_lines(_FakePlayer(), reveal=False)
        assert "Victory Points: 7" in lines
        assert "Victory Points: 9" not in lines

    def test_newdevcards_do_not_reduce_visible_vp(self) -> None:
        # A VP card scores immediately and never lands in ``newDevCards``
        # (engine/player.py), so a pending KNIGHT in that bucket must not be
        # subtracted. Regression guard against a ``- len(newDevCards)`` term.
        player = _FakePlayer()
        player.newDevCards = ["KNIGHT", "MONOPOLY"]
        assert "Victory Points: 7" in hand_panel_lines(player, reveal=False)

    def test_visible_vp_ignores_the_stale_cache(self) -> None:
        # ``player.visibleVictoryPoints`` is refreshed only at init + VP-card
        # buy, so it goes stale. A deliberately wrong cache must be ignored.
        player = _FakePlayer()
        player.visibleVictoryPoints = 999  # type: ignore[attr-defined]
        assert "Victory Points: 7" in hand_panel_lines(player, reveal=False)


class TestRevealedPanel:
    def test_reveal_restores_the_full_lines(self) -> None:
        lines = hand_panel_lines(_FakePlayer(), reveal=True)
        assert "WOOD: 2" in lines
        assert "ORE: 3" in lines
        assert "Victory Points: 9" in lines  # VP-card-inclusive total
        assert "Knight: 2" in lines  # 1 held + 1 just-bought
        assert "VP: 2" in lines

    def test_reveal_is_the_default(self) -> None:
        # ``_draw_hand_panel`` relies on this default for a player's OWN hand.
        assert hand_panel_lines(_FakePlayer()) == hand_panel_lines(_FakePlayer(), reveal=True)


class TestPublicProgressFacts:
    """Knights played and longest-road length are PUBLIC — both panels show them.

    Withholding them makes the playtest HARDER than real Catan (the bot reads
    both straight off the state) and biases every recorded result toward the
    bot, the same error as the reverted DISCARD/YOP over-blinding."""

    @pytest.mark.parametrize("reveal", [True, False])
    def test_both_facts_appear_in_both_reveal_modes(self, reveal: bool) -> None:
        lines = hand_panel_lines(_FakePlayer(), reveal=reveal)
        assert "Knights played: 2" in lines
        assert "Longest road: 4" in lines

    def test_the_knights_line_is_not_a_dev_card_type_leak(self) -> None:
        # The counterpart to ``test_no_dev_card_type_is_reachable``: that pin
        # forbids the ``Knight:`` label, and this one requires the count to be
        # present anyway. The wording clears the allow-list on purpose — the pin
        # is preserved, not weakened.
        blob = "\n".join(hand_panel_lines(_FakePlayer(), reveal=False))
        assert "Knights played: 2" in blob
        assert "Knight:" not in blob

    def test_a_board_makes_the_road_length_live(self) -> None:
        # FRESHNESS: with a board, the cache is ignored entirely.
        player = _FakePlayer()
        player.maxRoadLength = 99  # deliberately stale
        lines = hand_panel_lines(player, reveal=False, board={"road_length": 3})
        assert "Longest road: 3" in lines
        assert "Longest road: 99" not in lines

    def test_an_opponent_settlement_shortens_the_displayed_road(self) -> None:
        """The real engine path: a road broken mid-chain reads shorter at once."""
        from catan_rl.engine.board import catanBoard
        from catan_rl.engine.player import player as EnginePlayer

        board = catanBoard()
        mine = EnginePlayer("Mine", "red")
        theirs = EnginePlayer("Theirs", "blue")
        # Walk a 4-segment chain out from a start vertex.
        chain = []
        v = next(iter(board.boardGraph))
        seen = {v}
        while len(chain) < 4:
            nxt = next((n for n in board.boardGraph[v].neighbors if n not in seen), None)
            if nxt is None:
                break
            chain.append((v, nxt))
            seen.add(nxt)
            v = nxt
        assert len(chain) == 4
        for v1, v2 in chain:
            mine.build_road(v1, v2, board, is_free=True)
        before = hand_panel_lines(mine, reveal=False, board=board)
        assert f"Longest road: {mine.get_road_length(board)}" in before
        assert "Longest road: 4" in before

        # An OPPONENT settlement at the chain's middle vertex breaks the trail.
        board.updateBoardGraph_settlement(chain[1][1], theirs)
        after = hand_panel_lines(mine, reveal=False, board=board)
        assert "Longest road: 4" not in after
        assert f"Longest road: {mine.get_road_length(board)}" in after


class TestBroadcastBanner:
    """The banner carries resource types — and for DISCARD/YOP that is CORRECT.

    These events are PUBLIC. The engine says so itself (``tracker.track_steal``:
    "It is Public Information relative to the two players"), and the bot reads
    this same broadcast stream through a ``BroadcastHandTracker`` that does
    perfect opponent hand-tracking (``env/catan_env.py``). Blinding the human to
    the bot's discards while the bot tracks the human's would make the playtest
    HARDER than the real game and bias every result in the bot's favour.

    What the blind gate closes is the bot's HAND CONTENTS (``hand_panel_lines``),
    which are private. Public events stay public."""

    def test_bot_discard_types_stay_public_even_when_blinded(self) -> None:
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Agent", "resources": ["ORE", "WHEAT"]},
            blind_player="Agent",
        )
        assert "ORE" in text and "WHEAT" in text

    def test_bot_yop_types_stay_public_even_when_blinded(self) -> None:
        """Year-of-Plenty picks are taken openly from the bank — public, like discards."""
        text, _color = broadcast_message(
            {"type": "YOP", "player": "Agent", "resources": ["ORE", "ORE"]},
            blind_player="Agent",
        )
        assert "ORE" in text

    def test_the_humans_own_discard_is_unchanged(self) -> None:
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Opponent", "resources": ["ORE"]},
            blind_player="Agent",
        )
        assert "ORE" in text

    def test_no_blind_player_keeps_the_engine_message(self) -> None:
        # engine playCatan sets no bot_player -> nothing is hidden from anyone.
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Agent", "resources": ["ORE"]}
        )
        assert "ORE" in text

    def test_display_names_and_dice_still_work(self) -> None:
        text, _color = broadcast_message(
            {"type": "DICE_ROLL", "player": "Agent", "value": 8},
            name_display={"Agent": "Bot"},
            blind_player="Agent",
        )
        assert text == "Dice: Bot rolled 8"

    def test_unknown_event_renders_nothing(self) -> None:
        assert broadcast_message({"type": "BUILD", "player": "Agent"}) is None


class TestPanelRendering:
    """The box-sizing path (``panel_lines = len(lines) + 2``) is the only part
    of the blind panel the pure line tests cannot reach: the blind panel has 5
    lines where the revealed one has 12, so the backdrop must shrink with it."""

    def _view(self, size=(320, 240)):  # type: ignore[no-untyped-def]
        import pygame

        pygame.init()
        pygame.display.set_mode(size)
        view = catanGameView.__new__(catanGameView)
        view.screen = pygame.display.get_surface()
        view.font_resource = pygame.font.SysFont("cambria", 15)
        view.font_movelog = pygame.font.SysFont("cambria", 13)
        # No board -> the road length falls back to the cache (the pure path).
        view.board = None
        view.move_log = None
        return view

    def test_blind_and_revealed_panels_both_render(self) -> None:
        view = self._view()
        player = _FakePlayer()
        # No assertion on pixels (font rendering is platform-dependent); this
        # pins that the sizing arithmetic and the blit calls actually run.
        view._draw_hand_panel(player, 20, 20, "Bot — HAND", reveal=False)
        view._draw_hand_panel(player, 20, 140, "YOUR HAND", reveal=True)

    def test_the_blind_backdrop_is_shorter_than_the_revealed_one(self) -> None:
        blind = len(hand_panel_lines(_FakePlayer(), reveal=False)) + 2
        revealed = len(hand_panel_lines(_FakePlayer(), reveal=True)) + 2
        assert blind < revealed


class TestPanelLayout:
    """Arithmetic, not eyeballing: two extra lines per panel must not push the
    human panel into the BANK TRADE button, nor the REVEALED bot panel off the
    bottom of the 1000x800 window. Coordinates mirror ``displayPlayerStats``."""

    _WINDOW_H = 800
    _BANK_TRADE_Y = 400
    _HUMAN_PANEL_Y = 15
    _BOT_PANEL_Y = 460

    def _panel_bottom(self, y: int, *, reveal: bool) -> int:
        n_lines = len(hand_panel_lines(_FakePlayer(), reveal=reveal)) + 2
        return (y - 8) + HAND_PANEL_LINE_HEIGHT * n_lines + 12

    def test_the_revealed_human_panel_clears_the_bank_trade_button(self) -> None:
        assert self._panel_bottom(self._HUMAN_PANEL_Y, reveal=True) < self._BANK_TRADE_Y

    def test_the_revealed_bot_panel_stays_inside_the_window(self) -> None:
        # --reveal-bot draws the FULL 16-line panel at y=460; at the old 20px
        # leading it ran to y=824 and was clipped.
        assert self._panel_bottom(self._BOT_PANEL_Y, reveal=True) <= self._WINDOW_H

    def test_the_blind_bot_panel_stays_inside_the_window(self) -> None:
        assert self._panel_bottom(self._BOT_PANEL_Y, reveal=False) <= self._WINDOW_H

    def test_the_move_log_strip_fits_and_clears_the_end_turn_button(self) -> None:
        x, y, w, h = MOVE_LOG_RECT
        assert x >= 115, "must clear the END TURN button (x 20-100)"
        assert x + w <= 1000 and y + h <= self._WINDOW_H
        assert h >= MOVE_LOG_LINES * MOVE_LOG_LINE_HEIGHT


class TestMoveLogSurface:
    """D5: the log container is a bounded deque owned by the harness, and the
    GUI module never touches the recorder's destructive ``EventCollector``."""

    def test_the_gui_never_imports_the_replay_package(self) -> None:
        # AST, not a text grep: the prose explaining WHY names both symbols.
        import ast

        tree = ast.parse(Path(view_module.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported += [f"{node.module or ''}.{a.name}" for a in node.names]
        assert not [m for m in imported if "replay" in m], imported
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "EventCollector" not in names | attrs

    def test_no_log_draws_nothing(self) -> None:
        view = TestPanelRendering()._view((1000, 800))
        view.move_log = None
        view.displayMoveLog()  # must be a no-op, not an AttributeError
        view.move_log = []
        view.displayMoveLog()

    def test_the_strip_really_does_overlap_board_vertices(self) -> None:
        # The premise of the draw-order pin below, measured against the real
        # board rather than asserted: three of the 54 vertices sit at y=720,
        # inside MOVE_LOG_RECT.
        from catan_rl.engine.board import catanBoard

        x, y, w, h = MOVE_LOG_RECT
        board = catanBoard()
        inside = [v for v in board.boardGraph if x <= v.x <= x + w and y <= v.y <= y + h]
        assert len(inside) >= 3, inside

    def test_the_strip_is_drawn_before_the_buildings_loop(self) -> None:
        # Order matters: the strip covers three bottom-row vertices (above), so
        # painting it LAST hides settlements/cities/roads placed there — a new
        # withholding of a PUBLIC fact, which is the failure this feature exists
        # to fix. Pinned on the AST of ``displayGameScreen`` because a pixel
        # test would need a full engine game plus a real display.
        import ast

        tree = ast.parse(Path(view_module.__file__).read_text(encoding="utf-8"))
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "displayGameScreen"
        )

        def index_of(attr: str) -> int:
            return next(
                i
                for i, stmt in enumerate(fn.body)
                if any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == attr
                    for c in ast.walk(stmt)
                )
            )

        assert index_of("displayMoveLog") < index_of("draw_settlement")
        assert index_of("displayMoveLog") < index_of("displayPorts")

    def test_the_strip_really_does_overlap_two_ports(self) -> None:
        # Same premise-first measurement as the vertex case above: two of the
        # nine port ANCHORS fall inside MOVE_LOG_RECT, so the strip's backdrop
        # would erase them for the whole game.
        import math

        from catan_rl.engine.board import catanBoard
        from catan_rl.gui import render
        from catan_rl.gui import render_constants as RC

        x, y, w, h = MOVE_LOG_RECT
        board = catanBoard()
        vertex_pixel = board.vertex_index_to_pixel_dict
        bcx, bcy = board.width / 2.0, board.height / 2.0
        inside = []
        for v1_idx, v2_idx, _ratio, _res in render.collect_port_edges(board):
            p1, p2 = vertex_pixel[v1_idx], vertex_pixel[v2_idx]
            mx = (float(p1.x) + float(p2.x)) / 2.0
            my = (float(p1.y) + float(p2.y)) / 2.0
            dx, dy = mx - bcx, my - bcy
            d = math.hypot(dx, dy) or 1.0
            ax = int(mx + dx * RC.PORT_PUSH_DISTANCE / d)
            ay = int(my + dy * RC.PORT_PUSH_DISTANCE / d)
            if x <= ax <= x + w and y <= ay <= y + h:
                inside.append((ax, ay))
        assert len(inside) == 2, inside

    def test_the_ports_are_repainted_after_the_strip(self) -> None:
        # PIXEL PIN. Port access is a first-order PUBLIC planning fact — hiding
        # two of nine for the whole game biases the playtest toward the bot, the
        # same failure the buildings repaint guards against. Fill the surface
        # with a sentinel, draw the strip (which erases both anchors), repaint
        # the ports, and assert the anchors carry port pixels again.
        from collections import deque

        from catan_rl.engine.board import catanBoard

        view = TestPanelRendering()._view((1000, 800))
        view.board = catanBoard()
        view.move_log = deque(["Bot: Roll dice", "Bot: Play Knight"], maxlen=MOVE_LOG_LINES)

        sentinel = (255, 0, 0)
        anchors = [(541, 759), (296, 751)]
        view.screen.fill(sentinel)
        view.displayMoveLog()
        # Premise: the backdrop really did paint over both anchors.
        for px in anchors:
            assert view.screen.get_at(px)[:3] != sentinel, px
        blotted = [view.screen.get_at(px)[:3] for px in anchors]

        view.displayPorts()
        for px, before in zip(anchors, blotted, strict=True):
            got = view.screen.get_at(px)[:3]
            assert got != before, (px, got)
            assert got != sentinel, px

    def test_only_the_last_lines_are_drawn(self) -> None:
        from collections import deque

        view = TestPanelRendering()._view((1000, 800))
        log: deque[str] = deque(maxlen=MOVE_LOG_LINES)
        for i in range(MOVE_LOG_LINES * 3):
            log.append(f"line {i}")
        view.move_log = log
        assert len(log) == MOVE_LOG_LINES
        view.displayMoveLog()  # renders without overflowing the strip
