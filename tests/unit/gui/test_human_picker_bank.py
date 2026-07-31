"""Finite-bank pins for the GUI resource picker (``gui/view.get_resource_selection``).

The picker is the ONLY resource-mutating path in the engine that never went
through the spec-009 bank rework: its YOP branch minted cards out of nothing and
its DISCARD branch leaked them permanently out of the supply. Because
``resourceBank`` is a POV-neutral feature of the bot's observation
(``policy/obs_encoder.py``), a broken bank is not a cosmetic bug — it feeds the
policy a corrupted global feature for the rest of the game, and the corruption
flows into every recorded human-vs-bot game.

These tests drive the real pygame loop headlessly: ``tests/conftest.py`` forces
``SDL_VIDEODRIVER=dummy`` suite-wide and ``get_resource_selection`` polls
``pygame.event.get()``, so synthetic ``MOUSEBUTTONDOWN`` events posted before the
call are consumed by the loop's first poll exactly as real clicks would be.

Every click coordinate is DERIVED from the same arithmetic as the picker
(``view.py`` menu geometry, keyed off ``board.width`` / ``board.height``), never
hardcoded — the window size is expected to change and these pins must survive it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from catan_rl.engine.game import catanGame
from catan_rl.gui.view import catanGameView

#: The picker's own resource order (``view.get_resource_selection``).
_MENU_RESOURCES = ("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD")


def _menu_rects(board: Any) -> dict[str, pygame.Rect]:
    """Replica of the picker's menu geometry (``view.py`` menu layout).

    Derived from ``board.width`` / ``board.height`` so a window-size change moves
    the test clicks with the menu instead of silently missing it.
    """
    menu_width, menu_height = 500, 150
    menu_x = (board.width - menu_width) // 2
    menu_y = (board.height - menu_height) // 2

    res_size, spacing = 80, 15
    start_x = menu_x + (menu_width - (5 * res_size + 4 * spacing)) // 2
    res_y = menu_y + (menu_height - res_size) // 2
    return {
        res: pygame.Rect(start_x + i * (res_size + spacing), res_y, res_size, res_size)
        for i, res in enumerate(_MENU_RESOURCES)
    }


def _outside_menu_point(board: Any) -> tuple[int, int]:
    """A click position guaranteed to be outside the picker's menu rect."""
    menu_width, menu_height = 500, 150
    menu_x = (board.width - menu_width) // 2
    menu_y = (board.height - menu_height) // 2
    assert not pygame.Rect(menu_x, menu_y, menu_width, menu_height).collidepoint((5, 5))
    return (5, 5)


def _click(pos: tuple[int, int]) -> None:
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1}))


@pytest.fixture()
def picker() -> Any:
    """A real headless game + view, with the event queue drained."""
    pygame.init()
    game = catanGame(render_mode=None)
    view = catanGameView(game.board, game)
    game.boardView = view
    view.displayGameScreen()  # populates view.colorDict, as in live play
    pygame.event.clear()
    yield game, view
    pygame.event.clear()


def _players(game: Any) -> list[Any]:
    return list(game.playerQueue.queue)


def _grant(game: Any, player: Any, counts: dict[str, int]) -> None:
    """Give ``player`` cards the honest way — drawn from the bank."""
    for res, amt in counts.items():
        player.resources[res] += amt
    game.board.bank_draw(counts)
    game.board.assert_conservation(_players(game))


class TestYearOfPlenty:
    def test_two_picks_draw_from_the_bank_and_conserve(self, picker: Any) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        rects = _menu_rects(board)
        board.assert_conservation(_players(game))
        bank_before = dict(board.resourceBank)

        _click(rects["WOOD"].center)
        _click(rects["ORE"].center)
        result = view.get_resource_selection(human, "YOP", num_to_select=2)

        assert result == ["WOOD", "ORE"]
        assert human.resources["WOOD"] == 1 and human.resources["ORE"] == 1
        assert board.resourceBank["WOOD"] == bank_before["WOOD"] - 1
        assert board.resourceBank["ORE"] == bank_before["ORE"] - 1
        board.assert_conservation(_players(game))

    def test_a_pick_the_bank_cannot_supply_is_not_granted(self, picker: Any) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        # Drain ORE into the OTHER player's hand so conservation still holds.
        other = _players(game)[1]
        _grant(game, other, {"ORE": 19})
        rects = _menu_rects(board)

        # ORE is unavailable: the click must not register as a pick, so the two
        # WOOD clicks that follow are what completes the selection.
        _click(rects["ORE"].center)
        _click(rects["WOOD"].center)
        _click(rects["WOOD"].center)
        result = view.get_resource_selection(human, "YOP", num_to_select=2)

        assert result == ["WOOD", "WOOD"]
        assert human.resources["ORE"] == 0
        assert board.resourceBank["ORE"] == 0
        board.assert_conservation(_players(game))

    def test_cancelling_a_partial_pick_leaves_the_bank_untouched(self, picker: Any) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        rects = _menu_rects(board)
        bank_before = dict(board.resourceBank)

        _click(rects["WHEAT"].center)
        _click(_outside_menu_point(board))
        result = view.get_resource_selection(human, "YOP", num_to_select=2)

        assert result is None
        assert human.resources["WHEAT"] == 0
        assert board.resourceBank == bank_before
        board.assert_conservation(_players(game))


class TestDiscard:
    def test_discarded_cards_go_back_into_the_bank(self, picker: Any) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        _grant(game, human, {"SHEEP": 2, "BRICK": 1})
        rects = _menu_rects(board)
        bank_before = dict(board.resourceBank)

        _click(rects["SHEEP"].center)
        _click(rects["BRICK"].center)
        result = view.get_resource_selection(human, "DISCARD", num_to_select=2)

        assert result == ["SHEEP", "BRICK"]
        assert human.resources["SHEEP"] == 1 and human.resources["BRICK"] == 0
        assert board.resourceBank["SHEEP"] == bank_before["SHEEP"] + 1
        assert board.resourceBank["BRICK"] == bank_before["BRICK"] + 1
        board.assert_conservation(_players(game))


class TestMonopolyThroughPlayDevCard:
    """The human seat must reach the PICKER, not ``np.random.choice``."""

    def test_the_human_picks_and_the_ai_branch_is_never_taken(
        self, picker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        game, view = picker
        board = game.board
        human, bot = _players(game)
        human.isAI = False
        human.devCards["MONOPOLY"] = 1
        _grant(game, bot, {"WHEAT": 3})

        # The dev-card menu is not what is under test; the resource picker is.
        monkeypatch.setattr(view, "get_dev_card_selection", lambda _p: "MONOPOLY")

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("AI branch taken: np.random.choice must not run")

        monkeypatch.setattr("catan_rl.engine.player.np.random.choice", _boom)

        rects = _menu_rects(board)
        _click(rects["WHEAT"].center)  # select
        _click(rects["WHEAT"].center)  # confirm
        human.play_devCard(game)

        assert human.resources["WHEAT"] == 3
        assert bot.resources["WHEAT"] == 0
        assert human.devCards["MONOPOLY"] == 0
        board.assert_conservation(_players(game))

    def test_the_human_year_of_plenty_goes_through_the_picker(
        self, picker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        human.isAI = False
        human.devCards["YEAROFPLENTY"] = 1
        monkeypatch.setattr(view, "get_dev_card_selection", lambda _p: "YEAROFPLENTY")

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("AI branch taken: np.random.choice must not run")

        monkeypatch.setattr("catan_rl.engine.player.np.random.choice", _boom)

        rects = _menu_rects(board)
        _click(rects["BRICK"].center)
        _click(rects["BRICK"].center)
        human.play_devCard(game)

        assert human.resources["BRICK"] == 2
        assert board.resourceBank["BRICK"] == 17
        board.assert_conservation(_players(game))

    def test_cancelling_year_of_plenty_refunds_the_card(
        self, picker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        game, view = picker
        board = game.board
        human = _players(game)[0]
        human.isAI = False
        human.devCards["YEAROFPLENTY"] = 1
        monkeypatch.setattr(view, "get_dev_card_selection", lambda _p: "YEAROFPLENTY")
        bank_before = dict(board.resourceBank)

        rects = _menu_rects(board)
        _click(rects["SHEEP"].center)
        _click(_outside_menu_point(board))
        human.play_devCard(game)

        assert human.devCards["YEAROFPLENTY"] == 1
        assert human.devCardPlayedThisTurn is False
        assert board.resourceBank == bank_before
        board.assert_conservation(_players(game))
