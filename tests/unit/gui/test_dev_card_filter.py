"""Pins for the dev-card picker's optional pre-roll restriction.

``gui/view.get_dev_card_selection`` is reached from ``engine/player.play_devCard``
with a SINGLE argument, so the pre-roll window in ``scripts/play_vs_model.py``
cannot narrow the human's choices by passing a whitelist down — the restriction
has to live on the view. ``catanGameView.dev_card_filter`` is that seam: ``None``
(the default) is byte-identical to the historic behaviour, and a set of card
names makes everything outside it dim and unclickable.

Restricting the MENU rather than stashing ``player.devCards["ROADBUILDER"] = 0``
is deliberate: the human's hand is never mutated, so a ``sys.exit(0)`` from the
window's QUIT path or a partial-record write can neither capture nor leak a card,
and the on-screen count stays truthful.

Like ``test_human_picker_bank.py`` these drive the real pygame loop headlessly
(``tests/conftest.py`` forces ``SDL_VIDEODRIVER=dummy``) and derive every click
coordinate from the picker's own menu arithmetic.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from catan_rl.engine.game import catanGame
from catan_rl.gui.view import catanGameView

#: The picker's own card order (``view.get_dev_card_selection``).
_MENU_CARDS = ("KNIGHT", "VP", "MONOPOLY", "ROADBUILDER", "YEAROFPLENTY")


def _card_rects(board: Any) -> dict[str, pygame.Rect]:
    """Replica of the dev-card menu geometry (``view.get_dev_card_selection``)."""
    menu_width, menu_height = 500, 150
    menu_x = (board.width - menu_width) // 2
    menu_y = (board.height - menu_height) // 2

    card_size, spacing = 80, 15
    start_x = menu_x + (menu_width - (5 * card_size + 4 * spacing)) // 2
    card_y = menu_y + (menu_height - card_size) // 2
    return {
        card: pygame.Rect(start_x + i * (card_size + spacing), card_y, card_size, card_size)
        for i, card in enumerate(_MENU_CARDS)
    }


def _outside_menu_point(board: Any) -> tuple[int, int]:
    menu_width, menu_height = 500, 150
    menu_x = (board.width - menu_width) // 2
    menu_y = (board.height - menu_height) // 2
    assert not pygame.Rect(menu_x, menu_y, menu_width, menu_height).collidepoint((5, 5))
    return (5, 5)


def _click(pos: tuple[int, int]) -> None:
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1}))


@pytest.fixture()
def picker() -> Any:
    pygame.init()
    game = catanGame(render_mode=None)
    view = catanGameView(game.board, game)
    game.boardView = view
    view.displayGameScreen()
    pygame.event.clear()
    yield game, view
    pygame.event.clear()


def _players(game: Any) -> list[Any]:
    return list(game.playerQueue.queue)


class TestDefaultIsUnrestricted:
    def test_the_filter_defaults_to_none(self, picker: Any) -> None:
        _game, view = picker
        assert view.dev_card_filter is None

    def test_every_held_card_is_selectable_by_default(self, picker: Any) -> None:
        game, view = picker
        human = _players(game)[0]
        human.devCards["ROADBUILDER"] = 1
        rects = _card_rects(game.board)

        _click(rects["ROADBUILDER"].center)  # select
        _click(rects["ROADBUILDER"].center)  # confirm
        assert view.get_dev_card_selection(human) == "ROADBUILDER"


class TestFilteredMenu:
    def test_a_card_outside_the_filter_cannot_be_picked(self, picker: Any) -> None:
        """Select-then-confirm on a filtered-out card must not register.

        The allowed pick is clicked FIRST and the blocked one LAST, so an
        unfiltered picker (which keeps consuming events after a confirm) would
        return the blocked card — the assertion cannot pass by accident.
        """
        game, view = picker
        human = _players(game)[0]
        human.devCards["KNIGHT"] = 1
        human.devCards["ROADBUILDER"] = 1
        view.dev_card_filter = frozenset({"KNIGHT"})
        rects = _card_rects(game.board)

        _click(rects["KNIGHT"].center)  # select
        _click(rects["KNIGHT"].center)  # confirm
        _click(rects["ROADBUILDER"].center)  # blocked: select
        _click(rects["ROADBUILDER"].center)  # blocked: confirm
        assert view.get_dev_card_selection(human) == "KNIGHT"
        assert human.devCards["ROADBUILDER"] == 1

    def test_a_card_inside_the_filter_is_still_pickable(self, picker: Any) -> None:
        game, view = picker
        human = _players(game)[0]
        human.devCards["KNIGHT"] = 1
        human.devCards["ROADBUILDER"] = 1
        view.dev_card_filter = frozenset({"KNIGHT"})
        rects = _card_rects(game.board)

        _click(rects["KNIGHT"].center)
        _click(rects["KNIGHT"].center)
        assert view.get_dev_card_selection(human) == "KNIGHT"
