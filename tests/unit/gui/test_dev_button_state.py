"""Pins for the BUY DEV CARD button state and the dev-deck readout (``gui/view``).

The button was drawn gold unconditionally while the bot's mask
(``env/masks.py``) gates BUY_DEV_CARD on BOTH a non-empty deck and
affordability. ``player.draw_devCard`` no-ops on either half without spending, so
this is a UX gap rather than a rule break — but it is an asymmetry: the human had
to learn the rule by clicking a button that did nothing.

The deck readout is the PUBLIC-REVEAL-DERIVED figure the bot observes
(``policy/obs_encoder._build_global_features``), NOT ``board.devCardStack``. Deck
truth would tell the human how many cards sit in the opponent's hidden hand —
information the bot does not get, and handing it to the human would bias the
playtest the other way.

Both helpers are pure, so these tests need no pygame surface.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

import catan_rl
from catan_rl.engine.game import catanGame
from catan_rl.gui.view import (
    DEV_DECK_TOTAL,
    dev_card_button_enabled,
    public_dev_deck_remaining,
)
from catan_rl.policy.obs_schema import DEV_DECK_INITIAL

_COST = {"ORE": 1, "WHEAT": 1, "SHEEP": 1}


@pytest.fixture()
def game() -> Any:
    return catanGame(render_mode=None)


def _players(g: Any) -> list[Any]:
    return list(g.playerQueue.queue)


class TestBuyButtonPredicate:
    @pytest.mark.parametrize(
        ("afford", "deck_nonempty", "expected"),
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_full_predicate_is_deck_and_affordability(
        self, game: Any, afford: bool, deck_nonempty: bool, expected: bool
    ) -> None:
        human = _players(game)[0]
        if afford:
            for res, amt in _COST.items():
                human.resources[res] += amt
            game.board.bank_draw(_COST)
        if not deck_nonempty:
            game.board.devCardStack = dict.fromkeys(game.board.devCardStack, 0)

        assert dev_card_button_enabled(human, game.board) is expected

    def test_partial_affordability_is_not_enough(self, game: Any) -> None:
        human = _players(game)[0]
        human.resources["ORE"] += 1
        human.resources["WHEAT"] += 1
        game.board.bank_draw({"ORE": 1, "WHEAT": 1})

        assert dev_card_button_enabled(human, game.board) is False


class TestDeckRemainingReadout:
    def test_full_deck_at_game_start(self, game: Any) -> None:
        human, opponent = _players(game)[:2]

        assert public_dev_deck_remaining(human, opponent) == DEV_DECK_TOTAL == 25
        # the duplicated constant may not drift from the obs's deck composition
        assert sum(DEV_DECK_INITIAL) == DEV_DECK_TOTAL

    def test_own_held_and_both_seats_played_are_subtracted(self, game: Any) -> None:
        human, opponent = _players(game)[:2]
        human.devCards["KNIGHT"] = 2  # held, unplayed
        human.newDevCards.append("MONOPOLY")  # bought this turn
        human.knightsPlayed = 1
        opponent.yopPlayed = 1
        opponent.roadBuilderPlayed = 1

        assert public_dev_deck_remaining(human, opponent) == 25 - 3 - 1 - 2

    def test_it_does_not_leak_the_opponents_hidden_hand(self, game: Any) -> None:
        """Deck TRUTH would drop when the opponent buys; the honest figure must
        not, because a bought-but-unplayed card is indistinguishable from a card
        still in the deck."""
        human, opponent = _players(game)[:2]
        before = public_dev_deck_remaining(human, opponent)
        opponent.devCards["KNIGHT"] = 3
        opponent.newDevCards.append("VP")

        assert public_dev_deck_remaining(human, opponent) == before

    def test_it_never_goes_negative(self, game: Any) -> None:
        human, opponent = _players(game)[:2]
        human.knightsPlayed = 99

        assert public_dev_deck_remaining(human, opponent) == 0


class TestRenderedButtonColour:
    """The predicate is wired to the actual fill, not merely exported.

    A corner pixel is sampled rather than the centre: the "DEV CARD" label is
    blitted over the middle of the rect.
    """

    def _button_colour(self, game: Any, human: Any) -> Any:
        import pygame

        from catan_rl.gui.view import catanGameView

        pygame.init()
        v = catanGameView(game.board, game)
        game.boardView = v
        v.human_player = human
        v.displayGameScreen()
        return v.screen.get_at((v.devCard_button.right - 3, v.devCard_button.bottom - 3))

    def test_gold_when_buyable_grey_otherwise(self, game: Any) -> None:
        import pygame

        human = _players(game)[0]
        assert self._button_colour(game, human) == pygame.Color("gray50")

        for res, amt in _COST.items():
            human.resources[res] += amt
        game.board.bank_draw(_COST)
        assert self._button_colour(game, human) == pygame.Color("gold")


class TestGuiLayering:
    """CLAUDE.md rule 8: no RL/training path imports the GUI — and the GUI must
    not import the policy package either. ``from catan_rl.policy.obs_schema
    import ...`` executes ``policy/__init__``, which constructs the CatanPolicy
    import chain and pulls in torch, so the deck size is duplicated as
    ``view.DEV_DECK_TOTAL`` (pinned equal to ``DEV_DECK_INITIAL`` above) rather
    than imported. This test is what makes the duplication safe to keep.
    """

    def test_importing_the_gui_does_not_pull_in_torch(self) -> None:
        src = str(pathlib.Path(catan_rl.__file__).resolve().parent.parent)
        code = (
            "import os, sys\n"
            "os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')\n"
            "import catan_rl.gui.view\n"
            "print('TORCH', 'torch' in sys.modules)\n"
        )
        env = {**os.environ, "PYTHONPATH": src, "SDL_VIDEODRIVER": "dummy"}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert "TORCH False" in out.stdout
