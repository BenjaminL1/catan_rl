"""Conformance pins for the HUMAN robber picker (``gui/view.moveRobber_display``).

Three divergences from the bot, all of which the bot's own path
(``env/catan_env._apply_robber_placement`` + ``env/masks.py``) never had:

* ``currentPlayer`` was accepted and never used, and ``board.get_players_to_rob``
  is keyed on the hex alone — so the human's OWN building was offered as a steal
  target. ``player.steal_resource(self)`` is a net-zero self-transfer, so a
  mis-click FORFEITED the steal. The bug was biased AGAINST the human.
* With exactly one victim the picker still prompted, blocking on a click. The bot
  never prompts, and after self-filtering there is never a real choice in 1v1.
* An empty robber-spot set reached ``_animated_pick(..., allow_cancel=False)``
  with no matching rect, leaving the human no legal click and the turn
  unfinishable — a SOFT lock (``_animated_pick`` does handle ``pygame.QUIT``, and
  the driver salvages both artifacts on the way out), but unplayable all the
  same. ``env/masks.py`` falls open to all 19 tiles instead.

``_animated_pick`` is a blocking pygame event loop, so it is replaced by a spy in
every test here: a miss would hang the suite rather than fail it, and the spy is
also what lets us assert on the spot set the human was OFFERED.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from catan_rl.engine.game import catanGame
from catan_rl.gui.view import catanGameView


@pytest.fixture()
def view() -> Any:
    pygame.init()
    game = catanGame(render_mode=None)
    v = catanGameView(game.board, game)
    game.boardView = v
    v.displayGameScreen()
    pygame.event.clear()
    yield game, v
    pygame.event.clear()


def _players(game: Any) -> list[Any]:
    return list(game.playerQueue.queue)


def _spy_pick(v: Any, choose: Any) -> list[list[Any]]:
    """Replace the blocking hex picker with a spy returning ``choose(spots)``."""
    seen: list[list[Any]] = []

    def _fake(spots: Any, draw_fn: Any, allow_cancel: Any) -> Any:
        seen.append(list(spots))
        return choose(list(spots))

    v._animated_pick = _fake  # type: ignore[method-assign]
    return seen


def _settle_on_first_hex(game: Any, player: Any) -> int:
    """Give ``player`` a settlement on a corner of some hex; return that hex."""
    board = game.board
    idx, tile = next(iter(board.hexTileDict.items()))
    corner = next(v for v in tile.get_corners(board.flat) if v in board.boardGraph)
    player.build_settlement(corner, board, is_free=True)
    return idx


class TestSelfIsNeverAVictim:
    def test_the_robbing_player_is_filtered_out(self, view: Any) -> None:
        game, v = view
        human = _players(game)[0]
        hex_idx = _settle_on_first_hex(game, human)
        assert human in game.board.get_players_to_rob(hex_idx), (
            "fixture is degenerate: the engine must offer the self-owned hex"
        )
        _spy_pick(v, lambda spots: hex_idx)

        picked_hex, victim = v.moveRobber_display(human, game.board.get_robber_spots())

        assert picked_hex == hex_idx
        assert victim is None, (
            "the human was offered THEMSELVES as a steal victim; the resulting "
            "self-transfer is net-zero, i.e. a silently forfeited steal"
        )


class TestForcedVictimIsAutoSelected:
    def test_one_surviving_victim_needs_no_click(self, view: Any) -> None:
        """No MOUSEBUTTONDOWN is posted. Pre-change this blocked forever in
        ``choosePlayerToRob_display``; the bot never prompts at all."""
        game, v = view
        human, opponent = _players(game)[:2]
        hex_idx = _settle_on_first_hex(game, opponent)
        _spy_pick(v, lambda spots: hex_idx)
        pygame.event.clear()

        _picked_hex, victim = v.moveRobber_display(human, game.board.get_robber_spots())

        assert victim is opponent

    def test_zero_victims_returns_none(self, view: Any) -> None:
        game, v = view
        human = _players(game)[0]
        # No buildings exist at all, so every hex has zero steal candidates.
        _spy_pick(v, lambda spots: spots[0])

        _picked_hex, victim = v.moveRobber_display(human, game.board.get_robber_spots())

        assert victim is None


class TestFailOpenOnEmptySpotSet:
    def test_an_empty_spot_set_offers_all_19_hexes(self, view: Any) -> None:
        """Mirrors ``env/masks.py``, which sets every tile when the
        Friendly-Robber filter leaves nothing. Pre-change the empty set reached a
        non-cancellable picker with no clickable rect, so the turn could not be
        finished (the window still closed on QUIT)."""
        game, v = view
        human = _players(game)[0]
        seen = _spy_pick(v, lambda spots: spots[0])

        v.moveRobber_display(human, {})

        assert len(seen) == 1
        assert len(seen[0]) == 19
        assert set(seen[0]) == set(game.board.hexTileDict)

    def test_a_non_empty_spot_set_is_passed_through_unchanged(self, view: Any) -> None:
        game, v = view
        human = _players(game)[0]
        spots = game.board.get_robber_spots()
        seen = _spy_pick(v, lambda s: s[0])

        v.moveRobber_display(human, spots)

        assert set(seen[0]) == set(spots)
        assert len(seen[0]) < 19, "fixture is degenerate: the robber hex is excluded"
