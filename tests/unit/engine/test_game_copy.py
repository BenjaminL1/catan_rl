"""Tests for catanGame.copy() — needed by AlphaZero-style MCTS lookahead.

The copy contract is:

1. Returned instance is distinct from the original.
2. All mutable game-state objects (board, dice, players, broadcast) are
   distinct between the two instances.
3. Player <-> game back-references are rewired so each player's ``.game``
   points to its own catanGame instance (no cross-pollination).
4. Mutating the original after copying does not affect the copy.
5. The stateless headless boardView may be shared (it's a singleton-like
   no-op object).
6. copy() raises if the game is running in render_mode='human'.
"""

from __future__ import annotations

import pytest

from catan_rl.engine.game import _HeadlessView, catanGame


@pytest.fixture
def headless_game() -> catanGame:
    return catanGame(render_mode=None)


def test_copy_returns_distinct_instance(headless_game: catanGame) -> None:
    cp = headless_game.copy()
    assert cp is not headless_game
    assert isinstance(cp, catanGame)


def test_copy_owns_distinct_board_dice_broadcast(headless_game: catanGame) -> None:
    cp = headless_game.copy()
    assert cp.board is not headless_game.board
    assert cp.dice is not headless_game.dice
    assert cp.broadcast is not headless_game.broadcast


def test_copy_player_back_reference_rewired(headless_game: catanGame) -> None:
    orig_players = list(headless_game.playerQueue.queue)
    cp = headless_game.copy()
    cp_players = list(cp.playerQueue.queue)
    assert len(cp_players) == len(orig_players) == headless_game.numPlayers
    for op, cpp in zip(orig_players, cp_players, strict=True):
        assert cpp is not op, "player objects must be deep-copied"
        assert cpp.game is cp, "copied player's .game must point to the copy"
        assert op.game is headless_game, "original player's .game must be unchanged"


def test_copy_shares_stateless_headless_view(headless_game: catanGame) -> None:
    cp = headless_game.copy()
    assert isinstance(cp.boardView, _HeadlessView)
    # _HeadlessView opts out of deepcopy on purpose; sharing it is safe.
    assert cp.boardView is headless_game.boardView


def test_mutation_isolation_dice_bag(headless_game: catanGame) -> None:
    cp = headless_game.copy()
    before = len(cp.dice.bag)
    headless_game.dice.bag.append(99)
    assert len(cp.dice.bag) == before, "dice bag must not be aliased"


def test_mutation_isolation_player_resources(headless_game: catanGame) -> None:
    cp = headless_game.copy()
    orig_p0 = next(iter(headless_game.playerQueue.queue))
    cp_p0 = next(iter(cp.playerQueue.queue))
    res_key = next(iter(orig_p0.resources))
    before = cp_p0.resources[res_key]
    orig_p0.resources[res_key] = before + 17
    assert cp_p0.resources[res_key] == before, "player.resources must not be aliased"


def test_copy_rejects_render_mode_human(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bypass the real pygame init path: build a headless game, then swap the
    # view for a sentinel non-headless object so copy() trips the guard.
    g = catanGame(render_mode=None)

    class _NotHeadless:
        pass

    g.boardView = _NotHeadless()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="headless"):
        g.copy()
