"""D2 — the practice opponent's development-card policy.

Before this, ``heuristic_play_dev_card`` was an entirely commented-out stub:
the teacher bought dev cards and never played one, so action types 6/7/8/9
had never appeared as a positive example anywhere in the BC corpus.

These tests pin the two things a hand-rolled transition can silently break:
the spec-009 finite-bank conservation invariant, and the broadcast stream that
``BroadcastHandTracker`` uses for perfect 1v1 opponent tracking.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.engine.game import catanGame
from catan_rl.engine.tracker import ResourceTracker
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.policy.obs_schema import BANK_CAPACITY, RESOURCES_CW


def _play_scripted_game(seed: int, n_turns: int = 60):
    np.random.seed(seed)
    game = catanGame(render_mode=None)
    board = game.board
    p1 = heuristicAIPlayer("P1", "black")
    p2 = heuristicAIPlayer("P2", "darkslateblue")
    for p in (p1, p2):
        p.updateAI()
        p.game = game
    game.playerQueue = queue.Queue(2)
    game.playerQueue.put(p1)
    game.playerQueue.put(p2)
    game.resource_tracker = ResourceTracker([p1.name, p2.name])
    tracker = BroadcastHandTracker([p1.name, p2.name])
    tracker.subscribe(game.broadcast)

    for p in (p1, p2, p2, p1):
        game.currentPlayer = p
        p.initial_setup(board)
    game.gameSetup = False

    for _ in range(n_turns):
        for p in (p1, p2):
            game.currentPlayer = p
            p.updateDevCards()
            p.devCardPlayedThisTurn = False
            p.heuristic_pre_roll(board)
            dice = game.rollDice()
            if dice == 7:
                for pp in (p1, p2):
                    if sum(pp.resources.values()) > 9:
                        pp.discardResources(game)
                p.heuristic_move_robber(board)
            else:
                game.update_playerResources(dice, p)
            p.move(board)
            game.check_longest_road(p)
            game.check_largest_army(p)
            for r in RESOURCES_CW:
                total = board.resourceBank[r] + p1.resources[r] + p2.resources[r]
                assert total == BANK_CAPACITY, f"bank conservation broken on {r}: {total}"
    return game, p1, p2, tracker


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dev_card_plays_conserve_the_finite_bank(seed: int) -> None:
    """spec-009: ``bank[R] + Σ hands[R] == 19`` for every R, at every turn."""
    _play_scripted_game(seed)


def test_teacher_actually_plays_dev_cards() -> None:
    """Across a handful of seeds the teacher must play knights AND at least
    one of the three non-knight cards — otherwise there is still nothing for
    behavioural cloning to learn about types 6/7/8/9."""
    knights = 0
    others = 0
    for seed in range(4):
        _game, p1, p2, _tracker = _play_scripted_game(seed)
        for p in (p1, p2):
            knights += p.knightsPlayed
            others += p.yopPlayed + p.monopolyPlayed + p.roadBuilderPlayed
    assert knights > 0
    assert others > 0


def test_dev_card_plays_keep_the_hand_tracker_in_sync() -> None:
    """A play must not desync the broadcast-driven perfect hand tracker —
    every resource delta goes through a broadcast the tracker subscribes to."""
    _game, p1, p2, tracker = _play_scripted_game(seed=1, n_turns=40)
    tracker.verify_against_players({p.name: p for p in (p1, p2)})


def test_year_of_plenty_never_draws_from_an_empty_bank() -> None:
    """A pick the bank cannot supply is not granted, and the card is still
    consumed — the flag is set BEFORE the grant, which is what keeps YoP
    livelock-free (D4)."""
    np.random.seed(0)
    game = catanGame(render_mode=None)
    board = game.board
    p = heuristicAIPlayer("P1", "black")
    p.updateAI()
    p.game = game
    game.playerQueue = queue.Queue(2)
    game.playerQueue.put(p)
    game.resource_tracker = ResourceTracker([p.name])
    p.devCards["YEAROFPLENTY"] = 1
    # Drain the bank of WOOD entirely.
    drained = board.resourceBank["WOOD"]
    board.bank_draw({"WOOD": drained})
    before = dict(p.resources)
    p.heuristic_play_year_of_plenty(board, "WOOD", "WOOD")
    assert p.devCardPlayedThisTurn is True
    assert p.devCards["YEAROFPLENTY"] == 0
    assert p.resources == before  # nothing granted, but the card is spent
    assert board.resourceBank["WOOD"] == 0


def test_yop_picks_never_return_a_pair_the_bank_cannot_honour() -> None:
    """``_yop_picks``' "bank cannot double up" fallback used to re-select the
    very resource it was rejecting (its scan did not exclude ``first``), so it
    returned the ``(first, first)`` pair the branch exists to reject — a card
    that grants ONE resource, written into the corpus as a doubled pick."""
    np.random.seed(0)
    game = catanGame(render_mode=None)
    board = game.board
    p = heuristicAIPlayer("P1", "black")
    p.updateAI()
    p.game = game
    game.playerQueue = queue.Queue(2)
    game.playerQueue.put(p)
    game.resource_tracker = ResourceTracker([p.name])

    # One ORE short of a CITY (3 ORE + 2 WHEAT), with exactly one ORE left in
    # the bank: doubling ORE is impossible.
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.resources["ORE"] = 2
    p.resources["WHEAT"] = 2
    board.bank_draw({"ORE": board.resourceBank["ORE"] - 1})

    picks = p._yop_picks(board)
    assert picks is not None
    first, second = picks
    if first == second:
        assert board.resourceBank[first] >= 2, f"doubled pick {first} with bank < 2"
    else:
        assert board.resourceBank[first] >= 1 and board.resourceBank[second] >= 1

    # And the grant matches what was asked for.
    before = dict(p.resources)
    p.heuristic_play_year_of_plenty(board, first, second)
    granted = sum(p.resources[r] - before[r] for r in RESOURCES_CW)
    assert granted == 2, "the teacher asked for a pair the bank could not supply"
