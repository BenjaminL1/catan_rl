"""Conformance pins for the HUMAN dev-card path (``engine/player.play_devCard``).

The function interleaved *choice resolution* with *state commitment*, and that
one structural fact produced three separate divergences from the bot
(``env/catan_env.py``), which receives its choice pre-resolved and commits in a
single block:

* a cancelled Year-of-Plenty / Monopoly picker left ``yopPlayed`` /
  ``monopolyPlayed`` incremented forever. Those counters are PUBLIC state fed to
  the policy (``policy/obs_encoder.py``) and audited by
  ``eval/rules_invariants.py``, and because cancelling also restored the turn
  allowance the leak was UNBOUNDED within a single turn;
* Knight resolved the robber pick BEFORE ``check_largest_army``, so the
  Friendly-Robber filter (``board.get_robber_spots``) ran without Largest Army's
  +2 VP — the two seats could be offered different legal hexes from identical
  positions;
* Road Builder spent the card and then made two CANCELLABLE placements, so a
  stray click forfeited a free road with no feedback and no refund.

The pickers are stubbed rather than clicked. ``play_devCard`` reaches them
duck-typed through ``game.boardView``, and a real picker is a blocking pygame
event loop: driving two of them in sequence with pre-posted events is not
possible (the first ``pygame.event.get()`` drains the queue, including the
clicks meant for the second picker) and a miss hangs the suite. Stubs also let
each test assert on the arguments a picker was HANDED, which is the whole point
of the Knight-ordering pin.
"""

from __future__ import annotations

from typing import Any

import pytest

from catan_rl.engine.broadcast import BroadcastEventType
from catan_rl.engine.game import catanGame


class ScriptedView:
    """Duck-typed stand-in for ``game.boardView`` with programmable pickers.

    Every other attribute resolves to a no-op, matching ``game._HeadlessView``,
    so display calls inside the engine are silently absorbed.
    """

    def __init__(
        self,
        *,
        dev_card: Any = None,
        resource: Any = None,
        roads: Any = None,
    ) -> None:
        self.dev_card = dev_card
        #: A single value, or a list consumed one call at a time.
        self.resource = resource
        #: Road picks consumed one per ``buildRoad_display`` call. ``None``
        #: entries model a MISS (an empty-space click).
        self.roads = list(roads) if roads is not None else []
        self.road_calls: list[dict[str, Any]] = []
        self.robber_spots: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def get_dev_card_selection(self, player: Any) -> Any:
        return self.dev_card

    def get_resource_selection(self, player: Any, mode: str, num_to_select: int = 1) -> Any:
        if isinstance(self.resource, list):
            return self.resource.pop(0)
        return self.resource

    def buildRoad_display(
        self, currentPlayer: Any, roadsPossibleDict: Any, allow_cancel: Any = None
    ) -> Any:
        legal = [e for e in roadsPossibleDict if roadsPossibleDict[e]]
        self.road_calls.append({"allow_cancel": allow_cancel, "n_legal": len(legal)})
        pick = self.roads.pop(0) if self.roads else None
        return legal[0] if pick == "FIRST" else pick

    def moveRobber_display(self, currentPlayer: Any, possibleRobberDict: Any) -> Any:
        # Capture the legality set the human is OFFERED, alongside the set
        # ``env/masks.py`` would build from the identical position at the
        # identical moment, then take the first hex and steal from nobody so the
        # pin is about ordering and nothing else.
        self.robber_spots.append(
            {
                "offered": set(possibleRobberDict),
                "mask_would_offer": set(currentPlayer.game.board.get_robber_spots()),
                "largest_army": bool(currentPlayer.largestArmyFlag),
            }
        )
        return next(iter(possibleRobberDict)), None


@pytest.fixture()
def game() -> Any:
    g = catanGame(render_mode=None)
    return g


def _players(g: Any) -> list[Any]:
    return list(g.playerQueue.queue)


def _human(g: Any) -> Any:
    p = _players(g)[0]
    p.isAI = False
    return p


def _counters(p: Any) -> dict[str, int]:
    return {
        "knights": p.knightsPlayed,
        "yop": p.yopPlayed,
        "monopoly": p.monopolyPlayed,
        "roadbuilder": p.roadBuilderPlayed,
    }


class TestCancelStrandsNothing:
    """A cancelled picker must leave ZERO engine state behind."""

    @pytest.mark.parametrize(
        ("card", "counter"),
        [("YEAROFPLENTY", "yop"), ("MONOPOLY", "monopoly")],
    )
    def test_cancelling_the_picker_is_a_complete_no_op(
        self, game: Any, card: str, counter: str
    ) -> None:
        human = _human(game)
        human.devCards[card] = 1
        game.boardView = ScriptedView(dev_card=card, resource=None)
        before = _counters(human)

        human.play_devCard(game)

        assert _counters(human) == before, "a cancelled picker stranded a public counter"
        assert human.devCards[card] == 1, "the card was not refunded"
        assert human.devCardPlayedThisTurn is False

    def test_repeated_cancels_cannot_inflate_the_counters(self, game: Any) -> None:
        """The leak used to be UNBOUNDED: cancelling restored the turn allowance,
        so the same card could be re-cancelled arbitrarily many times in one turn
        and drive the public count past the number of that card in the deck."""
        human = _human(game)
        human.devCards["YEAROFPLENTY"] = 1
        human.devCards["MONOPOLY"] = 1
        game.boardView = ScriptedView(dev_card="YEAROFPLENTY", resource=None)

        for _ in range(3):
            human.play_devCard(game)
        game.boardView.dev_card = "MONOPOLY"
        for _ in range(3):
            human.play_devCard(game)

        assert human.yopPlayed == 0
        assert human.monopolyPlayed == 0
        assert human.devCards["YEAROFPLENTY"] == 1
        assert human.devCards["MONOPOLY"] == 1

    def test_a_completed_play_still_commits_exactly_once(self, game: Any) -> None:
        human = _human(game)
        human.devCards["YEAROFPLENTY"] = 1
        game.boardView = ScriptedView(dev_card="YEAROFPLENTY", resource=["WOOD", "ORE"])

        human.play_devCard(game)

        assert human.yopPlayed == 1
        assert human.devCards["YEAROFPLENTY"] == 0
        assert human.devCardPlayedThisTurn is True


class TestKnightOrdering:
    def test_largest_army_is_awarded_before_the_robber_legality_set(self, game: Any) -> None:
        """Bot order is counter -> ``check_largest_army`` -> robber placement, so
        the Friendly-Robber filter sees the +2 VP. Human order was robber first,
        which can offer a DIFFERENT legal hex set from an identical position."""
        human = _human(game)
        opponent = _players(game)[1]
        # Give the human two settlements so their own hexes are Friendly-Robber
        # protected while their visible VP is under 3, and the opponent one so
        # the board is not degenerate.
        vertices = sorted(game.board.boardGraph.keys(), key=lambda v: (v.x, v.y))
        human.build_settlement(vertices[0], game.board, is_free=True)
        human.build_settlement(vertices[len(vertices) // 2], game.board, is_free=True)
        opponent.build_settlement(vertices[-1], game.board, is_free=True)

        # Third knight -> Largest Army -> +2 VP -> visible VP crosses 3.
        human.knightsPlayed = 2
        human.devCards["KNIGHT"] = 1
        assert human.victoryPoints - human.devCards["VP"] < 3

        spots_before = set(game.board.get_robber_spots())
        view = ScriptedView(dev_card="KNIGHT")
        game.boardView = view

        human.play_devCard(game)

        assert human.largestArmyFlag is True
        pick = view.robber_spots[0]
        assert pick["largest_army"] is True, (
            "Largest Army was still unresolved when the human was asked to place "
            "the robber; the bot awards it first"
        )
        assert pick["offered"] == pick["mask_would_offer"], (
            "the human was offered a legality set computed at a different point "
            "in the ordering than the bot's mask would use"
        )
        assert pick["offered"] > spots_before, (
            "test position is degenerate: the +2 VP did not unlock any hex, so "
            "this cannot distinguish the two orderings"
        )


class TestRoadBuilder:
    def test_a_miss_cannot_forfeit_a_free_road(self, game: Any) -> None:
        """The card is spent before placement, so placement must be
        NON-cancellable while a legal road exists — mirroring the bot, which
        keeps ``road_building_roads_left`` until a road is actually placed."""
        human = _human(game)
        opponent = _players(game)[1]
        vertices = sorted(game.board.boardGraph.keys(), key=lambda v: (v.x, v.y))
        # ``get_potential_roads`` grows off EXISTING roads, so seed one.
        human.build_settlement(vertices[0], game.board, is_free=True)
        human.build_road(
            vertices[0], game.board.boardGraph[vertices[0]].neighbors[0], game.board, is_free=True
        )
        opponent.build_settlement(vertices[-1], game.board, is_free=True)
        human.devCards["ROADBUILDER"] = 1

        view = ScriptedView(dev_card="ROADBUILDER", roads=["FIRST", "FIRST"])
        game.boardView = view
        roads_before = len(human.buildGraph["ROADS"])

        human.play_devCard(game)

        assert len(human.buildGraph["ROADS"]) == roads_before + 2
        assert human.roadBuilderPlayed == 1
        assert human.devCards["ROADBUILDER"] == 0
        assert [c["allow_cancel"] for c in view.road_calls] == [False, False], (
            "a cancellable picker can forfeit an already-spent free road"
        )

    def test_placement_is_skipped_when_no_legal_road_exists(self, game: Any) -> None:
        """Matches the bot zeroing ``road_building_roads_left`` when no legal
        road exists — the card is still consumed, but nothing hangs."""
        human = _human(game)
        human.devCards["ROADBUILDER"] = 1
        assert not any(game.board.get_potential_roads(human).values())

        view = ScriptedView(dev_card="ROADBUILDER")
        game.boardView = view

        human.play_devCard(game)

        assert view.road_calls == []
        assert human.roadBuilderPlayed == 1


def _monopoly_sink(game: Any) -> list[dict[str, Any]]:
    """Subscribe to the broadcaster and collect every MONOPOLY event.

    ``GameBroadcast`` keeps only ``last_event``; subscription is the supported
    way to see the whole stream (it is what ``env/hand_tracker.py`` uses).
    """
    seen: list[dict[str, Any]] = []

    def _on(event: dict[str, Any]) -> None:
        if event.get("type") == BroadcastEventType.MONOPOLY.value:
            seen.append(event)

    game.broadcast.subscribe(_on)
    return seen


class TestMonopolyBroadcast:
    def test_human_monopoly_emits_the_structural_event(self, game: Any) -> None:
        human = _human(game)
        opponent = _players(game)[1]
        human.devCards["MONOPOLY"] = 1
        opponent.resources["WOOD"] += 3
        game.board.bank_draw({"WOOD": 3})
        game.boardView = ScriptedView(dev_card="MONOPOLY", resource="WOOD")
        events = _monopoly_sink(game)

        human.play_devCard(game)

        assert len(events) == 1
        assert events[0]["player"] == human.name
        assert events[0]["resource"] == "WOOD"
        assert events[0]["count"] == 3
        assert human.resources["WOOD"] == 3
        game.board.assert_conservation(_players(game))

    def test_it_fires_even_when_nothing_was_stolen(self, game: Any) -> None:
        """The bot emits unconditionally (``env/catan_env.py``). A zero-transfer
        Monopoly is still a played dev card, and swallowing it makes the play
        invisible to ``rules_invariants``' per-turn attribution."""
        human = _human(game)
        human.devCards["MONOPOLY"] = 1
        game.boardView = ScriptedView(dev_card="MONOPOLY", resource="ORE")
        events = _monopoly_sink(game)

        human.play_devCard(game)

        assert len(events) == 1
        assert events[0]["count"] == 0

    def test_heuristic_monopoly_already_emits_it(self, game: Any) -> None:
        """Characterization pin, NOT a fix: the audit reported this emitter
        missing from ``agents/heuristic.py``, but it is already present. Pinned
        so the two seats cannot silently diverge again."""
        from catan_rl.agents.heuristic import heuristicAIPlayer

        bot = heuristicAIPlayer("Bot", "black")
        bot.game = game
        game.playerQueue.queue[1] = bot
        bot.devCards["MONOPOLY"] = 1
        human = _human(game)
        human.resources["SHEEP"] += 2
        game.board.bank_draw({"SHEEP": 2})
        events = _monopoly_sink(game)

        bot.heuristic_play_monopoly(game.board, "SHEEP")

        assert len(events) == 1
        assert events[0]["count"] == 2
