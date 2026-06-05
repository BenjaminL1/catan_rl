"""Tests for Phase 2b: EventCollector + classify_step_events + extract_sub_actions."""

from __future__ import annotations

import pytest

from catan_rl.engine.broadcast import GameBroadcast
from catan_rl.replay.recorder import (
    EventCollector,
    classify_step_events,
    extract_sub_actions,
)
from catan_rl.replay.schema import (
    GameEnd,
    LargestArmyChange,
    LongestRoadChange,
    Monopoly,
    PlayerStateSnapshot,
    Robber,
    Steal,
    StepStateSnapshot,
    SubAction,
)

_SEAT_TO_ACTOR = {"Agent": "player_a", "Opponent": "player_b"}


# ---------------------------------------------------------------------------
# EventCollector
# ---------------------------------------------------------------------------


class TestEventCollector:
    def test_subscribe_then_drain_returns_events_in_order(self) -> None:
        bus = GameBroadcast()
        collector = EventCollector()
        collector.subscribe(bus)
        bus.dice_roll("Agent", 9)
        bus.discard("Opponent", ["WOOD"])
        out = collector.drain()
        assert len(out) == 2
        assert out[0]["type"] == "DICE_ROLL"
        assert out[1]["type"] == "DISCARD"

    def test_drain_empties_buffer(self) -> None:
        bus = GameBroadcast()
        collector = EventCollector()
        collector.subscribe(bus)
        bus.dice_roll("Agent", 9)
        collector.drain()
        assert collector.drain() == []

    def test_unsubscribe_stops_collection(self) -> None:
        bus = GameBroadcast()
        collector = EventCollector()
        collector.subscribe(bus)
        bus.dice_roll("Agent", 9)
        collector.unsubscribe(bus)
        bus.dice_roll("Opponent", 5)
        out = collector.drain()
        assert len(out) == 1
        assert out[0]["player"] == "Agent"

    def test_peek_does_not_drain(self) -> None:
        bus = GameBroadcast()
        collector = EventCollector()
        collector.subscribe(bus)
        bus.dice_roll("Agent", 9)
        assert len(collector.peek()) == 1
        assert len(collector.peek()) == 1  # idempotent
        assert len(collector.drain()) == 1


# ---------------------------------------------------------------------------
# classify_step_events
# ---------------------------------------------------------------------------


class TestClassifyStepEvents:
    def _capture(self, fn) -> list[dict]:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        fn(bus)
        return c.drain()

    def test_monopoly_produces_step_event_and_log(self) -> None:
        raw = self._capture(lambda b: b.monopoly("Agent", "WHEAT", 4))
        events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [Monopoly(player="player_a", resource="WHEAT", count=4)]
        assert lines == ["player_a played Monopoly on WHEAT (+4)"]

    def test_robber_translates_actor(self) -> None:
        raw = self._capture(lambda b: b.move_robber("Opponent", 7))
        events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [Robber(player="player_b", hex_idx=7)]
        assert lines == ["player_b moved robber to hex 7"]

    def test_steal_translates_both_actors(self) -> None:
        raw = self._capture(lambda b: b.steal("Agent", "Opponent", "ORE"))
        events, _ = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [Steal(robber="player_a", victim="player_b", resource="ORE")]

    def test_longest_road_change_handles_first_award_None_prev(self) -> None:
        raw = self._capture(
            lambda b: b.longest_road_change(prev_owner=None, new_owner="Agent", length=5)
        )
        events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [LongestRoadChange(prev_owner=None, new_owner="player_a", length=5)]
        assert "Longest Road → player_a (length 5)" in lines[0]

    def test_largest_army_change_transfer(self) -> None:
        raw = self._capture(
            lambda b: b.largest_army_change(prev_owner="Agent", new_owner="Opponent", knights=4)
        )
        events, _ = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [LargestArmyChange(prev_owner="player_a", new_owner="player_b", knights=4)]

    def test_game_end_translates_vp_breakdown_keys(self) -> None:
        raw = self._capture(lambda b: b.game_end("Agent", {"Agent": 15, "Opponent": 9}))
        events, _ = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == [
            GameEnd(
                winner="player_a",
                vp_breakdown={"player_a": 15, "player_b": 9},
            )
        ]

    def test_legacy_events_produce_log_lines_no_step_events(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.dice_roll("Agent", 9)
        bus.discard("Opponent", ["WOOD", "BRICK"])
        bus.year_of_plenty("Agent", ["ORE", "WHEAT"])
        bus.resource_change("Agent", {"WOOD": -1}, source="BUILD_ROAD")
        raw = c.drain()
        events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == []
        # 3 log lines (RESOURCE_CHANGE intentionally skipped).
        assert lines == [
            "player_a rolled 9",
            "player_b discarded WOOD, BRICK",
            "player_a Year of Plenty: took ORE, WHEAT",
        ]

    def test_build_event_produces_log_but_no_step_event(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="ROAD", location=-1)
        raw = c.drain()
        events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == []
        assert lines == ["player_a built ROAD"]

    def test_unmapped_player_raises_value_error(self) -> None:
        # Defensive: if an event references a player name not in
        # seat_to_actor, the classifier raises rather than writing
        # the engine name into the JSON.
        raw = [{"type": "MOVE_ROBBER", "player": "MysteryPlayer", "hex_idx": 7}]
        with pytest.raises(ValueError, match="not in seat_to_actor"):
            classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)

    def test_unknown_event_logs_but_does_not_crash(self, caplog) -> None:
        raw = [{"type": "MYSTERY_FUTURE_EVENT", "stuff": 1}]
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="catan_rl.replay"):
            events, lines = classify_step_events(raw, seat_to_actor=_SEAT_TO_ACTOR)
        assert events == []
        assert lines == ["<unknown event MYSTERY_FUTURE_EVENT>"]
        assert "unknown broadcast type" in caplog.text


# ---------------------------------------------------------------------------
# extract_sub_actions
# ---------------------------------------------------------------------------


def _empty_player(name: str) -> PlayerStateSnapshot:
    return PlayerStateSnapshot(
        name=name,
        vp=0,
        resources={"WOOD": 0, "BRICK": 0, "WHEAT": 0, "ORE": 0, "SHEEP": 0},
        dev_cards_hand={
            "KNIGHT": 0,
            "VP": 0,
            "ROAD_BUILDER": 0,
            "YEAR_OF_PLENTY": 0,
            "MONOPOLY": 0,
        },
        dev_cards_played={
            "KNIGHT": 0,
            "VP": 0,
            "ROAD_BUILDER": 0,
            "YEAR_OF_PLENTY": 0,
            "MONOPOLY": 0,
        },
    )


def _state(
    *,
    settlements_a: tuple[int, ...] = (),
    cities_a: tuple[int, ...] = (),
    roads_a: tuple[int, ...] = (),
    settlements_b: tuple[int, ...] = (),
    cities_b: tuple[int, ...] = (),
    roads_b: tuple[int, ...] = (),
    robber_hex: int = 9,
) -> StepStateSnapshot:
    return StepStateSnapshot(
        settlements={"player_a": settlements_a, "player_b": settlements_b},
        cities={"player_a": cities_a, "player_b": cities_b},
        roads={"player_a": roads_a, "player_b": roads_b},
        robber_hex=robber_hex,
        players={
            "player_a": _empty_player("Agent"),
            "player_b": _empty_player("Opponent"),
        },
        longest_road_holder=None,
        largest_army_holder=None,
    )


class TestExtractSubActions:
    def test_dice_roll_uses_provided_d1_d2(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.dice_roll("Agent", 9)
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(),
            dice_roll=(4, 5),
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="RollDice", args={"d1": 4, "d2": 5})]

    def test_build_settlement_back_fills_vertex_idx(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="SETTLEMENT", location=-1)
        raw = c.drain()
        prev = _state(settlements_a=())
        curr = _state(settlements_a=(17,))
        actions = extract_sub_actions(
            raw,
            prev_snapshot=prev,
            curr_snapshot=curr,
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="BuildSettlement", args={"vertex_idx": 17})]

    def test_build_road_back_fills_edge_idx(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="ROAD", location=-1)
        raw = c.drain()
        prev = _state(roads_a=())
        curr = _state(roads_a=(42,))
        actions = extract_sub_actions(
            raw,
            prev_snapshot=prev,
            curr_snapshot=curr,
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="BuildRoad", args={"edge_idx": 42})]

    def test_two_road_builds_take_successive_indices(self) -> None:
        # Road-builder dev card → 2 BUILD ROAD events in one step. The
        # back-fill must assign the diff'd edge ids in order.
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="ROAD", location=-1)
        bus.build("Agent", kind="ROAD", location=-1)
        raw = c.drain()
        prev = _state(roads_a=())
        curr = _state(roads_a=(10, 22))
        actions = extract_sub_actions(
            raw,
            prev_snapshot=prev,
            curr_snapshot=curr,
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [
            SubAction(kind="BuildRoad", args={"edge_idx": 10}),
            SubAction(kind="BuildRoad", args={"edge_idx": 22}),
        ]

    def test_two_road_builds_preserve_temporal_order_when_descending(
        self,
    ) -> None:
        # Regression for Phase 2b reviewer HIGH: the engine appends to
        # buildGraph["ROADS"] in temporal play order; snapshot_state
        # preserves that order. If road A (edge 22) is placed BEFORE
        # road B (edge 10), the recorder must emit
        # ``BuildRoad(22), BuildRoad(10)`` — NOT sorted-by-int.
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="ROAD", location=-1)
        bus.build("Agent", kind="ROAD", location=-1)
        raw = c.drain()
        prev = _state(roads_a=())
        curr = _state(roads_a=(22, 10))  # descending — temporal play order
        actions = extract_sub_actions(
            raw,
            prev_snapshot=prev,
            curr_snapshot=curr,
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [
            SubAction(kind="BuildRoad", args={"edge_idx": 22}),
            SubAction(kind="BuildRoad", args={"edge_idx": 10}),
        ]

    def test_move_robber_with_paired_steal_records_victim(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.move_robber("Agent", 7)
        bus.resource_change("Opponent", {"ORE": -1}, source="STEAL")
        bus.resource_change("Agent", {"ORE": 1}, source="STEAL")
        bus.steal("Agent", "Opponent", "ORE")
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(robber_hex=7),
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        # MoveRobber should pick up victim from the paired STEAL.
        move_actions = [a for a in actions if a.kind == "MoveRobber"]
        assert len(move_actions) == 1
        assert move_actions[0].args["hex_idx"] == 7
        assert move_actions[0].args["victim"] == "player_b"

    def test_move_robber_without_steal_records_victim_none(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.move_robber("Agent", 7)  # no STEAL follows
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(robber_hex=7),
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        move_actions = [a for a in actions if a.kind == "MoveRobber"]
        assert len(move_actions) == 1
        assert move_actions[0].args["victim"] is None

    def test_monopoly_subaction(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.monopoly("Agent", "WHEAT", 4)
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(),
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="PlayMonopoly", args={"resource": "WHEAT"})]

    def test_discard_records_resource_counts(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.discard("Opponent", ["WOOD", "WOOD", "BRICK"])
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(),
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="Discard", args={"discarded": {"WOOD": 2, "BRICK": 1}})]

    def test_yop_records_two_resources(self) -> None:
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.year_of_plenty("Agent", ["WHEAT", "ORE"])
        raw = c.drain()
        actions = extract_sub_actions(
            raw,
            prev_snapshot=_state(),
            curr_snapshot=_state(),
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [
            SubAction(
                kind="PlayYearOfPlenty",
                args={"res_a": "WHEAT", "res_b": "ORE"},
            )
        ]

    def test_build_no_diff_falls_back_to_neg1(self) -> None:
        # If somehow a BUILD fires without a corresponding diff
        # (engine bug, or a port that emits ahead of state mutation),
        # the recorder records location=-1 rather than crashing.
        bus = GameBroadcast()
        c = EventCollector()
        c.subscribe(bus)
        bus.build("Agent", kind="SETTLEMENT", location=-1)
        raw = c.drain()
        prev = _state(settlements_a=())
        curr = _state(settlements_a=())  # no change
        actions = extract_sub_actions(
            raw,
            prev_snapshot=prev,
            curr_snapshot=curr,
            dice_roll=None,
            seat_to_actor=_SEAT_TO_ACTOR,
        )
        assert actions == [SubAction(kind="BuildSettlement", args={"vertex_idx": -1})]
