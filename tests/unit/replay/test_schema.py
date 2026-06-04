"""Tests for `catan_rl.replay.schema`.

Covers:
1. `event_from_dict` round-trips every known StepEvent variant.
2. Unknown event kind raises in strict mode and yields UnknownEvent
   in lenient mode.
3. Missing required field raises with a clear message.
4. UnknownEvent round-trips its original payload bytes
   (forward-compat for v1 viewer reading a v2 replay).
5. EVENT_REGISTRY enumerates every concrete variant by `kind`.
"""

from __future__ import annotations

import logging

import pytest

from catan_rl.replay.schema import (
    EVENT_REGISTRY,
    GameEnd,
    LargestArmyChange,
    LongestRoadChange,
    Monopoly,
    ReplaySchemaError,
    Robber,
    Steal,
    UnknownEvent,
    event_from_dict,
    event_to_dict,
)

KNOWN_EVENTS = [
    LongestRoadChange(prev_owner="player_a", new_owner="player_b", length=6),
    LargestArmyChange(prev_owner=None, new_owner="player_a", knights=3),
    Monopoly(player="player_a", resource="WHEAT", count=4),
    Robber(player="player_b", hex_idx=11),
    Steal(robber="player_a", victim="player_b", resource="WOOD"),
    GameEnd(winner="player_a", vp_breakdown={"player_a": 15, "player_b": 11}),
]


class TestRoundTrip:
    @pytest.mark.parametrize("event", KNOWN_EVENTS)
    def test_known_event_roundtrips(self, event) -> None:
        payload = event_to_dict(event)
        restored = event_from_dict(payload, strict=True)
        assert restored == event

    def test_kind_field_in_payload(self) -> None:
        for event in KNOWN_EVENTS:
            payload = event_to_dict(event)
            assert payload["kind"] == event.kind

    def test_unknown_event_roundtrips_original_payload(self) -> None:
        # A v1 viewer reads a v2 event it doesn't know, captures it as
        # UnknownEvent, and on re-write must emit the original v2
        # payload bytes verbatim so a v2 reader can still parse them.
        v2_payload = {"kind": "FutureV2Event", "stash": [1, 2, 3], "tag": "x"}
        unknown = event_from_dict(v2_payload, strict=False)
        assert isinstance(unknown, UnknownEvent)
        assert unknown.original_kind == "FutureV2Event"
        round_tripped = event_to_dict(unknown)
        assert round_tripped == v2_payload


class TestStrictMode:
    def test_unknown_kind_raises_in_strict_mode(self) -> None:
        with pytest.raises(ReplaySchemaError, match="unknown event kind"):
            event_from_dict({"kind": "NoSuchEvent"}, strict=True)

    def test_unknown_kind_lenient_returns_unknown_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="catan_rl.replay"):
            ev = event_from_dict({"kind": "NewerVersionEvent", "x": 1}, strict=False)
        assert isinstance(ev, UnknownEvent)
        assert ev.original_kind == "NewerVersionEvent"
        assert "unknown event kind" in caplog.text


class TestValidation:
    def test_missing_kind_raises(self) -> None:
        with pytest.raises(ReplaySchemaError, match="missing 'kind'"):
            event_from_dict({"player": "a"}, strict=False)

    def test_non_string_kind_raises(self) -> None:
        with pytest.raises(ReplaySchemaError, match="'kind' must be str"):
            event_from_dict({"kind": 42}, strict=False)

    def test_missing_required_field_raises(self) -> None:
        # Monopoly requires player + resource + count.
        with pytest.raises(ReplaySchemaError, match="missing required field"):
            event_from_dict({"kind": "Monopoly", "player": "player_a"}, strict=True)


class TestDeepCopyIsolation:
    def test_unknown_event_payload_is_deepcopied_from_source(self) -> None:
        # event_from_dict(strict=False) must deep-copy the source
        # payload so a later mutation of the source dict cannot
        # corrupt the stored UnknownEvent.
        source = {"kind": "FutureEvent", "nested": {"a": 1, "b": [1, 2]}}
        ev = event_from_dict(source, strict=False)
        assert isinstance(ev, UnknownEvent)
        # Mutate the source nested dict + list.
        source["nested"]["a"] = 999
        source["nested"]["b"].append(99)
        # The stored UnknownEvent must NOT see the mutation.
        assert ev.payload["nested"]["a"] == 1
        assert ev.payload["nested"]["b"] == [1, 2]

    def test_event_to_dict_unknown_emits_independent_payload(self) -> None:
        # event_to_dict(UnknownEvent) must return a dict whose nested
        # values are independent of the UnknownEvent's payload.
        ev = UnknownEvent(original_kind="X", payload={"nested": {"a": 1}, "kind": "X"})
        out = event_to_dict(ev)
        out["nested"]["a"] = 999
        assert ev.payload["nested"]["a"] == 1


class TestRegistryCoverage:
    def test_every_concrete_variant_registered(self) -> None:
        # If we ever add a new concrete StepEvent subclass and forget
        # to register it, this test should catch it. We compare the
        # registry's set against the kind ClassVars on each known
        # variant.
        registered_kinds = set(EVENT_REGISTRY.keys())
        known_kinds = {
            LongestRoadChange.kind,
            LargestArmyChange.kind,
            Monopoly.kind,
            Robber.kind,
            Steal.kind,
            GameEnd.kind,
        }
        assert registered_kinds == known_kinds, (
            "EVENT_REGISTRY missing a concrete StepEvent variant; check "
            "that every new subclass appends itself to the registry."
        )

    def test_registry_keys_match_kind_classvars(self) -> None:
        # Each registry key string must equal the class's `kind`
        # ClassVar — protects against typos at registration time.
        for kind_str, cls in EVENT_REGISTRY.items():
            assert cls.kind == kind_str
