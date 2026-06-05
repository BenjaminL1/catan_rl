"""Unit tests for the Archetype enum (labeling §A archetype field).

The enum is the single source of truth for legal strategy declarations
recorded per-scenario. Tests pin the literal string values because they
appear verbatim in JSONL rows on disk; renaming a value would
retroactively break older labels.
"""

from __future__ import annotations

import json

import pytest

from catan_rl.labeling.archetypes import Archetype


class TestArchetypeValues:
    """The exact string values are part of the on-disk schema."""

    def test_all_archetypes_present(self) -> None:
        """Plan §B lists five archetypes; the enum must expose them all."""
        expected = {"balanced", "OWS", "OWS_hybrid", "road_builder", "other"}
        actual = {a.value for a in Archetype}
        assert actual == expected

    def test_balanced_value(self) -> None:
        assert Archetype.BALANCED.value == "balanced"

    def test_ows_value(self) -> None:
        assert Archetype.OWS.value == "OWS"

    def test_ows_hybrid_value(self) -> None:
        assert Archetype.OWS_HYBRID.value == "OWS_hybrid"

    def test_road_builder_value(self) -> None:
        assert Archetype.ROAD_BUILDER.value == "road_builder"

    def test_other_value(self) -> None:
        assert Archetype.OTHER.value == "other"


class TestArchetypeRoundtrip:
    """Archetypes survive JSON round-trip — they're written to JSONL rows."""

    def test_string_value_round_trip(self) -> None:
        for arch in Archetype:
            recovered = Archetype(arch.value)
            assert recovered is arch

    def test_json_roundtrip(self) -> None:
        for arch in Archetype:
            payload = json.dumps({"archetype": arch.value})
            decoded = json.loads(payload)
            assert Archetype(decoded["archetype"]) is arch

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Archetype("definitely-not-a-real-archetype")
