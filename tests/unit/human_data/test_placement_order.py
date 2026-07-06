"""Placement-order contract tests (step6 §3.1, harvest-blocking).

The openings CV reads a single order-blind post-setup frame, so a
:class:`~catan_rl.human_data.record.PlayerOpening`'s raw tuple order carries no
placement information. :func:`~catan_rl.human_data.orientation.establish_placement_order`
recovers LOG PLACEMENT ORDER (``settlements[1]`` = the 2nd/resource-granting
settlement) from two required signals — the log setup-event sequence (the grant
follows the 2nd settlement) and the grant-glyph adjacency (which vertex). These
tests pin: a normal snake establishes order; a missing setup/grant line or an
ambiguous grant leaves it UNESTABLISHED; the committed game-1 golden fixture
round-trips with order established and matches its log; and
:meth:`~catan_rl.human_data.record.GameRecord.is_scoreboard_eligible` requires the
flag while :meth:`~catan_rl.human_data.record.GameRecord.is_seed_eligible` does not.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from catan_rl.human_data import (
    GameRecord,
    OpponentStrength,
    PlayerOpening,
    establish_placement_order,
    identify_granting_settlement,
    order_openings_by_grant,
    parse_log,
)
from catan_rl.human_data.logparse import LogEvent
from catan_rl.human_data.orientation import granted_resources_under_orientation
from catan_rl.human_data.record import PROVENANCE_PLACEMENT_ORDER_ESTABLISHED
from catan_rl.human_data.topology import load_topology

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"

# The re-snapped game-1 board (desert=11), mirrored from test_scaffold / test_orientation.
_GAME1_HEXES: tuple[dict[str, Any], ...] = (
    {"hex_id": 0, "resource": "SHEEP", "number": 11},
    {"hex_id": 1, "resource": "BRICK", "number": 9},
    {"hex_id": 2, "resource": "WHEAT", "number": 10},
    {"hex_id": 3, "resource": "WHEAT", "number": 3},
    {"hex_id": 4, "resource": "BRICK", "number": 6},
    {"hex_id": 5, "resource": "SHEEP", "number": 5},
    {"hex_id": 6, "resource": "ORE", "number": 4},
    {"hex_id": 7, "resource": "WOOD", "number": 6},
    {"hex_id": 8, "resource": "SHEEP", "number": 2},
    {"hex_id": 9, "resource": "WOOD", "number": 5},
    {"hex_id": 10, "resource": "BRICK", "number": 8},
    {"hex_id": 11, "resource": "DESERT", "number": None},
    {"hex_id": 12, "resource": "SHEEP", "number": 4},
    {"hex_id": 13, "resource": "WOOD", "number": 11},
    {"hex_id": 14, "resource": "WHEAT", "number": 12},
    {"hex_id": 15, "resource": "ORE", "number": 9},
    {"hex_id": 16, "resource": "ORE", "number": 10},
    {"hex_id": 17, "resource": "WOOD", "number": 8},
    {"hex_id": 18, "resource": "WHEAT", "number": 3},
)

# Order-blind (CV-detection order) game-1 openings, as committed in the golden JSON.
_FRAME_OPENINGS: dict[str, PlayerOpening] = {
    "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
    "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
}
# The hand-verified game-1 grants (test_orientation): ThePhantom's 2nd settlement is
# v19 (SHEEP/ORE/ORE), rayman147's is v11 (WHEAT/WOOD/BRICK).
_GRANTS: dict[str, Counter[str] | None] = {
    "ThePhantom": Counter({"SHEEP": 1, "ORE": 2}),
    "rayman147": Counter({"WHEAT": 1, "WOOD": 1, "BRICK": 1}),
}


def _by_hex() -> dict[int, str]:
    return {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}


def _ev(actor: str, kind: str) -> LogEvent:
    return LogEvent(kind=kind, actor=actor, text="")  # type: ignore[arg-type]


def _normal_snake_events() -> list[LogEvent]:
    """A well-formed 1v1 snake setup stream: A, B, B(+grant), A(+grant)."""
    return [
        _ev("rayman147", "setup_settlement"),
        _ev("rayman147", "setup_road"),
        _ev("ThePhantom", "setup_settlement"),
        _ev("ThePhantom", "setup_road"),
        _ev("ThePhantom", "setup_settlement"),
        _ev("ThePhantom", "starting_resources"),
        _ev("ThePhantom", "setup_road"),
        _ev("rayman147", "setup_settlement"),
        _ev("rayman147", "starting_resources"),
        _ev("rayman147", "setup_road"),
    ]


# --- the order helper: normal snake establishes log-placement order ----------


def test_normal_snake_establishes_order() -> None:
    topo = load_topology()
    ordered, established = establish_placement_order(
        _normal_snake_events(), _FRAME_OPENINGS, _GRANTS, _by_hex(), topo
    )
    assert established is True
    # settlements[1] is the 2nd/resource-granting settlement; roads[i] incident to
    # settlements[i]. rayman147 frame (11, 3) re-orders to (3, 11); ThePhantom stays.
    assert ordered["rayman147"].settlements == (3, 11)
    assert ordered["rayman147"].roads == (8, 19)
    assert ordered["ThePhantom"].settlements == (1, 19)
    assert ordered["ThePhantom"].roads == (0, 35)
    # The granting settlement's adjacency equals the read grant, for both players.
    for name, opening in ordered.items():
        granting = opening.settlements[1]
        assert granted_resources_under_orientation(granting, _by_hex(), topo) == _GRANTS[name]


# --- unestablished: missing setup lines --------------------------------------


def test_missing_second_settlement_line_is_unestablished() -> None:
    topo = load_topology()
    # Drop rayman147's 2nd settlement placement — the log now shows only 1 settlement
    # for rayman, so the grant ordinal cannot be recovered.
    events = [
        _ev("rayman147", "setup_settlement"),
        _ev("rayman147", "setup_road"),
        _ev("ThePhantom", "setup_settlement"),
        _ev("ThePhantom", "setup_road"),
        _ev("ThePhantom", "setup_settlement"),
        _ev("ThePhantom", "starting_resources"),
        _ev("ThePhantom", "setup_road"),
        # rayman147's 2nd settlement line is MISSING
        _ev("rayman147", "starting_resources"),
        _ev("rayman147", "setup_road"),
    ]
    ordered, established = establish_placement_order(
        events, _FRAME_OPENINGS, _GRANTS, _by_hex(), topo
    )
    assert established is False
    # Unestablished ⟹ frame order returned unchanged (never a fabricated order).
    assert ordered["rayman147"].settlements == _FRAME_OPENINGS["rayman147"].settlements
    assert ordered["rayman147"].roads == _FRAME_OPENINGS["rayman147"].roads


def test_missing_grant_line_is_unestablished() -> None:
    topo = load_topology()
    # Drop ThePhantom's "received starting resources" line: the grant ordinal is
    # unknown for ThePhantom, so the whole record is order-unestablished.
    events = [
        e
        for e in _normal_snake_events()
        if e.kind != "starting_resources" or e.actor != "ThePhantom"
    ]
    ordered, established = establish_placement_order(
        events, _FRAME_OPENINGS, _GRANTS, _by_hex(), topo
    )
    assert established is False
    assert ordered == dict(_FRAME_OPENINGS)


def test_no_setup_events_is_unestablished() -> None:
    topo = load_topology()
    ordered, established = establish_placement_order([], _FRAME_OPENINGS, _GRANTS, _by_hex(), topo)
    assert established is False
    assert ordered == dict(_FRAME_OPENINGS)


# --- unestablished: the grant glyph cannot disambiguate the vertex -----------


def test_unreadable_grant_is_unestablished() -> None:
    topo = load_topology()
    grants: dict[str, Counter[str] | None] = {
        "ThePhantom": _GRANTS["ThePhantom"],
        "rayman147": None,
    }
    ordered, established = establish_placement_order(
        _normal_snake_events(), _FRAME_OPENINGS, grants, _by_hex(), topo
    )
    assert established is False
    assert ordered == dict(_FRAME_OPENINGS)


def test_grant_matching_neither_settlement_is_unestablished() -> None:
    topo = load_topology()
    # A grant multiset that matches neither of rayman147's settlements (a joint flip).
    grants: dict[str, Counter[str] | None] = {
        "ThePhantom": _GRANTS["ThePhantom"],
        "rayman147": Counter({"ORE": 3}),
    }
    _, established = establish_placement_order(
        _normal_snake_events(), _FRAME_OPENINGS, grants, _by_hex(), topo
    )
    assert established is False


def test_grant_collision_both_settlements_is_unestablished() -> None:
    topo = load_topology()
    by_hex = _by_hex()
    # Find two board vertices that SHARE a 3-hex grant multiset (the granted multiset
    # is a weak discriminator; 38/54 game-1 vertices collide). An opening on such a
    # colliding pair cannot be disambiguated by the grant alone.
    groups: dict[tuple[tuple[str, int], ...], list[int]] = {}
    for v in range(54):
        key = tuple(sorted(granted_resources_under_orientation(v, by_hex, topo).items()))
        groups.setdefault(key, []).append(v)
    shared_key, colliding = next((key, vs) for key, vs in groups.items() if key and len(vs) >= 2)
    v_a, v_b = colliding[0], colliding[1]
    shared = Counter(dict(shared_key))
    opening = PlayerOpening(settlements=(v_a, v_b), roads=(0, 1))
    # identify returns None (two matches ⟹ ambiguous), so order is not established.
    assert identify_granting_settlement(opening, shared, by_hex, topo) is None
    _, established = order_openings_by_grant({"P": opening}, {"P": shared}, by_hex, topo)
    assert established is False


# --- the game-1 golden fixture round-trips with order established -------------


def test_golden_game1_round_trips_with_order_established() -> None:
    golden = json.loads((FIXTURES / "game1_openings.json").read_text(encoding="utf-8"))
    topo = load_topology()
    handles = list(golden["openings"])

    frame_openings = {
        name: PlayerOpening(settlements=tuple(o["settlements"]), roads=tuple(o["roads"]))
        for name, o in golden["openings"].items()
    }
    granted = {name: Counter(m) for name, m in golden["granted_resources"].items()}
    # The committed log_setup_sequence, parsed into the ordered LogEvent stream the
    # harvest driver would feed (setup_settlement / setup_road / starting_resources).
    setup_events = parse_log(golden["log_setup_sequence"], handles=handles).events

    ordered, established = establish_placement_order(
        setup_events, frame_openings, granted, _by_hex(), topo
    )
    assert established is True
    assert golden["placement_order"]["established"] is True
    for name in handles:
        expected = golden["placement_order"][name]
        assert ordered[name].settlements == tuple(expected["settlements"]), name
        assert ordered[name].roads == tuple(expected["roads"]), name
        # settlements[1] is the resource-granting settlement whose adjacency == the
        # committed granted multiset (the log-derived, glyph-pinned 2nd settlement).
        granting = ordered[name].settlements[1]
        assert granted_resources_under_orientation(granting, _by_hex(), topo) == granted[name]


# --- eligibility wiring: scoreboard requires the flag, seed does not ---------


def _record(*, order_established: bool | None, winner: str | None = "ThePhantom") -> GameRecord:
    provenance: dict[str, Any] = {
        "resolution": 1080,
        "ts": 247,
        "board_desert_hex": 11,
        "openings_desert_hex": 11,
    }
    if order_established is not None:
        provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] = order_established
    return GameRecord(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=_GAME1_HEXES,
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        openings={
            "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
            "rayman147": PlayerOpening(settlements=(3, 11), roads=(8, 19)),
        },
        dice_log=(8, 6, 11, 4),
        winner=winner,
        episode_source="natural",
        passed_crosscheck=True,
        provenance=provenance,
        rejection_reason=None,
    )


def test_scoreboard_ineligible_when_order_unestablished_seed_unaffected() -> None:
    # A won, high-tournament, crosscheck-passing record — the ONLY missing clause is
    # the placement-order flag. Scoreboard-excluded; seed-eligible unaffected.
    rec = _record(order_established=False)
    assert rec.is_scoreboard_eligible() is False
    assert rec.is_seed_eligible() is True


def test_scoreboard_ineligible_when_order_flag_absent() -> None:
    # A missing flag reads as unestablished (the safe default) — still seed-eligible.
    rec = _record(order_established=None)
    assert rec.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is None
    assert rec.is_scoreboard_eligible() is False
    assert rec.is_seed_eligible() is True


def test_scoreboard_eligible_when_order_established() -> None:
    rec = _record(order_established=True)
    assert rec.is_scoreboard_eligible() is True
    assert rec.is_seed_eligible() is True
