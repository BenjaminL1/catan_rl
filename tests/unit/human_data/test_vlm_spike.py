"""Tests for the VLM opening-localization decision spike (``scripts/vlm_spike.py``).

The spike delegates ONLY perception to a (mock/file) localizer that describes each
opening piece by tile adjacency; the engine-id snap, the log-sourced placement
order, and the fail-closed validators (GameRecord invariants + joint-flip glyph
anchor) are all deterministic. These tests prove:

* a scripted CORRECT adjacency (game-1's true openings) snaps to the expected
  vertex/edge ids AND passes the existing validators (accepted GameRecord);
* an AMBIGUOUS adjacency (0 or 2 matching vertices) yields a typed
  ``ambiguous_snap`` rejection — never a guess (fail-closed);
* placement ORDER is taken from the supplied LOG sequence, not the localizer.

No network: the board is a synthetic ``BoardRead`` (the re-snapped game-1 desert=11
board) and the frames are never read — the MockLocalizer ignores them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catan_rl.human_data import load_topology
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.orientation import granted_resources_under_orientation
from catan_rl.human_data.record import OpponentStrength

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_vlm_spike() -> Any:
    spec = importlib.util.spec_from_file_location(
        "vlm_spike_mod", REPO_ROOT / "scripts" / "vlm_spike.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so slots=True dataclasses can resolve their own module
    # (dataclasses reads sys.modules[__module__].__dict__ during KW_ONLY detection).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vlm = _load_vlm_spike()


# The re-snapped game-1 board (desert=11) — the same legal standard 19-tile board
# used in test_validate; v11 adjacency == rayman's grant (WHEAT/WOOD/BRICK) and v19
# adjacency == ThePhantom's grant (SHEEP/ORE/ORE), so the glyph anchor + order both
# resolve on the true openings.
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

# Game-1 ground truth (order-blind CV-detection order, as committed in the fixture).
_TRUE_SETTLEMENTS: dict[str, tuple[int, ...]] = {"ThePhantom": (1, 19), "rayman147": (11, 3)}
_TRUE_ROADS: dict[str, tuple[int, ...]] = {"ThePhantom": (0, 35), "rayman147": (19, 8)}
_PLAYERS = {"agent": "ThePhantom", "opponent": "rayman147"}
_DRAFT_ORDER = ("rayman147", "ThePhantom", "ThePhantom", "rayman147")
_LOG_SEQUENCE = [
    "rayman147 placed a Settlement",
    "rayman147 placed a Road",
    "ThePhantom placed a Settlement",
    "ThePhantom placed a Road",
    "ThePhantom placed a Settlement",
    "ThePhantom received starting resources",
    "ThePhantom placed a Road",
    "rayman147 placed a Settlement",
    "rayman147 received starting resources",
    "rayman147 placed a Road",
    "rayman147 rolled (= setup complete)",
]


def _board(desert_hex: int = 11) -> BoardRead:
    return BoardRead(
        hexes=_GAME1_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=desert_hex,
        residual_px=0.77,
        screen_rule_gap=43.5,
        pip_ok=True,
    )


def _granted() -> dict[str, Counter[str] | None]:
    topo = load_topology()
    by_hex = {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}
    # The 2nd/granting settlements: ThePhantom v19, rayman147 v11 (hand-verified).
    return {
        "ThePhantom": granted_resources_under_orientation(19, by_hex, topo),
        "rayman147": granted_resources_under_orientation(11, by_hex, topo),
    }


def _true_localized(
    settlements: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    topo = load_topology()
    setts = settlements if settlements is not None else _TRUE_SETTLEMENTS
    return {
        handle: vlm.localized_from_ids(setts[handle], _TRUE_ROADS[handle], topo)
        for handle in _TRUE_SETTLEMENTS
    }


def _strength() -> OpponentStrength:
    return OpponentStrength(tier="high", source="tournament", confidence=0.8)


# --- (c) SNAP: adjacency -> engine id ---------------------------------------


def test_snap_settlement_recovers_true_vertices() -> None:
    topo = load_topology()
    # v1 touches hexes {0,2,3}; the snap recovers exactly v1.
    assert vlm.snap_settlement_vertex(topo.vertex_adjacent_hexes[1], topo) == 1
    assert vlm.snap_settlement_vertex(topo.vertex_adjacent_hexes[11], topo) == 11


def test_snap_road_recovers_true_edge() -> None:
    topo = load_topology()
    road = vlm.localized_from_ids((), (19,), topo).roads[0]
    assert vlm.snap_road_edge(road, topo) == 19


def test_scripted_correct_adjacency_snaps_to_expected_ids() -> None:
    topo = load_topology()
    result = vlm.snap_localized_openings(_true_localized(), topo)
    assert result.rejection_reason is None
    assert result.openings is not None
    for handle in _TRUE_SETTLEMENTS:
        assert set(result.openings[handle].settlements) == set(_TRUE_SETTLEMENTS[handle])
        assert set(result.openings[handle].roads) == set(_TRUE_ROADS[handle])


# --- MockLocalizer end-to-end: snap+validate passes the existing gate --------


def test_mock_localizer_correct_adjacency_accepted_and_exact() -> None:
    topo = load_topology()
    localizer = vlm.MockLocalizer(scripted=_true_localized())
    localized = localizer.localize(Path("post.png"), Path("baseline.png"), _board())
    setup_events = vlm.build_setup_events(_LOG_SEQUENCE, _PLAYERS.values())
    result = vlm.snap_and_validate(
        localized=localized,
        board=_board(),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8, 6, 11, 4),
        winner="ThePhantom",
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=setup_events,
    )
    assert result.accepted is True
    assert result.cross_check_result.anchor_ran is True  # joint-flip glyph firewall ran
    assert result.rejection_reason is None
    rec = result.record
    # Exact vertex/edge match against ground truth (order-blind sets).
    truth = {
        handle: vlm.PlayerOpening(settlements=_TRUE_SETTLEMENTS[handle], roads=_TRUE_ROADS[handle])
        for handle in _TRUE_SETTLEMENTS
    }
    score = vlm.score_openings(rec.openings, truth)
    assert score.all_exact is True
    assert score.settlements_correct == 4
    assert score.roads_correct == 4
    # Placement order established from the log + grant (settlements[1] == granting).
    assert rec.provenance["placement_order_established"] is True
    assert rec.openings["rayman147"].settlements[1] == 11
    assert rec.openings["ThePhantom"].settlements[1] == 19
    assert rec.is_scoreboard_eligible() is True


# --- (c) fail-closed: ambiguous adjacency -> typed ambiguous_snap ------------


def test_ambiguous_zero_match_settlement_is_typed_reject() -> None:
    topo = load_topology()
    # No board corner touches exactly {0,1,2,3}: 0 matches -> ambiguous_snap.
    bad = {
        "ThePhantom": vlm.LocalizedPlayer(
            settlements=(
                vlm.LocalizedSettlement(hexes=(0, 1, 2, 3)),
                vlm.LocalizedSettlement(hexes=topo.vertex_adjacent_hexes[19]),
            ),
            roads=(),
        ),
        "rayman147": vlm.localized_from_ids((11, 3), (19, 8), topo),
    }
    result = vlm.snap_localized_openings(bad, topo)
    assert result.openings is None
    assert result.rejection_reason == "ambiguous_snap:settlement:ThePhantom"


def test_ambiguous_two_match_settlement_is_typed_reject() -> None:
    topo = load_topology()
    # A single-hex adjacency {7} is shared by two outer board-tip vertices: >1 match.
    from collections import Counter as _C

    single_hex_sets = [
        h
        for h, cnt in _C(
            frozenset(adj) for adj in topo.vertex_adjacent_hexes if len(adj) == 1
        ).items()
        if cnt > 1
    ]
    assert single_hex_sets, "expected shared single-hex corner sets on the standard board"
    ambiguous = tuple(next(iter(single_hex_sets)))
    bad = {
        "rayman147": vlm.LocalizedPlayer(
            settlements=(
                vlm.LocalizedSettlement(hexes=ambiguous),
                vlm.LocalizedSettlement(hexes=topo.vertex_adjacent_hexes[3]),
            ),
            roads=(),
        )
    }
    result = vlm.snap_localized_openings(bad, topo)
    assert result.openings is None
    assert result.rejection_reason == "ambiguous_snap:settlement:rayman147"


def test_ambiguous_road_endpoint_is_typed_reject() -> None:
    topo = load_topology()
    bad = {
        "ThePhantom": vlm.LocalizedPlayer(
            settlements=(),
            roads=(vlm.LocalizedRoad(endpoint_a=(0, 2, 3), endpoint_b=(4, 5, 9)),),
        )
    }
    # {0,2,3}=v1 snaps, but {4,5,9} matches no vertex AND no edge joins them anyway.
    result = vlm.snap_localized_openings(bad, topo)
    assert result.openings is None
    assert result.rejection_reason == "ambiguous_snap:road:ThePhantom"


# --- (d) ORDER comes from the LOG, not the localizer ------------------------


def test_order_from_log_not_localizer() -> None:
    topo = load_topology()
    # Feed the localizer rayman's settlements in the OPPOSITE order to the truth
    # (3 then 11); the final record order must still put the GRANTING vertex (11)
    # at settlements[1], driven by the log + grant — never the localizer order.
    reversed_setts = {"ThePhantom": (19, 1), "rayman147": (3, 11)}
    localizer = vlm.MockLocalizer(scripted=_true_localized(reversed_setts))
    localized = localizer.localize(Path("p.png"), Path("b.png"), _board())
    setup_events = vlm.build_setup_events(_LOG_SEQUENCE, _PLAYERS.values())
    result = vlm.snap_and_validate(
        localized=localized,
        board=_board(),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8,),
        winner=None,
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=setup_events,
    )
    assert result.accepted is True
    assert result.record.provenance["placement_order_established"] is True
    assert result.record.openings["rayman147"].settlements[1] == 11
    assert result.record.openings["ThePhantom"].settlements[1] == 19


def test_missing_log_grant_downgrades_order_but_still_accepts() -> None:
    topo = load_topology()
    localizer = vlm.MockLocalizer(scripted=_true_localized())
    localized = localizer.localize(Path("p.png"), Path("b.png"), _board())
    # A setup sequence where NEITHER grant line appears -> log cannot confirm the
    # 2nd-settlement ordinal, so the record is downgraded to order-unestablished
    # (EVAL-excluded) but still ACCEPTED (seed-eligible). Order is the log's job.
    no_grant_sequence = [line for line in _LOG_SEQUENCE if "starting resources" not in line]
    setup_events = vlm.build_setup_events(no_grant_sequence, _PLAYERS.values())
    result = vlm.snap_and_validate(
        localized=localized,
        board=_board(),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8,),
        winner=None,
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=setup_events,
    )
    assert result.accepted is True
    assert result.record.provenance["placement_order_established"] is False
    assert result.record.is_seed_eligible() is True


# --- real-video meta round-trip: grant line survives -> order establishes ----


# The per-game setup LogEvent kinds as ``segment.events`` carries them (real-video
# path). ``prepare_frames_from_video`` serialises these to ``log_setup_sequence``.
_GAME1_SETUP_KINDS: tuple[tuple[str, str], ...] = (
    ("rayman147", "setup_settlement"),
    ("rayman147", "setup_road"),
    ("ThePhantom", "setup_settlement"),
    ("ThePhantom", "setup_road"),
    ("ThePhantom", "setup_settlement"),
    ("ThePhantom", "starting_resources"),
    ("ThePhantom", "setup_road"),
    ("rayman147", "setup_settlement"),
    ("rayman147", "starting_resources"),
    ("rayman147", "setup_road"),
)


def test_real_video_meta_setup_sequence_carries_grant_line() -> None:
    # Emit log_setup_sequence EXACTLY as prepare_frames_from_video now does (canonical
    # phrase per kind, INCLUDING starting_resources) then re-parse via build_setup_events.
    emitted = [
        f"{actor} {vlm._SETUP_PHRASE[kind]}"
        for actor, kind in _GAME1_SETUP_KINDS
        if kind in vlm._SETUP_PHRASE
    ]
    events = vlm.build_setup_events(emitted, _PLAYERS.values())
    kinds = [e.kind for e in events]
    # The grant lines survive the serialization boundary (the bug dropped them).
    assert kinds.count("starting_resources") == 2
    assert [(e.actor, e.kind) for e in events] == list(_GAME1_SETUP_KINDS)


def test_real_video_round_trip_establishes_placement_order() -> None:
    # The regression: the real-video path must be able to establish placement order.
    # Serialize setup events as prepare_frames_from_video does, re-parse, and confirm
    # snap_and_validate stamps placement_order_established (grant read + grant line).
    topo = load_topology()
    localizer = vlm.MockLocalizer(scripted=_true_localized())
    localized = localizer.localize(Path("p.png"), Path("b.png"), _board())
    emitted = [
        f"{actor} {vlm._SETUP_PHRASE[kind]}"
        for actor, kind in _GAME1_SETUP_KINDS
        if kind in vlm._SETUP_PHRASE
    ]
    setup_events = vlm.build_setup_events(emitted, _PLAYERS.values())
    result = vlm.snap_and_validate(
        localized=localized,
        board=_board(),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8,),
        winner=None,
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=setup_events,
    )
    assert result.accepted is True
    assert result.record.provenance["placement_order_established"] is True
    assert result.record.openings["rayman147"].settlements[1] == 11
    assert result.record.openings["ThePhantom"].settlements[1] == 19


# --- (e) joint-flip glyph firewall still guards (a D6-flipped board rejects) --


def test_joint_flip_board_rejected_by_glyph_anchor() -> None:
    topo = load_topology()
    # Feed the D6-flipped (desert=17) openings the fixture note documents as
    # physically wrong; the grant multiset then matches no settlement adjacency, so
    # the NON-OPTIONAL glyph anchor rejects even though every structural gate passes.
    flipped_setts = {"ThePhantom": (4, 10), "rayman147": (20, 0)}
    flipped_roads = {"ThePhantom": (7, 20), "rayman147": (34, 2)}
    localized = {
        handle: vlm.localized_from_ids(flipped_setts[handle], flipped_roads[handle], topo)
        for handle in flipped_setts
    }
    result = vlm.snap_and_validate(
        localized=localized,
        board=_board(desert_hex=11),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8,),
        winner=None,
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=None,
    )
    assert result.accepted is False
    assert result.rejection_reason is not None


# --- score helper ------------------------------------------------------------


def test_score_counts_partial_and_exact() -> None:
    truth = {
        "ThePhantom": vlm.PlayerOpening(settlements=(1, 19), roads=(0, 35)),
        "rayman147": vlm.PlayerOpening(settlements=(11, 3), roads=(19, 8)),
    }
    # One settlement wrong for rayman (3 -> 4), roads intact.
    got = {
        "ThePhantom": vlm.PlayerOpening(settlements=(1, 19), roads=(0, 35)),
        "rayman147": vlm.PlayerOpening(settlements=(11, 4), roads=(19, 8)),
    }
    score = vlm.score_openings(got, truth)
    assert score.settlements_correct == 3
    assert score.settlements_total == 4
    assert score.roads_correct == 4
    assert score.all_exact is False
    assert score.per_player_exact == {"ThePhantom": True, "rayman147": False}


# --- FileLocalizer round-trip ------------------------------------------------


def test_file_localizer_round_trips(tmp_path: Path) -> None:
    topo = load_topology()
    localized = _true_localized()
    payload = {"players": {h: vlm.localized_player_to_payload(p) for h, p in localized.items()}}
    path = tmp_path / "game1__g1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = vlm.FileLocalizer(path).localize(Path("p.png"), Path("b.png"), _board())
    result = vlm.snap_localized_openings(parsed, topo)
    assert result.openings is not None
    assert set(result.openings["rayman147"].settlements) == {11, 3}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_categorize_rejection_buckets() -> None:
    assert vlm.categorize_rejection("ambiguous_snap:settlement:foo") == "ambiguous_snap"
    assert vlm.categorize_rejection("unlocalizable:endgame_stats_overlay") == "board"
    assert vlm.categorize_rejection("glyph_unreadable") == "unreadable"
    assert vlm.categorize_rejection("orientation_joint_flip_glyph_mismatch") == "flip"
    assert vlm.categorize_rejection("affine_residual_exceeded") == "board"
    assert vlm.categorize_rejection("resolution_below_1080p") == "board"
    assert vlm.categorize_rejection("record_contract_violation:x") == "hud"
    assert vlm.categorize_rejection(None) == "hud"


def test_wilson_ci_bounds() -> None:
    assert vlm.wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = vlm.wilson_ci(0, 2)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = vlm.wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_unlocalizable_reason_absent_and_marker(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert vlm._unlocalizable_reason(missing) == "unlocalizable:no_frame"
    marked = tmp_path / "marked.json"
    marked.write_text(json.dumps({"unlocalizable": "endgame_overlay"}), encoding="utf-8")
    assert vlm._unlocalizable_reason(marked) == "unlocalizable:endgame_overlay"
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"players": {}}), encoding="utf-8")
    assert vlm._unlocalizable_reason(real) is None
