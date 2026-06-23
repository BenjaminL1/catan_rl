"""Orientation-firewall tests (FIX 4 glyph anchor + FIX 5 content-lock gates).

The glyph anchor is the only check that catches a JOINTLY-flipped board+openings
(the provenance-binding cannot — both stages flipped together so they still
agree). The glyph *classifier* is deferred (see ``orientation.py`` and the task
report); the CHECK side is tested here on the re-snapped game-1 fixture, and the
deferred classifier is enforced by the scale-up hard gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from catan_rl.human_data import (
    GameRecord,
    GlyphClassifierNotValidated,
    OpponentStrength,
    PlayerOpening,
    assert_glyph_anchor,
    assert_scale_up_orientation_gates,
    granted_multiset_matches_a_settlement,
    granted_resources_under_orientation,
    load_topology,
)

# The re-snapped game-1 board (desert=11) + openings, mirrored from the scaffold
# sample. ThePhantom s[1,19], rayman147 s[11,3].
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


def _record() -> GameRecord:
    return GameRecord(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="known_window", confidence=0.8),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=_GAME1_HEXES,
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        openings={
            "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
            "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
        },
        dice_log=(8, 6, 11, 4),
        winner="ThePhantom",
        episode_source="natural",
        passed_crosscheck=True,
        provenance={
            "resolution": 1080,
            "ts": 247,
            "board_desert_hex": 11,
            "openings_desert_hex": 11,
        },
        rejection_reason=None,
    )


# --- granted-resource prediction (orientation-DEPENDENT CHECK side) ---------


def test_granted_resources_excludes_desert() -> None:
    topo = load_topology()
    board = {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}
    # rayman147 v11 adj hexes 2/9/10 = WHEAT/WOOD/BRICK (no desert).
    assert granted_resources_under_orientation(11, board, topo) == Counter(
        {"WHEAT": 1, "WOOD": 1, "BRICK": 1}
    )
    # ThePhantom v1 adj hexes 0/2/3 = SHEEP/WHEAT/WHEAT.
    assert granted_resources_under_orientation(1, board, topo) == Counter({"SHEEP": 1, "WHEAT": 2})


# --- glyph anchor: the orientation-INDEPENDENT firewall (FIX 4 CHECK) -------


def test_glyph_anchor_accepts_correct_grants() -> None:
    """Hand-verified game-1 grants (a settlement's true adjacency) match under the
    correct desert=11 orientation."""
    topo = load_topology()
    rec = _record()
    granted = {
        # rayman147's resource-granting (2nd) settlement is v11: WHEAT/WOOD/BRICK.
        "rayman147": Counter({"WHEAT": 1, "WOOD": 1, "BRICK": 1}),
        # ThePhantom's v19: SHEEP/ORE/ORE.
        "ThePhantom": Counter({"SHEEP": 1, "ORE": 2}),
    }
    assert granted_multiset_matches_a_settlement("rayman147", granted["rayman147"], rec, topo)
    assert granted_multiset_matches_a_settlement("ThePhantom", granted["ThePhantom"], rec, topo)
    assert_glyph_anchor(rec, granted, topo)  # no raise


def test_glyph_anchor_rejects_jointly_flipped_grants() -> None:
    """A jointly-flipped board+openings moves the settlements onto different hexes,
    so the externally-read granted cards match no settlement adjacency. Simulate
    by feeding grants that belong to neither game-1 settlement."""
    topo = load_topology()
    rec = _record()
    bogus = {"rayman147": Counter({"ORE": 3})}  # no game-1 rayman settlement grants 3 ore
    assert not granted_multiset_matches_a_settlement("rayman147", bogus["rayman147"], rec, topo)
    with pytest.raises(ValueError, match="glyph-anchor mismatch"):
        assert_glyph_anchor(rec, bogus, topo)


def test_glyph_anchor_rejects_a_REAL_d6_joint_flip() -> None:
    """The load-bearing negative: a *real* D6 joint flip, not a fabricated multiset.

    The wrong-orientation (desert=17) game-1 openings are the committed flipped IDs
    (TP s[4,10], ray s[20,0]; see test_road_incidence_is_NOT_an_orientation_check).
    The externally-read granted cards (the orientation-INDEPENDENT log glyphs) are
    the true game-1 grants — TP SHEEP/ORE/ORE (v19), ray WHEAT/WOOD/BRICK (v11). A
    joint flip moves both players' settlements onto hexes whose adjacency diverges
    from the true grants, so the anchor must reject. This proves the anchor catches
    an ACTUAL flip, not just garbage — including the near-miss where flipped ray-v20
    grants SHEEP/ORE/ORE (colliding with the CORRECT *TP* grant): because each
    player's grant is checked only against that player's own flipped settlements,
    the collision does not sneak through.
    """
    topo = load_topology()
    # Re-point the openings to the REJECTED desert=17 wrong-orientation IDs (a D6
    # joint flip of the desert=11 openings) while the board stays the true read.
    rec = replace(
        _record(),
        openings={
            "ThePhantom": PlayerOpening(settlements=(4, 10), roads=(7, 20)),
            "rayman147": PlayerOpening(settlements=(20, 0), roads=(34, 2)),
        },
    )
    # The SAME externally-read true grants (TP v19, ray v11) the correct test uses.
    true_grants = {
        "ThePhantom": Counter({"SHEEP": 1, "ORE": 2}),
        "rayman147": Counter({"WHEAT": 1, "WOOD": 1, "BRICK": 1}),
    }
    assert not granted_multiset_matches_a_settlement(
        "ThePhantom", true_grants["ThePhantom"], rec, topo
    )
    assert not granted_multiset_matches_a_settlement(
        "rayman147", true_grants["rayman147"], rec, topo
    )
    with pytest.raises(ValueError, match="glyph-anchor mismatch"):
        assert_glyph_anchor(rec, true_grants, topo)


# --- scale-up hard gate (FIX 4 deferred classifier + FIX 5 gates) -----------


def test_scale_up_gate_blocks_on_unvalidated_glyph_classifier() -> None:
    with pytest.raises(GlyphClassifierNotValidated, match="glyph anchor"):
        assert_scale_up_orientation_gates(
            resolution=1080, affine_residual_px=0.8, glyph_classifier_validated=False
        )


def test_scale_up_gate_rejects_sub_1080p() -> None:
    with pytest.raises(ValueError, match="resolution"):
        assert_scale_up_orientation_gates(
            resolution=720, affine_residual_px=0.8, glyph_classifier_validated=True
        )


def test_scale_up_gate_rejects_blown_residual() -> None:
    with pytest.raises(ValueError, match="residual"):
        assert_scale_up_orientation_gates(
            resolution=1080, affine_residual_px=42.0, glyph_classifier_validated=True
        )


def test_scale_up_gate_passes_when_all_satisfied() -> None:
    # When a validated classifier exists and the frame is clean, the gate is silent.
    assert_scale_up_orientation_gates(
        resolution=1080, affine_residual_px=0.77, glyph_classifier_validated=True
    )
