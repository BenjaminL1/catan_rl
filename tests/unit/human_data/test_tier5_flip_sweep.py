"""Tests for the TIER-5 joint-D6 flip-sweep analysis (``scripts/tier5_flip_sweep.py``).

Pins the reference-record sweep numbers and the game-1 board leakage surface so the
tier5 report's figures are reproducible and a regression in the anchor / D6 tables is
caught. The board-collision figures (28 distinct / 38 colliding) mirror the committed
``test_glyph_anchor_multiset_collision_rate`` on the same board.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "tier5_flip_sweep", _REPO / "scripts" / "tier5_flip_sweep.py"
)
assert _SPEC is not None and _SPEC.loader is not None
t5 = importlib.util.module_from_spec(_SPEC)
sys.modules["tier5_flip_sweep"] = t5
_SPEC.loader.exec_module(t5)

from catan_rl.human_data.record import GameRecord, OpponentStrength, PlayerOpening  # noqa: E402
from catan_rl.human_data.topology import load_topology  # noqa: E402

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


def _reference_record() -> GameRecord:
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
            "placement_order_established": True,
        },
        rejection_reason=None,
    )


def test_reference_flip_sweep_rejects_all_flips_both_modes() -> None:
    topo = load_topology()
    result = t5.flip_sweep([_reference_record()], topo)
    assert result.n_games == 1
    assert result.n_flips_per_game == 11
    # Both players' openings together catch every non-identity joint flip: 0 leakage.
    assert result.reject_either == 11
    assert result.reject_second_only == 11
    assert result.leak_either == 0
    assert result.leak_second_only == 0


def test_reconstructed_grant_equals_granting_settlement_adjacency() -> None:
    topo = load_topology()
    rec = _reference_record()
    grants = t5.reconstruct_true_grants(rec, topo)
    # settlements[1] is the granting settlement; TP s[1,19] -> v19 grants SHEEP+2*ORE.
    assert grants["ThePhantom"] == Counter({"SHEEP": 1, "ORE": 2})
    assert grants["rayman147"] == Counter({"SHEEP": 1, "BRICK": 1, "ORE": 1})


def test_board_leakage_surface_matches_committed_collision_numbers() -> None:
    topo = load_topology()
    surface = t5.board_leakage_surface(_GAME1_HEXES, topo)
    assert surface.desert_hex == 11
    assert surface.n_vertices == 54
    # Same board as test_glyph_anchor_multiset_collision_rate: 28 distinct multisets,
    # 38/54 vertices share a multiset -> the leakage-bounding collision structure.
    assert surface.distinct_multisets == 28
    assert surface.colliding_vertices == 38
    # Moved-settlement leak surface is a strict subset of the moved pairs.
    assert 0 <= surface.leak_pairs <= surface.moved_pairs <= 54 * 11


def test_order_unestablished_record_excluded_from_sweep() -> None:
    topo = load_topology()
    rec = _reference_record()
    unestablished = t5.replace(
        rec, provenance={**rec.provenance, "placement_order_established": False}
    )
    assert t5.flip_sweep([unestablished], topo).n_games == 0
