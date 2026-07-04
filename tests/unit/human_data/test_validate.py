"""Cross-check gate tests (Stage-2 ``validate`` slice, build brief §4 / §5).

``validate.cross_check`` is the single accept/reject gate that fuses the parse
artifacts (a cross-frame-stable :class:`BoardRead`, an :class:`OpeningResult`,
the segment winner + opponent strength) into a :class:`GameRecord` that is either
ACCEPTED (``passed_crosscheck=True``) or REJECTED (``passed_crosscheck=False`` +
a typed ``rejection_reason``) — a rejected game still emits its features for the
§5.6 rejection-bias audit.

The **orientation firewall** the gate enforces is the provenance-binding
``board.desert_hex == openings_desert_hex`` (a D6 flip preserves the
resource/number multisets, so this is the only structural gate that catches a
board/openings weld). Road-incidence is **SANITY-ONLY** — it is D6-invariant and
must NEVER be treated as the orientation gate (proven here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan_rl.human_data import (
    GameRecord,
    OpeningResult,
    OpponentStrength,
    PlayerOpening,
    load_topology,
)
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.validate import (
    CrossCheckResult,
    cross_check,
    road_incidence_offenders,
)

# The re-snapped game-1 board (desert=11) + openings, mirrored from the scaffold
# sample: a legal standard 19-tile board.
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

_GAME1_OPENINGS: dict[str, PlayerOpening] = {
    "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
    "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
}

# The REJECTED desert=17 D6-flipped openings (physically wrong but self-consistent
# under the flipped lattice — road-incidence still holds for all 4 roads).
_GAME1_OPENINGS_FLIPPED: dict[str, PlayerOpening] = {
    "ThePhantom": PlayerOpening(settlements=(4, 10), roads=(7, 20)),
    "rayman147": PlayerOpening(settlements=(20, 0), roads=(34, 2)),
}


def _board_read(desert_hex: int = 11) -> BoardRead:
    """A cross-frame-stable BoardRead standing in for read_board_stable output."""
    return BoardRead(
        hexes=_GAME1_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=desert_hex,
        residual_px=0.77,
        screen_rule_gap=43.5,
        pip_ok=True,
    )


def _cross_check(
    *,
    board: BoardRead | None = None,
    opening_result: OpeningResult | None = None,
    openings_desert_hex: int = 11,
    winner: str | None = "ThePhantom",
    resolution: int = 1080,
    residual_px: float = 0.77,
) -> CrossCheckResult:
    return cross_check(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
        board=board if board is not None else _board_read(),
        openings_desert_hex=openings_desert_hex,
        opening_result=(
            opening_result
            if opening_result is not None
            else OpeningResult(openings=dict(_GAME1_OPENINGS), rejection_reason=None)
        ),
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        dice_log=(8, 6, 11, 4),
        winner=winner,
        resolution=resolution,
        residual_px=residual_px,
        topology=load_topology(),
    )


# --- acceptance: the golden game-1 record passes the gate -------------------


def test_gate_accepts_game_1() -> None:
    result = _cross_check()
    assert result.accepted is True
    assert result.record.passed_crosscheck is True
    assert result.record.rejection_reason is None
    assert result.record.provenance["board_desert_hex"] == 11
    assert result.record.provenance["openings_desert_hex"] == 11
    assert result.record.winner == "ThePhantom"
    # Accepted records are constructible (validate() ran) and scoreboard-eligible.
    assert isinstance(result.record, GameRecord)
    assert result.record.is_scoreboard_eligible() is True


def test_gate_accepts_null_winner_concession() -> None:
    # A resign / cutoff game (winner=None) is accepted (seed) but not scoreboard.
    result = _cross_check(winner=None)
    assert result.accepted is True
    assert result.record.is_scoreboard_eligible() is False
    assert result.record.is_seed_eligible() is True


# --- THE orientation firewall: welded desert11 board / desert17 openings ----


def test_gate_rejects_welded_desert11_desert17() -> None:
    """THE board-orientation weld: board locked desert=11, openings snapped under
    the REJECTED desert=17 orientation. A D6 flip preserves the resource/number
    multisets, so only the board_desert==openings_desert firewall catches it."""
    result = _cross_check(
        opening_result=OpeningResult(openings=dict(_GAME1_OPENINGS_FLIPPED), rejection_reason=None),
        openings_desert_hex=17,
    )
    assert result.accepted is False
    assert result.record.passed_crosscheck is False
    assert result.record.rejection_reason is not None
    assert "orientation" in result.record.rejection_reason
    # Rejected record still emits its features (§5.6 bias audit) and is loadable.
    assert result.record.is_scoreboard_eligible() is False
    assert result.record.is_seed_eligible() is False


# --- road-incidence is D6-INVARIANT (SANITY-ONLY, never the orientation gate) -


def test_road_incidence_is_d6_invariant_sanity_only() -> None:
    """The load-bearing negative: road-incidence PASSES the wrong-orientation
    (desert=17) game-1 openings just as readily as the correct desert=11 ones — it
    is D6-invariant, so it can NEVER be the orientation firewall. Both the correct
    and the jointly-flipped openings are clean under road-incidence."""
    topo = load_topology()
    correct = road_incidence_offenders(_GAME1_OPENINGS, topo.edge_vertices)
    flipped = road_incidence_offenders(_GAME1_OPENINGS_FLIPPED, topo.edge_vertices)
    assert correct == {"ThePhantom": [], "rayman147": []}
    # The D6-flipped openings ALSO pass road-incidence — that is exactly why it is
    # sanity-only and NOT the orientation gate (which is the desert-hex binding).
    assert flipped == {"ThePhantom": [], "rayman147": []}


def test_gate_rejects_isolated_road_snap_error() -> None:
    # An isolated bad road snap (nowhere near its settlement) IS caught by the
    # sanity gate — this is all road-incidence is for.
    bad = {
        "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 50)),  # 50 isolated
        "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
    }
    result = _cross_check(opening_result=OpeningResult(openings=bad, rejection_reason=None))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "road" in result.record.rejection_reason


# --- resolution / residual gates --------------------------------------------


def test_gate_rejects_sub_1080p() -> None:
    result = _cross_check(resolution=720)
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "resolution" in result.record.rejection_reason


def test_gate_rejects_blown_residual() -> None:
    result = _cross_check(residual_px=42.0)
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "residual" in result.record.rejection_reason


# --- winner must be one of the two handles ----------------------------------


def test_gate_rejects_winner_not_in_handles() -> None:
    result = _cross_check(winner="somebody_else")
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "winner" in result.record.rejection_reason


# --- upstream openings rejection is carried through -------------------------


def test_gate_carries_upstream_openings_rejection() -> None:
    # When the openings stage already rejected (no openings), the gate emits a
    # rejected record carrying the upstream reason (§5.6 bias audit).
    result = _cross_check(
        opening_result=OpeningResult(
            openings=None, rejection_reason="settlement_blob_shortfall:GREEN"
        )
    )
    assert result.accepted is False
    assert result.record.rejection_reason == "settlement_blob_shortfall:GREEN"
    assert result.record.passed_crosscheck is False


# --- a rejected record is always a loadable GameRecord (bias audit) ----------


def test_rejected_record_is_a_loadable_game_record() -> None:
    result = _cross_check(resolution=720)
    line = result.record.to_json_line()
    restored = GameRecord.from_json_line(line)
    assert restored.passed_crosscheck is False
    assert restored.rejection_reason is not None
