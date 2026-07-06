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

from collections import Counter
from typing import Any

import numpy as np
import pytest

from catan_rl.human_data import (
    GameRecord,
    OpeningResult,
    OpponentStrength,
    PlayerOpening,
    load_topology,
)
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.orientation import granted_resources_under_orientation
from catan_rl.human_data.validate import (
    GLYPH_MISMATCH_REASON,
    GLYPH_UNREADABLE_REASON,
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


def _board_read(
    desert_hex: int = 11,
    *,
    hexes: tuple[dict[str, Any], ...] = _GAME1_HEXES,
    residual_px: float = 0.77,
    pip_ok: bool = True,
) -> BoardRead:
    """A cross-frame-stable BoardRead standing in for read_board_stable output."""
    return BoardRead(
        hexes=hexes,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=desert_hex,
        residual_px=residual_px,
        screen_rule_gap=43.5,
        pip_ok=pip_ok,
    )


def _granted_for(vertex: int) -> Counter[str]:
    """The setup granted-card multiset for a settlement at ``vertex`` on the game-1
    board (adjacent-hex resources, DESERT excluded) — the ground truth the glyph
    reader would produce under the correct orientation."""
    topo = load_topology()
    by_hex = {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}
    return granted_resources_under_orientation(vertex, by_hex, topo)


# Readable, orientation-consistent grant reads for BOTH players. The 2nd/
# resource-granting settlements are the hand-verified game-1 grants (test_orientation):
# ThePhantom v19 (SHEEP/ORE/ORE), rayman147 v11 (WHEAT/WOOD/BRICK) — so the openings
# re-order to log-placement order (settlements[1] == granting) and the accepted
# record is placement_order_established (step6 §3.1). The joint-flip firewall is
# NON-OPTIONAL (expert BLOCKER 1): acceptance requires the anchor to run for both
# players, so the default fixture must supply grants that let it run and pass.
_VALID_GRANTS: dict[str, Counter[str] | None] = {
    "ThePhantom": _granted_for(19),
    "rayman147": _granted_for(11),
}


def _cross_check(
    *,
    board: BoardRead | None = None,
    opening_result: OpeningResult | None = None,
    openings_desert_hex: int = 11,
    winner: str | None = "ThePhantom",
    resolution: int = 1080,
    dice_log: tuple[int, ...] = (8, 6, 11, 4),
    granted_by_player: dict[str, Any] | None = _VALID_GRANTS,
) -> CrossCheckResult:
    # The gate reads the residual from the board it is validating (SHOULD-FIX 2); the
    # ``residual_px`` param is cross-asserted equal, so mirror the board here.
    resolved_board = board if board is not None else _board_read()
    return cross_check(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
        board=resolved_board,
        openings_desert_hex=openings_desert_hex,
        opening_result=(
            opening_result
            if opening_result is not None
            else OpeningResult(openings=dict(_GAME1_OPENINGS), rejection_reason=None)
        ),
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        dice_log=dice_log,
        winner=winner,
        resolution=resolution,
        residual_px=resolved_board.residual_px,
        granted_by_player=granted_by_player,
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
    # The residual is read from the board the gate is validating (SHOULD-FIX 2), not
    # a caller scalar — a caller cannot pass a clean residual alongside a board that
    # fit at 42px.
    result = _cross_check(board=_board_read(residual_px=42.0))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "residual" in result.record.rejection_reason


def test_gate_residual_param_must_match_board() -> None:
    # If a caller passes a residual_px that disagrees with board.residual_px it is a
    # caller bug (a stale scalar re-supplied alongside a different board) — the gate
    # fails loud rather than gate on the wrong number.
    with pytest.raises(ValueError, match="residual_px"):
        cross_check(
            video_id="9Sm86ml04aI",
            game_index=1,
            players={"agent": "ThePhantom", "opponent": "rayman147"},
            opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
            board=_board_read(),  # board.residual_px == 0.77
            openings_desert_hex=11,
            opening_result=OpeningResult(openings=dict(_GAME1_OPENINGS), rejection_reason=None),
            draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
            dice_log=(8, 6, 11, 4),
            winner="ThePhantom",
            resolution=1080,
            residual_px=9.9,  # disagrees with the board
            topology=load_topology(),
        )


def test_gate_rejects_pip_count_mismatch() -> None:
    # The independent pip-count corroboration (§5.2): a board whose number tokens
    # fail the pip re-count is rejected even if every other check passes.
    result = _cross_check(board=_board_read(pip_ok=False))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "pip" in result.record.rejection_reason


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


# --- reject path NEVER crashes: sanitize dice_log so the audit row always loads --


def test_reject_empty_dice_with_valid_winner_does_not_raise() -> None:
    # A game rejected for an unrelated reason (blown residual) whose winner IS a valid
    # handle but whose dice_log failed to parse (empty). The record contract forbids
    # an empty dice_log with a winner set; the reject path must sanitize (null the
    # winner) so the §5.6 audit row LOADS instead of raising (BLOCKER 1).
    result = _cross_check(board=_board_read(residual_px=42.0), winner="ThePhantom", dice_log=())
    assert result.accepted is False
    assert "residual" in (result.record.rejection_reason or "")
    # Loadable, not raised.
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


def test_reject_garbage_dice_is_sanitized() -> None:
    # An OCR-misread roll (13) on a game rejected for an unrelated reason (residual):
    # the reject record must filter the garbage token, not raise (BLOCKER 2).
    result = _cross_check(
        board=_board_read(residual_px=42.0), winner="ThePhantom", dice_log=(8, 13, 6)
    )
    assert result.accepted is False
    restored = GameRecord.from_json_line(result.record.to_json_line())
    # The 13 was filtered; 8 and 6 survive; winner kept (dice_log non-empty).
    assert restored.dice_log == (8, 6)
    assert restored.winner == "ThePhantom"


# --- accept path NEVER crashes on finer contract invariants (BLOCKER 3) -------


def test_accept_path_dice_misread_is_rejected_not_raised() -> None:
    # A record that passes the coarse pre-screen but trips the finer dice-log range
    # invariant (a '13' OCR misread) must REJECT, not raise out of cross_check.
    result = _cross_check(dice_log=(8, 6, 13, 4))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "record_contract_violation" in result.record.rejection_reason
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


def test_accept_path_bad_resource_multiset_is_rejected_not_raised() -> None:
    # A resource-multiset misclassification (hex0 SHEEP->WOOD) fails the finer
    # multiset invariant. It must reject with a loadable audit row, not crash — even
    # though the board hexes themselves are contract-invalid (placeholder fallback).
    bad_hexes = ({"hex_id": 0, "resource": "WOOD", "number": 11}, *_GAME1_HEXES[1:])
    result = _cross_check(board=_board_read(hexes=bad_hexes))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


def test_accept_path_settlement_double_snap_is_rejected_not_raised() -> None:
    # A settlement double-snap (two pieces rounded to one vertex) fails the finer
    # distinctness invariant. It must reject, not raise.
    doubled = {
        "ThePhantom": PlayerOpening(settlements=(1, 1), roads=(0, 35)),
        "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
    }
    result = _cross_check(opening_result=OpeningResult(openings=doubled, rejection_reason=None))
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


def test_accept_path_bad_hex_number_is_rejected_not_raised() -> None:
    # An invalid hex number (a 7, illegal on a hex token) fails the finer number
    # invariant. It must reject with a loadable audit row, not crash.
    bad_hexes = ({"hex_id": 0, "resource": "SHEEP", "number": 7}, *_GAME1_HEXES[1:])
    result = _cross_check(board=_board_read(hexes=bad_hexes))
    assert result.accepted is False
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


# --- glyph anchor wired into the accept path (NON-OPTIONAL joint-flip firewall) -


def test_gate_accepts_with_matching_glyph_anchor() -> None:
    # ThePhantom's 2nd settlement is vertex 19, rayman147's is vertex 11; the granted
    # glyphs match their adjacencies under the correct orientation → the joint-flip
    # firewall runs for BOTH players and passes.
    granted: dict[str, Any] = {"ThePhantom": _granted_for(19), "rayman147": _granted_for(11)}
    result = _cross_check(granted_by_player=granted)
    assert result.accepted is True
    assert result.record.passed_crosscheck is True
    assert result.anchor_ran is True
    assert result.anchor_unreadable is False
    assert result.anchor_mismatch is False


def test_gate_rejects_joint_flip_via_glyph_anchor() -> None:
    # A jointly-flipped-but-stable board sails through the desert-binding (both stages
    # flipped together) and read_board_stable (stable across frames). The glyph anchor
    # is the ONLY defense: granted cards that match NO settlement adjacency under this
    # orientation → reject, not silently accept.
    #
    # Simulate the granted glyphs the log reports (from the *true* orientation) while
    # the board+openings are welded to a different orientation: force a granted
    # multiset that matches neither of ThePhantom's opening settlements' adjacencies.
    ph_adj = {tuple(sorted(_granted_for(v).items())) for v in (1, 19)}
    # A deliberately impossible-for-those-settlements grant (3 ORE — no vertex on this
    # board touches 3 ore hexes; certainly not vertices 1/19). rayman147's read stays
    # valid so the game reaches the anchor (an unreadable read would reject earlier).
    impossible = Counter({"ORE": 3})
    assert tuple(sorted(impossible.items())) not in ph_adj
    result = _cross_check(
        granted_by_player={"ThePhantom": impossible, "rayman147": _granted_for(11)}
    )
    assert result.accepted is False
    assert result.record.rejection_reason == GLYPH_MISMATCH_REASON
    assert result.anchor_ran is True
    assert result.anchor_mismatch is True
    assert result.anchor_unreadable is False
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


# --- the firewall is NON-OPTIONAL: unreadable/absent grants REJECT (BLOCKER 1) --


def test_gate_rejects_unreadable_grant_none_read() -> None:
    # A player whose grant read came back None (the glyph reader's honest "could
    # not read") must be a typed REJECT — never an accepted game that silently
    # skipped the only joint-flip defence.
    result = _cross_check(granted_by_player={"ThePhantom": None, "rayman147": _granted_for(11)})
    assert result.accepted is False
    assert result.record.passed_crosscheck is False
    assert result.record.rejection_reason == GLYPH_UNREADABLE_REASON
    assert result.record.is_scoreboard_eligible() is False
    assert result.record.is_seed_eligible() is False
    assert result.anchor_ran is False
    assert result.anchor_unreadable is True
    assert result.anchor_mismatch is False
    # The audit row still loads (§5.6).
    restored = GameRecord.from_json_line(result.record.to_json_line())
    assert restored.passed_crosscheck is False


def test_gate_rejects_absent_grant_reads_entirely() -> None:
    # No grant read at all (granted_by_player=None — the old fail-open path) must
    # now be the same typed reject: the anchor never ran, so nothing is accepted.
    result = _cross_check(granted_by_player=None)
    assert result.accepted is False
    assert result.record.rejection_reason == GLYPH_UNREADABLE_REASON
    assert result.record.is_scoreboard_eligible() is False
    assert result.record.is_seed_eligible() is False
    assert result.anchor_ran is False
    assert result.anchor_unreadable is True


def test_gate_rejects_grant_read_missing_one_player() -> None:
    # A read covering only ONE of the two granting players is not enough — the
    # anchor must run for BOTH players before a game can be accepted.
    result = _cross_check(granted_by_player={"ThePhantom": _granted_for(19)})
    assert result.accepted is False
    assert result.record.rejection_reason == GLYPH_UNREADABLE_REASON
    assert result.anchor_ran is False
    assert result.anchor_unreadable is True


def test_accepted_record_implies_anchor_ran_for_both_players() -> None:
    # "The anchor actually ran" is an explicit precondition of acceptance: every
    # accepted result carries anchor_ran=True (both players' grants were readable
    # and assert_glyph_anchor executed on the assembled record).
    result = _cross_check()
    assert result.accepted is True
    assert result.anchor_ran is True
    assert result.anchor_unreadable is False


def test_coarse_reject_before_glyph_gate_keeps_its_reason() -> None:
    # A game already rejected by a coarse capture gate (sub-1080p) keeps its true
    # reason even when the grants are ALSO unreadable — but the unreadable-coverage
    # telemetry still reports the read (the §5.6 audit reason stays feature-true).
    result = _cross_check(resolution=720, granted_by_player=None)
    assert result.accepted is False
    assert result.record.rejection_reason is not None
    assert "resolution" in result.record.rejection_reason
    assert result.anchor_ran is False
    assert result.anchor_unreadable is True
