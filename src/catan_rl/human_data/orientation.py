"""Orientation firewalls for the human-data opening parser.

The board-orientation bug (a D6 flip of board+openings) has two distinct
firewalls, of increasing power:

1. **Provenance orientation-binding** (:meth:`catan_rl.human_data.record.GameRecord.validate`):
   asserts the board CV stage and the openings CV stage agree on which engine
   hex is the desert. This catches a record where only *one* of the two artifacts
   was flipped (the committed game-1 weld bug: board=desert11 / openings=desert17).

2. **Glyph anchor** (this module, :func:`assert_glyph_anchor`): the only check
   that catches a *jointly* flipped board+openings, where stage (1) trivially
   agrees because both stages flipped together. It is **orientation-independent**:
   it ties the parse to an external ground-truth that does not come from the
   affine — the resources the Colonist log says each player was *granted* for
   their 2nd setup settlement. Those granted cards are rendered as small
   **card-icon glyphs** in the log (the text OCR drops them). Under the correct
   orientation, the 2nd-settlement's 3 adjacent-hex resource multiset MUST equal
   the granted-card multiset; a joint D6 flip moves the settlement onto different
   hexes and the multisets diverge.

Scope-lock (brief §6): the pure value contract (``record.py``) stays
topology-free; the topology-aware orientation checks live here.

**FIX 4 status — glyph READER built, scale-up still HARD-GATED on VALIDATION.**
The CHECK (:func:`granted_resources_under_orientation`,
:func:`granted_multiset_matches_a_settlement`) and the glyph *reader*
(:mod:`catan_rl.human_data.glyph_anchor`:
:func:`~catan_rl.human_data.glyph_anchor.classify_granted_glyphs` colour-classifies
the granted resource card icons into a per-player multiset) are both implemented and
tested. The reader is PER-GAME calibrated (a
:class:`~catan_rl.human_data.glyph_anchor.GlyphPalette` derived from the game's own
board tiles via
:func:`~catan_rl.human_data.glyph_anchor.calibrate_glyph_palette`, never a global
hue constant) and BEST-EFFORT / fails closed: an ambiguous / too-dark / near-white /
text-abutting (impure) glyph returns ``None`` rather than a guess, the hue median is
wrap-safe, and a per-game multi-frame CONSENSUS
(:func:`~catan_rl.human_data.glyph_anchor.consensus_granted_glyphs`) rejects a lone
noisy frame. The scale-up gate is NOT unblocked by merely
having a reader — it flips to allowed only when the reader is *validated* on a real
labelled post-grant corpus via
:func:`~catan_rl.human_data.glyph_anchor.validate_glyph_classifier`
(>= ``MIN_VALIDATION_FRAMES`` frames at >= ``MIN_VALIDATION_ACCURACY`` accuracy),
whose result is threaded through
:func:`~catan_rl.human_data.glyph_anchor.glyph_classifier_is_validated` into
:func:`assert_scale_up_orientation_gates`'s ``glyph_classifier_validated`` argument.
The validation harness exists (``scripts/glyph_valset.py`` + the labelled crops
under ``data/human/glyph_valset/``; result artifact
``data/human/glyph_validation.json``), and the batch path
(:func:`catan_rl.human_data.batch.run_batch`) now calls
:func:`assert_scale_up_orientation_gates` **structurally, once per harvest run**
— an absent/failed validation raises before any video is parsed, so the batch can
never run with the glyph anchor unvalidated. The anchor itself is NON-OPTIONAL
per game inside :func:`catan_rl.human_data.validate.cross_check`: an unreadable
grant read is a typed reject
(:data:`~catan_rl.human_data.validate.GLYPH_UNREADABLE_REASON`), and "the anchor
ran for both players" is an explicit precondition of acceptance (expert BLOCKER 1,
2026-07-05).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catan_rl.human_data.record import GameRecord
    from catan_rl.human_data.topology import Topology

#: Minimum capture resolution (brief / FIX 5): 1080p is mandatory; 360-480p OCRs
#: number tokens and log glyphs to garbage and silently corrupts the parse.
MIN_RESOLUTION = 1080

#: Maximum permitted mean affine-fit residual (px) before a frame is rejected
#: (FIX 5 / BLOCKER-2 residual gate). A token detector that drops/adds a token
#: during board animations blows the residual past this and the orientation lock
#: must skip the frame rather than emit a mis-snapped board.
MAX_AFFINE_RESIDUAL_PX = 5.0


def granted_resources_under_orientation(
    settlement_vertex: int, board_resource_by_hex: dict[int, str], topology: Topology
) -> Counter[str]:
    """Resource multiset a settlement at ``settlement_vertex`` grants at setup:
    the (≤3) adjacent-hex resources, ``DESERT`` excluded (a desert grants nothing).

    This is the orientation-DEPENDENT *prediction* side of the glyph anchor: it
    reads off the chosen affine's board + openings IDs. The glyph read is the
    orientation-INDEPENDENT ground truth it is compared against.
    """
    adjacent = topology.vertex_adjacent_hexes[settlement_vertex]
    granted: Counter[str] = Counter()
    for hex_id in adjacent:
        resource = board_resource_by_hex[hex_id]
        if resource != "DESERT":
            granted[resource] += 1
    return granted


def granted_multiset_matches_a_settlement(
    player: str,
    granted: Counter[str],
    record: GameRecord,
    topology: Topology,
) -> bool:
    """True iff the granted-card multiset equals the adjacent-resource multiset of
    *one* of ``player``'s two opening settlements (the 2nd / resource-granting
    settlement, whichever it is) under the record's chosen orientation.

    This is the orientation-independent assertion: a jointly-flipped board+openings
    moves the settlements onto different hexes, so no settlement's adjacency
    matches the externally-read granted cards and this returns False.

    **NECESSARY, NOT SUFFICIENT — the granted multiset is a weak discriminator
    (review finding: glyph-anchor discriminative power).** On the committed game-1
    board only 28 distinct 3-hex adjacency multisets exist across the 54 vertices,
    and 38/54 vertices SHARE their multiset with at least one other vertex (see
    ``test_glyph_anchor_multiset_collision_rate``). So a joint D6 flip that lands a
    settlement on a collision-partner vertex can false-accept on the multiset
    alone. Two mitigations the batch MUST apply before trusting this:

    1. Match only the **2nd (resource-granting) settlement** once the draft order
       tells you which of the two it is — the 1st settlement grants nothing, so
       allowing *either* to match (as this helper does, lacking draft order)
       doubles the collision surface. Pass the record's ``draft_order`` down and
       restrict the match at the call site.
    2. Corroborate with the **number-token adjacency**: grant multisets tie-break
       far better when the adjacent NUMBERS must also match. The glyph anchor is a
       necessary check, not a sufficient one, and must be combined with the
       number-adjacency (tracked as a Stage-2 gate condition).
    """
    board_resource_by_hex = {int(h["hex_id"]): str(h["resource"]) for h in record.hexes}
    for settlement_vertex in record.openings[player].settlements:
        predicted = granted_resources_under_orientation(
            settlement_vertex, board_resource_by_hex, topology
        )
        if predicted == granted:
            return True
    return False


def assert_glyph_anchor(
    record: GameRecord,
    granted_by_player: dict[str, Counter[str]],
    topology: Topology,
) -> None:
    """The orientation-independent firewall (the CHECK side). For each player whose
    granted-card glyphs were read from the log, assert the multiset equals one of
    that player's opening settlements' adjacent-resource multisets under the
    record's orientation. Raises :class:`ValueError` on mismatch.

    ``granted_by_player`` is produced by the (not-yet-validated) glyph classifier;
    until that classifier is validated the batch path is hard-gated by
    :func:`assert_scale_up_orientation_gates` and this is exercised only on the
    game-1 fixture with a hand-verified granted multiset.
    """
    for player, granted in granted_by_player.items():
        if player not in record.openings:
            raise ValueError(f"glyph-anchor player {player!r} has no opening in the record")
        if not granted_multiset_matches_a_settlement(player, granted, record, topology):
            adjacencies = {
                v: dict(
                    granted_resources_under_orientation(
                        v, {int(h["hex_id"]): str(h["resource"]) for h in record.hexes}, topology
                    )
                )
                for v in record.openings[player].settlements
            }
            raise ValueError(
                f"glyph-anchor mismatch for {player!r}: granted cards {dict(granted)} match no "
                f"opening settlement adjacency {adjacencies} under this orientation — the board "
                "and openings may be JOINTLY flipped (provenance-binding cannot see a joint flip)"
            )


class GlyphClassifierNotValidated(NotImplementedError):
    """Raised by the scale-up gate while the log-glyph colour classifier is not yet
    validated. Blocks the 300-game batch from running without the glyph anchor."""


def assert_scale_up_orientation_gates(
    *,
    resolution: int,
    affine_residual_px: float,
    glyph_classifier_validated: bool,
) -> None:
    """HARD GATE for the 300-game batch path (FIX 4 + FIX 5).

    The batch MUST call this before parsing. It enforces, in order:

    - **resolution ≥ 1080** (FIX 5): sub-1080p OCRs to garbage.
    - **mean affine residual ≤ 5 px** (FIX 5): a blown residual = dropped/added
      token during an animation; skip the frame, never emit a mis-snapped board.
    - **glyph classifier validated** (FIX 4): the jointly-flipped-board firewall
      is the glyph anchor. Until the log-glyph colour classifier is validated this
      raises :class:`GlyphClassifierNotValidated` — the batch can never silently
      run with the only joint-flip defense absent.

    Raises :class:`ValueError` (resolution / residual) or
    :class:`GlyphClassifierNotValidated` (glyph) on failure.
    """
    if resolution < MIN_RESOLUTION:
        raise ValueError(
            f"capture resolution {resolution} < required {MIN_RESOLUTION} — sub-1080p OCRs "
            "number tokens and log glyphs to garbage (FIX 5)"
        )
    if affine_residual_px > MAX_AFFINE_RESIDUAL_PX:
        raise ValueError(
            f"affine residual {affine_residual_px:.2f}px > {MAX_AFFINE_RESIDUAL_PX}px — a "
            "dropped/added token; skip the frame rather than emit a mis-snapped board (FIX 5)"
        )
    if not glyph_classifier_validated:
        raise GlyphClassifierNotValidated(
            "the log-glyph resource classifier is not validated, so the jointly-flipped "
            "board+openings firewall (FIX 4 glyph anchor) is absent. The 300-game scale-up is "
            "BLOCKED until the classifier is built + validated and assert_glyph_anchor runs per "
            "game. Build it (a colour-classify of the 'received starting resources' card icons), "
            "validate it on labelled games, then pass glyph_classifier_validated=True."
        )
