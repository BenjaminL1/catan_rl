"""The cross-check gate — the single accept/reject decision (Stage-2 ``validate``).

:func:`cross_check` fuses the parse artifacts of one game — a cross-frame-stable
:class:`~catan_rl.human_data.board_cv.BoardRead`, an
:class:`~catan_rl.human_data.openings.OpeningResult`, the segment winner + the
manifest-derived :class:`~catan_rl.human_data.record.OpponentStrength` — into a
single :class:`~catan_rl.human_data.record.GameRecord` that is either **ACCEPTED**
(``passed_crosscheck=True``) or **REJECTED** (``passed_crosscheck=False`` plus a
typed ``rejection_reason``). It rejects **on any disagreement** and **emits the
rejected game anyway** so the §5.6 rejection-bias audit can bucket it — a rejected
game never silently disappears.

**The ORIENTATION firewall is the provenance-binding**
``board.desert_hex == openings_desert_hex`` (build brief §5.2 / FIX 2). A D6 flip
of the board+openings preserves the resource/number multisets, the snake-draft
shape, and road↔settlement incidence, so those are all blind to a board/openings
weld — the ONLY structural signal is that the board CV stage and the openings CV
stage disagree about which engine hex is the desert. :func:`cross_check` binds
them and rejects on mismatch.

**Road-incidence is SANITY-ONLY, never the orientation gate** (build brief §5.2 /
FIX 3). :func:`road_incidence_offenders` is **D6-invariant**: a joint flip
relabels the settlement and road IDs by the same lattice permutation, so
incidence holds for the flipped IDs exactly as for the correct ones (proven in
``test_road_incidence_is_d6_invariant_sanity_only``). It only catches an
*isolated* snap error (a road blob that snapped nowhere near its settlement). It
must never be presented as the orientation defence — that is the desert-hex
binding above.

Every check here is a **pure value / topology cross-check** (no ``gui`` /
training / engine-rule import; scope-lock, build brief §6). The deeper
jointly-flipped-board firewall (the glyph anchor) lives in
:mod:`catan_rl.human_data.orientation`; this gate covers the provenance-binding +
the standard multiset / stability / winner-in-handles / resolution / residual
checks the build brief enumerates, **and it runs the glyph anchor NON-OPTIONALLY**
(expert BLOCKER 1, 2026-07-05): a game whose grant read is absent or unreadable
for either player is REJECTED with :data:`GLYPH_UNREADABLE_REASON` — "the anchor
actually ran for both players" is an explicit precondition of ``accepted=True``
(``CrossCheckResult.anchor_ran``), never a fail-open skip.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.openings import OpeningResult
from catan_rl.human_data.orientation import (
    MAX_AFFINE_RESIDUAL_PX,
    MIN_RESOLUTION,
    assert_glyph_anchor,
)
from catan_rl.human_data.record import (
    PROVENANCE_BOARD_DESERT,
    PROVENANCE_OPENINGS_DESERT,
    VALID_DICE_VALUES,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
)
from catan_rl.human_data.topology import Topology

#: Typed rejection reason for a game whose grant read is absent or unreadable for
#: at least one player. The glyph anchor is the ONLY joint-flip defence, so a game
#: it could not run on is REJECTED — never accepted fail-open (expert BLOCKER 1).
GLYPH_UNREADABLE_REASON = "glyph_unreadable"

#: Typed rejection reason for a glyph-anchor mismatch (the joint-D6-flip catch).
GLYPH_MISMATCH_REASON = "orientation_joint_flip_glyph_mismatch"


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
    """Outcome of :func:`cross_check`.

    ``record`` is ALWAYS a valid :class:`GameRecord` (accepted or rejected) so the
    §5.6 rejection-bias audit can load every game; ``accepted`` mirrors
    ``record.passed_crosscheck``. On rejection ``record.rejection_reason`` is the
    typed reason.

    Firewall telemetry (additive — how often the joint-flip anchor actually
    executed, expert review 2026-07-05):

    - ``anchor_ran`` — :func:`orientation.assert_glyph_anchor` executed for BOTH
      players (it is an explicit precondition of ``accepted=True``; also ``True``
      when the anchor ran and REJECTED the game).
    - ``anchor_unreadable`` — the grant read was absent / ``None`` for at least
      one player (the anchor could not run on this game).
    - ``anchor_mismatch`` — the anchor ran and rejected
      (:data:`GLYPH_MISMATCH_REASON`).
    """

    accepted: bool
    record: GameRecord
    anchor_ran: bool = False
    anchor_unreadable: bool = False
    anchor_mismatch: bool = False


def road_incidence_offenders(
    openings: dict[str, PlayerOpening], edge_vertices: tuple[tuple[int, int], ...]
) -> dict[str, list[int]]:
    """SANITY-ONLY snap check: each opening road must touch one of its owner's
    opening settlements. Returns ``{player: [offending edge_ids]}`` (empty ⟹
    clean).

    **NOT an orientation check.** A D6 board+openings flip relabels the settlement
    *and* road IDs by the same lattice permutation, so road↔settlement incidence
    is D6-invariant — it passes the welded desert17/desert11 record just as readily
    as the correct one. It only catches an *isolated* snap error (a road blob that
    snapped to an edge nowhere near its settlement). The cross-orientation firewall
    is the ``board_desert == openings_desert`` binding in :func:`cross_check`, NOT
    this. Mirrors :func:`catan_rl.human_data.record.check_road_incidence` on a raw
    openings map (so the gate can run it before a :class:`GameRecord` exists).
    """
    offenders: dict[str, list[int]] = {}
    for name, opening in openings.items():
        sset = set(opening.settlements)
        bad = [
            edge_id
            for edge_id in opening.roads
            if edge_vertices[edge_id][0] not in sset and edge_vertices[edge_id][1] not in sset
        ]
        offenders[name] = bad
    return offenders


# A legal standard 19-tile board (correct resource / number multisets, desert at
# hex 11) used as the LAST-RESORT reject-record board when the real ``board.hexes``
# are themselves contract-invalid (a CV multiset / number / hex-id misclassification
# is often the rejection cause). Its only job is to make the §5.6 audit row LOAD —
# the row is ``passed_crosscheck=False`` and carries the true cause in
# ``rejection_reason``, so a placeholder board is never consumed as archetype data.
_PLACEHOLDER_HEXES: tuple[dict[str, object], ...] = (
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
_PLACEHOLDER_DESERT_HEX = 11


def _contract_rejection_reason(exc: ValueError) -> str:
    """Map a contract / glyph ``ValueError`` from the accepted-path construction to a
    typed ``rejection_reason``. The glyph anchor is the joint-flip firewall, so its
    failure gets its own reason (:data:`GLYPH_MISMATCH_REASON`); every other finer
    contract invariant is folded into ``record_contract_violation:{msg}``."""
    msg = str(exc)
    if "glyph-anchor" in msg:
        return GLYPH_MISMATCH_REASON
    return f"record_contract_violation:{msg}"


def _placeholder_openings(players: dict[str, str], topology: Topology) -> dict[str, PlayerOpening]:
    """Structurally-valid stand-in openings for a rejected record whose real
    openings are unusable (upstream CV rejection).

    The board features carry the §5.6 archetype signal; the openings are unknown
    for such a game, so a legal disjoint placeholder lets the record load for the
    bias audit without fabricating a real human opening (the record is
    ``passed_crosscheck=False`` and never seed/scoreboard-eligible, so the
    placeholder is never consumed as data). Built from the topology so both roads
    are genuinely incident and both settlements distinct + disjoint.
    """
    handles = sorted(players.values())
    # Two disjoint settlement vertices + a genuinely-incident road each, disjoint
    # across the two players. Edges 0 and 1 anchor player A; a far pair for player B.
    a0, a1 = topology.edge_vertices[0]
    b0, b1 = topology.edge_vertices[71]
    return {
        handles[0]: PlayerOpening(settlements=(a0, a1), roads=(0, _incident_edge(a1, topology, 0))),
        handles[1]: PlayerOpening(
            settlements=(b0, b1), roads=(71, _incident_edge(b1, topology, 71))
        ),
    }


def _incident_edge(vertex: int, topology: Topology, exclude: int) -> int:
    """The first edge (other than ``exclude``) incident to ``vertex``."""
    for edge_id, (a, b) in enumerate(topology.edge_vertices):
        if edge_id != exclude and (a == vertex or b == vertex):
            return edge_id
    raise ValueError(f"no incident edge for vertex {vertex}")  # pragma: no cover


def cross_check(
    *,
    video_id: str,
    game_index: int,
    players: dict[str, str],
    opponent_strength: OpponentStrength,
    board: BoardRead,
    openings_desert_hex: int,
    opening_result: OpeningResult,
    draft_order: tuple[str, ...],
    dice_log: tuple[int, ...],
    winner: str | None,
    resolution: int,
    residual_px: float,
    topology: Topology,
    granted_by_player: dict[str, Counter[str] | None] | None = None,
    ts: int = 0,
    max_residual_px: float = MAX_AFFINE_RESIDUAL_PX,
    min_resolution: int = MIN_RESOLUTION,
) -> CrossCheckResult:
    """Fuse one game's artifacts into an accepted or rejected :class:`GameRecord`.

    Runs the build-brief cross-checks in order and REJECTS on the first
    disagreement (a rejected game still emits its features for the §5.6 bias
    audit). Checks:

    1. **resolution ≥ 1080** (FIX 5) — sub-1080p OCRs number tokens / log glyphs to
       garbage; ``rejection_reason="resolution_below_1080p"``.
    2. **mean affine residual ≤ 5px** (FIX 5) — a blown residual is a dropped/added
       token during an animation. Gated on ``board.residual_px`` (the diagnostic the
       board that produced these hexes actually fit at), NOT on a caller scalar; the
       ``residual_px`` param is cross-asserted equal to ``board.residual_px`` so the
       two can't silently diverge. ``rejection_reason="affine_residual_exceeded"``.
    3. **pip-count corroboration** — ``board.pip_ok`` (the independent pip-count that
       the number tokens OCR to the standard bag, §5.2); ``rejection_reason=
       "pip_count_mismatch"``.
    4. **upstream openings** — if the openings stage already rejected
       (``opening_result.openings is None``) carry its reason through.
    5. **ORIENTATION firewall** — ``board.desert_hex == openings_desert_hex`` (the
       provenance-binding; the only gate that catches a board/openings weld);
       ``rejection_reason="orientation_mismatch_desert_hex"``.
    6. **winner in handles** — winner is ``None`` or one of the two player handles;
       ``rejection_reason="winner_not_a_player_handle"``.
    7. **road-incidence sanity** — every road touches an owner settlement
       (D6-invariant, so NOT the orientation gate); ``rejection_reason=
       "road_snap_isolated:{player}:{edge}"``.
    8. **grant-read coverage** (expert BLOCKER 1 — the firewall is NON-OPTIONAL):
       ``granted_by_player`` must carry a readable (non-``None``) grant multiset for
       BOTH player handles. A ``None`` / absent read means the glyph anchor cannot
       run, so the game is REJECTED with
       ``rejection_reason=`` :data:`GLYPH_UNREADABLE_REASON` — never accepted
       fail-open (an unreadable grant would otherwise silently skip the only
       joint-flip defence).
    9. **glyph anchor** — the jointly-flipped-board firewall
       (:func:`orientation.assert_glyph_anchor`) runs on the assembled record for
       BOTH players BEFORE it is accepted; "the anchor actually ran" is an explicit
       precondition of ``accepted=True`` (``CrossCheckResult.anchor_ran``). A joint
       D6 flip that the desert-binding cannot see is caught here;
       ``rejection_reason=`` :data:`GLYPH_MISMATCH_REASON`.

    A record that passes every check is built with ``passed_crosscheck=True``. The
    ``GameRecord`` constructor then re-runs its own pure-value :meth:`validate`
    (standard resource/number multisets, distinctness, snake-draft, provenance
    orientation-binding, sub-1080p, dice-log range). Those finer contract invariants
    are a SUPERSET of the coarse ``_first_rejection`` pre-screen, so the accepted-path
    construction is wrapped: a record that passes the coarse gate but trips a finer
    contract invariant (a dice-log OCR misread, a multiset misclassification, a
    settlement double-snap) is REJECTED with
    ``rejection_reason="record_contract_violation:{msg}"`` rather than crashing — the
    §5.6 guarantee that every game emits a loadable audit row holds on every path.
    """
    board_hexes = board.hexes

    if residual_px != board.residual_px:
        # The residual param is redundant with the diagnostic the board carries; a
        # divergence is a caller bug (a stale scalar re-supplied alongside a
        # different board). Fail loud rather than gate on the wrong number.
        raise ValueError(
            f"residual_px={residual_px} disagrees with board.residual_px={board.residual_px}; "
            "the gate reads the residual from the board it is validating (they must match)"
        )

    reason = _first_rejection(
        board=board,
        openings_desert_hex=openings_desert_hex,
        opening_result=opening_result,
        winner=winner,
        players=players,
        resolution=resolution,
        topology=topology,
        max_residual_px=max_residual_px,
        min_resolution=min_resolution,
    )

    # Grant-read coverage (BLOCKER 1): the anchor must be RUNNABLE for BOTH players
    # before a game can be accepted. ``None``/absent reads are an honest "could not
    # read" from the glyph reader — the game is rejected, never accepted fail-open.
    handles = set(players.values())
    readable_grants: dict[str, Counter[str]] = {
        player: granted
        for player, granted in (granted_by_player or {}).items()
        if granted is not None
    }
    anchor_unreadable = not handles <= set(readable_grants)
    if reason is None and anchor_unreadable:
        reason = GLYPH_UNREADABLE_REASON

    anchor_ran = False
    anchor_mismatch = False
    if reason is None:
        assert opening_result.openings is not None  # guaranteed by check (4)
        try:
            record = GameRecord(
                video_id=video_id,
                game_index=game_index,
                players=dict(players),
                opponent_strength=opponent_strength,
                ruleset={"num_players": 2, "win_vp": 15},
                hexes=board_hexes,
                draft_order=draft_order,
                openings=dict(opening_result.openings),
                dice_log=dice_log,
                winner=winner,
                episode_source="natural",
                passed_crosscheck=True,
                provenance={
                    "resolution": resolution,
                    "ts": ts,
                    PROVENANCE_BOARD_DESERT: board.desert_hex,
                    PROVENANCE_OPENINGS_DESERT: openings_desert_hex,
                },
                rejection_reason=None,
            )
            # The jointly-flipped-board firewall (the desert-binding is blind to a
            # JOINT flip). NON-OPTIONAL (BLOCKER 1): the coverage gate above
            # guarantees a readable grant for both players, so the anchor ALWAYS
            # runs on the accept path; a glyph mismatch is a rejection, not an
            # acceptance, and ``anchor_ran`` records that the firewall executed.
            anchor_ran = True
            assert_glyph_anchor(record, readable_grants, topology)
        except ValueError as exc:
            # A finer contract invariant (dice-log range / multiset / distinctness /
            # snake-draft / free-anchor) or the glyph anchor failed. Treat it as a
            # rejection so the §5.6 audit row still loads — NEVER let it crash out of
            # cross_check (the coarse pre-screen is a strict subset of the contract).
            reason = _contract_rejection_reason(exc)
            anchor_mismatch = reason == GLYPH_MISMATCH_REASON
            # ``anchor_ran`` is only true if the anchor was actually invoked: a
            # contract ValueError raised DURING record construction happened before
            # the anchor line, so the anchor never executed for that game.
            anchor_ran = anchor_mismatch
        else:
            return CrossCheckResult(accepted=True, record=record, anchor_ran=True)

    # --- REJECTED: emit a structurally-valid record for the §5.6 bias audit ---
    # Sanitize the fields the record contract would itself reject on, so the
    # rejected game still loads (its BOARD features carry the archetype signal the
    # audit buckets by). The record is passed_crosscheck=False and is never
    # seed / scoreboard-eligible, so the sanitized winner / placeholder openings
    # are never consumed as data — they only make the audit row loadable.
    safe_winner = winner if winner in players.values() else None
    # Sanitize dice_log the SAME way winner/openings/resolution are: the reject
    # record's fields only need to LOAD (the true reason is in rejection_reason),
    # not be faithful. An empty or garbage-token dice_log otherwise trips the record
    # contract (dice range / empty-with-winner) and the audit row would fail to
    # load — dropping the game from the §5.6 pool, and rejection is
    # feature-correlated so the drop would be biased, not random.
    safe_dice_log = tuple(roll for roll in dice_log if roll in VALID_DICE_VALUES)
    if not safe_dice_log:
        # The contract permits an empty dice_log only when winner is None.
        safe_winner = None
    if opening_result.openings is not None and set(opening_result.openings.keys()) == set(
        players.values()
    ):
        safe_openings = dict(opening_result.openings)
    else:
        safe_openings = _placeholder_openings(players, topology)
    # A sub-1080p / residual-blown record must still load: the record contract
    # rejects a sub-1080p resolution stamp, so a rejected record stamps a
    # sentinel-safe resolution (the game is already rejected; the true low value is
    # recorded in the reason string). The desert stamps are bound to the board's
    # OWN desert so the record's free-anchor + orientation-binding both pass.
    safe_resolution = max(resolution, min_resolution)

    def _build(
        hexes: tuple[dict[str, object], ...],
        desert_hex: int,
        openings: dict[str, PlayerOpening],
    ) -> GameRecord:
        return GameRecord(
            video_id=video_id,
            game_index=game_index,
            players=dict(players),
            opponent_strength=opponent_strength,
            ruleset={"num_players": 2, "win_vp": 15},
            hexes=hexes,
            draft_order=draft_order,
            openings=openings,
            dice_log=safe_dice_log,
            winner=safe_winner,
            episode_source="natural",
            passed_crosscheck=False,
            provenance={
                "resolution": safe_resolution,
                "ts": ts,
                PROVENANCE_BOARD_DESERT: desert_hex,
                PROVENANCE_OPENINGS_DESERT: desert_hex,
            },
            rejection_reason=reason,
        )

    try:
        # Preserve the real board (the §5.6 archetype signal) when it loads.
        record = _build(board_hexes, board.desert_hex, safe_openings)
    except ValueError:
        # The rejection cause IS the board (a CV multiset / number / hex-id
        # misclassification): the real hexes don't satisfy the record contract, so
        # they can't be preserved faithfully anyway. Fall back to a canonical legal
        # board + placeholder openings so the audit row still LOADS with its typed
        # rejection_reason — never crash out of cross_check (§5.6 guarantee).
        record = _build(
            _PLACEHOLDER_HEXES,
            _PLACEHOLDER_DESERT_HEX,
            _placeholder_openings(players, topology),
        )
    return CrossCheckResult(
        accepted=False,
        record=record,
        anchor_ran=anchor_ran,
        anchor_unreadable=anchor_unreadable,
        anchor_mismatch=anchor_mismatch,
    )


def _first_rejection(
    *,
    board: BoardRead,
    openings_desert_hex: int,
    opening_result: OpeningResult,
    winner: str | None,
    players: dict[str, str],
    resolution: int,
    topology: Topology,
    max_residual_px: float,
    min_resolution: int,
) -> str | None:
    """Return the first rejection reason, or ``None`` if the game passes the gate.

    Ordered so a coarse capture defect (resolution / residual / pip) is reported
    before the finer structural cross-checks. The residual and pip diagnostics are
    read from ``board`` itself (the board that produced these hexes), not a
    caller-supplied scalar — a caller cannot pass a clean residual alongside a board
    that fit at 30px.
    """
    if resolution < min_resolution:
        return f"resolution_below_1080p:{resolution}"
    if board.residual_px > max_residual_px:
        return f"affine_residual_exceeded:{board.residual_px:.2f}px"
    if not board.pip_ok:
        # The independent pip-count corroboration the brief §5.2 requires as a second
        # OCR anchor: the number tokens must also OCR to the standard bag by pip
        # count, not only by digit read.
        return "pip_count_mismatch"
    if opening_result.openings is None:
        # Carry the upstream openings-stage reason through unchanged (§5.6).
        return opening_result.rejection_reason or "openings_unresolved"
    # ORIENTATION firewall (the only gate that catches a board/openings weld).
    if board.desert_hex != openings_desert_hex:
        return (
            f"orientation_mismatch_desert_hex:board={board.desert_hex}:"
            f"openings={openings_desert_hex}"
        )
    if winner is not None and winner not in players.values():
        return f"winner_not_a_player_handle:{winner}"
    # Road-incidence SANITY (D6-invariant; NOT the orientation gate).
    offenders = road_incidence_offenders(opening_result.openings, topology.edge_vertices)
    for name, bad in offenders.items():
        if bad:
            return f"road_snap_isolated:{name}:{bad[0]}"
    return None
