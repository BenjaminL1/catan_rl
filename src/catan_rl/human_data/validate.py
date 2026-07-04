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
checks the build brief enumerates.
"""

from __future__ import annotations

from dataclasses import dataclass

from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.openings import OpeningResult
from catan_rl.human_data.orientation import MAX_AFFINE_RESIDUAL_PX, MIN_RESOLUTION
from catan_rl.human_data.record import (
    PROVENANCE_BOARD_DESERT,
    PROVENANCE_OPENINGS_DESERT,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
)
from catan_rl.human_data.topology import Topology


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
    """Outcome of :func:`cross_check`.

    ``record`` is ALWAYS a valid :class:`GameRecord` (accepted or rejected) so the
    §5.6 rejection-bias audit can load every game; ``accepted`` mirrors
    ``record.passed_crosscheck``. On rejection ``record.rejection_reason`` is the
    typed reason.
    """

    accepted: bool
    record: GameRecord


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
       token during an animation; ``rejection_reason="affine_residual_exceeded"``.
    3. **upstream openings** — if the openings stage already rejected
       (``opening_result.openings is None``) carry its reason through.
    4. **ORIENTATION firewall** — ``board.desert_hex == openings_desert_hex`` (the
       provenance-binding; the only gate that catches a board/openings weld);
       ``rejection_reason="orientation_mismatch_desert_hex"``.
    5. **winner in handles** — winner is ``None`` or one of the two player handles;
       ``rejection_reason="winner_not_a_player_handle"``.
    6. **road-incidence sanity** — every road touches an owner settlement
       (D6-invariant, so NOT the orientation gate); ``rejection_reason=
       "road_snap_isolated:{player}:{edge}"``.

    A record that passes every check is built with ``passed_crosscheck=True``. The
    ``GameRecord`` constructor then re-runs its own pure-value :meth:`validate`
    (standard resource/number multisets, distinctness, snake-draft, provenance
    orientation-binding, sub-1080p) — so the standard multiset + stability checks
    the brief lists are enforced by the record contract the accepted board flows
    into, not duplicated here.
    """
    board_hexes = board.hexes

    reason = _first_rejection(
        board=board,
        openings_desert_hex=openings_desert_hex,
        opening_result=opening_result,
        winner=winner,
        players=players,
        resolution=resolution,
        residual_px=residual_px,
        topology=topology,
        max_residual_px=max_residual_px,
        min_resolution=min_resolution,
    )

    if reason is None:
        assert opening_result.openings is not None  # guaranteed by check (3)
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
        return CrossCheckResult(accepted=True, record=record)

    # --- REJECTED: emit a structurally-valid record for the §5.6 bias audit ---
    # Sanitize the fields the record contract would itself reject on, so the
    # rejected game still loads (its BOARD features carry the archetype signal the
    # audit buckets by). The record is passed_crosscheck=False and is never
    # seed / scoreboard-eligible, so the sanitized winner / placeholder openings
    # are never consumed as data — they only make the audit row loadable.
    safe_winner = winner if winner in players.values() else None
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
    record = GameRecord(
        video_id=video_id,
        game_index=game_index,
        players=dict(players),
        opponent_strength=opponent_strength,
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=board_hexes,
        draft_order=draft_order,
        openings=safe_openings,
        dice_log=dice_log,
        winner=safe_winner,
        episode_source="natural",
        passed_crosscheck=False,
        provenance={
            "resolution": max(resolution, min_resolution),
            "ts": ts,
            PROVENANCE_BOARD_DESERT: board.desert_hex,
            PROVENANCE_OPENINGS_DESERT: board.desert_hex,
        },
        rejection_reason=reason,
    )
    return CrossCheckResult(accepted=False, record=record)


def _first_rejection(
    *,
    board: BoardRead,
    openings_desert_hex: int,
    opening_result: OpeningResult,
    winner: str | None,
    players: dict[str, str],
    resolution: int,
    residual_px: float,
    topology: Topology,
    max_residual_px: float,
    min_resolution: int,
) -> str | None:
    """Return the first rejection reason, or ``None`` if the game passes the gate.

    Ordered so a coarse capture defect (resolution / residual) is reported before
    the finer structural cross-checks.
    """
    if resolution < min_resolution:
        return f"resolution_below_1080p:{resolution}"
    if residual_px > max_residual_px:
        return f"affine_residual_exceeded:{residual_px:.2f}px"
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
