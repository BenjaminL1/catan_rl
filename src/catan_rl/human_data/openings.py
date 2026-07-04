"""Opening-placement detector (Stage-2 ``openings`` slice, build brief §4/§5).

Given the orientation-locked board (:class:`~catan_rl.human_data.board_cv.BoardRead`,
its projected 54 ``vertex_px``) plus **two** native-geometry RGB frames of one game
— the **post-setup** frame (the 8 opening pieces down, robber still on the desert,
pre-first-roll) and an **empty-baseline** frame (the same board, no pieces) — this
reads each player's snake-draft opening: **2 settlements → vertex IDs, 2 roads →
edge IDs**, keyed by the per-game player→colour binding.

Why two frames (build brief §2, spike VERDICT caveat 1): a player's piece colour
collides with a same-hued *tile* (GREEN pieces vs the forest/pasture tiles). The
empty baseline supplies a **green-tile subtraction** mask (green present with no
pieces down ⟹ tile, not piece), so only the piece green survives. The incremental
frame-diff approach (build brief §2, caveat 3) is NOT used — it is swamped by the
play-GUI available-spot glow; the post-setup single frame + colour is far cleaner.

Correctness constraints honoured (build brief §5):

- **§5.14 player→colour from the per-game HUD, never a global constant.**
  :func:`read_hud_seat_colors` reads the HUD seat-avatar ring colours (top →
  bottom) for THIS game. :func:`detect_openings_result` requires the HUD to yield
  exactly two DISTINCT seat colours (else ``hud_unreadable``) and that they equal
  the two bound colours as a SET (else ``hud_set_mismatch``). When the caller
  supplies the ``seat_order`` (the top→bottom handle order), it *also* validates
  the handle→colour **assignment** positionally (else ``hud_assignment_mismatch``)
  — a swapped ``{handle: colour}`` binding is set-equal to the truth and would
  otherwise silently invert ownership. Without ``seat_order`` only the set is
  validated (the weaker legacy check), so callers that know the seat order must
  pass it.
- **§5.7 road tiebreak.** A setup settlement + its road fuse into one blob whose
  pixel mass bleeds onto several incident edges; the road is resolved among the
  edges *incident to the owner's settlement* by a colour-specific road mask (the
  vivid road bar, tile-subtracted for green) counted along each full edge segment
  — the correct edge wins, the thin subtraction artefact / adjacent tile loses.
- **§5.7 never trust a snapped piece.** Each opening is exactly 2 settlements + 2
  roads per player (known from the log); a detection yielding a different count,
  a double-snap, an un-voted blob (see :data:`_MIN_HEAD_VOTES`), or a binding that
  disagrees with the HUD is **rejected with a typed reason** (see
  :class:`OpeningResult`), never emitted as a confidently-wrong opening.
- **§5.6 the green-tile subtraction bias is MEASURED, not hidden.** Only GREEN
  pieces collide with a same-hued tile, so a GREEN settlement on a green tile can
  be eaten by the (minimal-kernel) tile subtraction while a BLACK one never is —
  an asymmetric, feature-correlated rejection. This bias is inherent to the same-
  hue collision and cannot be fully removed at the mask level (on real frames
  piece-green and tile-green are near-colocated in HSV), so a shortfall for a
  ``tile_subtract`` colour carries the distinct ``:green_tile_suppressed`` reason
  so the §5.6 per-archetype acceptance-rate audit (batch.py / validate.py) can
  separate it from a generic count shortfall.

**Palette precondition.** The palette is a **per-colour table** (:data:`PALETTE`),
not a two-colour hardcode, but only the calibrated colours (currently
``GREEN``/``BLACK``, the committed spike game's seats) are populated. A game whose
seats use other Colonist colours is **rejected with a named reason**
(``player_colors_invalid`` if the binding names a non-palette colour, or
``hud_unreadable`` if the HUD shows none of the palette colours) — never a silent
mislabel — so the §5.6 audit surfaces the palette-coverage yield limit. Extend
:data:`PALETTE` / :data:`_HUD_RING` with data-derived HSV ranges to widen coverage.

CPU-only; ``cv2`` is imported lazily. Never imports ``gui/`` or the training
path (brief §6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from catan_rl.human_data.board_cv import BoardRead, load_engine_template
from catan_rl.human_data.record import PlayerOpening
from catan_rl.human_data.topology import load_topology

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class OpeningResult:
    """The outcome of :func:`detect_openings`: either the per-player openings, or a
    typed ``rejection_reason`` (build brief §5.6 — every rejected game must emit a
    reason so the per-archetype acceptance-rate audit can bucket it).

    Exactly one of ``openings`` / ``rejection_reason`` is set. ``openings`` is the
    ``{handle: PlayerOpening}`` map on success; ``rejection_reason`` is ``None`` on
    success and a distinct reason string on every rejection branch:

    - ``"player_colors_invalid"`` — ``player_colors`` is not two handles bound to
      two distinct :data:`PALETTE` colours.
    - ``"hud_unreadable"`` — the HUD did not yield exactly two distinct seat colours
      (:func:`read_hud_seat_colors`); the per-game binding cannot be corroborated.
    - ``"hud_set_mismatch"`` — the two HUD seat colours are not the two bound colours
      (a colour named that the HUD never shows).
    - ``"hud_assignment_mismatch"`` — the HUD seat *order* disagrees with the passed
      handle→colour assignment (a swapped binding, §5.14) when a ``seat_order`` was
      supplied to check against.
    - ``"settlement_blob_shortfall:{color}"`` — fewer than two settlement blobs
      survived for a colour. A ``tile_subtract`` colour (GREEN) additionally carries
      the ``:green_tile_suppressed`` suffix so the §5.6 bias audit can separate a
      green-tile-collision suppression from a genuine count shortfall.
    - ``"settlement_double_snap:{color}"`` — the two settlement blobs snapped to one
      vertex (a §5.7 double-snap).
    - ``"road_unresolved:{color}:{settlement}"`` — a settlement had no resolvable
      incident road (§5.7).
    """

    openings: dict[str, PlayerOpening] | None
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class ColorProfile:
    """HSV segmentation profile for one Colonist player colour (OpenCV RGB→HSV
    ranges, hue 0..180). ``piece`` isolates the settlement/city body; ``road``
    the vivid road bar. ``tile_subtract`` marks a colour whose *pieces* collide
    with a same-hued board tile (only GREEN in the standard skin), so the
    empty-baseline tile mask must be subtracted before segmentation."""

    name: str
    piece_lo: tuple[int, int, int]
    piece_hi: tuple[int, int, int]
    road_lo: tuple[int, int, int]
    road_hi: tuple[int, int, int]
    tile_subtract: bool
    #: Broad same-hue tile range (empty-baseline) dilated & subtracted when
    #: ``tile_subtract`` — the forest/pasture green that masquerades as a piece.
    tile_lo: tuple[int, int, int] = (0, 0, 0)
    tile_hi: tuple[int, int, int] = (0, 0, 0)


#: Per-colour segmentation table (build brief §5.13 — a table, not a two-colour
#: hardcode). GREEN pieces collide with the forest/pasture tiles ⟹ the baseline
#: tile-subtraction; BLACK pieces are dark low-value and need no subtraction (the
#: grey robber is excluded separately by the on-hex-centre test). The two ACTIVE
#: colours for any game are whichever the HUD seats show (:func:`read_hud_seat_colors`).
PALETTE: dict[str, ColorProfile] = {
    "GREEN": ColorProfile(
        name="GREEN",
        piece_lo=(35, 70, 70),
        piece_hi=(90, 255, 255),
        road_lo=(40, 150, 120),
        road_hi=(90, 255, 255),
        tile_subtract=True,
        tile_lo=(35, 60, 60),
        tile_hi=(92, 255, 255),
    ),
    "BLACK": ColorProfile(
        name="BLACK",
        piece_lo=(0, 0, 0),
        piece_hi=(179, 95, 88),
        road_lo=(0, 0, 0),
        road_hi=(179, 110, 72),
        tile_subtract=False,
    ),
}

#: HUD seat-avatar ring HSV ranges (RGB→HSV). One per palette colour; the two
#: seats' rings are matched against these to read the per-game player→colour
#: binding off the authoritative HUD (§5.14), never a global constant.
_HUD_RING: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "GREEN": ((40, 120, 90), (90, 255, 255)),
    "BLACK": ((0, 0, 0), (179, 90, 90)),
}

#: Fraction of the frame taken by the bottom-right HUD seat panel (x0, y0) — the
#: two stacked seat-avatar rings live in this region. Screen-space landmark, skin-
#: stable at 1080p (the HUD is a fixed Colonist overlay), independent of board CV.
_HUD_REGION_FRAC: tuple[float, float] = (0.76, 0.78)

#: Minimum piece-blob area (px) at 1080p — smaller connected components are OCR /
#: anti-alias speckle, never a settlement or road piece.
_MIN_PIECE_AREA = 250

#: A blob whose centroid is within this many px of a hex CENTRE sits on the number
#: token / robber, not on a vertex/edge — excluded (spike final_detect).
_HEX_CENTER_EXCLUSION_PX = 18.0

#: Settlement-head vote radius (px): a blob's settlement vertex is the lattice
#: vertex collecting the most blob pixels within this radius.
_HEAD_VOTE_RADIUS_PX = 16.0

#: Minimum head votes a settlement blob's winning vertex must collect (blob pixels
#: within :data:`_HEAD_VOTE_RADIUS_PX` of it). WITHOUT this floor, ``argmax`` on an
#: all-zero vote array silently returns index 0, so a blob that sits nowhere near
#: any lattice vertex is snapped to engine vertex 0 (a fabricated opening), and a
#: blob that grazes a vertex by a single stray pixel is accepted as a settlement
#: (review finding: red-team counterexample). A real setup settlement paints a
#: solid disk on its vertex, so its head collects tens-to-hundreds of votes (the
#: weakest real game-1 settlement head collects 29); this floor rejects the
#: zero-vote fabrication and single-stray-pixel occlusion remnants far below that,
#: turning "confidently wrong" into an honest rejection (``None``).
_MIN_HEAD_VOTES = 10

#: Road tiebreak band: along an incident edge's [lo, hi]·L midsection, count the
#: colour's road-mask pixels within ``_ROAD_PERP_PX`` of the edge line. The
#: midsection (skipping the settlement-fused ends) + perp band isolates the road
#: bar from the settlement body and adjacent tiles (§5.7).
_ROAD_SPAN_LO = 0.2
_ROAD_SPAN_HI = 0.8
_ROAD_PERP_PX = 10.0


def _color_masks(
    frame_hsv: npt.NDArray[np.uint8],
    baseline_hsv: npt.NDArray[np.uint8],
    board_mask: npt.NDArray[np.uint8],
    profile: ColorProfile,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """(piece_mask, road_mask) for one colour, both clipped to the board hull.

    For a ``tile_subtract`` colour the same-hue tile pixels present in the
    empty baseline are dilated and removed from both masks (green pieces vs green
    tiles, §5.13 / spike VERDICT caveat 1)."""
    import cv2

    def _b(bound: tuple[int, int, int]) -> npt.NDArray[np.uint8]:
        return np.array(bound, np.uint8)

    piece = cv2.inRange(frame_hsv, _b(profile.piece_lo), _b(profile.piece_hi))
    road = cv2.inRange(frame_hsv, _b(profile.road_lo), _b(profile.road_hi))
    if profile.tile_subtract:
        # Same-hue tile subtraction is an asymmetric, feature-correlated rejection
        # source (review finding §5.6): a GREEN piece ON a green tile is eaten by
        # this dilated tile mask, so keep the dilation to the MINIMUM that clears
        # the tile's anti-alias fringe (9/11 px, shrunk from 11/13) — a larger
        # kernel over-eats legitimate on-green-tile settlements and inflates the
        # ``:green_tile_suppressed`` rejection rate. The residual bias cannot be
        # removed at the mask level (piece-green and tile-green are near-colocated
        # in HSV on real frames), so it is MEASURED via that rejection reason, not
        # hidden.
        tile = cv2.inRange(baseline_hsv, _b(profile.tile_lo), _b(profile.tile_hi))
        tile_piece = cv2.dilate(tile, np.ones((9, 9), np.uint8))
        tile_road = cv2.dilate(tile, np.ones((11, 11), np.uint8))
        piece = cv2.bitwise_and(piece, cv2.bitwise_not(tile_piece))
        road = cv2.bitwise_and(road, cv2.bitwise_not(tile_road))
    piece = cv2.bitwise_and(piece, board_mask)
    road = cv2.bitwise_and(road, board_mask)
    piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return np.asarray(piece, np.uint8), np.asarray(road, np.uint8)


def _detect_settlements(
    piece_mask: npt.NDArray[np.uint8],
    vertex_px: npt.NDArray[np.float64],
    hex_px: npt.NDArray[np.float64],
    count: int,
) -> tuple[list[int] | None, str | None]:
    """The ``count`` (=2) largest on-board piece blobs → their settlement-head
    vertices as ``(heads, None)``, or ``(None, reason)`` on rejection.

    Each blob's head is the vertex collecting the most blob pixels within
    :data:`_HEAD_VOTE_RADIUS_PX` (a fused settlement+road blob votes for its
    settlement vertex, not the road midpoint). A blob on a hex centre (robber /
    number token) is excluded, as is a blob whose winning vertex collects fewer
    than :data:`_MIN_HEAD_VOTES` votes — WITHOUT that floor an all-zero vote array
    makes ``argmax`` fabricate engine vertex 0, and a single-stray-pixel occlusion
    remnant is accepted as a settlement (review finding: red-team counterexample).
    Such a blob is dropped from candidacy (it is not a piece), so it can never be
    snapped to a confidently-wrong vertex.

    Reasons (suffix ``:double_snap`` is disambiguated by the caller via the
    :class:`OpeningResult` reason strings): ``"shortfall"`` — fewer than ``count``
    real settlement blobs survived (the guard turned a fabricated/occluded piece
    into an honest count shortfall); ``"double_snap"`` — the ``count`` largest
    survivors snapped to fewer than ``count`` distinct vertices (§5.7)."""
    import cv2

    _n, labels, stats, centroids = cv2.connectedComponentsWithStats(piece_mask)
    blobs: list[tuple[int, int]] = []
    for i in range(1, _n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < _MIN_PIECE_AREA:
            continue
        cx, cy = centroids[i]
        if float(np.linalg.norm(hex_px - [cx, cy], axis=1).min()) < _HEX_CENTER_EXCLUSION_PX:
            continue
        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], 1).astype(float)
        dv = np.linalg.norm(vertex_px[:, None, :] - pts[None, :, :], axis=2)
        votes = (dv < _HEAD_VOTE_RADIUS_PX).sum(1)
        # THE guard (review finding: red-team counterexample). A zero-vote blob
        # (nowhere near any vertex) or a single-stray-pixel blob (grazes a vertex)
        # is NOT a settlement; dropping it here avoids the argmax→vertex-0
        # fabrication and the occluded-remnant mis-snap. Compare max BEFORE argmax.
        if int(votes.max()) < _MIN_HEAD_VOTES:
            continue
        head = int(votes.argmax())
        blobs.append((area, head))
    if len(blobs) < count:
        return None, "shortfall"
    blobs.sort(key=lambda z: -z[0])
    heads = [head for _area, head in blobs[:count]]
    if len(set(heads)) != count:
        return None, "double_snap"  # two blobs snapped to one vertex, reject
    return heads, None


def _road_for_settlement(
    road_pts: npt.NDArray[np.float64],
    settlement: int,
    vertex_px: npt.NDArray[np.float64],
    edge_vertices: tuple[tuple[int, int], ...],
    used: set[int],
) -> int | None:
    """The opening road for one settlement (§5.7 road tiebreak): among the edges
    incident to ``settlement`` and not already claimed, pick the one collecting the
    most colour road-mask pixels along its midsection perpendicular band. Returns
    ``None`` if no incident edge collects any road pixels."""
    best: int | None = None
    best_count = 0
    for edge_id, (a, b) in enumerate(edge_vertices):
        if settlement not in (a, b) or edge_id in used:
            continue
        p1 = vertex_px[a]
        p2 = vertex_px[b]
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length == 0.0:
            continue
        u = d / length
        rel = road_pts - p1
        t = rel @ u
        perp = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
        count = int(
            (
                (t > _ROAD_SPAN_LO * length) & (t < _ROAD_SPAN_HI * length) & (perp < _ROAD_PERP_PX)
            ).sum()
        )
        if count > best_count:
            best_count = count
            best = edge_id
    return best


def read_hud_seat_colors(frame_rgb: npt.NDArray[np.uint8]) -> tuple[str, ...]:
    """The HUD seat-avatar ring colours for THIS game, ordered top → bottom
    (build brief §5.14 — the authoritative per-game player→colour source, never a
    global constant).

    Segments the bottom-right HUD seat panel for each :data:`PALETTE` colour's
    ring (:data:`_HUD_RING`); each colour whose largest ring blob clears the seat-
    ring area threshold contributes a seat at its blob's y. Returns the colours
    sorted by screen y (top seat first). The caller pairs this order with the
    draft/seat order to build the ``player_colors`` binding.
    """
    import cv2

    height, width = frame_rgb.shape[:2]
    x0 = int(_HUD_REGION_FRAC[0] * width)
    y0 = int(_HUD_REGION_FRAC[1] * height)
    hsv = cv2.cvtColor(frame_rgb[y0:height, x0:width], cv2.COLOR_RGB2HSV)
    # Seat-ring area floor: the avatar ring is a substantial blob; speckle in the
    # HUD (card pips, timer) is far smaller. 400px at 1080p separates them.
    seat_ring_area = 400
    seats: list[tuple[float, str]] = []
    for color, (lo, hi) in _HUD_RING.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        _n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        best_area = 0
        best_y = 0.0
        for i in range(1, _n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_y = float(centroids[i][1])
        if best_area >= seat_ring_area:
            seats.append((best_y, color))
    seats.sort(key=lambda z: z[0])
    return tuple(color for _y, color in seats)


def detect_openings_result(
    frame_rgb: npt.NDArray[np.uint8],
    empty_baseline_rgb: npt.NDArray[np.uint8],
    board: BoardRead,
    *,
    player_colors: dict[str, str],
    seat_order: Sequence[str] | None = None,
) -> OpeningResult:
    """Detect the 2-settlement + 2-road snake-draft opening per player, keyed by the
    per-game ``player_colors`` binding (``{handle: colour}``), returning an
    :class:`OpeningResult` that carries either the openings **or** a distinct
    ``rejection_reason`` on every rejection branch (build brief §5.6 — the batch
    path needs the reason to compute the per-archetype acceptance rate the bias
    audit gates on; a bare ``None`` throws that signal away).

    ``frame_rgb`` is the post-setup RGB frame (8 pieces down, robber on the desert,
    pre-first-roll); ``empty_baseline_rgb`` the same board with no pieces (the
    green-tile subtraction source, §5.13); ``board`` the orientation-locked
    :class:`~catan_rl.human_data.board_cv.BoardRead` (its projected ``vertex_px``
    and ``affine`` supply the lattice / hex centres).

    ``seat_order`` (optional) is the handle order top→bottom (the seat / draft
    order). When supplied, the HUD is checked for the full **assignment** — each
    handle's bound colour must equal the HUD's positional colour at that seat —
    not merely the colour *set* (§5.14: a swapped ``{handle: colour}`` binding is
    set-equal to the truth and would otherwise pass, silently inverting every
    per-player scoreboard row). When absent, only the colour set is validated (the
    weaker legacy check) — callers that know the seat order should pass it.

    Rejection reasons (see :class:`OpeningResult`): ``player_colors_invalid``,
    ``hud_unreadable``, ``hud_set_mismatch``, ``hud_assignment_mismatch``,
    ``settlement_blob_shortfall:{color}[:green_tile_suppressed]``,
    ``settlement_double_snap:{color}``, ``road_unresolved:{color}:{settlement}``.

    Vertex/edge IDs are engine integer IDs under the SAME orientation ``board`` was
    locked at, so ``provenance.openings_desert_hex`` == ``board.desert_hex`` by
    construction (the schema-v2 orientation-binding; :meth:`GameRecord.validate`).
    """
    import cv2

    colors = list(player_colors.values())
    if len(player_colors) != 2 or len(set(colors)) != 2 or any(c not in PALETTE for c in colors):
        return OpeningResult(None, "player_colors_invalid")

    # §5.14: corroborate the bound colours against the authoritative HUD.
    hud = read_hud_seat_colors(frame_rgb)
    # The HUD must yield exactly two DISTINCT seat colours; anything else is an
    # unreadable HUD (not this game's palette / a spurious hit), a distinct cause
    # from a genuine binding mismatch (finding: HUD-unreadable vs mismatch).
    if len(hud) != 2 or len(set(hud)) != 2:
        return OpeningResult(None, "hud_unreadable")
    if set(hud) != set(colors):
        return OpeningResult(None, "hud_set_mismatch")
    # §5.14 assignment check: validate the handle→colour ASSIGNMENT, not just the
    # set — a swapped binding is set-equal but mislabels ownership. Only possible
    # when the caller supplies the seat order the HUD's positional read pairs with.
    # ``seat_order`` must name exactly the two bound handles (top→bottom), and each
    # seat's bound colour must equal the HUD's colour at that seat.
    if seat_order is not None:
        if set(seat_order) != set(player_colors) or len(seat_order) != len(hud):
            return OpeningResult(None, "hud_assignment_mismatch")
        for seat_idx, handle in enumerate(seat_order):
            if player_colors.get(handle) != hud[seat_idx]:
                return OpeningResult(None, "hud_assignment_mismatch")

    topology = load_topology()
    edge_vertices = topology.edge_vertices
    template = load_engine_template()
    affine = board.affine
    hex_px: npt.NDArray[np.float64] = (affine[:, :2] @ template.hex_centers.T).T + affine[:, 2]
    vertex_px = board.vertex_px

    frame_hsv: npt.NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV), np.uint8
    )
    baseline_hsv: npt.NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(empty_baseline_rgb, cv2.COLOR_RGB2HSV), np.uint8
    )
    height, width = frame_rgb.shape[:2]
    hull = cv2.convexHull(vertex_px.astype(np.float32)).astype(np.int32)
    hull_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    board_mask: npt.NDArray[np.uint8] = np.asarray(
        cv2.erode(hull_mask, np.ones((8, 8), np.uint8)), np.uint8
    )

    openings: dict[str, PlayerOpening] = {}
    for handle, color in player_colors.items():
        profile = PALETTE[color]
        piece_mask, road_mask = _color_masks(frame_hsv, baseline_hsv, board_mask, profile)
        settlements, seat_reason = _detect_settlements(piece_mask, vertex_px, hex_px, count=2)
        if settlements is None:
            if seat_reason == "double_snap":
                return OpeningResult(None, f"settlement_double_snap:{color}")
            # A shortfall for a tile_subtract colour (GREEN) is flagged distinctly
            # so the §5.6 audit can separate a green-tile-collision suppression
            # (asymmetric, feature-correlated) from a generic count shortfall.
            suffix = ":green_tile_suppressed" if profile.tile_subtract else ""
            return OpeningResult(None, f"settlement_blob_shortfall:{color}{suffix}")
        ys, xs = np.where(road_mask > 0)
        road_pts = np.stack([xs, ys], 1).astype(float)
        used: set[int] = set()
        roads: list[int] = []
        for settlement in settlements:
            edge_id = _road_for_settlement(road_pts, settlement, vertex_px, edge_vertices, used)
            if edge_id is None:
                return OpeningResult(None, f"road_unresolved:{color}:{settlement}")
            used.add(edge_id)
            roads.append(edge_id)
        openings[handle] = PlayerOpening(settlements=tuple(settlements), roads=tuple(roads))
    return OpeningResult(openings, None)


def detect_openings(
    frame_rgb: npt.NDArray[np.uint8],
    empty_baseline_rgb: npt.NDArray[np.uint8],
    board: BoardRead,
    *,
    player_colors: dict[str, str],
    seat_order: Sequence[str] | None = None,
) -> dict[str, PlayerOpening] | None:
    """Back-compat thin wrapper over :func:`detect_openings_result`: the openings
    ``{handle: PlayerOpening}`` on success, or ``None`` on any rejection (the
    reason is discarded). Callers that need the §5.6 ``rejection_reason`` (batch.py)
    must call :func:`detect_openings_result` directly. See that function for the
    ``seat_order`` §5.14 assignment check and the full rejection-reason vocabulary.
    """
    return detect_openings_result(
        frame_rgb,
        empty_baseline_rgb,
        board,
        player_colors=player_colors,
        seat_order=seat_order,
    ).openings
