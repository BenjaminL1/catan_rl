"""Opening-detector tests (Stage-2 ``openings`` slice, build brief §4/§5).

Two layers, matching the ``board_cv`` convention:

- **Pure geometry** (no ``cv2``): the palette is a per-colour table (not a
  two-colour hardcode); the §5.7 road tiebreak resolves the correct incident edge
  from a synthetic road-pixel cloud and honours the ``used`` claim; the binding /
  count guards in :func:`detect_openings` reject a bad ``player_colors`` up front.

- **Integration** (``slow``, real ``cv2``): :func:`detect_openings` reproduces the
  committed ``game1_openings.json`` EXACTLY off the two committed 1080p frames —
  the re-snapped desert=11 IDs (ThePhantom s[1,19] r[0,35]; rayman147 s[11,3]
  r[19,8]), NOT the rejected-orientation desert=17 IDs — and
  :func:`read_hud_seat_colors` reads GREEN(top)/BLACK(bottom) off the HUD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from catan_rl.human_data.board_cv import BoardRead

from catan_rl.human_data import (
    PALETTE,
    ColorProfile,
    PlayerOpening,
    load_topology,
)
from catan_rl.human_data.openings import _road_for_settlement

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"
_GAME1_FRAME = "game1_postsetup_t247.png"
_GAME1_BASELINE = "game1_empty_baseline_t105.png"

#: The golden game-1 opening (desert=11 re-snap), byte-identical to the committed
#: ``game1_openings.json`` and ``test_scaffold._sample_record()``. GREEN =
#: rayman147, BLACK = ThePhantom (HUD-authoritative, §5.14).
_GAME1_PLAYER_COLORS = {"rayman147": "GREEN", "ThePhantom": "BLACK"}
_GAME1_OPENINGS = {
    "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
    "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
}


# --- pure geometry (no cv2) -------------------------------------------------


def test_palette_is_a_per_color_table() -> None:
    # The palette is a table keyed by colour name, not a two-colour hardcode.
    assert set(PALETTE) >= {"GREEN", "BLACK"}
    for color, profile in PALETTE.items():
        assert isinstance(profile, ColorProfile)
        assert profile.name == color
    # Only GREEN pieces collide with a same-hued tile ⟹ baseline subtraction.
    assert PALETTE["GREEN"].tile_subtract is True
    assert PALETTE["BLACK"].tile_subtract is False


def test_road_tiebreak_picks_the_incident_edge_with_most_road_pixels() -> None:
    """§5.7: among the edges incident to a settlement, the road is the one the
    colour road-pixels lie along. A synthetic cloud placed on the v3→v9 edge (id 8)
    must be picked over the two other v3-incident edges (5, 7)."""
    topo = load_topology()
    vertex_px = _synthetic_lattice()
    p1 = vertex_px[3]
    p2 = vertex_px[9]
    # A dense cloud along the v3→v9 segment midsection.
    ts = np.linspace(0.25, 0.75, 60)
    cloud = np.array([p1 + t * (p2 - p1) for t in ts])
    got = _road_for_settlement(cloud, 3, vertex_px, topo.edge_vertices, used=set())
    assert got == 8


def test_road_tiebreak_respects_used_edges() -> None:
    # If edge 8 is already claimed by the other settlement's road, the tiebreak
    # must not re-award it — a duplicate road edge is a §5.7 double-snap.
    topo = load_topology()
    vertex_px = _synthetic_lattice()
    p1, p2 = vertex_px[3], vertex_px[9]
    cloud = np.array([p1 + t * (p2 - p1) for t in np.linspace(0.25, 0.75, 60)])
    got = _road_for_settlement(cloud, 3, vertex_px, topo.edge_vertices, used={8})
    assert got != 8


def test_road_tiebreak_none_when_no_road_pixels() -> None:
    topo = load_topology()
    vertex_px = _synthetic_lattice()
    empty = np.zeros((0, 2), float)
    assert _road_for_settlement(empty, 3, vertex_px, topo.edge_vertices, used=set()) is None


def test_road_tiebreak_rejects_single_stray_pixel() -> None:
    """BLOCKER (red-team §openings): mirror the settlement head-vote guard on the
    road path. WITHOUT a minimum-pixel floor, ``best_count`` starts at 0 and any
    incident edge with even ONE road-mask pixel wins — the exact "fabricated from
    stray pixels" class guarded for settlements but left unguarded for roads.

    A single road-mask pixel on the midsection of edge 7 (verts 3,4), incident to
    settlement 3, must NOT be accepted as that settlement's road; the correct
    output is ``None`` (road_unresolved), not edge 7."""
    from catan_rl.human_data.openings import _MIN_ROAD_PIXELS

    assert _MIN_ROAD_PIXELS > 1
    topo = load_topology()
    vertex_px = _synthetic_lattice()
    p1, p2 = vertex_px[3], vertex_px[4]  # edge 7
    stray = np.array([p1 + 0.5 * (p2 - p1)])  # one pixel on the midsection
    got = _road_for_settlement(stray, 3, vertex_px, topo.edge_vertices, used=set())
    assert got is None


def test_road_tiebreak_rejects_below_floor_leaked_pixels() -> None:
    """BLOCKER (red-team §openings): a handful of leaked settlement-body/tile-bleed
    pixels on a WRONG incident edge (finding: edge 5 collects ~11px on the real
    frame when the true road is occluded) is below the floor, so the detector
    rejects (``None``) rather than confidently assigning the wrong road. A cloud at
    or above the floor is still resolved."""
    from catan_rl.human_data.openings import _MIN_ROAD_PIXELS

    topo = load_topology()
    vertex_px = _synthetic_lattice()
    p1, p2 = vertex_px[2], vertex_px[3]  # edge 5 (verts 2,3), a WRONG incident edge
    ts = np.linspace(0.3, 0.7, _MIN_ROAD_PIXELS - 1)  # below the floor
    leaked = np.array([p1 + t * (p2 - p1) for t in ts])
    assert _road_for_settlement(leaked, 3, vertex_px, topo.edge_vertices, used=set()) is None
    # At the floor it resolves.
    ts_ok = np.linspace(0.3, 0.7, _MIN_ROAD_PIXELS)
    at_floor = np.array([p1 + t * (p2 - p1) for t in ts_ok])
    assert _road_for_settlement(at_floor, 3, vertex_px, topo.edge_vertices, used=set()) == 5


def test_detect_openings_rejects_bad_player_colors() -> None:
    from catan_rl.human_data import detect_openings
    from catan_rl.human_data.board_cv import BoardRead

    board = BoardRead(
        hexes=(),
        affine=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        vertex_px=_synthetic_lattice(),
        desert_hex=11,
        residual_px=0.5,
        screen_rule_gap=40.0,
        pip_ok=True,
    )
    frame = np.zeros((1080, 1920, 3), np.uint8)
    # one handle, one colour, an unknown colour, and two handles sharing a colour.
    for bad in (
        {"a": "GREEN"},
        {"a": "GREEN", "b": "PURPLE"},
        {"a": "GREEN", "b": "GREEN"},
    ):
        assert detect_openings(frame, frame, board, player_colors=bad) is None


def _blank_board() -> BoardRead:
    from catan_rl.human_data.board_cv import BoardRead

    return BoardRead(
        hexes=(),
        affine=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        vertex_px=_synthetic_lattice(),
        desert_hex=11,
        residual_px=0.5,
        screen_rule_gap=40.0,
        pip_ok=True,
    )


def test_detect_openings_result_carries_typed_rejection_reasons() -> None:
    """§5.6 (finding §3): every rejection branch of ``detect_openings_result``
    must carry a *distinct* ``rejection_reason``, not a bare ``None`` — the batch
    path attributes rejections (green-tile-collision vs HUD-mismatch vs shortfall)
    off this string. A blank frame has no readable HUD, so any two-colour binding
    lands on ``hud_unreadable``; a bad binding lands on ``player_colors_invalid``
    *before* the HUD read."""
    pytest.importorskip("cv2")
    from catan_rl.human_data import detect_openings_result
    from catan_rl.human_data.openings import OpeningResult

    board = _blank_board()
    frame = np.zeros((1080, 1920, 3), np.uint8)

    # player_colors_invalid fires first (before the HUD read).
    for bad in ({"a": "GREEN"}, {"a": "GREEN", "b": "PURPLE"}, {"a": "GREEN", "b": "GREEN"}):
        res = detect_openings_result(frame, frame, board, player_colors=bad)
        assert isinstance(res, OpeningResult)
        assert res.openings is None
        assert res.rejection_reason == "player_colors_invalid"

    # A well-formed two-colour binding on a blank (HUD-less) frame -> hud_unreadable.
    res = detect_openings_result(
        frame, frame, board, player_colors={"ThePhantom": "GREEN", "rayman147": "BLACK"}
    )
    assert res.openings is None
    assert res.rejection_reason == "hud_unreadable"


def _synthetic_lattice() -> np.ndarray:
    """The real game-1 projected vertex pixels (board_cv on the committed frame),
    frozen so the pure road-tiebreak tests need no cv2 decode. Only the handful of
    vertices the tests touch need be exact; the rest are placeholders far away."""
    lattice = np.zeros((54, 2), float)
    # Real game-1 read_board vertex_px for the v3 neighbourhood (edges 5/7/8).
    lattice[2] = (771.20, 449.81)
    lattice[3] = (689.51, 496.69)
    lattice[4] = (689.41, 590.82)
    lattice[9] = (607.92, 449.44)
    return lattice


def _spread_lattice(near_vertex: int, at: tuple[float, float]) -> np.ndarray:
    """A 54-vertex lattice with exactly one vertex at ``at`` and every other vertex
    far away, so a blob near ``at`` votes only for ``near_vertex`` and a blob
    elsewhere collects zero votes (the head-vote-guard counterexample)."""
    lattice = np.zeros((54, 2), float)
    for i in range(54):
        lattice[i] = (2000.0 + 10 * i, 2000.0 + 10 * i)
    lattice[near_vertex] = at
    return lattice


def test_detect_settlements_rejects_zero_vote_blob_no_vertex0_fabrication() -> None:
    """BLOCKER (red-team §1): a blob nowhere near any lattice vertex yields an
    all-zero vote array; WITHOUT the head-vote guard ``argmax`` returns index 0 and
    the blob is fabricated as engine vertex 0. With the guard the blob is dropped,
    so the two-settlement count is short and the detector honestly rejects
    (``None``, ``"shortfall"``) instead of returning ``[7, 0]``."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import _detect_settlements

    vertex_px = _spread_lattice(7, (300.0, 300.0))
    hex_px = np.array([[9000.0, 9000.0]])  # no hex near either blob
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (300, 300), 12, 255, -1)  # real settlement at v7 (many votes)
    cv2.circle(mask, (100, 100), 12, 255, -1)  # far from every vertex -> 0 votes
    heads, reason = _detect_settlements(mask, vertex_px, hex_px, count=2)
    assert heads is None
    assert reason == "shortfall"


def test_detect_settlements_rejects_single_stray_pixel_blob() -> None:
    """BLOCKER (red-team §1): a blob that grazes a vertex by a single stray pixel
    is not a settlement; the guard drops it (a max vote of 1 is below the floor),
    so it can never be snapped to a confidently-wrong vertex."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import _MIN_HEAD_VOTES, _detect_settlements

    assert _MIN_HEAD_VOTES > 1
    vertex_px = _spread_lattice(9, (100.0, 100.0))
    hex_px = np.array([[9000.0, 9000.0]])
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (300, 300), 12, 255, -1)  # one real blob far from v9
    # a >min-area blob (occlusion remnant) whose bulk sits >16px from v9 so only a
    # few graze pixels reach it: 3 head votes, well below _MIN_HEAD_VOTES and above
    # 1 (the finding's ~1-vote occluded-remnant counterexample).
    cv2.circle(mask, (118, 118), 10, 255, -1)
    heads, reason = _detect_settlements(mask, vertex_px, hex_px, count=2)
    # v9 blob is guarded out (<_MIN_HEAD_VOTES votes) -> only one real blob -> reject
    assert heads is None
    assert reason == "shortfall"


def _spread_lattice_multi(placements: dict[int, tuple[float, float]]) -> np.ndarray:
    """A 54-vertex lattice with each named vertex at its given pixel and every
    other vertex far away, so a blob near a placed vertex votes only for it."""
    lattice = np.zeros((54, 2), float)
    for i in range(54):
        lattice[i] = (5000.0 + 13 * i, 5000.0 + 13 * i)
    for vertex, at in placements.items():
        lattice[vertex] = at
    return lattice


def test_detect_settlements_rejects_leak_displacing_occluded_settlement() -> None:
    """BLOCKER (red-team §openings): the head-vote floor does NOT distinguish a real
    settlement from a green-tile-subtraction-leak blob — both collect tens-to-
    hundreds of votes. The detector's only remaining discriminator is that a real
    settlement towers in area over every leak. When a real settlement is occluded a
    leak floats into the top-2, indistinguishable in area from the next leak.

    Two real settlements (v3, v11) are big solid disks; several leaks (v18, v19)
    are small vote-passing blobs of similar area. Occluding the v3 real settlement
    leaves top-2 = {v11 real (big), v18 leak (small)}; the accepted-min area (v18)
    does NOT dominate the next rejected leak (v19), so the area-margin guard fires
    and the detector rejects (``None``, ``"ambiguous"``) instead of emitting the
    confidently-wrong ``[11, 18]`` opening the finding names."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import _MIN_SETTLEMENT_AREA_MARGIN, _detect_settlements

    assert _MIN_SETTLEMENT_AREA_MARGIN > 1.0
    vertex_px = _spread_lattice_multi({11: (200.0, 200.0), 18: (200.0, 400.0), 19: (400.0, 200.0)})
    hex_px = np.array([[9000.0, 9000.0]])  # no hex near any blob
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (200, 200), 20, 255, -1)  # v11 real settlement (big)
    cv2.circle(mask, (200, 400), 11, 255, -1)  # v18 leak (small, vote-passing)
    cv2.circle(mask, (400, 200), 11, 255, -1)  # v19 leak (small, vote-passing)
    # v11 (big) + v18 (small) survive as top-2 by area, but v18 does not dominate
    # v19 -> the leak may have displaced an occluded real settlement -> reject.
    heads, reason = _detect_settlements(mask, vertex_px, hex_px, count=2)
    assert heads is None
    assert reason == "ambiguous"


def test_detect_settlements_accepts_when_settlements_dominate() -> None:
    """The area-margin guard must NOT over-reject: two real settlements that tower
    over every leak in area (the intact-frame case, ~5x margin) are accepted. Two
    big disks (v3, v11) plus a small leak (v18) -> the min accepted area dominates
    the leak by well over the margin -> accept [3, 11]."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import _detect_settlements

    vertex_px = _spread_lattice_multi({3: (200.0, 200.0), 11: (400.0, 400.0), 18: (200.0, 400.0)})
    hex_px = np.array([[9000.0, 9000.0]])
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (200, 200), 24, 255, -1)  # v3 real (big)
    cv2.circle(mask, (400, 400), 22, 255, -1)  # v11 real (big)
    cv2.circle(mask, (200, 400), 10, 255, -1)  # v18 leak (small)
    heads, reason = _detect_settlements(mask, vertex_px, hex_px, count=2)
    assert reason is None
    assert set(heads or []) == {3, 11}


def test_detect_settlements_accepts_exactly_two_vote_passing_blobs() -> None:
    """When exactly ``count`` blobs pass the vote floor there is no rejected
    candidate, so the area-margin is vacuous and both are accepted (the BLACK-seat
    case: two real settlements, no green-tile leaks)."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import _detect_settlements

    vertex_px = _spread_lattice_multi({7: (200.0, 200.0), 33: (400.0, 400.0)})
    hex_px = np.array([[9000.0, 9000.0]])
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (200, 200), 14, 255, -1)
    cv2.circle(mask, (400, 400), 14, 255, -1)
    heads, reason = _detect_settlements(mask, vertex_px, hex_px, count=2)
    assert reason is None
    assert set(heads or []) == {7, 33}


# --- integration: real frame decode (slow, cv2) -----------------------------


@pytest.mark.slow
def test_detect_openings_reproduces_game1_exactly() -> None:
    """The load-bearing test-first guarantee: :func:`detect_openings` reproduces
    the committed ``game1_openings.json`` opening IDs EXACTLY off the two committed
    1080p frames (desert=11 re-snap; NOT the rejected desert=17 IDs)."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import detect_openings, read_board

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    baseline = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_BASELINE)), cv2.COLOR_BGR2RGB)
    board = read_board(frame)
    assert board is not None and board.desert_hex == 11

    got = detect_openings(frame, baseline, board, player_colors=_GAME1_PLAYER_COLORS)
    assert got is not None
    # Order-insensitive on the two placements (settlement/road pairing is by blob
    # area, not draft order); compare as sets against the golden IDs.
    for handle, opening in _GAME1_OPENINGS.items():
        assert set(got[handle].settlements) == set(opening.settlements), handle
        assert set(got[handle].roads) == set(opening.roads), handle

    # Cross-check against the committed golden JSON directly (the true fixture).
    golden = json.loads((_FIXTURES / "game1_openings.json").read_text(encoding="utf-8"))
    assert golden["fit"]["desert_hex"] == board.desert_hex == 11
    for handle in _GAME1_PLAYER_COLORS:
        assert set(got[handle].settlements) == set(golden["openings"][handle]["settlements"])
        assert set(got[handle].roads) == set(golden["openings"][handle]["roads"])


@pytest.mark.slow
def test_detect_settlements_rejects_occluded_green_on_real_frame() -> None:
    """BLOCKER (red-team §openings), on the real unmodified game-1 GREEN frame:
    occlude the real v3 settlement blob (a winning-spot glow / card overlay /
    robber-over-piece — the spike's named failure modes) plus the v15 leak blob and
    confirm the detector does NOT emit the confidently-wrong ``[11, 18]`` opening
    the finding names, but rejects with ``"ambiguous"``. WITHOUT the area-margin
    guard the surviving top-2 by area would be {v11 real, v18 green-tile leak}, both
    past the head-vote floor, yielding a fully-legal but WRONG opening."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import read_board
    from catan_rl.human_data.board_cv import load_engine_template
    from catan_rl.human_data.openings import (
        _HEAD_VOTE_RADIUS_PX,
        _HEX_CENTER_EXCLUSION_PX,
        _MIN_PIECE_AREA,
        PALETTE,
        _color_masks,
        _detect_settlements,
    )

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    baseline = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_BASELINE)), cv2.COLOR_BGR2RGB)
    board = read_board(frame)
    assert board is not None and board.desert_hex == 11
    vertex_px = board.vertex_px
    affine = board.affine
    hex_px = (affine[:, :2] @ load_engine_template().hex_centers.T).T + affine[:, 2]

    h, w = frame.shape[:2]
    hull = cv2.convexHull(vertex_px.astype(np.float32)).astype(np.int32)
    hull_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    board_mask = cv2.erode(hull_mask, np.ones((8, 8), np.uint8))
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    baseline_hsv = cv2.cvtColor(baseline, cv2.COLOR_RGB2HSV)
    piece_mask, _road = _color_masks(frame_hsv, baseline_hsv, board_mask, PALETTE["GREEN"])

    # Intact: the two real GREEN settlements are v3 and v11.
    heads, reason = _detect_settlements(piece_mask, vertex_px, hex_px, count=2)
    assert reason is None
    assert set(heads or []) == {3, 11}

    # Occlude the real v3 blob and the v15 leak (models the finding's break).
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(piece_mask)
    occluded = piece_mask.copy()
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < _MIN_PIECE_AREA:
            continue
        cx, cy = centroids[i]
        if float(np.linalg.norm(hex_px - [cx, cy], axis=1).min()) < _HEX_CENTER_EXCLUSION_PX:
            continue
        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], 1).astype(float)
        dv = np.linalg.norm(vertex_px[:, None, :] - pts[None, :, :], axis=2)
        head = int((dv < _HEAD_VOTE_RADIUS_PX).sum(1).argmax())
        if head in (3, 15):
            occluded[labels == i] = 0

    broke_heads, broke_reason = _detect_settlements(occluded, vertex_px, hex_px, count=2)
    assert broke_heads is None, f"emitted a confidently-wrong opening {broke_heads}"
    assert broke_reason == "ambiguous"


@pytest.mark.slow
def test_read_hud_seat_colors_game1() -> None:
    """§5.14: the per-game HUD seat colours read top→bottom = GREEN, BLACK for
    game 1 (rayman147 top seat / green, ThePhantom bottom seat / black)."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data import read_hud_seat_colors

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    assert read_hud_seat_colors(frame) == ("GREEN", "BLACK")


@pytest.mark.slow
def test_detect_openings_rejects_hud_binding_mismatch() -> None:
    """§5.14: if the passed player→colour binding disagrees with the authoritative
    HUD seat colours (here a swap onto a colour the HUD does not show), reject."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import detect_openings, read_board

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    baseline = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_BASELINE)), cv2.COLOR_BGR2RGB)
    board = read_board(frame)
    assert board is not None
    # Bind ThePhantom to a colour the HUD never shows for this game.
    bad = {"rayman147": "GREEN", "ThePhantom": "PURPLE"}
    assert detect_openings(frame, baseline, board, player_colors=bad) is None


@pytest.mark.slow
def test_detect_openings_rejects_swapped_assignment_with_seat_order() -> None:
    """§5.14 (finding §4): a SET-equal but swapped handle→colour binding must be
    rejected when the seat order is supplied. The HUD reads GREEN(top)/BLACK(bottom)
    = rayman147/ThePhantom; binding rayman147→BLACK, ThePhantom→GREEN is set-equal
    (both colours present) yet inverts ownership, so with ``seat_order`` the
    positional assignment check fires ``hud_assignment_mismatch``. The correct
    binding with the same seat order is accepted."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import detect_openings_result, read_board

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    baseline = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_BASELINE)), cv2.COLOR_BGR2RGB)
    board = read_board(frame)
    assert board is not None
    seat_order = ["rayman147", "ThePhantom"]  # top -> bottom (matches the HUD read)

    swapped = {"rayman147": "BLACK", "ThePhantom": "GREEN"}
    res = detect_openings_result(
        frame, baseline, board, player_colors=swapped, seat_order=seat_order
    )
    assert res.openings is None
    assert res.rejection_reason == "hud_assignment_mismatch"

    correct = detect_openings_result(
        frame, baseline, board, player_colors=_GAME1_PLAYER_COLORS, seat_order=seat_order
    )
    assert correct.rejection_reason is None
    assert correct.openings is not None


@pytest.mark.slow
def test_detect_openings_rejects_non_palette_game_with_named_reason() -> None:
    """§5.6 (finding §5): a game whose seats use colours not in the calibrated
    PALETTE is rejected with a NAMED reason (never a silent mislabel), so the
    palette-coverage yield limit is auditable. A binding naming a non-palette colour
    -> ``player_colors_invalid``; a valid-palette binding the HUD does not show ->
    ``hud_set_mismatch``."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import detect_openings_result, read_board

    frame = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_FRAME)), cv2.COLOR_BGR2RGB)
    baseline = cv2.cvtColor(cv2.imread(str(_FIXTURES / _GAME1_BASELINE)), cv2.COLOR_BGR2RGB)
    board = read_board(frame)
    assert board is not None

    non_palette = detect_openings_result(
        frame, baseline, board, player_colors={"a": "BLUE", "b": "ORANGE"}
    )
    assert non_palette.openings is None
    assert non_palette.rejection_reason == "player_colors_invalid"


@pytest.mark.slow
def test_green_settlement_on_green_baseline_tile_survives_subtraction() -> None:
    """§5.6 (finding §2): a vivid GREEN settlement straddling a green baseline tile
    seam (the real vertex geometry — pieces sit on hex corners between tiles) must
    survive the minimal-kernel tile subtraction, i.e. still register a >min-area
    blob whose head votes for its vertex. Guards against the dilation over-eating a
    legitimate on-green-tile settlement (an asymmetric, feature-correlated loss)."""
    cv2 = pytest.importorskip("cv2")
    from catan_rl.human_data.openings import (
        _MIN_HEAD_VOTES,
        _MIN_PIECE_AREA,
        PALETTE,
        _color_masks,
    )

    h, w = 200, 200
    tile_rgb = (46, 110, 52)  # duller forest/pasture tile green
    piece_rgb = (60, 220, 90)  # vivid flat settlement paint
    board_bg = (200, 180, 120)  # non-green board border (the seam side)
    baseline = np.full((h, w, 3), board_bg, np.uint8)
    baseline[:, :90] = tile_rgb  # green tile occupies x<90; the vertex is on the border
    frame = baseline.copy()
    vx, vy = 100, 100  # vertex on the hex border, adjacent to (not inside) the green tile
    cv2.circle(frame, (vx, vy), 14, piece_rgb, -1)
    board_mask = np.full((h, w), 255, np.uint8)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    baseline_hsv = cv2.cvtColor(baseline, cv2.COLOR_RGB2HSV)

    piece_mask, _road = _color_masks(frame_hsv, baseline_hsv, board_mask, PALETTE["GREEN"])
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(piece_mask)
    vertex_px = np.full((54, 2), 5000.0)
    vertex_px[3] = (float(vx), float(vy))
    survived = False
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < _MIN_PIECE_AREA:
            continue
        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], 1).astype(float)
        dv = np.linalg.norm(vertex_px[:, None, :] - pts[None, :, :], axis=2)
        if int((dv < 16.0).sum(1).max()) >= _MIN_HEAD_VOTES:
            survived = True
    assert survived, "on-green-tile GREEN settlement was eaten by the tile subtraction"
