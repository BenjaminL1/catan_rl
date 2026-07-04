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

import numpy as np
import pytest

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
