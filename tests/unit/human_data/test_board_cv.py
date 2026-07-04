"""Board-CV tests (Stage-2 ``board_cv`` slice).

Two layers, matching the ``logparse`` convention (pure core unit-tested; heavy
CV/OCR marked ``slow``):

- **Pure geometry** (no ``cv2`` / ``easyocr``): the committed engine template
  loads at the right shape; the per-game resource classifier generalizes (never a
  hardcoded palette) and reconstructs the game-1 resources from the real sampled
  HSV; the D6 orientation math + screen-space rule pick the unique correct
  orientation and reject a mis-oriented (D6-flipped) fit.

- **Integration** (``slow``, real ``cv2`` + ``easyocr``): :func:`read_board`
  reproduces the game-1 board map (19/19) **byte-identical across the two
  committed 1080p frames**, and rejects a deliberately mis-oriented fit via the
  content-anchor cross-check.

The golden board is the re-snapped game-1 board (desert=11), identical to
``record._GAME1_HEXES`` / ``test_orientation._GAME1_HEXES`` and the spike
``AGREEMENT_TABLE.txt``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catan_rl.human_data import (
    ContentAnchor,
    classify_resources,
    load_engine_template,
    load_topology,
)
from catan_rl.human_data.board_cv import (
    _candidate_affines,
    _score_screen_rule,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"

#: The re-snapped game-1 board (desert=11), byte-identical to the value the two
#: committed 1080p frames must both decode to (spike AGREEMENT_TABLE.txt).
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

_GAME1_FRAMES = ("game1_postsetup_t247.png", "game1_empty_baseline_t105.png")


def _resources_by_hex() -> dict[int, str]:
    return {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}


# --- committed template fixture --------------------------------------------


def test_engine_template_loads_at_standard_shape() -> None:
    tpl = load_engine_template()
    assert tpl.hex_centers.shape == (19, 2)
    assert tpl.vertex_px.shape == (54, 2)
    # The template must be geometrically consistent with the committed topology:
    # every hex-corner vertex sits ~one edge-length from its hex center.
    topo = load_topology()
    for h in range(19):
        for corner in range(6):
            v = topo.hex_corner_to_vertex[h][corner]
            dist = float(np.linalg.norm(tpl.vertex_px[v] - tpl.hex_centers[h]))
            assert 75.0 < dist < 85.0


# --- per-game resource classifier (generalization, no hardcoded palette) ----


def _game1_hsv_samples() -> np.ndarray:
    """The real per-hex median HSV sampled from the committed t=247 frame under
    the locked affine (engine hex order). Pins the classifier's input so the
    generalization test runs WITHOUT cv2 — these are the measured values."""
    # (H, S, V) OpenCV ranges, measured from game1_postsetup_t247.png.
    return np.array(
        [
            [37, 238, 172],  # H0 SHEEP
            [9, 211, 212],  # H1 BRICK
            [21, 203, 230],  # H2 WHEAT
            [21, 207, 234],  # H3 WHEAT
            [9, 210, 220],  # H4 BRICK
            [36, 233, 171],  # H5 SHEEP
            [60, 10, 154],  # H6 ORE
            [67, 209, 137],  # H7 WOOD
            [37, 231, 171],  # H8 SHEEP
            [64, 194, 140],  # H9 WOOD
            [10, 212, 222],  # H10 BRICK
            [26, 98, 196],  # H11 DESERT (token-less)
            [36, 239, 172],  # H12 SHEEP
            [66, 182, 138],  # H13 WOOD
            [21, 205, 230],  # H14 WHEAT
            [48, 10, 153],  # H15 ORE
            [60, 10, 154],  # H16 ORE
            [65, 212, 135],  # H17 WOOD
            [21, 212, 226],  # H18 WHEAT
        ],
        float,
    )


def test_classify_resources_reconstructs_game1_from_real_samples() -> None:
    """The per-game cluster classifier (ORE=lowest-sat, hue-ordered rest) rebuilds
    the exact game-1 resources from the real sampled HSV, with the desert taken as
    the token-less hex — no hardcoded colour thresholds."""
    labels = classify_resources(_game1_hsv_samples(), desert_hex=11)
    expected = [str(h["resource"]) for h in _GAME1_HEXES]
    assert labels == expected
    assert dict(Counter(labels)) == {
        "SHEEP": 4,
        "WHEAT": 4,
        "WOOD": 4,
        "ORE": 3,
        "BRICK": 3,
        "DESERT": 1,
    }


def test_classify_resources_is_palette_agnostic() -> None:
    """Shift the whole palette (add a constant hue + scale saturation) and the
    RELATIVE clustering still recovers the same labels — proof the classifier is
    per-game calibrated, not thresholded on a fixed palette."""
    samples = _game1_hsv_samples()
    shifted = samples.copy()
    shifted[:, 0] = shifted[:, 0] + 5.0  # hue offset (a different render skin)
    shifted[:, 1] = shifted[:, 1] * 0.8  # global saturation scale
    assert classify_resources(shifted, desert_hex=11) == classify_resources(samples, desert_hex=11)


def test_classify_resources_rejects_bad_desert() -> None:
    with pytest.raises(ValueError, match="desert_hex"):
        classify_resources(_game1_hsv_samples(), desert_hex=19)


# --- D6 orientation lock: screen rule picks the unique orientation ----------


def _synthetic_tokens() -> np.ndarray:
    """Build synthetic token pixels by applying a known screen-like affine to the
    engine template hex centers (identity rotation, refl=+1, scale/offset). Under
    this affine the correct orientation is refl=+1 rot=0."""
    tpl = load_engine_template()
    # scale down + translate so H8 lands top-center and H11 rightmost as on screen.
    return tpl.hex_centers.copy()


def test_screen_rule_locks_a_unique_orientation() -> None:
    """The screen-space rule (H8 top-center, H11 rightmost) picks ONE orientation
    with a large penalty gap — the residual is D6-degenerate and cannot, so this
    is the frame-stable lock (build brief §5.2)."""
    tpl = load_engine_template()
    tokens = _synthetic_tokens()  # engine-space == the identity orientation
    candidates = _candidate_affines(tokens, tpl.hex_centers)
    # all 12 candidates fit with ~identical (near-zero) residual → residual cannot
    # disambiguate orientation.
    residuals = sorted(c[3] for c in candidates)
    assert residuals[-1] - residuals[0] < 1.0
    scored = _score_screen_rule(candidates, tokens, tpl.hex_centers)
    best = scored[0]
    second = scored[1]
    # The winning orientation is the identity (refl=+1, rot=0) and it wins by a
    # wide margin over every D6 sibling.
    assert best[1] == 1 and best[2] == 0
    assert second[0] > best[0] + 10.0


def test_screen_rule_rejects_the_mis_oriented_fit() -> None:
    """A deliberately mis-oriented (D6-flipped) affine is NOT the screen-rule
    winner: its H8/H11 do not land top-center/rightmost, so it scores strictly
    worse. This is the geometric half of the mis-orientation rejection (the
    content-anchor half is exercised in the slow integration test)."""
    tpl = load_engine_template()
    tokens = _synthetic_tokens()
    candidates = _candidate_affines(tokens, tpl.hex_centers)
    scored = _score_screen_rule(candidates, tokens, tpl.hex_centers)
    winner_key = (scored[0][1], scored[0][2])
    # Every non-winning D6 orientation (a flip/rotation) scores worse — none ties.
    for penalty, refl, rot, _affine, _resid in scored[1:]:
        assert (refl, rot) != winner_key
        assert penalty > scored[0][0]


# --- integration: real frame decode (slow, cv2 + easyocr) -------------------


@pytest.mark.slow
def test_read_board_reproduces_game1_byte_identical_across_frames() -> None:
    """The load-bearing test-first guarantee: :func:`read_board` decodes the game-1
    board (19/19 hexes, resource + number) BYTE-IDENTICAL across the two committed
    1080p frames, and equal to the golden board (desert=11)."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import read_board

    reads = []
    for name in _GAME1_FRAMES:
        bgr = cv2.imread(str(_FIXTURES / name))
        assert bgr is not None, f"fixture {name} did not load"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = read_board(rgb)
        assert result is not None, f"{name} was rejected"
        reads.append(result)

    for result in reads:
        assert list(result.hexes) == list(_GAME1_HEXES)
        assert result.desert_hex == 11
        assert result.pip_ok
        assert result.residual_px < 5.0
        assert result.screen_rule_gap > 5.0
    # byte-identical across the two frames
    assert list(reads[0].hexes) == list(reads[1].hexes)


@pytest.mark.slow
def test_read_board_rejects_mis_oriented_content_anchor() -> None:
    """A deliberately mis-oriented fit is rejected: feed content anchors that
    correspond to a D6-flipped orientation (the desert=17 wrong IDs) and the
    content-anchor cross-check must return ``None`` rather than emit a flipped
    board. Proves the anchor half of the §5.2 lock."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import read_board

    bgr = cv2.imread(str(_FIXTURES / "game1_postsetup_t247.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Anchors asserting H8 is BRICK#9 (it is really SHEEP#2) — a mis-oriented
    # ground truth. The correct screen-locked board disagrees, so the read is
    # rejected rather than silently relabelled.
    bogus = (
        ContentAnchor(engine_id=8, resource="BRICK", number=9),
        ContentAnchor(engine_id=0, resource="ORE", number=4),
    )
    assert read_board(rgb, content_anchors=bogus) is None
