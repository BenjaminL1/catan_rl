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
    hue_cluster_margin,
    load_engine_template,
    load_topology,
)
from catan_rl.human_data.board_cv import (
    MIN_DESERT_COVERAGE_MARGIN,
    MIN_HUE_CLUSTER_MARGIN,
    MIN_SCREEN_RULE_GAP,
    _candidate_affines,
    _desert_hex,
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
    pytest.importorskip("cv2")  # load_engine_template reaches cv2 at runtime
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
    pytest.importorskip("cv2")  # load_engine_template reaches cv2 at runtime
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


# --- resource hue-cluster-margin gate (the real resource firewall) ----------


def test_hue_cluster_margin_wide_on_game1() -> None:
    """The game-1 palette has wide inter-cluster hue gaps (BRICK≈9 | WHEAT≈21 |
    SHEEP≈36 | WOOD≈65), so the margin clears :data:`MIN_HUE_CLUSTER_MARGIN` with
    room — the golden board is accepted."""
    margin = hue_cluster_margin(_game1_hsv_samples(), desert_hex=11)
    assert margin > MIN_HUE_CLUSTER_MARGIN


def test_hue_cluster_margin_rejects_near_boundary_swap() -> None:
    """Collapse the WHEAT/SHEEP hue gap (drift a SHEEP hex down to a WHEAT hue) so
    the two clusters touch: the rank-slice would still force a 4/4 multiset (a
    silent WHEAT↔SHEEP swap), but the margin drops below the gate, so
    :func:`read_board` REJECTS the frame rather than confidently mislabel it. This
    is the resource firewall the multiset gate cannot be (BLOCKER)."""
    samples = _game1_hsv_samples()
    # H5 is a real SHEEP (hue 36); pull it down to a WHEAT-ish hue (22) so the
    # WHEAT and SHEEP clusters are no longer separated.
    samples[5, 0] = 22.0
    margin = hue_cluster_margin(samples, desert_hex=11)
    assert margin < MIN_HUE_CLUSTER_MARGIN


def test_hue_cluster_margin_rejects_bad_desert() -> None:
    with pytest.raises(ValueError, match="desert_hex"):
        hue_cluster_margin(_game1_hsv_samples(), desert_hex=19)


# --- relative desert split (no fixed white threshold) -----------------------


def test_desert_hex_relative_bimodal_split() -> None:
    """The desert is the single lowest-white-coverage hex when the split is
    bimodal (18 high, 1 low with a clear margin) — a RELATIVE detector, no fixed
    threshold. Even under a global brightness shift (all coverages scaled) the
    relative split still isolates the same desert."""
    cov = np.full(19, 0.6)
    cov[11] = 0.05  # the token-less desert
    assert _desert_hex(cov) == 11
    # a darker skin lowers every token hex's white coverage, but the desert is
    # still the clear minimum — relative, not absolute.
    darker = cov * 0.5
    assert _desert_hex(darker) == 11


def test_desert_hex_rejects_ambiguous_split() -> None:
    """When no single hex is clearly token-less (two hexes tie low, within
    :data:`MIN_DESERT_COVERAGE_MARGIN`), the detector returns ``None`` — the
    desert stamp (orientation-binding provenance) is rejected rather than guessed."""
    cov = np.full(19, 0.6)
    cov[11] = 0.05
    cov[3] = 0.05 + MIN_DESERT_COVERAGE_MARGIN / 2.0  # a second near-low hex
    assert _desert_hex(cov) is None


# --- screen-rule gap is load-bearing (rejects a near-tie) -------------------


def test_screen_rule_gap_gate_threshold_is_conservative() -> None:
    """The game-1 fixture's screen-rule margin is far above the gate floor, so the
    gate never spuriously rejects the golden board (the slow decode asserts the
    realised ratio > 5.0; the floor is 3.0)."""
    assert MIN_SCREEN_RULE_GAP < 5.0


# --- cross-frame stability gate (§5.2, mandatory) ---------------------------


@pytest.mark.slow
def test_read_board_stable_agrees_across_two_frames() -> None:
    """The mandatory §5.2 cross-frame stability gate: :func:`read_board_stable`
    accepts the game-1 board only because both committed 1080p frames decode to
    the byte-identical map, and returns that shared board."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data import read_board_stable

    frames = []
    for name in _GAME1_FRAMES:
        bgr = cv2.imread(str(_FIXTURES / name))
        assert bgr is not None
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = read_board_stable(frames)
    assert result is not None
    assert list(result.hexes) == list(_GAME1_HEXES)
    assert result.desert_hex == 11


def test_read_board_stable_requires_two_frames() -> None:
    from catan_rl.human_data import read_board_stable

    with pytest.raises(ValueError, match=">= 2 frames"):
        read_board_stable([np.zeros((4, 4, 3), np.uint8)])


# --- rejection DIAGNOSTICS (why a board_unreadable game failed) -----------------
#
# read_board has ~10 independent fail-closed gates and used to return a bare None from
# every one, so a `board_unreadable` game could not say WHICH gate rejected it — the
# same observability hole the grant path had (fixed in 7151308). 6 of 14 games in the
# 8-video sweep died here, so naming the gate is the prerequisite for fixing it.
# Purely diagnostic: the reject behaviour is byte-identical (None either way).


def test_read_board_diag_names_the_token_count_gate() -> None:
    pytest.importorskip("cv2")
    from catan_rl.human_data.board_cv import read_board

    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)  # no number-token disks at all
    diag: dict[str, Any] = {}
    assert read_board(blank, diag=diag) is None
    assert diag["fail"] == "token_count"
    assert diag["n_tokens"] == 0  # and it records HOW MANY it found


@pytest.mark.slow
def test_read_board_diag_is_none_on_success_and_reject_is_unchanged() -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("easyocr")
    from catan_rl.human_data.board_cv import read_board

    bgr = cv2.imread(str(_FIXTURES / "game1_postsetup_t247.png"))
    assert bgr is not None
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    diag: dict[str, Any] = {}
    board = read_board(frame, diag=diag)
    assert board is not None
    assert diag["fail"] is None
    # the diag out-param must not change the RESULT — identical hexes without it
    plain = read_board(frame)
    assert plain is not None
    assert list(plain.hexes) == list(board.hexes)


def test_read_board_stable_diag_list_records_one_entry_per_frame() -> None:
    pytest.importorskip("cv2")
    from catan_rl.human_data.board_cv import read_board_stable

    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diags: list[dict[str, Any]] = []
    assert read_board_stable([blank, blank.copy()], diag_list=diags) is None
    assert len(diags) == 2
    assert all(d["fail"] == "token_count" for d in diags)


# --- off-board token outlier trim (the `board_unreadable` root cause) ------------
#
# MEASURED over all 34 saved frames of the 8-video sweep: EVERY real board token sits
# at <= 5.40 token-diameters from the median token centroid (range 5.30-5.40), while
# the false disks — the on-screen score display's "0" digit in the top-left HUD, which
# the disk detector mistakes for a number token — sit at 10.16-12.96 diameters. Total
# separation, so a cut at 8.0 diameters cannot touch a real token.
#
# The false disk broke boards TWO ways, and both were counted as `board_unreadable`:
#   * 19 tokens -> trips the 16-18 count gate outright (9Sm86ml04aI g2/g5/g6);
#   * 17-18 tokens (the false one replacing an occluded real one) -> PASSES the count
#     gate and then POISONS the RANSAC lattice fit -> residual 33-67px (g4).
# Scale-invariant by construction: the board's own median token diameter sets the unit,
# so it holds at any resolution. Additive-accepting: a no-op on every frame that reads
# today (none has a token beyond 5.40 diameters).


def test_trim_token_outliers_drops_the_off_board_score_digit() -> None:
    from catan_rl.human_data.board_cv import _trim_token_outliers

    # 18 real tokens on a tight cloud (~5 diameters) + 1 far HUD digit (~12 diameters)
    diam = 60.0
    real = [(700.0 + 80 * (i % 5), 500.0 + 80 * (i // 5), diam) for i in range(18)]
    false_disk = (137.0, 133.0, 66.0)  # the "0" of the 0-0 score, top-left
    kept, dropped = _trim_token_outliers([*real, false_disk])
    assert dropped == 1
    assert len(kept) == 18
    assert false_disk not in kept


def test_trim_token_outliers_is_a_noop_on_a_clean_board() -> None:
    from catan_rl.human_data.board_cv import _trim_token_outliers

    diam = 60.0
    real = [(700.0 + 80 * (i % 5), 500.0 + 80 * (i // 5), diam) for i in range(18)]
    kept, dropped = _trim_token_outliers(real)
    assert dropped == 0
    assert kept == real  # byte-identical: every currently-accepted frame is unchanged


def test_trim_token_outliers_refuses_to_drop_too_many() -> None:
    from catan_rl.human_data.board_cv import MAX_TOKEN_OUTLIERS, _trim_token_outliers

    diam = 60.0
    real = [(700.0 + 80 * (i % 5), 500.0 + 80 * (i // 5), diam) for i in range(18)]
    far = [(100.0 + 10 * i, 100.0, diam) for i in range(MAX_TOKEN_OUTLIERS + 2)]
    kept, dropped = _trim_token_outliers([*real, *far])
    # more outliers than the cap means the frame is genuinely wrong — trim NOTHING and
    # let the unchanged count gate reject it (fail closed, never salvage a bad frame)
    assert dropped == 0
    assert len(kept) == len(real) + len(far)
