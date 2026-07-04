"""Glyph-classifier tests (the READER side of the FIX-4 glyph anchor).

The glyph anchor is the only firewall that catches a JOINTLY-flipped board+openings
(:mod:`catan_rl.human_data.orientation` §2). ``orientation.py`` owns the CHECK (does
the granted multiset match a settlement adjacency under the affine?) and the
scale-up HARD GATE; :mod:`catan_rl.human_data.glyph_anchor` owns the READER that
colour-classifies the granted resource card glyphs into that multiset.

Two layers:

- **classify accept/reject** on card-glyph swatches: the classifier maps each
  granted resource's canonical card colour to the right literal and, critically,
  FAILS CLOSED (returns ``None``) on an ambiguous / too-dark swatch — the
  BEST-EFFORT honesty the task spec requires (never guess an unreadable glyph).
- **the scale-up gate flips to allowed ONLY when validated**: a passing
  :class:`GlyphValidation` flips
  :func:`~catan_rl.human_data.orientation.assert_scale_up_orientation_gates` to
  silent; anything less (too few frames, below the accuracy bar, or no validation
  at all) keeps it raising :class:`GlyphClassifierNotValidated`.

The card-glyph swatches here are synthesised at the module's calibrated card hues
(``RESOURCE_CARD_HUES``) — a palette-faithful stand-in for the ~14px log icons,
since the committed ``game1_log_crop_t120.png`` fixture is captured BEFORE any
``"received starting resources"`` event and has no card glyphs (see orientation.py
FIX-4 status note). They exercise the classifier's colour logic and the
validation→gate wiring end-to-end; a real labelled post-grant corpus is what a
future session feeds :func:`validate_glyph_classifier` to flip the batch gate.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from catan_rl.human_data import (
    GlyphClassifierNotValidated,
    assert_scale_up_orientation_gates,
)
from catan_rl.human_data.glyph_anchor import (
    HUE_RESOURCES_BY_RANK,
    MIN_GRANT_CONSENSUS_FRAMES,
    MIN_VALIDATION_FRAMES,
    ORE_MAX_SATURATION,
    RESOURCE_CARD_HUES,
    GlyphValidation,
    LabeledGrantFrame,
    calibrate_glyph_palette,
    classify_glyph,
    classify_granted_glyphs,
    consensus_granted_glyphs,
    glyph_classifier_is_validated,
    validate_glyph_classifier,
)


def _hsv_to_rgb_swatch(hue: float, sat: float, val: float, size: int = 14) -> np.ndarray:
    """A ``size``x``size`` RGB card-glyph swatch of a single HSV colour."""
    import cv2

    hsv = np.full((size, size, 3), (hue, sat, val), np.uint8)
    return np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), np.uint8)


# --- classify_glyph: accept the calibrated card colours ---------------------


@pytest.mark.parametrize("resource", ["WOOD", "BRICK", "WHEAT", "SHEEP"])
def test_classify_glyph_accepts_hue_resources(resource: str) -> None:
    # A vivid swatch at a resource's calibrated card hue classifies to it.
    hsv = (RESOURCE_CARD_HUES[resource], 200.0, 210.0)
    assert classify_glyph(hsv) == resource


def test_classify_glyph_accepts_ore_as_desaturated_grey() -> None:
    # ORE is the low-saturation grey card branch (mirrors board_cv ORE-by-sat).
    assert classify_glyph((0.0, ORE_MAX_SATURATION - 20.0, 180.0)) == "ORE"


# --- classify_glyph: FAIL CLOSED on unreadable / ambiguous glyphs -----------


def test_classify_glyph_rejects_dark_swatch() -> None:
    # A near-black swatch is log text / background bleed, not a card icon.
    assert classify_glyph((25.0, 200.0, 20.0)) is None


def test_classify_glyph_rejects_ambiguous_between_families() -> None:
    # A hue exactly between WHEAT (25) and SHEEP (40) centres is within the margin
    # of BOTH — it must fail closed, never be guessed toward one family.
    midpoint = (RESOURCE_CARD_HUES["WHEAT"] + RESOURCE_CARD_HUES["SHEEP"]) / 2.0
    assert classify_glyph((midpoint, 200.0, 210.0)) is None


def test_classify_glyph_rejects_dark_grey_background() -> None:
    # A DARK low-saturation swatch (grey-dark log background, not a bright ORE
    # card) fails the value floor first — it is NOT mistaken for ORE.
    hsv = (0.0, ORE_MAX_SATURATION - 20.0, 30.0)
    assert classify_glyph(hsv) is None


# --- classify_granted_glyphs: multiset read + fail-closed on any bad glyph ---


def test_classify_granted_glyphs_reads_multiset() -> None:
    # Two BRICK glyphs + one WHEAT glyph laid out horizontally in a log crop.
    brick = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["BRICK"], 200.0, 210.0)
    wheat = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["WHEAT"], 200.0, 210.0)
    crop = np.zeros((14, 42, 3), np.uint8)
    crop[:, 0:14] = brick
    crop[:, 14:28] = brick
    crop[:, 28:42] = wheat
    boxes = [(0, 0, 14, 14), (14, 0, 28, 14), (28, 0, 42, 14)]
    assert classify_granted_glyphs(crop, boxes) == Counter({"BRICK": 2, "WHEAT": 1})


def test_classify_granted_glyphs_fails_closed_on_empty_boxes() -> None:
    crop = np.zeros((14, 14, 3), np.uint8)
    assert classify_granted_glyphs(crop, []) is None


def test_classify_granted_glyphs_fails_closed_on_one_unreadable() -> None:
    # A good BRICK glyph + a dark (unreadable) swatch → the WHOLE read is None,
    # never a partial multiset that could false-accept a joint flip.
    brick = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["BRICK"], 200.0, 210.0)
    dark = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["BRICK"], 200.0, 15.0)
    crop = np.zeros((14, 28, 3), np.uint8)
    crop[:, 0:14] = brick
    crop[:, 14:28] = dark
    boxes = [(0, 0, 14, 14), (14, 0, 28, 14)]
    assert classify_granted_glyphs(crop, boxes) is None


# --- validation → scale-up gate wiring --------------------------------------


def _good_frame(expected: Counter[str]) -> LabeledGrantFrame:
    """A labelled frame whose glyphs are the calibrated card colours of ``expected``."""
    boxes: list[tuple[int, int, int, int]] = []
    swatches: list[np.ndarray] = []
    x = 0
    for resource, count in expected.items():
        for _ in range(count):
            if resource == "ORE":
                sw = _hsv_to_rgb_swatch(0.0, ORE_MAX_SATURATION - 20.0, 180.0)
            else:
                sw = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES[resource], 200.0, 210.0)
            swatches.append(sw)
            boxes.append((x, 0, x + 14, 14))
            x += 14
    crop = np.zeros((14, max(x, 14), 3), np.uint8)
    for i, sw in enumerate(swatches):
        crop[:, i * 14 : i * 14 + 14] = sw
    return LabeledGrantFrame(log_crop_rgb=crop, glyph_boxes=boxes, expected=expected)


def test_validation_passes_on_enough_correct_frames() -> None:
    frames = [
        _good_frame(Counter({"BRICK": 1, "WHEAT": 1, "WOOD": 1}))
        for _ in range(MIN_VALIDATION_FRAMES)
    ]
    v = validate_glyph_classifier(frames)
    assert v.passed
    assert v.n_correct == v.n_frames == MIN_VALIDATION_FRAMES
    assert v.reason is None


def test_validation_fails_below_min_frames() -> None:
    frames = [
        _good_frame(Counter({"ORE": 2, "SHEEP": 1})) for _ in range(MIN_VALIDATION_FRAMES - 1)
    ]
    v = validate_glyph_classifier(frames)
    assert not v.passed
    assert v.reason is not None and "frame" in v.reason


def test_validation_fails_below_accuracy_bar() -> None:
    # Enough frames, but one is mislabelled (expected WHEAT where the glyph is BRICK)
    # → accuracy drops below the bar, validation fails with the accuracy reason.
    frames = [_good_frame(Counter({"BRICK": 1, "WOOD": 1})) for _ in range(MIN_VALIDATION_FRAMES)]
    bad = frames[0]
    frames[0] = LabeledGrantFrame(
        log_crop_rgb=bad.log_crop_rgb,
        glyph_boxes=bad.glyph_boxes,
        expected=Counter({"WHEAT": 1, "WOOD": 1}),  # wrong label
    )
    v = validate_glyph_classifier(frames)
    assert not v.passed
    assert v.reason is not None and "accuracy" in v.reason


def test_scale_up_gate_blocked_without_validation() -> None:
    # No validation ever run → gate stays engaged (harvest blocked).
    assert not glyph_classifier_is_validated(None)
    with pytest.raises(GlyphClassifierNotValidated):
        assert_scale_up_orientation_gates(
            resolution=1080,
            affine_residual_px=0.8,
            glyph_classifier_validated=glyph_classifier_is_validated(None),
        )


def test_scale_up_gate_blocked_on_failed_validation() -> None:
    failed = GlyphValidation(
        passed=False, n_frames=3, n_correct=3, accuracy=1.0, reason="too few frames"
    )
    assert not glyph_classifier_is_validated(failed)
    with pytest.raises(GlyphClassifierNotValidated):
        assert_scale_up_orientation_gates(
            resolution=1080,
            affine_residual_px=0.8,
            glyph_classifier_validated=glyph_classifier_is_validated(failed),
        )


def test_scale_up_gate_ALLOWED_only_when_validated() -> None:
    # THE load-bearing wiring: a passing validation flips the gate to silent.
    frames = [_good_frame(Counter({"SHEEP": 1, "ORE": 2})) for _ in range(MIN_VALIDATION_FRAMES)]
    v = validate_glyph_classifier(frames)
    assert v.passed
    assert glyph_classifier_is_validated(v)
    # No raise: resolution + residual clean AND the glyph classifier validated.
    assert_scale_up_orientation_gates(
        resolution=1080,
        affine_residual_px=0.8,
        glyph_classifier_validated=glyph_classifier_is_validated(v),
    )


# --- per-game calibration (brief §5.13 — never a global hue constant) --------


def _board_samples_from_hues(
    hue_by_resource: dict[str, float], ore_sat: float = 20.0, coloured_sat: float = 180.0
) -> tuple[np.ndarray, int]:
    """A synthetic 19x3 board median-HSV array + desert_hex mirroring board_cv's
    per-game convention: 3 low-saturation ORE hexes, then BRICK x3 / WHEAT x4 /
    SHEEP x4 / WOOD x4 at the given per-game hue centres."""
    order = ["ORE"] * 3 + ["BRICK"] * 3 + ["WHEAT"] * 4 + ["SHEEP"] * 4 + ["WOOD"] * 4
    rows: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]  # index 0 = desert
    for resource in order:
        if resource == "ORE":
            rows.append((100.0, ore_sat, 150.0))
        else:
            rows.append((hue_by_resource[resource], coloured_sat, 200.0))
    return np.array(rows, dtype=np.float64), 0


def test_calibrate_glyph_palette_tracks_per_game_hues() -> None:
    # An OFF-PALETTE game whose card hues are shifted far from the fallback prior:
    # the palette must derive centres from the game's OWN tiles, not the constants.
    shifted = {"BRICK": 12.0, "WHEAT": 33.0, "SHEEP": 55.0, "WOOD": 95.0}
    samples, desert = _board_samples_from_hues(shifted)
    palette = calibrate_glyph_palette(samples, desert)
    for resource, hue in shifted.items():
        assert abs(palette.hue_centres[resource] - hue) < 1e-6
    # ORE ceiling sits between the ORE tiles (sat 20) and coloured tiles (sat 180).
    assert 20.0 < palette.ore_max_saturation < 180.0


def test_classify_glyph_uses_the_per_game_palette_not_the_constant() -> None:
    # On the shifted game a swatch at the game's WHEAT hue (33) classifies WHEAT
    # under the per-game palette; the fallback constant (WHEAT=25) would be wrong
    # for a swatch that on this skin is a different resource.
    shifted = {"BRICK": 12.0, "WHEAT": 33.0, "SHEEP": 55.0, "WOOD": 95.0}
    samples, desert = _board_samples_from_hues(shifted)
    palette = calibrate_glyph_palette(samples, desert)
    assert classify_glyph((33.0, 200.0, 210.0), palette) == "WHEAT"
    assert classify_glyph((95.0, 200.0, 210.0), palette) == "WOOD"


def test_calibrate_glyph_palette_rejects_bad_desert() -> None:
    samples, _ = _board_samples_from_hues(RESOURCE_CARD_HUES)
    with pytest.raises(ValueError):
        calibrate_glyph_palette(samples, 99)


def test_hue_rank_order_matches_board_cv() -> None:
    # The rank-slice order must be ascending-hue BRICK→WHEAT→SHEEP→WOOD.
    assert HUE_RESOURCES_BY_RANK == ("BRICK", "WHEAT", "SHEEP", "WOOD")


# --- circular hue median: a red BRICK glyph must NOT become WOOD -------------


def test_classify_glyph_brick_across_red_wrap_is_not_wood() -> None:
    # A red BRICK glyph whose body pixels straddle the 0/180 hue wrap (half at ~178,
    # half at ~4 — both red). A LINEAR median lands ~91 → WOOD (a confident wrong
    # label). The circular median keeps it red → BRICK (or an honest None), never
    # a WOOD/SHEEP swap.
    import cv2

    top = np.full((14, 7, 3), (178, 200, 210), np.uint8)
    bot = np.full((14, 7, 3), (4, 200, 210), np.uint8)
    hsv = np.concatenate([top, bot], axis=1)
    rgb = np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), np.uint8)
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = rgb
    result = classify_granted_glyphs(crop, [(0, 0, 14, 14)])
    assert result != Counter({"WOOD": 1})
    assert result != Counter({"SHEEP": 1})
    assert result in (Counter({"BRICK": 1}), None)


# --- ORE branch fails closed on near-white / blank UI (red-team) -------------


def test_classify_glyph_rejects_pure_white_as_not_ore() -> None:
    # Red-team: pure white (0,0,255) is bright + zero-saturation UI/text, NOT a
    # grey stone card. The ORE branch must return None, never a confident ORE.
    assert classify_glyph((0.0, 0.0, 255.0)) is None
    assert classify_glyph((30.0, 5.0, 250.0)) is None


def test_classify_granted_glyphs_rejects_white_background_swatch() -> None:
    # Red-team end-to-end: a 14x14 swatch that is entirely the log's light-grey/
    # white background (RGB ~235) — a glyph box mis-landed on a blank gap / bright
    # text — reads as an honest None, NOT Counter({'ORE': 1}).
    white = np.full((14, 14, 3), 250, np.uint8)
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = white
    assert classify_granted_glyphs(crop, [(0, 0, 14, 14)]) is None


# --- median-drag by abutting bright text → honest None (red-team) ------------


def test_classify_glyph_impure_text_abutting_swatch_fails_closed() -> None:
    # A genuine WHEAT card body abutting bright white log text 50/50 in the box.
    # The two-sided body mask drops white text; the surviving body is pure WHEAT,
    # so it either reads WHEAT or (if too little body survives) None — never a
    # confident wrong literal like BRICK/ORE dragged by the text.
    import cv2

    wheat_hsv = np.full((14, 7, 3), (RESOURCE_CARD_HUES["WHEAT"], 200, 200), np.uint8)
    wheat_rgb = np.asarray(cv2.cvtColor(wheat_hsv, cv2.COLOR_HSV2RGB), np.uint8)
    text_rgb = np.full((14, 7, 3), 250, np.uint8)  # bright white text
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, 0:7] = wheat_rgb
    crop[:, 7:14] = text_rgb
    result = classify_granted_glyphs(crop, [(0, 0, 14, 14)])
    assert result != Counter({"BRICK": 1})
    assert result != Counter({"ORE": 1})
    assert result in (Counter({"WHEAT": 1}), None)


# --- per-game multi-frame consensus (mirror the board's >=2-frame rule) ------


def _grant_crop(expected: Counter[str]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    boxes: list[tuple[int, int, int, int]] = []
    swatches: list[np.ndarray] = []
    x = 0
    for resource, count in expected.items():
        for _ in range(count):
            if resource == "ORE":
                sw = _hsv_to_rgb_swatch(0.0, ORE_MAX_SATURATION - 20.0, 180.0)
            else:
                sw = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES[resource], 200.0, 210.0)
            swatches.append(sw)
            boxes.append((x, 0, x + 14, 14))
            x += 14
    crop = np.zeros((14, max(x, 14), 3), np.uint8)
    for i, sw in enumerate(swatches):
        crop[:, i * 14 : i * 14 + 14] = sw
    return crop, boxes


def test_consensus_agrees_across_frames() -> None:
    expected = Counter({"BRICK": 1, "WHEAT": 1})
    frames = [_grant_crop(expected) for _ in range(3)]
    assert consensus_granted_glyphs(frames) == expected


def test_consensus_fails_closed_without_enough_agreement() -> None:
    # One readable frame + noise frames that read None → not enough agreement.
    good = _grant_crop(Counter({"SHEEP": 1}))
    dark = np.zeros((14, 14, 3), np.uint8)  # all-dark → None
    frames = [good, (dark, [(0, 0, 14, 14)])]
    # Only 1 agreeing read < MIN_GRANT_CONSENSUS_FRAMES → honest None.
    assert MIN_GRANT_CONSENSUS_FRAMES >= 2
    assert consensus_granted_glyphs(frames) is None


def test_consensus_all_unreadable_is_none() -> None:
    dark = np.zeros((14, 14, 3), np.uint8)
    frames = [(dark, [(0, 0, 14, 14)]) for _ in range(3)]
    assert consensus_granted_glyphs(frames) is None
