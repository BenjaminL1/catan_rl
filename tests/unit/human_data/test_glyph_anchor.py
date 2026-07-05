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

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Every test here reaches cv2 at runtime through the glyph classifier; skip
# (not fail) where opencv isn't installed — the repo convention for CV deps
# (see test_board_cv.py). Without this, CI reports all 18 as FAILURES
# (audit 2026-07 follow-up: found once the collection error was fixed).
pytest.importorskip("cv2")

import catan_rl.human_data.glyph_anchor as glyph_anchor
from catan_rl.human_data import (
    GlyphClassifierNotValidated,
    assert_scale_up_orientation_gates,
)
from catan_rl.human_data.glyph_anchor import (
    CARD_PALETTE,
    GRANT_RE,
    HUE_RESOURCES_BY_RANK,
    MIN_GRANT_CONSENSUS_FRAMES,
    MIN_VALIDATION_ACCURACY,
    MIN_VALIDATION_FRAMES,
    ORE_MAX_SATURATION,
    RESOURCE_CARD_HUES,
    GlyphValidation,
    LabeledGrantFrame,
    calibrate_glyph_palette,
    classify_glyph,
    classify_granted_glyphs,
    consensus_granted_glyphs,
    detect_glyph_boxes,
    glyph_classifier_fingerprint,
    glyph_classifier_is_validated,
    load_glyph_validation,
    validate_glyph_classifier,
    validation_fingerprint,
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
    by_box: list[str] = []
    x = 0
    for resource, count in expected.items():
        for _ in range(count):
            if resource == "ORE":
                sw = _hsv_to_rgb_swatch(0.0, ORE_MAX_SATURATION - 20.0, 180.0)
            else:
                sw = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES[resource], 200.0, 210.0)
            swatches.append(sw)
            boxes.append((x, 0, x + 14, 14))
            by_box.append(resource)
            x += 14
    crop = np.zeros((14, max(x, 14), 3), np.uint8)
    for i, sw in enumerate(swatches):
        crop[:, i * 14 : i * 14 + 14] = sw
    return LabeledGrantFrame(
        log_crop_rgb=crop,
        glyph_boxes=boxes,
        expected=expected,
        expected_by_box=tuple(by_box),
    )


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


def test_validation_zero_tolerance_on_ore_brick_confusion() -> None:
    # USER-APPROVED BAR: a single ORE->BRICK confusion fails validation regardless of
    # the other numbers — it is the systematic, firewall-blinding misread. Build a
    # frame whose TRUE label is ORE but whose glyph is a vivid brick colour, so the
    # classifier confidently reads BRICK.
    frames = [_good_frame(Counter({"BRICK": 1, "WOOD": 1})) for _ in range(MIN_VALIDATION_FRAMES)]
    brick_swatch = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["BRICK"], 200.0, 210.0)
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = brick_swatch
    frames.append(
        LabeledGrantFrame(
            log_crop_rgb=crop,
            glyph_boxes=[(0, 0, 14, 14)],
            expected=Counter({"ORE": 1}),
            expected_by_box=("ORE",),
        )
    )
    v = validate_glyph_classifier(frames)
    assert not v.passed
    assert v.reason is not None and "ORE<->BRICK" in v.reason
    assert ("ORE", "BRICK", 1) in v.confusion


def test_validation_confusion_matrix_and_coverage_fields() -> None:
    # Per-box scoring: correct reads land on the confusion diagonal; a fail-closed
    # unreadable box counts as COVERAGE (n_unread_boxes), never as a confusion entry.
    frames = [_good_frame(Counter({"SHEEP": 1, "ORE": 2})) for _ in range(MIN_VALIDATION_FRAMES)]
    v = validate_glyph_classifier(frames)
    assert v.passed
    assert v.n_boxes == 3 * MIN_VALIDATION_FRAMES
    assert v.n_unread_boxes == 0
    assert all(true == pred for true, pred, _ in v.confusion)  # diagonal only

    dark = _hsv_to_rgb_swatch(RESOURCE_CARD_HUES["WOOD"], 200.0, 15.0)  # unreadably dark
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = dark
    frames.append(
        LabeledGrantFrame(
            log_crop_rgb=crop,
            glyph_boxes=[(0, 0, 14, 14)],
            expected=Counter({"WOOD": 1}),
            expected_by_box=("WOOD",),
        )
    )
    v2 = validate_glyph_classifier(frames)
    assert v2.n_unread_boxes == 1
    assert not any(true == "WOOD" and pred != "WOOD" for true, pred, _ in v2.confusion)


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


# --- the PASS is bound to classifier identity (expert SHOULD-FIX 2026-07-05) --


def test_validation_is_stamped_with_the_current_classifier_fingerprint() -> None:
    # Every validate_glyph_classifier result — pass AND fail — carries the
    # fingerprint of the classifier that produced it, so the artifact written to
    # data/human/glyph_validation.json is self-identifying.
    frames = [_good_frame(Counter({"SHEEP": 1, "ORE": 2})) for _ in range(MIN_VALIDATION_FRAMES)]
    v = validate_glyph_classifier(frames)
    assert v.classifier_fingerprint == glyph_classifier_fingerprint()
    failed = validate_glyph_classifier(frames[:1])  # under-sampled → passed=False
    assert not failed.passed
    assert failed.classifier_fingerprint == glyph_classifier_fingerprint()
    # The fingerprint is a stable sha256 hex digest of the classifier's identity.
    fp = glyph_classifier_fingerprint()
    assert len(fp) == 64 and fp == glyph_classifier_fingerprint()


def test_scale_up_gate_blocked_on_fabricated_or_stale_validation() -> None:
    # A hand-built passed=True record (a fabricated PASS — e.g. a doctored
    # glyph_validation.json) has no / a foreign fingerprint and must NOT satisfy
    # the gate; nor may a stale PASS stamped by a since-edited classifier.
    fabricated = GlyphValidation(passed=True, n_frames=24, n_correct=24, accuracy=1.0, reason=None)
    assert not glyph_classifier_is_validated(fabricated)
    stale = GlyphValidation(
        passed=True,
        n_frames=24,
        n_correct=24,
        accuracy=1.0,
        reason=None,
        classifier_fingerprint="f" * 64,
    )
    assert not glyph_classifier_is_validated(stale)
    for unbound in (fabricated, stale):
        with pytest.raises(GlyphClassifierNotValidated):
            assert_scale_up_orientation_gates(
                resolution=1080,
                affine_residual_px=0.8,
                glyph_classifier_validated=glyph_classifier_is_validated(unbound),
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


@pytest.mark.parametrize("bg", [235, 238, 240])
def test_classify_granted_glyphs_rejects_light_grey_log_background(bg: int) -> None:
    # Red-team (the realistic band the 250-only tests miss): Colonist's log panel
    # background is a light grey at RGB ~235-240 (HSV val 235-240, ~0 saturation),
    # which sits AT/BELOW the old strict `> 240` near-white boundary. A glyph box
    # that mis-lands on that panel background must read an honest None, NOT a
    # confident ORE — a spurious ORE corrupts the granted multiset that feeds the
    # joint-flip firewall (brief §5: confidently wrong, not merely noisy).
    grey = np.full((14, 14, 3), bg, np.uint8)
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = grey
    assert classify_granted_glyphs(crop, [(0, 0, 14, 14)]) is None


@pytest.mark.parametrize(
    ("hue", "sat", "val"),
    [(0.0, 20.0, 250.0), (30.0, 8.0, 245.0), (0.0, 40.0, 255.0), (10.0, 59.0, 250.0)],
)
def test_classify_glyph_rejects_faint_tint_near_white_as_not_ore(
    hue: float, sat: float, val: float
) -> None:
    # Red-team counterexample: a faint-tinted near-white swatch (sat in the stone
    # band [8, ore_ceiling) but val above the card body — a UI highlight / setup
    # glow with a slight colour cast). The old ORE guard only rejected the
    # desaturated corner (sat < MIN_ORE_SATURATION AND val > MAX_GLYPH_VALUE), so
    # this whole band slipped past as a confident ORE. It must fail closed (None).
    assert classify_glyph((hue, sat, val)) is None


def test_classify_granted_glyphs_rejects_faint_tint_swatch_pixel_path() -> None:
    # End-to-end pixel path for the faint-tint counterexample: a stable tinted
    # background reads identically every frame, so multi-frame consensus can't save
    # it — the single-frame reader itself must fail closed.
    import cv2

    tint = np.asarray(
        cv2.cvtColor(np.full((14, 14, 3), (0, 20, 250), np.uint8), cv2.COLOR_HSV2RGB), np.uint8
    )
    crop = np.zeros((14, 14, 3), np.uint8)
    crop[:, :] = tint
    assert classify_granted_glyphs(crop, [(0, 0, 14, 14)]) is None
    assert consensus_granted_glyphs([(crop, [(0, 0, 14, 14)])] * 4) is None


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


def test_consensus_rejects_contradictory_readable_reads() -> None:
    # RED-TEAM: two distinct multisets each backed by >= MIN_GRANT_CONSENSUS_FRAMES
    # readable frames. A plurality/tie-break would emit one CONFIDENTLY; full
    # agreement must reject (None) like read_board_stable does on any disagreement.
    a = Counter({"WOOD": 1, "BRICK": 1})
    b = Counter({"WHEAT": 1, "ORE": 1})
    frames = [_grant_crop(a), _grant_crop(b), _grant_crop(a), _grant_crop(b)]
    assert consensus_granted_glyphs(frames) is None
    # Order-independence: reversing the reads must not flip the reject to an accept.
    assert consensus_granted_glyphs(list(reversed(frames))) is None


def test_classify_glyph_borderline_grey_never_confident_warm_hue() -> None:
    # Review BLOCKER, re-pinned to the MEASURED card palette (2026-07-04 valset):
    # a grey stone whose saturation drifts past the ORE ceiling must fail closed —
    # never read as a confident warm hue (the firewall-blinding ORE<->BRICK path).
    # Measured bands: ORE cards S 44-57, coloured cards S >= 99; ceiling 75 with a
    # fail-closed dead band [75, 95) between the branches.
    assert classify_glyph((5.0, 80.0, 175.0)) is None  # dead band — refuses to guess
    assert classify_glyph((5.0, 90.0, 175.0)) is None  # still dead band
    assert classify_glyph((5.8, 110.0, 188.0)) == "BRICK"  # measured-median BRICK
    assert classify_glyph((5.0, 50.0, 179.0)) == "ORE"  # measured-band ORE
    assert classify_glyph((5.0, 62.0, 175.0)) == "ORE"  # below ceiling: grey band


# --- detect_glyph_boxes: the PROMOTED production detector (expert BLOCKERs 2+3) --
#
# The 24/24 glyph-validation PASS scored a COMPOSITE (detector + classifier), so
# the detector must live in production code and its merged-box failure mode —
# boxes spanning >1 icon classify CONFIDENTLY WRONG and survive consensus — must
# fail CLOSED. Two real-frame regression suites pin both sides:
#
# - the GOOD path: the committed game1_postsetup_t247.png fixture (2 grant lines)
#   yields exactly the 3+3 boxes the detector was smoke-validated on;
# - the MERGED path: the 8 SKIP-labelled crops of the 2026-07-04 valset (5 log
#   frames, committed under data/human/glyph_valset/) must ALL end in an honest
#   no-read — the detector returns no boxes, so classify_granted_glyphs is None
#   and the game falls out of the harvest instead of feeding the firewall the
#   measured wrong-but-stable multisets (Yejbe2-q4_o read {BRICK:1, WHEAT:1};
#   MZBLarAmNXw_t145_b0 read a confident ORE).
#
# The OCR token boxes below are PINNED easyocr output (recorded 2026-07-05, CPU,
# the same reader the harness uses) so these tests exercise the detector
# deterministically without importing easyocr; the OCR side has its own
# fixture-pinned tests in test_logparse.py. Texts are stored post-_normalise,
# lowercase — exactly what the harness feeds GRANT_RE.

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"
_VALSET = Path(__file__).resolve().parents[3] / "data" / "human" / "glyph_valset"

#: Pinned OCR tokens of the game1_postsetup_t247.png log crop (normalised text,
#: (x0, y0, x1, y1)). The first grant line is OCR-mangled ("receivea") — exactly
#: the misread GRANT_RE is built to tolerate.
_GAME1_TOKENS: list[tuple[str, tuple[int, int, int, int]]] = [
    ("inernantom receivea starting resources", (291, 0, 617, 21)),
    ("thephantom placed a road", (286, 54, 514, 80)),
    ("rayman147 placed a settlement", (285, 108, 548, 135)),
    ("rayman147 received starting resources", (285, 139, 604, 166)),
    ("rayman147 placed a road", (285, 171, 502, 196)),
]

#: The exact boxes per grant line the detector was validated on (3 + 3): line 1
#: is two icons on the grant row plus one on the wrap row below; line 2 is a
#: tightly-packed 3-icon run that must pitch-split into three 19px cells.
_GAME1_EXPECTED: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {
    (291, 0, 617, 21): [(620, 6, 632, 16), (639, 6, 652, 16), (268, 26, 280, 44)],
    (285, 139, 604, 166): [(604, 140, 619, 160), (623, 140, 638, 160), (642, 140, 657, 160)],
}

#: Pinned OCR tokens for the 5 committed SKIP-frame log crops (the merged-icon
#: regression corpus — every frame owning a SKIP label in labels.json).
_SKIP_FRAME_TOKENS: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {
    # b0 is a 26px merged box (2 icons) INSIDE the single-box aspect band.
    "sG05DoaOmM4_t200": [
        ("thephantom placed a", (280, 18, 460, 42)),
        ("thephantom placed a", (280, 45, 460, 72)),
        ("thephantom received starting resources", (280, 76, 608, 102)),
        ("secpra placed a", (279, 135, 408, 161)),
        ("secpra placed a", (277, 163, 410, 191)),
        ("received starting resources", (340, 194, 558, 218)),
        ("secpra", (275, 189, 341, 220)),
    ],
    # A 46px 3-icon run whose line-height pitch mis-rounds to k=2 -> two 19px
    # boxes each spanning >1 icon; b0 read a confident (wrong) ORE.
    "MZBLarAmNXw_t145": [
        ("8 realninos placed a", (266, 36, 442, 64)),
        ("realninos placed a", (280, 70, 442, 94)),
        ("8 realninos placed a", (266, 132, 442, 156)),
        ("8 realninos placed a", (264, 162, 442, 186)),
        ("realninos received starting resources", (280, 192, 588, 218)),
    ],
    # b1 is a 29px merged box (2 icons) inside the single-box aspect band.
    "mY2X6Dw0iFE_t127": [
        ("mo061101 placed a", (293, 3, 439, 23)),
        ("thephantom placed", (288, 47, 446, 72)),
        ("thephantom placed a", (290, 76, 456, 100)),
        ("thephantom placed a", (288, 123, 456, 148)),
        ("thephantom received starting resources", (292, 150, 594, 176)),
        ("thephantom placed a", (288, 178, 458, 202)),
        ("chat is", (291, 251, 345, 271)),
        ("being monitored: please be respectful.", (340, 247, 628, 275)),
    ],
    # Same mis-rounded k=2 run at two sampled times; the two 19px merged boxes
    # read a STABLE wrong {BRICK:1, WHEAT:1} that survives consensus — the
    # expert-verified firewall-defeating case.
    "Yejbe2-q4_o_t250": [
        ("8 thelew placed a", (266, 45, 414, 70)),
        ("8 thejew placed a", (266, 74, 414, 100)),
        ("thejew placed a", (280, 133, 414, 158)),
        ("thelew placed a", (280, 162, 412, 188)),
        ("thejew received starting resources", (280, 188, 560, 218)),
        ("8 thejew: thephantom, love ur content: keep it", (264, 273, 644, 298)),
        ("up", (261, 305, 285, 321)),
    ],
    "Yejbe2-q4_o_t260": [
        ("thejew placed a", (280, 45, 414, 70)),
        ("8 thejew placed a", (266, 73, 414, 100)),
        ("thejew placed a", (280, 133, 414, 158)),
        ("thejew placed a", (280, 164, 412, 188)),
        ("thejew received starting resources", (280, 188, 560, 218)),
        ("8 thejew: thephantom, love ur content: keep it", (264, 273, 644, 298)),
        ("up", (261, 305, 285, 321)),
    ],
}


def _load_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    assert bgr is not None, f"missing committed fixture {path}"
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8)


def test_grant_re_tolerates_ocr_misreads() -> None:
    assert GRANT_RE.search("thephantom received starting resources")
    assert GRANT_RE.search("inernantom receivea starting resources")  # game1 line 1
    assert GRANT_RE.search("recelved starting resources")
    assert not GRANT_RE.search("thephantom placed a road")


def test_detector_pins_the_game1_fixture_boxes_exactly() -> None:
    # The GOOD path, end-to-end on the committed real frame: both grant lines of
    # game1_postsetup_t247.png yield EXACTLY the 3+3 boxes the promoted detector
    # was validated on (any drift here is a detector regression — investigate,
    # never re-pin without re-validating the composite).
    frame = _load_rgb(_FIXTURES / "game1_postsetup_t247.png")
    from catan_rl.human_data.logparse import LOG_CROP_FRAC

    h, w = frame.shape[:2]
    x0f, y0f, x1f, y1f = LOG_CROP_FRAC
    crop = frame[int(y0f * h) : int(y1f * h), int(x0f * w) : int(x1f * w)]
    text_boxes = [b for _, b in _GAME1_TOKENS]
    grant_boxes = [b for t, b in _GAME1_TOKENS if GRANT_RE.search(t)]
    assert grant_boxes == list(_GAME1_EXPECTED)  # both grant lines, pinned order
    for line_box, expected in _GAME1_EXPECTED.items():
        assert detect_glyph_boxes(crop, line_box, text_boxes) == expected


def test_skip_corpus_covers_every_skip_label() -> None:
    # The 5 pinned frames must own ALL SKIP labels in the committed labels.json —
    # if the labelled corpus grows a new SKIP stratum, this corpus (and the
    # committed crops) must grow with it, never silently under-cover.
    labels: dict[str, str] = json.loads((_VALSET / "labels.json").read_text())
    skip_frames = {cid.rsplit("_b", 1)[0] for cid, lab in labels.items() if lab == "SKIP"}
    assert skip_frames == set(_SKIP_FRAME_TOKENS)
    n_skip = sum(1 for lab in labels.values() if lab == "SKIP")
    assert n_skip == 8


@pytest.mark.parametrize("crop_id", sorted(_SKIP_FRAME_TOKENS))
def test_detector_no_reads_every_skip_labelled_merged_frame(crop_id: str) -> None:
    # The MERGED path fails CLOSED end-to-end: on every frame that produced a
    # SKIP-labelled (unlabelable, merged-icon) crop, the promoted detector
    # returns NO boxes — never a box spanning >1 icon — and the grant read is an
    # honest None. Before the merged-box rule these frames produced boxes that
    # classified confidently WRONG and survived consensus_granted_glyphs.
    crop = _load_rgb(_VALSET / f"{crop_id}_log.png")
    tokens = _SKIP_FRAME_TOKENS[crop_id]
    text_boxes = [b for _, b in tokens]
    grant_box = next(b for t, b in tokens if GRANT_RE.search(t))
    boxes = detect_glyph_boxes(crop, grant_box, text_boxes)
    assert boxes == []  # detector no-read: the merged-box rule fired
    assert classify_granted_glyphs(crop, boxes, CARD_PALETTE) is None


# --- load_glyph_validation: the artifact-side gate (expert SHOULD-FIXes a/b) --
#
# The on-disk PASS (data/human/glyph_validation.json) must not outlive the
# constants / detector / bar it measured: load_glyph_validation recomputes the
# validation fingerprint (LIVE palette/threshold constant values + the detector
# source) at load time and fails CLOSED (None + a logged reason) on any doubt.


def _artifact_payload(v: GlyphValidation) -> dict[str, Any]:
    """Serialize a GlyphValidation exactly as `scripts/glyph_valset.py score` does."""
    return {
        "passed": v.passed,
        "n_frames": v.n_frames,
        "n_correct": v.n_correct,
        "accuracy": v.accuracy,
        "n_boxes": v.n_boxes,
        "n_unread_boxes": v.n_unread_boxes,
        "confusion": [list(c) for c in v.confusion],
        "reason": v.reason,
        "classifier_fingerprint": v.classifier_fingerprint,
        "validation_fingerprint": validation_fingerprint(),
        "git_rev": "deadbeef",
        "bar": {
            "min_frames": MIN_VALIDATION_FRAMES,
            "min_accuracy": MIN_VALIDATION_ACCURACY,
            "zero_ore_brick": True,
        },
    }


def _passing_validation() -> GlyphValidation:
    frames = [_good_frame(Counter({"SHEEP": 1, "ORE": 2})) for _ in range(MIN_VALIDATION_FRAMES)]
    v = validate_glyph_classifier(frames)
    assert v.passed
    return v


def test_load_glyph_validation_returns_pass_on_matching_fingerprint(tmp_path: Path) -> None:
    p = tmp_path / "glyph_validation.json"
    v = _passing_validation()
    p.write_text(json.dumps(_artifact_payload(v)))
    loaded = load_glyph_validation(p)
    assert loaded is not None
    assert loaded.passed
    assert loaded.n_frames == v.n_frames
    assert loaded.n_correct == v.n_correct
    assert loaded.confusion == v.confusion
    assert loaded.classifier_fingerprint == v.classifier_fingerprint
    # The round-tripped record still satisfies the module-source gate check.
    assert glyph_classifier_is_validated(loaded)


@pytest.mark.parametrize(
    "constant",
    [
        "ORE_MAX_SATURATION",  # palette threshold
        "MIN_GLYPH_HUE_MARGIN",  # classifier threshold
        "MERGED_BOX_PITCH_FACTOR",  # DETECTOR tuning — the fingerprint covers it too
    ],
)
def test_load_glyph_validation_none_after_constant_perturbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    # A stored PASS written under the CURRENT constants...
    p = tmp_path / "glyph_validation.json"
    p.write_text(json.dumps(_artifact_payload(_passing_validation())))
    assert load_glyph_validation(p) is not None
    # ...must be refused once ANY palette/threshold/detector constant changes
    # (live values, so even a runtime retune — not just a source edit — mismatches).
    monkeypatch.setattr(glyph_anchor, constant, getattr(glyph_anchor, constant) + 1.0)
    assert load_glyph_validation(p) is None


def test_load_glyph_validation_rejects_failed_or_lowered_bar(tmp_path: Path) -> None:
    p = tmp_path / "glyph_validation.json"
    v = _passing_validation()
    # A non-PASS record never loads, whatever its fingerprint says.
    failed = _artifact_payload(v)
    failed["passed"] = False
    p.write_text(json.dumps(failed))
    assert load_glyph_validation(p) is None
    # A PASS scored under a LOOSER bar than the current constants never loads:
    # raising MIN_VALIDATION_* retroactively invalidates the old artifact.
    for weak_bar in (
        {"min_frames": MIN_VALIDATION_FRAMES - 1, "min_accuracy": 0.98, "zero_ore_brick": True},
        {"min_frames": MIN_VALIDATION_FRAMES, "min_accuracy": 0.9, "zero_ore_brick": True},
        {"min_frames": MIN_VALIDATION_FRAMES, "min_accuracy": 0.98, "zero_ore_brick": False},
        None,
    ):
        weak = _artifact_payload(v)
        weak["bar"] = weak_bar
        p.write_text(json.dumps(weak))
        assert load_glyph_validation(p) is None


def test_load_glyph_validation_fails_closed_on_missing_or_malformed(tmp_path: Path) -> None:
    assert load_glyph_validation(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_glyph_validation(bad) is None
    bad.write_text(json.dumps(["passed"]))  # not an object
    assert load_glyph_validation(bad) is None


def test_committed_validation_artifact_is_hardened_and_current() -> None:
    # The COMMITTED artifact must load under the CURRENT code — this is the
    # re-score discipline made executable: any edit to the glyph constants or
    # the detector fails this test until `scripts/glyph_valset.py score` is
    # re-run over the committed labels + *_log.png crops (all in-repo, so the
    # score IS reproducible from a clean clone). Never fix this test by editing
    # the JSON by hand.
    artifact = Path(__file__).resolve().parents[3] / "data" / "human" / "glyph_validation.json"
    loaded = load_glyph_validation(artifact)
    assert loaded is not None
    assert loaded.passed
    assert glyph_classifier_is_validated(loaded)
    raw = json.loads(artifact.read_text())
    assert raw["validation_fingerprint"] == validation_fingerprint()
    assert raw["git_rev"] not in ("", "unknown")
    # Every scored frame's log crop is committed alongside the artifact.
    valset = artifact.parent / "glyph_valset"
    assert raw["scored_frames"]
    for cid in raw["scored_frames"]:
        assert (valset / f"{cid}_log.png").exists(), f"scored crop {cid}_log.png not committed"
    # (c) the video-disjoint 2-fold cross-check is present and both folds pass.
    folds = raw["folds"]
    assert len(folds) == 2
    assert all(f["passed"] for f in folds)
    assert all(f["n_frames"] >= 1 for f in folds)
    fold_vids = [set(f["score_videos"]) for f in folds]
    assert not (fold_vids[0] & fold_vids[1])  # video-disjoint
    # (d) per-box margin diagnostics (min/median summaries) are present.
    margins = raw["margins"]
    assert margins["dead_band_sat"] == [75.0, 95.0]
    for key in ("sat_gap_ore_side", "sat_gap_hue_side", "hue_margin"):
        assert margins[key]["n"] > 0
        assert margins[key]["min"] is not None
        assert margins[key]["median"] is not None
