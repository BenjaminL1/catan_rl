"""Glyph classifier — the READER side of the orientation-INDEPENDENT firewall.

:mod:`catan_rl.human_data.orientation` owns the glyph-anchor CHECK (does the
granted-card multiset equal a 2nd-settlement's adjacent-resource multiset under
the chosen affine?) and the scale-up HARD GATE
(:func:`~catan_rl.human_data.orientation.assert_scale_up_orientation_gates`,
which raises :class:`~catan_rl.human_data.orientation.GlyphClassifierNotValidated`
until a validated glyph classifier is wired). This module supplies that reader:
given a frame captured right after a player's ``"received starting resources"``
log event, it **colour-classifies the granted resource card-icon glyphs** in the
log panel into a per-player ``Counter[str]`` resource multiset that feeds
:func:`~catan_rl.human_data.orientation.assert_glyph_anchor`.

Why this is the only joint-flip defence (orientation.py §2): the provenance
desert-binding cannot see a board+openings pair that were *jointly* D6-flipped
(both stages flipped together, so they still agree on the desert hex). The glyph
anchor ties the parse to an EXTERNAL ground truth — the resources Colonist says
each player was granted for their 2nd settlement — that does not come from the
affine. A joint flip moves the 2nd settlement onto different hexes, so the
predicted adjacency multiset diverges from the granted-card multiset and the
anchor rejects. This reader produces that granted-card multiset.

**BEST-EFFORT honesty (brief §5 / task spec).** The card glyphs are ~14px,
line-wrapped, and abut the adjacent log text. If a glyph cannot be classified
*reliably* — its nearest resource-card colour is not clearly separated from the
runner-up (:data:`MIN_GLYPH_HUE_MARGIN`), or the swatch is too small / desaturated
to be a card icon — :func:`classify_granted_glyphs` returns ``None`` (an honest
"could not read"), never a guessed multiset. And :func:`validate_glyph_classifier`
only reports ``passed=True`` when the classifier reproduces the labelled grants on
enough frames at the pre-registered accuracy bar. The scale-up gate is wired to
that validation via :func:`glyph_classifier_is_validated`, so the 300-game batch
harvest stays BLOCKED (the gate keeps raising) until a genuinely-validated
classifier exists — the firewall is never faked validated.

The per-resource card-glyph colours are a per-game calibrated *table* keyed by
resource literal (:data:`RESOURCE_CARD_HUES`), not a two-colour hardcode — the
same convention as :data:`~catan_rl.human_data.openings.PALETTE`. DESERT is never
granted, so it is absent from the table. CPU-only; ``cv2`` is imported lazily.
Never imports ``gui/`` or the training path (brief §6).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    import numpy.typing as npt

#: The five GRANTABLE resource literals (DESERT is never granted at setup, so it
#: is not a card-glyph class). Resource literals are strings, never an enum
#: (brief §5.8). Order is stable for deterministic nearest-class iteration.
GRANTABLE_RESOURCES: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Canonical OpenCV hue (0..180) of each granted resource's card-icon glyph in the
#: Colonist skin. The card icons use the same resource colour family as the board
#: tiles: BRICK red-orange, WHEAT gold, SHEEP yellow-green, WOOD forest green, ORE
#: a desaturated blue-grey. Classification is nearest-hue with a MARGIN gate
#: (:data:`MIN_GLYPH_HUE_MARGIN`), so these are calibration centres, not hard
#: thresholds — a glyph whose nearest and runner-up centres are within the margin
#: is rejected as unreadable rather than guessed. ORE is handled by the low-
#: saturation branch (a grey card has no meaningful hue), mirroring the board_cv
#: ORE-by-saturation rule.
RESOURCE_CARD_HUES: dict[str, float] = {
    "BRICK": 8.0,
    "WHEAT": 25.0,
    "SHEEP": 40.0,
    "WOOD": 70.0,
}

#: Saturation below which a (sufficiently bright) card glyph is treated as ORE (a
#: desaturated grey card has no meaningful hue), mirroring board_cv's ORE-by-low-
#: saturation rule. A card glyph with saturation >= this is classified by hue
#: against :data:`RESOURCE_CARD_HUES`.
ORE_MAX_SATURATION = 60.0

#: Minimum value (brightness) for a swatch to be a card glyph at all. A near-black
#: swatch (log text / background bleed, not a card icon) — whether desaturated
#: (grey-dark) or coloured-dark — is unreadable, so :func:`classify_glyph` returns
#: ``None`` and the read fails closed. This value floor is what distinguishes a
#: bright grey ORE card from dark background: ORE requires low saturation AND value
#: above this floor. Card icons are vivid/bright, well above it.
MIN_GLYPH_VALUE = 90.0

#: Minimum OpenCV-hue gap (0..180 scale) between a glyph's nearest and second-
#: nearest resource-card hue centre for a HUE-classified (non-ORE) glyph to be
#: accepted. Below this the glyph is ambiguous (its colour sits between two card
#: families) and :func:`classify_glyph` returns ``None`` — the BEST-EFFORT honest
#: "could not read reliably" that keeps the scale-up gate engaged (task spec).
MIN_GLYPH_HUE_MARGIN = 8.0

#: Pre-registered validation bars for :func:`validate_glyph_classifier`. The
#: classifier is only reported ``passed`` when it reproduces the labelled grants
#: on at least :data:`MIN_VALIDATION_FRAMES` labelled post-grant frames at
#: >= :data:`MIN_VALIDATION_ACCURACY` per-player-grant accuracy. These gate the
#: scale-up firewall flip (task spec: the gate flips to allowed ONLY when
#: validated), so they are deliberately strict — a mislabelled grant silently
#: welds a jointly-flipped board.
MIN_VALIDATION_FRAMES = 8
MIN_VALIDATION_ACCURACY = 0.95


def _hue_distance(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (0..180 wraps)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def classify_glyph(hsv: tuple[float, float, float]) -> str | None:
    """Classify one card-glyph median HSV into a granted resource literal, or
    ``None`` when it cannot be read RELIABLY (BEST-EFFORT honesty).

    ``hsv`` is the median OpenCV HSV (hue 0..180, sat/val 0..255) of a candidate
    card-icon swatch. Returns ``None`` — never a guess — when:

    - the swatch is too dark to be a vivid/bright card icon (below
      :data:`MIN_GLYPH_VALUE`), i.e. log text / background bleed, or
    - a hue-classified swatch's nearest and runner-up resource-card hue centres
      are within :data:`MIN_GLYPH_HUE_MARGIN` (an ambiguous between-families
      colour).

    ORE is the desaturated-grey branch (saturation < :data:`ORE_MAX_SATURATION`
    with value above the floor — a BRIGHT grey card, not black text), mirroring
    board_cv's ORE-by-saturation rule; the other four are nearest-hue with the
    margin gate. The value floor is what separates a bright grey ORE card from
    dark background (a dark low-sat swatch fails the value check first).
    """
    hue, sat, val = hsv
    if val < MIN_GLYPH_VALUE:
        return None  # too dark to be a card icon (log text / gap)
    if sat < ORE_MAX_SATURATION:
        return "ORE"  # bright desaturated grey card
    ranked = sorted(
        RESOURCE_CARD_HUES,
        key=lambda r: _hue_distance(hue, RESOURCE_CARD_HUES[r]),
    )
    nearest, runner_up = ranked[0], ranked[1]
    margin = _hue_distance(hue, RESOURCE_CARD_HUES[runner_up]) - _hue_distance(
        hue, RESOURCE_CARD_HUES[nearest]
    )
    if margin < MIN_GLYPH_HUE_MARGIN:
        return None  # between two card families — ambiguous, fail closed
    return nearest


def _glyph_median_hsv(
    glyph_rgb: npt.NDArray[np.uint8],
) -> tuple[float, float, float]:
    """Median HSV over the vivid (non-background) pixels of one card-glyph swatch.

    Restricts to pixels above a small saturation+value floor so the card body —
    not the anti-aliased edge onto the dark log panel — drives the median, then
    returns the median hue/sat/val. An all-background swatch yields a low-value
    median that :func:`classify_glyph` rejects.
    """
    import cv2

    hsv = np.asarray(cv2.cvtColor(glyph_rgb, cv2.COLOR_RGB2HSV), np.uint8)
    flat = hsv.reshape(-1, 3).astype(np.float64)
    # Keep pixels that are plausibly card body (some sat OR bright grey ORE),
    # dropping the near-black log panel that would drag the median dark.
    keep = flat[:, 2] >= MIN_GLYPH_VALUE
    if not bool(keep.any()):
        return (0.0, 0.0, 0.0)
    body = flat[keep]
    med = np.median(body, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def classify_granted_glyphs(
    log_crop_rgb: npt.NDArray[np.uint8],
    glyph_boxes: list[tuple[int, int, int, int]],
) -> Counter[str] | None:
    """Colour-classify the granted resource card glyphs for ONE player into a
    resource multiset, or ``None`` if ANY glyph cannot be read reliably.

    ``log_crop_rgb`` is the RGB log-panel crop of a frame captured right after the
    player's ``"received starting resources"`` event; ``glyph_boxes`` are the
    ``(x0, y0, x1, y1)`` pixel boxes of that line's card icons (from the glyph
    detector — not this function's concern; a box per granted card). Returns a
    ``Counter`` over the granted resource literals on success.

    **Fail closed (BEST-EFFORT).** If ``glyph_boxes`` is empty, or any box does
    not classify to a granted resource (:func:`classify_glyph` returned ``None``),
    the whole read returns ``None`` — a partial/uncertain grant multiset must never
    feed :func:`~catan_rl.human_data.orientation.assert_glyph_anchor`, or the
    firewall would compare against a wrong multiset and false-accept a joint flip.
    """
    if not glyph_boxes:
        return None
    granted: Counter[str] = Counter()
    for x0, y0, x1, y1 in glyph_boxes:
        swatch = np.asarray(log_crop_rgb[y0:y1, x0:x1], np.uint8)
        if swatch.size == 0:
            return None
        resource = classify_glyph(_glyph_median_hsv(swatch))
        if resource is None:
            return None  # one unreadable glyph → the whole grant is unreliable
        granted[resource] += 1
    return granted


@dataclass(frozen=True, slots=True)
class LabeledGrantFrame:
    """One labelled post-grant frame for :func:`validate_glyph_classifier`.

    ``log_crop_rgb`` is the RGB log-panel crop right after a
    ``"received starting resources"`` event; ``glyph_boxes`` the granted card
    boxes for the ONE player that line grants; ``expected`` the hand-labelled
    granted multiset (ground truth). The classifier is scored per frame: a frame
    is CORRECT iff :func:`classify_granted_glyphs` returns exactly ``expected``.
    """

    log_crop_rgb: npt.NDArray[np.uint8]
    glyph_boxes: list[tuple[int, int, int, int]]
    expected: Counter[str]


@dataclass(frozen=True, slots=True)
class GlyphValidation:
    """The outcome of :func:`validate_glyph_classifier`.

    ``passed`` is ``True`` ONLY when the classifier reproduced the labelled grants
    on >= :data:`MIN_VALIDATION_FRAMES` frames at >= :data:`MIN_VALIDATION_ACCURACY`
    accuracy. ``n_frames`` / ``n_correct`` / ``accuracy`` carry the measured
    numbers; ``reason`` explains a fail (too few frames, or below the accuracy bar)
    so the scale-up gate can report EXACTLY why the harvest stays blocked (task
    spec: never fake it validated — report why).
    """

    passed: bool
    n_frames: int
    n_correct: int
    accuracy: float
    reason: str | None


def validate_glyph_classifier(frames: list[LabeledGrantFrame]) -> GlyphValidation:
    """Score the glyph classifier against labelled post-grant frames.

    Runs :func:`classify_granted_glyphs` on each frame and counts a frame CORRECT
    iff the returned multiset equals the frame's ``expected`` label (an unreadable
    ``None`` counts as wrong — an honest miss, not a pass). Reports ``passed=True``
    only when both pre-registered bars clear (:data:`MIN_VALIDATION_FRAMES`,
    :data:`MIN_VALIDATION_ACCURACY`); otherwise ``passed=False`` with a ``reason``.
    This is the sole switch the scale-up firewall consults
    (:func:`glyph_classifier_is_validated`), so a below-bar or under-sampled run
    keeps the 300-game harvest BLOCKED.
    """
    n_frames = len(frames)
    n_correct = sum(
        1 for f in frames if classify_granted_glyphs(f.log_crop_rgb, f.glyph_boxes) == f.expected
    )
    accuracy = (n_correct / n_frames) if n_frames else 0.0
    if n_frames < MIN_VALIDATION_FRAMES:
        return GlyphValidation(
            passed=False,
            n_frames=n_frames,
            n_correct=n_correct,
            accuracy=accuracy,
            reason=(
                f"only {n_frames} labelled post-grant frame(s) < required "
                f"{MIN_VALIDATION_FRAMES} — not enough to validate the glyph "
                "classifier; scale-up stays blocked"
            ),
        )
    if accuracy < MIN_VALIDATION_ACCURACY:
        return GlyphValidation(
            passed=False,
            n_frames=n_frames,
            n_correct=n_correct,
            accuracy=accuracy,
            reason=(
                f"glyph accuracy {accuracy:.3f} < required {MIN_VALIDATION_ACCURACY} "
                f"({n_correct}/{n_frames} frames) — classifier not reliable enough; "
                "scale-up stays blocked"
            ),
        )
    return GlyphValidation(
        passed=True,
        n_frames=n_frames,
        n_correct=n_correct,
        accuracy=accuracy,
        reason=None,
    )


def glyph_classifier_is_validated(validation: GlyphValidation | None) -> bool:
    """The single boolean the scale-up firewall consults for the glyph gate.

    Pass the result through to
    :func:`~catan_rl.human_data.orientation.assert_scale_up_orientation_gates`'s
    ``glyph_classifier_validated`` argument. Returns ``True`` — flipping the
    scale-up gate to allowed — ONLY when a :class:`GlyphValidation` with
    ``passed=True`` is supplied. ``None`` (no validation ever run) or any
    ``passed=False`` validation returns ``False``, so the gate keeps raising
    :class:`~catan_rl.human_data.orientation.GlyphClassifierNotValidated` and the
    300-game harvest stays blocked (task spec: gate flips to allowed only when
    validated).
    """
    return validation is not None and validation.passed
