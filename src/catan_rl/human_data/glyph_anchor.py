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

**Per-game calibration (brief §5.13 / §13 / §14 — never a global constant).**
The card icons render in the SAME resource colour family as the board tiles, so
the reader is calibrated PER GAME from the board's own 19 hex HSV samples via
:func:`calibrate_glyph_palette` — the identical rank-relative convention
``board_cv.classify_resources`` uses (ORE = lowest-saturation cluster; the rest
hue-sorted and rank-sliced BRICK→WHEAT→SHEEP→WOOD). :func:`classify_glyph` /
:func:`classify_granted_glyphs` take that :class:`GlyphPalette` as an argument;
they never read absolute module constants for a real game. :data:`FALLBACK_PALETTE`
(built from the module-level prior centres) exists ONLY as a default for the
synthetic unit tests and as a last-resort prior — it is documented as such and
CANNOT pass real-corpus validation without a per-game palette.

**BEST-EFFORT honesty (brief §5 / task spec).** The card glyphs are ~14px,
line-wrapped, and abut the adjacent log text. :func:`classify_glyph` returns
``None`` — an honest "could not read" — whenever a swatch is not clearly a card
body:

- too dark (below :data:`MIN_GLYPH_VALUE`) — log text / background bleed;
- too bright / washed out (above :data:`MAX_GLYPH_VALUE` with near-zero
  saturation) — white log text or a blank inter-icon gap, NOT a grey ORE card;
- a hue-classified swatch whose nearest and runner-up card centres are within
  :data:`MIN_GLYPH_HUE_MARGIN` (an ambiguous between-families colour);
- an impure swatch whose body pixels do not agree with the chosen class (the
  abutting-text / mis-boxed case) — :func:`_glyph_median_hsv` reports a purity
  fraction and :func:`classify_glyph` rejects a bimodal swatch.

The ORE branch is NOT a catch-all: it requires low saturation AND a value inside
a card-body band AND enough body purity, so a white/near-white swatch fails
closed rather than being labelled a confident ORE. :func:`classify_granted_glyphs`
returns ``None`` if ANY glyph is unreadable, and :func:`validate_glyph_classifier`
only reports ``passed=True`` when the classifier reproduces the labelled grants on
enough frames at the pre-registered accuracy bar. The scale-up gate is wired to
that validation via :func:`glyph_classifier_is_validated`, so the 300-game batch
harvest stays BLOCKED until a genuinely-validated classifier exists — the firewall
is never faked validated.

CPU-only; ``cv2`` is imported lazily. Never imports ``gui/`` or the training path
(brief §6).
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

#: The four HUE-classified granted resources (ORE is the desaturated branch), in
#: ascending-hue rank order — the same slice order ``board_cv.classify_resources``
#: applies to the non-ORE hexes (BRICK reddest → WOOD greenest).
HUE_RESOURCES_BY_RANK: tuple[str, ...] = ("BRICK", "WHEAT", "SHEEP", "WOOD")

#: **FALLBACK PRIOR ONLY — not a per-game palette.** Canonical OpenCV hues
#: (0..180) of the four hue-classified resource card icons in one Colonist skin.
#: A real game MUST be classified against a :class:`GlyphPalette` calibrated from
#: that game's own board tiles (:func:`calibrate_glyph_palette`); these absolute
#: centres are used only to seed :data:`FALLBACK_PALETTE` for the synthetic unit
#: tests and cannot pass real-corpus validation across skins / compression.
RESOURCE_CARD_HUES: dict[str, float] = {
    "BRICK": 8.0,
    "WHEAT": 25.0,
    "SHEEP": 40.0,
    "WOOD": 70.0,
}

#: Fallback ORE saturation ceiling (0..255) for :data:`FALLBACK_PALETTE`. A card
#: glyph with saturation below the palette's ORE ceiling AND a value in the card
#: band is treated as ORE (a desaturated grey card has no meaningful hue). A real
#: game derives its own ceiling from its ORE tiles (:func:`calibrate_glyph_palette`).
ORE_MAX_SATURATION = 60.0

#: Minimum value (brightness) for a swatch to be a card glyph at all. A near-black
#: swatch (log text / background bleed, not a card icon) — whether desaturated
#: (grey-dark) or coloured-dark — is unreadable, so :func:`classify_glyph` returns
#: ``None`` and the read fails closed.
MIN_GLYPH_VALUE = 90.0

#: Maximum value (brightness) for the ORE branch. A near-white swatch (val above
#: this AND near-zero saturation) is bright log text / a blank inter-icon gap, NOT
#: a grey stone card — the ORE branch must fail closed on it rather than emit a
#: confident ``ORE`` (red-team: ``(0,0,255)`` must be ``None``, not ``ORE``).
MAX_GLYPH_VALUE = 240.0

#: Minimum saturation (0..255) an ORE-branch swatch must have to be a grey stone
#: card rather than pure white UI. Stone icons carry a faint blue-grey tint; pure
#: white / near-white text sits below this and fails closed.
MIN_ORE_SATURATION = 8.0

#: Minimum OpenCV-hue gap (0..180 scale) between a glyph's nearest and second-
#: nearest resource-card hue centre for a HUE-classified (non-ORE) glyph to be
#: accepted. Below this the glyph is ambiguous (its colour sits between two card
#: families) and :func:`classify_glyph` returns ``None`` — the BEST-EFFORT honest
#: "could not read reliably" that keeps the scale-up gate engaged (task spec).
MIN_GLYPH_HUE_MARGIN = 8.0

#: Minimum fraction of a swatch's bright pixels that must agree with the chosen
#: card class (hue within :data:`MIN_GLYPH_HUE_MARGIN` of the winning centre for
#: the hue branch, or low-saturation for the ORE branch) for the read to be
#: trusted. Below this the swatch is bimodal — abutting log text or a mis-landed
#: box straddling two icons — and :func:`classify_glyph` fails closed. This turns
#: text contamination into an honest ``None`` rather than a dragged median.
MIN_GLYPH_BODY_PURITY = 0.6

#: Pre-registered validation bars for :func:`validate_glyph_classifier`. The
#: classifier is only reported ``passed`` when it reproduces the labelled grants
#: on at least :data:`MIN_VALIDATION_FRAMES` labelled post-grant frames at
#: >= :data:`MIN_VALIDATION_ACCURACY` per-player-grant accuracy. These gate the
#: scale-up firewall flip (task spec: the gate flips to allowed ONLY when
#: validated), so they are deliberately strict — a mislabelled grant silently
#: welds a jointly-flipped board.
MIN_VALIDATION_FRAMES = 8
MIN_VALIDATION_ACCURACY = 0.95

#: Minimum agreeing single-frame reads for the per-game grant CONSENSUS
#: (:func:`consensus_granted_glyphs`). The grant line persists across several
#: sampled frames; a mid-animation / compression-artefact frame must not drive the
#: multiset that feeds the joint-flip firewall, so a per-game mode vote requires at
#: least this many identical non-``None`` reads. Mirrors the board's >=2-frame
#: agreement discipline (orientation.py / brief §5.2).
MIN_GRANT_CONSENSUS_FRAMES = 2


@dataclass(frozen=True, slots=True)
class GlyphPalette:
    """Per-game card-glyph colour calibration (brief §5.13 — never a constant).

    ``hue_centres`` maps each of the four HUE-classified resources to that game's
    own card-icon hue (OpenCV 0..180), clustered from the board tiles the same way
    ``board_cv.classify_resources`` clusters the non-ORE hexes (hue-sorted, rank-
    sliced BRICK→WHEAT→SHEEP→WOOD). ``ore_max_saturation`` is the game's ORE-vs-
    coloured saturation boundary derived from its own ORE tiles. Build one per game
    with :func:`calibrate_glyph_palette`; :data:`FALLBACK_PALETTE` is the synthetic-
    test / last-resort prior only.
    """

    hue_centres: dict[str, float]
    ore_max_saturation: float


def calibrate_glyph_palette(
    board_samples: npt.NDArray[np.float64], desert_hex: int
) -> GlyphPalette:
    """Derive a per-game :class:`GlyphPalette` from the game's own board tiles.

    ``board_samples`` is the 19x3 median-HSV array ``board_cv.read_board`` already
    computes for this game (engine hex order); ``desert_hex`` the locked desert.
    The card icons share the tile colour family, so we cluster exactly as
    ``board_cv.classify_resources`` does — RELATIVE ranks, never absolute centres:

    - **ORE** = the 3 lowest-saturation non-desert hexes. The ORE saturation
      ceiling for the glyph branch is set just above those hexes' max saturation
      (midway to the next-lowest coloured hex), so a real game's own grey band —
      not a hardcoded 60 — decides ORE.
    - the remaining 15 are hue-sorted and rank-sliced into BRICK[0:3] / WHEAT[3:7]
      / SHEEP[7:11] / WOOD[11:15]; each resource's hue centre is the MEDIAN hue of
      its slice.

    This makes the palette track skin / theme / compression drift per game. Raises
    :class:`ValueError` on a malformed ``desert_hex`` (mirrors board_cv).
    """
    n = int(board_samples.shape[0])
    if not 0 <= desert_hex < n:
        raise ValueError(f"desert_hex {desert_hex} out of 0..{n - 1}")
    idxs = [i for i in range(n) if i != desert_hex]
    sats = np.array([board_samples[i][1] for i in idxs], dtype=np.float64)
    ore_local = np.argsort(sats)[:3].tolist()
    ore_idxs = {idxs[j] for j in ore_local}
    ore_sat_max = float(max(board_samples[i][1] for i in ore_idxs))

    rest = sorted((i for i in idxs if i not in ore_idxs), key=lambda i: float(board_samples[i][0]))
    lowest_coloured_sat = min(float(board_samples[i][1]) for i in rest) if rest else ore_sat_max
    # ORE ceiling: above the ORE tiles' saturation, below the least-saturated
    # coloured tile — a per-game boundary, not a global 60.
    ore_ceiling = (ore_sat_max + lowest_coloured_sat) / 2.0

    hue_centres: dict[str, float] = {}
    slices = {"BRICK": rest[0:3], "WHEAT": rest[3:7], "SHEEP": rest[7:11], "WOOD": rest[11:15]}
    for resource, slice_idxs in slices.items():
        if slice_idxs:
            hue_centres[resource] = float(
                np.median([float(board_samples[i][0]) for i in slice_idxs])
            )
        else:  # pragma: no cover - a full 19-hex board always fills every slice
            hue_centres[resource] = RESOURCE_CARD_HUES[resource]
    return GlyphPalette(hue_centres=hue_centres, ore_max_saturation=ore_ceiling)


#: **Synthetic-test / last-resort prior ONLY.** A :class:`GlyphPalette` built from
#: the module-level fallback centres. It is the default when no per-game palette is
#: threaded into :func:`classify_glyph`; a REAL game must pass a
#: :func:`calibrate_glyph_palette` palette instead (this fallback cannot pass
#: real-corpus validation — brief §5.13 / §13).
FALLBACK_PALETTE = GlyphPalette(
    hue_centres=dict(RESOURCE_CARD_HUES),
    ore_max_saturation=ORE_MAX_SATURATION,
)


def _hue_distance(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (0..180 wraps)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _circular_hue_median(hues: npt.NDArray[np.float64]) -> float:
    """Circular central tendency of OpenCV hues (0..180), wrap-safe.

    A linear ``np.median`` breaks at the red 0/180 wrap: a BRICK glyph whose
    fringed body pixels straddle the wrap (some ~178, some ~6, both red) medians to
    ~90 (WOOD/SHEEP) and gets confidently mislabelled. We map each hue onto the
    unit circle at DOUBLE angle (0..180 → full 0..2π turn), take the mean vector,
    and read the resultant angle back — the standard circular mean, robust to the
    wrap. Empty input returns 0.0.
    """
    if hues.size == 0:  # pragma: no cover - callers guard against empty bodies
        return 0.0
    angles = hues * (2.0 * np.pi / 180.0)
    mean_cos = float(np.mean(np.cos(angles)))
    mean_sin = float(np.mean(np.sin(angles)))
    ang = np.arctan2(mean_sin, mean_cos)
    if ang < 0.0:
        ang += 2.0 * np.pi
    return float(ang * (180.0 / (2.0 * np.pi)))


@dataclass(frozen=True, slots=True)
class _GlyphStats:
    """Robust body statistics of one candidate card-glyph swatch.

    ``hue`` is the circular-median hue over the kept body pixels; ``sat`` / ``val``
    their linear medians; ``hues`` the kept body hues (for the hue-branch purity
    vote); ``sats`` the kept body saturations (for the ORE-branch purity vote).
    ``ok`` is ``False`` when no body pixel survived the two-sided card-body mask
    (an all-dark or all-bright swatch), so :func:`classify_glyph` fails closed.
    """

    ok: bool
    hue: float
    sat: float
    val: float
    hues: npt.NDArray[np.float64]
    sats: npt.NDArray[np.float64]


def classify_glyph(
    hsv: tuple[float, float, float] | _GlyphStats,
    palette: GlyphPalette = FALLBACK_PALETTE,
) -> str | None:
    """Classify one card-glyph into a granted resource literal, or ``None`` when it
    cannot be read RELIABLY (BEST-EFFORT honesty).

    ``hsv`` is either the robust body statistics of a swatch (a :class:`_GlyphStats`
    from :func:`_glyph_median_hsv`, used on real frames) or a bare
    ``(hue, sat, val)`` median tuple (OpenCV hue 0..180, sat/val 0..255 — used by
    the synthetic tests and callers with a single colour). ``palette`` is the
    PER-GAME calibration (:func:`calibrate_glyph_palette`); it defaults to the
    synthetic-test :data:`FALLBACK_PALETTE`, which a real game must NOT rely on.

    Returns ``None`` — never a guess — when:

    - the swatch is too dark to be a vivid/bright card icon (below
      :data:`MIN_GLYPH_VALUE`) — log text / background bleed;
    - the ORE branch's swatch is near-white (value above :data:`MAX_GLYPH_VALUE`
      with saturation below :data:`MIN_ORE_SATURATION`) — white log text / a blank
      gap, not a grey stone card;
    - a hue-classified swatch's nearest and runner-up card hue centres are within
      :data:`MIN_GLYPH_HUE_MARGIN` (an ambiguous between-families colour);
    - the body pixels do not agree with the chosen class at
      :data:`MIN_GLYPH_BODY_PURITY` (a bimodal / text-abutting swatch).

    ORE is the desaturated-grey branch (saturation below the palette's ORE ceiling,
    value inside the card band) — a BRIGHT-BUT-NOT-WHITE grey card — mirroring
    board_cv's ORE-by-saturation rule; the other four are wrap-aware nearest-hue
    with the margin + purity gates.
    """
    stats = hsv if isinstance(hsv, _GlyphStats) else _stats_from_tuple(hsv)
    if not stats.ok:
        return None
    hue, sat, val = stats.hue, stats.sat, stats.val
    if val < MIN_GLYPH_VALUE:
        return None  # too dark to be a card icon (log text / gap)
    if sat < palette.ore_max_saturation:
        # ORE branch — but fail closed on near-white (bright, ~0 saturation) UI /
        # text and on a value outside the card body band, and require body purity.
        if sat < MIN_ORE_SATURATION and val > MAX_GLYPH_VALUE:
            return None  # near-white log text / blank gap, not a grey stone card
        ore_purity = _ore_purity(stats.sats, palette.ore_max_saturation)
        if ore_purity < MIN_GLYPH_BODY_PURITY:
            return None  # bimodal — abutting text / mis-boxed, not a clean grey card
        return "ORE"
    ranked = sorted(
        palette.hue_centres,
        key=lambda r: _hue_distance(hue, palette.hue_centres[r]),
    )
    nearest, runner_up = ranked[0], ranked[1]
    margin = _hue_distance(hue, palette.hue_centres[runner_up]) - _hue_distance(
        hue, palette.hue_centres[nearest]
    )
    if margin < MIN_GLYPH_HUE_MARGIN:
        return None  # between two card families — ambiguous, fail closed
    purity = _hue_purity(stats.hues, palette.hue_centres[nearest])
    if purity < MIN_GLYPH_BODY_PURITY:
        return None  # bimodal — abutting text / mis-boxed, not a clean card body
    return nearest


def _stats_from_tuple(hsv: tuple[float, float, float]) -> _GlyphStats:
    """Wrap a single ``(hue, sat, val)`` median as trivially-pure :class:`_GlyphStats`.

    Used by callers (and the synthetic tests) that pass one solid colour rather
    than a pixel swatch: the body is a single sample, so purity is 1.0 by
    construction. ``ok`` is ``True`` even for a dark tuple so the value floor in
    :func:`classify_glyph` (not this wrapper) makes the dark→``None`` decision.
    """
    hue, sat, val = hsv
    return _GlyphStats(
        ok=True,
        hue=hue,
        sat=sat,
        val=val,
        hues=np.array([hue], dtype=np.float64),
        sats=np.array([sat], dtype=np.float64),
    )


def _hue_purity(hues: npt.NDArray[np.float64], centre: float) -> float:
    """Fraction of body hues within :data:`MIN_GLYPH_HUE_MARGIN` of ``centre``."""
    if hues.size == 0:  # pragma: no cover - callers guard against empty bodies
        return 0.0
    dists = np.array([_hue_distance(float(h), centre) for h in hues], dtype=np.float64)
    return float(np.mean(dists <= MIN_GLYPH_HUE_MARGIN))


def _ore_purity(sats: npt.NDArray[np.float64], ceiling: float) -> float:
    """Fraction of body saturations below the ORE ceiling (grey-card agreement)."""
    if sats.size == 0:  # pragma: no cover - callers guard against empty bodies
        return 0.0
    return float(np.mean(sats < ceiling))


def _glyph_median_hsv(glyph_rgb: npt.NDArray[np.uint8]) -> _GlyphStats:
    """Robust body statistics over the CARD-BODY pixels of one glyph swatch.

    A one-sided value floor is not enough: it drops dark background but KEEPS bright
    white/grey log text abutting the icon, which then drags the median (a genuine
    WHEAT can median to BRICK/ORE). We keep a two-sided card-BODY band — pixels
    bright enough to be a card yet not near-white text (value in
    ``[MIN_GLYPH_VALUE, MAX_GLYPH_VALUE]`` OR value >= floor with meaningful
    saturation) — then report the CIRCULAR hue median (wrap-safe) plus the kept
    hues/sats so :func:`classify_glyph` can vote body purity and turn an impure /
    text-abutting swatch into an honest ``None`` instead of a confident mislabel.
    """
    import cv2

    hsv = np.asarray(cv2.cvtColor(glyph_rgb, cv2.COLOR_RGB2HSV), np.uint8)
    flat = hsv.reshape(-1, 3).astype(np.float64)
    val_c = flat[:, 2]
    sat_c = flat[:, 1]
    # Card-body pixels: bright enough to be a card, and either coloured (has
    # saturation) or a mid-bright grey ORE body — but NOT near-white text (bright
    # AND ~0 saturation), which we exclude two-sidedly.
    bright = val_c >= MIN_GLYPH_VALUE
    not_white_text = ~((val_c > MAX_GLYPH_VALUE) & (sat_c < MIN_ORE_SATURATION))
    keep = bright & not_white_text
    if not bool(keep.any()):
        return _GlyphStats(
            ok=False,
            hue=0.0,
            sat=0.0,
            val=0.0,
            hues=np.empty(0, dtype=np.float64),
            sats=np.empty(0, dtype=np.float64),
        )
    body = flat[keep]
    hues = body[:, 0]
    sats = body[:, 1]
    vals = body[:, 2]
    return _GlyphStats(
        ok=True,
        hue=_circular_hue_median(hues),
        sat=float(np.median(sats)),
        val=float(np.median(vals)),
        hues=hues,
        sats=sats,
    )


def classify_granted_glyphs(
    log_crop_rgb: npt.NDArray[np.uint8],
    glyph_boxes: list[tuple[int, int, int, int]],
    palette: GlyphPalette = FALLBACK_PALETTE,
) -> Counter[str] | None:
    """Colour-classify the granted resource card glyphs for ONE player into a
    resource multiset, or ``None`` if ANY glyph cannot be read reliably.

    ``log_crop_rgb`` is the RGB log-panel crop of a frame captured right after the
    player's ``"received starting resources"`` event; ``glyph_boxes`` are the
    ``(x0, y0, x1, y1)`` pixel boxes of that line's card icons (from the glyph
    detector — not this function's concern; a box per granted card). ``palette`` is
    the PER-GAME calibration (:func:`calibrate_glyph_palette`); it defaults to the
    synthetic-test :data:`FALLBACK_PALETTE`. Returns a ``Counter`` over the granted
    resource literals on success.

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
        resource = classify_glyph(_glyph_median_hsv(swatch), palette)
        if resource is None:
            return None  # one unreadable glyph → the whole grant is unreliable
        granted[resource] += 1
    return granted


def consensus_granted_glyphs(
    frames: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]],
    palette: GlyphPalette = FALLBACK_PALETTE,
) -> Counter[str] | None:
    """Per-game CONSENSUS grant read across every frame the grant line is visible.

    The ``"received starting resources"`` line persists across several sampled
    frames; trusting a single crop lets one mid-animation / compression-artefact
    frame drive the multiset the joint-flip firewall compares against. This mirrors
    the board's byte-identical agreement rule
    (:func:`~catan_rl.human_data.board_cv.read_board_stable`, brief §5.2): read the
    grant crop on EVERY frame it is visible, drop the ``None`` (unreadable) reads,
    and require the non-``None`` reads to ALL agree on one multiset backed by
    >= :data:`MIN_GRANT_CONSENSUS_FRAMES` frames. ANY disagreement among the
    readable reads is an honest reject (``None``) — like ``read_board_stable``,
    consensus never tie-breaks a bimodal read, because a wrong-but-confident grant
    multiset would either false-accept or false-reject the joint-D6-flip firewall
    (:func:`~catan_rl.human_data.orientation.assert_glyph_anchor`). Returns the
    agreed multiset, or ``None`` when reads disagree or too few agree.
    """
    reads = [
        r
        for frame_rgb, boxes in frames
        if (r := classify_granted_glyphs(frame_rgb, boxes, palette)) is not None
    ]
    if not reads:
        return None
    # Full agreement, not a plurality: every readable frame must read the SAME
    # multiset. A single disagreeing frame is exactly the instability the multi-
    # frame gate exists to catch — fail closed rather than tie-break a coin flip.
    distinct = {tuple(sorted(r.items())) for r in reads}
    if len(distinct) > 1:
        return None  # readable reads disagree → bimodal/unstable, fail closed
    if len(reads) < MIN_GRANT_CONSENSUS_FRAMES:
        return None  # unanimous but too few frames agree — unreliable, fail closed
    return Counter(dict(distinct.pop()))


@dataclass(frozen=True, slots=True)
class LabeledGrantFrame:
    """One labelled post-grant frame for :func:`validate_glyph_classifier`.

    ``log_crop_rgb`` is the RGB log-panel crop right after a
    ``"received starting resources"`` event; ``glyph_boxes`` the granted card
    boxes for the ONE player that line grants; ``expected`` the hand-labelled
    granted multiset (ground truth); ``palette`` the per-game glyph calibration
    (defaults to the synthetic-test :data:`FALLBACK_PALETTE`). The classifier is
    scored per frame: a frame is CORRECT iff :func:`classify_granted_glyphs`
    returns exactly ``expected`` under that palette.
    """

    log_crop_rgb: npt.NDArray[np.uint8]
    glyph_boxes: list[tuple[int, int, int, int]]
    expected: Counter[str]
    palette: GlyphPalette = FALLBACK_PALETTE


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

    Runs :func:`classify_granted_glyphs` on each frame UNDER THAT FRAME'S per-game
    palette and counts a frame CORRECT iff the returned multiset equals the frame's
    ``expected`` label (an unreadable ``None`` counts as wrong — an honest miss, not
    a pass). Reports ``passed=True`` only when both pre-registered bars clear
    (:data:`MIN_VALIDATION_FRAMES`, :data:`MIN_VALIDATION_ACCURACY`); otherwise
    ``passed=False`` with a ``reason``. This is the sole switch the scale-up
    firewall consults (:func:`glyph_classifier_is_validated`), so a below-bar or
    under-sampled run keeps the 300-game harvest BLOCKED.
    """
    n_frames = len(frames)
    n_correct = sum(
        1
        for f in frames
        if classify_granted_glyphs(f.log_crop_rgb, f.glyph_boxes, f.palette) == f.expected
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
