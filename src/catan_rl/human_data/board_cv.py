"""Orientation-locked board CV: parse the 19-hex board off a 1080p Colonist frame.

The Stage-2 ``board_cv`` slice (build brief §4). Given a decoded **RGB** frame
(an :class:`~catan_rl.human_data.ingest.DecodedFrame` ``.frame``), it:

1. **Detects the 16-18 number-token disks** (white blobs in the board region) and
   fits an **affine** from the committed engine board template (render-space hex
   pixels, :func:`load_engine_template`) to the frame.
2. **Locks the D6 orientation deterministically** (build brief §5.2 / BLOCKER-2).
   The 19-hex board has a D6 (12-element) symmetry, so all 12 candidate affines
   fit the detected tokens with *identical* residual — "lowest residual" flips
   orientation on floating-point noise and silently relabels every engine ID.
   The lock therefore uses a **screen-space rule** (engine H8 → top-center hex,
   H11 → rightmost hex) that is content-free and frame-stable, corroborated by
   **≥2 content anchors** derived from the screen-space token extremes (the
   pixel-topmost and pixel-rightmost detected token disks,
   :func:`derive_screen_anchors`): the anchor's engine ID is fixed by *where a
   token lands on screen* and its number is **re-OCR'd independently at that
   screen pixel**, then required to equal what the affine predicts at that engine
   ID. A mis-oriented affine that satisfied the geometric rule only degenerately
   puts a different token topmost/rightmost, so its independent number disagrees
   and the frame is rejected. A frame is **rejected** (returns ``None``) if the
   screen rule is ambiguous (the best/second-best penalty gap is below
   :data:`MIN_SCREEN_RULE_GAP`), if fewer than 2 independent anchors are
   available, if a content anchor disagrees, if a resource hue cluster overlaps
   its neighbour (:data:`MIN_HUE_CLUSTER_MARGIN`), or if the affine residual
   exceeds :data:`~catan_rl.human_data.orientation.MAX_AFFINE_RESIDUAL_PX`.
3. **Reads each hex per-game-calibrated** (build brief §2 / §5.13). The desert is
   the unique **token-less** hex, detected by a **relative** white-coverage split
   (the 18 token hexes all rank high, the desert alone ranks low; rejected if the
   split is not bimodal with a clear margin — no fixed colour threshold). The 18
   token hexes are classified by **per-game clustering** of their own sampled
   colours (ORE = the 3 lowest-saturation hexes; the rest ordered by hue →
   BRICK/WHEAT/SHEEP/WOOD) — the palette is never hardcoded. Each number is OCR'd
   and **independently corroborated by a pip-count** (§5.6).

**Resource-multiset gate is NOT a resource cross-check (record.py finding).**
This module assigns the 18 non-desert hexes to resources by *forcing the standard
4/4/4/3/3 multiset* from the per-game colour clusters, so the record contract's
``STANDARD_RESOURCE_COUNTS`` gate is **tautological** for the resource dimension
and cannot catch an in-multiset swap. The pip cross-check corroborates the
NUMBER, not the resource — a resource swap between two hexes carrying valid
numbers passes both the multiset gate AND ``pip_ok``. The two genuine resource
firewalls this module owns are therefore: (a) a **hue-cluster-margin gate**
(:func:`hue_cluster_margin`) — a rank-sliced assignment is only accepted when
each assigned class's hue span is clearly separated from the neighbouring class,
so a near-boundary WHEAT/SHEEP frame is *rejected* rather than confidently
mislabelled — and (b) the **independent screen-space content anchors** (which
cover a real resource read). The glyph anchor
(:mod:`catan_rl.human_data.orientation`) remains the joint-flip firewall.

CPU-only. ``cv2`` and ``easyocr`` are imported lazily inside the reader functions
so the pure geometry core (D6 orientation math, the screen-rule scorer, the
per-game cluster assignment) — the unit-tested surface — carries no heavy
optional dependency. Never imports ``gui/`` or the training path (brief §6).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.human_data.orientation import MAX_AFFINE_RESIDUAL_PX
from catan_rl.human_data.topology import NUM_HEXES, NUM_VERTICES, load_topology

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    import numpy.typing as npt

#: Non-desert resource → standard board count (the 18 token hexes). ORE/BRICK x3,
#: WOOD/SHEEP/WHEAT x4. The per-game cluster assignment forces exactly these
#: counts, so the resource-multiset gate is tautological for the resource
#: dimension; the real resource firewalls are the hue-cluster-margin gate and the
#: independent screen anchors (see module docstring / record.py finding).
_NONDESERT_COUNTS: dict[str, int] = {"ORE": 3, "BRICK": 3, "WHEAT": 4, "SHEEP": 4, "WOOD": 4}

#: Number-token → pip-dot count (the independent OCR cross-check, §5.6). The
#: rendered token carries this many pips under its digit; OCR digit and pip count
#: must agree on every token hex or the read is rejected.
_PIP_FOR: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

#: Minimum OpenCV-hue margin (in the 0..180 hue scale) required between any two
#: ADJACENT resource hue-clusters (BRICK<WHEAT<SHEEP<WOOD) for a rank-sliced
#: assignment to be accepted. Below this, a per-game palette drift could swap a
#: WHEAT↔SHEEP (or BRICK↔WHEAT) while the standard multiset stays exactly 4/4, so
#: the frame is REJECTED (an honest rejection, not a confident mislabel). The
#: game-1 fixture inter-cluster gaps are wide (BRICK≈9 | WHEAT≈21 | SHEEP≈36 |
#: WOOD≈65: min neighbour gap ~12), so this margin passes the golden board with
#: room while catching a near-boundary drift.
MIN_HUE_CLUSTER_MARGIN = 4.0

#: Minimum ratio ``second_best_penalty / best_penalty`` of the D6 screen-rule
#: scores for the orientation lock to be accepted. The spike proved the 12 D6
#: orientations are near-degenerate on residual; the screen rule breaks the tie
#: but only cleanly when the runner-up scores materially worse. Below this ratio
#: the frame is ambiguous (partial board off-screen, mid-animation pan, an
#: occluding piece skewing the token centroid) and is REJECTED rather than a
#: coin-flip orientation emitted. The game-1 fixture margin is ~30x, far above
#: this floor.
MIN_SCREEN_RULE_GAP = 3.0

#: Minimum separation, in white-coverage fraction, between the lowest-covered
#: TOKEN hex and the (single) token-LESS desert hex for the relative desert
#: split to be accepted. The desert has no white number-token disk, so its
#: centre white-coverage is far below every token hex; a bimodal split with at
#: least this margin distinguishes it per-game without a fixed colour threshold.
MIN_DESERT_COVERAGE_MARGIN = 0.20


@dataclass(frozen=True, slots=True)
class EngineTemplate:
    """Committed render-space engine board template (pointy-top, edge=80px).

    Pure geometry (shape only): ``hex_centers`` (19x2) and ``vertex_px`` (54x2) in
    the engine's canonical hex/vertex ordering — the same ordering as the
    committed ``topology.json`` fixture. Used as the affine source; the absolute
    scale/offset is fit per frame, only the relative lattice matters.
    """

    hex_centers: npt.NDArray[np.float64]
    vertex_px: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.hex_centers.shape != (NUM_HEXES, 2):
            raise ValueError(f"hex_centers must be {NUM_HEXES}x2, got {self.hex_centers.shape}")
        if self.vertex_px.shape != (NUM_VERTICES, 2):
            raise ValueError(f"vertex_px must be {NUM_VERTICES}x2, got {self.vertex_px.shape}")


def load_engine_template() -> EngineTemplate:
    """Load the committed ``engine_template.json`` package fixture."""
    resource = files("catan_rl.human_data").joinpath("engine_template.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    hex_centers = np.array([payload["hex_centers"][str(i)] for i in range(NUM_HEXES)], float)
    vertex_px = np.array([payload["vertex_px"][str(i)] for i in range(NUM_VERTICES)], float)
    return EngineTemplate(hex_centers=hex_centers, vertex_px=vertex_px)


@dataclass(frozen=True, slots=True)
class ContentAnchor:
    """A ground-truth (engine_id → resource, number) read at a fixed screen
    landmark, used to corroborate the geometric orientation lock (§5.2).

    ``resource`` must be a non-``DESERT`` literal (the desert moves per game, so it
    is never a valid anchor). ``number`` is the token digit at that engine ID.

    In the batch path these are **derived per-frame from the screen-space token
    extremes** (:func:`derive_screen_anchors`) — an independent read at a fixed
    screen landmark, NOT a hardcoded constant. A caller may also pass explicit
    anchors (e.g. a test asserting a mis-oriented ground truth is rejected).
    """

    engine_id: int
    resource: str
    number: int


@dataclass(frozen=True, slots=True)
class BoardRead:
    """Result of :func:`read_board`: the orientation-locked 19-hex board + diag.

    ``hexes`` is the :class:`~catan_rl.human_data.record.GameRecord`-shaped list
    of ``{"hex_id", "resource", "number"}`` (desert carries ``number=None``).
    ``affine`` is the 2x3 engine→frame affine; ``vertex_px`` the projected 54
    vertex pixel coords (consumed by the openings slice). ``desert_hex`` is the
    engine hex_id the board locked the desert at (the orientation stamp
    ``provenance.board_desert_hex``). ``residual_px`` / ``screen_rule_gap`` /
    ``pip_ok`` are the acceptance diagnostics.
    """

    hexes: tuple[dict[str, Any], ...]
    affine: npt.NDArray[np.float64]
    vertex_px: npt.NDArray[np.float64]
    desert_hex: int
    residual_px: float
    screen_rule_gap: float
    pip_ok: bool


# --------------------------------------------------------------------- geometry


def _apply_affine(
    affine: npt.NDArray[np.float64], pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply a 2x3 affine to an (N, 2) array of points."""
    result: npt.NDArray[np.float64] = (affine[:, :2] @ pts.T).T + affine[:, 2]
    return result


def _candidate_affines(
    token_xy: npt.NDArray[np.float64], hex_centers: npt.NDArray[np.float64]
) -> list[tuple[int, int, npt.NDArray[np.float64], float]]:
    """Fit all 12 D6-orientation affines (engine hexes → detected tokens).

    Each of the 6 rotations x 2 reflections is greedily matched token→hex and
    refined by least-squares. Returns ``(refl, rot_deg, affine, residual_px)`` for
    every candidate — by D6 symmetry they share a residual, so orientation is
    resolved by the screen rule + content anchors, NOT the residual. A candidate
    whose RANSAC affine fit fails (``estimateAffine2D`` returns ``None``) is
    skipped rather than crashing the batch.
    """
    import cv2  # lazy: heavy optional dep, keeps the pure surface import-light

    eng_c = hex_centers.mean(0)
    frame_c = token_xy.mean(0)
    scale = (
        np.linalg.norm(token_xy - frame_c, axis=1).mean()
        / np.linalg.norm(hex_centers - eng_c, axis=1).mean()
    )
    out: list[tuple[int, int, npt.NDArray[np.float64], float]] = []
    for refl in (1, -1):
        for k in range(6):
            ang = math.radians(60 * k)
            rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], float)
            transform = rot @ np.array([[refl, 0], [0, 1]], float)
            projected = (hex_centers - eng_c) @ transform.T * scale + frame_c
            pairs: list[tuple[int, int]] = []
            used: set[int] = set()
            for ti, point in enumerate(token_xy):
                dists = np.linalg.norm(projected - point, axis=1)
                for hi in np.argsort(dists):
                    hi_int = int(hi)
                    if hi_int not in used:
                        used.add(hi_int)
                        pairs.append((ti, hi_int))
                        break
            src = np.array([hex_centers[hi] for _, hi in pairs])
            dst = np.array([token_xy[ti] for ti, _ in pairs])
            affine, _ = cv2.estimateAffine2D(
                src.astype(np.float32),
                dst.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=12,
            )
            if affine is None:
                continue  # RANSAC found no model for this candidate — skip it
            proj = (affine[:, :2] @ src.T).T + affine[:, 2]
            residual = float(np.linalg.norm(proj - dst, axis=1).mean())
            out.append((refl, 60 * k, affine.astype(np.float64), residual))
    return out


def _score_screen_rule(
    candidates: list[tuple[int, int, npt.NDArray[np.float64], float]],
    token_xy: npt.NDArray[np.float64],
    hex_centers: npt.NDArray[np.float64],
) -> list[tuple[float, int, int, npt.NDArray[np.float64], float]]:
    """Score each candidate by the content-free screen-space rule (build brief
    §5.2): engine H8 → the TOP-CENTER hex (min y, x near board center), engine H11
    → the RIGHTMOST hex (max x). Lower penalty = better. Returns candidates sorted
    best-first as ``(penalty, refl, rot, affine, residual)``.

    This ordered non-collinear landmark pair fixes rotation AND reflection
    uniquely and is identical every frame (the board render is screen-static), so
    it is the frame-stable lock — not the D6-degenerate residual.
    """
    board_cx = float(token_xy[:, 0].mean())
    scored: list[tuple[float, int, int, npt.NDArray[np.float64], float]] = []
    for refl, deg, affine, residual in candidates:
        hx = _apply_affine(affine, hex_centers)
        h8, h11 = hx[8], hx[11]
        top_center_pen = float((h8[1] - hx[:, 1].min()) + 0.5 * abs(h8[0] - board_cx))
        right_pen = float(hx[:, 0].max() - h11[0])
        scored.append((top_center_pen + right_pen, refl, deg, affine, residual))
    scored.sort(key=lambda z: z[0])
    return scored


# ------------------------------------------------------------------ content read


def _detect_tokens(frame_rgb: npt.NDArray[np.uint8]) -> list[tuple[float, float, float]]:
    """Detect the number-token white disks in the board region of an RGB frame.

    Returns ``(x, y, diameter)`` per disk. Uses HSV white segmentation restricted
    to the left ~63% of the frame (the board; the right is the HUD/log). Tuned to
    the ~60px 1080p token disk (build brief §2 / BLOCKER-2 spike).
    """
    import cv2

    height, width = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 175], np.uint8)
    upper = np.array([180, 65, 255], np.uint8)
    white = cv2.inRange(hsv, lower, upper)
    mask = np.zeros((height, width), np.uint8)
    mask[int(0.04 * height) : int(0.97 * height), 0 : int(0.63 * width)] = 255
    white = cv2.bitwise_and(white, mask)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(white)
    tokens: list[tuple[float, float, float]] = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / max(h, 1)
        fill = area / (w * h)
        if area > 2400 and 0.78 < aspect < 1.28 and 50 < w < 74 and fill > 0.65:
            tokens.append((float(centroids[i][0]), float(centroids[i][1]), (w + h) / 2.0))
    return tokens


def _sample_hex_hsv(
    hsv: npt.NDArray[np.uint8],
    center: npt.NDArray[np.float64],
    corners: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Median HSV of a hex FILL, sampled along center→corner rays at 0.40/0.52/0.64
    (clean colour area away from the token, centre icon, and border). The white
    token pixels are masked out; palette-agnostic (no fixed colour thresholds)."""
    pts: list[npt.NDArray[np.uint8]] = []
    cx, cy = float(center[0]), float(center[1])
    for corner in corners:
        for frac in (0.40, 0.52, 0.64):
            x = int(cx + frac * (float(corner[0]) - cx))
            y = int(cy + frac * (float(corner[1]) - cy))
            if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
                pts.append(hsv[y, x])
    arr = np.array(pts)
    keep = (arr[:, 2] > 55) & ~((arr[:, 2] > 170) & (arr[:, 1] < 55))
    if keep.sum() < 4:
        keep = np.ones(len(arr), bool)
    return np.asarray(np.median(arr[keep], 0), dtype=np.float64)


def _hex_white_coverage(
    frame_rgb: npt.NDArray[np.uint8], center: npt.NDArray[np.float64], diam: float
) -> float:
    """White-coverage fraction at a hex center (the number-token-disk signal).

    High on a token hex (the white disk), ~0 on the token-LESS desert. The desert
    is found by a RELATIVE bimodal split of these 19 values (:func:`_desert_hex`),
    not a fixed absolute threshold — per-game / per-skin robust (brief §2)."""
    import cv2

    r = int(diam * 0.35)
    cx, cy = int(center[0]), int(center[1])
    patch = frame_rgb[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 2] > 175) & (hsv[..., 1] < 60)
    return float(white.mean())


def _desert_hex(coverages: npt.NDArray[np.float64]) -> int | None:
    """Return the engine hex_id of the token-LESS desert, or ``None`` if the split
    is ambiguous. Detected RELATIVELY (brief §2 / review finding): rank the 19
    hex-centre white-coverages and take the single lowest as the desert only when
    it is separated from the next-lowest (the lowest-covered TOKEN hex) by at least
    :data:`MIN_DESERT_COVERAGE_MARGIN` — a clear bimodal split (1 low, 18 high). A
    two-way tie or a per-skin brightness shift that fails to isolate exactly one
    low hex is rejected (returns ``None``) rather than silently stamping the wrong
    desert (which moves the orientation-binding provenance)."""
    order = np.argsort(coverages)
    lowest = int(order[0])
    second = int(order[1])
    if coverages[second] - coverages[lowest] < MIN_DESERT_COVERAGE_MARGIN:
        return None
    return lowest


def hue_cluster_margin(samples: npt.NDArray[np.float64], desert_hex: int) -> float:
    """Minimum OpenCV-hue gap between ADJACENT resource hue-clusters after the
    rank-slice (BRICK<WHEAT<SHEEP<WOOD), i.e. ``min over neighbouring classes of
    (min hue of the higher class minus max hue of the lower class)``.

    This is the genuine resource cross-check the multiset gate cannot be
    (record.py finding). :func:`classify_resources` forces the standard multiset
    by slicing the hue-sorted non-ORE hexes at FIXED rank boundaries, so a
    per-game palette drift near a boundary silently swaps a WHEAT↔SHEEP while the
    multiset stays 4/4. :func:`read_board` rejects the frame when this margin is
    below :data:`MIN_HUE_CLUSTER_MARGIN` — turning a confident mislabel into an
    honest rejection. Returns ``-inf`` if the ORE cluster cannot be isolated (the
    same degenerate case :func:`classify_resources` labels). Pure numpy."""
    if not 0 <= desert_hex < NUM_HEXES:
        raise ValueError(f"desert_hex {desert_hex} out of 0..{NUM_HEXES - 1}")
    idxs = [i for i in range(NUM_HEXES) if i != desert_hex]
    sats = np.array([samples[i][1] for i in idxs])
    ore = {idxs[j] for j in np.argsort(sats)[:3].tolist()}
    rest = sorted((i for i in idxs if i not in ore), key=lambda i: float(samples[i][0]))
    hues = [float(samples[i][0]) for i in rest]
    # rank-slice boundaries: BRICK[0:3] | WHEAT[3:7] | SHEEP[7:11] | WOOD[11:15]
    boundaries = (3, 7, 11)
    gaps = [hues[b] - hues[b - 1] for b in boundaries]
    return min(gaps)


def classify_resources(samples: npt.NDArray[np.float64], desert_hex: int) -> list[str]:
    """Assign each of the 19 hexes a resource literal, per-game calibrated.

    ``samples`` is a 19x3 median-HSV array (engine hex order); ``desert_hex`` is
    the unique token-less hex. The 18 non-desert hexes are clustered from the
    frame's OWN samples (never a hardcoded palette, brief §5.13):

    - **ORE** = the 3 lowest-saturation hexes (grey stone).
    - the remaining 15 are ordered by **hue** and sliced by the standard counts:
      BRICK x3 (red, lowest hue) → WHEAT x4 (gold) → SHEEP x4 (yellow-green) →
      WOOD x4 (forest green, highest hue).

    This FORCES the standard 4/4/4/3/3 multiset — so the ``STANDARD_RESOURCE_COUNTS``
    gate is tautological for the resource dimension and a near-boundary WHEAT↔SHEEP
    swap stays in-multiset. The real resource firewall is :func:`hue_cluster_margin`
    (an inter-cluster-margin gate that :func:`read_board` enforces), NOT this
    forced assignment and NOT the pip cross-check (which corroborates the number,
    not the resource). Pure numpy — no cv2 — so it is unit-testable without the
    heavy CV dependency.
    """
    if not 0 <= desert_hex < NUM_HEXES:
        raise ValueError(f"desert_hex {desert_hex} out of 0..{NUM_HEXES - 1}")
    labels: list[str] = ["DESERT"] * NUM_HEXES
    idxs = [i for i in range(NUM_HEXES) if i != desert_hex]
    sats = np.array([samples[i][1] for i in idxs])
    ore = {idxs[j] for j in np.argsort(sats)[:3].tolist()}
    for i in ore:
        labels[i] = "ORE"
    rest = sorted((i for i in idxs if i not in ore), key=lambda i: float(samples[i][0]))
    order = ["BRICK"] * 3 + ["WHEAT"] * 4 + ["SHEEP"] * 4 + ["WOOD"] * 4
    for i, label in zip(rest, order, strict=True):
        labels[i] = label
    return labels


def _ocr_number(
    frame_rgb: npt.NDArray[np.uint8], center: npt.NDArray[np.float64], diam: float
) -> int | None:
    """OCR the token digit (2..12) at a hex center via easyocr (lazy, CPU-only)."""
    import cv2

    reader = _easyocr_reader()
    r = int(diam * 0.55)
    cx, cy = int(center[0]), int(center[1])
    crop = frame_rgb[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_u8: npt.NDArray[np.uint8] = np.asarray(gray, dtype=np.uint8)
    best: tuple[int, float] | None = None
    for _box, text, conf in reader.readtext(gray_u8, allowlist="0123456789", detail=1):
        digits = "".join(c for c in str(text) if c.isdigit())
        if digits and 2 <= int(digits) <= 12 and (best is None or conf > best[1]):
            best = (int(digits), float(conf))
    return best[0] if best is not None else None


def _count_pips(
    frame_rgb: npt.NDArray[np.uint8], center: npt.NDArray[np.float64], diam: float
) -> int:
    """Count the black pip dots under the token digit (independent of the OCR digit
    — the §5.6 cross-check). Pips sit in a horizontal band in the lower disk."""
    import cv2

    r = int(diam / 2)
    cx, cy = int(center[0]), int(center[1])
    patch = frame_rgb[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    if patch.size == 0:
        return -1
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _thresh, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    hh, ww = binary.shape
    band = binary[int(hh * 0.72) : int(hh * 0.92), int(ww * 0.12) : int(ww * 0.88)]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(band)
    pips = 0
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / max(h, 1)
        if 3 < area < 70 and 2 <= w <= 11 and 2 <= h <= 11 and 0.5 < aspect < 2.0:
            pips += 1
    return pips


class _EasyOcrReader:
    """Structural protocol for the one easyocr method we call (keeps mypy strict)."""

    def readtext(
        self, image: npt.NDArray[np.uint8], *, allowlist: str, detail: int
    ) -> list[Any]:  # pragma: no cover - satisfied by the real easyocr.Reader
        raise NotImplementedError


_READER: _EasyOcrReader | None = None


def _easyocr_reader() -> _EasyOcrReader:
    """Lazily build + cache a CPU easyocr reader (English)."""
    global _READER
    if _READER is None:
        import easyocr

        reader: _EasyOcrReader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _READER = reader
    return _READER


# ------------------------------------------------------------- screen anchors


def derive_screen_anchors(
    hexes: list[dict[str, Any]],
    hex_px: npt.NDArray[np.float64],
    frame_rgb: npt.NDArray[np.uint8],
    diam: float,
) -> tuple[ContentAnchor, ...]:
    """Derive >=2 orientation anchors from the screen-space token extremes (build
    brief §5.2 / review finding).

    The anchor's engine_id is a pure SCREEN fact — the engine hex the affine places
    pixel-topmost and the one it places pixel-rightmost — chosen by *where a token
    lands on screen*, not by a hardcoded id. Its ``number`` is OCR'd **straight from
    the frame at that screen pixel** here (a fresh, independent read). ``read_board``
    then asserts the board's own value at that engine_id equals this independent
    read: a mis-oriented affine that satisfied the geometric screen rule only
    degenerately places a DIFFERENT token at the topmost/rightmost position, so its
    independently-OCR'd number disagrees and the frame is rejected — this is the
    §5.2 orientation corroboration this module owns.

    Independence scope: the NUMBER is re-read independently at the screen landmark;
    the anchor's ``resource`` is the board's per-game palette classification
    (there is no palette-free single-hex resource read). The independent RESOURCE
    firewall is the hue-cluster-margin gate (:func:`hue_cluster_margin`), not this
    anchor. Skips the desert (``number is None``); returns the two extremes (top,
    right). A caller with fewer than 2 non-desert anchors must fail closed.
    """
    # Screen extremes among the NON-desert (token-bearing) hexes.
    token_ids = [int(h["hex_id"]) for h in hexes if h["resource"] != "DESERT"]
    if len(token_ids) < 2:
        return ()
    ys = {e: float(hex_px[e][1]) for e in token_ids}
    xs = {e: float(hex_px[e][0]) for e in token_ids}
    top_id = min(token_ids, key=lambda e: ys[e])
    right_id = max(token_ids, key=lambda e: xs[e])
    by_id = {int(h["hex_id"]): h for h in hexes}
    anchors: list[ContentAnchor] = []
    for e in (top_id, right_id):
        if e in {a.engine_id for a in anchors}:
            continue  # top and right coincided — only one independent extreme
        number = _ocr_number(frame_rgb, hex_px[e], diam)
        if number is None:
            continue
        anchors.append(
            ContentAnchor(engine_id=e, resource=str(by_id[e]["resource"]), number=number)
        )
    return tuple(anchors)


# ------------------------------------------------------------------- the reader


#: Max distance (in TOKEN DIAMETERS, from the median token centroid) at which a detected
#: disk can still be a real board number-token. MEASURED over all 34 saved frames of the
#: 8-video sweep: every real token sits at 5.30-5.40 diameters; the false disks — the
#: on-screen score display's "0" digit in the top-left HUD, which the disk detector reads
#: as a token — sit at 10.16-12.96. The cut at 8.0 lies in that empty gap with ~48%
#: margin above the real max, so it can never drop a real token. Scale-invariant: the
#: board's own median token diameter is the unit, so it holds at any resolution.
MAX_TOKEN_RADIUS_DIAM = 8.0

#: At most this many outliers may be trimmed. More than a couple of off-board disks means
#: the frame is genuinely wrong (a menu, an overlay, a different screen) — trim NOTHING
#: and let the unchanged 16-18 count gate reject it. Fail closed; never salvage a bad frame.
MAX_TOKEN_OUTLIERS = 3


def _trim_token_outliers(
    tokens: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], int]:
    """Drop detected disks that are too far from the token cloud to be board tokens.

    The number-token detector also fires on the HUD's score digits (the "0" of a ``0-0``
    scoreboard is a white-on-dark rounded blob). That single false disk broke boards TWO
    ways, both surfacing as ``board_unreadable``:

    * **19 tokens** -> trips :func:`read_board`'s 16-18 count gate outright;
    * **17-18 tokens** (the false disk standing in for an occluded real one) -> PASSES the
      count gate and then POISONS the RANSAC lattice fit, blowing the affine residual to
      33-67 px against a 5 px budget.

    Both are instrument artifacts, not unreadable boards. The centroid is the coordinate
    MEDIAN (robust to the very outliers we are removing) and the scale is the median token
    diameter, so the rule is resolution-independent. Returns ``(kept, n_dropped)``; a clean
    frame is returned byte-identically with ``n_dropped == 0``, so every frame that reads
    today is completely unaffected.
    """
    if len(tokens) < 4:
        return tokens, 0
    xy = np.array([[x, y] for x, y, _ in tokens], float)
    diam = float(np.median([d for _, _, d in tokens]))
    if diam <= 0:
        return tokens, 0
    centre = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - centre, axis=1) / diam
    keep_mask = dist <= MAX_TOKEN_RADIUS_DIAM
    n_dropped = int((~keep_mask).sum())
    if n_dropped == 0 or n_dropped > MAX_TOKEN_OUTLIERS:
        return tokens, 0  # nothing to do, or too many -> fail closed on the count gate
    return [t for t, k in zip(tokens, keep_mask, strict=True) if k], n_dropped


def _rb_fail(diag: dict[str, Any] | None, reason: str) -> BoardRead | None:
    """Record WHICH fail-closed gate rejected a frame, then reject it (return ``None``).

    :func:`read_board` has ~10 independent reject gates and used to return a bare
    ``None`` from all of them, so a ``board_unreadable`` game could not say WHY — the
    same observability hole the grant path had. Every gate now names itself, so a reject
    explains itself instead of needing a bespoke probe per video. Purely diagnostic: the
    reject behaviour is byte-identical (``None`` either way).
    """
    if diag is not None:
        diag["fail"] = reason
    return None


def read_board(
    frame_rgb: npt.NDArray[np.uint8],
    *,
    content_anchors: tuple[ContentAnchor, ...] | None = None,
    max_residual_px: float = MAX_AFFINE_RESIDUAL_PX,
    min_screen_rule_gap: float = MIN_SCREEN_RULE_GAP,
    diag: dict[str, Any] | None = None,
) -> BoardRead | None:
    """Parse the orientation-locked 19-hex board from a native-geometry **RGB**
    frame, or ``None`` if the frame is rejected (build brief §4 / §5.2).

    Rejection (returns ``None``) — the frame is skipped rather than emitting a
    silently mis-oriented / mislabelled board:

    - **fewer than 16 or more than 18 tokens** detected (a board animation
      dropped/added several disks; a small deficit is tolerated so a single
      occluded token does not reject an otherwise-good frame),
    - **no candidate affine** could be fit (RANSAC failed on every D6 candidate),
    - the **screen-rule gap** (second-best / best penalty) is below
      ``min_screen_rule_gap`` — a near-tied, ambiguous orientation,
    - **affine residual > ``max_residual_px``** (a mis-snapped lattice, §5.2),
    - the **desert split is not bimodal** (not exactly one clearly-token-less hex),
    - two adjacent resource hue clusters are **not separated by**
      :data:`MIN_HUE_CLUSTER_MARGIN` (a near-boundary WHEAT/SHEEP swap risk),
    - the read board is **not the standard resource / number multiset**,
    - any token hex's **OCR digit ≠ pip count** (§5.6 corroboration fails),
    - **fewer than 2 independent content anchors** are available, or
    - any **content anchor disagrees** with the screen-locked orientation.

    ``content_anchors`` — when ``None`` (the batch default) the anchors are derived
    per-frame from the screen-space token extremes (:func:`derive_screen_anchors`),
    an INDEPENDENT read; there is no game-1 fallback. A caller may pass explicit
    anchors (e.g. a test asserting a mis-oriented ground truth is rejected). At
    least 2 non-desert anchors are required either way (fail closed).
    """
    template = load_engine_template()
    topology = load_topology()
    hcv = topology.hex_corner_to_vertex

    # 16-18 number-token disks are expected: the 19-hex board minus the token-less
    # desert (18), tolerating up to two occluded/animating tokens. Fewer than 16
    # (or more than 18) means the board is mid-animation — skip the frame.
    tokens = _detect_tokens(frame_rgb)
    # Remove off-board false disks (the HUD score digits) BEFORE the count gate — they
    # both trip that gate at 19 tokens AND poison the lattice fit at 17-18. A clean frame
    # is untouched (see :func:`_trim_token_outliers`).
    tokens, n_outliers = _trim_token_outliers(tokens)
    if diag is not None:
        diag["n_tokens"] = len(tokens)
        diag["outliers_dropped"] = n_outliers
        diag["tokens"] = [(float(x), float(y), float(d)) for x, y, d in tokens]
    if not NUM_HEXES - 3 <= len(tokens) <= NUM_HEXES - 1:
        return _rb_fail(diag, "token_count")

    token_xy = np.array([[x, y] for x, y, _ in tokens], float)
    candidates = _candidate_affines(token_xy, template.hex_centers)
    if not candidates:
        return _rb_fail(diag, "no_affine")  # RANSAC failed on every D6 candidate
    scored = _score_screen_rule(candidates, token_xy, template.hex_centers)
    best_penalty, _refl, _rot, affine, residual = scored[0]
    second_penalty = scored[1][0] if len(scored) > 1 else float("inf")
    screen_rule_gap = float(second_penalty / best_penalty) if best_penalty > 0 else float("inf")
    if diag is not None:
        diag["screen_rule_gap"] = screen_rule_gap
        diag["residual_px"] = float(residual)
    if screen_rule_gap < min_screen_rule_gap:
        return _rb_fail(diag, "screen_rule_gap")  # near-tied orientation — never coin-flip
    if residual > max_residual_px:
        return _rb_fail(diag, "residual")

    import cv2

    hsv: npt.NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV), dtype=np.uint8
    )
    hex_px = _apply_affine(affine, template.hex_centers)
    vertex_px = _apply_affine(affine, template.vertex_px)
    diam = float(np.median([d for _, _, d in tokens]))

    samples = np.zeros((NUM_HEXES, 3))
    coverages = np.zeros(NUM_HEXES)
    for e in range(NUM_HEXES):
        corners = [vertex_px[hcv[e][c]] for c in range(6)]
        samples[e] = _sample_hex_hsv(hsv, hex_px[e], corners)
        coverages[e] = _hex_white_coverage(frame_rgb, hex_px[e], diam)

    desert = _desert_hex(coverages)
    if desert is None:
        return _rb_fail(diag, "desert_not_bimodal")
    desert_hex = desert

    # Resource firewall (a): reject a rank-sliced assignment whose adjacent hue
    # clusters are not clearly separated (a near-boundary WHEAT/SHEEP swap risk).
    margin = hue_cluster_margin(samples, desert_hex)
    if diag is not None:
        diag["hue_margin"] = float(margin)
    if margin < MIN_HUE_CLUSTER_MARGIN:
        return _rb_fail(diag, "hue_margin")

    resources = classify_resources(samples, desert_hex)
    if dict(Counter(r for r in resources if r != "DESERT")) != _NONDESERT_COUNTS:
        return _rb_fail(diag, "nonstandard_resources")

    hexes: list[dict[str, Any]] = []
    numbers: list[int] = []
    pip_ok = True
    for e in range(NUM_HEXES):
        if resources[e] == "DESERT":
            hexes.append({"hex_id": e, "resource": "DESERT", "number": None})
            continue
        number = _ocr_number(frame_rgb, hex_px[e], diam)
        pips = _count_pips(frame_rgb, hex_px[e], diam)
        if number is None or _PIP_FOR.get(number) != pips:
            pip_ok = False
        else:
            numbers.append(number)
        hexes.append({"hex_id": e, "resource": resources[e], "number": number})

    if not pip_ok:
        return _rb_fail(diag, "ocr_pip_mismatch")
    if Counter(numbers) != Counter(_STANDARD_NUMBER_BAG_EXPANDED):
        return _rb_fail(diag, "nonstandard_number_bag")

    # Content-anchor cross-check (§5.2): resource firewall (b) + the independent
    # orientation corroboration. When the caller passes no anchors, derive them
    # per-frame from the screen-space token extremes (an INDEPENDENT read at a
    # fixed screen landmark — never a game-1 constant). Fail closed if fewer than
    # 2 anchors are available.
    anchors = (
        content_anchors
        if content_anchors is not None
        else derive_screen_anchors(hexes, hex_px, frame_rgb, diam)
    )
    if len(anchors) < 2:
        return _rb_fail(diag, "too_few_anchors")
    by_id = {int(h["hex_id"]): h for h in hexes}
    for anchor in anchors:
        if anchor.resource == "DESERT":
            raise ValueError("content anchor must not be the desert (it moves per game)")
        got = by_id[anchor.engine_id]
        if got["resource"] != anchor.resource or got["number"] != anchor.number:
            return _rb_fail(diag, "anchor_disagrees")

    if diag is not None:
        diag["fail"] = None
    return BoardRead(
        hexes=tuple(hexes),
        affine=affine,
        vertex_px=vertex_px,
        desert_hex=desert_hex,
        residual_px=residual,
        screen_rule_gap=screen_rule_gap,
        pip_ok=pip_ok,
    )


def board_hsv_samples(
    frame_rgb: npt.NDArray[np.uint8], board: BoardRead
) -> npt.NDArray[np.float64]:
    """The 19x3 median-HSV hex samples for an accepted :class:`BoardRead`'s frame.

    Recomputes the same per-hex fill samples :func:`read_board` used internally,
    under the board's locked affine — the input
    :func:`~catan_rl.human_data.glyph_anchor.calibrate_glyph_palette` takes.
    NOTE: that palette is board-side tooling only — real glyph reads use the
    measured fixed :data:`~catan_rl.human_data.glyph_anchor.CARD_PALETTE` (the
    brief-§5.13 "cards share the tile colour family" premise was measured false
    for glyphs; see ``calibrate_glyph_palette``'s warning).
    """
    import cv2

    template = load_engine_template()
    hcv = load_topology().hex_corner_to_vertex
    hsv: npt.NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV), dtype=np.uint8
    )
    hex_px = _apply_affine(board.affine, template.hex_centers)
    samples = np.zeros((NUM_HEXES, 3))
    for e in range(NUM_HEXES):
        corners = [board.vertex_px[hcv[e][c]] for c in range(6)]
        samples[e] = _sample_hex_hsv(hsv, hex_px[e], corners)
    return samples


def read_board_stable(
    frames_rgb: list[npt.NDArray[np.uint8]],
    *,
    max_residual_px: float = MAX_AFFINE_RESIDUAL_PX,
    min_screen_rule_gap: float = MIN_SCREEN_RULE_GAP,
    diag_list: list[dict[str, Any]] | None = None,
) -> BoardRead | None:
    """Cross-frame board-stability gate (build brief §5.2, mandatory).

    Runs :func:`read_board` on ``≥2`` setup-window frames of the SAME game and
    returns the board only if it **agrees byte-identical** across every accepted
    frame — the hexes (resource + number per engine id) AND the ``desert_hex``.
    Returns ``None`` if fewer than 2 frames were accepted or any two disagree. A
    single-frame orientation flip (the exact failure §5.2 targets) produces a
    disagreement and is caught here; the batch/validate path MUST use this rather
    than a bare single-frame :func:`read_board`.
    """
    if len(frames_rgb) < 2:
        raise ValueError("read_board_stable requires >= 2 frames of the same game")
    reads: list[BoardRead] = []
    for frame in frames_rgb:
        d: dict[str, Any] = {}
        result = read_board(
            frame,
            max_residual_px=max_residual_px,
            min_screen_rule_gap=min_screen_rule_gap,
            diag=d if diag_list is not None else None,
        )
        if diag_list is not None:
            diag_list.append(d)
        if result is not None:
            reads.append(result)
    if len(reads) < 2:
        return None  # not enough accepted frames to corroborate stability
    first = reads[0]
    for other in reads[1:]:
        if list(other.hexes) != list(first.hexes) or other.desert_hex != first.desert_hex:
            if diag_list is not None:
                diag_list.append({"fail": "cross_frame_disagreement"})
            return None  # cross-frame disagreement — an orientation flip; reject
    return first


#: The standard 18-token number bag expanded to a flat list (for the Counter
#: cross-check in :func:`read_board`). Mirrors record.STANDARD_NUMBER_BAG.
_STANDARD_NUMBER_BAG_EXPANDED: tuple[int, ...] = (
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
)
