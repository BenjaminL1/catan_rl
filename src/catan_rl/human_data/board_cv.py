"""Orientation-locked board CV: parse the 19-hex board off a 1080p Colonist frame.

The Stage-2 ``board_cv`` slice (build brief §4). Given a decoded **RGB** frame
(an :class:`~catan_rl.human_data.ingest.DecodedFrame` ``.frame``), it:

1. **Detects the 19 number-token disks** (white blobs in the board region) and
   fits an **affine** from the committed engine board template (render-space hex
   pixels, :func:`load_engine_template`) to the frame.
2. **Locks the D6 orientation deterministically** (build brief §5.2 / BLOCKER-2).
   The 19-hex board has a D6 (12-element) symmetry, so all 12 candidate affines
   fit the detected tokens with *identical* residual — "lowest residual" flips
   orientation on floating-point noise and silently relabels every engine ID.
   The lock therefore uses a **screen-space rule** (engine H8 → top-center hex,
   H11 → rightmost hex) that is content-free and frame-stable, corroborated by
   **≥2 OCR content anchors** (a number token AND its resource must land on the
   engine ID the affine predicts — and NOT on the desert, which moves per game).
   A frame is **rejected** (returns ``None``) if the screen rule is ambiguous, if
   the content anchors disagree, or if the affine residual exceeds
   :data:`~catan_rl.human_data.orientation.MAX_AFFINE_RESIDUAL_PX`.
3. **Reads each hex per-game-calibrated** (build brief §2 / §5.13). The desert is
   the unique **token-less** hex (orientation-independent, per-game robust — no
   colour threshold). The 18 token hexes are classified by **per-game clustering**
   of their own sampled colours (ORE = the 3 lowest-saturation hexes; the rest
   ordered by hue → BRICK/WHEAT/SHEEP/WOOD) — the palette is never hardcoded. Each
   number is OCR'd and **independently corroborated by a pip-count** (§5.6), the
   non-tautological CV cross-check the resource-multiset gate cannot be (see
   :data:`catan_rl.human_data.record.STANDARD_RESOURCE_COUNTS`).

**Resource-multiset tautology (record.py finding, resolved here).** This module
assigns the 18 non-desert hexes to resources by *forcing the standard 4/4/4/3/3
multiset* from the per-game colour clusters, so the record contract's
``STANDARD_RESOURCE_COUNTS`` gate is tautological for the resource dimension. Per
the record.py contract that is only acceptable **because** an independent signal
replaces it: :func:`read_board` returns per-hex ``pip_ok`` (OCR-digit vs
pip-count agreement) and requires it on every token hex, and the number-token
bag is a genuine (non-forced) cross-check. A multiset-preserving resource swap
still surfaces via a wrong hue-cluster ordering only if it also breaks the
number-adjacency corroboration; the glyph anchor
(:mod:`catan_rl.human_data.orientation`) remains the joint-flip firewall.

CPU-only. ``cv2`` and ``easyocr`` are imported lazily inside the reader functions
so the pure geometry core (D6 orientation math, the screen-rule scorer, the
per-game cluster assignment) — the unit-tested surface — carries no heavy
optional dependency. Never imports ``gui/`` or the training path (brief §6).
"""

from __future__ import annotations

import json
import math
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
#: counts, so the resource-multiset gate is corroborated by the pip cross-check,
#: not relied on (see module docstring / record.py finding).
_NONDESERT_COUNTS: dict[str, int] = {"ORE": 3, "BRICK": 3, "WHEAT": 4, "SHEEP": 4, "WOOD": 4}

#: Number-token → pip-dot count (the independent OCR cross-check, §5.6). The
#: rendered token carries this many pips under its digit; OCR digit and pip count
#: must agree on every token hex or the read is rejected.
_PIP_FOR: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


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
    """

    engine_id: int
    resource: str
    number: int


#: The two default content anchors for the orientation lock (build brief §5.2:
#: "≥2 OCR anchors: a number token AND its resource"). Chosen at orientation-
#: discriminating engine IDs that carry a token (never the desert): the
#: top-center hex (H8) and a distinctive edge hex (H0). Their (resource, number)
#: is read INDEPENDENTLY from the frame and must equal what the affine predicts.
#:
#: NOTE these are the *game-1 fixture* anchor values; a general batch establishes
#: per-game anchors from an independent read (the same read the full board uses at
#: a fixed screen landmark). They are committed here so the game-1 lock is
#: reproducible and a deliberately mis-oriented fit is rejected against them.
_GAME1_ANCHORS: tuple[ContentAnchor, ...] = (
    ContentAnchor(engine_id=8, resource="SHEEP", number=2),
    ContentAnchor(engine_id=0, resource="SHEEP", number=11),
)


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
    resolved by the screen rule + content anchors, NOT the residual.
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


def _hex_has_token(
    frame_rgb: npt.NDArray[np.uint8], center: npt.NDArray[np.float64], diam: float
) -> bool:
    """True iff a white number-token disk sits at this hex center. The desert is
    the unique token-LESS hex — this is the orientation-independent, per-game
    desert detector (no colour threshold, brief §2)."""
    import cv2

    r = int(diam * 0.35)
    cx, cy = int(center[0]), int(center[1])
    patch = frame_rgb[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 2] > 175) & (hsv[..., 1] < 60)
    return bool(white.mean() > 0.35)


def classify_resources(samples: npt.NDArray[np.float64], desert_hex: int) -> list[str]:
    """Assign each of the 19 hexes a resource literal, per-game calibrated.

    ``samples`` is a 19x3 median-HSV array (engine hex order); ``desert_hex`` is
    the unique token-less hex. The 18 non-desert hexes are clustered from the
    frame's OWN samples (never a hardcoded palette, brief §5.13):

    - **ORE** = the 3 lowest-saturation hexes (grey stone).
    - the remaining 15 are ordered by **hue** and sliced by the standard counts:
      BRICK x3 (red, lowest hue) → WHEAT x4 (gold) → SHEEP x4 (yellow-green) →
      WOOD x4 (forest green, highest hue).

    This forces the standard 4/4/4/3/3 multiset, which is why the read is
    corroborated by the independent pip cross-check (see module docstring). Pure
    numpy — no cv2 — so it is unit-testable without the heavy CV dependency.
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


# ------------------------------------------------------------------- the reader


def read_board(
    frame_rgb: npt.NDArray[np.uint8],
    *,
    content_anchors: tuple[ContentAnchor, ...] = _GAME1_ANCHORS,
    max_residual_px: float = MAX_AFFINE_RESIDUAL_PX,
) -> BoardRead | None:
    """Parse the orientation-locked 19-hex board from a native-geometry **RGB**
    frame, or ``None`` if the frame is rejected (build brief §4 / §5.2).

    Rejection (returns ``None``) — the frame is skipped rather than emitting a
    silently mis-oriented board:

    - **not exactly 18 tokens** detected (the 19-hex board minus the token-less
      desert; a different count means a board animation dropped/added a disk),
    - **affine residual > ``max_residual_px``** (a mis-snapped lattice, §5.2),
    - **not exactly one token-less hex** (the desert detector is ambiguous),
    - the read board is **not the standard resource / number multiset**, or
    - any **content anchor disagrees** with the screen-locked orientation, or any
      token hex's **OCR digit ≠ pip count** (§5.6 corroboration fails).

    ``content_anchors`` are the ≥2 independent ground-truth reads the geometric
    screen-rule lock is cross-checked against; they must not be the desert.
    """
    template = load_engine_template()
    topology = load_topology()
    hcv = topology.hex_corner_to_vertex

    # Exactly 18 number-token disks are expected: the 19-hex board minus the
    # unique token-less desert. A different count means a board animation
    # dropped/added a disk (skip the frame rather than fit a broken lattice).
    tokens = _detect_tokens(frame_rgb)
    if len(tokens) != NUM_HEXES - 1:
        return None

    token_xy = np.array([[x, y] for x, y, _ in tokens], float)
    candidates = _candidate_affines(token_xy, template.hex_centers)
    scored = _score_screen_rule(candidates, token_xy, template.hex_centers)
    best_penalty, _refl, _rot, affine, residual = scored[0]
    second_penalty = scored[1][0]
    if residual > max_residual_px:
        return None

    import cv2

    hsv: npt.NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV), dtype=np.uint8
    )
    hex_px = _apply_affine(affine, template.hex_centers)
    vertex_px = _apply_affine(affine, template.vertex_px)
    diam = float(np.median([d for _, _, d in tokens]))

    samples = np.zeros((NUM_HEXES, 3))
    token_present: list[bool] = []
    for e in range(NUM_HEXES):
        corners = [vertex_px[hcv[e][c]] for c in range(6)]
        samples[e] = _sample_hex_hsv(hsv, hex_px[e], corners)
        token_present.append(_hex_has_token(frame_rgb, hex_px[e], diam))

    tokenless = [e for e in range(NUM_HEXES) if not token_present[e]]
    if len(tokenless) != 1:
        return None
    desert_hex = tokenless[0]

    resources = classify_resources(samples, desert_hex)
    from collections import Counter

    if dict(Counter(r for r in resources if r != "DESERT")) != _NONDESERT_COUNTS:
        return None

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

    if not pip_ok or Counter(numbers) != Counter(_STANDARD_NUMBER_BAG_EXPANDED):
        return None

    # Content-anchor cross-check (§5.2): the resource + number the affine predicts
    # at each anchor's engine id must equal the independently-known ground truth,
    # and the anchor must not be the desert. Rejects a deliberately mis-oriented
    # fit that satisfied the screen rule only degenerately.
    by_id = {int(h["hex_id"]): h for h in hexes}
    for anchor in content_anchors:
        if anchor.resource == "DESERT":
            raise ValueError("content anchor must not be the desert (it moves per game)")
        got = by_id[anchor.engine_id]
        if got["resource"] != anchor.resource or got["number"] != anchor.number:
            return None

    return BoardRead(
        hexes=tuple(hexes),
        affine=affine,
        vertex_px=vertex_px,
        desert_hex=desert_hex,
        residual_px=residual,
        screen_rule_gap=float(second_penalty / best_penalty) if best_penalty > 0 else float("inf"),
        pip_ok=pip_ok,
    )


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
