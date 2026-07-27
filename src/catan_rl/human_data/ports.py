"""Port-slot harvest: read the 9 fixed port slots off a retained Colonist frame.

Implements the ratified ``port-harvest`` spec (D1-D8). The module is split into
three layers that are deliberately independent, because the spec (D8) requires
**geometry** and **photometry** to be verified separately:

1. **Geometry** (:func:`predicted_slot_centres`, :func:`localise_board_slots`) — where
   the 9 port sprites sit in frame pixels. Each slot centre is the reflection of
   its water-hex centre through the slot-edge midpoint, ``2*mid - hex_centre``,
   plus the **constant** :data:`PORT_SLOT_OFFSET_PX` measured across all 9 slots
   on all 34 retained boards. Refinement is confined to a **tight**
   ``±SEARCH_WINDOW_FRAC * edge`` window (D1): a loose window lets slots 5 and 6
   lock onto the white HUD panels (dice widget / log panel) that overlap the
   water there.
2. **Photometry** (:func:`crop_features`, :func:`kmeans`, :class:`PortClassifier`)
   — CLUSTER-THEN-LABEL (D2). The 6 sprites recur on every board, so the
   class->name binding is a 6-item problem solved once by hand-labelling the
   cluster centroids. This module NEVER self-generates that binding; it emits
   centroids and consumes a labels file.
3. **Decode** (:func:`decode_slots`) — dual decode (D3): unconstrained per-slot
   argmax vs a Hungarian assignment under the fixed composition
   (:data:`PORT_COMPOSITION`). Agreement is required; disagreement is a typed
   whole-board rejection (D4).

**Fail-closed (D4).** :func:`build_port_assignment` is the ONLY constructor of a
9-slot port map and it requires 9 independently read slot names. Deducing the
9th slot from the other 8 is banned even though the fixed composition always
permits it -- it would let a classifier that never reads a slot report success.

**Known blind spot (D3).** A *consistent global transposition* of two classes
(every ore read as brick and every brick as ore) yields a legal multiset, so both
decodes agree and are wrong on every board. No self-check detects it; the
hand-labelled centroids are the only defence.

CPU-only. ``cv2`` is imported lazily inside the pixel functions so the pure
geometry / decode core -- the unit-tested, CI-safe surface -- carries no heavy
optional dependency. Never imports ``gui/`` or the training path.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.human_data.board_cv import (
    MIN_SCREEN_RULE_GAP,
    EngineTemplate,
    _apply_affine,
    _candidate_affines,
    _desert_hex,
    _detect_tokens,
    _hex_white_coverage,
    _score_screen_rule,
    _trim_token_outliers,
    load_engine_template,
)
from catan_rl.human_data.orientation import MAX_AFFINE_RESIDUAL_PX
from catan_rl.human_data.topology import NUM_HEXES, NUM_PORTS, Topology, load_topology

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    import numpy.typing as npt

# --------------------------------------------------------------------- D1 geometry

#: Engine-template edge length in render pixels (pointy-top, ``engine_template.json``).
ENGINE_EDGE_PX = 80.0

#: Constant frame-pixel offset added to the reflected hex centre (D1). Measured
#: across 9 slots x 34 boards; the slot-wise spread is +-0.5 px (x -10.6..-11.6,
#: y -5.7..-7.4), i.e. it is genuinely a single constant, not 9 of them.
PORT_SLOT_OFFSET_PX: tuple[float, float] = (-11.0, -7.0)

#: Half-width of the refinement search window as a fraction of the frame edge
#: length. **Tight by design** (D1): at 0.95 a loose window locked 38/306 slots
#: (12.4%) onto white HUD rectangles, concentrated in slot 5 (dice/action
#: widgets) and slot 6 (log panel). Do not widen without re-running the HUD pin.
SEARCH_WINDOW_FRAC = 0.35

#: Side of the scale-normalised slot crop, in pixels.
CROP_PX = 64

#: Side of the sampled slot patch as a fraction of the frame edge length. Framed
#: on the port SIGN (not the whole dock): resizing this box to :data:`CROP_PX`
#: makes crops comparable across the 1.1486-1.2321 px/engine-unit render-scale
#: span of the corpus.
CROP_SIDE_FRAC = 0.72

#: Displacement from the localised sail anchor to the sign centre, in units of the
#: frame edge length. The sail blob the localiser locks onto sits at the sign's
#: lower edge, so cropping on the anchor itself clips the icon off the top.
CROP_SIGN_OFFSET_FRAC: tuple[float, float] = (-0.09, -0.24)

#: HSV thresholds isolating the port sprite's white SAIL, which is the one
#: landmark every one of the 6 sprites shares. ``V >= 170`` and ``S <= 70``.
SAIL_MIN_VALUE = 170
SAIL_MAX_SATURATION = 70

#: Admissible sail-blob area as a fraction of the search window. The measured
#: sprite sail occupies 0.04-0.065 of the tight window on every slot; the HUD
#: components that intrude on slots 5/6 run to 0.15-0.44. The upper bound is
#: therefore a real discriminator, not a tuning knob.
SPRITE_AREA_FRAC_RANGE: tuple[float, float] = (0.02, 0.09)

#: Radius (px) within which two slots' candidate sail offsets count as agreeing
#: when voting for the board's rigid anchor offset.
CONSENSUS_SUPPORT_TOL_PX = 4.0

#: Maximum distance (px) a slot's chosen sail anchor may sit from the board
#: consensus. Beyond this the slot is UNLOCALISED and -- fail-closed, D4 -- the
#: whole board is rejected. The measured per-slot p95 deviation is <= 0.9 px, so
#: this floor is ~7x the observed noise: it rejects a lock-on, not jitter.
MAX_SLOT_DEVIATION_PX = 6.0


class PortRejection(StrEnum):
    """Typed whole-board rejection reasons (D4 -- there is no partial success)."""

    UNLOCALISED_SLOT = "unlocalised_slot"
    BLANK_SLOT = "blank_slot"
    DECODE_DISAGREEMENT = "decode_disagreement"
    LOW_MARGIN = "low_margin"
    NO_AFFINE = "no_affine"
    ORIENTATION_MISMATCH = "orientation_mismatch"


class PortHarvestError(Exception):
    """Raised when a caller tries to build a port map from fewer than 9 slots."""


@dataclass(frozen=True, slots=True)
class AffineFit:
    """Orientation-locked engine->frame affine, fitted WITHOUT OCR.

    ``read_board`` needs easyocr to read the number tokens; the port harvest only
    needs the lattice, so this reuses the same token detection + D6 screen-rule
    lock and stops there. ``desert_hex`` (the unique token-less hex) is carried so
    the caller can corroborate the orientation against the recorded
    ``provenance.board_desert_hex`` -- an independent check that the screen rule
    picked the same D6 element the full reader did.
    """

    affine: npt.NDArray[np.float64]
    scale: float
    residual_px: float
    screen_rule_gap: float
    desert_hex: int


def affine_scale(affine: npt.NDArray[np.float64]) -> float:
    """Isotropic render scale (frame px per engine unit) of a 2x3 affine.

    The measured affines are ``scale * I`` to within 0.2% (no rotation, skew or
    perspective), so a single scalar is a faithful summary.
    """
    return float(math.sqrt(abs(float(np.linalg.det(affine[:, :2])))))


def fit_board_affine(
    frame_rgb: npt.NDArray[np.uint8],
    *,
    max_residual_px: float = MAX_AFFINE_RESIDUAL_PX,
    min_screen_rule_gap: float = MIN_SCREEN_RULE_GAP,
) -> AffineFit | None:
    """Fit the orientation-locked affine for ``frame_rgb``, or ``None`` if rejected.

    Same gates as :func:`~catan_rl.human_data.board_cv.read_board` up to the
    lattice: token count in 16..18, a fittable candidate, screen-rule gap above
    ``min_screen_rule_gap``, residual below ``max_residual_px``, and a bimodal
    token-less desert split. Everything downstream of that (hue clusters, OCR,
    pip cross-check) is skipped -- it reads hex CONTENT, which the port harvest
    does not use.
    """
    template = load_engine_template()
    topology = load_topology()
    tokens, _ = _trim_token_outliers(_detect_tokens(frame_rgb))
    if not NUM_HEXES - 3 <= len(tokens) <= NUM_HEXES - 1:
        return None
    token_xy = np.array([[x, y] for x, y, _ in tokens], float)
    candidates = _candidate_affines(token_xy, template.hex_centers)
    if not candidates:
        return None
    scored = _score_screen_rule(candidates, token_xy, template.hex_centers)
    best_penalty, _refl, _rot, affine, residual = scored[0]
    second_penalty = scored[1][0] if len(scored) > 1 else float("inf")
    gap = float(second_penalty / best_penalty) if best_penalty > 0 else float("inf")
    if gap < min_screen_rule_gap or residual > max_residual_px:
        return None

    hex_px = _apply_affine(affine, template.hex_centers)
    diam = float(np.median([d for _, _, d in tokens]))
    coverages = np.array(
        [_hex_white_coverage(frame_rgb, hex_px[e], diam) for e in range(NUM_HEXES)], float
    )
    desert = _desert_hex(coverages)
    if desert is None:
        return None
    del topology  # loaded only to fail fast on a malformed fixture
    return AffineFit(
        affine=affine,
        scale=affine_scale(affine),
        residual_px=float(residual),
        screen_rule_gap=gap,
        desert_hex=int(desert),
    )


def predicted_slot_centres(
    affine: npt.NDArray[np.float64],
    template: EngineTemplate,
    topology: Topology,
    *,
    offset_px: tuple[float, float] = PORT_SLOT_OFFSET_PX,
) -> npt.NDArray[np.float64]:
    """D1 slot centres in FRAME pixels: ``2*mid - hex_centre + offset``, (9, 2).

    ``mid`` is the midpoint of the slot's two coast vertices and ``hex_centre``
    the centre of the land hex the slot hangs off, both projected through
    ``affine``. The reflection is affine-equivariant (the translation cancels:
    ``A(2m-h)+t == 2(Am+t)-(Ah+t)``), so the only frame-space term is the
    constant ``offset_px``.
    """
    centres = np.zeros((NUM_PORTS, 2), float)
    for i, slot in enumerate(topology.port_slots):
        hexc = _apply_affine(affine, template.hex_centers[int(slot["hex"]) : int(slot["hex"]) + 1])[
            0
        ]
        v0, v1 = (int(v) for v in slot["vertices"])
        mid = 0.5 * (
            _apply_affine(affine, template.vertex_px[v0 : v0 + 1])[0]
            + _apply_affine(affine, template.vertex_px[v1 : v1 + 1])[0]
        )
        centres[i] = 2.0 * mid - hexc + np.asarray(offset_px, float)
    return centres


def sail_candidates(
    frame_rgb: npt.NDArray[np.uint8],
    centre: tuple[float, float] | npt.NDArray[np.float64],
    *,
    edge_px: float,
    window_frac: float = SEARCH_WINDOW_FRAC,
    area_range: tuple[float, float] = SPRITE_AREA_FRAC_RANGE,
) -> tuple[tuple[float, float, float], ...]:
    """Candidate sail blobs in the ``±window_frac * edge_px`` window around ``centre``.

    Each candidate is ``(dx, dy, area_frac)``: the connected white component's
    centroid **relative to** ``centre``, plus its area as a fraction of the
    window. Returns ``()`` when the window falls off the frame or no component
    is admissible. Ordered by descending area, so the caller's choice is
    deterministic.
    """
    import cv2

    cx, cy = float(centre[0]), float(centre[1])
    half = round(window_frac * edge_px)
    h, w = frame_rgb.shape[:2]
    x0, y0 = round(cx) - half, round(cy) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
        return ()
    patch = frame_rgb[y0 : y0 + 2 * half + 1, x0 : x0 + 2 * half + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] >= SAIL_MIN_VALUE) & (hsv[:, :, 1] <= SAIL_MAX_SATURATION)).astype(
        np.uint8
    )
    n_lab, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lo, hi = area_range
    out: list[tuple[float, float, float]] = []
    for k in range(1, n_lab):
        frac = float(stats[k, cv2.CC_STAT_AREA]) / float(mask.size)
        if lo <= frac <= hi:
            out.append((x0 + float(centroids[k][0]) - cx, y0 + float(centroids[k][1]) - cy, frac))
    return tuple(sorted(out, key=lambda c: (-c[2], c[0], c[1])))


@dataclass(frozen=True, slots=True)
class SlotLocalisation:
    """One localised slot: the D1 prediction, its sail anchor, and the deviation."""

    slot: int
    predicted_px: tuple[float, float]
    anchor_px: tuple[float, float]
    deviation_px: tuple[float, float]
    area_frac: float


@dataclass(frozen=True, slots=True)
class BoardLocalisation:
    """All 9 slots of one board, or nothing (D4 -- there is no partial board)."""

    slots: tuple[SlotLocalisation, ...]
    anchor_offset_px: tuple[float, float]
    support: int
    edge_px: float


def localise_board_slots(
    frame_rgb: npt.NDArray[np.uint8],
    centres: npt.NDArray[np.float64],
    *,
    edge_px: float,
    window_frac: float = SEARCH_WINDOW_FRAC,
    max_deviation_px: float = MAX_SLOT_DEVIATION_PX,
    area_range: tuple[float, float] = SPRITE_AREA_FRAC_RANGE,
    naive: bool = False,
) -> BoardLocalisation | PortRejection:
    """Localise all 9 slots, fail-closed, using the board's RIGID sail offset.

    The sail sits at a *fixed* displacement from the D1 point (measured
    ``(+3.7, -30.5)`` px, per-slot p95 spread <= 0.9 px), so the 9 slots vote for
    one board-level offset: the candidate offset supported -- within
    :data:`CONSENSUS_SUPPORT_TOL_PX` -- by the most slots wins, ties broken by
    total distance then lexicographically. Each slot then takes the candidate
    nearest that consensus; a slot with no candidate inside
    ``max_deviation_px`` is UNLOCALISED and the **whole board is rejected**.

    ``naive=True`` disables the consensus and takes each slot's largest blob
    independently. That is the ablation the HUD pin (acceptance criterion 3)
    compares tight vs loose windows under; it is never the production path.
    """
    if centres.shape != (NUM_PORTS, 2):
        raise ValueError(f"expected ({NUM_PORTS}, 2) centres, got {centres.shape}")
    cands = [
        sail_candidates(
            frame_rgb,
            centres[i],
            edge_px=edge_px,
            window_frac=window_frac,
            area_range=area_range,
        )
        for i in range(NUM_PORTS)
    ]
    if any(not c for c in cands):
        return PortRejection.BLANK_SLOT

    if naive:
        chosen = [c[0] for c in cands]
        offset = (
            float(np.median([c[0] for c in chosen])),
            float(np.median([c[1] for c in chosen])),
        )
        support = NUM_PORTS
    else:
        best: tuple[int, float, tuple[float, float]] | None = None
        for slot_cands in cands:
            for cand in slot_cands:
                support_n, total = 0, 0.0
                for other in cands:
                    d = min(math.hypot(o[0] - cand[0], o[1] - cand[1]) for o in other)
                    if d <= CONSENSUS_SUPPORT_TOL_PX:
                        support_n += 1
                        total += d
                key = (support_n, -total, -cand[0], -cand[1])
                if best is None or key > (best[0], -best[1], -best[2][0], -best[2][1]):
                    best = (support_n, total, (cand[0], cand[1]))
        assert best is not None
        support, _total, offset = best
        chosen = []
        for slot_cands in cands:
            nearest = min(slot_cands, key=lambda c: math.hypot(c[0] - offset[0], c[1] - offset[1]))
            chosen.append(nearest)

    slots: list[SlotLocalisation] = []
    for i, cand in enumerate(chosen):
        dev = (cand[0] - offset[0], cand[1] - offset[1])
        if not naive and math.hypot(*dev) > max_deviation_px:
            return PortRejection.UNLOCALISED_SLOT
        slots.append(
            SlotLocalisation(
                slot=i,
                predicted_px=(float(centres[i, 0]), float(centres[i, 1])),
                anchor_px=(float(centres[i, 0] + cand[0]), float(centres[i, 1] + cand[1])),
                deviation_px=dev,
                area_frac=cand[2],
            )
        )
    return BoardLocalisation(
        slots=tuple(slots), anchor_offset_px=offset, support=support, edge_px=edge_px
    )


def sign_centre(
    anchor_px: tuple[float, float] | npt.NDArray[np.float64], edge_px: float
) -> tuple[float, float]:
    """Centre of the port SIGN, given the slot's sail anchor.

    The sail blob the localiser locks onto is a small, highly repeatable patch at
    the sign's lower edge, not the sign's middle; :data:`CROP_SIGN_OFFSET_FRAC`
    walks from one to the other. Keeping them distinct matters: the anchor is
    what geometry is measured against, the sign centre is what photometry is
    cropped around.
    """
    return (
        float(anchor_px[0]) + CROP_SIGN_OFFSET_FRAC[0] * edge_px,
        float(anchor_px[1]) + CROP_SIGN_OFFSET_FRAC[1] * edge_px,
    )


def slot_crop(
    frame_rgb: npt.NDArray[np.uint8],
    centre: tuple[float, float] | npt.NDArray[np.float64],
    *,
    edge_px: float,
    size: int = CROP_PX,
) -> npt.NDArray[np.uint8]:
    """Scale-normalised ``size x size`` RGB crop centred on ``centre``.

    The sampled box is ``CROP_SIDE_FRAC * edge_px`` on a side, so a board rendered
    at 1.2321 px/unit and one at 1.1486 produce crops of the same sprite at the
    same apparent size (the corpus render-scale span is 7.3%). Sampling is
    sub-pixel (``cv2.getRectSubPix``), which both removes a half-pixel quantisation
    that the 0.7 px localisation jitter would otherwise be swamped by and
    replicates the border instead of failing when a slot sits near a frame edge.
    """
    import cv2

    side = round(CROP_SIDE_FRAC * edge_px)
    patch = cv2.getRectSubPix(frame_rgb, (side, side), (float(centre[0]), float(centre[1])))
    resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    out: npt.NDArray[np.uint8] = np.asarray(resized, dtype=np.uint8)
    return out


@dataclass(frozen=True, slots=True)
class BoardCrops:
    """The 9 aligned slot crops of one board plus the geometry that produced them.

    ``desert_corroborated`` records whether the D6 orientation cross-check actually
    RAN (i.e. whether the caller supplied ``expected_desert_hex``). It is carried
    because a skipped check is not a passed check: a wrong D6 element relabels all
    9 slots at once, which is a global slot permutation that the composition check,
    both decode legs and the jitter envelope all wave through. Callers must report
    the boards where it was skipped rather than let the skip be invisible.
    """

    fit: AffineFit
    localisation: BoardLocalisation
    crops: tuple[npt.NDArray[np.uint8], ...]
    desert_corroborated: bool = False


def read_board_slots(
    frame_rgb: npt.NDArray[np.uint8],
    *,
    expected_desert_hex: int | None = None,
    window_frac: float = SEARCH_WINDOW_FRAC,
) -> BoardCrops | PortRejection:
    """Geometry leg end-to-end: frame -> affine -> 9 localised slots -> 9 crops.

    ``expected_desert_hex`` (the recorded ``provenance.board_desert_hex``) is an
    INDEPENDENT orientation corroboration: the port slots are labelled by engine
    id, so a D6 flip would relabel all 9 of them. When the recorded value is
    absent the check is SKIPPED and :attr:`BoardCrops.desert_corroborated` is
    ``False`` -- it is a corroboration, not the lock itself, but a board that never
    ran it has one guard fewer and the caller is expected to say so.

    Returns a :class:`PortRejection` rather than a partial board (D4).
    """
    fit = fit_board_affine(frame_rgb)
    if fit is None:
        return PortRejection.NO_AFFINE
    if expected_desert_hex is not None and fit.desert_hex != int(expected_desert_hex):
        return PortRejection.ORIENTATION_MISMATCH
    template, topology = load_engine_template(), load_topology()
    edge_px = fit.scale * ENGINE_EDGE_PX
    centres = predicted_slot_centres(fit.affine, template, topology)
    loc = localise_board_slots(frame_rgb, centres, edge_px=edge_px, window_frac=window_frac)
    if isinstance(loc, PortRejection):
        return loc
    crops = tuple(
        slot_crop(frame_rgb, sign_centre(s.anchor_px, edge_px), edge_px=edge_px) for s in loc.slots
    )
    return BoardCrops(
        fit=fit,
        localisation=loc,
        crops=crops,
        desert_corroborated=expected_desert_hex is not None,
    )


# ------------------------------------------------------------------ D2 photometry

#: The 6 port classes, in the engine's ``updatePorts`` naming. This is the NAME
#: space; the cluster->name binding is hand-supplied (D2), never inferred here.
PORT_TYPES: tuple[str, ...] = (
    "2:1 BRICK",
    "2:1 ORE",
    "2:1 SHEEP",
    "2:1 WHEAT",
    "2:1 WOOD",
    "3:1 PORT",
)

#: The fixed composition of every standard board: one 2:1 per resource, four 3:1.
PORT_COMPOSITION: dict[str, int] = {
    "2:1 BRICK": 1,
    "2:1 ORE": 1,
    "2:1 SHEEP": 1,
    "2:1 WHEAT": 1,
    "2:1 WOOD": 1,
    "3:1 PORT": 4,
}

#: Sub-box of the aligned crop holding the printed icon -- the resource glyph on a
#: 2:1 sign, the "?" on a 3:1 -- as ``(x0, x1, y0, y1)`` fractions. Deliberately
#: strictly INSIDE the white sign: a box that reaches the scalloped sign edge lets
#: dark ocean (and, on slots 5/6, HUD) into the ink map, which is what made a
#: naive whole-crop clustering split the 3:1 class in two.
ICON_BOX: tuple[float, float, float, float] = (0.26, 0.64, 0.12, 0.44)

#: Side of the square grid the icon ink map is pooled onto.
ICON_GRID = 14

#: Ink threshold on HSV *value*: the printed glyph is dark, the sign is not.
INK_MAX_VALUE = 150


def sign_ink(crop_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Binary ink map of one aligned crop: dark, NON-BLUE pixels.

    The blue exclusion matters even inside the sign box: ocean showing through a
    scalloped edge and the translucent blue HUD are both dark, and without the
    hue guard they enter the descriptor as if they were glyph strokes.
    """
    import cv2

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = (hue >= 95) & (hue <= 135) & (sat > 80)
    return ((val < INK_MAX_VALUE) & ~blue).astype(np.float32)


def crop_features(crop_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Descriptor for one aligned slot crop: the pooled icon ink map.

    Mean-subtracted and L2-normalised, so overall ink coverage (which varies with
    render scale and JPEG-ish blur across the 34 videos) drops out and only the
    glyph's SHAPE survives. The "2:1"/"3:1" digit row is deliberately NOT included
    -- the icon alone already separates generic from specific ports exactly
    (measured 128/160 at k=2), and the digits add a font-rendering nuisance
    dimension for no gain.
    """
    import cv2

    ink = sign_ink(crop_rgb)
    n = ink.shape[0]
    x0, x1, y0, y1 = ICON_BOX
    sub = ink[int(y0 * n) : int(y1 * n), int(x0 * n) : int(x1 * n)]
    grid = cv2.resize(sub, (ICON_GRID, ICON_GRID), interpolation=cv2.INTER_AREA)
    vec = grid.reshape(-1).astype(np.float64)
    vec = vec - vec.mean()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def kmeans(
    features: npt.NDArray[np.float64], k: int, *, seed: int = 0, iters: int = 100
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Deterministic k-means (seeded k-means++ init), returning ``(centroids, labels)``.

    Hand-rolled rather than ``sklearn`` (not a dependency) and seeded so the same
    corpus in the same order always yields the same clusters -- determinism is an
    acceptance criterion, and the emitted centroids are hand-labelled downstream,
    so they must not drift between runs.
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    if n < k:
        raise ValueError(f"cannot cluster {n} points into {k} clusters")
    centroids = np.empty((k, features.shape[1]), float)
    centroids[0] = features[rng.integers(n)]
    closest = ((features - centroids[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = float(closest.sum())
        probs = np.full(n, 1.0 / n) if total <= 0 else closest / total
        centroids[c] = features[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((features - centroids[c]) ** 2).sum(axis=1))
    labels = np.zeros(n, dtype=np.int64)
    for step in range(iters):
        dists = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1).astype(np.int64)
        if step > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = features[labels == c]
            if members.size:
                centroids[c] = members.mean(axis=0)
    return centroids, labels


def kmeans_best(
    features: npt.NDArray[np.float64], k: int, *, n_init: int = 16
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Lowest-inertia k-means over seeds ``0..n_init-1`` (deterministic n-init).

    Flat k=6 is seed-sensitive on this data (it lands anywhere between a clean
    split and 97/74/33/32/30/22), so the seed must never be a lucky constant.
    """
    best: tuple[float, npt.NDArray[np.float64], npt.NDArray[np.int64]] | None = None
    for seed in range(n_init):
        centroids, labels = kmeans(features, k, seed=seed)
        inertia = float(((features - centroids[labels]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, centroids, labels)
    assert best is not None
    return best[1], best[2]


#: Cluster index reserved for the generic 3:1 port in :func:`cluster_slot_crops`.
GENERIC_CLUSTER = 5


def cluster_slot_crops(
    features: npt.NDArray[np.float64], *, n_init: int = 16
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Two-level CLUSTER step (D2): generic-vs-specific, then the 5 resources.

    Flat k=6 does not separate these classes (measured: one class always splits
    and two merge). Two levels do, and for a structural reason -- the "?" glyph is
    far from all 5 resource glyphs, so the k=2 split is the dominant axis and
    swallows the k=6 objective. Level 1 (k=2) lands 128/160 exactly; level 2 (k=5)
    over the specific members lands 32/32/32/32/32 exactly.

    Returns ``(centroids, labels)`` with 6 centroids; index
    :data:`GENERIC_CLUSTER` is the larger-membership *generic* cluster. Both are
    still UNLABELLED -- binding them to :data:`PORT_TYPES` is the hand step (D2).

    **Only index** :data:`GENERIC_CLUSTER` **is stable.** Indices 0..4 are k-means
    output over ``features`` and permute when the board set changes, so they are
    never a durable name key: persist the centroids and bind through
    :func:`classifier_from_labels`.
    """
    _c2, l2 = kmeans_best(features, 2, n_init=n_init)
    counts = [int((l2 == c).sum()) for c in (0, 1)]
    # The generic 3:1 port is 4 of every 9 slots and each specific type is 1, so
    # the SPECIFIC side is the larger of the two (5/9 vs 4/9). Ties cannot occur.
    generic = 0 if counts[0] < counts[1] else 1
    specific = np.where(l2 != generic)[0]
    _c5, l5 = kmeans_best(features[specific], 5, n_init=n_init)
    labels = np.full(features.shape[0], GENERIC_CLUSTER, dtype=np.int64)
    labels[specific] = l5
    centroids = np.stack(
        [
            features[labels == c].mean(axis=0)
            if int((labels == c).sum())
            else np.zeros(features.shape[1])
            for c in range(len(PORT_TYPES))
        ]
    )
    return centroids, labels


@dataclass(frozen=True, slots=True)
class PortClassifier:
    """A FROZEN centroid set plus its hand-supplied cluster->name binding (D2).

    Decoding is nearest-frozen-centroid, never a re-clustering: one frame maps to
    one port map bit-for-bit, independently of which other boards are in the run
    (acceptance criterion 9). That property belongs to this object, not to the
    pipeline around it — a caller that re-derives centroids and binds names to the
    fresh cluster INDICES throws it away. Build one via
    :func:`classifier_from_labels`, which refuses a labels file whose recorded
    :func:`centroid_fingerprint` is not the fingerprint of ``centroids``.
    """

    centroids: npt.NDArray[np.float64]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) != self.centroids.shape[0]:
            raise ValueError("one name is required per centroid")
        unknown = set(self.names) - set(PORT_TYPES)
        if unknown:
            raise ValueError(f"unknown port type(s) in labels: {sorted(unknown)}")

    def scores(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """(N, C) similarity = negative Euclidean distance to each centroid."""
        d = np.sqrt(((features[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        return -d


#: Key under which a labels file carries the fingerprint of the centroids its
#: names were read off, and the key holding the index->name map itself.
LABELS_FINGERPRINT_KEY = "centroids_sha256"
LABELS_NAMES_KEY = "names"


def centroid_fingerprint(centroids: npt.NDArray[np.float64]) -> str:
    """Stable sha256 identity of a centroid set — what a hand-label is bound TO.

    Cluster indices are *not* stable: they come out of k-means over whatever
    feature matrix the run happened to build, so the same archetype can be
    cluster 1 in one run and cluster 3 in the next (adding or dropping a single
    board is enough). A name bound to a bare index therefore silently rebinds —
    which is precisely the consistent global transposition D3 says nothing can
    detect. The fix is that the binding is bound to *these* centroid vectors:
    the fingerprint is written next to them, copied into the labels file, and
    re-checked before a single slot is decoded (:func:`classifier_from_labels`).

    Values are rounded to 9 decimals before hashing so that the identity survives
    a float round-trip through ``.npz``, but not an actual re-clustering.
    """
    arr = np.ascontiguousarray(np.round(np.asarray(centroids, dtype=np.float64), 9))
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def classifier_from_labels(
    centroids: npt.NDArray[np.float64], labels: Mapping[str, Any]
) -> PortClassifier:
    """Bind FROZEN centroids to a hand-supplied labels file (D2), fail-closed.

    ``labels`` is the parsed ``centroid_labels.json``::

        {"centroids_sha256": "<fingerprint of the centroids that were looked at>",
         "names": {"0": "2:1 ORE", ..., "5": "3:1 PORT"}}

    Every cluster index ``"0"``..``"C-1"`` must be present as a **string** key,
    every value must be one of :data:`PORT_TYPES`, and the fingerprint must match
    ``centroids``. Each of those is a :class:`PortHarvestError` rather than a bare
    ``KeyError``, because a wrong or stale binding is the one failure the harvest
    cannot detect downstream.
    """
    n = int(np.asarray(centroids).shape[0])
    recorded = labels.get(LABELS_FINGERPRINT_KEY)
    actual = centroid_fingerprint(centroids)
    if not recorded:
        raise PortHarvestError(
            f"labels file has no {LABELS_FINGERPRINT_KEY!r}; it cannot be shown to name "
            f"these centroids (expected {actual})"
        )
    if str(recorded) != actual:
        raise PortHarvestError(
            "labels were bound to a DIFFERENT centroid set "
            f"({recorded} != {actual}); cluster indices are not stable across runs, so "
            "re-label the emitted centroid images instead of reusing this file"
        )
    names_obj = labels.get(LABELS_NAMES_KEY)
    if not isinstance(names_obj, Mapping):
        raise PortHarvestError(f"labels file has no {LABELS_NAMES_KEY!r} object")
    missing = [str(c) for c in range(n) if str(c) not in names_obj]
    if missing:
        raise PortHarvestError(f"labels file is missing cluster key(s) {missing}")
    unknown = sorted({str(v) for v in names_obj.values()} - set(PORT_TYPES))
    if unknown:
        raise PortHarvestError(f"unknown port type(s) in labels: {unknown}")
    return PortClassifier(
        centroids=np.asarray(centroids, dtype=np.float64),
        names=tuple(str(names_obj[str(c)]) for c in range(n)),
    )


# ---------------------------------------------------------------------- D3 decode


@dataclass(frozen=True, slots=True)
class PortDecode:
    """Result of the D3 dual decode over a (9, C) score matrix."""

    names: tuple[str, ...] | None
    argmax_names: tuple[str, ...]
    hungarian_names: tuple[str, ...]
    margins: tuple[float, ...]
    rejection: PortRejection | None


def _hungarian_names(
    scores: npt.NDArray[np.float64],
    names: tuple[str, ...],
    composition: dict[str, int],
) -> tuple[str, ...]:
    """Composition-constrained assignment: each class duplicated to its count."""
    from scipy.optimize import linear_sum_assignment

    columns: list[int] = []
    for c, name in enumerate(names):
        columns.extend([c] * composition.get(name, 0))
    if len(columns) != scores.shape[0]:
        raise ValueError(
            f"composition totals {len(columns)} ports, expected {scores.shape[0]} slots"
        )
    cost = -scores[:, columns]
    rows, cols = linear_sum_assignment(cost)
    out = [""] * scores.shape[0]
    for r, c in zip(rows, cols, strict=True):
        out[int(r)] = names[columns[int(c)]]
    return tuple(out)


def decode_slots(
    scores: npt.NDArray[np.float64],
    *,
    names: tuple[str, ...],
    composition: dict[str, int] | None = None,
    min_margin: float = 0.0,
) -> PortDecode:
    """D3 dual decode of a ``(9, C)`` similarity matrix (higher = better match).

    Leg (a) is the **unconstrained** per-slot argmax; leg (b) the **constrained**
    Hungarian assignment under the fixed composition. They must agree. A
    disagreement means the classifier's free reading is not a legal board, which
    the permutation-invariant multiset check alone cannot see -- so it is a typed
    rejection, not a silent snap-to-legal.

    ``min_margin`` additionally rejects a board where any slot's best-vs-second
    score gap is below the floor (an unconfident read is not evidence).

    ``names`` maps score COLUMN -> port type and is **required, keyword-only**. It
    previously defaulted to :data:`PORT_TYPES`, i.e. an identity binding on
    alphabetical order -- and that default silently shipped a globally transposed
    harvest, because every caller omitted it and so discarded
    ``PortClassifier.names`` (the D2 hand binding). A global transposition yields a
    LEGAL multiset, so both decode legs agree and the composition check passes on
    every board: exactly the D3 blind spot, with no self-check able to fire. There
    is deliberately no default now -- the binding must be stated at the call site.
    """
    if scores.shape[0] != NUM_PORTS:
        raise ValueError(f"expected {NUM_PORTS} slot rows, got {scores.shape[0]}")
    comp = PORT_COMPOSITION if composition is None else composition
    order = np.argsort(-scores, axis=1)
    argmax = tuple(names[int(order[i, 0])] for i in range(scores.shape[0]))
    margins = tuple(
        float(scores[i, order[i, 0]] - scores[i, order[i, 1]]) for i in range(scores.shape[0])
    )
    hung = _hungarian_names(scores, names, comp)
    if argmax != hung:
        return PortDecode(None, argmax, hung, margins, PortRejection.DECODE_DISAGREEMENT)
    if min_margin > 0.0 and min(margins) < min_margin:
        return PortDecode(None, argmax, hung, margins, PortRejection.LOW_MARGIN)
    return PortDecode(argmax, argmax, hung, margins, None)


def composition_ok(names: tuple[str, ...], composition: dict[str, int] | None = None) -> bool:
    """Permutation-invariant multiset check -- photometry only (D8).

    Deliberately weak. It cannot see a within-board PERMUTATION, and neither can
    :func:`decode_slots`: a permutation of a legal multiset is still legal, so both
    decode legs agree on it (verified in the unit tests). What rules a within-board
    permutation out is the *measured geometric jitter* (D8, p95 <= 0.9 px): each slot
    is cropped at its own predicted lattice point, so two slots cannot exchange
    contents without a displacement two orders of magnitude larger than the observed
    one. Do not weaken the geometry gates on the assumption that D3 covers this.
    It also cannot see a global transposition (nothing can -- see the module
    docstring). Its power is that a systematic photometric confusion breaks nearly
    every board at once, which is why the pass rate over N boards is a real
    photometry test.
    """
    comp = PORT_COMPOSITION if composition is None else composition
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return counts == comp


def build_port_assignment(
    names: tuple[str, ...] | list[str], topology: Topology | None = None
) -> dict[str, list[int]]:
    """The ONLY constructor of a 9-slot port map (D4).

    ``names`` must carry exactly 9 independently read slot names in slot order.
    There is intentionally no overload that fills a missing slot from the
    composition residual: a classifier that never read slot 8 must not be able to
    report a complete board.
    """
    if len(names) != NUM_PORTS:
        raise PortHarvestError(
            f"a port map requires {NUM_PORTS} independently read slots, got {len(names)}; "
            "deducing a slot from the composition is banned (D4)"
        )
    if any(n not in PORT_TYPES for n in names):
        raise PortHarvestError(f"unknown port type in {list(names)}")
    topo = load_topology() if topology is None else topology
    assignment: dict[str, list[int]] = {}
    for slot, name in zip(topo.port_slots, names, strict=True):
        assignment.setdefault(name, []).extend(int(v) for v in slot["vertices"])
    return assignment
