"""BLOCKER 2: deterministic, desert-free board orientation lock.

Root cause (proven by diag_residuals.py): the 19 hex centers form a regular
hexagonal lattice with a D6 (12-element) symmetry group. All 12 candidate
affines fit the detected token centroids with IDENTICAL geometric residual
(gap 0.00px), so 'pick lowest residual' is decided by FP noise and flips
between frames -> the hex->resource/number labels silently rotate/reflect.

Fix: geometry can't disambiguate, so disambiguate by CONTENT against
independently-OCR'd screen anchors.

  1. Detect number tokens; fit ONE base affine (engine hexes -> screen px) via
     least squares on a greedy match (orientation is arbitrary at this step).
  2. Enumerate the 12 D6 relabelings as permutations of engine hex ids that map
     the engine lattice onto itself (computed from geometry, exact).
  3. For each candidate labeling, project hexes and read (resource, number) at
     every engine id.
  4. SCREEN ANCHORS: read ground-truth content at >=2 fixed screen positions
     (independently: a NUMBER-token OCR and that hex's RESOURCE color). Pick the
     unique labeling whose (resource, number) at the engine id covering each
     anchor's screen pixel matches the anchor's own OCR/color read.
  5. Reject if not exactly one labeling is consistent (catches a bad fit /
     ambiguous frame instead of silently emitting a flipped board).

This module is self-contained and used by board_read.py.
"""

import json
import math

import cv2
import numpy as np

WORK = "/tmp/phantom/blockers"
M0 = "/tmp/phantom/m0"

TPL = json.load(open(f"{WORK}/engine_template.json"))
TOPO = json.load(open(f"{WORK}/topology.json"))
ENG_HEX = np.array([TPL["hex_centers"][str(i)] for i in range(19)], float)
ENG_VTX = np.array([TPL["vertex_px"][str(i)] for i in range(54)], float)
EDGE_V = TOPO["edgeVertices"]
HCV = TOPO["hexCornerToVertex"]
ENG_C = ENG_HEX.mean(0)

PIP_FOR = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
STD_NUMS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
STD_RES = {"WOOD": 4, "SHEEP": 4, "WHEAT": 4, "ORE": 3, "BRICK": 3, "DESERT": 1}


# ----------------------------------------------------------------- D6 perms
def d6_hex_perms():
    """Return the 12 permutations of engine hex ids induced by the lattice's D6
    symmetry. perm[i] = j means: after applying symmetry op, the hex that WAS at
    engine position i now sits where engine position j was. Computed exactly by
    nearest-neighbour on the centered lattice (snap distance ~0)."""
    perms = []
    centered = ENG_HEX - ENG_C
    for refl in (1, -1):
        for k in range(6):
            ang = math.radians(60 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            Rf = R @ np.array([[refl, 0], [0, 1]], float)
            moved = centered @ Rf.T
            perm = []
            ok = True
            for i in range(19):
                d = np.linalg.norm(centered - moved[i], axis=1)
                j = int(d.argmin())
                if d[j] > 1.0:
                    ok = False
                    break
                perm.append(j)
            if ok and len(set(perm)) == 19:
                perms.append((refl, 60 * k, perm))
    return perms


# ----------------------------------------------------------------- tokens
def detect_tokens(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 175), (180, 65, 255))
    m = np.zeros((H, W), np.uint8)
    m[int(0.04 * H) : int(0.97 * H), 0 : int(0.63 * W)] = 255
    white = cv2.bitwise_and(white, m)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(white)
    toks = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ar = w / max(h, 1)
        fill = a / (w * h)
        if a > 2400 and 0.78 < ar < 1.28 and 50 < w < 74 and fill > 0.65:
            toks.append((cent[i][0], cent[i][1], (w + h) / 2.0))
    return toks


# ----------------------------------------------------------------- base affine
def fit_base_affine(toks):
    """Fit ONE affine engine-hex -> screen px. Orientation chosen here is
    arbitrary (any of the 12); the D6 relabel is resolved later by content."""
    tok = np.array([[x, y] for x, y, _ in toks], float)
    fr_c = tok.mean(0)
    s0 = np.linalg.norm(tok - fr_c, axis=1).mean() / np.linalg.norm(ENG_HEX - ENG_C, axis=1).mean()
    best = None
    for refl in (1, -1):
        for k in range(12):
            ang = math.radians(30 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            Rf = R @ np.array([[refl, 0], [0, 1]], float)
            eng_t = (ENG_HEX - ENG_C) @ Rf.T * s0 + fr_c
            pairs = []
            used = set()
            for ti, p in enumerate(tok):
                d = np.linalg.norm(eng_t - p, axis=1)
                for hi in np.argsort(d):
                    if hi not in used:
                        used.add(hi)
                        pairs.append((ti, hi, d[hi]))
                        break
            resid = np.mean([d for _, _, d in pairs])
            if best is None or resid < best[0]:
                best = (resid, pairs)
    _, pairs = best
    src = np.array([ENG_HEX[hi] for _, hi, _ in pairs])
    dst = np.array([tok[ti] for ti, _, _ in pairs])
    A, _ = cv2.estimateAffine2D(
        src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=12
    )
    proj = (A[:, :2] @ src.T).T + A[:, 2]
    fin = np.linalg.norm(proj - dst, axis=1)
    return A, float(fin.mean()), float(fin.max())


def apply_affine(A, pts):
    return (A[:, :2] @ pts.T).T + A[:, 2]


# ----------------------------------------------------------------- content read
def sample_fill(img_hsv, cx, cy, corners):
    pts = []
    for vx, vy in corners:
        for frac in (0.40, 0.52, 0.64):
            x = int(cx + frac * (vx - cx))
            y = int(cy + frac * (vy - cy))
            if 0 <= y < img_hsv.shape[0] and 0 <= x < img_hsv.shape[1]:
                pts.append(img_hsv[y, x])
    pts = np.array(pts)
    keep = (pts[:, 2] > 55) & ~((pts[:, 2] > 170) & (pts[:, 1] < 55))
    if keep.sum() < 4:
        keep = np.ones(len(pts), bool)
    return np.median(pts[keep], 0)


def classify_one(h, s, v):
    if s < 60:
        return "DESERT" if (10 <= h <= 35 and v > 165) else "ORE"
    if h < 15:
        return "BRICK"
    if h < 29:
        return "WHEAT"
    if h < 50:
        return "SHEEP"
    return "WOOD"


def read_screen_hexes(img, A):
    """Read (resource, number) at every SCREEN position given by the base affine,
    indexed by the *base* engine id (pre-relabel). Returns dict id->(res,num,px)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hex_px = apply_affine(A, ENG_HEX)
    vtx_px = apply_affine(A, ENG_VTX)
    out = {}
    for i in range(19):
        corners = [vtx_px[HCV[i][c]] for c in range(6)]
        h, s, v = sample_fill(hsv, hex_px[i, 0], hex_px[i, 1], corners)
        res = classify_one(h, s, v)
        num = ocr_number(img, hex_px[i, 0], hex_px[i, 1]) if res != "DESERT" else None
        out[i] = (res, num, tuple(hex_px[i]))
    return out, hex_px, vtx_px


_reader = None


def reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def ocr_number(img, cx, cy, diam=60.0):
    r = int(diam * 0.55)
    crop = img[max(0, int(cy - r)) : int(cy + r), max(0, int(cx - r)) : int(cx + r)]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res = reader().readtext(g, allowlist="0123456789", detail=1)
    best = None
    for _, txt, conf in res:
        t = "".join(c for c in txt if c.isdigit())
        if t and 2 <= int(t) <= 12 and (best is None or conf > best[1]):
            best = (int(t), conf)
    return best[0] if best else None


# ----------------------------------------------------------------- the lock
def orientation_lock(img, anchors, verbose=False):
    """Returns (perm, A, screen_read, diag).

    perm maps engine-id -> base-id: the engine hex `e` is rendered at the screen
    position of base id `perm_inv`, i.e. board[e] = screen_read[ perm[e] ].

    anchors: list of dicts {name, px:(x,y), number:int|None, resource:str}.
      Each is an INDEPENDENT ground-truth read at a fixed screen pixel.
    """
    toks = detect_tokens(img)
    A, res_mean, res_max = fit_base_affine(toks)
    screen_read, hex_px, vtx_px = read_screen_hexes(img, A)
    perms = d6_hex_perms()

    # Which base-id covers each anchor screen pixel?
    anchor_base = []
    for a in anchors:
        d = np.linalg.norm(hex_px - np.array(a["px"]), axis=1)
        anchor_base.append(int(d.argmin()))

    consistent = []
    diag = {
        "residual_mean": res_mean,
        "residual_max": res_max,
        "n_tokens": len(toks),
        "candidates": [],
    }
    for refl, deg, perm in perms:
        # perm[e] = base id where engine id e is rendered.
        ok = True
        details = []
        for a, base_id in zip(anchors, anchor_base, strict=False):
            # engine id e that maps to this base id:
            e = perm.index(base_id)
            res_b, num_b, _ = screen_read[base_id]
            match_res = (a["resource"] is None) or (res_b == a["resource"])
            match_num = (a["number"] is None) or (num_b == a["number"])
            details.append((a["name"], e, base_id, res_b, num_b, match_res and match_num))
            if not (match_res and match_num):
                ok = False
        diag["candidates"].append(
            {"refl": refl, "rot": deg, "consistent": ok, "anchor_details": details}
        )
        if ok:
            consistent.append((refl, deg, perm))

    if len(consistent) != 1:
        return None, A, screen_read, diag, consistent
    refl, deg, perm = consistent[0]
    diag["locked"] = {"refl": refl, "rot": deg}
    return perm, A, screen_read, diag, consistent


def build_board(perm, screen_read):
    """engine board: board[e] = screen_read[perm[e]]."""
    hexes = []
    for e in range(19):
        base = perm[e]
        res, num, _ = screen_read[base]
        hexes.append({"hex_id": e, "resource": res, "number": num})
    return hexes
