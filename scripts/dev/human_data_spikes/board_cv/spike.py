"""Board-CV spike — full go/no-go probe.

Pipeline:
 1. Detect number-token disks (HSV white blobs, round, board region).
 2. Fit affine engine-pixel -> frame-pixel by matching detected token
    centroids to the engine's 19 hex centers (brute-force over discrete
    rotations/reflections since Colonist render is flat-top vs engine pointy-top).
 3. Project all 19 hex centers, 54 vertices, 72 edges into the frame.
 4. Read each hex: median resource color (calibrated from frame) + digit OCR
    + pip-dot count. Cross-check digit vs pips.
 5. Detect pieces by player-color segmentation; snap to nearest vertex/edge.
 6. Emit engine-indexed board JSON + annotated overlays.
"""

import json
import math
import sys

import cv2
import numpy as np

WORK = "/tmp/phantom/board_cv"
M0 = "/tmp/phantom/m0"

TPL = json.load(open(f"{WORK}/engine_template.json"))
TOPO = json.load(open(f"{M0}/topology.json"))
ENG_HEX = np.array([TPL["hex_centers"][str(i)] for i in range(19)], float)  # 19x2
ENG_VTX = np.array([TPL["vertex_px"][str(i)] for i in range(54)], float)  # 54x2
EDGE_V = TOPO["edgeVertices"]  # 72 [v1,v2]
PORT_SLOTS = TOPO["portSlots"]


# ------------------------------------------------------------------ tokens
def detect_tokens(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    m = np.zeros((H, W), np.uint8)
    m[int(0.05 * H) : int(0.96 * H), 0 : int(0.62 * W)] = 255
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
        if a > 2600 and 0.8 < ar < 1.25 and 54 < w < 72 and fill > 0.7:
            toks.append((cent[i][0], cent[i][1], (w + h) / 2.0))
    return toks  # list (x,y,diam)


# ------------------------------------------------------------------ lattice fit
def fit_affine(tok_xy):
    """Find affine A (2x3) mapping ENGINE hex pixels -> FRAME pixels, using the
    detected token centroids. Strategy: estimate the frame board center/scale,
    then for each of 12 candidate orientations (6 rot * 2 reflect of the engine
    template about its center) greedily match tokens to engine hexes by nearest
    neighbour and keep the orientation with the lowest residual. Refine with a
    full least-squares affine on the matched pairs.
    """
    tok = np.array([[x, y] for x, y, _ in tok_xy], float)
    fr_c = tok.mean(0)
    eng_c = ENG_HEX.mean(0)
    # scale guess: ratio of mean radius from center
    fr_rad = np.linalg.norm(tok - fr_c, axis=1).mean()
    eng_rad = np.linalg.norm(ENG_HEX - eng_c, axis=1).mean()
    s0 = fr_rad / eng_rad

    best = None
    for refl in (1, -1):
        for k in range(12):  # 30-degree steps to be safe
            ang = math.radians(30 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            Rf = R @ np.array([[refl, 0], [0, 1]], float)
            eng_t = (ENG_HEX - eng_c) @ Rf.T * s0 + fr_c
            # greedy NN match token->hex
            pairs = []
            used = set()
            for ti, p in enumerate(tok):
                d = np.linalg.norm(eng_t - p, axis=1)
                order = np.argsort(d)
                for hi in order:
                    if hi not in used:
                        used.add(hi)
                        pairs.append((ti, hi, d[hi]))
                        break
            resid = np.mean([d for _, _, d in pairs])
            if best is None or resid < best[0]:
                best = (resid, Rf, pairs)
    resid0, Rf, pairs = best
    # least-squares affine refine ENG_HEX[hi] -> tok[ti]
    src = np.array([ENG_HEX[hi] for _, hi, _ in pairs])
    dst = np.array([tok[ti] for ti, _, _ in pairs])
    A, inl = cv2.estimateAffine2D(
        src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=20
    )
    # final residual on all matched tokens
    proj = (A[:, :2] @ src.T).T + A[:, 2]
    fin = np.linalg.norm(proj - dst, axis=1)
    return A, resid0, fin, pairs


def apply_affine(A, pts):
    return (A[:, :2] @ pts.T).T + A[:, 2]


# ------------------------------------------------------------------ resource color
RES_REF_HSV = None  # calibrated per frame


def calibrate_colors(img, hex_px, hex_diam):
    """Sample median HSV at each projected hex center (small patch) and cluster
    into the 6 resource classes by hand-anchored hue ranges, but anchors derived
    from the frame's own samples (palette-robust)."""
    samples = []
    r = int(hex_diam * 0.7)
    for x, y in hex_px:
        x, y = int(x), int(y)
        patch = img[max(0, y - r) : y + r, max(0, x - r) : x + r]
        # avoid the white token: mask it out
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        notwhite = ~((hsv[..., 2] > 175) & (hsv[..., 1] < 60))
        if notwhite.sum() < 30:
            med = np.median(hsv.reshape(-1, 3), 0)
        else:
            med = np.median(hsv[notwhite], 0)
        samples.append(med)
    return np.array(samples)  # 19x3 HSV


def classify_resources(hsv_samples):
    """Rule-based classification using HSV. Hue (OpenCV 0-179)."""
    out = []
    for h, s, v in hsv_samples:
        if s < 45 and v > 110:  # gray-ish, low sat -> ORE (stone) or DESERT
            # desert is tan (higher hue ~20, moderate sat); ore is bluish-gray
            out.append("?gray")
        elif v < 60:
            out.append("?dark")
        else:
            out.append("hue:%d s:%d v:%d" % (h, s, v))
    return out


# ------------------------------------------------------------------ pip count
def count_pips(img, cx, cy, diam):
    """Count black pip dots under the number in the token. Pips are small dark
    blobs in the lower portion of the disk."""
    r = int(diam / 2)
    x0, y0 = int(cx - r), int(cy - r)
    patch = img[max(0, y0) : y0 + 2 * r, max(0, x0) : x0 + 2 * r].copy()
    if patch.size == 0:
        return -1
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # pips are tiny round dark spots; threshold dark, find small blobs in lower band
    _, th = cv2.threshold(g, 90, 255, cv2.THRESH_BINARY_INV)
    hh, ww = th.shape
    band = th[int(hh * 0.62) :, :]  # pips sit below the digit
    n, lab, stats, cent = cv2.connectedComponentsWithStats(band)
    cnt = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 4 < a < 80 and 2 <= w <= 12 and 2 <= h <= 12:
            cnt += 1
    return cnt


PIP_FOR = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def main(t):
    img = cv2.imread(f"{M0}/f1080_{t}.png")
    toks = detect_tokens(img)
    print(f"[t={t}] tokens detected: {len(toks)}")
    A, resid0, fin, pairs = fit_affine(toks)
    print(
        f"  pre-fit residual {resid0:.1f}px ; post-affine residual mean {fin.mean():.1f}px max {fin.max():.1f}px"
    )

    hex_px = apply_affine(A, ENG_HEX)
    vtx_px = apply_affine(A, ENG_VTX)
    hex_diam = np.median([d for _, _, d in toks])

    # save lattice overlay
    ov = img.copy()
    for x, y, d in toks:
        cv2.circle(ov, (int(x), int(y)), int(d / 2), (0, 0, 255), 2)
    for hi, (x, y) in enumerate(hex_px):
        cv2.circle(ov, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.putText(
            ov,
            f"H{hi}",
            (int(x) - 14, int(y) - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )
    for vi, (x, y) in enumerate(vtx_px):
        cv2.circle(ov, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.putText(
            ov, str(vi), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 0), 1
        )
    for v1, v2 in EDGE_V:
        p1 = vtx_px[v1]
        p2 = vtx_px[v2]
        cv2.line(ov, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (180, 180, 0), 1)
    cv2.imwrite(f"{WORK}/lattice_{t}.png", ov)

    # color calibration + pip readout
    hsv_samp = calibrate_colors(img, hex_px, hex_diam)
    print("  per-hex HSV (engine id order):")
    for hi in range(19):
        h, s, v = hsv_samp[hi]
        cx, cy = hex_px[hi]
        pip = count_pips(img, cx, cy, hex_diam)
        print(f"   H{hi:2d} HSV=({h:3.0f},{s:3.0f},{v:3.0f}) pips={pip}")

    np.save(f"{WORK}/hsv_{t}.npy", hsv_samp)
    json.dump(
        {"A": A.tolist(), "hex_px": hex_px.tolist(), "vtx_px": vtx_px.tolist()},
        open(f"{WORK}/fit_{t}.json", "w"),
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
