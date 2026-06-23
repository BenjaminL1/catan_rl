"""Board-CV spike v2 — refined resource classification + digit OCR + cross-checks.

Reuses the affine fit from spike.py (fit_<t>.json). Improves:
  * hex background color sampled from a RING (avoid token + center icon)
  * data-driven resource classification: cluster the 19 samples, then assign
    cluster->resource by the known standard multiset (palette-agnostic)
  * digit OCR with easyocr on each token crop
  * pip count cross-check
  * resource + number multiset cross-checks
Emits engine-indexed board_<t>.json + annotated overlay_<t>.png.
"""

import json
import sys

import cv2
import numpy as np

WORK = "/tmp/phantom/board_cv"
M0 = "/tmp/phantom/m0"

TOPO = json.load(open(f"{M0}/topology.json"))
EDGE_V = TOPO["edgeVertices"]
PORT_SLOTS = TOPO["portSlots"]

PIP_FOR = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
STD_NUMS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
STD_RES = {"WOOD": 4, "SHEEP": 4, "WHEAT": 4, "ORE": 3, "BRICK": 3, "DESERT": 1}

_READER = None


def reader():
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def sample_fill(img, cx, cy, vtx_xy):
    """Median HSV of the hex FILL sampled along center->corner rays at 0.40,
    0.52, 0.64 (clean colored area, away from token, center icon, and border).
    vtx_xy: list of 6 (x,y) corner pixel coords for this hex."""
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pts = []
    for vx, vy in vtx_xy:
        for frac in (0.40, 0.52, 0.64):
            x = int(cx + frac * (vx - cx))
            y = int(cy + frac * (vy - cy))
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pts.append(hsvimg[y, x])
    pts = np.array(pts)
    keep = (pts[:, 2] > 55) & ~((pts[:, 2] > 170) & (pts[:, 1] < 55))
    if keep.sum() < 4:
        keep = np.ones(len(pts), bool)
    return np.median(pts[keep], 0)


def classify(hsv19):
    """hsv19: 19x3. Palette-agnostic resource labels.

    Calibrated hue anchors (OpenCV 0-179), verified on the Colonist render:
      brick  hue<15  (bright red-orange)
      wheat  15-29   (gold)
      sheep  29-50   (yellow-green pasture)
      wood   >50     (forest green)
      desert low/mid sat + warm hue (tan)
      ore    low sat + cool/neutral (blue-gray)
    """
    H = hsv19[:, 0]
    S = hsv19[:, 1]
    V = hsv19[:, 2]
    labels = [None] * 19
    for i in range(19):
        h, s, v = H[i], S[i], V[i]
        if s < 60:  # unsaturated -> ore or desert
            labels[i] = "DESERT" if (10 <= h <= 35 and v > 165) else "ORE"
        elif h < 15:
            labels[i] = "BRICK"
        elif h < 29:
            labels[i] = "WHEAT"
        elif h < 50:
            labels[i] = "SHEEP"
        else:
            labels[i] = "WOOD"
    # desert can also be a moderately-saturated tan (hue~26, s~100): if no DESERT
    # was found and exactly one hex is tan-ish, reassign it.
    if "DESERT" not in labels:
        cand = [
            (i, abs(H[i] - 26) + (200 - V[i]) * 0.1)
            for i in range(19)
            if 18 <= H[i] <= 34 and S[i] < 130 and V[i] > 165
        ]
        if cand:
            labels[min(cand, key=lambda z: z[1])[0]] = "DESERT"
    return labels


def ocr_digit(img, cx, cy, diam):
    r = int(diam * 0.55)
    crop = img[max(0, int(cy - r)) : int(cy + r), max(0, int(cx - r)) : int(cx + r)]
    if crop.size == 0:
        return None, 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res = reader().readtext(g, allowlist="0123456789", detail=1)
    best = None
    for _, txt, conf in res:
        t = "".join(c for c in txt if c.isdigit())
        if t and 2 <= int(t) <= 12 and (best is None or conf > best[1]):
            best = (int(t), conf)
    return best if best else (None, 0.0)


def count_pips(img, cx, cy, diam):
    r = int(diam / 2)
    patch = img[max(0, int(cy - r)) : int(cy + r), max(0, int(cx - r)) : int(cx + r)]
    if patch.size == 0:
        return -1
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY_INV)
    hh, ww = th.shape
    # pips sit in the bottom ~22% of the disk in a single horizontal row
    band = th[int(hh * 0.72) : int(hh * 0.92), int(ww * 0.12) : int(ww * 0.88)]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(band)
    cnt = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ar = w / max(h, 1)
        if 3 < a < 70 and 2 <= w <= 11 and 2 <= h <= 11 and 0.5 < ar < 2.0:
            cnt += 1
    return cnt


def has_token(img, cx, cy, diam):
    """True if a white disk sits at this hex center (non-desert)."""
    r = int(diam * 0.35)
    patch = img[max(0, int(cy - r)) : int(cy + r), max(0, int(cx - r)) : int(cx + r)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 175) & (hsv[..., 1] < 60)
    return white.mean() > 0.35


def main(t):
    img = cv2.imread(f"{M0}/f1080_{t}.png")
    fit = json.load(open(f"{WORK}/fit_{t}.json"))
    hex_px = np.array(fit["hex_px"])
    vtx_px = np.array(fit["vtx_px"])
    diam = 60.0
    hcv = TOPO["hexCornerToVertex"]

    hsv19 = np.zeros((19, 3))
    for i in range(19):
        corners = [vtx_px[hcv[i][c]] for c in range(6)]
        hsv19[i] = sample_fill(img, hex_px[i, 0], hex_px[i, 1], corners)
    res = classify(hsv19)

    rows = []
    for i in range(19):
        tok = has_token(img, hex_px[i, 0], hex_px[i, 1], diam)
        if not tok:
            num, conf, pip = None, 0.0, 0
        else:
            num, conf = ocr_digit(img, hex_px[i, 0], hex_px[i, 1], diam)
            pip = count_pips(img, hex_px[i, 0], hex_px[i, 1], diam)
        rows.append((i, res[i], num, conf, pip, tok))

    print(f"[t={t}] hex readout (engine id order):")
    print("  id  res     num conf pip  pip_ok  token")
    for i, r, num, conf, pip, tok in rows:
        exp = PIP_FOR.get(num) if num else None
        ok = "—" if num is None else ("OK" if exp == pip else f"MISMATCH(exp{exp})")
        print(f"  H{i:2d} {r:7s} {num!s:>4} {conf:4.2f} {pip:3d}  {ok:14s} {tok}")

    # cross-checks
    rescount = {}
    for _, r, *_ in rows:
        rescount[r] = rescount.get(r, 0) + 1
    nums = sorted([num for _, _, num, _, _, _ in rows if num is not None])
    print("\n  RESOURCE multiset:", rescount, "(std", STD_RES, ")")
    print("  resource multiset matches std:", rescount == STD_RES)
    print("  NUMBER multiset:", nums)
    print("  number multiset matches std:", nums == STD_NUMS, f"(got {len(nums)}/18)")

    # ---- engine-indexed JSON ----
    hexes = [{"hex_id": i, "resource": r, "number": num} for i, r, num, _, _, _ in rows]
    board = {
        "frame_t": t,
        "hexes": hexes,
        "ports": [{"slot": p["slot"], "type": "UNKNOWN"} for p in PORT_SLOTS],
        "pieces": [],
    }
    json.dump(board, open(f"{WORK}/board_{t}.json", "w"), indent=2)

    # ---- annotated overlay ----
    ov = img.copy()
    for v1, v2 in EDGE_V:
        p1, p2 = vtx_px[v1], vtx_px[v2]
        cv2.line(ov, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (170, 170, 0), 1)
    for vi, (x, y) in enumerate(vtx_px):
        cv2.circle(ov, (int(x), int(y)), 3, (0, 255, 255), -1)
    for i, r, num, conf, pip, tok in rows:
        x, y = hex_px[i]
        cv2.putText(
            ov,
            f"H{i}:{r[:3]}",
            (int(x) - 30, int(y) - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            2,
        )
        if num is not None:
            col = (0, 200, 0) if PIP_FOR.get(num) == pip else (0, 0, 255)
            cv2.putText(
                ov, f"#{num}", (int(x) - 14, int(y) + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2
            )
    cv2.imwrite(f"{WORK}/overlay_{t}.png", ov)
    print(f"\n  wrote board_{t}.json, overlay_{t}.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
