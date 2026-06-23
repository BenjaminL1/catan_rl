"""Diagnose BLOCKER 2 root cause: show that the 12 candidate orientations have
near-tied geometric residuals (so 'lowest residual' silently flips between frames).
"""

import json
import math

import cv2
import numpy as np

WORK = "/tmp/phantom/blockers"
M0 = "/tmp/phantom/m0"
TPL = json.load(open(f"{WORK}/engine_template.json"))
ENG_HEX = np.array([TPL["hex_centers"][str(i)] for i in range(19)], float)


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
    return toks


def orientation_residuals(toks):
    tok = np.array([[x, y] for x, y, _ in toks], float)
    fr_c = tok.mean(0)
    eng_c = ENG_HEX.mean(0)
    s0 = np.linalg.norm(tok - fr_c, axis=1).mean() / np.linalg.norm(ENG_HEX - eng_c, axis=1).mean()
    rows = []
    for refl in (1, -1):
        for k in range(12):
            ang = math.radians(30 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            Rf = R @ np.array([[refl, 0], [0, 1]], float)
            eng_t = (ENG_HEX - eng_c) @ Rf.T * s0 + fr_c
            used = set()
            resid = 0.0
            npair = 0
            for p in tok:
                d = np.linalg.norm(eng_t - p, axis=1)
                for hi in np.argsort(d):
                    if hi not in used:
                        used.add(hi)
                        resid += d[hi]
                        npair += 1
                        break
            rows.append((refl, k * 30, resid / npair))
    return rows


for t in (300, 350, 500):
    img = cv2.imread(f"{M0}/f1080_{t}.png")
    toks = detect_tokens(img)
    rows = orientation_residuals(toks)
    rows.sort(key=lambda r: r[2])
    print(f"\n[t={t}] {len(toks)} tokens; top-6 orientations by greedy residual:")
    for refl, deg, r in rows[:6]:
        print(f"   refl={refl:+d} rot={deg:3d}deg  resid={r:6.2f}px")
    best = rows[0][2]
    second = rows[1][2]
    print(f"   --> best {best:.2f}px vs 2nd {second:.2f}px  (gap {second - best:.2f}px)")
