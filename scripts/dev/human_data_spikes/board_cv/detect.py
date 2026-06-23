"""Board-CV spike: detect hexes, classify resources, read numbers, fit engine lattice.

Brutally-honest go/no-go feasibility probe. Not the full pipeline.
"""

import json
import sys

import cv2
import numpy as np

WORK = "/tmp/phantom/board_cv"
M0 = "/tmp/phantom/m0"

# ---- engine template (canonical pixel positions, pointy-top) ----
TPL = json.load(open(f"{WORK}/engine_template.json"))
TOPO = json.load(open(f"{M0}/topology.json"))
ENG_HEX = {int(k): np.array(v, float) for k, v in TPL["hex_centers"].items()}
ENG_VTX = {int(k): np.array(v, float) for k, v in TPL["vertex_px"].items()}

# Engine axial coords per hex (from getHexCoords) for ordering checks.
AXIAL = {
    0: (0, 0),
    1: (0, -1),
    2: (1, -1),
    3: (1, 0),
    4: (0, 1),
    5: (-1, 1),
    6: (-1, 0),
    7: (0, -2),
    8: (1, -2),
    9: (2, -2),
    10: (2, -1),
    11: (2, 0),
    12: (1, 1),
    13: (0, 2),
    14: (-1, 2),
    15: (-2, 2),
    16: (-2, 1),
    17: (-2, 0),
    18: (-1, -1),
}


def detect_tokens(img):
    """HoughCircles on number-token white disks. Returns list of (x,y,r)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # disks ~ 40px diameter at 1080p; tune.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=55,
        param1=120,
        param2=28,
        minRadius=16,
        maxRadius=34,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def main(t):
    path = f"{M0}/f1080_{t}.png"
    img = cv2.imread(path)
    H, W = img.shape[:2]

    tokens = detect_tokens(img)
    # Keep only tokens within the board region (rough: left 60% of width, exclude UI).
    tokens = [c for c in tokens if c[0] < W * 0.62 and 0.05 * H < c[1] < 0.98 * H]
    print(f"[t={t}] detected {len(tokens)} candidate number tokens")
    for x, y, r in sorted(tokens, key=lambda c: (c[1], c[0])):
        print(f"   token at ({x:.0f},{y:.0f}) r={r:.0f}")

    # overlay
    ov = img.copy()
    for x, y, r in tokens:
        cv2.circle(ov, (int(x), int(y)), int(r), (0, 0, 255), 2)
        cv2.circle(ov, (int(x), int(y)), 2, (0, 255, 0), 3)
    cv2.imwrite(f"{WORK}/tokens_{t}.png", ov)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
