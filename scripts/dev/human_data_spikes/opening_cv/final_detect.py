"""Final opening detector for post-setup frame t=247.

GREEN (rayman147): tile-green subtraction (green present in EMPTY t=105 = tile,
  not piece) isolates green pieces from green forest/pasture tiles.
BLACK (ThePhantom): dark low-V mask; gray robber excluded by V band + hex-center test.
Settlement = compact blob snapping to a VERTEX; road = elongated blob snapping to
an EDGE midpoint. We KNOW from the log: 2 settlements + 2 roads per player.
"""

import json

import cv2
import numpy as np

FIT = json.load(open("fit_247.json"))
VTX = np.array(FIT["vtx_px"])
HEXC = np.array(FIT["hex_px"])
TOPO = json.load(open("/tmp/phantom/m0/topology.json"))
EDGE_V = TOPO["edgeVertices"]
EM = np.array([(VTX[a] + VTX[b]) / 2 for a, b in EDGE_V])
POST = cv2.imread("g_247.png")
EMP = cv2.imread("g_105.png")
H, W = POST.shape[:2]
hull = cv2.convexHull(VTX.astype(np.float32)).astype(np.int32)
BM = np.zeros((H, W), np.uint8)
cv2.fillConvexPoly(BM, hull, 255)
BM = cv2.erode(BM, np.ones((8, 8), np.uint8))


def green_piece_mask():
    he = cv2.cvtColor(EMP, cv2.COLOR_BGR2HSV)
    hp = cv2.cvtColor(POST, cv2.COLOR_BGR2HSV)
    g = lambda h: cv2.inRange(h, (35, 70, 70), (90, 255, 255))
    tile = cv2.dilate(g(he), np.ones((11, 11), np.uint8))  # green that is TILE
    pm = cv2.bitwise_and(g(hp), cv2.bitwise_not(tile))
    pm = cv2.bitwise_and(pm, BM)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return pm


def black_piece_mask():
    hp = cv2.cvtColor(POST, cv2.COLOR_BGR2HSV)
    bm = cv2.inRange(hp, (0, 0, 0), (179, 95, 88))  # robber is grayer/brighter -> excluded
    bm = cv2.bitwise_and(bm, BM)
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return bm


def extract(mask, color):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    sets, roads = [], []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = cent[i]
        if a < 250:
            continue
        dh = np.linalg.norm(HEXC - [cx, cy], axis=1).min()
        if dh < 18:  # on a hex center -> robber/number, skip
            continue
        # A setup blob is often settlement+road fused. Find the vertex it touches
        # (settlement head) and the edge it runs along (road). Report both.
        dv = np.linalg.norm(VTX - [cx, cy], axis=1)
        vi = int(dv.argmin())
        vmin = float(dv.min())
        # the component pixel set: use its bbox to find vertex inside bbox
        ys, xs = np.where(lab == i)
        # vertex with most piece-pixels nearby = settlement head
        vd = np.linalg.norm(
            VTX[:, None, :] - np.stack([xs, ys], 1)[None, :, :], axis=2
        )  # 54 x npix
        vcount = (vd < 16).sum(1)
        head_v = int(vcount.argmax())
        head_n = int(vcount.max())
        # edge with most piece-pixels along it (road)
        ed = np.linalg.norm(EM[:, None, :] - np.stack([xs, ys], 1)[None, :, :], axis=2)
        ecount = (ed < 14).sum(1)
        road_e = int(ecount.argmax())
        road_n = int(ecount.max())
        sets.append(
            dict(color=color, vertex_id=head_v, vpix=head_n, px=[float(cx), float(cy)], area=int(a))
        )
        if road_n > 30:
            roads.append(dict(color=color, edge_id=road_e, epix=road_n, area=int(a)))
    return sets, roads


PLAYER = {"GREEN": "rayman147", "BLACK": "ThePhantom"}
gm = green_piece_mask()
bm = black_piece_mask()
cv2.imwrite("mask_green.png", gm)
cv2.imwrite("mask_black.png", bm)
gs, gr = extract(gm, "GREEN")
bs, br = extract(bm, "BLACK")
print("GREEN settlement-heads:", [(s["vertex_id"], s["vpix"], s["area"]) for s in gs])
print("GREEN road-edges:", [(r["edge_id"], r["epix"]) for r in gr])
print("BLACK settlement-heads:", [(s["vertex_id"], s["vpix"], s["area"]) for s in bs])
print("BLACK road-edges:", [(r["edge_id"], r["epix"]) for r in br])

# overlay
ov = POST.copy()
for v1, v2 in EDGE_V:
    cv2.line(ov, tuple(VTX[v1].astype(int)), tuple(VTX[v2].astype(int)), (120, 120, 0), 1)
for vi, (x, y) in enumerate(VTX):
    cv2.circle(ov, (int(x), int(y)), 2, (0, 255, 255), -1)
allsets = gs + bs
allroads = gr + br
for s in allsets:
    x, y = VTX[s["vertex_id"]]
    bgr = (0, 220, 0) if s["color"] == "GREEN" else (10, 10, 10)
    cv2.circle(ov, (int(x), int(y)), 12, bgr, 3)
    cv2.circle(ov, (int(x), int(y)), 12, (0, 0, 255), 1)
    cv2.putText(
        ov,
        f"{PLAYER[s['color']][:4]} S v{s['vertex_id']}",
        (int(x) + 12, int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
for r in allroads:
    v1, v2 = EDGE_V[r["edge_id"]]
    bgr = (0, 220, 0) if r["color"] == "GREEN" else (10, 10, 10)
    cv2.line(ov, tuple(VTX[v1].astype(int)), tuple(VTX[v2].astype(int)), bgr, 5)
    mx, my = EM[r["edge_id"]]
    cv2.putText(
        ov,
        f"{PLAYER[r['color']][:4]} R e{r['edge_id']}",
        (int(mx) + 4, int(my) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        2,
    )
cv2.imwrite("overlay_openings.png", ov)
print("wrote overlay_openings.png")
