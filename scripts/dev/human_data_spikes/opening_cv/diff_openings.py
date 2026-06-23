"""Incremental-diff opening detector.

Board is pixel-static across setup frames; we lock ONE affine (fit_247) as the
canonical engine<->frame lattice. Between consecutive setup frames exactly the
new piece(s) appear, so a frame-diff isolates them regardless of tile color
(this sidesteps the green-piece-vs-green-tile collision).

For each diff blob: classify settlement (compact, snaps to a VERTEX) vs road
(elongated, snaps to an EDGE midpoint); assign player by the blob's median hue
(GREEN=rayman147 vivid green; BLACK=ThePhantom dark). Robber excluded (it lands
on a HEX CENTER, far from any vertex/edge — and only after the first roll).
"""

import json

import cv2
import numpy as np

TOPO = json.load(open("/tmp/phantom/m0/topology.json"))
EDGE_V = TOPO["edgeVertices"]
FIT = json.load(open("fit_247.json"))
VTX = np.array(FIT["vtx_px"])
HEXC = np.array(FIT["hex_px"])
EDGE_MID = np.array([(VTX[a] + VTX[b]) / 2 for a, b in EDGE_V])
PLAYER = {"GREEN": "rayman147", "BLACK": "ThePhantom"}

# ordered setup frames (t, expected new pieces label)
SEQ = [105, 140, 160, 180, 220, 240, 247]
LABELS = {
    140: "rayman147 S1 + R1",
    160: "ThePhantom S1",
    180: "ThePhantom R1",
    220: "ThePhantom S2",
    240: "ThePhantom R2",
    247: "rayman147 S2 + R2",
}

H, W = cv2.imread("g_247.png").shape[:2]
hull = cv2.convexHull(VTX.astype(np.float32)).astype(np.int32)
BOARD = np.zeros((H, W), np.uint8)
cv2.fillConvexPoly(BOARD, hull, 255)
BOARD = cv2.erode(BOARD, np.ones((6, 6), np.uint8))  # stay off the sea/coast


def classify_blob(cx, cy, w, h, area, img):
    dv = np.linalg.norm(VTX - [cx, cy], axis=1)
    vmin = dv.min()
    vi = int(dv.argmin())
    de = np.linalg.norm(EDGE_MID - [cx, cy], axis=1)
    emin = de.min()
    ei = int(de.argmin())
    dh = np.linalg.norm(HEXC - [cx, cy], axis=1).min()
    elong = max(w, h) / max(min(w, h), 1)
    # player color from blob median hue
    patch = cv2.cvtColor(
        img[int(cy) - 8 : int(cy) + 8, int(cx) - 8 : int(cx) + 8], cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    hue = np.median(patch[:, 0])
    sat = np.median(patch[:, 1])
    val = np.median(patch[:, 2])
    color = "GREEN" if (38 <= hue <= 85 and sat > 110 and val > 110) else "BLACK"
    # settlement vs road: road is elongated and snaps closer to an edge than vertex
    is_road = (emin < vmin) and (elong > 1.6 or emin + 4 < vmin)
    return dict(
        color=color,
        vi=vi,
        vmin=vmin,
        ei=ei,
        emin=emin,
        dh=dh,
        elong=round(elong, 2),
        hue=int(hue),
        sat=int(sat),
        val=int(val),
        is_road=is_road,
    )


def main():
    results = {}  # accumulate accepted pieces
    settle = {}
    roads = {}
    prev = cv2.imread(f"g_{SEQ[0]}.png")
    for t in SEQ[1:]:
        cur = cv2.imread(f"g_{t}.png")
        d = cv2.absdiff(cur, prev).max(2)
        _, dm = cv2.threshold(d, 45, 255, cv2.THRESH_BINARY)
        dm = cv2.bitwise_and(dm, BOARD)
        dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(dm)
        blobs = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = cent[i]
            if a < 120:
                continue
            info = classify_blob(cx, cy, w, h, a, cur)
            info.update(cx=int(cx), cy=int(cy), area=int(a), w=int(w), h=int(h))
            blobs.append(info)
        # keep the biggest few (new pieces), reject diff noise far from any vtx & edge
        blobs = [b for b in blobs if min(b["vmin"], b["emin"]) < 40 and b["dh"] > 18]
        blobs.sort(key=lambda b: -b["area"])
        print(f"\n=== diff -> t={t}  ({LABELS.get(t, '?')})  blobs={len(blobs)} ===")
        for b in blobs:
            tag = f"ROAD e{b['ei']}" if b["is_road"] else f"SET  v{b['vi']}"
            print(
                f"   {b['color']:6s} {tag:9s} area={b['area']:4d} elong={b['elong']:.1f} "
                f"vmin={b['vmin']:.1f} emin={b['emin']:.1f} dh={b['dh']:.0f} hue={b['hue']} "
                f"sat={b['sat']} val={b['val']} ctr=({b['cx']},{b['cy']})"
            )
            if b["is_road"]:
                roads.setdefault(
                    b["ei"],
                    dict(
                        player=b["color"],
                        owner=PLAYER[b["color"]],
                        edge_id=b["ei"],
                        area=b["area"],
                        snap=round(b["emin"], 1),
                        seen=t,
                    ),
                )
            else:
                settle.setdefault(
                    b["vi"],
                    dict(
                        player=b["color"],
                        owner=PLAYER[b["color"]],
                        vertex_id=b["vi"],
                        area=b["area"],
                        snap=round(b["vmin"], 1),
                        seen=t,
                    ),
                )
        prev = cur

    print("\n========== ACCUMULATED (first-seen) ==========")
    print("SETTLEMENTS:")
    for vi, p in sorted(settle.items()):
        print(f"   {p['owner']:10s} v{vi:2d} snap={p['snap']:.1f} (frame t={p['seen']})")
    print("ROADS:")
    for ei, p in sorted(roads.items()):
        print(f"   {p['owner']:10s} e{ei:2d} snap={p['snap']:.1f} (frame t={p['seen']})")
    json.dump(
        {"settlements": settle, "roads": roads}, open("diff_raw.json", "w"), indent=2, default=int
    )
    return settle, roads


if __name__ == "__main__":
    main()
