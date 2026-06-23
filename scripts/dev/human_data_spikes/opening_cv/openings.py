"""Opening-placement detector (post-setup single frame).

Color map (READ FROM HUD, t=120): rayman147 = GREEN, ThePhantom = BLACK.
Detects: settlements (snap to vertex), roads (snap to edge midpoint).
Excludes robber (dark blob on a HEX CENTER, not near a vertex).
"""

import json
import sys

import cv2
import numpy as np

TOPO = json.load(open("/tmp/phantom/m0/topology.json"))
EDGE_V = TOPO["edgeVertices"]
PLAYER_COLOR = {"GREEN": "rayman147", "BLACK": "ThePhantom"}


def player_masks(hsv):
    # GREEN pieces: bright saturated green (distinct from hex-tile greens which
    # are more muted/yellowish). Pieces are vivid.
    green = cv2.inRange(hsv, (40, 120, 90), (85, 255, 255))
    # BLACK pieces: very dark, low value. Robber is gray (higher V); pieces are darker.
    black = cv2.inRange(hsv, (0, 0, 0), (179, 80, 95))
    return {"GREEN": green, "BLACK": black}


def detect(path, tag, fit):
    img = cv2.imread(path)
    vtx_px = np.array(fit["vtx_px"])
    hex_px = np.array(fit["hex_px"])
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hull = cv2.convexHull(vtx_px.astype(np.float32)).astype(np.int32)
    board_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(board_mask, hull, 255)
    board_mask = cv2.dilate(board_mask, np.ones((25, 25), np.uint8))

    edge_mid = np.array([(vtx_px[a] + vtx_px[b]) / 2 for a, b in EDGE_V])

    overlay = img.copy()
    for v1, v2 in EDGE_V:
        cv2.line(
            overlay, tuple(vtx_px[v1].astype(int)), tuple(vtx_px[v2].astype(int)), (150, 150, 0), 1
        )
    for vi, (x, y) in enumerate(vtx_px):
        cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 255), -1)

    buildings, roads = {}, {}
    raw = {"GREEN": [], "BLACK": []}
    for color, m in player_masks(hsv).items():
        m = cv2.bitwise_and(m, board_mask)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = cent[i]
            if a < 70:
                continue
            fill = a / max(w * h, 1)
            dv = np.linalg.norm(vtx_px - [cx, cy], axis=1)
            vmin = dv.min()
            vi = int(dv.argmin())
            de = np.linalg.norm(edge_mid - [cx, cy], axis=1)
            emin = de.min()
            ei = int(de.argmin())
            dh = np.linalg.norm(hex_px - [cx, cy], axis=1).min()
            raw[color].append(
                (
                    int(cx),
                    int(cy),
                    int(a),
                    round(fill, 2),
                    round(vmin, 1),
                    vi,
                    round(emin, 1),
                    ei,
                    round(dh, 1),
                    w,
                    h,
                )
            )
            # robber sits ON a hex center; pieces never do. Exclude.
            if dh < 22:
                continue
            road_like = (emin + 3 < vmin) and (emin < 30) and (fill < 0.60)
            if road_like:
                if ei not in roads or a > roads[ei]["area"]:
                    roads[ei] = {
                        "player": color,
                        "owner": PLAYER_COLOR[color],
                        "kind": "road",
                        "edge_id": ei,
                        "px": [float(cx), float(cy)],
                        "area": int(a),
                        "snap_err": float(emin),
                    }
            elif vmin < 26:
                if vi not in buildings or a > buildings[vi]["area"]:
                    buildings[vi] = {
                        "player": color,
                        "owner": PLAYER_COLOR[color],
                        "kind": "settlement",
                        "vertex_id": vi,
                        "px": [float(cx), float(cy)],
                        "area": int(a),
                        "snap_err": float(vmin),
                    }

    # annotate
    for p in buildings.values():
        cx, cy = p["px"]
        bgr = (0, 200, 0) if p["player"] == "GREEN" else (20, 20, 20)
        cv2.circle(overlay, (int(cx), int(cy)), 13, bgr, 3)
        cv2.circle(overlay, (int(cx), int(cy)), 13, (0, 0, 255), 1)
        cv2.putText(
            overlay,
            f"{p['owner'][:4]} S v{p['vertex_id']}",
            (int(cx) + 12, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    for p in roads.values():
        v1, v2 = EDGE_V[p["edge_id"]]
        bgr = (0, 200, 0) if p["player"] == "GREEN" else (20, 20, 20)
        cv2.line(overlay, tuple(vtx_px[v1].astype(int)), tuple(vtx_px[v2].astype(int)), bgr, 5)
        mx, my = edge_mid[p["edge_id"]]
        cv2.putText(
            overlay,
            f"{p['owner'][:4]} R e{p['edge_id']}",
            (int(mx) + 4, int(my) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(f"overlay_{tag}.png", overlay)

    print(f"[{tag}] settlements={len(buildings)} roads={len(roads)}")
    print("  --- raw blobs (cx,cy,area,fill,vmin,vi,emin,ei,dhex,w,h) ---")
    for color in ("GREEN", "BLACK"):
        for r in sorted(raw[color], key=lambda z: -z[2]):
            print(f"   {color:6s}", r)
    print("  --- accepted ---")
    for p in sorted(buildings.values(), key=lambda z: z["vertex_id"]):
        print(
            f"   SET  {p['owner']:10s} v{p['vertex_id']:2d} area={p['area']:4d} snap={p['snap_err']:.1f}"
        )
    for p in sorted(roads.values(), key=lambda z: z["edge_id"]):
        print(
            f"   ROAD {p['owner']:10s} e{p['edge_id']:2d} area={p['area']:4d} snap={p['snap_err']:.1f}"
        )
    return buildings, roads


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "247"
    fit = json.load(open(f"fit_{tag}.json"))
    detect(f"g_{tag}.png", tag, fit)
