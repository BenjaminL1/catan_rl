"""Piece detection spike v2 — calibrated to the two players in the clip.

Players in this game: GREEN (ThePhantom) = bright saturated green
(HSV hue~63, S>225, V>200, brighter+more saturated than any hex/icon green);
BLACK (rayman147) = dark, low-sat (V<120, S<70).

Pieces snap to engine vertices (settlement/city) or edges (road).
Settlement vs city by blob area. Robber (dark blob on a HEX CENTER) is
excluded by requiring buildings to snap close to a vertex, not a hex center.
"""

import json
import sys

import cv2
import numpy as np

WORK = "/tmp/phantom/board_cv"
M0 = "/tmp/phantom/m0"
TOPO = json.load(open(f"{M0}/topology.json"))
EDGE_V = TOPO["edgeVertices"]


def player_masks(hsv):
    green = cv2.inRange(hsv, (45, 200, 170), (78, 255, 255))
    black = cv2.inRange(hsv, (0, 0, 30), (179, 70, 120))
    return {"GREEN": green, "BLACK": black}


def main(t):
    img = cv2.imread(f"{M0}/f1080_{t}.png")
    fit = json.load(open(f"{WORK}/fit_{t}.json"))
    vtx_px = np.array(fit["vtx_px"])
    hex_px = np.array(fit["hex_px"])
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hull = cv2.convexHull(vtx_px.astype(np.float32)).astype(np.int32)
    board_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(board_mask, hull, 255)
    board_mask = cv2.dilate(board_mask, np.ones((20, 20), np.uint8))

    edge_mid = np.array([(vtx_px[a] + vtx_px[b]) / 2 for a, b in EDGE_V])

    overlay = img.copy()
    for v1, v2 in EDGE_V:
        cv2.line(
            overlay, tuple(vtx_px[v1].astype(int)), tuple(vtx_px[v2].astype(int)), (160, 160, 0), 1
        )

    buildings = {}  # vertex_id -> piece
    roads = {}  # edge_id -> piece
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
            if a < 90:
                continue
            fill = a / max(w * h, 1)  # roads = sparse line in bbox; buildings = filled blob
            dv = np.linalg.norm(vtx_px - [cx, cy], axis=1)
            vmin = dv.min()
            vi = int(dv.argmin())
            de = np.linalg.norm(edge_mid - [cx, cy], axis=1)
            emin = de.min()
            ei = int(de.argmin())
            dh = np.linalg.norm(hex_px - [cx, cy], axis=1).min()  # dist to nearest hex center
            # Road: snaps better to an edge-midpoint than a vertex, and is a
            # sparse (low-fill) shape in its bounding box.
            road_like = (emin < vmin) and (emin < 26) and (fill < 0.55)
            if road_like:
                if ei not in roads or a > roads[ei]["area"]:
                    roads[ei] = {
                        "player": color,
                        "kind": "road",
                        "edge_id": ei,
                        "px": [float(cx), float(cy)],
                        "area": int(a),
                        "snap_err": float(emin),
                    }
            elif vmin < 24 and dh > 35:  # building at vertex, NOT a robber on a hex center
                kind = "city" if a > 600 else "settlement"
                if vi not in buildings or a > buildings[vi]["area"]:
                    buildings[vi] = {
                        "player": color,
                        "kind": kind,
                        "vertex_id": vi,
                        "px": [float(cx), float(cy)],
                        "area": int(a),
                        "snap_err": float(vmin),
                    }

    for p in buildings.values():
        cx, cy = p["px"]
        col = (0, 200, 0) if p["player"] == "GREEN" else (40, 40, 40)
        cv2.circle(overlay, (int(cx), int(cy)), 11, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{p['player'][:1]}{p['kind'][:1]}v{p['vertex_id']}",
            (int(cx) + 10, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            2,
        )
    for p in roads.values():
        v1, v2 = EDGE_V[p["edge_id"]]
        cv2.line(
            overlay, tuple(vtx_px[v1].astype(int)), tuple(vtx_px[v2].astype(int)), (0, 0, 255), 3
        )
        mx, my = edge_mid[p["edge_id"]]
        cv2.putText(
            overlay,
            f"{p['player'][:1]}r{p['edge_id']}",
            (int(mx) + 4, int(my)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

    print(f"[t={t}] buildings: {len(buildings)}  roads: {len(roads)}")
    for p in sorted(buildings.values(), key=lambda z: z["vertex_id"]):
        print(
            f"   {p['player']:6s} {p['kind']:10s} v{p['vertex_id']:2d} area={p['area']:4d} snap={p['snap_err']:.1f}"
        )
    for p in sorted(roads.values(), key=lambda z: z["edge_id"]):
        print(
            f"   {p['player']:6s} road       e{p['edge_id']:2d} area={p['area']:4d} snap={p['snap_err']:.1f}"
        )

    cv2.imwrite(f"{WORK}/pieces_{t}.png", overlay)
    board = json.load(open(f"{WORK}/board_{t}.json"))
    board["pieces"] = list(buildings.values()) + list(roads.values())
    json.dump(board, open(f"{WORK}/board_{t}.json", "w"), indent=2)
    print(f"   wrote pieces_{t}.png + merged into board_{t}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
