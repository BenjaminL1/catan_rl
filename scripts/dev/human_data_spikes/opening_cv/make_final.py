import json

import cv2
import numpy as np

FIT = json.load(open("fit_247.json"))
VTX = np.array(FIT["vtx_px"])
TOPO = json.load(open("/tmp/phantom/m0/topology.json"))
EDGE_V = TOPO["edgeVertices"]
EM = np.array([(VTX[a] + VTX[b]) / 2 for a, b in EDGE_V])
img = cv2.imread("g_247.png")

FINAL = {
    "rayman147": {"color": "GREEN", "settlements": [20, 0], "roads": [34, 2]},
    "ThePhantom": {"color": "BLACK", "settlements": [4, 10], "roads": [7, 20]},
}
BGR = {"GREEN": (0, 210, 0), "BLACK": (15, 15, 15)}

ov = img.copy()
# faint full lattice
for v1, v2 in EDGE_V:
    cv2.line(ov, tuple(VTX[v1].astype(int)), tuple(VTX[v2].astype(int)), (110, 110, 0), 1)
for vi, (x, y) in enumerate(VTX):
    cv2.circle(ov, (int(x), int(y)), 2, (0, 255, 255), -1)
# draw detected pieces
for pl, d in FINAL.items():
    c = BGR[d["color"]]
    for s in d["settlements"]:
        x, y = VTX[s]
        cv2.circle(ov, (int(x), int(y)), 14, c, -1)
        cv2.circle(ov, (int(x), int(y)), 14, (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"{pl[:4]} S v{s}",
            (int(x) + 16, int(y) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    for r in d["roads"]:
        v1, v2 = EDGE_V[r]
        cv2.line(ov, tuple(VTX[v1].astype(int)), tuple(VTX[v2].astype(int)), c, 6)
        cv2.line(ov, tuple(VTX[v1].astype(int)), tuple(VTX[v2].astype(int)), (0, 0, 255), 2)
        mx, my = EM[r]
        cv2.putText(
            ov,
            f"{pl[:4]} R e{r}",
            (int(mx) + 6, int(my) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
cv2.imwrite("overlay_openings.png", ov)
# zoomed
crop = ov[40:880, 150:1180]
crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("overlay_openings_zoom.png", crop)
print("wrote overlay_openings.png + overlay_openings_zoom.png")

# openings.json (engine indices) + cross-check metadata
out = {
    "video": "https://www.youtube.com/watch?v=9Sm86ml04aI",
    "game": 1,
    "frame_used": "g_247.png (t=247s, post-setup, pre-first-roll)",
    "fit": {
        "affine_resid_mean_px": FIT["resid_mean"],
        "affine_resid_max_px": FIT["resid_max"],
        "desert_hex": 17,
    },
    "player_colors": {"rayman147": "GREEN", "ThePhantom": "BLACK"},
    "draft_order": ["rayman147", "ThePhantom", "ThePhantom", "rayman147"],
    "log_setup_sequence": [
        "rayman147 placed a Settlement",
        "rayman147 placed a Road",
        "ThePhantom placed a Settlement",
        "ThePhantom placed a Road",
        "ThePhantom placed a Settlement",
        "ThePhantom received starting resources",
        "ThePhantom placed a Road",
        "rayman147 placed a Settlement",
        "rayman147 received starting resources",
        "rayman147 placed a Road",
        "rayman147 rolled (= setup complete)",
    ],
    "openings": {
        "rayman147": {"settlements": [20, 0], "roads": [34, 2]},
        "ThePhantom": {"settlements": [4, 10], "roads": [7, 20]},
    },
}
json.dump(out, open("openings.json", "w"), indent=2)
print(json.dumps(out["openings"], indent=2))
