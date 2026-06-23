"""Annotated overlays of the orientation-locked board for eyeballing."""

import cv2
import orient_lock2 as ol2

OUT = "/tmp/phantom/blockers"
ANCHORS = [
    {"name": "topcenter", "engine_id": 8, "resource": "SHEEP", "number": 2},
    {"name": "lowerleft", "engine_id": 16, "resource": "ORE", "number": 10},
]
EDGE_V = ol2.EDGE_V


def draw(t):
    img = cv2.imread(f"/tmp/phantom/m0/f1080_{t}.png")
    toks = ol2.base.detect_tokens(img)
    scored = ol2.pick_screen_orientation(toks)
    _, refl, deg, A, resid = scored[0]
    hx = ol2.apply(A, ol2.ENG_HEX)
    vx = ol2.apply(A, ol2.ENG_VTX)
    hexes, _, _ = ol2.read_full_board(img, A)
    # enforce 1 desert (the lone no-number hex)
    nn = [h for h in hexes if h["number"] is None]
    if len(nn) == 1:
        nn[0]["resource"] = "DESERT"
    ov = img.copy()
    for v1, v2 in EDGE_V:
        cv2.line(ov, tuple(vx[v1].astype(int)), tuple(vx[v2].astype(int)), (170, 170, 0), 1)
    for vi, (x, y) in enumerate(vx):
        cv2.circle(ov, (int(x), int(y)), 3, (0, 255, 255), -1)
    for h in hexes:
        e = h["hex_id"]
        x, y = hx[e]
        label = f"H{e}:{h['resource'][:3]}"
        cv2.putText(
            ov, label, (int(x) - 32, int(y) - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
        )
        if h["number"] is not None:
            cv2.putText(
                ov,
                f"#{h['number']}",
                (int(x) - 14, int(y) + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )
    # mark anchors
    for a in ANCHORS:
        x, y = hx[a["engine_id"]]
        cv2.circle(ov, (int(x), int(y)), 30, (0, 0, 255), 3)
        cv2.putText(
            ov,
            f"ANCHOR {a['resource']}#{a['number']}",
            (int(x) - 60, int(y) + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        ov,
        f"t={t} LOCK refl={refl} rot={deg} resid={resid:.2f}px",
        (20, 1060),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(f"{OUT}/overlays/B2_locked_overlay_{t}.png", ov)
    print(f"wrote B2_locked_overlay_{t}.png")


for t in (240, 500):
    draw(t)
