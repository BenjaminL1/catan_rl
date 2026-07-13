#!/usr/bin/env python3
"""Render a VERTEX-ID overlay on a prepared game's post-setup frame.

The VLM localization step must name, for each settlement, the hex-set its corner
touches (and for each road, both endpoint hex-sets). Eyeballing that off a bare
screenshot is error-prone; this draws the engine's own vertex ids at their true image
positions so the read is CHECKABLE rather than guessed.

Board geometry comes from the SAME source the snapper uses, in preference order:
  1. ``read_board_stable`` on the empty-baseline frame (a real affine + vertex_px);
  2. failing that (occlusion / a splash baseline), a 3-point affine fitted from
     hex-centre references the caller supplies (``--refs '{"7":[655,245], ...}'``),
     mapping the engine's axial hex layout onto the frame.

Also prints the board's hex layout (top row -> bottom, with resource+number) so the
hex ids can be matched to what is visible, and the per-player GRANT multiset — which
is the mandatory self-check: a player's grant must equal the adjacent-resource multiset
of EXACTLY ONE of their two settlements.

Usage::

    PYTHONPATH=src python3 scripts/dev/localize_overlay.py 33KR75rhTgo__g2
    PYTHONPATH=src python3 scripts/dev/localize_overlay.py AoOXWyxaTkA__g1 \
        --refs '{"7":[655,245],"9":[985,245],"14":[820,825]}'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2
import imageio.v3 as iio
import numpy as np

from catan_rl.engine.board import catanBoard
from catan_rl.human_data.board_cv import read_board_stable
from catan_rl.human_data.topology import load_topology

FRAMES = REPO / "data/human/vlm_spike/frames"


def hex_model_centres() -> dict[int, tuple[float, float]]:
    """Engine hex id -> (x, y) in model space (axial -> pointy-top pixel)."""
    b = catanBoard()
    return {hid: (math.sqrt(3) * (t.q + t.r / 2.0), 1.5 * t.r) for hid, t in b.hexTileDict.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game", help="prepared game key, e.g. 33KR75rhTgo__g2")
    ap.add_argument("--refs", default="", help='JSON {"hex_id":[img_x,img_y], ...} (>=3)')
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = FRAMES / args.game
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    res = {int(h["hex_id"]): (str(h["resource"]), h["number"]) for h in meta["board_hexes"]}
    topo = load_topology()
    post = np.asarray(iio.imread(root / "post_setup.png"))

    # --- board layout (so hex ids can be matched to what is on screen) ---
    mc = hex_model_centres()
    rows: dict[float, list[tuple[float, int]]] = {}
    for hid, (x, y) in mc.items():
        rows.setdefault(round(y, 2), []).append((round(x, 2), hid))
    print(f"=== {args.game}: BOARD (top row -> bottom) ===")
    for y in sorted(rows):
        cells = [f"h{hid}={res[hid][0][:3]}{res[hid][1] or ''}" for _, hid in sorted(rows[y])]
        print("   " + "   ".join(cells))

    # --- vertex pixel positions ---
    vpx: dict[int, tuple[int, int]] | None = None
    base = root / "empty_baseline.png"
    if base.exists():
        arr = np.asarray(iio.imread(base))
        board = read_board_stable([arr, arr.copy()])
        if board is not None:
            print(
                f"[overlay] board fitted from empty_baseline (residual {board.residual_px:.2f}px)"
            )
            vpx = {v: (int(board.vertex_px[v][0]), int(board.vertex_px[v][1])) for v in range(54)}

    if vpx is None:
        if not args.refs:
            raise SystemExit(
                "board unreadable from the baseline — supply >=3 hex-centre refs, e.g.\n"
                '  --refs \'{"7":[655,245],"9":[985,245],"14":[820,825]}\''
            )
        refs = json.loads(args.refs)
        src = np.array([mc[int(h)] for h in refs], dtype=np.float32)[:3]
        dst = np.array([refs[h] for h in refs], dtype=np.float32)[:3]
        A = cv2.getAffineTransform(src, dst)
        print("[overlay] board fitted from 3 hex-centre refs")
        vpx = {}
        for v, hexes in enumerate(topo.vertex_adjacent_hexes):
            x = sum(mc[h][0] for h in hexes) / len(hexes)
            y = sum(mc[h][1] for h in hexes) / len(hexes)
            p = A @ np.array([x, y, 1.0])
            vpx[v] = (int(p[0]), int(p[1]))

    img = cv2.cvtColor(post.copy(), cv2.COLOR_RGB2BGR)
    for v, (x, y) in vpx.items():
        cv2.circle(img, (x, y), 14, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 14, (0, 0, 255), 2)
        cv2.putText(
            img, str(v), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
    out = Path(args.out) if args.out else root / "overlay.png"
    cv2.imwrite(str(out), img)
    print(f"[overlay] wrote {out}")

    # --- the MANDATORY self-check inputs ---
    print(
        "\n=== GRANTS (each must equal the hexes of EXACTLY ONE of that player's settlements) ==="
    )
    print(f"  colours: {meta.get('player_colors')}")
    for handle, grant in meta.get("granted_resources", {}).items():
        print(f"  {handle:12s} {grant}")
    print("\n=== vertex -> hexes (resources) ===")
    for v in range(54):
        hx = sorted(topo.vertex_adjacent_hexes[v])
        desc = " ".join(f"{res[h][0][:3]}{res[h][1] or ''}" for h in hx)
        print(f"  v{v:2d} {hx}  {desc}")


if __name__ == "__main__":
    main()
