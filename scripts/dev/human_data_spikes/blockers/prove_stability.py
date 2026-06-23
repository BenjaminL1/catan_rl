"""BLOCKER 2 proof: run the deterministic orientation lock on >=3 game-1 frames
and show the engine-indexed board map is IDENTICAL across all of them.
Prints a per-hex agreement table. Also runs the rejection test on a deliberately
mis-oriented fit.
"""

import json
import subprocess

import cv2
import orient_lock2 as ol2

OUT = "/tmp/phantom/blockers"
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
URL = open(f"{OUT}/stream_url.txt").read().strip()

# Ground-truth content anchors (engine ids established by the screen-rule lock at
# t=500, eyeball-confirmed: H8=SHEEP/2 top-center, H16=ORE/10 lower-left).
ANCHORS = [
    {"name": "topcenter", "engine_id": 8, "resource": "SHEEP", "number": 2},
    {"name": "lowerleft", "engine_id": 16, "resource": "ORE", "number": 10},
]

STD_NUMS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
STD_RES = {"WOOD": 4, "SHEEP": 4, "WHEAT": 4, "ORE": 3, "BRICK": 3, "DESERT": 1}


def grab(t, dst):
    subprocess.run(
        [FFMPEG, "-nostdin", "-ss", str(t), "-i", URL, "-frames:v", "1", "-q:v", "2", "-y", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def fix_desert(hexes):
    """Standard board has exactly 1 desert. The desert hex carries a robber and
    has no number token. If the color classifier mislabeled it, the hex with
    number=None (no token) is the desert. Enforce exactly-one-desert."""
    no_num = [h for h in hexes if h["number"] is None]
    if len(no_num) == 1:
        no_num[0]["resource"] = "DESERT"
    return hexes


def main():
    # GAME A (the blocker's "game 1"): one continuous game, setup t=80-240,
    # reset (next game) at t=620. Anchor content SHEEP#2 / ORE#10 verified on each
    # of these frames. t=120/180/240 = setup phase; 350/450/500/540 = mid-game.
    # t=300 is an animation/occlusion frame (only 12 tokens detected) -> REJECTS.
    frame_ts = [120, 180, 240, 300, 350, 450, 500, 540]
    boards = {}
    diags = {}
    for t in frame_ts:
        path = f"/tmp/phantom/m0/f1080_{t}.png"
        import os

        if not os.path.exists(path):
            path = f"{OUT}/frames/stab_{t}.png"
            grab(t, path)
        img = cv2.imread(path)
        hexes, A, diag = ol2.lock_and_read(img, ANCHORS)
        diags[t] = diag
        if hexes is None:
            print(f"[t={t}] REJECTED by lock: {diag.get('REJECTED')}")
            print(f"        content_checks: {diag['content_checks']}")
            boards[t] = None
            continue
        hexes = fix_desert(hexes)
        boards[t] = hexes
        json.dump(
            {"frame_t": t, "hexes": hexes},
            open(f"{OUT}/board/locked_board_{t}.json", "w"),
            indent=2,
        )
        rescount = {}
        for h in hexes:
            rescount[h["resource"]] = rescount.get(h["resource"], 0) + 1
        nums = sorted(h["number"] for h in hexes if h["number"] is not None)
        print(
            f"[t={t}] LOCKED refl={diag['refl']} rot={diag['rot']} "
            f"pen={diag['screen_rule_penalty']:.1f}(2nd {diag['second_penalty']:.0f}) "
            f"resid={diag['residual_mean']:.2f} content_ok={diag['content_consistent']}"
        )
        print(f"        res multiset {rescount} std_ok={rescount == STD_RES}")
        print(f"        num multiset matches std: {nums == STD_NUMS} ({len(nums)}/18)")

    # ---- per-hex agreement table across frames ----
    good_ts = [t for t in frame_ts if boards[t] is not None]
    print("\n=== PER-HEX AGREEMENT TABLE (engine id -> resource#number per frame) ===")
    hdr = "hex | " + " | ".join(f"t={t:<8}" for t in good_ts) + " | AGREE"
    print(hdr)
    print("-" * len(hdr))
    all_agree = True
    for e in range(19):
        cells = []
        vals = []
        for t in good_ts:
            h = boards[t][e]
            s = f"{h['resource'][:4]}#{h['number']}"
            cells.append(f"{s:<10}")
            vals.append((h["resource"], h["number"]))
        agree = len(set(vals)) == 1
        all_agree = all_agree and agree
        print(f"H{e:2d} | " + " | ".join(cells) + f" | {'OK' if agree else 'MISMATCH'}")
    print("-" * len(hdr))
    print(f"\nALL HEXES IDENTICAL ACROSS {len(good_ts)} FRAMES: {all_agree}")

    # desert location stability
    deserts = {t: [h["hex_id"] for h in boards[t] if h["resource"] == "DESERT"] for t in good_ts}
    print(f"desert hex_id per frame: {deserts}")

    # ---- rejection test: deliberately mis-orient and confirm REJECT ----
    print("\n=== REJECTION TEST (deliberate mis-orientation) ===")
    img = cv2.imread("/tmp/phantom/m0/f1080_500.png")
    toks = ol2.base.detect_tokens(img)
    scored = ol2.pick_screen_orientation(toks)
    # take the 2nd-best (wrong) orientation's affine and run PART B content check
    _, refl, deg, A_wrong, _ = scored[1]
    checks = []
    for a in ANCHORS:
        res, num, px = ol2.read_screen_at(img, A_wrong, a["engine_id"])
        checks.append(
            {
                "name": a["name"],
                "want": (a["resource"], a["number"]),
                "got": (res, num),
                "ok": (res == a["resource"] and num == a["number"]),
            }
        )
    rejected = not all(c["ok"] for c in checks)
    print(f"forced WRONG orientation refl={refl} rot={deg}:")
    for c in checks:
        print(f"   {c['name']}: want {c['want']} got {c['got']} -> {'PASS' if c['ok'] else 'FAIL'}")
    print(f"consistency check REJECTS the mis-oriented fit: {rejected}")

    json.dump(
        {
            "boards": {t: boards[t] for t in good_ts},
            "all_agree": all_agree,
            "deserts": deserts,
            "rejection_works": rejected,
        },
        open(f"{OUT}/board/stability_result.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
