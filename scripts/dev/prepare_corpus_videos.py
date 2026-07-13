#!/usr/bin/env python3
"""Prepare frames for N high-tier videos, sequentially, for the corpus build.

ONE ingest per video serves BOTH deliverables:
  * the YIELD numbers (games seen / clean opening / both grants readable), and
  * the persisted ``post_setup.png`` + ``meta.json`` the VLM localization step needs.

Running the yield sweep and ``prepare-frames`` separately would ingest every video
TWICE (~50 min each, OCR-bound) — this does it once.

A game is LOCALIZABLE iff the fail-closed router produced a clean 8-pieces-down frame
AND both players' grant glyphs were read (the glyph anchor is the only joint-flip
defence, so a game missing either grant is rejected no matter how good the frame is).

Writes a per-video tally to ``runs/human/corpus_prep.json`` after EACH video, so an
interrupted overnight run loses nothing. Skips videos already prepared.

Usage::

    PYTHONPATH=src python3 scripts/dev/prepare_corpus_videos.py --videos 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "data/human/strength_manifest.json"
FRAMES = REPO / "data/human/vlm_spike/frames"
OUT = REPO / "runs/human/corpus_prep.json"


def high_tier_videos() -> list[str]:
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    vids = d if isinstance(d, list) else d.get("videos", d)
    items = vids.items() if isinstance(vids, dict) else [(v.get("video_id"), v) for v in vids]
    return [k for k, v in items if isinstance(v, dict) and v.get("strength") == "high"]


def tally(video: str) -> dict:
    """Per-game tally from the frame dirs prepare-frames persisted for this video."""
    games = []
    for d in sorted(FRAMES.glob(f"{video}__g*")):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        grants = meta.get("granted_resources", {})
        players = list(meta.get("players", {}).values())
        both = len(players) == 2 and all(p in grants for p in players)
        games.append(
            {
                "game": d.name,
                "winner": meta.get("winner"),
                "grants_readable": both,
                "localizable": both,  # the dir exists => the router gave a clean frame
                "players": players,
            }
        )
    return {"video": video, "games": games, "n_games": len(games)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))

    todo = [v for v in high_tier_videos() if v not in results and v not in args.skip]
    todo = todo[: args.videos]
    print(f"[prep] {len(todo)} videos: {todo}", flush=True)

    for i, video in enumerate(todo, 1):
        print(f"[prep] {i}/{len(todo)} {video}: ingesting…", flush=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/vlm_spike.py"),
                    "prepare-frames",
                    "--video",
                    video,
                ],
                cwd=str(REPO),
                env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO / "src")},
                check=True,
                capture_output=True,
                timeout=5400,
            )
            results[video] = tally(video)
            r = results[video]
            loc = sum(g["localizable"] for g in r["games"])
            print(
                f"[prep] {i}/{len(todo)} {video}: games={r['n_games']} localizable={loc}",
                flush=True,
            )
        except Exception as exc:  # one bad video must not kill the night
            results[video] = {"video": video, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[prep] {i}/{len(todo)} {video}: FAILED {exc}", flush=True)
            traceback.print_exc()
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = [r for r in results.values() if "error" not in r]
    games = [g for r in ok for g in r["games"]]
    if games and ok:
        loc = sum(g["localizable"] for g in games)
        print("\n=== CORPUS PREP TALLY ===")
        print(f"videos prepared : {len(ok)} (failed {len(results) - len(ok)})")
        print(f"games (clean)   : {len(games)}   games/video: {len(games) / len(ok):.2f}")
        print(f"LOCALIZABLE     : {loc}/{len(games)} = {loc / len(games):.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
