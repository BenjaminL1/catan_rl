#!/usr/bin/env python3
"""Tier-5 yield re-run after the setup-parse fix (spike report Step 5).

For each high-tier video: run the SAME harvest ingest + (now fail-closed) frame
routing the real pipeline uses, and record per GAME:

  * ``routed``            — the window got frames at all;
  * ``clean_opening``     — the router produced an honest 8-pieces-down post-setup
                            frame (i.e. NOT the ``post_setup_frame_unresolved`` typed
                            reject that used to be a silent END-GAME fallback);
  * ``grants_readable``   — BOTH players' setup-grant glyphs were read (the glyph
                            anchor is the only joint-flip defence, so a game missing
                            either grant is a fail-closed ``glyph_unreadable`` reject
                            no matter how good the frame is);
  * ``localizable``       — clean_opening AND grants_readable, i.e. everything the
                            pipeline needs before the VLM's perception step. This is
                            the honest ACCEPT-eligibility rate: on the games measured
                            by hand, localization+snap+validate accepted whenever both
                            held, and the remaining failure modes are perception-side.

Writes ``runs/human/tier5_yield.json``. CPU-frugal: the ``--workers`` cap exists
because a training run may own the machine (keep it <= 3 while one is live).

Usage::

    PYTHONPATH=src python3 scripts/dev/tier5_yield_rerun.py --videos 8 --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data import harvest
from catan_rl.human_data.segment import segment_games

MANIFEST = REPO / "data/human/strength_manifest.json"
OUT = REPO / "runs/human/tier5_yield.json"


def high_tier_videos() -> list[str]:
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    vids = d if isinstance(d, list) else d.get("videos", d)
    items = vids.items() if isinstance(vids, dict) else [(v.get("video_id"), v) for v in vids]
    return [k for k, v in items if isinstance(v, dict) and v.get("strength") == "high"]


def probe_video(video: str) -> dict:
    """Ingest + route one video; report per-game routing/grant readability."""
    # Two-pass ingest: the sparse pass finds the grant lines, a 1 s dense pass
    # re-samples those windows so the >=2-frame grant consensus is not starved.
    frames, per_frame_lines = harvest._ingest_two_pass(video, download_gate=None, work_dir=None)
    ctx = harvest._extract_context(video, frames, per_frame_lines)
    segments = segment_games(ctx.events, list(ctx.handles))

    games: list[dict] = []
    game_index = 0
    for seg_idx, segment in enumerate(segments):
        if not harvest.ruleset_ok(segment):
            continue  # not a 1v1 game window — not a game, not a denominator row
        game_index += 1
        routed = seg_idx < len(ctx.game_frames) and ctx.game_frames[seg_idx] is not None
        clean_opening = False
        grants_readable = False
        if routed:
            gf = ctx.game_frames[seg_idx]
            assert gf is not None
            clean_opening = gf.post_setup_frame is not None
            grants = {
                h: harvest._consensus_grant(h, gf.grant_frames, ctx.handles) for h in ctx.handles
            }
            grants_readable = all(g is not None for g in grants.values())
        games.append(
            {
                "game_index": game_index,
                "routed": routed,
                "clean_opening": clean_opening,
                "grants_readable": grants_readable,
                "localizable": bool(clean_opening and grants_readable),
                "winner": segment.winner,
            }
        )
    return {"video": video, "games": games, "n_games": len(games)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=8, help="how many high-tier videos to probe")
    ap.add_argument("--skip", nargs="*", default=[], help="video ids already measured")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    todo = [v for v in high_tier_videos() if v not in results and v not in args.skip]
    todo = todo[: args.videos]
    print(f"[yield] probing {len(todo)} videos: {todo}", flush=True)

    for i, video in enumerate(todo, 1):
        try:
            r = probe_video(video)
            results[video] = r
            n = r["n_games"]
            loc = sum(g["localizable"] for g in r["games"])
            clean = sum(g["clean_opening"] for g in r["games"])
            print(
                f"[yield] {i}/{len(todo)} {video}: games={n} clean_opening={clean} "
                f"localizable={loc}",
                flush=True,
            )
        except Exception as exc:  # a per-video failure must not kill the sweep
            results[video] = {"video": video, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[yield] {i}/{len(todo)} {video}: FAILED {exc}", flush=True)
            traceback.print_exc()
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = [r for r in results.values() if "error" not in r]
    games = [g for r in ok for g in r["games"]]
    if games:
        print("\n=== TIER-5 YIELD (post setup-parse fix) ===")
        print(f"videos probed      : {len(ok)}  (failed: {len(results) - len(ok)})")
        print(f"games detected     : {len(games)}")
        print(f"games/video        : {len(games) / len(ok):.2f}")
        print(f"clean_opening      : {sum(g['clean_opening'] for g in games)}/{len(games)}")
        print(f"grants_readable    : {sum(g['grants_readable'] for g in games)}/{len(games)}")
        print(
            f"LOCALIZABLE (yield): {sum(g['localizable'] for g in games)}/{len(games)} "
            f"= {sum(g['localizable'] for g in games) / len(games):.3f}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
