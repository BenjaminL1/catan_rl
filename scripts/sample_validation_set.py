#!/usr/bin/env python3
"""Draw a stratified hand-check validation set from the strength manifest.

Over-samples the DECISION-CRITICAL cases (where a wrong label costs the most) rather
than a flat random slice:

  - excluded          : ALL of them (each is a "drop this game" call; a false-exclude
                        loses a good game). Usually few.
  - high @ boundary   : ranked highs with the largest rank number (closest to the <=200
                        cutoff) — the reads most likely to be wrong-side-of-the-line.
  - high ranked spread: a random draw across the rank distribution (both UI eras).
  - unknown           : a random draw — to eyeball whether a Global leaderboard was
                        actually missed (recall check).
  - tournament        : a random draw — confirm the keyword really is a tournament.
  - random            : an unbiased draw across everything.

Emits data/human/strength_manifest_validation.md — one row per video with a
timestamp-anchored check link (jumps to the exact frame) + the saved-frame path +
the manifest's claim + a blank verdict box for you to mark.

Usage: python scripts/sample_validation_set.py [--seed 0] [--per 8]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
FRAMES = REPO / "data" / "human" / "frames"
OUT_MD = REPO / "data" / "human" / "strength_manifest_validation.md"


def _check_link(v: dict[str, Any]) -> str:
    vid = v["video_id"]
    ev = v.get("evidence", {})
    base = f"https://www.youtube.com/watch?v={vid}"
    if v["source"] == "tournament":
        t = ev.get("bracket_frame_s")
        return f"[title/bracket]({base}&t={int(t)}s)" if t is not None else f"[open]({base})"
    if v["source"] == "ranked_rank":
        t = ev.get("frame_s")
        anchor = f"{base}&t={int(t)}s" if t is not None else base
        return f"[leaderboard]({anchor})"
    return f"[open — scan first/last ~90s]({base})"


def _claim(v: dict[str, Any]) -> str:
    ev = v.get("evidence", {})
    if v["source"] == "tournament":
        return f"high · tournament `{ev.get('keyword', '')}`"
    if v["source"] == "ranked_rank":
        return f"{v['strength']} · world rank **#{ev.get('rank')}**"
    return f"{v['strength']} · {ev.get('reason', 'no signal')}"


def _frame(v: dict[str, Any]) -> str:
    p = FRAMES / f"{v['video_id']}_rank.png"
    return f"`{p.relative_to(REPO)}`" if p.exists() else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=8, help="draw size per random bucket")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    videos = json.loads(MANIFEST.read_text())["videos"]
    by: dict[str, list[dict[str, Any]]] = {"high": [], "excluded": [], "unknown": []}
    for v in videos:
        by.setdefault(v["strength"], []).append(v)
    ranked_high = [v for v in by.get("high", []) if v["source"] == "ranked_rank"]
    tourney_high = [v for v in by.get("high", []) if v["source"] == "tournament"]

    def draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rng.sample(pool, min(k, len(pool)))

    buckets: list[tuple[str, list[dict[str, Any]]]] = [
        ("excluded (verify all — false-exclude loses a game)", by.get("excluded", [])[:20]),
        (
            "high @ boundary (largest rank, closest to cutoff)",
            sorted(ranked_high, key=lambda v: -v["evidence"].get("rank", 0))[:5],
        ),
        ("high ranked (random spread)", draw(ranked_high, args.per + 2)),
        (
            "unknown (recall check — was a leaderboard missed?)",
            draw(by.get("unknown", []), args.per),
        ),
        ("tournament (confirm real tournament)", draw(tourney_high, 5)),
        ("random (unbiased)", draw(videos, 5)),
    ]

    seen: set[str] = set()
    rows: list[tuple[str, dict[str, Any]]] = []
    for label, pool in buckets:
        for v in pool:
            if v["video_id"] not in seen:
                seen.add(v["video_id"])
                rows.append((label, v))

    lines = [
        "# Strength manifest — hand-check validation set",
        "",
        f"Stratified sample of **{len(rows)}** videos (seed={args.seed}), weighted toward the "
        "decisions that cost the most if wrong. Click **Check** to jump to the exact frame the "
        "label came from, or open the saved frame. Mark each ✓ (agree) or ✗ (wrong) at the end.",
        "",
        "For `unknown` rows there is no frame — open the video and scan the first/last ~90s: "
        "if you DO see a Global leaderboard with his rank, that's a missed `high` "
        "(recall gap, not a wrong label).",
        "",
        "| # | Bucket (why sampled) | Manifest claim | Video | Check | Frame | ✓/✗ |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (label, v) in enumerate(rows, 1):
        title = v["title"].replace("|", "\\|")[:48]
        lines.append(
            f"| {i} | {label} | {_claim(v)} | {title} | {_check_link(v)} | {_frame(v)} |  |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}  ({len(rows)} videos to hand-check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
