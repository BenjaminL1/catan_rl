#!/usr/bin/env python3
"""Render data/human/strength_manifest.json -> a human-checkable Markdown table.

Every row gets a timestamp-anchored "check" link that opens YouTube at the exact frame
the label was read from (the bracket for tournaments, the leaderboard frame for ranked),
so any verdict is auditable in one click.

Usage: python scripts/render_strength_manifest_md.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
OUT_MD = REPO / "data" / "human" / "strength_manifest.md"


def _check_link(v: dict[str, Any]) -> str:
    vid = v["video_id"]
    ev = v.get("evidence", {})
    base = f"https://www.youtube.com/watch?v={vid}"
    if v["source"] == "tournament":
        t = ev.get("bracket_frame_s")
        url = f"{base}&t={int(t)}s" if t is not None else base
        return f"[bracket/title]({url})"
    if v["source"] == "ranked_rank":
        t = ev.get("frame_s")
        url = f"{base}&t={int(t)}s" if t is not None else base
        return f"[leaderboard @rank #{ev.get('rank')}]({url})"
    return f"[watch]({base})"


def _why(v: dict[str, Any]) -> str:
    ev = v.get("evidence", {})
    if v["source"] == "tournament":
        return f"tournament (`{ev.get('keyword', '')}`)"
    if v["source"] == "ranked_rank":
        return f"world rank **#{ev.get('rank')}** ({ev.get('where', '')})"
    return str(ev.get("reason", "no signal"))


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    videos = m["videos"]
    order = {"high": 0, "excluded": 1, "unknown": 2}
    videos.sort(
        key=lambda v: (order.get(v["strength"], 9), v.get("evidence", {}).get("rank", 9999))
    )

    counts: dict[str, int] = {}
    for v in videos:
        counts[v["strength"]] = counts.get(v["strength"], 0) + 1

    lines = [
        "# ThePhantom strength manifest",
        "",
        f"Source: {m.get('generated_from', '')}  ·  high-rank threshold: world rank ≤ "
        f"{m.get('rank_high_max', 200)} (or any tournament).",
        "",
        f"**Totals:** {len(videos)} videos — "
        + ", ".join(f"`{k}` {counts[k]}" for k in sorted(counts)),
        "",
        "- `high` → scoreboard **and** seed corpus",
        "- `unknown` → seed corpus only (no verified rank found)",
        "- `excluded` → dropped (world rank > threshold)",
        "",
        "Each **check** link opens the video at the exact frame the label was read from.",
        "",
        "| Verdict | Why | Video | Check |",
        "|---|---|---|---|",
    ]
    for v in videos:
        title = v["title"].replace("|", "\\|")[:60]
        lines.append(f"| `{v['strength']}` | {_why(v)} | {title} | {_check_link(v)} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}  ({len(videos)} rows; counts={counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
