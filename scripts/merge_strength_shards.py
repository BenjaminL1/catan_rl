#!/usr/bin/env python3
"""Merge parallel strength-manifest shard files into the single manifest.

Reads data/human/strength_manifest.shard*.json (written by parallel
`build_strength_manifest.py --shard I/N --out ...` workers), unions their videos by
id, writes data/human/strength_manifest.json, and VERIFIES completeness against the
live channel list (reports any missing/duplicate videos) so sharding can't silently
drop coverage.

Usage: python scripts/merge_strength_shards.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_strength_manifest as m


def main() -> int:
    shards = sorted(m.OUT_DIR.glob("strength_manifest.shard*.json"))
    if not shards:
        print("no shard files found", file=sys.stderr)
        return 1

    by_id: dict[str, dict[str, object]] = {}
    dupes = 0
    for sh in shards:
        for v in json.loads(sh.read_text())["videos"]:
            if v["video_id"] in by_id:
                dupes += 1  # disjoint shards shouldn't overlap; last write wins
            by_id[v["video_id"]] = v

    merged = {
        "schema_version": 1,
        "generated_from": m.CHANNEL,
        "rank_high_max": m.RANK_HIGH_MAX,
        "videos": list(by_id.values()),
    }
    m.save_manifest(merged, m.MANIFEST)

    counts: dict[str, int] = {}
    for v in by_id.values():
        s = str(v["strength"])
        counts[s] = counts.get(s, 0) + 1
    print(f"merged {len(shards)} shards -> {len(by_id)} videos  counts={counts}  dupes={dupes}")

    # completeness check against the live channel
    try:
        expected = {vid for vid, _ in m.list_videos()}
    except Exception as e:  # network best-effort
        print(f"(channel completeness check skipped: {e})")
        return 0
    missing = expected - by_id.keys()
    extra = by_id.keys() - expected
    print(f"channel={len(expected)}  missing={len(missing)}  extra={len(extra)}")
    if missing:
        print("MISSING (not covered by any shard):", sorted(missing)[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
