#!/usr/bin/env python3
"""mine_phantom — CLI for the ThePhantom human-data video-parsing pipeline.

Parses ThePhantom's YouTube 1v1 Colonist.io games into a dataset of human
openings + outcomes (see ``docs/plans/human_data_pipeline.md``). Opponent
strength is read from the **committed strength manifest**
(``data/human/strength_manifest.json``) — the source of truth built by
``scripts/build_strength_manifest.py`` — never a rank-OCR guess here.

Subcommands:

  * ``ingest`` — the Stage-1 acquisition slice: download one video's 1080p
    stream, two-pass sample it into in-memory frames (``ingest.py``), report the
    frame count, then delete the download. A runnable smoke for the ingest path.

Downstream stages (``logparse`` / ``segment`` / ``board_cv`` / ``openings`` /
``validate`` / ``batch``) attach further subcommands as they land. CPU-only; no
``gui/`` or training-path import (build brief §6).

Usage:
    PYTHONPATH=src python3 scripts/mine_phantom.py ingest 9Sm86ml04aI --duration 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.ingest import (
    build_sampling_schedule,
    ingest_video,
    schedule_ocr_eta_s,
)

MANIFEST = REPO / "data" / "human" / "strength_manifest.json"


def _manifest_entry(video_id: str) -> dict[str, object] | None:
    """Return the strength-manifest row for ``video_id`` (or ``None``)."""
    if not MANIFEST.exists():
        return None
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in payload.get("videos", []):
        if row.get("video_id") == video_id:
            return dict(row)
    return None


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Download → two-pass sample → count frames → delete (ingest smoke)."""
    entry = _manifest_entry(args.video_id)
    strength = entry.get("strength") if entry else "MISSING-FROM-MANIFEST"
    print(f"[ingest] {args.video_id} manifest strength={strength}")
    if entry is None:
        print(
            "[ingest] WARNING: video not in strength_manifest.json — it will produce "
            "no scoreboard/seed record downstream (segment.py gates on the manifest)."
        )

    # ETA (brief §5.10): fps x 0.58s x n_videos / n_procs, with fps derived
    # honestly from this video's two-pass schedule (not the naive 1 fps spike).
    schedule = build_sampling_schedule(args.duration, sparse_interval_s=args.sparse_interval)
    fps = len(schedule) / args.duration
    eta_s = schedule_ocr_eta_s(
        schedule,
        args.duration,
        n_videos=args.n_videos,
        n_procs=args.n_procs,
    )
    print(
        f"[ingest] OCR ETA: {len(schedule)} frames/video, fps={fps:.4f} "
        f"x 0.58s x n_videos={args.n_videos:g} / n_procs={args.n_procs} "
        f"= {eta_s:.0f}s ({eta_s / 3600.0:.2f}h)"
    )

    n = 0
    sparse = dense = 0
    for frame in ingest_video(
        args.video_id,
        duration_s=args.duration,
        sparse_interval_s=args.sparse_interval,
    ):
        n += 1
        if frame.pass_name == "dense":
            dense += 1
        else:
            sparse += 1
    print(f"[ingest] decoded {n} frames (sparse={sparse}, dense={dense}); download deleted")
    return 0 if n > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mine_phantom", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="download + two-pass sample one video (smoke)")
    ing.add_argument("video_id", help="YouTube video id (11 chars)")
    ing.add_argument(
        "--duration",
        type=float,
        required=True,
        help="video duration in seconds (bounds the sparse pass)",
    )
    ing.add_argument(
        "--sparse-interval",
        dest="sparse_interval",
        type=float,
        default=4.0,
        help="sparse-pass cadence in seconds (default 4.0)",
    )
    ing.add_argument(
        "--n-videos",
        dest="n_videos",
        type=float,
        default=1.0,
        help="corpus size to project the OCR ETA over (brief §5.10; default 1)",
    )
    ing.add_argument(
        "--n-procs",
        dest="n_procs",
        type=int,
        default=1,
        help="perf cores OCR fans out across in the ETA (brief §5.10; default 1)",
    )
    ing.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
