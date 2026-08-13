#!/usr/bin/env python
"""Thin CLI for the records→labels adapter (spec D3).

Harvests the owner's setup decisions out of ``scripts/play_vs_model.py`` replay
records and appends them to the SAME JSONL label store the labeling tool writes,
marked ``source="game"``.

All logic lives in :mod:`catan_rl.labeling.from_record`; this file only parses
arguments, per the repo's thin-script convention.

Usage::

    python scripts/export_game_labels.py \\
        --replay-dir runs/human_playtest/replays \\
        --out data/labels/setup/v1/scenarios.jsonl \\
        --labeler-id ben

Re-running is safe: ``scenario_id`` is a deterministic uuid5 of the game plus
draft position, so already-exported decisions are skipped rather than
duplicated. Records that cannot yield trustworthy labels (synthesized setup,
partial game, no human seat, an auto-played human seat) are REPORTED and
skipped, never written — as are files that fail to load at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from catan_rl.labeling.from_record import export_replay_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("runs/human_playtest/replays"),
        help="Directory of Replay JSON files written by play_vs_model.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/labels/setup/v1/scenarios.jsonl"),
        help="JSONL label store to append to (the tool's store; same schema).",
    )
    parser.add_argument(
        "--labeler-id",
        type=str,
        required=True,
        help="Identity recorded on every emitted row.",
    )
    args = parser.parse_args(argv)

    written, skipped = export_replay_dir(args.replay_dir, args.out, labeler_id=args.labeler_id)
    for path, reason in skipped:
        print(f"SKIP {path.name}: {reason}", file=sys.stderr)
    print(f"appended {written} label row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
