"""CLI entry point for the interactive setup-labeling tool.

Usage::

    # First time / fresh dataset
    python scripts/label_setup.py

    # With a custom data dir and labeler id
    LABELER_ID=ben python scripts/label_setup.py --data-dir data/labels/setup/v1

    # Deterministic seed (for testing — normally omit)
    python scripts/label_setup.py --seed 42

Controls inside the app:
  - Click a green vertex to place a settlement.
  - Click a blue edge to place a road.
  - S = submit (only when both picks are made).
  - K = skip current draft and jump to a fresh board.
  - U = undo last pick within the current scenario.
  - Q = quit.
  - B / O / H / R / X = set archetype (Balanced / OWS / OWS-hybrid /
    Road-builder / Other).

Data:
  - Labels appended to ``<data-dir>/scenarios.jsonl``.
  - Per-session manifest in ``<data-dir>/sessions/<uuid>/manifest.json``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# REPO_ROOT used for the default ``--out`` argparse path; computed
# from this file's location: ``src/catan_rl/cli/...`` →
# ``parents[3]`` = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.ui import LabelingUI


def main() -> int:
    parser = argparse.ArgumentParser(description="Catan setup-labeling tool")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "setup" / "v1",
        help="Directory for scenarios.jsonl + sessions/ subdir.",
    )
    parser.add_argument(
        "--labeler-id",
        type=str,
        default=os.environ.get("LABELER_ID", "unknown"),
        help="Identity recorded per row (defaults to $LABELER_ID, then 'unknown').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for the session (omit for non-deterministic per-session boards).",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1100,
        help="Window width in pixels.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=900,
        help="Window height in pixels.",
    )
    args = parser.parse_args()

    session = LabelingSession(
        data_dir=args.data_dir,
        labeler_id=args.labeler_id,
        session_seed=args.seed,
    )
    session.start()
    print(
        f"[label_setup] Session {session.session_id[:8]}… ready. "
        f"Total in dataset: {session.total_scenarios_in_dataset()}. "
        f"Labels → {session.scenarios_path}",
        flush=True,
    )

    ui = LabelingUI(
        session=session,
        screen_size=(args.screen_width, args.screen_height),
    )
    try:
        ui.run()
    finally:
        if not session._quit:
            session.quit()
        print(
            f"[label_setup] Session ended. "
            f"You labeled {session.scenarios_completed} this session. "
            f"Dataset now contains {session.total_scenarios_in_dataset()} labels.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
