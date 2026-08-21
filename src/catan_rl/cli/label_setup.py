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
  - S = submit tagging the pick "close call" (only when both picks are made).
        S is the pre-existing submit key and keeps the CONSERVATIVE tag: only
        ``clear`` picks face D4's >=70% top-1 bar, so reflex must not assert it.
  - B = submit tagging the pick "clear BEST" (D3's second submit key).
  - K = skip current draft and jump to a fresh board.
  - U = undo last pick within the current scenario (inert once the reveal is up).
  - any key = dismiss the post-submit reveal overlay.
  - Q = quit.

Modes (spec ``setup-scorer-and-blind-reveal``):
  - ``--replay-session <id>`` re-presents that session's exact boards blind, for
    the D0 self-consistency measurement. Rows link to the originals via
    ``replay_of`` and are excluded from scorer fitting. It REFUSES
    ``--scorer-weights``: D0 is an owner-vs-owner measurement, and a reveal
    overlay mid-replay anchors the owner on the scorer.
  - ``--replay-boards N`` FOLDS N replayed boards into an otherwise normal
    session — how D0's ceiling estimate is extended past its n=20 pilot without
    spending a whole sitting on it. The folded boards come first and are
    replayed exactly as above (forced-original, linked, never graded); the rest
    of the session is fresh boards. Unlike ``--replay-session`` it DOES work
    with ``--scorer-weights``, because the reveal is suppressed board-by-board
    rather than session-wide. Add ``--replay-session <id>`` to name the source;
    without it the most recent ENDED session holding N boards of original
    labels is chosen.
  - ``--scorer-weights <path>`` loads a fitted scorer; after each SUBMIT an
    overlay shows its top-1 (plus top-3 dimmed) beside your pick.
  - ``--no-reveal`` is the D3 anchoring CONTROL: the scorer is not shown and the
    rows carry no scorer fields at all. At least 20% of fresh exam picks must
    come from these sessions, and the D4 gate REFUSES to pass below that. It
    still REQUIRES ``--scorer-weights``: the version stamp is what makes a
    control pick countable, and a control session run without one contributes
    nothing.

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
from catan_rl.labeling.store import REVEAL_MODE_NO_REVEAL, REVEAL_MODE_REVEAL
from catan_rl.labeling.ui import LabelingUI
from catan_rl.setup_phase.scorer import load_weights


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
        "--replay-session",
        type=str,
        default=None,
        help="Re-present the boards of a past session id (D0 self-consistency replay).",
    )
    parser.add_argument(
        "--replay-boards",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Fold N replayed boards into an otherwise normal session (D0 "
            "ceiling estimate, extended). Source = --replay-session if given, "
            "else the most recent ended session with N boards of original labels."
        ),
    )
    parser.add_argument(
        "--scorer-weights",
        type=Path,
        default=None,
        help="Fitted scorer artifact to reveal AFTER each submit (omit for no scorer).",
    )
    parser.add_argument(
        "--no-reveal",
        action="store_true",
        help=(
            "D3 anchoring control: never show the scorer; write no scorer row "
            "fields. Still requires --scorer-weights (the manifest's version "
            # ``%%`` because argparse %-formats help strings: a bare "20%" makes
            # --help itself raise (pre-existing), which is how --replay-boards
            # would have stayed undiscoverable.
            "stamp is what makes these picks count toward the 20%% bar)."
        ),
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

    if args.no_reveal and args.scorer_weights is None:
        # Without a scorer there is no version to stamp on the manifest, and
        # ``gate.fresh_exam_picks`` drops every pick whose ``scorer_version``
        # cannot be resolved. A control session run this way would produce a
        # full sitting of labels that contribute 0% toward D3's 20% bar, with
        # no error anywhere — so refuse up front instead.
        parser.error(
            "--no-reveal requires --scorer-weights: the session manifest must "
            "stamp the scorer version live at label time or the control picks "
            "are invisible to the D4 gate."
        )
    if args.replay_boards < 0:
        parser.error("--replay-boards must be >= 0")
    exclusive_replay = args.replay_session is not None and args.replay_boards == 0
    if exclusive_replay and args.scorer_weights is not None:
        # D0's replay measures the owner against THEMSELVES — it is the
        # labeler-noise ceiling every later bar is read against. A scorer
        # overlay after each submit anchors the owner mid-replay, which is the
        # exact contamination D3's ``--no-reveal`` control exists to detect, and
        # it would corrupt the one number nothing downstream can be recalibrated
        # without. In the EXCLUSIVE mode every board is part of that
        # measurement, so the two flags are mutually exclusive, not merely
        # inadvisable. A FOLD (--replay-boards) is the sanctioned way to have
        # both: there the reveal is suppressed board-by-board, so the replayed
        # boards stay owner-vs-owner while the fresh ones — which are not part
        # of the measurement — reveal normally.
        parser.error(
            "--replay-session cannot be combined with --scorer-weights: the D0 "
            "self-consistency replay must be owner-vs-owner, and a post-submit "
            "reveal anchors the owner on the scorer mid-measurement. To do both "
            "in one sitting, fold a few boards in with --replay-boards N instead."
        )
    scorer = load_weights(args.scorer_weights) if args.scorer_weights else None
    reveal_mode = REVEAL_MODE_NO_REVEAL if args.no_reveal else REVEAL_MODE_REVEAL

    session = LabelingSession(
        data_dir=args.data_dir,
        labeler_id=args.labeler_id,
        session_seed=args.seed,
        replay_of_session=args.replay_session,
        replay_boards=args.replay_boards,
        reveal_mode=reveal_mode,
        scorer_version=None if scorer is None else scorer.version,
    )
    session.start()
    print(
        f"[label_setup] Session {session.session_id[:8]}… ready. "
        f"Total in dataset: {session.total_scenarios_in_dataset()}. "
        f"Labels → {session.scenarios_path}",
        flush=True,
    )
    if args.replay_boards > 0:
        # The source may have been auto-chosen, so name the resolved id: it is
        # the only place the owner can see WHICH sitting they are being measured
        # against before the boards start coming.
        print(
            f"[label_setup] Folding {args.replay_boards} replayed board(s) from "
            f"session {str(session.replay_of_session)[:8]}… first (never graded), "
            f"then fresh boards.",
            flush=True,
        )

    ui = LabelingUI(
        session=session,
        screen_size=(args.screen_width, args.screen_height),
        # The control arm is enforced at the UI too, not only in the row writer:
        # a scorer the renderer cannot see is a scorer that cannot anchor.
        scorer=None if args.no_reveal else scorer,
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
