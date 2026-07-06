#!/usr/bin/env python3
"""gold_gate — the 30-game blind-labeling EXAM apparatus for the human-data pipeline.

Instruments only (labeling happens in a later phase). Three subcommands over
:mod:`catan_rl.human_data.gold_gate`:

  * ``prepare`` — select ~30 gold games (accepted games from the harvested corpus)
    and write a BLIND-LABELING packet per game under ``<gold-dir>/<game_id>/``: the
    post-setup full frame, 1-2 mid-setup frames, the setup + terminal log-crop PNGs,
    the static engine-id reference grid, and a blank label template. The pipeline's
    own record (the answer key ``score`` grades against) is written OUTSIDE the
    packet, under ``<gold-dir>/_answers/``. The packet contains nothing derived from
    the pipeline's parse — blindness is the point.

  * ``score`` — grade completed label files against the pipeline records field-by-
    field, report per-field accuracy vs the pre-registered bars (board ≥98% of
    hexes, openings ≥95% of placements, winner ~100%, orientation flips 0) with
    Wilson CIs + a verdict line, written to ``docs/plans/gold_gate_report.md``.

  * ``grid`` — render the standalone engine-id reference grid to a PNG.

CPU-only; no ``gui/`` or training-path import. The ``prepare`` frame acquisition
re-ingests each video (a real-run CV path: yt-dlp + ffmpeg + easyocr).

Usage:
    PYTHONPATH=src python3 scripts/gold_gate.py grid --out /tmp/grid.png
    PYTHONPATH=src python3 scripts/gold_gate.py prepare --corpus OUT/corpus.jsonl --count 30
    PYTHONPATH=src python3 scripts/gold_gate.py score --gold-dir data/human/gold
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.gold_gate import (
    DEFAULT_GOLD_COUNT,
    GoldFrames,
    load_records,
    prepare_packets,
    render_reference_grid,
    score_gold,
    select_gold_games,
    write_score_report,
)
from catan_rl.human_data.record import GameRecord

DEFAULT_GOLD_DIR = REPO / "data" / "human" / "gold"
DEFAULT_REPORT = REPO / "docs" / "plans" / "gold_gate_report.md"


def _real_frame_provider(record: GameRecord) -> GoldFrames:  # pragma: no cover - real-run CV path
    """Re-ingest a video and route its frames to the game (the real ``prepare`` seam).

    Composes the harvest ingest + context + routing to recover the game's setup
    frames, then crops the setup and terminal log panels. This runs yt-dlp / ffmpeg
    / easyocr and is exercised only on hardware; the tested ``prepare`` core
    (:func:`~catan_rl.human_data.gold_gate.prepare_packets`) injects a stub provider.
    """
    import numpy as np

    from catan_rl.human_data import harvest
    from catan_rl.human_data.batch import VideoParseError
    from catan_rl.human_data.logparse import crop_log, ocr_log_crop, parse_log
    from catan_rl.human_data.segment import ruleset_ok, segment_games

    frames = harvest._ingest(record.video_id)
    ctx = harvest._extract_context(record.video_id, frames)
    segments = segment_games(ctx.events, list(ctx.handles))

    ordinal = 0
    seg_idx: int | None = None
    for i, seg in enumerate(segments):
        if not ruleset_ok(seg):
            continue
        ordinal += 1
        if ordinal == record.game_index:
            seg_idx = i
            break
    if seg_idx is None or seg_idx >= len(ctx.game_frames):
        raise VideoParseError(f"could not route {record.video_id!r} game {record.game_index}")
    gf = ctx.game_frames[seg_idx]
    if gf is None:
        raise VideoParseError(f"no routed frames for {record.video_id!r} game {record.game_index}")

    post = gf.post_setup_frame.frame
    mids = tuple(f.frame for f in gf.setup_frames[:2])
    setup_log = crop_log(post)

    # Terminal log crop: scan frames after the post-setup ts for the victory line.
    terminal_log = crop_log(frames[-1].frame)
    for frame in frames:
        if frame.ts < gf.post_setup_frame.ts:
            continue
        lines = ocr_log_crop(crop_log(frame.frame))
        events = parse_log(lines, ctx.handles).events
        if any(e.kind == "victory" for e in events):
            terminal_log = crop_log(frame.frame)
            break

    return GoldFrames(
        post_setup=np.ascontiguousarray(post),
        mid_setup=tuple(np.ascontiguousarray(m) for m in mids),
        setup_log_crop=np.ascontiguousarray(setup_log),
        terminal_log_crop=np.ascontiguousarray(terminal_log),
    )


def _cmd_grid(args: argparse.Namespace) -> int:
    """Render the standalone engine-id reference grid PNG."""
    out = render_reference_grid(args.out)
    print(f"[grid] wrote engine-id reference grid to {out}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Select the gold games and write a blind packet per game (+ sibling answer keys)."""
    corpus_paths: Sequence[str] = args.corpus
    records = load_records(corpus_paths)
    accepted = select_gold_games(records, args.count)
    print(
        f"[prepare] loaded {len(records)} records from {len(corpus_paths)} file(s); "
        f"{len(accepted)} accepted gold games selected (target {args.count})"
    )
    if not accepted:
        print(
            "[prepare] no accepted (passed_crosscheck) games in the corpus — nothing to "
            "prepare. Point --corpus at a harvested corpus.jsonl with accepted games."
        )
        return 1
    if len(accepted) < args.count:
        print(
            f"[prepare] WARNING: only {len(accepted)} accepted games available (< {args.count}); "
            "run the harvest driver on more 'high' videos to grow the corpus, then re-run."
        )
    packets = prepare_packets(accepted, _real_frame_provider, args.gold_dir, count=args.count)
    print(f"[prepare] wrote {len(packets)} blind packet(s) under {args.gold_dir}")
    for packet in packets:
        print(f"[prepare]   {packet}")
    answers = Path(args.gold_dir) / "_answers"
    print(f"[prepare] answer keys written under {answers} (NOT in packets)")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Grade completed labels vs the pipeline answers → gold_gate_report.md + verdict."""
    report = score_gold(
        args.gold_dir,
        labels_dir=args.labels_dir,
        answers_dir=args.answers_dir,
    )
    path = write_score_report(report, args.report)
    verdict = "READY" if report.ready else "NOT READY"
    print(
        f"[score] {report.n_games} game(s) scored "
        f"({len(report.skipped_unlabeled)} unlabeled skipped) → {path}"
    )
    print(
        f"[score] board={report.board.accuracy:.4f} openings={report.openings.accuracy:.4f} "
        f"winner={report.winner.accuracy:.4f} orientation_flips={report.orientation_flips}"
    )
    print(f"[score] VERDICT: {verdict}")
    if report.failures():
        print("[score] open failures: " + "; ".join(report.failures()))
    return 0 if report.ready else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gold_gate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    grid = sub.add_parser("grid", help="render the engine-id reference grid PNG")
    grid.add_argument(
        "--out",
        default=str(DEFAULT_GOLD_DIR / "reference_grid.png"),
        help="output PNG path (default data/human/gold/reference_grid.png)",
    )
    grid.set_defaults(func=_cmd_grid)

    prepare = sub.add_parser(
        "prepare", help="select ~30 gold games and write blind-labeling packets"
    )
    prepare.add_argument(
        "--corpus",
        nargs="+",
        required=True,
        help="one or more harvested corpus.jsonl files to select accepted gold games from",
    )
    prepare.add_argument(
        "--gold-dir",
        dest="gold_dir",
        default=str(DEFAULT_GOLD_DIR),
        help="destination for the blind packets + _answers/ (default data/human/gold)",
    )
    prepare.add_argument(
        "--count",
        type=int,
        default=DEFAULT_GOLD_COUNT,
        help=f"target number of gold games (default {DEFAULT_GOLD_COUNT})",
    )
    prepare.set_defaults(func=_cmd_prepare)

    score = sub.add_parser(
        "score", help="grade completed labels vs the pipeline records → gold_gate_report.md"
    )
    score.add_argument(
        "--gold-dir",
        dest="gold_dir",
        default=str(DEFAULT_GOLD_DIR),
        help="the gold tree written by prepare (default data/human/gold)",
    )
    score.add_argument(
        "--labels-dir",
        dest="labels_dir",
        default=None,
        help="read filled labels from <labels-dir>/<game_id>.json instead of in-packet "
        "label_template.json",
    )
    score.add_argument(
        "--answers-dir",
        dest="answers_dir",
        default=None,
        help="read answer keys from here (default <gold-dir>/_answers)",
    )
    score.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="destination for the markdown report (default docs/plans/gold_gate_report.md)",
    )
    score.set_defaults(func=_cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
