"""Console-script body for the Torevan conformance recorder.

Wired onto PATH as ``catan-rl-conformance`` via
``pyproject.toml [project.scripts]``. The thin ``scripts/record_conformance.py``
shim delegates here so ``python scripts/record_conformance.py`` keeps
working.

See :mod:`catan_rl.conformance.recorder` for the replay-log schema.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from catan_rl.conformance.recorder import record_game, save_log


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="catan-rl-conformance",
        description="Record reference games to Torevan conformance replay-logs.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more integer seeds; one replay-log is written per seed.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write game-seed-N.json files into.",
    )
    p.add_argument(
        "--max-main-turns",
        type=int,
        default=200,
        help="Per-game main-turn cap (the random policy rarely reaches a win).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing replay-log files. Default refuses.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("catan_rl.conformance")

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        out_path = out_dir / f"game-seed-{seed}.json"
        if out_path.exists() and not args.force:
            raise SystemExit(f"error: {out_path} already exists; pass --force to overwrite")
        replay = record_game(seed, max_main_turns=args.max_main_turns)
        save_log(replay, out_path)
        log.info("wrote %s: %d steps, seed=%d", out_path.name, len(replay["steps"]), seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
