"""CLI for recording one 1v1 Catan game to a JSON replay file.

Usage::

    python scripts/record_game.py \\
        --player-a heuristic \\
        --player-b heuristic \\
        --seed 42 \\
        --out runs/replays/sample.json

For policy seats, pass ``--ckpt-a`` / ``--ckpt-b`` pointing at a
Phase 8 checkpoint file. ``--device`` resolves the policy's compute
target (``auto`` walks cuda → mps → cpu).

CLI-level matchup validation rules:
* If ``--player-{a,b}=policy`` then ``--ckpt-{a,b}`` is REQUIRED.
* ``(policy, policy)`` raises ``NotImplementedError`` in v1 (the
  v2 env's snapshot-opponent path hasn't landed yet; revisit when
  Phase 8 ships the snapshot opponent).
* ``--out`` refuses to overwrite an existing file unless ``--force``
  is set — protects against accidental clobbers.

This script is the entry point only. The actual simulation +
recording loop is built in Recorder Phases 2a-2d (which haven't
landed yet); for now the script validates args, prints a summary,
and exits with a clear "not yet wired" error if asked to actually
record.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from catan_rl.replay.player_factory import PlayerSpec

#: Exit code used when the recorder simulation logic is not yet
#: implemented (Phases 2a-2d). 64 = "do not retry" per the POSIX
#: convention used elsewhere in this codebase (see scripts/train.py).
EXIT_RECORDER_NOT_WIRED = 64


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="record_game.py",
        description="Record one 1v1 Catan game to a JSON replay.",
    )
    p.add_argument(
        "--player-a",
        choices=["random", "heuristic", "policy"],
        required=True,
        help="Player A kind (seat 0 / 'player_a' in the JSON).",
    )
    p.add_argument(
        "--ckpt-a",
        type=Path,
        default=None,
        help="Path to player A's checkpoint (required if --player-a=policy).",
    )
    p.add_argument(
        "--player-b",
        choices=["random", "heuristic", "policy"],
        required=True,
        help="Player B kind (seat 1 / 'player_b' in the JSON).",
    )
    p.add_argument(
        "--ckpt-b",
        type=Path,
        default=None,
        help="Path to player B's checkpoint (required if --player-b=policy).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (controls StackedDice + board layout + per-actor random sampling).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the JSON replay file.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=400,
        help="Per-agent turn cap before truncation (matches v2 CatanEnv default).",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="cpu",
        help="Compute device for policy actors (random/heuristic ignore this).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists. Default refuses.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """Apply matchup-level validation. Raises with clear messages on
    misuse so the user doesn't burn cycles only to crash mid-game.

    Order matters: the v1-scope ``(policy, policy)`` rejection runs
    first so a user with both ckpt paths missing sees the actual
    blocker (the unsupported matchup) rather than a "ckpt required"
    error they'd fix only to hit the matchup error on the next try.
    """
    if args.player_a == "policy" and args.player_b == "policy":
        raise NotImplementedError(
            "(policy, policy) recording is not supported in v1. The v2 "
            "CatanEnv only hosts one policy as the agent; the opponent "
            "slot accepts 'random' or 'heuristic' only until the Phase "
            "8 snapshot-opponent path lands. Set one side to 'heuristic' "
            "or 'random' for now."
        )
    if args.player_a == "policy" and args.ckpt_a is None:
        raise SystemExit("error: --player-a=policy requires --ckpt-a path/to/ckpt.pt")
    if args.player_b == "policy" and args.ckpt_b is None:
        raise SystemExit("error: --player-b=policy requires --ckpt-b path/to/ckpt.pt")
    out_path: Path = args.out.expanduser().resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"error: --out already exists at {out_path}; pass --force to overwrite")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("catan_rl.replay")

    _validate_args(args)

    # Build player specs so the CLI validates the policy ckpt paths
    # exist (raises FileNotFoundError from build_actor) — but we don't
    # actually run the recorder loop yet (Phases 2a-2d).
    spec_a = PlayerSpec(
        kind=args.player_a,
        ckpt_path=str(args.ckpt_a) if args.ckpt_a is not None else None,
    )
    spec_b = PlayerSpec(
        kind=args.player_b,
        ckpt_path=str(args.ckpt_b) if args.ckpt_b is not None else None,
    )

    log.info(
        "player_a=%s ckpt_a=%s player_b=%s ckpt_b=%s seed=%d max_turns=%d device=%s out=%s",
        spec_a.kind,
        spec_a.ckpt_path,
        spec_b.kind,
        spec_b.ckpt_path,
        args.seed,
        args.max_turns,
        args.device,
        args.out,
    )

    log.error(
        "recorder simulation loop is Phases 2a-2d of the replay system "
        "build-out; this script currently validates args + player specs "
        "only. Wait for the recorder loop to land before invoking with "
        "a real --out path."
    )
    return EXIT_RECORDER_NOT_WIRED


if __name__ == "__main__":
    raise SystemExit(main())
