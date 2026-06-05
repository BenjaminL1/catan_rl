"""CLI for upgrading a checkpoint to the current schema version.

Usage::

    python scripts/migrate_checkpoint.py --in old.pt --out upgraded.pt

The script:

1. ``torch.load``s the input file.
2. Runs :func:`catan_rl.checkpoint.migrations.apply_migrations` against
   :data:`catan_rl.checkpoint.manager.SCHEMA_VERSION`.
3. Atomically writes the upgraded payload to ``--out`` (a tmp file
   plus ``os.replace`` — same machinery as a normal save, so a
   partial write doesn't corrupt an existing target).

If the input is already at the current schema, the script prints
``already at v<N>`` and exits 0 without writing anything (unless
``--force`` is passed, in which case it re-saves so the on-disk file
gets the canonical layout).

Exit codes:
* 0 — migration applied or already current.
* 1 — load failure, migration failure, or schema downgrade attempt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from catan_rl.checkpoint.manager import SCHEMA_VERSION, CheckpointError
from catan_rl.checkpoint.migrations import MigrationError, apply_migrations


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Path to the old checkpoint file.",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        type=Path,
        help="Path to write the upgraded checkpoint to.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-save even if the input is already at the current schema.",
    )
    return p.parse_args(argv)


def _atomic_torch_save(payload: dict, dest: Path) -> None:
    import contextlib

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    # Best-effort parent-dir fsync — see manager._fsync_dir for why.
    try:
        dir_fd = os.open(dest.parent, os.O_DIRECTORY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return 1

    try:
        raw = torch.load(args.in_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"error: torch.load failed: {e}", file=sys.stderr)
        return 1

    if not isinstance(raw, dict):
        print(
            f"error: input is not a dict; got {type(raw).__name__}",
            file=sys.stderr,
        )
        return 1

    if "schema_version" not in raw:
        print(
            "error: input missing 'schema_version' — pre-versioned "
            "checkpoint, no migration available",
            file=sys.stderr,
        )
        return 1

    current = int(raw["schema_version"])
    if current == SCHEMA_VERSION and not args.force:
        print(f"already at v{SCHEMA_VERSION}; pass --force to re-save")
        return 0

    try:
        upgraded = apply_migrations(raw, target_version=SCHEMA_VERSION)
    except MigrationError as e:
        print(f"error: migration failed: {e}", file=sys.stderr)
        return 1
    except CheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        _atomic_torch_save(upgraded, args.out_path)
    except OSError as e:
        print(f"error: write failed: {e}", file=sys.stderr)
        return 1

    print(f"migrated v{current} -> v{SCHEMA_VERSION}; wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
