#!/usr/bin/env python3
"""One-shot: transplant a legacy v2 checkpoint into the pointer-arch (D5).

Usage::

    python scripts/migrate_pointer_arch.py --in runs/anchors/v11_cand_u724.pt \
        --out /tmp/v11_pointer_arch.pt

Loads a checkpoint payload (``policy_state_dict``), transplants the tile-encoder
and GNN verbatim, zero-pads the new player-encoder / fusion input columns, and
fresh-initialises the pointer readouts + aux head; the optimizer state is
dropped (a transplant restarts the optimizer). Prints the per-block disposition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from catan_rl.checkpoint.pointer_arch_migration import migrate_checkpoint_payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return 1
    payload = torch.load(args.in_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print(f"error: input is not a payload dict; got {type(payload).__name__}", file=sys.stderr)
        return 1

    upgraded, report = migrate_checkpoint_payload(payload)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(upgraded, args.out_path)
    print(
        f"transplanted={len(report['transplanted'])} "
        f"zero_padded={len(report['zero_padded'])} "
        f"fresh_init={len(report['fresh_init'])}; wrote {args.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
