"""One-shot pre-Phase-0 checkpoint migrator.

Phase 0 introduces:
  - new config keys (eval_harness_seeds, frozen_champion_path, entropy_collapse_*)
  - new arrays in CompositeRolloutBuffer (terminated/truncated split)
  - per-head entropy logging hooks

None of those changes touch the *policy* state dict, optimizer state, or value
normalizer. So the migration is a tiny config patch that lets old checkpoints
load through the new ``CatanPPO.load`` without error.

Usage::

    python scripts/migrate_checkpoint.py checkpoints/train/checkpoint_07390040.pt
    python scripts/migrate_checkpoint.py checkpoints/train/checkpoint_07390040.pt --in-place
    python scripts/migrate_checkpoint.py checkpoints/train/checkpoint_07390040.pt --output checkpoints/train/checkpoint_07390040_p0.pt

The script verifies the migrated checkpoint loads cleanly through the new
trainer before declaring success.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

# Allow `python scripts/migrate_checkpoint.py` without `pip install -e .`.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

warnings.filterwarnings("ignore", message="enable_nested_tensor.*norm_first")

import torch

# New config keys added in Phase 0. Each is set to its default if missing.
PHASE0_CONFIG_DEFAULTS: dict[str, object] = {
    "eval_harness_seeds": list(range(0, 200)),
    "eval_harness_swap_first_player": True,
    "frozen_champion_path": "checkpoints/train/checkpoint_07390040.pt",
    "entropy_collapse_threshold": 0.0005,
    "entropy_collapse_consecutive_updates": 3,
}


def migrate_checkpoint(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    in_place: bool = False,
    verify: bool = True,
) -> Path:
    """Migrate a pre-Phase-0 checkpoint and write the result.

    Args:
        src: Source checkpoint path.
        dst: Destination path. Required unless ``in_place=True``.
        in_place: If True, overwrite ``src`` (with a ``.bak`` backup created
            alongside it for safety).
        verify: If True, load the migrated checkpoint via ``CatanPPO.load`` to
            confirm the schema is acceptable before reporting success.

    Returns:
        The path to the written migrated checkpoint.
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {src_path}")

    if in_place:
        dst_path = src_path
        backup = src_path.with_suffix(src_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(src_path, backup)
            print(f"  backup: {backup}")
    else:
        if dst is None:
            raise ValueError("either --in-place or --output is required")
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  loading: {src_path}")
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise ValueError(
            f"checkpoint at {src_path} has no 'config' field; not a CatanPPO checkpoint"
        )

    config = checkpoint["config"]
    added_keys = []
    for key, default in PHASE0_CONFIG_DEFAULTS.items():
        if key not in config:
            config[key] = default
            added_keys.append(key)

    # Keep the policy state dict untouched. The Phase 0 changes are in the
    # rollout buffer (memory only) and the trainer's logging path; no model
    # weights are renamed or reshaped.
    if added_keys:
        print(f"  added keys: {added_keys}")
    else:
        print("  no config keys missing — checkpoint already Phase-0-compatible")

    print(f"  writing: {dst_path}")
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(dst_path)

    if verify:
        print("  verifying with CatanPPO.load …")
        from catan_rl.algorithms.ppo.trainer import CatanPPO

        trainer = CatanPPO.load(str(dst_path))
        print(f"  OK — global_step={trainer.global_step:,}")

    return dst_path


def main() -> int:
    p = argparse.ArgumentParser(description="Migrate pre-Phase-0 CatanPPO checkpoints")
    p.add_argument("src", type=str, help="Source checkpoint path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--output", "-o", type=str, default=None, help="Destination path for migrated checkpoint"
    )
    g.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the source file (a .bak copy is kept alongside)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-migration CatanPPO.load verification",
    )
    args = p.parse_args()

    out = migrate_checkpoint(
        args.src, args.output, in_place=args.in_place, verify=not args.no_verify
    )
    print(f"migrated → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
