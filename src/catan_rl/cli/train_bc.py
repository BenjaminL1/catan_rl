"""CLI driver for :func:`catan_rl.bc.train.train_bc`.

Loads ``configs/bc.yaml`` for defaults; CLI flags override::

    python scripts/train_bc.py --data data/bc/v1 --out runs/bc/v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# Used for argparse defaults pointing at ``data/`` / ``configs/`` /
# ``runs/``. Computed from this file's location:
# ``src/catan_rl/cli/train_bc.py`` → ``parents[3]`` = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

from catan_rl.bc.train import train_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BC anchor policy.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "bc" / "v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "bc" / "v1")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "bc.yaml",
        help="YAML config — CLI flags override these values.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-every-steps", type=int, default=None)
    parser.add_argument("--val-pct", type=float, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=None)
    parser.add_argument("--belief-weight", type=float, default=None)
    parser.add_argument("--aug-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Rows per decompressed shard chunk (loader RAM knob; smaller ⇒ lower peak).",
    )
    parser.add_argument(
        "--max-cached-chunks",
        type=int,
        default=None,
        help="LRU size in chunks (default 1; keep 1 with the chunk-grouped sampler).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap optimizer steps (smoke guard); default None trains the full budget.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    loader_cfg = config.get("loader", {})
    optim_cfg = config.get("optimizer", {})
    sched_cfg = config.get("schedule", {})
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})

    kwargs: dict[str, Any] = {}

    def _set(key, cli_val, fallback):
        kwargs[key] = cli_val if cli_val is not None else fallback

    _set("max_epochs", args.max_epochs, train_cfg.get("max_epochs", 10))
    _set("batch_size", args.batch_size, loader_cfg.get("batch_size", 1024))
    _set("val_every_steps", args.val_every_steps, train_cfg.get("val_every_steps", 500))
    _set("val_pct", args.val_pct, loader_cfg.get("val_split_pct", 0.10))
    _set("peak_lr", args.peak_lr, optim_cfg.get("lr", 3e-4))
    _set("warmup_steps", args.warmup_steps, sched_cfg.get("warmup_steps", 500))
    _set("patience", args.patience, train_cfg.get("patience", 3))
    _set("value_weight", args.value_weight, loss_cfg.get("value_weight", 0.10))
    _set("belief_weight", args.belief_weight, loss_cfg.get("belief_weight", 0.05))
    _set("aug_prob", args.aug_prob, loader_cfg.get("aug_prob", 0.5))
    _set("chunk_rows", args.chunk_rows, loader_cfg.get("chunk_rows", 200_000))
    _set("max_cached_chunks", args.max_cached_chunks, loader_cfg.get("max_cached_chunks", 1))
    kwargs.update(
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    summary = train_bc(data_dir=args.data, out_dir=args.out, **kwargs)
    print(
        f"[train_bc] done in {summary['wall_clock_seconds']:.1f}s; "
        f"best_val_nll={summary['best_val_nll']:.4f}@{summary['best_step']:,} "
        f"early_stopped={summary['early_stopped']}"
    )


if __name__ == "__main__":
    main()
