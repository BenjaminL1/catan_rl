"""CLI driver for :func:`catan_rl.bc.dataset.generate_dataset`.

Default values reflect ``configs/bc.yaml``; CLI flags override the
config. Common invocation::

    python scripts/generate_bc_dataset.py --out data/bc/v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

# REPO_ROOT used for default ``--out`` path; computed from this
# file's location: ``src/catan_rl/cli/...`` → ``parents[3]``.
REPO_ROOT = Path(__file__).resolve().parents[3]

from catan_rl.bc.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the BC training dataset.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "bc" / "v1")
    parser.add_argument("--n-games", type=int, default=30_000)
    parser.add_argument("--perturb-pct", type=float, default=0.30)
    parser.add_argument("--epsilon-greedy-share", type=float, default=0.50)
    parser.add_argument("--shard-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--discount", type=float, default=0.998)
    parser.add_argument("--include-forced", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    manifest = generate_dataset(
        out_dir=args.out,
        n_games=args.n_games,
        perturb_pct=args.perturb_pct,
        epsilon_greedy_share_of_perturbed=args.epsilon_greedy_share,
        shard_size=args.shard_size,
        seed=args.seed,
        max_turns=args.max_turns,
        discount=args.discount,
        include_forced=args.include_forced,
        progress_every=args.progress_every,
    )
    print(f"[generate_bc_dataset] wrote {args.out}/manifest.json")
    print(
        f"  n_games={manifest['n_games']} "
        f"perturbations={manifest['perturbation_counts']} "
        f"shards={len(manifest['shards'])}"
    )
    print(
        f"  total_decisions_post_filter={manifest['total_decisions_post_filter']:,}, "
        f"forced_drop_pct={manifest['forced_move_drop_pct']:.3f}"
    )
    print(f"  wall_clock_seconds={manifest['wall_clock_seconds']:.1f}")


if __name__ == "__main__":
    main()
