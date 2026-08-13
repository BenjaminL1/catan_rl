#!/usr/bin/env python
"""Thin CLI for the label store → BC shard converter (spec D4, plan §E).

All logic lives in :mod:`catan_rl.labeling.to_shard`; this file only parses
arguments, per the repo's thin-script convention.

Usage::

    python scripts/convert_labels_to_bc_shard.py \\
        --labels data/labels/setup/v1/scenarios.jsonl \\
        --out data/bc/human_openings/v1

The output directory loads directly through
:class:`catan_rl.bc.loader.BcDataset`. The manifest is stamped with the
CURRENT ``FORCED_RULE_VERSION`` / ``RULESET_VERSION``, so re-running after a
schema bump is the whole migration story — the JSONL never goes stale.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from catan_rl.labeling.to_shard import DEFAULT_OPPONENT_KINDS, convert

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels/setup/v1/scenarios.jsonl"),
        help="JSONL label store written by the labeling tool / game exporter.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write shard_00000.npz + manifest.json into.",
    )
    parser.add_argument(
        "--opponent-kinds",
        type=int,
        nargs="+",
        default=list(DEFAULT_OPPONENT_KINDS),
        help=(
            "Opponent-kind ids to duplicate every human row across. The policy "
            "conditions on an opponent-id embedding, so a single kind leaves the "
            "openings unexpressed under the others."
        ),
    )
    parser.add_argument(
        "--held-out-frac",
        type=float,
        default=0.2,
        help=(
            "Fraction of game_seeds withheld from the shard for the D7 gate-1 "
            "agreement measurement. The held-out seeds are stamped into the "
            "manifest and scripts/eval_setup_agreement.py reads them from there, "
            "so the gate can never be measured on rows the fine-tune trained on. "
            "0 disables the split — and makes the gate CLI refuse the shard."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = convert(
        args.labels,
        args.out,
        opponent_kinds=tuple(args.opponent_kinds),
        held_out_frac=args.held_out_frac,
        split_seed=args.split_seed,
    )
    print(
        f"wrote {manifest['n_pairs']} rows from {manifest['n_scenarios']} scenarios "
        f"to {args.out} (sources: {manifest['label_source_counts']}; "
        f"held out {manifest['n_held_out_scenarios']} scenarios over "
        f"{len(manifest['held_out_game_seeds'])} seeds)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
