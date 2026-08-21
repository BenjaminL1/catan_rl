#!/usr/bin/env python
"""RECONSTRUCTION of the 2026-08-15 pilot scorer fit — provenance, not authority.

**This is not the original artifact.** The pilot's script lived in a session
scratchpad that no longer exists; it was never committed. What is committed here
is a reconstruction from the pilot's reported design — a 10-feature linear
scorer under a masked softmax over the legal vertices — expressed as a COLUMN
SUBSET of the current design matrix
(:data:`catan_rl.setup_phase.scorer_features.PILOT_FEATURE_NAMES`). It exists so
the 34.1% held-out / 45.2% train numbers the program was adopted on have a
runnable definition in the repo, and so acceptance criterion 3 has something to
compare against.

Bit-exactness against the original is IMPOSSIBLE and is not claimed. Read the
output as "the pilot reproduces to within a couple of points", never as "the
pilot is reproduced".

The pilot's split is the one PINNED in
``data/bc/human_openings/v1/manifest.json`` (``held_out_game_seeds``), read out
of the manifest rather than re-derived: ``held_out_split`` is seed-count-relative
and would silently change the moment the owner labels another board.

The pilot CORPUS is the earliest ``n_scenarios + n_held_out_scenarios`` rows by
``labeled_at`` — the store has grown since, and fitting on rows that postdate the
pilot would not be the pilot.

Usage::

    python scripts/dev/fit_scorer_pilot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from catan_rl.eval.wilson import wilson_interval
from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.fit import fit_scorer, top1_agreement
from catan_rl.setup_phase.scorer_features import PILOT_FEATURE_NAMES

PILOT_HELD_OUT_AGREEMENT = 0.341
"""The pilot's reported held-out top-1 settlement agreement."""

PILOT_TRAIN_AGREEMENT = 0.452
"""The pilot's reported train top-1 settlement agreement."""


def pilot_split(labels_path: Path, manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """``(train, held_out)`` exactly as the pilot's shard manifest pins them."""
    manifest = json.loads(manifest_path.read_text())
    held = {int(s) for s in manifest["held_out_game_seeds"]}
    n_pilot = int(manifest["n_scenarios"]) + int(manifest["n_held_out_scenarios"])
    rows = sorted(
        load_scenarios(labels_path),
        key=lambda r: (str(r["labeled_at"]), str(r["scenario_id"])),
    )[:n_pilot]
    train = [r for r in rows if int(r["game_seed"]) not in held]
    held_out = [r for r in rows if int(r["game_seed"]) in held]
    if len(train) != int(manifest["n_scenarios"]) or len(held_out) != int(
        manifest["n_held_out_scenarios"]
    ):
        raise SystemExit(
            f"pilot split did not reconstruct: got {len(train)}/{len(held_out)}, "
            f"manifest says {manifest['n_scenarios']}/{manifest['n_held_out_scenarios']}"
        )
    return train, held_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "bc" / "human_openings" / "v1" / "manifest.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, held_out = pilot_split(args.labels, args.manifest)
    subset = list(PILOT_FEATURE_NAMES)
    result = fit_scorer(
        train, version="pilot-reconstruction", seed=args.seed, settlement_feature_subset=subset
    )
    held_agree = top1_agreement(result.scorer, held_out, settlement_feature_subset=subset)
    ci = wilson_interval(wins=int(np.sum(held_agree)), n=len(held_agree))
    pilot_ci = wilson_interval(
        wins=round(PILOT_HELD_OUT_AGREEMENT * len(held_agree)), n=len(held_agree)
    )
    print(
        json.dumps(
            {
                "n_train": len(train),
                "n_held_out": len(held_out),
                "train_agreement": result.metrics["settlement"]["agreement"],
                "pilot_train_agreement": PILOT_TRAIN_AGREEMENT,
                "held_out_agreement": float(np.mean(held_agree)),
                "pilot_held_out_agreement": PILOT_HELD_OUT_AGREEMENT,
                "held_out_ci": [ci.lower, ci.upper],
                "pilot_held_out_ci": [pilot_ci.lower, pilot_ci.upper],
                "note": (
                    "RECONSTRUCTION. Bit-exactness against the pilot is impossible "
                    "(its script is gone); read this as agreement to within a couple "
                    "of points, not as a reproduction."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
