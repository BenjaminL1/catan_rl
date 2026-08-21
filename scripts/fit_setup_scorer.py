#!/usr/bin/env python
"""D2 — fit the setup scorer on the owner's labels and write the artifact.

Thin CLI. The fit lives in :mod:`catan_rl.setup_phase.fit`.

Usage::

    python scripts/fit_setup_scorer.py \\
        --labels data/labels/setup/v1/scenarios.jsonl \\
        --out data/setup_phase/scorer_weights_v1.json \\
        --version v1

Replay rows (``replay_of``) are excluded from the fit by construction. The fit
REFUSES a corpus with duplicate ``(game_seed, draft_position)`` rows that carry
no ``replay_of`` — pass ``--on-duplicate first-labeled`` to keep the earliest of
each pair, a choice that is stamped into the artifact's provenance.

FIRST RUN, on the corpus as it stands today::

    python scripts/fit_setup_scorer.py \\
        --labels data/labels/setup/v1/scenarios.jsonl \\
        --out data/setup_phase/scorer_weights_v1.json \\
        --version v1 \\
        --on-duplicate first-labeled

The flag is REQUIRED today and the default refusal is not a bug: the owner ran a
free replay before ``replay_of`` existed, so those rows re-label positions
already in the corpus with no link to say so. Run it once WITHOUT the flag —
the refusal names the offending positions, and seeing them is how you decide
whether ``first-labeled`` is the right answer. Once a ``--replay-session`` run
has happened its rows carry ``replay_of`` and are excluded by identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.fit import DUPLICATE_POLICIES, fit_scorer
from catan_rl.setup_phase.scorer import save_weights
from catan_rl.setup_phase.scorer_features import PILOT_FEATURE_NAMES


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "setup_phase" / "scorer_weights_v1.json",
    )
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--pilot-features",
        action="store_true",
        help="Fit the 2026-08-15 pilot's 10-feature subset instead of the full block.",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=DUPLICATE_POLICIES,
        default="refuse",
        help="How to handle repeated (game_seed, draft_position) rows with no replay_of.",
    )
    args = parser.parse_args()

    rows = load_scenarios(args.labels)
    result = fit_scorer(
        rows,
        version=args.version,
        seed=args.seed,
        l2=args.l2,
        iters=args.iters,
        lr=args.lr,
        settlement_feature_subset=list(PILOT_FEATURE_NAMES) if args.pilot_features else None,
        duplicate_policy=args.on_duplicate,
        provenance={
            "labels_path": str(args.labels),
            "git_sha": _git_sha(),
            "fit_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": args.seed,
            "l2": args.l2,
            "iters": args.iters,
            "lr": args.lr,
            "duplicate_policy": args.on_duplicate,
            "pilot_features": bool(args.pilot_features),
        },
    )
    save_weights(result.scorer, args.out)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"[fit_setup_scorer] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
