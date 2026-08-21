#!/usr/bin/env python
"""D0 — the labeler-noise ceiling: owner-vs-owner top-1 agreement.

Thin CLI. The arithmetic lives in
:mod:`catan_rl.labeling.consistency`.

Usage::

    python scripts/report_label_consistency.py \\
        --labels data/labels/setup/v1/scenarios.jsonl

Reads the whole label store, pairs re-labeled positions under a NAMED
estimator, and prints per-position + overall settlement self-agreement with
Wilson CIs, plus ``road_given_same_settlement`` — roads are scored ONLY where
the settlement agreed, because a setup road must be incident to the settlement
just placed and a road compared across two different settlements is a
comparison across two different legal sets.

``--estimator`` selects how re-labeled positions are paired (default ``auto``):

  ``linked``         ``replay_of`` rows from ``--replay-session``. Unbiased —
                     the forced-original replay advances the draft with the
                     ORIGINAL pick, so every pick stands on the same position.
  ``free_replay``    joined on ``(game_seed, draft_position)`` alone. **This is
                     the estimator the banked D0 RESULT (7/20 = 35%, Wilson
                     [18, 57]) was computed with**, and rerunning it on the
                     shipped store reproduces that figure exactly. Biased
                     DOWNWARD on picks 2-4 (a free replay diverges after a
                     disagreement).
  ``same_position``  ``free_replay`` narrowed to identical ``prior_picks``.
                     Biased UPWARD — it selects on the outcome — and cannot
                     reach draft position 4. Reads 6/8 = 75% on the same store.
  ``auto``           ``linked`` if the corpus has any, else ``free_replay``.

Whatever is chosen, the report names it under ``estimator``, states its bias
under ``estimator_bias``, and prints all three overall rates side by side under
``estimators`` — a 40pp spread the report refuses to hide behind one number.

This is NOT a gate. It is the number every later bar is read against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from catan_rl.labeling.consistency import ESTIMATOR_AUTO, ESTIMATORS, consistency_report
from catan_rl.labeling.store import load_scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--estimator",
        choices=[*ESTIMATORS, ESTIMATOR_AUTO],
        default=ESTIMATOR_AUTO,
        help=(
            "How re-labeled positions are paired. 'free_replay' reproduces the "
            "banked D0 RESULT; 'same_position' is upward-biased; 'auto' prefers "
            "replay_of-linked pairs and falls back to 'free_replay'."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the report as JSON.")
    args = parser.parse_args()

    rows = load_scenarios(args.labels)
    report = consistency_report(rows, alpha=args.alpha, estimator=args.estimator)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
