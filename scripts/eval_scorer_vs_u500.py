#!/usr/bin/env python
"""D4 v2 — the tie-aware, paired, per-position forward exam: scorer vs u500.

PRIMARY metric: paired mean LOG-PROBABILITY of the owner's pick (a proper
scoring rule). Top-1 agreement is reported alongside but no longer decides —
D0 measured the owner's own top-1 self-agreement at ~35%, a labeler ceiling the
pilot scorer already sits at.

Thin CLI. The arithmetic lives in :func:`catan_rl.setup_phase.gate.evaluate_gate`.

Usage::

    python scripts/eval_scorer_vs_u500.py \\
        --labels data/labels/setup/v1/scenarios.jsonl \\
        --baseline runs/anchors/ptr_v1_u500.pt \\
        --scorer v1=data/setup_phase/scorer_weights_v1.json

Unlike ``scripts/eval_setup_agreement.py`` (which measures a fine-tuned
CHECKPOINT on a shard manifest's held-out seeds), this runs on an arbitrary
LABEL SUBSET: the fresh blind-first picks, selected by their ``reveal_mode``
session. That is what D4 asks for, and it is why the shard-manifest CLI is left
untouched rather than generalised.

``--scorer VERSION=PATH`` may be repeated. Every fresh pick is graded by the
scorer version that was LIVE WHEN IT WAS LABELED (D6); a pick whose stamp has no
matching artifact makes the run fail rather than borrow another scorer.

PASS requires ALL of: >=150 fresh picks, D3's >=20% no-reveal anchoring control
satisfied, overall paired log-prob CI lower bound > 0, a positive picks-2-4
paired log-prob delta, and >=70% scorer top-1 on the picks the owner tagged
``clear``. The exit code follows ``report["passes"]``, and every clause is
itemised under ``report["pass_clauses"]``.

Eval is pinned to CPU per the repo device policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.gate import evaluate_gate, session_metadata
from catan_rl.setup_phase.scorer import load_weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "runs" / "anchors" / "ptr_v1_u500.pt",
        help="The u500 champion checkpoint the scorer is paired against.",
    )
    parser.add_argument(
        "--scorer",
        action="append",
        default=[],
        metavar="VERSION=PATH",
        help="A fitted scorer artifact and the version stamp it answers to.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-fresh-picks", type=int, default=150)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.scorer:
        parser.error("at least one --scorer VERSION=PATH is required")
    scorers = {}
    for spec in args.scorer:
        if "=" not in spec:
            parser.error(f"--scorer must be VERSION=PATH, got {spec!r}")
        version, path = spec.split("=", 1)
        scorers[version] = load_weights(Path(path))

    rows = load_scenarios(args.labels)
    report = evaluate_gate(
        rows,
        scorers_by_version=scorers,
        baseline_ckpt=args.baseline,
        session_meta=session_metadata(args.labels.parent),
        alpha=args.alpha,
        min_fresh_picks=args.min_fresh_picks,
        device="cpu",
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
