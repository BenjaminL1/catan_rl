#!/usr/bin/env python
"""D7 gate 1 — held-out top-1 settlement agreement (candidate vs baseline).

Thin CLI. The arithmetic lives in :func:`catan_rl.bc.gates.setup_agreement_gate`
and the measurement in :func:`catan_rl.bc.finetune.setup_agreement`.

Usage::

    python scripts/eval_setup_agreement.py \\
        --candidate runs/finetune/human_openings/candidate.pt \\
        --baseline runs/anchors/ptr_v1_u500.pt \\
        --labels data/labels/setup/v1/scenarios.jsonl \\
        --shard-dir data/bc/human_openings/v1

**The held-out set is not chosen here.** It is read out of the shard's
``manifest.json`` (``held_out_game_seeds``), written by
``scripts/convert_labels_to_bc_shard.py`` when it EXCLUDED those seeds from the
training shard. Re-deriving a split at gate time would measure agreement on the
very rows the candidate was fine-tuned on, and a memorisation readout is exactly
what D7 exists to remove. A shard built with ``--held-out-frac 0`` carries no
held-out seeds, and this CLI refuses it rather than inventing one.

The bar is ``max(0.30, baseline_agreement + 0.10)``. The reported CI is
information, not a second gate: at ~40 held-out scenarios the estimate carries
roughly +-14pp, so the remedy for a marginal result is MORE LABELS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from catan_rl.bc.finetune import setup_agreement
    from catan_rl.bc.gates import setup_agreement_gate
    from catan_rl.labeling.store import load_scenarios

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/anchors/ptr_v1_u500.pt"),
        help="The PRE-fine-tune champion. Its agreement calibrates the bar.",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help=(
            "The shard the candidate was fine-tuned on. Its manifest names the "
            "held-out game_seeds this gate measures on."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    manifest_path = args.shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "held_out_game_seeds" not in manifest:
        raise SystemExit(
            f"{manifest_path} carries no 'held_out_game_seeds' — it was built by a "
            f"converter that did not withhold anything, so every label in "
            f"{args.labels} is in-sample. Re-run convert_labels_to_bc_shard.py "
            f"with --held-out-frac > 0 and re-run the fine-tune on that shard."
        )
    held_seeds = {int(s) for s in manifest["held_out_game_seeds"]}
    if not held_seeds:
        raise SystemExit(
            f"{manifest_path} has an EMPTY held-out set (--held-out-frac 0). "
            f"Gate 1 would report memorisation, not generalisation."
        )

    labels = load_scenarios(args.labels)
    held = [row for row in labels if int(row["game_seed"]) in held_seeds]
    if not held:
        raise SystemExit(
            f"none of the {len(labels)} rows in {args.labels} carry a held-out "
            f"game_seed from {manifest_path} — the label file and the shard do "
            f"not describe the same corpus."
        )

    baseline = setup_agreement(args.baseline, held, device=args.device)
    candidate = setup_agreement(args.candidate, held, device=args.device)
    result = setup_agreement_gate(
        agreements=candidate,
        baseline=sum(baseline) / len(baseline),
    )
    result["held_out_game_seeds"] = sorted(held_seeds)
    result["labels_path"] = str(args.labels)
    result["shard_dir"] = str(args.shard_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
