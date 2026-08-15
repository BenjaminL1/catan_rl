#!/usr/bin/env python
"""D7 gate 2 — paired-seed WR non-inferiority (candidate vs pre-fine-tune champion).

Thin CLI, mirroring ``scripts/eval_setup_agreement.py``. The arithmetic lives in
:func:`catan_rl.bc.gates.paired_wr_non_inferiority` and the games are played by
:class:`catan_rl.eval.harness.EvalHarness`; this file only orchestrates.

Usage::

    python scripts/eval_wr_non_inferiority.py \\
        --candidate runs/finetune/human_openings/candidate.pt \\
        --baseline runs/anchors/ptr_v1_u500.pt \\
        --n-games-per-seat 300

**Pairing is by construction, not by request.** ``EvalHarness`` derives its seed
plan from ``self.seed`` plus the opponent label alone, so two harnesses built
with the same ``seed`` / ``opponent_types`` / ``n_games_per_seat`` play the same
``(seed, agent_seat)`` keys. That is what lets the gate difference out the
board-and-dice variance a 5pp margin would otherwise be invisible under; if the
harness ever stopped doing so, ``paired_wr_non_inferiority`` raises
``UnpairableEvalError`` rather than reporting an unpaired number.

**Both checkpoints must be R0** (D6). ``ptr_v1_u500.pt`` carries no ruleset
stamp, so the lineage this fine-tune extends is R0, and R0/R1 numbers are not
comparable. A mismatch fails CLOSED — this CLI refuses rather than annotating,
the same rule ``eval.harness`` applies to cross-epoch head-to-heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HEURISTIC = "heuristic"


def main(argv: list[str] | None = None) -> int:
    import torch

    from catan_rl.bc.finetune import build_policy
    from catan_rl.bc.gates import paired_wr_non_inferiority
    from catan_rl.env.ruleset import RULESET_R0
    from catan_rl.eval.harness import EvalHarness, checkpoint_ruleset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/anchors/ptr_v1_u500.pt"),
        help="The PRE-fine-tune champion the candidate must be non-inferior to.",
    )
    parser.add_argument(
        "--n-games-per-seat",
        type=int,
        default=300,
        help=(
            "Games per seat per checkpoint. The paired SE is ~2pp at 600 total, "
            "which is what makes a -5pp margin detectable at all."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Shared seed plan for both rounds.")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help=(
            "Non-inferiority margin, as an absolute win-rate difference. PASS iff the "
            "CI lower bound on WR_ft - WR_pre exceeds MINUS this (spec D7.2) — i.e. "
            "the candidate is provably not worse by more than the margin. It is NOT a "
            "superiority test: a candidate slightly behind on the point estimate "
            "passes, which is the question the owner actually has."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    for label, path in (("candidate", args.candidate), ("baseline", args.baseline)):
        epoch = checkpoint_ruleset(path)
        if epoch != RULESET_R0:
            raise SystemExit(
                f"{label} {path} is stamped {epoch}, but this gate runs {RULESET_R0} "
                f"(D6: ptr_v1_u500.pt carries no ruleset stamp, so the lineage is "
                f"{RULESET_R0}). {epoch} and {RULESET_R0} win rates are not comparable; "
                f"re-run the fine-tune under the epoch you mean to gate."
            )

    device = torch.device(args.device)

    def _wr(path: Path) -> Any:
        policy = build_policy(path, device)
        policy.eval()
        # A FRESH harness per checkpoint, identically configured: the seed plan
        # derives from these arguments alone, so the two rounds are paired
        # without either round knowing about the other.
        harness = EvalHarness(
            opponent_types=(HEURISTIC,),
            n_games_per_seat=args.n_games_per_seat,
            seed=args.seed,
            device=device,
            max_turns=args.max_turns,
            ruleset=RULESET_R0,
        )
        result = harness.run(policy).by_opponent(HEURISTIC)
        if result is None:
            raise SystemExit(f"no '{HEURISTIC}' matchup in the eval report for {path}")
        return result

    finetuned = _wr(args.candidate)
    pre = _wr(args.baseline)

    payload = paired_wr_non_inferiority(finetuned=finetuned, pre=pre, margin=args.margin)
    payload["candidate"] = str(args.candidate)
    payload["baseline"] = str(args.baseline)
    payload["ruleset"] = RULESET_R0
    payload["seed"] = int(args.seed)
    payload["n_games_per_seat"] = int(args.n_games_per_seat)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
