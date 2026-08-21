#!/usr/bin/env python
"""D5 — forced-opening win-rate probe (machine time).

Thin CLI. The driver lives in :func:`catan_rl.setup_phase.wr_probe.run_probe`.

Usage::

    python scripts/run_forced_opening_probe.py \\
        --ckpt runs/anchors/ptr_v1_u500.pt \\
        --scorer data/setup_phase/scorer_weights_v1.json \\
        --n-seeds 100

The SAME checkpoint plays both sides. Paired seeds x both seats; one arm's
opening is forced to the scorer's picks, the other to the checkpoint's own
argmax picks. The reading is PRE-REGISTERED (see the module docstring): a
delta near zero is AMBIGUOUS, not a refutation.

Pinned to CPU per the repo device policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from catan_rl.setup_phase.scorer import load_weights
from catan_rl.setup_phase.wr_probe import run_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", type=Path, default=REPO_ROOT / "runs" / "anchors" / "ptr_v1_u500.pt"
    )
    parser.add_argument(
        "--scorer",
        type=Path,
        default=REPO_ROOT / "data" / "setup_phase" / "scorer_weights_v1.json",
    )
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from typing import cast

    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        _PolicyActor,
        build_actor(PlayerSpec(kind="policy", ckpt_path=str(args.ckpt)), seed=0, device="cpu"),
    )
    policy = actor.policy
    policy.eval()
    opponent = FrozenSnapshotOpponent(policy, device=actor.device, seed=0)

    result = run_probe(
        scorer=load_weights(args.scorer),
        policy=policy,
        opponent=opponent,
        seeds=range(args.seed_base, args.seed_base + args.n_seeds),
        device="cpu",
        max_turns=args.max_turns,
        alpha=args.alpha,
    )
    text = json.dumps(result.report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
