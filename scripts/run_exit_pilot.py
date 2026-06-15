"""T016 — run the 004 expert-iteration PILOT GATE (label -> distill -> gate).

Search-labeled data generation (CPU search) -> warm-started BC distillation of v6
(MPS if available) -> SEARCH-FREE distilled-vs-raw-v6 gate (CPU). PASS iff the
distilled policy beats raw v6 with Wilson LB > 0.50 at n>=200 then n>=500.

Writes data/exit/round_0/ (shards) + runs/exit/round_0/{distill/, gate.json}.
Launch detached:
    nohup python scripts/run_exit_pilot.py --sims 50 --n-positions 5000 > runs/exit/pilot.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--n-positions", type=int, default=5000)
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--data", default="data/exit/round_0")
    ap.add_argument("--out", default="runs/exit/round_0")
    ap.add_argument("--n-quick", type=int, default=200)
    ap.add_argument("--n-confirm", type=int, default=500)
    args = ap.parse_args()

    import torch

    from catan_rl.expert_iteration.config import V6_BASE_CKPT, DistillConfig, SearchLabelConfig
    from catan_rl.expert_iteration.distill import distill
    from catan_rl.expert_iteration.gate import run_gate
    from catan_rl.expert_iteration.labeler import generate_search_labels

    distill_device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _log(f"PILOT START sims={args.sims} n_positions={args.n_positions} opp={args.opponent}")
    t0 = time.time()
    label_cfg = SearchLabelConfig(
        out_dir=args.data,
        sims_per_move=args.sims,
        opponent=args.opponent,
        n_positions=args.n_positions,
        seed=args.seed,
    )
    manifest = generate_search_labels(label_cfg)
    _log(f"LABEL done: {manifest['n_pairs_total']} positions ({(time.time() - t0) / 60:.1f} min)")

    t1 = time.time()
    distill_cfg = DistillConfig(
        data_dir=args.data,
        out_dir=str(out / "distill"),
        init_ckpt=V6_BASE_CKPT,
        peak_lr=args.lr,
        max_epochs=args.epochs,
        seed=args.seed,
        device=distill_device,
    )
    distilled = distill(distill_cfg)
    _log(f"DISTILL done: {distilled} ({(time.time() - t1) / 60:.1f} min, device={distill_device})")

    t2 = time.time()
    verdict = run_gate(
        str(distilled),
        V6_BASE_CKPT,
        n_quick=args.n_quick,
        n_confirm=args.n_confirm,
        seed=args.seed,
        device="cpu",
    )
    (out / "gate.json").write_text(json.dumps(verdict, indent=2))
    _log(
        f"GATE {'PASS' if verdict.get('passed') else 'FAIL'} "
        f"wr_vs_v6={verdict.get('wr_confirm_vs_v6', verdict.get('wr_quick_vs_v6')):.3f} "
        f"({(time.time() - t2) / 60:.1f} min) -> {out / 'gate.json'}"
    )
    _log(f"PILOT total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
