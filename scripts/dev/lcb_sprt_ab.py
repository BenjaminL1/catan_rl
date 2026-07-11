"""LCB-vs-max-visit final-move A/B on the SPRT gate (spec 008 STAGE-A, FR-002/SC-003).

The no-regression check: does switching the final root move from the deployed
``max_visit`` rule to ``lcb`` (``argmax mean_Q - z*stderr``) regress at MATCHED
total sim budget? Both sides wrap the FROZEN v8 net; only ``final_move_mode``
differs, so ``assert_matched_budget`` passes trivially (same sims, n_det=1). The
pentanomial SPRT plays seat-swapped common-seed pairs vs a frozen v8 reference
until PROMOTE / REJECT / INCONCLUSIVE.

Verdict (SC-003): the LCB rule PASSES iff the SPRT does NOT return REJECT (i.e.
LCB does not lose strength vs max-visit). PROMOTE would additionally mean LCB is
a small win; INCONCLUSIVE means no regression detected within the game cap.

Inference-only, additive, CPU, fixed seeds. Diagnostic driver (no source change).

Usage (smoke)::

    python scripts/dev/lcb_sprt_ab.py --sims 25 --max-pairs 2 \
        --out runs/search/lcb_sprt_ab_smoke.json

Usage (modest, run alongside training)::

    python scripts/dev/lcb_sprt_ab.py --sims 48 --max-pairs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"


def run(*, sims: int, max_pairs: int, lcb_z: float, seed: int, out_path: Path) -> dict[str, Any]:
    from catan_rl.search.config import SearchConfig
    from catan_rl.search.sprt import (
        SPRTConfig,
        config_total_sim_budget,
        run_sprt_config_vs_config,
    )

    cfg_lcb = SearchConfig(sims_per_move=sims, seed=seed, final_move_mode="lcb", lcb_z=lcb_z)
    cfg_max = SearchConfig(sims_per_move=sims, seed=seed, final_move_mode="max_visit")

    t0 = time.time()
    res = run_sprt_config_vs_config(
        cfg_lcb,
        cfg_max,
        ckpt=V8_CKPT,
        reference_ckpt=V8_CKPT,
        sprt_cfg=SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=max_pairs),
        seed=seed,
        on_pair=lambda i, s, sp: print(
            f"[lcb-sprt] pair {i}: score={s} llr={sp.llr():+.3f} "
            f"elo_est={sp.elo_estimate():+.1f} decision={sp.decision()} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        ),
    )

    no_regression = res.decision != "REJECT"
    out = {
        "probe": "LCB-vs-max-visit final-move A/B on the SPRT gate (spec 008 FR-002/SC-003)",
        "ckpt": V8_CKPT,
        "configs": {
            "agent_a": f"final_move_mode='lcb' (z={lcb_z})",
            "agent_b": "final_move_mode='max_visit' (deployed)",
            "reference": "frozen v8 (raw, no search)",
            "sims_per_move": sims,
            "total_sim_budget_a": config_total_sim_budget(cfg_lcb),
            "total_sim_budget_b": config_total_sim_budget(cfg_max),
            "sprt": {
                "elo0": 0.0,
                "elo1": 10.0,
                "alpha": 0.05,
                "beta": 0.05,
                "max_pairs": max_pairs,
            },
        },
        "sprt_results": {
            "decision": res.decision,
            "llr": res.llr,
            "n_pairs": res.n_pairs,
            "n_games": res.n_games,
            "pentanomial_counts": res.counts,
            "elo_estimate": res.elo_estimate,
        },
        "no_regression": no_regression,
        "reason": (
            "SC-003 PASS: LCB does not regress vs max-visit (SPRT != REJECT)"
            if no_regression
            else "SC-003 FAIL: LCB regresses vs max-visit (SPRT REJECT)"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 80)
    print("LCB-vs-max-visit A/B (spec 008 SC-003)")
    print(
        f"  SPRT={res.decision} llr={res.llr:+.3f} elo_est={res.elo_estimate:+.1f} "
        f"games={res.n_games} counts={res.counts}"
    )
    print(f"  no_regression={no_regression}")
    print(f"  wrote {out_path}")
    print("=" * 80)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=48, help="sims/move, matched on both sides")
    parser.add_argument("--max-pairs", type=int, default=10, help="SPRT pair cap (4 games/pair)")
    parser.add_argument("--lcb-z", type=float, default=1.96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/search/lcb_sprt_ab.json"))
    args = parser.parse_args(argv)
    run(
        sims=args.sims,
        max_pairs=args.max_pairs,
        lcb_z=args.lcb_z,
        seed=args.seed,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
