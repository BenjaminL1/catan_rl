"""T016 — run the inference-search bake-off GATE on the v6 frontier checkpoint.

Two-stage Wilson gate (mirrors search.bakeoff.run_bakeoff, but writes the n=200
screen result to disk before the n=500 confirm so progress is visible on the
multi-hour run): minimal determinized search (sims=50) vs the RAW policy it
wraps. PASS iff Wilson lower bound > 0.50 at n>=200, re-confirmed on a DISJOINT
n>=500 sample. CPU-pinned (eval device policy).

Writes runs/search/bakeoff_n200.json, runs/search/bakeoff_n500.json,
runs/search/bakeoff_gate.json. Launch detached:
    nohup python scripts/run_search_bakeoff.py > runs/search/bakeoff.log 2>&1 &
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from catan_rl.eval.harness import EvalMatchupResult
from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import evaluate_search_vs_policy

CKPT = "runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt"
SIMS = 50
N_QUICK = 200
N_CONFIRM = 500
SEED = 0
OUT = Path("runs/search")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _summary(r: EvalMatchupResult) -> dict[str, object]:
    return {
        "n": r.n,
        "wins": r.wins,
        "wr": r.wr,
        "ci_lower": r.ci.lower,
        "ci_upper": r.ci.upper,
        "rules_violations": len(r.rules_violations),
        "n_seat0": r.n_seat0,
        "n_seat1": r.n_seat1,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = SearchConfig(sims_per_move=SIMS, seed=SEED)
    _log(f"BAKE-OFF START ckpt={CKPT} sims={SIMS} n_quick={N_QUICK} n_confirm={N_CONFIRM}")

    t0 = time.time()
    quick = evaluate_search_vs_policy(
        cfg, CKPT, CKPT, n_games=N_QUICK, seed=SEED, device="cpu", max_turns=400
    )
    q = _summary(quick)
    q["passed"] = quick.ci.lower > 0.5
    (OUT / "bakeoff_n200.json").write_text(json.dumps(q, indent=2))
    _log(
        f"QUICK done wr={quick.wr:.3f} LB={quick.ci.lower:.3f} "
        f"pass={q['passed']} violations={q['rules_violations']} "
        f"({(time.time() - t0) / 60:.1f} min)"
    )

    gate: dict[str, object] = {"ckpt": CKPT, "sims": SIMS, "seed": SEED, "quick": q}
    if not quick.ci.lower > 0.5:
        gate["passed"] = False
        gate["failure_mode"] = f"quick Wilson LB {quick.ci.lower:.3f} <= 0.50 at n={quick.n}"
        (OUT / "bakeoff_gate.json").write_text(json.dumps(gate, indent=2))
        _log(f"GATE FAIL at quick screen: {gate['failure_mode']}")
        return

    t1 = time.time()
    confirm = evaluate_search_vs_policy(
        cfg, CKPT, CKPT, n_games=N_CONFIRM, seed=SEED + N_QUICK, device="cpu", max_turns=400
    )
    c = _summary(confirm)
    c["passed"] = confirm.ci.lower > 0.5
    (OUT / "bakeoff_n500.json").write_text(json.dumps(c, indent=2))
    _log(
        f"CONFIRM done wr={confirm.wr:.3f} LB={confirm.ci.lower:.3f} "
        f"pass={c['passed']} violations={c['rules_violations']} "
        f"({(time.time() - t1) / 60:.1f} min)"
    )

    gate["confirm"] = c
    gate["passed"] = bool(c["passed"])
    gate["failure_mode"] = (
        None
        if c["passed"]
        else f"confirm Wilson LB {confirm.ci.lower:.3f} <= 0.50 at n={confirm.n}"
    )
    (OUT / "bakeoff_gate.json").write_text(json.dumps(gate, indent=2))
    _log(
        f"GATE {'PASS' if gate['passed'] else 'FAIL'} "
        f"(total {(time.time() - t0) / 60:.1f} min) -> runs/search/bakeoff_gate.json"
    )


if __name__ == "__main__":
    main()
