"""T021 — strength-budget ladder (SC-002): WR vs the raw policy scales with compute.

Runs the determinized search vs the RAW v6 policy at increasing simulation budgets
on the SAME (paired) seat-symmetrized games, so a monotone win-rate isolates the
effect of search depth from board/dice variance. We use a reproducible SIMS ladder
(sims is compute; bit-reproducible, FR-006) rather than wall-clock tiers, which are
machine-dependent and would cost ~17h at the spec's 5s/move x 200 games.

Writes runs/search/ladder_<sims>.json per tier + runs/search/ladder_summary.json
(with the monotonicity verdict). Launch detached:
    nohup python scripts/run_search_ladder.py > runs/search/ladder.log 2>&1 &
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import evaluate_search_vs_policy

CKPT = "runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt"
SIMS_TIERS = (10, 50, 100)
N_GAMES = 100
SEED = 0  # same eval seed across tiers => paired games (lower-variance comparison)
OUT = Path("runs/search")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _log(f"LADDER START ckpt={CKPT} tiers={SIMS_TIERS} n={N_GAMES} (paired, seed={SEED})")
    rows: list[dict[str, object]] = []
    for sims in SIMS_TIERS:
        t0 = time.time()
        cfg = SearchConfig(sims_per_move=sims, seed=0)
        res = evaluate_search_vs_policy(
            cfg, CKPT, CKPT, n_games=N_GAMES, seed=SEED, device="cpu", max_turns=400
        )
        row: dict[str, object] = {
            "sims": sims,
            "n": res.n,
            "wins": res.wins,
            "wr": res.wr,
            "ci_lower": res.ci.lower,
            "ci_upper": res.ci.upper,
            "rules_violations": len(res.rules_violations),
        }
        rows.append(row)
        (OUT / f"ladder_{sims}.json").write_text(json.dumps(row, indent=2))
        _log(
            f"tier sims={sims}: WR {res.wr:.3f} [{res.ci.lower:.3f},{res.ci.upper:.3f}] "
            f"violations={row['rules_violations']} ({(time.time() - t0) / 60:.1f} min)"
        )

    wrs = [r["wr"] for r in rows]
    monotone = all(wrs[i] <= wrs[i + 1] for i in range(len(wrs) - 1))  # type: ignore[operator]
    summary = {
        "ckpt": CKPT,
        "tiers": list(SIMS_TIERS),
        "n_games": N_GAMES,
        "rows": rows,
        "wr_by_sims": {r["sims"]: r["wr"] for r in rows},
        "monotone_in_budget": monotone,
    }
    (OUT / "ladder_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"LADDER DONE monotone={monotone} wr_by_sims={summary['wr_by_sims']}")


if __name__ == "__main__":
    main()
