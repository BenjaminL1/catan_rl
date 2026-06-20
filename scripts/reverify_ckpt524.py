"""High-n re-verification of the ckpt_524-vs-v8 promotion gate (ruler follow-up).

The n=600 transitive ladder ranked ckpt_524 #1 and its per-anchor Elo deltas vs v8
were point-POSITIVE (v7 +54.5, v6 +25.9), but the binding clause-1 signal — the MIN
over the un-gamed anchors of [pairwise-Elo(ckpt_524 vs a) - pairwise-Elo(v8 vs a)] —
had a bootstrap CI [-16.2, +59.4] that STRADDLES 0, so the gate withheld promotion
(underpowered, not flat). This re-runs ONLY the five gate-relevant matchups at high n
with a FRESH independent seed (a true 2nd measurement, not a re-run of the same dice),
merges them into the ladder, and re-applies the EXACT hardened promotion_check. If the
min-anchor-delta CI now clears 0, ckpt_524 is a confirmed GLOBAL gain (new champion);
if it still straddles, the edge is too marginal to bank and we move on.

Reuses scripts/elo_ladder.py wholesale (run_match, fit_elo, promotion_check,
bootstrap_elo_ci, min_anchor_delta_dist, nontransitivity_residual) — no reimplemented
eval or gate logic. Policy-vs-policy, seat-symmetric, CPU-pinned per the device policy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "elo_ladder.py"
_spec = importlib.util.spec_from_file_location("elo_ladder_module", _SCRIPT)
assert _spec is not None and _spec.loader is not None
elo = importlib.util.module_from_spec(_spec)
sys.modules["elo_ladder_module"] = elo
_spec.loader.exec_module(elo)

CAND = "ckpt_524"
BASE = "v8_u243"
ANCHORS = ("v6_u1399", "v7_u399")
NPS = 1000  # n = 2*nps = 2000 games/pair (vs the ladder's 600); ~1.6x tighter delta CI
SEED_BASE = 4242  # fresh seeds -> independent dice from the ladder's seed=0.. measurement
LADDER_JSON = _ROOT / "runs" / "elo_ladder_transitive.json"
OUT_JSON = _ROOT / "runs" / "reverify_ckpt524.json"

# Resolve checkpoint paths from the ladder's own rung table (single source of truth).
_PATH = {nm: p for nm, k, p in elo.RUNGS}


def _task(a: str, b: str, seed: int) -> dict[str, Any]:
    return {
        "a_name": a,
        "a_kind": "policy",
        "a_path": _PATH[a],
        "a_sims": 0,
        "b_name": b,
        "b_kind": "policy",
        "b_path": _PATH[b],
        "nps": NPS,
        "seed": seed,
    }


def main() -> None:
    # The 5 gate-relevant pairs: candidate & baseline each vs both un-gamed anchors
    # (feeds the min-anchor delta), plus the candidate-vs-baseline head-to-head (context).
    pairs = [
        (CAND, "v6_u1399"),
        (CAND, "v7_u399"),
        (BASE, "v6_u1399"),
        (BASE, "v7_u399"),
        (CAND, BASE),
    ]
    tasks = [_task(a, b, SEED_BASE + i) for i, (a, b) in enumerate(pairs)]
    print(f"[reverify] {len(tasks)} matchups @ n={2 * NPS}/pair, seed base {SEED_BASE}", flush=True)

    t0 = time.time()
    hi: dict[frozenset[str], tuple[str, str, int, int]] = {}
    with Pool(len(tasks)) as pool:
        for i, m in enumerate(pool.imap_unordered(elo.run_match, tasks), 1):
            hi[frozenset((m[0], m[1]))] = m
            print(
                f"[reverify] {i}/{len(tasks)} {m[0]} vs {m[1]}: {m[2]}/{m[3]} "
                f"(WR {m[2] / m[3]:.4f})",
                flush=True,
            )
    dt = time.time() - t0

    # Merge: replace the 5 ladder matchups with their high-n versions, keep the rest.
    led = json.loads(LADDER_JSON.read_text())
    names = list(led["ratings"].keys())
    merged: list[tuple[str, str, int, int]] = []
    n_replaced = 0
    for mm in led["matches"]:
        key = frozenset((mm["a"], mm["b"]))
        if key in hi:
            merged.append(hi[key])
            n_replaced += 1
        else:
            merged.append((mm["a"], mm["b"], int(mm["wins_a"]), int(mm["n"])))
    print(
        f"[reverify] merged {n_replaced}/{len(hi)} high-n matchups into the ladder "
        f"({len(merged)} total); refitting",
        flush=True,
    )

    ratings = elo.fit_elo(merged, names)
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    resid, _detail, ev = elo.nontransitivity_residual(merged, ratings)
    boot = elo.bootstrap_elo_ci(merged, names, n_boot=2000, candidate=CAND, baseline=BASE)
    gate = elo.promotion_check(
        merged, names, ratings, boot, candidate=CAND, baseline=BASE, anchors=ANCHORS
    )

    print("\n=== ELO LADDER (re-verified, heuristic pinned @ 500) ===")
    for nm, r in ranked:
        ci = boot["elo_ci"].get(nm) if "elo_ci" in boot else None
        cis = f"  CI [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
        print(f"  {r:7.1f}  {nm}{cis}")
    print(f"[reverify] nontransitivity_residual={resid:.4f}  explained_variance={ev:.4f}")
    print(f"\n=== PROMOTION GATE ({CAND} vs {BASE}) — HIGH-N RE-VERIFY ===")
    print(f"  per_anchor_delta_elo : {gate['per_anchor_delta_elo']}")
    print(f"  min_anchor_delta_ci  : {gate['min_anchor_delta_ci']}  (clause-1 needs LB > 0)")
    print(f"  clause1_global_gain  : {gate['clause1_global_gain_vs_anchors']}")
    print(f"  clause2_non_regress  : {gate['clause2_non_regression']}  {gate['non_regression']}")
    print(
        f"  clause3_residual_ok  : {gate['clause3_residual_non_increase']}  "
        f"(with {gate['residual_with']:.4f} <= without {gate['residual_without']:.4f} + 0.01)"
    )
    verdict = (
        "PROMOTE — ckpt_524 is a CONFIRMED global gain over v8 (new champion)"
        if gate["passed"]
        else "WITHHOLD — ckpt_524's global edge over v8 is not robust at this n"
    )
    print(f"\n  GATE passed = {gate['passed']}  ->  {verdict}")

    OUT_JSON.write_text(
        json.dumps(
            {
                "high_n_matchups": [
                    {"a": m[0], "b": m[1], "wins_a": m[2], "n": m[3], "wr_a": m[2] / m[3]}
                    for m in hi.values()
                ],
                "n_per_pair": 2 * NPS,
                "seed_base": SEED_BASE,
                "seconds": dt,
                "ratings": ratings,
                "ranked": ranked,
                "nontransitivity_residual": resid,
                "explained_variance": ev,
                "elo_ci": boot.get("elo_ci", {}),
                "gate": gate,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
