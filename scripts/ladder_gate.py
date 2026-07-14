"""Parametric ladder fold-back gate: is CANDIDATE a GLOBAL gain over BASELINE (default v8)?

Games CANDIDATE vs the gate opponents (baseline + the un-gamed anchors + heuristic) at high n,
ADDS those matchups (and the candidate rung) to the cached transitive ladder, refits Bradley-Terry
Elo, and applies the EXACT hardened promotion_check from scripts/elo_ladder.py:
  clause-1 = MIN over un-gamed anchors {v6_u1399, v7_u399} of
             [elo_from_wr(WR(cand vs a)) - elo_from_wr(WR(baseline vs a))], bootstrap CI LB > 0.
A candidate that only COUNTERS the baseline head-to-head is flat vs the un-gamed anchors → fails
by design; a true global gain beats EVERY un-gamed anchor by more than the baseline → passes.

This is the v6-seeded EXPLOITER's fold-back decision (candidate = exploiter final ckpt, baseline =
v8). Reuses elo_ladder.py wholesale (run_match, fit_elo, promotion_check, bootstrap_elo_ci,
nontransitivity_residual) — no reimplemented eval or gate logic. Seat-symmetric, CPU-pinned.

If --reverify-json is given, the baseline-vs-anchor matchups in the cached ladder are upgraded to
that file's high-n versions so the per-anchor delta compares like-with-like at matched n.
"""

from __future__ import annotations

import argparse
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

# Checkpoint paths for the ladder rungs (single source of truth = elo_ladder's RUNGS table).
_PATH: dict[str, str | None] = {nm: p for nm, _k, p in elo.RUNGS}
_ENGINE_KINDS = {"heuristic", "random"}
_DEFAULT_REVERIFY = "runs/reverify_ckpt524.json"


def validate_inputs(
    candidate_ckpt: str, opponents: list[str], ladder_json: str, reverify_json: str
) -> None:
    """Fail-fast on operator errors BEFORE the multi-hour eval Pool (a typo or missing file
    otherwise crashes loudly only AFTER wasting the eval). Raises SystemExit on any bad input."""
    if not Path(candidate_ckpt).is_file():
        raise SystemExit(f"candidate-ckpt {candidate_ckpt} does not exist or is not a file")
    for o in opponents:
        kind, path = _kind_and_path(o)
        if kind == "policy" and path is None:
            raise SystemExit(f"unknown opponent {o!r}: not a rung in elo_ladder RUNGS")
    if not Path(ladder_json).is_file():
        raise SystemExit(f"ladder-json {ladder_json} not found — run scripts/elo_ladder.py first")
    if reverify_json != _DEFAULT_REVERIFY and not Path(reverify_json).is_file():
        raise SystemExit(f"reverify-json {reverify_json} not found")


def _kind_and_path(name: str) -> tuple[str, str | None]:
    """Resolve an opponent NAME to (kind, ckpt_path). Engine-driven kinds carry no path."""
    if name in _ENGINE_KINDS:
        return name, None
    return "policy", _PATH.get(name)


def _task(cand_name: str, cand_ckpt: str, opp: str, nps: int, seed: int) -> dict[str, Any]:
    bk, bp = _kind_and_path(opp)
    return {
        "a_name": cand_name,
        "a_kind": "policy",
        "a_path": cand_ckpt,
        "a_sims": 0,
        "b_name": opp,
        "b_kind": bk,
        "b_path": bp,
        "nps": nps,
        "seed": seed,
    }


def build_merged_matches(
    ladder_matches: list[dict[str, Any]],
    cand_matches: list[tuple[str, str, int, int]],
    *,
    overrides: dict[frozenset[str], tuple[str, str, int, int]] | None = None,
) -> list[tuple[str, str, int, int]]:
    """Cached ladder (as JSON dicts) + the candidate's NEW matchups, with optional high-n
    overrides of existing pairs (e.g. baseline-vs-anchor upgraded from a reverify run).
    Pure function (no eval) so it is unit-testable."""
    overrides = overrides or {}
    merged: list[tuple[str, str, int, int]] = []
    for mm in ladder_matches:
        key = frozenset((mm["a"], mm["b"]))
        if key in overrides:
            merged.append(overrides[key])
        else:
            merged.append((mm["a"], mm["b"], int(mm["wins_a"]), int(mm["n"])))
    merged.extend(cand_matches)
    return merged


def fit_name_universe(
    cached_names: list[str], matches: list[tuple[str, str, int, int]]
) -> list[str]:
    """Name universe for the Bradley-Terry fit = cached ladder names UNION every
    participant appearing in ``matches`` (order-preserving). Deriving it from the cached
    ratings ALONE KeyErrors in ``fit_elo`` whenever a matchup references a rung added to
    RUNGS after the cached ladder was last rebuilt — e.g. the v9 baseline (or the
    candidate) absent from a pre-v9 ``elo_ladder_transitive.json``."""
    names = list(cached_names)
    for a, b, _w, _n in matches:
        for nm in (a, b):
            if nm not in names:
                names.append(nm)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-ckpt", required=True, help="path to the candidate checkpoint")
    ap.add_argument("--candidate-name", required=True, help="rung name for the candidate")
    ap.add_argument("--baseline", default="v8_u243", help="champion to beat (default v8)")
    ap.add_argument("--anchors", default="v6_u1399,v7_u399", help="comma-sep un-gamed anchors")
    ap.add_argument(
        "--extra-opponents",
        default="heuristic",
        help="comma-sep extra opponents gamed for ladder anchoring + general-skill sanity",
    )
    ap.add_argument("--nps", type=int, default=1000, help="n_games_per_seat (total/pair = 2*nps)")
    ap.add_argument("--seed-base", type=int, default=4242)
    ap.add_argument("--ladder-json", default="runs/elo_ladder_transitive.json")
    ap.add_argument(
        "--reverify-json",
        default=_DEFAULT_REVERIFY,
        help="optional: upgrade baseline-vs-anchor pairs to these high-n matchups",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand, base = args.candidate_name, args.baseline
    anchors = tuple(a.strip() for a in args.anchors.split(",") if a.strip())
    extras = [o.strip() for o in args.extra_opponents.split(",") if o.strip()]
    # Gate opponents: baseline + un-gamed anchors + extras (dedup, order-preserving).
    opponents: list[str] = []
    for o in [base, *anchors, *extras]:
        if o not in opponents:
            opponents.append(o)
    out_path = Path(args.out or f"runs/ladder_gate_{cand}.json")

    validate_inputs(args.candidate_ckpt, opponents, args.ladder_json, args.reverify_json)
    print(f"[gate] candidate={cand} ({args.candidate_ckpt})", flush=True)
    print(f"[gate] baseline={base}  anchors={anchors}  opponents={opponents}", flush=True)
    print(f"[gate] n={2 * args.nps}/pair, seed base {args.seed_base}", flush=True)

    led = json.loads(Path(args.ladder_json).read_text())
    cached_pairs = {frozenset((m["a"], m["b"])) for m in led["matches"]}

    cand_tasks = [
        _task(cand, args.candidate_ckpt, opp, args.nps, args.seed_base + i)
        for i, opp in enumerate(opponents)
    ]
    # clause-1 (MIN-over-anchors pairwise-Elo delta) needs the BASELINE's WR vs each
    # un-gamed anchor. When the cached ladder predates the baseline rung — e.g. v9 is
    # gated against v10 BEFORE v9 was itself added to the transitive ladder — those
    # baseline-vs-anchor pairs are absent from the cache, so PLAY them here (else clause-1
    # has no baseline leg and spuriously fails). Chosen over reading a specific prior gate
    # JSON: no coupling to a run filename, fully general, and with >=len(tasks) workers the
    # extra pairs finish in the same wave (~no wall-clock cost).
    base_kind, base_path = _kind_and_path(base)
    baseline_anchor_tasks: list[dict[str, Any]] = []
    if base not in _ENGINE_KINDS:
        for j, anc in enumerate(anchors):
            if frozenset((base, anc)) in cached_pairs:
                continue
            bk, bp = _kind_and_path(anc)
            baseline_anchor_tasks.append(
                {
                    "a_name": base,
                    "a_kind": base_kind,
                    "a_path": base_path,
                    "a_sims": 0,
                    "b_name": anc,
                    "b_kind": bk,
                    "b_path": bp,
                    "nps": args.nps,
                    "seed": args.seed_base + len(cand_tasks) + j,
                }
            )
    tasks = [*cand_tasks, *baseline_anchor_tasks]
    if baseline_anchor_tasks:
        played = [t["b_name"] for t in baseline_anchor_tasks]
        print(
            f"[gate] baseline {base} absent from cached ladder — also playing "
            f"baseline-vs-anchor pair(s) {played} for clause-1",
            flush=True,
        )

    t0 = time.time()
    all_matches: list[tuple[str, str, int, int]] = []
    with Pool(min(args.workers, len(tasks))) as pool:
        for i, m in enumerate(pool.imap_unordered(elo.run_match, tasks), 1):
            all_matches.append(m)
            print(
                f"[gate] {i}/{len(tasks)} {m[0]} vs {m[1]}: {m[2]}/{m[3]} (WR {m[2] / m[3]:.4f})",
                flush=True,
            )
    dt = time.time() - t0
    # Belt-and-suspenders: a SIGKILL'd worker is the one path imap_unordered handles less
    # crisply than its re-raise of ordinary exceptions; never gate on partial matchups.
    if len(all_matches) != len(tasks):
        raise SystemExit(f"expected {len(tasks)} matchups, got {len(all_matches)} — a worker died")

    # run_match never swaps a/b, so m[0] is always the acting champion of its task.
    cand_matches = [m for m in all_matches if m[0] == cand]
    baseline_anchor_matches = [m for m in all_matches if m[0] == base]

    # Optional high-n upgrade of baseline-vs-anchor pairs from a reverify run.
    overrides: dict[frozenset[str], tuple[str, str, int, int]] = {}
    rj = Path(args.reverify_json)
    if rj.exists():
        for m in json.loads(rj.read_text()).get("high_n_matchups", []):
            if base in (m["a"], m["b"]) and (m["a"] in anchors or m["b"] in anchors):
                key = frozenset((m["a"], m["b"]))
                overrides[key] = (m["a"], m["b"], int(m["wins_a"]), int(m["n"]))
    if overrides:
        print(
            f"[gate] upgraded {len(overrides)} baseline-vs-anchor pair(s) to reverify high-n",
            flush=True,
        )

    merged = build_merged_matches(
        led["matches"], [*cand_matches, *baseline_anchor_matches], overrides=overrides
    )
    # Fit universe = cached names UNION every merged participant (covers the candidate AND a
    # baseline rung absent from a stale cached ladder — otherwise fit_elo KeyErrors).
    names = fit_name_universe(list(led["ratings"].keys()), merged)
    ratings = elo.fit_elo(merged, names)
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    resid, _detail, ev = elo.nontransitivity_residual(merged, ratings)
    boot = elo.bootstrap_elo_ci(merged, names, n_boot=2000, candidate=cand, baseline=base)
    gate = elo.promotion_check(
        merged, names, ratings, boot, candidate=cand, baseline=base, anchors=anchors
    )

    print("\n=== LADDER (candidate added, heuristic pinned @ 500) ===")
    for nm, r in ranked:
        ci = boot.get("elo_ci", {}).get(nm)
        cis = f"  CI [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
        print(f"  {r:7.1f}  {nm}{cis}")
    print(f"[gate] nontransitivity_residual={resid:.4f}  explained_variance={ev:.4f}")
    print(f"\n=== FOLD-BACK GATE ({cand} vs {base}) ===")
    print(f"  per_anchor_delta_elo : {gate['per_anchor_delta_elo']}")
    print(f"  min_anchor_delta_ci  : {gate['min_anchor_delta_ci']}  (clause-1 needs LB > 0)")
    print(f"  clause1_global_gain  : {gate['clause1_global_gain_vs_anchors']}")
    print(f"  clause2_non_regress  : {gate['clause2_non_regression']}  {gate['non_regression']}")
    print(f"  clause3_residual_ok  : {gate['clause3_residual_non_increase']}")
    verdict = (
        "FOLD IN — confirmed global gain over the champion (new champion)"
        if gate["passed"]
        else "DO NOT FOLD IN — not a robust global gain (lateral counter or underpowered)"
    )
    print(f"\n  GATE passed = {gate['passed']}  ->  {verdict}")

    out_path.write_text(
        json.dumps(
            {
                "candidate": cand,
                "candidate_ckpt": args.candidate_ckpt,
                "baseline": base,
                "anchors": list(anchors),
                "opponents": opponents,
                "candidate_matchups": [
                    {"a": m[0], "b": m[1], "wins_a": m[2], "n": m[3], "wr_a": m[2] / m[3]}
                    for m in cand_matches
                ],
                "baseline_anchor_matchups": [
                    {"a": m[0], "b": m[1], "wins_a": m[2], "n": m[3], "wr_a": m[2] / m[3]}
                    for m in baseline_anchor_matches
                ],
                "n_per_pair": 2 * args.nps,
                "seed_base": args.seed_base,
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
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
