"""Elo / Bradley-Terry strength ladder over on-disk checkpoints (+ a search rung).

Round-robin via the wired eval primitives (no training):
  - checkpoint-vs-checkpoint: ``eval.harness.evaluate_policy_vs_policy``
  - checkpoint-vs-{heuristic,random}: ``eval.harness.EvalHarness``
  - search-vs-X (US3): ``search.eval_search`` (``evaluate_search_vs_policy`` /
    ``run_search_matchup``) — pass ``--search-ckpt`` to add a determinized-search
    rung at ``--search-sims`` simulations/move.
Fits a logistic-Elo MLE (Bradley-Terry, base-10 / 400) with the HEURISTIC pinned
at 500 so every rung is an interpretable offset from it. Parallel across cores.

This script writes the round-robin ladder to ``--out`` (default ``runs/elo_ladder.json``).
For the 003 search uplift we do NOT run a fresh multi-hour search round-robin — the
n=500 bake-off already measured search-vs-raw-v6; the reusable :func:`elo_from_wr`
converter is what the separate uplift step imports to populate
``runs/search/elo_uplift.json`` from that WR + its Wilson CI.

Usage:
  python scripts/elo_ladder.py --nps 100                       # full policy ladder
  python scripts/elo_ladder.py --smoke                         # 2 rungs, 3 games
  python scripts/elo_ladder.py --search-ckpt <ckpt> --search-sims 50 --nps 25
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

V6 = "runs/train/selfplay_v6_20260611_065459/checkpoints"
V5 = "runs/train/selfplay_v5_20260610_022256/checkpoints"
V3B = "runs/train/selfplay_v3_20260609_140731/best"
BOOT = "runs/train/bootstrap_v1_20260607_233931/checkpoints"
V7_FINAL = "runs/anchors/v7_final_u399.pt"
V8 = "runs/anchors/v8_promobar_u243.pt"
V8CONT = "runs/train/selfplay_v8_cont_20260618_014258/checkpoints/ckpt_000000524.pt"

# (name, kind, path)  — kind: "policy" | "heuristic" | "random"
# Frontier (v7/v8/ckpt_524) on top so the transitive ruler covers where the
# intransitivity lives. ckpt_524 is the intransitive v8-counter — a CORRECT ruler
# must rank it ~= v8 (NOT above) and fail its promotion check (the built-in validity test).
RUNGS: list[tuple[str, str, str | None]] = [
    ("ckpt_524", "policy", V8CONT),
    ("v8_u243", "policy", V8),
    ("v7_u399", "policy", V7_FINAL),
    ("v6_u1499", "policy", f"{V6}/ckpt_000001499.pt"),
    ("v6_u1449", "policy", f"{V6}/ckpt_000001449.pt"),
    ("v6_u1399", "policy", f"{V6}/ckpt_000001399.pt"),
    ("v5_u849", "policy", f"{V5}/ckpt_000000849.pt"),
    ("v3_u299", "policy", f"{V3B}/ckpt_u299_strength0.77.pt"),
    ("bootstrap_u799", "policy", f"{BOOT}/ckpt_000000799.pt"),
    ("heuristic", "heuristic", None),
    ("random", "random", None),
]
# Redundant intra-v6 micro-rungs dropped from the DEFAULT frontier set (kept behind
# --full so old behavior stays reproducible); v6_u1399 (ladder top / search base) and
# v6_u1499 (the +54.6-uplift base) are both retained.
_FRONTIER_DROP = {"v6_u1449"}
PIN_NAME, PIN_VALUE, SCALE = "heuristic", 500.0, 400.0


def elo_from_wr(wr: float, scale: float = SCALE) -> float:
    """Pairwise Elo delta implied by a win-rate (clamped away from 0/1)."""
    wr = min(max(wr, 1e-6), 1 - 1e-6)
    return scale * math.log10(wr / (1.0 - wr))


def bt_prob(ra: float, rb: float, scale: float = SCALE) -> float:
    """Bradley-Terry P(a beats b), clamped away from 0/1. Single source of the
    win-prob formula reused by the fit, the residual, the matrices, and the bootstrap."""
    p = 1.0 / (1.0 + 10 ** ((rb - ra) / scale))
    return min(max(p, 1e-12), 1 - 1e-12)


def run_match(task: dict[str, Any]) -> tuple[str, str, int, int]:
    """One matchup a-vs-b. Returns (a_name, b_name, wins_a, n) — integer wins."""
    from catan_rl.eval.harness import EvalHarness, evaluate_policy_vs_policy
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor

    a_name, a_kind, a_path = task["a_name"], task["a_kind"], task["a_path"]
    b_name, b_kind, b_path = task["b_name"], task["b_kind"], task["b_path"]
    nps, seed = task["nps"], task["seed"]

    if a_kind == "search":
        from catan_rl.search.agent import SearchAgent
        from catan_rl.search.config import SearchConfig
        from catan_rl.search.eval_search import evaluate_search_vs_policy, run_search_matchup

        # seed=0 fixed: search determinization RNG pinned for FR-006 reproducibility;
        # per-matchup env/game seeds still vary via `seed`.
        cfg = SearchConfig(sims_per_move=task["a_sims"], seed=0)
        if b_kind == "policy":
            res = evaluate_search_vs_policy(
                cfg, a_path, b_path, n_games=2 * nps, seed=seed, device="cpu"
            )
            return (a_name, b_name, int(res.wins), int(res.n))
        actor = build_actor(PlayerSpec(kind="policy", ckpt_path=a_path), seed=seed, device="cpu")
        assert isinstance(actor, _PolicyActor)
        agent = SearchAgent(actor.policy, cfg, device=actor.device)
        r = run_search_matchup(  # opponent=None -> engine drives the heuristic/random body
            agent,
            opponent_type=b_kind,
            opponent=None,
            n_games=2 * nps,
            seed=seed,
            opponent_ref=b_kind,
        )
        return (a_name, b_name, int(r.wins), int(r.n))

    actor = build_actor(PlayerSpec(kind="policy", ckpt_path=a_path), seed=0, device="cpu")
    assert isinstance(actor, _PolicyActor)
    champ = actor.policy
    if b_kind == "policy":
        res = evaluate_policy_vs_policy(champ, b_path, n_games=2 * nps, seed=seed, device="cpu")
        return (a_name, b_name, int(res.wins), int(res.n))
    harness = EvalHarness(opponent_types=(b_kind,), n_games_per_seat=nps, seed=seed, device="cpu")
    r2 = harness.run(champ).results[0]
    return (a_name, b_name, int(r2.wins), int(r2.n))


def fit_elo(matches: list[tuple[str, str, int, int]], names: list[str]) -> dict[str, float]:
    idx = {n: i for i, n in enumerate(names)}
    pin = idx[PIN_NAME]
    free = [i for i in range(len(names)) if i != pin]

    def nll(theta: np.ndarray) -> float:
        r = np.zeros(len(names))
        for k, i in enumerate(free):
            r[i] = theta[k]
        ll = 0.0
        for a, b, wins, n in matches:
            ia, ib = idx[a], idx[b]
            wa = float(wins)
            wb = float(n - wins)
            pa = bt_prob(r[ia], r[ib])
            ll += wa * math.log(pa) + wb * math.log(1 - pa)
        # Tiny ridge toward the pin: finitizes flat directions from saturated 0%/100%
        # pairs (e.g. a strong rung vs random) without moving ratings on the connected
        # ladder (>4 sig figs unchanged); keeps L-BFGS-B off the rails.
        ridge = 1e-6 * float(sum((r[i] - PIN_VALUE) ** 2 for i in free))
        return -ll + ridge

    res = minimize(nll, np.zeros(len(free)), method="L-BFGS-B")
    r = np.zeros(len(names))
    for k, i in enumerate(free):
        r[i] = res.x[k]
    r = r - r[pin] + PIN_VALUE
    return {names[i]: float(r[i]) for i in range(len(names))}


def nontransitivity_residual(
    matches: list[tuple[str, str, int, int]], ratings: dict[str, float]
) -> tuple[float, list[dict[str, Any]], float]:
    """Game-weighted mean |observed_WR - BT_predicted_WR| (a rock-paper-scissors
    detector) + per-pair detail (sorted by |resid|) + the BT explained-variance.
    Noise floor ~0.020 at n=400; R>0.05 AND some per-pair |resid|>0.10 flags a real
    intransitive triangle (transitive truth has only binomial noise)."""
    num = den = ss_res = ss_tot = 0.0
    detail: list[dict[str, Any]] = []
    for a, b, wins, n in matches:
        w_obs = wins / n
        p_hat = bt_prob(ratings[a], ratings[b])
        resid = w_obs - p_hat
        num += n * abs(resid)
        den += n
        ss_res += n * resid * resid
        ss_tot += n * (w_obs - 0.5) ** 2
        detail.append({"a": a, "b": b, "w_obs": w_obs, "p_hat": p_hat, "resid": resid, "n": n})
    r = num / den if den else 0.0
    ev = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    detail.sort(key=lambda d: -abs(float(d["resid"])))
    return r, detail, ev


def pairwise_matrices(
    matches: list[tuple[str, str, int, int]], names: list[str], ratings: dict[str, float]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Observed WR matrix (mirror-filled wr[b][a]=1-wr[a][b]), BT-predicted matrix,
    and the signed residual matrix (observed - predicted). An A>B>C>A cycle shows as a
    coherent sign pattern in the residual matrix."""
    wr: dict[str, dict[str, float]] = {nm: {} for nm in names}
    pred: dict[str, dict[str, float]] = {nm: {} for nm in names}
    resid: dict[str, dict[str, float]] = {nm: {} for nm in names}
    for a, b, wins, n in matches:
        w = wins / n
        wr[a][b], wr[b][a] = w, 1 - w
        pred[a][b], pred[b][a] = bt_prob(ratings[a], ratings[b]), bt_prob(ratings[b], ratings[a])
        resid[a][b] = w - pred[a][b]
        resid[b][a] = (1 - w) - pred[b][a]
    return wr, pred, resid


def bootstrap_elo_ci(
    matches: list[tuple[str, str, int, int]],
    names: list[str],
    *,
    n_boot: int = 1000,
    rng_seed: int = 0,
    candidate: str | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Parametric Binomial bootstrap of the BT fit (games iid + seat-balanced within a
    pair). Returns per-rung 95% percentile Elo CIs and — if candidate+baseline given —
    the PAIRED delta distribution (candidate - baseline on the SAME resample, which
    preserves shared-opponent correlation; independent marginals overstate variance)."""
    rng = np.random.default_rng(rng_seed)
    samples: dict[str, list[float]] = {nm: [] for nm in names}
    deltas: list[float] = []
    for _ in range(n_boot):
        resampled = [(a, b, int(rng.binomial(n, wins / n)), n) for a, b, wins, n in matches]
        rb = fit_elo(resampled, names)
        for nm in names:
            samples[nm].append(rb[nm])
        if candidate is not None and baseline is not None:
            deltas.append(rb[candidate] - rb[baseline])
    cis = {
        nm: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for nm, v in samples.items()
    }
    out: dict[str, Any] = {"elo_ci": cis}
    if deltas:
        out["delta_ci"] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        out["delta_frac_positive"] = float(np.mean([d > 0 for d in deltas]))
    return out


def _candidate_wr_vs(
    matches: list[tuple[str, str, int, int]], candidate: str, opponent: str
) -> tuple[int, int] | None:
    """(wins_for_candidate, n) for the candidate-vs-opponent pair (either orientation)."""
    for a, b, wins, n in matches:
        if a == candidate and b == opponent:
            return wins, n
        if a == opponent and b == candidate:
            return n - wins, n
    return None


def promotion_check(
    matches: list[tuple[str, str, int, int]],
    names: list[str],
    ratings: dict[str, float],
    boot: dict[str, Any],
    *,
    candidate: str,
    baseline: str = "v8_u243",
    anchors: tuple[str, ...] = ("v6_u1399", "v7_u399"),
    search_anchor: str | None = None,
) -> dict[str, Any]:
    """3-clause gate that a candidate is GLOBALLY better than ``baseline`` (defeats the
    intransitive trap): (1) paired-bootstrap joint-Elo delta CI strictly > 0; (2)
    non-regression vs un-gamed anchors (Wilson LB > 0.45 each); (3) non-transitivity
    residual does not increase (<= without-candidate R + 0.01)."""
    from catan_rl.eval.wilson import wilson_interval

    out: dict[str, Any] = {"candidate": candidate, "baseline": baseline}
    delta_ci = boot.get("delta_ci")
    clause1 = bool(delta_ci is not None and delta_ci[0] > 0.0)
    out["joint_elo_delta_ci"] = delta_ci
    out["clause1_joint_elo_gain"] = clause1

    reg: dict[str, Any] = {}
    clause2 = True
    for anc in [*anchors, *([search_anchor] if search_anchor else [])]:
        wn = _candidate_wr_vs(matches, candidate, anc)
        if wn is None:
            continue
        wins, n = wn
        lb = wilson_interval(wins=wins, n=n).lower
        ok = lb > 0.45
        reg[anc] = {"wr": wins / n, "wilson_lb": lb, "ok": ok}
        clause2 = clause2 and ok
    out["non_regression"] = reg
    out["clause2_non_regression"] = clause2

    r_with, _, _ = nontransitivity_residual(matches, ratings)
    matches_wo = [m for m in matches if candidate not in (m[0], m[1])]
    names_wo = [nm for nm in names if nm != candidate]
    r_wo = (
        nontransitivity_residual(matches_wo, fit_elo(matches_wo, names_wo))[0]
        if matches_wo and len(names_wo) >= 2
        else r_with
    )
    clause3 = r_with <= r_wo + 0.01
    out["residual_with"], out["residual_without"] = r_with, r_wo
    out["clause3_residual_non_increase"] = clause3
    out["passed"] = bool(clause1 and clause2 and clause3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nps", type=int, default=100, help="n_games_per_seat (total/pair = 2*nps)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--search-ckpt", default=None, help="add a determinized-search referee rung")
    ap.add_argument("--search-sims", type=int, default=50)
    ap.add_argument(
        "--full", action="store_true", help="include the redundant intra-v6 micro-rungs"
    )
    ap.add_argument(
        "--bootstrap-B", type=int, default=1000, help="bootstrap resamples for Elo CIs (0=off)"
    )
    ap.add_argument(
        "--gate-candidate", default="ckpt_524", help="rung to run the v8 promotion check on"
    )
    ap.add_argument("--out", default="runs/elo_ladder.json")
    args = ap.parse_args()

    if args.smoke:
        rungs: list[tuple[str, str, str | None]] = [RUNGS[0], RUNGS[1], RUNGS[-2]]
    elif args.full:
        rungs = list(RUNGS)
    else:
        rungs = [r for r in RUNGS if r[0] not in _FRONTIER_DROP]
    nps = 3 if args.smoke else args.nps
    # Optional search rung (a_kind="search"); carries its own sims via search_sims map.
    search_sims: dict[str, int] = {}
    if args.search_ckpt is not None:
        name = f"search@{args.search_sims}"
        rungs = [(name, "search", args.search_ckpt), *rungs]
        search_sims[name] = args.search_sims

    names = [r[0] for r in rungs]
    tasks: list[dict[str, Any]] = []
    seed = 0
    for (an, ak, ap_), (bn, bk, bp) in itertools.combinations(rungs, 2):
        # Only "policy"/"search" can be the acting champion (a); engine-driven
        # kinds (heuristic/random) only ever appear as opponent b.
        if ak not in ("policy", "search"):
            an, ak, ap_, bn, bk, bp = bn, bk, bp, an, ak, ap_
        if ak not in ("policy", "search"):
            continue  # heuristic-vs-random: skip (no champion)
        tasks.append(
            {
                "a_name": an,
                "a_kind": ak,
                "a_path": ap_,
                "a_sims": search_sims.get(an, 0),
                "b_name": bn,
                "b_kind": bk,
                "b_path": bp,
                "nps": nps,
                "seed": seed,
            }
        )
        seed += 1

    print(f"[elo] {len(rungs)} rungs, {len(tasks)} matchups, {nps} games/seat", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        results: list[tuple[str, str, int, int]] = []
        for i, m in enumerate(pool.imap_unordered(run_match, tasks), 1):
            results.append(m)
            print(f"[elo] {i}/{len(tasks)} {m[0]} vs {m[1]}: {m[2]}/{m[3]} wins", flush=True)
    dt = time.time() - t0

    ratings = fit_elo(results, names)
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    print("\n=== ELO LADDER (heuristic pinned @ 500) ===")
    for n, r in ranked:
        print(f"  {r:7.1f}  {n}")

    # --- transitive-ruler diagnostics (additive) ---
    resid_scalar, resid_detail, ev = nontransitivity_residual(results, ratings)
    wr_mat, pred_mat, resid_mat = pairwise_matrices(results, names, ratings)
    rps = resid_scalar > 0.05 and any(abs(float(d["resid"])) > 0.10 for d in resid_detail)
    print(
        f"[elo] nontransitivity_residual={resid_scalar:.4f}  explained_variance={ev:.4f}  "
        f"rock-paper-scissors_flag={rps}"
    )

    # Search-referee calibration tripwire: the un-gameable axis must reproduce the
    # banked +54.6 [23.9,85.4] search uplift over raw v6, else the referee regressed.
    search_rung = next((nm for nm in names if nm.startswith("search@")), None)
    if search_rung is not None and "v6_u1399" in ratings:
        joint_delta = ratings[search_rung] - ratings["v6_u1399"]
        if not (20.0 <= joint_delta <= 90.0):
            print(
                f"[elo] WARNING: search referee Elo over v6_u1399 = {joint_delta:.1f} "
                f"outside [20,90] — referee axis may have regressed (banked ~+55)."
            )
        else:
            print(f"[elo] search-referee calibration OK: +{joint_delta:.1f} Elo over v6_u1399")

    # Paired bootstrap CIs + the v8 promotion gate on the candidate.
    boot: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    cand, base = args.gate_candidate, "v8_u243"
    if args.bootstrap_B > 0 and cand in names and base in names:
        boot = bootstrap_elo_ci(
            results, names, n_boot=args.bootstrap_B, candidate=cand, baseline=base
        )
        gate = promotion_check(
            results,
            names,
            ratings,
            boot,
            candidate=cand,
            baseline=base,
            search_anchor=search_rung,
        )
        print(
            f"[elo] promotion_check({cand} vs {base}): passed={gate['passed']} "
            f"(joint_elo_gain={gate['clause1_joint_elo_gain']}, "
            f"non_regression={gate['clause2_non_regression']}, "
            f"residual_ok={gate['clause3_residual_non_increase']})"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "ratings": ratings,
                "ranked": ranked,
                "matches": [
                    {"a": a, "b": b, "wins_a": w, "wr_a": w / n, "n": n} for a, b, w, n in results
                ],
                "pin": {PIN_NAME: PIN_VALUE},
                "nps": nps,
                "seconds": dt,
                "nontransitivity_residual": resid_scalar,
                "explained_variance": ev,
                "rock_paper_scissors_flag": rps,
                "residual_per_pair": resid_detail,
                "wr_matrix": wr_mat,
                "pred_matrix": pred_mat,
                "residual_matrix": resid_mat,
                "elo_ci": boot.get("elo_ci", {}),
                "gate": gate,
            },
            indent=2,
        )
    )
    print(f"[elo] wrote {args.out}")


if __name__ == "__main__":
    main()
