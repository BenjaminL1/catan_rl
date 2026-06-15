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

# (name, kind, path)  — kind: "policy" | "heuristic" | "random"
RUNGS: list[tuple[str, str, str | None]] = [
    ("v6_u1499", "policy", f"{V6}/ckpt_000001499.pt"),
    ("v6_u1449", "policy", f"{V6}/ckpt_000001449.pt"),
    ("v6_u1399", "policy", f"{V6}/ckpt_000001399.pt"),
    ("v5_u849", "policy", f"{V5}/ckpt_000000849.pt"),
    ("v3_u299", "policy", f"{V3B}/ckpt_u299_strength0.77.pt"),
    ("bootstrap_u799", "policy", f"{BOOT}/ckpt_000000799.pt"),
    ("heuristic", "heuristic", None),
    ("random", "random", None),
]
PIN_NAME, PIN_VALUE, SCALE = "heuristic", 500.0, 400.0


def elo_from_wr(wr: float, scale: float = SCALE) -> float:
    """Pairwise Elo delta implied by a win-rate (clamped away from 0/1)."""
    wr = min(max(wr, 1e-6), 1 - 1e-6)
    return scale * math.log10(wr / (1.0 - wr))


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
            pa = 1.0 / (1.0 + 10 ** ((r[ib] - r[ia]) / SCALE))
            pa = min(max(pa, 1e-12), 1 - 1e-12)
            ll += wa * math.log(pa) + wb * math.log(1 - pa)
        return -ll

    res = minimize(nll, np.zeros(len(free)), method="L-BFGS-B")
    r = np.zeros(len(names))
    for k, i in enumerate(free):
        r[i] = res.x[k]
    r = r - r[pin] + PIN_VALUE
    return {names[i]: float(r[i]) for i in range(len(names))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nps", type=int, default=100, help="n_games_per_seat (total/pair = 2*nps)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--search-ckpt", default=None, help="add a determinized-search rung")
    ap.add_argument("--search-sims", type=int, default=50)
    ap.add_argument("--out", default="runs/elo_ladder.json")
    args = ap.parse_args()

    rungs: list[tuple[str, str, str | None]] = (
        [RUNGS[0], RUNGS[1], RUNGS[-2]] if args.smoke else list(RUNGS)
    )
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
            },
            indent=2,
        )
    )
    print(f"[elo] wrote {args.out}")


if __name__ == "__main__":
    main()
