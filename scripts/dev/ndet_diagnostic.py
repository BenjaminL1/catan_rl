"""Fixed-budget n-determinization diagnostic (spec 008 STAGE-A, FR-003).

Sweeps ``n_determinizations`` K∈{2,4,8} at a FIXED total sim budget (the
``split_sims_across_determinizations`` budget split — each of the K trees runs
``sims//K`` sims, total ~``sims``), so every K spends the SAME leaf-eval budget
(the matched-total control). For each K it records, on a small both-seats game
sample vs the frozen v8 reference:

  * **per-K win-rate** (Wilson-free point WR — this is a diagnostic, not a gate);
  * **per-depth visit-concentration** — the fraction of decision nodes at depths
    0/1/2 whose visits collapse >50% onto one action (``collect_depth_stats``),
    averaged over every non-forced agent decision in the sample;
  * **Spearman(root-child value, ex-post terminal outcome)** — rank correlation
    between the backed-up mean value of the chosen root child at each decision and
    whether that game was eventually won. Low Spearman => the root value is too
    noisy for completed-Q (the STAGE-B max-Q-fallback trigger, spec US1.2).

Diagnostic only: no gate, no config/obs/engine change, CPU, fixed seeds, additive
new file. The full run is the human's (``--n-games 50``); ``--n-games 4`` is the
smoke used to prove the readout is well-formed.

Usage (smoke)::

    python scripts/dev/ndet_diagnostic.py --sims 32 --n-games 4 \
        --out runs/search/ndet_diagnostic_smoke.json

Usage (full — run by the human)::

    python scripts/dev/ndet_diagnostic.py --sims 96 --n-games 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.search.agent import SearchAgent

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"
K_SWEEP: tuple[int, ...] = (2, 4, 8)
_SEED_STRIDE = 1_000_003


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (manual, dependency-free).

    Ranks (average-tie) each series, then Pearson on the ranks. Returns 0.0 for
    <2 points or a degenerate (zero-variance) series.
    """
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank (1-indexed) for the tie block
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return cov / (vx**0.5 * vy**0.5)


def _play_collecting(
    env: CatanEnv, agent: SearchAgent, *, seed: int, agent_seat: int
) -> tuple[list[dict[str, Any]], bool]:
    """Play one game; collect per-decision root-child value + per-depth collapse."""
    from catan_rl.search.node import agent_outcome

    env.reset(seed=seed, options={"agent_seat": agent_seat})
    decisions: list[dict[str, Any]] = []
    terminated = truncated = False
    n_steps = 0
    cap = env.max_turns * 50
    while not terminated and not truncated:
        action = agent.choose_action(env)
        diag = agent.last_diagnostics
        if not diag.get("forced", False) and diag.get("visit_counts"):
            chosen = tuple(int(v) for v in action.tolist())
            aq: dict[tuple[int, ...], float] = diag.get("action_q", {})
            pdc = diag.get("per_depth_concentration")
            if chosen in aq and pdc is not None:
                decisions.append({"q": float(aq[chosen]), "pdc": pdc})
        _obs, _r, terminated, truncated, _ = env.step(action)
        n_steps += 1
        if n_steps > cap:
            truncated = True
            break
    won = bool(agent_outcome(env)) and not truncated
    return decisions, won


def _run_for_k(
    k: int, *, sims: int, n_games: int, seed: int, device: torch.device
) -> dict[str, Any]:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = actor.policy
    reference = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(reference)

    cfg = SearchConfig(
        sims_per_move=sims,
        seed=seed,
        n_determinizations=k,
        split_sims_across_determinizations=True,
        collect_depth_stats=True,
    )
    agent = SearchAgent(policy, cfg, device=device)

    from catan_rl.search.sprt import config_total_sim_budget

    wins = 0
    per_depth_collapse: dict[int, list[float]] = {0: [], 1: [], 2: []}
    q_series: list[float] = []
    win_series: list[float] = []
    base = (seed * _SEED_STRIDE) % (2**31 - 1)
    for g in range(n_games):
        agent_seat = g % 2
        reference.reset_rng(seed=(base + g) % (2**31 - 1))
        decisions, won = _play_collecting(
            env, agent, seed=(base + g) % (2**31 - 1), agent_seat=agent_seat
        )
        wins += int(won)
        for d in decisions:
            for depth in (0, 1, 2):
                m = d["pdc"].get(depth)
                if m is not None and m["n_nodes"] > 0:
                    per_depth_collapse[depth].append(m["collapse_frac"])
            q_series.append(d["q"])
            win_series.append(1.0 if won else 0.0)
    env.close()

    return {
        "k": k,
        "sims_per_tree": max(1, sims // k),
        "total_sim_budget": config_total_sim_budget(cfg),
        "n_games": n_games,
        "wins": wins,
        "win_rate": wins / n_games if n_games else 0.0,
        "n_decisions": len(q_series),
        "per_depth_collapse_frac": {
            str(depth): (sum(v) / len(v) if v else None) for depth, v in per_depth_collapse.items()
        },
        "root_child_value_spearman": _spearman(q_series, win_series),
    }


def run(*, sims: int, n_games: int, seed: int, out_path: Path) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    t0 = time.time()
    per_k = []
    for k in K_SWEEP:
        res = _run_for_k(k, sims=sims, n_games=n_games, seed=seed, device=device)
        per_k.append(res)
        print(
            f"[ndet] K={k} sims/tree={res['sims_per_tree']} total={res['total_sim_budget']} "
            f"WR={res['win_rate']:.3f} depth0_collapse={res['per_depth_collapse_frac']['0']} "
            f"spearman={res['root_child_value_spearman']:+.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    result = {
        "diagnostic": "fixed-budget n-det sweep (spec 008 STAGE-A, FR-003)",
        "ckpt": V8_CKPT,
        "config": {"sims_total": sims, "n_games": n_games, "seed": seed, "k_sweep": list(K_SWEEP)},
        "per_k": per_k,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ndet] wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=96, help="total sim budget/move (split by K)")
    parser.add_argument("--n-games", type=int, default=50, help="games/K (50 full; 4 smoke)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/search/ndet_diagnostic.json"))
    args = parser.parse_args(argv)
    run(sims=args.sims, n_games=args.n_games, seed=args.seed, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
