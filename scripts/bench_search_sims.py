#!/usr/bin/env python3
"""CPU search sims/s micro-benchmark (pointer-arch fork acceptance gate AC-7).

Measures determinized-PUCT search throughput (simulations per second) on CPU
under a FIXED :class:`SearchConfig`, so the pointer-arch's location-head change
can be shown to stay within 10% of the v11 baseline. Run it on the baseline
branch (v11 arch) AND on this branch (pointer-arch) on the SAME machine/settings
and compare the printed ``sims_per_s``; a regression > 10% is a BLOCKER
regardless of any training metric.

Determinism / fairness knobs are pinned: CPU device, single thread of control,
fixed seed, fresh env, identical ``sims_per_move`` and move count. A freshly
constructed policy (no checkpoint needed) is used by default so the harness runs
in CI; pass ``--checkpoint`` to benchmark specific weights (the migrated v11).

Usage::

    python scripts/bench_search_sims.py --sims 100 --moves 20 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _build_policy(checkpoint: Path | None) -> object:
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["policy_state_dict"] if isinstance(payload, dict) else payload
        policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def run_benchmark(*, sims: int, moves: int, seed: int, checkpoint: Path | None) -> dict[str, float]:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    policy = _build_policy(checkpoint)
    cfg = SearchConfig(sims_per_move=sims, seed=seed)
    agent = SearchAgent(policy, cfg, device=device)  # type: ignore[arg-type]

    env = CatanEnv()
    env.reset(seed=seed)

    # Warm up one decision (JIT/allocator warmup) — not counted.
    with torch.no_grad():
        agent.choose_action(env)
    env.reset(seed=seed)

    total_sims = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(moves):
            action = agent.choose_action(env)
            total_sims += sims
            _obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                env.reset(seed=seed)
    elapsed = time.perf_counter() - t0
    sims_per_s = total_sims / elapsed if elapsed > 0 else 0.0
    return {"sims_per_s": sims_per_s, "total_sims": float(total_sims), "elapsed_s": elapsed}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sims", type=int, default=100, help="sims_per_move (fixed budget).")
    p.add_argument("--moves", type=int, default=20, help="Number of timed search decisions.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional policy weights.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    result = run_benchmark(
        sims=args.sims, moves=args.moves, seed=args.seed, checkpoint=args.checkpoint
    )
    print(
        f"sims_per_s={result['sims_per_s']:.2f} "
        f"total_sims={int(result['total_sims'])} elapsed_s={result['elapsed_s']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
