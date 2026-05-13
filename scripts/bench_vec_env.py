"""Benchmark serial vs subproc vec env modes.

Constructs a phase4_full-flagged GameManager / SubprocGameManager, runs N
steps with random actions, and reports steps/sec. The benchmark intentionally
uses random actions (no policy forward pass) so the result reflects the
env-stepping cost only — which is the bottleneck the SubprocVecEnv addresses.

Usage:
    python scripts/bench_vec_env.py --n-envs 8 --steps 256
    python scripts/bench_vec_env.py --n-envs 4 --steps 512 --modes serial,subproc

Output is a markdown-style table — paste into PRs / docs as-is.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from catan_rl.selfplay.vec_env_factory import make_vec_env


def _bench_one(mode: str, n_envs: int, steps: int) -> tuple[float, float]:
    """Run ``steps`` parallel-step iterations and return (elapsed_sec, fps)."""
    gm = make_vec_env(
        mode,
        n_envs=n_envs,
        opponent_type="random",
        max_turns=500,
        league=None,
        build_policy_fn=None,
        device="cpu",
        use_thermometer_encoding=False,  # phase4_full default
        use_opponent_id_emb=True,
        use_belief_head=True,
    )
    gm.reset_all()
    masks = gm.get_masks()
    rng = np.random.default_rng(0)

    def sample_action(mask_dict: dict) -> np.ndarray:
        # Pick any valid action type; corner/edge/tile/res default to 0.
        # The env masks-out invalid actions internally so any sampled value
        # is fine for the no-policy benchmark.
        valid_types = np.flatnonzero(mask_dict["type"])
        t = int(rng.choice(valid_types)) if len(valid_types) > 0 else 12
        return np.array([t, 0, 0, 0, 0, 0], dtype=np.int64)

    # Warmup — JIT/page cache priming, especially relevant for fork startup.
    for _ in range(2):
        actions = [sample_action(masks[i]) for i in range(n_envs)]
        out = gm.step_all(actions)
        masks = gm.get_masks()
        # Drive any deferred opponent turns to completion so the next step
        # is a clean main-agent step. We have no opponent NN here; with
        # opponent_type="random" the env's internal random opp acts inline.
        for _env_i, info in enumerate(out[4]):
            if info.get("opp_turn_pending"):
                # Should not happen with random opponent; defensive.
                pass

    t0 = time.time()
    for _ in range(steps):
        actions = [sample_action(masks[i]) for i in range(n_envs)]
        gm.step_all(actions)
        masks = gm.get_masks()
    elapsed = time.time() - t0
    gm.close()
    fps = (steps * n_envs) / elapsed if elapsed > 0 else float("inf")
    return elapsed, fps


def main() -> None:
    p = argparse.ArgumentParser(description="serial vs subproc vec env benchmark")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--modes", type=str, default="serial,subproc", help="comma-separated mode list")
    args = p.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"\n## bench_vec_env (n_envs={args.n_envs}, steps={args.steps})\n")
    print("| mode | wall (s) | env-steps/sec | speedup |")
    print("|---|---|---|---|")
    serial_fps: float | None = None
    for mode in modes:
        elapsed, fps = _bench_one(mode, args.n_envs, args.steps)
        if mode == "serial":
            serial_fps = fps
            speedup = "1.00×"  # noqa: RUF001
        elif serial_fps is not None:
            speedup = f"{fps / serial_fps:.2f}×"  # noqa: RUF001
        else:
            speedup = "—"
        print(f"| {mode} | {elapsed:.2f} | {fps:.1f} | {speedup} |")
    print()


if __name__ == "__main__":
    main()
