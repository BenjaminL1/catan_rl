"""Build-test the proposed training speedups without disturbing a live run.

For each configuration variant we:
  1. Construct the trainer (this is what crashes if a flag is broken).
  2. Run ``collect_rollouts`` for a small ``n_steps``, timed.
  3. Run one ``update`` to confirm the loss path is intact.
  4. Report a tentative FPS number — *with a strong caveat* that the live
     training run is competing for the CPU, so absolute numbers are
     pessimistic. Relative comparisons are still informative.

This script is throwaway — it exists to answer "does flag X actually
work on the current branch?" not to produce publication-quality
benchmarks.

Usage:
    python scripts/benchmark_speedups.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch

from catan_rl.algorithms.ppo.arguments import resolve_config
from catan_rl.algorithms.ppo.trainer import CatanPPO


@contextmanager
def timed(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"  ⏱ {label}: {dt:.2f}s")


def _smoke_config(base: str = "configs/phase4_full.yaml", **overrides) -> dict:
    """Build a minimal phase4 config — small rollout, single update."""
    cfg = resolve_config(base)
    cfg.update(
        n_steps=128,
        n_envs=overrides.pop("n_envs", 2),
        batch_size=64,
        n_epochs=1,
        total_timesteps=128,
        checkpoint_freq=10**9,
        eval_freq=10**9,
        log_dir="/tmp/bench_logs",
        checkpoint_dir="/tmp/bench_ckpts",
        device=overrides.pop("device", "cpu"),
        max_turns=200,
    )
    cfg.update(overrides)
    return cfg


def _bench(label: str, cfg_overrides: dict) -> tuple[bool, float | None]:
    """Build, run one rollout + one update; return (ok, fps_or_None)."""
    print(f"\n— {label} —")
    try:
        cfg = _smoke_config(**cfg_overrides)
        with timed("trainer init"):
            t = CatanPPO(cfg)
        n_params = sum(p.numel() for p in t.policy.parameters() if p.requires_grad)
        print(f"  params: {n_params:,}")
        t0 = time.time()
        with timed("collect_rollouts"):
            t.collect_rollouts()
        rollout_dt = time.time() - t0
        with timed("update"):
            stats = t.update()
        print(
            f"  loss: pg={stats['policy_loss']:+.4f} v={stats['value_loss']:.4f} "
            f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f}"
        )
        if "belief_loss" in stats:
            print(f"  belief_loss={stats['belief_loss']:.4f}")
        if "opp_action_loss" in stats:
            print(f"  opp_action_loss={stats['opp_action_loss']:.4f}")
        fps = cfg["n_steps"] / rollout_dt if rollout_dt > 0 else None
        return True, fps
    except Exception as e:
        import traceback

        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def main() -> int:
    print("=" * 70)
    print("Speedup build-tests for phase4_full (live training process is")
    print("running concurrently — FPS numbers are PESSIMISTIC).")
    print("=" * 70)

    results: list[tuple[str, bool, float | None]] = []

    # Tier 1.1 — torch.compile on
    ok, fps = _bench("phase4_full + torch_compile=True", {"torch_compile": True})
    results.append(("torch_compile=True", ok, fps))

    # Tier 1.2 — n_envs=16 (compared against n_envs=2 baseline)
    ok, fps = _bench("phase4_full + n_envs=16", {"n_envs": 16})
    results.append(("n_envs=16", ok, fps))

    # Tier 1.3 — leave-one-out: drop GNN (the heaviest Phase 4 add)
    ok, fps = _bench(
        "phase4_no_graph (drops GNN)",
        {"base": "configs/phase4_no_graph.yaml"},
    )
    results.append(("phase4_no_graph", ok, fps))

    # Tier 1.4 — leave-one-out: drop recurrent value
    ok, fps = _bench(
        "phase4_no_recurrent_value",
        {"base": "configs/phase4_no_recurrent_value.yaml"},
    )
    results.append(("phase4_no_recurrent_value", ok, fps))

    # Tier 1.5 — phase3_full (no Phase 4 features at all)
    ok, fps = _bench(
        "phase3_full (skip Phase 4 entirely)",
        {"base": "configs/phase3_full.yaml"},
    )
    results.append(("phase3_full", ok, fps))

    # Tier 3.1 — MPS (Apple GPU) availability + build
    if torch.backends.mps.is_available():
        ok, fps = _bench("phase4_full on MPS", {"device": "mps"})
        results.append(("device=mps", ok, fps))
    else:
        print("\n— device=mps —\n  ✗ MPS unavailable on this machine")
        results.append(("device=mps", False, None))

    # Tier 3.2 — CUDA availability + build
    if torch.cuda.is_available():
        ok, fps = _bench("phase4_full on CUDA", {"device": "cuda"})
        results.append(("device=cuda", ok, fps))
    else:
        print("\n— device=cuda —\n  · CUDA not available (expected on M1)")
        results.append(("device=cuda", None, None))  # type: ignore[arg-type]

    # ── Reference baseline (phase4_full default flags) ─────────────────────
    ok, fps = _bench("phase4_full BASELINE (no overrides)", {})
    results.append(("BASELINE", ok, fps))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'config':<32} {'build':<8} {'rollout FPS (smoke)':<22}")
    for label, ok, fps in results:
        if ok is None:
            print(f"{label:<32} {'n/a':<8} -")
        elif ok:
            fps_str = f"{fps:.1f}" if fps is not None else "—"
            print(f"{label:<32} {'OK':<8} {fps_str:<22}")
        else:
            print(f"{label:<32} {'FAIL':<8} -")
    print("=" * 70)
    print(
        "\nNote: smoke FPS is from 128-step rollouts at n_envs=2 (or 16 where\n"
        "noted), with the live phase4 training process competing for CPU.\n"
        "Trends between rows are informative; absolute numbers are not.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
