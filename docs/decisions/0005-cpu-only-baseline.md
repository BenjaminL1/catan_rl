# ADR 0005: CPU-Only Baseline on Apple M1 Pro

**Status:** Accepted
**Date:** 2026-04-30

## Context

Training runs on developer hardware: Apple M1 Pro (8-core CPU). MPS is documented to be ~9× slower than CPU at batch=1 inference for this 1.54M-param network (kernel launch overhead dominates). CUDA is unavailable.

## Decision

The reference training pipeline is **CPU-only**. CUDA is supported as an additive opt-in (`--device cuda`) but never required. CPU performance must not regress in any phase of the roadmap.

## Consequences

- Multi-env rollouts are **in-process** (sequential across `n_envs`). Subprocess parallelism via Python's `spawn` (default on macOS) has prohibitive overhead at this batch size.
- Opponent NN inference is batched across all `n_envs` envs sharing a single rollout opponent policy (`GameManager.rollout_opp_policy`).
- `torch.compile` is disabled by default — `mode="reduce-overhead"` requires CUDA graphs; `mode="default"` adds overhead at our batch sizes.
- Real-world throughput is ~30-35 steps/sec ≈ 2.5M steps/day.
- Architecture choices favor simplicity over raw FLOPs. Phase 2.3's GNN must not regress CPU FPS by more than 25% or it falls back to PPG.

## Alternatives Considered

- **CUDA-first.** Not viable on the user's hardware.
- **MPS.** Tested and rejected — slower than CPU at batch=1.
- **Distributed training.** Out of scope for solo development.

## Related

- `src/catan_rl/algorithms/ppo/trainer.py` (`device` selection)
- `src/catan_rl/selfplay/game_manager.py` (in-process multi-env)
- `docs/plans/superhuman_roadmap.md` §1.3 (non-goals)
