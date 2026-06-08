# Quickstart / Validation: self-play snapshot opponent

Runnable scenarios that prove the feature works end to end. Each maps to a
success criterion (SC) and is the acceptance gate for its increment. Implement
the test FIRST (TDD), then make it pass.

## Prerequisites

- A working tree on the feature branch with the four increments implemented.
- A small policy checkpoint for the eval scenario (any `bootstrap_v1` checkpoint).

## Scenario 1 — Snapshot opponent is actually used (SC-002, US1)

Unit/integration test:
1. Build a league with one snapshot whose policy is a deterministic stub that
   always selects `EndTurn`.
2. Configure `snapshot_weight=1.0`; run a short rollout.
3. **Expect**: rollout completes with no `NotImplementedError`; every opponent
   action observed is `EndTurn` (the stub), proving the heuristic is bypassed.

## Scenario 2 — Self-play smoke (SC-001)

Integration test (short, CPU ok):
1. `snapshot_weight=0.5`, seed the league with one real snapshot.
2. Run ~3 PPO updates.
3. **Expect**: no `NotImplementedError`, no device errors; training advances.

## Scenario 3 — Fresh snapshot enters play (US3)

1. Start with a one-snapshot league; run one rollout.
2. `add_snapshot(new)`, then `vec_env.set_opponents(build_env_opponent_mix(...))`.
3. **Expect**: the next rollout's opponent assignment includes the new
   snapshot id.

## Scenario 4 — Policy-vs-policy eval (SC-004)

1. `evaluate_policy_vs_policy(champion, opponent_ref=<ckpt>, n_games=100, seed=0,
   device="cpu")`.
2. Run it twice at the same seed.
3. **Expect**: a finite WR + Wilson 95% CI; the two runs are bit-for-bit
   identical; a TB scalar `eval/wr_vs_<opp>` is written.

## Scenario 5 — Empty-pool fallback (FR-011)

1. `snapshot_weight=0.5` with an **empty** league.
2. Run a rollout.
3. **Expect**: no error; opponents are heuristic/random only.

## Checkpoint-compat guard (Constitution III)

After the change, load an existing `bootstrap_v1` checkpoint into the policy.
**Expect**: loads cleanly (no shape mismatch) — the `_OppIdEmbedding` was fed
real values, not resized.

## Out of scope (do NOT validate here)

PG-1 (self-play equilibrium / strength rise) and PG-2 (phase advancement gate)
require full training runs and later-phase machinery — they are downstream gates,
not acceptance tests for this PR.
