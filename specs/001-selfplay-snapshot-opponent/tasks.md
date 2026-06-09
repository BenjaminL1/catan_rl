# Tasks: Frozen-policy self-play opponent + policy-vs-policy eval

**Feature**: `specs/001-selfplay-snapshot-opponent/` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

Re-scoped 2026-06-08 after the senior-RL review + clarify pass. Decisions: full
opponent sub-turn state machine; shared `_apply_action`; stochastic opponent
sampling with an isolated `torch.Generator`; main turn run-to-completion + hard
action cap. TDD required (Constitution IV): write the listed tests first, watch
them fail, implement to green. `[P]` = parallelizable.

## Phase 1: Setup

- [ ] T001 Work on local branch `feat/selfplay-snapshot-opponent`; merge to main + push when green (no PR)
- [ ] T002 [P] Stub policies in `tests/conftest.py`: (a) always-EndTurn, (b) build/buy-then-end (exercises the apply path), (c) never-EndTurn (adversarial, for the cap test), each a `CatanPolicy`-compatible `sample(obs,masks)`

## Phase 2: Foundational (blocking — shared by US1/US2/US3)

- [ ] T003 Extract a shared **`_apply_action(player, action, ...)`** helper from `CatanEnv.step()` (the ~100-line inlined apply block), used by both agent and opponent — single code path, no rules drift, in `src/catan_rl/env/catan_env.py`
- [ ] T004 [P] Behavior-identical regression test: the agent path through the refactored `_apply_action` produces identical transitions to pre-refactor (golden trace), in `tests/unit/env/test_apply_action_parity.py`
- [ ] T005 Add an **opponent-POV obs+mask builder** — parameterize `_get_obs`/`get_action_masks` (and an opponent-local `EnvObsState`) by acting player, reusing the existing `agent_seat` machinery, in `src/catan_rl/env/catan_env.py`
- [ ] T006 [P] **POV no-leak test (correctness-critical, FR-012)**: opponent-POV obs shows the opponent's OWN hidden dev cards and only the agent's *played* cards; the agent's hidden info NEVER appears in the opponent's obs, in `tests/unit/env/test_opponent_pov_obs.py`
- [ ] T007 Define `OpponentAssignment` (kind + snapshot_id) in `src/catan_rl/selfplay/league.py`
- [ ] T008 Add the isolated-RNG, batched, inference-only frozen-opponent helper (load snapshot → `CatanPolicy.eval()`, cached per id, `sample` under `no_grad` with a dedicated `torch.Generator`) in `src/catan_rl/selfplay/snapshot_opponent.py`
- [ ] T009 Feed the existing `_OppIdEmbedding` real opponent kind/id WITHOUT resizing (FR-008), in the opp-id wiring

## Phase 3: User Story 1 — Train against a frozen past self (P1) 🟢 MVP

**Goal**: a snapshot plays a *full* Catan turn via its policy; self-play runs with no error and no info leak.

Tests:
- [ ] T010 [P] [US1] Test: `build_env_opponent_mix` returns `snapshot`+id for a non-empty pool, falls back when empty, in `tests/unit/selfplay/test_opponent_mix.py`
- [ ] T011 [P] [US1] Test: a **build/buy stub** snapshot opponent actually builds/buys then ends via the driver (exercises `_apply_action`), opponent actions come from the policy not the heuristic, in `tests/unit/env/test_snapshot_opponent.py`
- [ ] T012 [P] [US1] Test: **7-roll discard interleave** — opponent rolls a 7, agent owes a discard; the opponent turn suspends/resumes correctly via `_opp_pending`, in `tests/integration/test_snapshot_discard_interleave.py`
- [ ] T013 [P] [US1] Test: **action cap** — a never-EndTurn stub triggers the hard cap (turn force-ends, anomaly logged), in `tests/unit/env/test_opponent_turn_cap.py`
- [ ] T014 [P] [US1] Test: **determinism** — same seed + snapshot id + isolated generator → identical opponent action sequence; AND the learner's rollout RNG is unperturbed vs a heuristic-opponent run, in `tests/unit/env/test_opponent_determinism.py`

Impl:
- [ ] T015 [US1] Build the opponent-side **turn-driver state machine** (roll → knight/robber → road-builder → dev-card plays → bank trades → build → EndTurn) driving `snapshot_opponent.sample` and applying via `_apply_action`, replacing `opp.move()` in `_run_opponent_main_turn` (`catan_env.py:699`). The **roll phase MUST hook the existing `_opp_pending` discard-suspension** so a 7-roll that makes the agent owe a discard suspends/resumes correctly across `env.step` (no new re-entrancy machinery — reuse the existing path). **Phase-2 review contracts:** (a) the opponent-local `EnvObsState` MUST set `opp_kind`/`opp_policy_id` to describe the AGENT from the opponent's POV (`OPP_KIND_SELF_LATEST`), NOT reuse the agent's values; (b) the driver MUST call `snapshot_opponent.reset_rng()` at each game start for per-game determinism
- [ ] T016 [US1] Hard per-turn action cap in the driver (FR-013) — force EndTurn + log anomaly on exceed
- [ ] T017 [US1] Wire snapshot sampling into `build_env_opponent_mix`; remove the `NotImplementedError` at `selfplay/league.py:245`
- [ ] T018 [US1] Remove the snapshot `NotImplementedError` at `env/catan_env.py:181`; inject the frozen-opponent helper when `opponent_type='snapshot'`
- [ ] T019 [US1] Batched frozen-opponent inference across envs in `src/catan_rl/ppo/game_manager.py` (mixed opponent kinds per env handled coherently)
- [ ] T020 [US1] Empty-pool / evicted-snapshot fallback to a non-snapshot opponent (FR-011)

**Checkpoint**: self-play rollout runs, opponent plays full turns via its policy, no leak (T006), no livelock (T013).

## Phase 4: User Story 2 — Measure the champion (P2) — depends on US1

**Corrected**: policy-vs-policy eval seats the opponent via the **US1 in-env driver** (NOT the recorder actor). `build_actor` only *loads* a checkpoint into a frozen `CatanPolicy`.

- [ ] T021 [P] [US2] Test: `evaluate_policy_vs_policy` returns WR + Wilson CI, seat-symmetrized, bit-for-bit identical across two CPU runs at one seed, in `tests/unit/eval/test_policy_vs_policy.py`
- [ ] T022 [US2] Implement `evaluate_policy_vs_policy(champion, opponent_ref, n_games, seed, device="cpu")` — load opponent via `replay/player_factory.build_actor`, seat it through the US1 snapshot-opponent driver, reuse `eval/wilson.py`; `EvalMatchupResult` EXTENDS the existing `EvalResult` (no parallel type), in `src/catan_rl/eval/harness.py`
- [ ] T023 [US2] Emit additive TB scalar `eval/wr_vs_<opp>` (no renames), in `src/catan_rl/ppo/training_loop.py`

## Phase 5: User Story 3 — Fresh snapshots enter play (P3) — depends on US1

- [ ] T024 [P] [US3] Test: after `add_snapshot` + `set_opponents`, the next rollout's opponent is the new id; observed mix matches the configured ratio, in `tests/unit/ppo/test_set_opponents.py`
- [ ] T025 [US3] Add `vec_env.set_opponents(assignments)` threading `(kind, snapshot_id)` into each env's next `reset`, in `src/catan_rl/ppo/vec_env.py`
- [ ] T026 [US3] Refresh per-env opponent assignment each rollout (replace the construction-time lock at `training_loop.py:201`)
- [ ] T027 [US3] Static heuristic:snapshot mix knob + a **non-zero heuristic floor** (can't be tuned to 0 — guards PG-2) in `configs/ppo_default.yaml` + `LeagueConfig` (`arguments.py`)

## Phase 6: Polish & cross-cutting

- [ ] T028 [P] Checkpoint-compat guard: load a `bootstrap_v1` checkpoint (u799 seed) after the change → no shape mismatch (Constitution III), in `tests/unit/policy/test_checkpoint_compat.py`
- [ ] T029 [P] Integration smoke: `snapshot_weight=0.5` + seeded league (u799) → ~3 PPO updates, no errors, AND run the `eval/rules_invariants` gate to assert the 1v1 ruleset is unchanged (FR-009), in `tests/integration/test_selfplay_smoke.py`
- [ ] T030 [P] Update `MEMORY.md`/`README.md` to record self-play snapshot opponent is wired
- [ ] T031 Run full `pytest` + `mypy` + `ruff`; confirm green; merge to main + push

## Dependencies

- Phase 2 (T003–T009) blocks all user stories. **T003 (`_apply_action`) blocks T015**; **T005 (opponent-POV) blocks T006/T015**.
- US1 (P1) is the MVP and the keystone driver.
- **US2 (P2) depends on US1** (it seats the opponent via the US1 driver — corrected from the prior "independent" claim).
- US3 (P3) depends on US1.
- Polish last.

## Parallel opportunities

- Test files T004/T006/T010/T011/T012/T013/T014 are independent (different files), but T006 + T011 gate the US1 impl (write them first).
- T028/T029/T030 are independent.

## Implementation strategy

1. **Foundational first** (T003–T009): the `_apply_action` extraction + opponent-POV builder are the prerequisites; land the POV no-leak test (T006) before any driver code.
2. **US1 = MVP**: the full turn-driver. Validate quickstart Scenarios 1/2/5 + no-leak + cap.
3. **US2** (eval, needs US1), then **US3** (swap + curriculum).
4. Seed the first self-play run from `bootstrap_v1` checkpoint **u799** (~0.66 WR vs heuristic).
