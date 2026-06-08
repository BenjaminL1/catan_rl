# Tasks: Frozen-policy self-play opponent + policy-vs-policy eval

**Feature**: `specs/001-selfplay-snapshot-opponent/` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

TDD is required (constitution Principle IV). Within each user-story phase,
write the listed tests FIRST, watch them fail, then implement to green.
`[P]` = parallelizable (different files, no incomplete-task dependency).

## Phase 1: Setup

- [ ] T001 Create implementation branch `feat/selfplay-snapshot-opponent` off the spec branch
- [ ] T002 [P] Add a deterministic stub policy fixture (a `CatanPolicy`-compatible object whose `sample` always returns `EndTurn`) in `tests/conftest.py`

## Phase 2: Foundational (blocking — shared by US1 + US3)

- [ ] T003 Define `OpponentAssignment` (`kind: {"heuristic","random","snapshot"}`, `snapshot_id: int | None`) in `src/catan_rl/selfplay/league.py`
- [ ] T004 Add a batched, inference-only frozen-opponent helper (load a snapshot → `CatanPolicy.eval()` on the learner device, cached per snapshot_id, `sample(obs,masks)` under `torch.no_grad`) in `src/catan_rl/selfplay/snapshot_opponent.py`
- [ ] T005 Feed the existing `_OppIdEmbedding` real opponent kind/id values WITHOUT resizing, in the opp-id wiring (`src/catan_rl/policy/network.py` consumers / env obs build)

## Phase 3: User Story 1 — Train against a frozen past self (P1) 🟢 MVP

**Goal**: a league snapshot acts as the in-env opponent; self-play runs with no error.
**Independent test**: stub-EndTurn snapshot → opponent only ends turns; rollout completes.

- [ ] T006 [P] [US1] Test: `build_env_opponent_mix` returns `kind="snapshot"`+id for a non-empty pool and falls back (no snapshot) when empty, in `tests/unit/selfplay/test_opponent_mix.py`
- [ ] T007 [P] [US1] Test: an env assigned a stub-EndTurn snapshot opponent takes only `EndTurn` actions, raises no `NotImplementedError`, AND produces an identical opponent action sequence across two runs at the same seed + device (FR-006 rollout determinism), in `tests/unit/env/test_snapshot_opponent.py`
- [ ] T008 [US1] Wire snapshot sampling into `build_env_opponent_mix`; remove the `NotImplementedError` at `src/catan_rl/selfplay/league.py:245`
- [ ] T009 [US1] Add the snapshot branch to `_run_opponent_main_turn` (encode opponent POV obs+masks → frozen-opponent act fn instead of `opp.move()`); remove the `NotImplementedError` at `src/catan_rl/env/catan_env.py:181`
- [ ] T010 [US1] Construct + batch the frozen-policy opponent across snapshot-assigned envs in the rollout, in `src/catan_rl/ppo/game_manager.py`
- [ ] T011 [US1] Implement empty-pool / evicted-snapshot fallback to a non-snapshot opponent (FR-011), in `src/catan_rl/selfplay/league.py`

**Checkpoint**: US1 alone delivers a runnable self-play rollout (MVP).

## Phase 4: User Story 2 — Measure the champion vs any policy (P2)

**Goal**: champion vs any loaded policy → WR + Wilson CI. **Independent test**: eval vs a checkpoint twice at one seed → identical WR+CI on CPU.

- [ ] T012 [P] [US2] Test: `evaluate_policy_vs_policy` returns a finite WR + Wilson CI and is bit-for-bit identical across two CPU runs at a fixed seed, in `tests/unit/eval/test_policy_vs_policy.py`
- [ ] T013 [US2] Implement `evaluate_policy_vs_policy(champion, opponent_ref, n_games, seed, device="cpu")` reusing `replay/player_factory.build_actor`, seat-symmetrized, reusing `eval/wilson.py`, in `src/catan_rl/eval/harness.py`
- [ ] T014 [US2] Emit the additive TB scalar `eval/wr_vs_<opp>` (never rename existing scalars), in `src/catan_rl/ppo/training_loop.py`

## Phase 5: User Story 3 — Bring fresh snapshots into play (P3)

**Goal**: newly-added snapshots enter subsequent rollouts under a configurable mix.
**Independent test**: add_snapshot + refresh → next assignment includes the new id.

- [ ] T015 [P] [US3] Test: after `add_snapshot` + `set_opponents`, the next assignment includes the new snapshot id and the observed heuristic:snapshot mix matches the configured ratio in expectation, in `tests/unit/ppo/test_set_opponents.py`
- [ ] T016 [US3] Add `vec_env.set_opponents(assignments)` threading `(kind, snapshot_id)` into each env's next `reset`, in `src/catan_rl/ppo/vec_env.py`
- [ ] T017 [US3] Refresh the per-env opponent assignment each rollout (replacing the construction-time lock at `:201`), in `src/catan_rl/ppo/training_loop.py`
- [ ] T018 [US3] Add the static heuristic:snapshot mix knob to `configs/ppo_default.yaml` + `LeagueConfig` in `src/catan_rl/ppo/arguments.py`

## Phase 6: Polish & cross-cutting

- [ ] T019 [P] Checkpoint-compat guard test: load a `bootstrap_v1` checkpoint after the change → no shape mismatch (Constitution III), in `tests/unit/policy/test_checkpoint_compat.py`
- [ ] T020 [P] Integration smoke: `snapshot_weight=0.5` + a seeded league → ~3 PPO updates with no `NotImplementedError`/device errors, AND run the `eval/rules_invariants` gate to assert the 1v1 ruleset / reward / action space is unchanged (FR-009), in `tests/integration/test_selfplay_smoke.py`
- [ ] T021 [P] Update `MEMORY.md` / `README.md` to record that the self-play snapshot opponent is wired (remove "unwired" claims)
- [ ] T022 Run full `pytest` + `mypy` + `ruff`; confirm green; open ONE PR (`feat/selfplay-snapshot-opponent`)

## Dependencies

- Phase 2 (T003–T005) blocks all user stories.
- US1 (P1) is the MVP and has no dependency on US2/US3.
- US2 (P2) depends only on Foundational (eval reuses `build_actor`, not the rollout path) — can proceed in parallel with US1 after Phase 2.
- US3 (P3) depends on US1 (snapshot opponents must exist before refreshing assignments is meaningful).
- Polish (Phase 6) runs last.

## Parallel opportunities

- T006 ∥ T007 (different test files).
- After Phase 2: US1 and US2 test-writing can proceed in parallel.
- T019 ∥ T020 ∥ T021 (independent files).

## Implementation strategy

1. **MVP = US1** (T001–T011): self-play runs against frozen snapshots. Stop and validate (quickstart Scenarios 1, 2, 5).
2. Add **US2** (eval) so strength is measurable, then **US3** (fresh snapshots + mix).
3. Polish + single PR. Seed the first real run from the strongest `bootstrap_v1` checkpoint.
