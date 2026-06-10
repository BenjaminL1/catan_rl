# Implementation Plan: PFSP Opponent Sampling

**Branch**: `main` (short-lived `feat/pfsp-opponent-sampling` local branch) | **Date**: 2026-06-10 | **Spec**: [spec.md](./spec.md)

## Summary

Replace the uniform draw over the snapshot pool with a win-rate-weighted draw, so the learner trains preferentially against the snapshots it is **not** reliably beating — arresting the catastrophic-forgetting drift that made v2/v3 peak (strength-vs-u799 ≈ 0.77) then collapse (≈ 0.43) under uniform sampling. The league gains a per-snapshot win-rate store fed by rollout outcomes; `build_env_opponent_assignments` weights the *pool* category by a configurable curve; the frozen anchor + heuristic floor keep their fixed reserved weights. Off by default (byte-identical). Reuses the existing `OpponentAssignment` + snapshot-opponent driver + checkpoint capture — no new opponent driver.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: numpy (sampling); torch only via the existing snapshot policy load — PFSP itself adds none
**Storage**: in-memory per-snapshot `(wins, games)` counts in `League`, persisted via the existing checkpoint league-capture
**Testing**: pytest — unit (league WR + weighting + persistence), integration (PFSP-on rollout records outcomes; PFSP-off byte-identity)
**Target Platform**: M1 (MPS train / CPU eval); PFSP is device-agnostic CPU bookkeeping
**Project Type**: single library (`src/catan_rl/`)
**Performance Goals**: O(n_envs) per rollout, no extra policy inference; per-update wall-clock within ~2% of PFSP-off (SC-006)
**Constraints**: PFSP off ⇒ byte-identical assignments (FR-005); correct attribution under auto-reset (FR-002); finite weights at p∈{0,1} + equal-WR (FR-008); resumable (FR-007)
**Scale/Scope**: league ≤ `maxlen` (=100) snapshots; n_envs=128

## Constitution Check

*GATE: passed.*

- **I. 1v1 ruleset sacred** — PFSP changes opponent SELECTION only; no game-rule constant, action space, obs schema, or trading change. ✓
- **II. Engine integrity** — no engine change. ✓
- **III. Backward-compatible / additive** — no policy state-dict shape change (PFSP state is league bookkeeping, not policy weights); the checkpoint schema gains an additive, optional `opponent_stats` field with a default-empty fallback for old checkpoints. ✓
- **IV. Test-first & green CI** — TDD: league WR/weighting/persistence + attribution + byte-identity tests precede implementation. ✓
- **V. Self-play 2-player zero-sum** — PFSP is defined over the symmetric 2-player league; win-rate is the single zero-sum outcome. ✓
- **Additional**: `arguments.py` stays the config SoT (new `LeagueConfig` fields); any TensorBoard scalars are additive (optional `pfsp/*`, no renames).

No violations → Complexity Tracking empty.

## Project Structure

### Documentation (this feature)

```text
specs/002-pfsp-opponent-sampling/
├── plan.md          # this file
├── research.md      # design decisions (estimator, curve, attribution, persistence)
├── data-model.md    # OpponentStats + PFSP config entities
├── contracts/
│   └── internal-interfaces.md   # changed internal APIs
├── quickstart.md    # validation scenarios
└── tasks.md         # /speckit-tasks output (not created here)
```

### Source Code (touched)

```text
src/catan_rl/
├── selfplay/league.py        # OpponentStats store, record_outcome(), PFSP pool weighting in
│                             #   build_env_opponent_assignments, capture/restore hooks
├── ppo/
│   ├── arguments.py          # LeagueConfig: pfsp_enabled/pfsp_curve/pfsp_k/pfsp_min_games/pfsp_cold_start_weight + validation
│   ├── vec_env.py            # current_opponent_ids() accessor (per-env in-progress opponent snapshot_id)
│   ├── game_manager.py       # attribute finished games -> league.record_outcome (pre-step opponent ids)
│   └── training_loop.py      # (no new call site; refresh already calls build_env_opponent_assignments)
└── checkpoint/manager.py     # _capture_league_state / apply_to_league extended with opponent_stats

tests/unit/selfplay/test_league.py        # WR store, record_outcome, PFSP weighting, degenerate cases
tests/unit/ppo/test_game_manager.py       # outcome->opponent attribution under auto-reset
tests/unit/ppo/test_arguments.py          # PFSP config validation
tests/unit/checkpoint/...                  # opponent_stats round-trip
tests/integration/test_selfplay_smoke.py  # PFSP-on rollout updates WR; PFSP-off byte-identity
```

**Structure Decision**: Single-library layout (existing). PFSP lives almost entirely in `league.py` (state + weighting); the only cross-module plumbing is outcome attribution (vec_env accessor + game_manager hook) and the additive checkpoint field.

## Increments (implementation order)

**Increment 1 — Win-rate tracking + outcome attribution (US1 core, P1).**
`OpponentStats` (wins, games) keyed by snapshot_id in `League`; `record_outcome(snapshot_id, agent_won)`; evicted-with-snapshot cleanup. `vec_env.current_opponent_ids()` returns the in-progress opponent snapshot_id per env. `game_manager.collect` reads pre-step opponent ids and, on each terminated|truncated env, records the agent win/loss. No sampling change yet (uniform pool unchanged). Tests: attribution correct under mid-rollout auto-reset; WR updates.

**Increment 2 — PFSP-weighted sampling (US1 + US2, P1/P2).**
`LeagueConfig` PFSP fields + validation. `build_env_opponent_assignments` weights the *pool* draw by the curve when `pfsp_enabled`; cold-start (games < min_games → cold_start_weight); degenerate handling (equal-WR → uniform; p∈{0,1} finite via a floor). PFSP-off ⇒ byte-identical. Tests: hard-curve ordering (SC-001), cold-start (SC-002), off byte-identity (SC-003), degenerate weights (FR-008).

**Increment 3 — Persistence (US3, P3).**
Extend `_capture_league_state` / `apply_to_league` with `opponent_stats`; old checkpoints restore to empty (additive). Seeded sampling is already resume-reproducible (per-update RNG). Tests: round-trip equality (SC-004), same-seed assignment reproducibility.

## Complexity Tracking

> No constitution violations — section intentionally empty.
