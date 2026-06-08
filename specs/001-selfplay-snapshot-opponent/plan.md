# Implementation Plan: Frozen-policy self-play opponent + policy-vs-policy eval

**Branch**: `001-selfplay-snapshot-opponent` (spec); implementation lands on its own `feat/` branch | **Date**: 2026-06-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-selfplay-snapshot-opponent/spec.md`

## Summary

Wire the one primitive that lets the agent train against frozen snapshots of its
own past selves (and measure itself against any loaded policy). Today both the
league consumer (`league.py:245`) and the env (`catan_env.py:181`) raise
`NotImplementedError` when a snapshot opponent is requested, the opponent mix is
locked at construction (`training_loop.py:201`), and there is no policy-vs-policy
eval. The approach delivers four small, independently-testable increments:
(1) a **frozen in-env opponent** that drives the opponent turn by encoding its
POV obs + masks and sampling a loaded `CatanPolicy` (replacing the `opp.move()`
call), (2) the **league consumer** that maps `'snapshot'` → a concrete snapshot
id and deletes the two guards, (3) **mid-rollout opponent swap** via a new
`vec_env.set_opponents()`, and (4) **policy-vs-policy eval** reusing
`replay/player_factory.build_actor` + a `eval/wr_vs_<opp>` champion TB scalar.
Opponent inference is batched in the main process across envs. No obs/action
shape change — the existing `_OppIdEmbedding` is fed real values, keeping the
in-flight `bootstrap_v1` checkpoint loadable as the first snapshot.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: PyTorch (MPS/CPU), custom PPO (`src/catan_rl/ppo`),
Gymnasium-style env (`src/catan_rl/env`), league (`src/catan_rl/selfplay`).

**Storage**: League snapshots are in-memory CPU state-dict clones (bounded deque,
producer already wired); checkpoints on disk (existing `CheckpointManager`).

**Testing**: pytest. TDD — unit tests for the snapshot-opponent primitive and
eval path written before implementation; an integration smoke for a self-play
rollout with `snapshot_weight>0`.

**Target Platform**: Apple Silicon (MPS) for training, CPU for eval; CUDA opt-in.

**Project Type**: Single internal research library/CLI.

**Performance Goals**: Opponent NN inference batched across the N rollout envs in
the main process (the measured bottleneck is the small-batch policy forward);
no per-env batch-of-one snapshot forwards.

**Constraints**: NO observation/action-head shape change (checkpoint back-compat,
`bootstrap_v1` must load); 1v1 ruleset + reward + action space unchanged;
deterministic eval on CPU; opponent inference is inference-only (`no_grad`,
never trained).

**Scale/Scope**: ~1.4M-param policy; n_envs=128 rollout; league deque maxlen
per `LeagueConfig`. Four increments, each one PR-sized slice of one larger PR.

## Constitution Check

*GATE: passes — this feature is plumbing under the existing invariants.*

| Principle | Status | How |
|---|---|---|
| I. 1v1 ruleset sacred | ✅ | No rule/action-space/obs/reward change; opponent is just a different move source. |
| II. Engine integrity | ✅ | Engine untouched; only the env's opponent-turn driver gains a snapshot branch. |
| III. Backward-compatible / additive | ✅ | No shape change (feed existing `_OppIdEmbedding`); new TB scalar `eval/wr_vs_<opp>` is additive; `bootstrap_v1` stays loadable. |
| IV. Test-first & green CI | ✅ | TDD per increment; stub-opponent + eval-determinism + fallback tests. |
| V. Self-play is 2-player zero-sum | ✅ | Single frozen opponent per env; PFSP/rating deferred (non-goals). |

**No violations** → Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-selfplay-snapshot-opponent/
├── plan.md              # this file
├── research.md          # Phase 0 decisions
├── data-model.md        # entities (snapshot, assignment, frozen opp, eval result)
├── quickstart.md        # runnable validation scenarios
├── contracts/
│   └── internal-interfaces.md   # new/changed internal function contracts
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks output (not created here)
```

### Source Code (repository root)

```text
src/catan_rl/
├── env/catan_env.py          # +snapshot opponent branch in _run_opponent_main_turn; delete guard :181
├── selfplay/league.py        # +snapshot sampling into build_env_opponent_mix; delete guard :245
├── ppo/
│   ├── vec_env.py            # +set_opponents(); batched opponent inference seam
│   ├── game_manager.py       # opponent construction → frozen-policy opponent
│   └── training_loop.py      # refresh opponent assignment each rollout (was locked :201)
├── eval/
│   └── harness.py            # +policy-vs-policy eval (reuse build_actor) + champion WR scalar
├── policy/network.py         # feed _OppIdEmbedding real kind/id (no resize)
└── replay/player_factory.py  # REUSE build_actor/_PolicyActor for eval (no new loader)

configs/ppo_default.yaml      # snapshot_weight + static heuristic:snapshot mix knob

tests/
├── unit/selfplay/            # snapshot sampling, fallback
├── unit/env/                 # snapshot-opponent turn uses loaded policy (stub)
├── unit/eval/                # policy-vs-policy WR + CI + determinism
└── integration/              # self-play rollout smoke (snapshot_weight>0)
```

**Structure Decision**: Single-project layout; changes are surgical edits at the
named seams plus one new eval function. No new top-level packages.

## Complexity Tracking

> No constitution violations — section intentionally empty.
