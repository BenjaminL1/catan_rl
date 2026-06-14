# Implementation Plan: Inference-Time Search (Determinized MCTS)

**Branch**: `main` (solo project — no PR; short-lived local branch optional) | **Date**: 2026-06-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-inference-search/spec.md`

## Summary

Add an **offline** determinized PUCT-MCTS search agent that wraps the trained v6 policy, giving it the lookahead a single-forward-pass policy lacks. Search uses the policy's action priors + a **squashed** value-head leaf (`sigmoid(3.22·V−1.14)`), determinizes stochasticity by cloning the env (perfect 1v1 hand-tracking ⇒ no opponent-hand belief sampling), and expands the action space via progressive widening on the 13-way type head. The **first deliverable is a go/no-go bake-off** (minimal search vs the raw policy, Wilson LB > 0.50 at n≥200→500) that retires the one residual risk (off-distribution value quality) before any scaling. All design forks are pre-resolved in [research.md](research.md); the approach is de-risked by probes (value head ranks+calibrates, engine ~120 sims/sec NN-bound, clones faithful).

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: PyTorch (existing `CatanPolicy`), the pure-Python `catan_rl.engine` + `CatanEnv` (cloned via `deepcopy`), `eval/harness.py` (`evaluate_policy_vs_policy`, Wilson CI), `replay/player_factory.build_actor`, `selfplay/snapshot_opponent.FrozenSnapshotOpponent`. numpy/scipy for the Elo fit.
**Storage**: N/A (in-memory search; JSON results under `runs/search/`)
**Testing**: pytest (unit: `tests/unit/search/`; integration: search-vs-policy smoke); ruff + mypy strict; CI Python 3.11+
**Target Platform**: local CLI, macOS/Linux; CPU-pinned (eval device policy)
**Project Type**: single library + CLI (the `src/catan_rl/` package)
**Performance Goals**: ~120 sims/sec/move baseline (NN-forward-bound); ~100 sims @ 1s/move, ~550 @ 5s; batched leaf eval is the scaling lever if needed
**Constraints**: offline-only; additive + isolated (training path/policy/obs/action/checkpoints untouched, byte-identical when search absent); 1v1 ruleset sacred; no GUI import; uses existing v2 checkpoints as-is
**Scale/Scope**: one new `src/catan_rl/search/` module (~5-7 files) + a CLI entry + tests; offline eval over hundreds-to-500 games

## Constitution Check

*GATE: passes — no violations.*

- **I. 1v1 Ruleset Sacred** ✅ — search only proposes legal actions and simulates the *unchanged* engine; no rule constant, action-space, obs-schema, or trading change. (FR-007, SC-006)
- **II. Engine Integrity** ✅ — engine not modified; search clones + reads it.
- **III. Backward-Compatible, Additive Artifacts** ✅ — new isolated module; no policy state-dict change/migration (loads existing checkpoints); any new TB/diagnostic scalars are new names (append-only). (FR-009/010)
- **IV. Test-First & Green CI** ✅ — bake-off gate, determinism, legality, and no-regression are the gating tests, written alongside; ruff+mypy+pytest green.
- **V. Self-Play Is 2-Player Zero-Sum** ✅ — search models the symmetric 2-player game and uses the *known* opponent hand (perfect tracking); no >2-player or hidden-trade assumptions.
- **Device policy** ✅ CPU-pinned. **Config SoT** ✅ — `SearchConfig` dataclass (isolated; not bolted onto `TrainConfig`). **No v1 artifacts** ✅ — v2-lineage checkpoints only.

No entries in Complexity Tracking (no violations to justify).

## Project Structure

### Documentation (this feature)

```text
specs/003-inference-search/
├── plan.md              # this file
├── research.md          # Phase 0 — 8 resolved decisions (done)
├── data-model.md        # Phase 1 — SearchConfig/Node/Agent/Determinization/Result (done)
├── quickstart.md        # Phase 1 — 5 runnable validation scenarios (done)
├── contracts/
│   └── internal-interfaces.md   # Phase 1 — C1..C6 module surfaces (done)
└── tasks.md             # Phase 2 — created by /speckit-tasks (NOT yet)
```

### Source Code (repository root)

```text
src/catan_rl/search/                # NEW, isolated module
├── __init__.py
├── config.py            # SearchConfig (dataclass + validation)
├── value.py             # squash_value, leaf_value (C1)
├── priors.py            # action_priors over the autoregressive heads (C2)
├── node.py              # SearchNode (state/stats/expansion)
├── mcts.py              # determinized PUCT loop + progressive widening
├── agent.py             # SearchAgent.choose_action(env) (C3)
├── eval_search.py       # evaluate_search_vs_policy — env-access loop (C4)
└── bakeoff.py           # run_bakeoff gate + time-budget ladder (C6)

src/catan_rl/cli/search_eval.py     # NEW console script catan-rl-search-eval (C5)
pyproject.toml                      # add the console-script entry (additive)

tests/unit/search/                  # NEW
├── test_value.py        # squash bounded (0,1), perspective sign
├── test_priors.py       # priors legal + normalized + mask-consistent
├── test_mcts.py         # PUCT select/expand/backup; progressive widening; determinism
├── test_agent.py        # forced-move short-circuit; legality; no env mutation; reproducible
└── test_eval_search.py  # seat-symmetry; mirrors evaluate_policy_vs_policy semantics
tests/integration/test_search_smoke.py   # tiny-budget search vs heuristic completes + legal
```

**Structure Decision**: Single-project library layout (matches the repo). All search code is a **new, self-contained `src/catan_rl/search/` package** that only *reads* the engine/policy/env and *reuses* eval primitives — nothing in `engine/`, `policy/`, `ppo/`, `env/`, or `checkpoint/` is modified, satisfying the additive/isolated constitution gate. The one new public entry is the `catan-rl-search-eval` console script.

## Complexity Tracking

No constitution violations — table intentionally empty.

## Phase 2 sequencing (for /speckit-tasks)

Build order with the **bake-off as task #1** (the off-distribution-V gate): `config` + `value` (squash) + `priors` → a *minimal* `mcts`/`agent` → `eval_search` → **run `bakeoff` at n≥200 (GATE)** → if PASS: harden `mcts` (progressive widening, determinization count, anytime budget) → CLI + Elo rung + full tests; if FAIL: stop, document the pivot (priors-weighted / rollout-to-late-state). Determinization-variance measurement happens inside the prototype.
