# Implementation Plan: Expert Iteration (search-as-teacher distillation)

**Branch**: `main` (solo project — no PR) | **Date**: 2026-06-15 | **Spec**: [spec.md](spec.md)

## Summary

Bank the offline 003 search lever into the base policy via **expert iteration**: play
games where the agent is the 003 determinized-search agent, record each agent decision as
a training target, and **distill** those targets into a fine-tune of v6 — reusing the
existing **behavior-cloning pipeline** wholesale. The decisive first deliverable is a
**pilot gate** (one distillation round, distilled-vs-raw-v6 search-free, Wilson LB > 0.50
at n≥200→500) that mirrors the 003 bake-off's go/no-go. Only if it passes do we build the
flywheel (≥2 rounds) and ship the banked fast agent. All design forks are resolved in
[research.md](research.md); the binding constraint is single-M1 compute (the throughput
multiplier, batched leaf eval, is in scope only after the gate).

## Technical Context

**Language/Version**: Python 3.11+
**Primary deps**: the 003 search (`catan_rl.search.SearchAgent`), the BC pipeline
(`catan_rl.bc.dataset` shard format + `_DecisionRecord`/`_flatten_records`, `catan_rl.bc.loss.bc_loss`, `catan_rl.bc.train.train_bc`), the env (`CatanEnv` + `compute_action_masks` + `hidden_belief_target`), eval (`eval.harness.evaluate_policy_vs_policy`, Wilson CI), checkpoint manager, `scripts/elo_ladder.py`.
**Storage**: search-labeled NPZ shards + manifest under `data/exit/round_<n>/` (BcDataset-compatible); distilled checkpoints under `runs/exit/`; JSON results under `runs/exit/`.
**Testing**: pytest (`tests/unit/expert_iteration/`); ruff + mypy strict; CI 3.11+.
**Target**: local CLI; MPS for the distillation SGD, CPU-pinned for search labeling + eval.
**Performance**: a search game (~32 s @ 50 sims) yields ~40-80 non-forced labeled positions → ~5k-position pilot ≈ ~60-80 games ≈ ~45-60 min; distillation fine-tune (few epochs, ~5k samples) ~10-20 min; gate eval is SEARCH-FREE (`evaluate_policy_vs_policy`) so n=500 is minutes.
**Constraints**: 1v1 ruleset sacred; engine/policy-arch/obs/action/checkpoint UNCHANGED; the 003 search consumed as-is; additive + isolated (new `expert_iteration/` module; one backward-compatible `train_bc(init_ckpt=...)` warm-start param); MPS-train/CPU-eval; no GUI import; v2-lineage checkpoints.
**Scale/Scope**: one new `src/catan_rl/expert_iteration/` module (~3-5 files) + a 1-line `train_bc` warm-start hook + CLI/runner + tests; offline generate→distill→eval.

## Constitution Check

*GATE: passes — no violations.*

- **I. 1v1 Ruleset Sacred** ✅ — labeling plays the unchanged engine via the 003 search; no rule/action/obs/trading change; distilled policy only proposes legal actions (rules-invariant audited, FR-010, SC-005).
- **II. Engine Integrity** ✅ — engine not modified; labeler reads/clones it through the search + env.
- **III. Backward-Compatible, Additive Artifacts** ✅ — new isolated module; distilled checkpoints are v2-lineage, no state-dict shape change (FR-004); the only edit to an existing module is an additive `train_bc(init_ckpt=None)` warm-start param (default None = today's behavior); new TB scalars are new names.
- **IV. Test-First & Green CI** ✅ — the pilot gate, labeler-format, warm-start-load, and no-regression are the gating tests, written alongside; ruff+mypy+pytest green.
- **V. Self-Play Is 2-Player Zero-Sum** ✅ — labeling uses the symmetric 2-player game; the search teacher already models the known-hand 1v1 opponent.
- **Device** ✅ MPS-train / CPU-eval+search. **Config SoT** ✅ — distillation knobs in a config (mirrors `configs/bc.yaml`); the ExIt round config is isolated. **No v1 artifacts** ✅ — v2-lineage only.

No Complexity Tracking entries.

## Project Structure

```text
src/catan_rl/expert_iteration/      # NEW, isolated module
├── __init__.py
├── labeler.py        # play search-vs-opponent games -> BcDataset-compatible NPZ shards
├── distill.py        # warm-start v6 + run train_bc on search shards -> distilled ckpt
├── gate.py           # distilled-vs-raw-v6 search-free eval -> PASS/FAIL (Wilson LB>0.50)
└── round.py          # one ExIt round: generate -> distill -> eval (flywheel unit, US2)

src/catan_rl/bc/train.py            # +init_ckpt warm-start param (additive, backward-compat)
scripts/run_exit_pilot.py           # NEW runner: the US1 pilot gate (detached)

tests/unit/expert_iteration/        # NEW
├── test_labeler.py     # shards are BcDataset-loadable; action == search choice; forced skipped
├── test_distill.py     # warm-start loads v6; distillation runs; output is v2-lineage-loadable
└── test_gate.py        # gate logic: PASS iff Wilson LB>0.50; search-free eval reused
tests/unit/bc/test_train_warmstart.py   # train_bc(init_ckpt=...) loads weights, trains
```

**Structure Decision**: a new self-contained `src/catan_rl/expert_iteration/` package (module name avoids shadowing the `exit` builtin). It only *reads/reuses* the search, BC pipeline, env, and eval; the sole edit to existing code is a backward-compatible `train_bc` warm-start param (ExIt fundamentally needs to fine-tune, not train-from-scratch). Everything else is reuse.

## Complexity Tracking

No constitution violations — table intentionally empty.

## Phase 2 sequencing (for /speckit-tasks)

Build with the **pilot gate as the goal of US1**: `train_bc` warm-start hook → labeler (search games → BC shards) → distill (warm-start + BC train) → gate eval → **run the pilot (GATE)**. If PASS: `round.py` flywheel (US2) + the Elo-ladder placement (US3) + (only if throughput-bound) batched leaf eval (FR-012). If FAIL: stop, document the pivot. Per-phase review gate (`specs/003-inference-search/reviewers.md`, adapted) after each code phase.
