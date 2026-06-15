# Tasks: Expert Iteration (search-as-teacher distillation)

**Input**: `specs/004-expert-iteration/` (plan, spec, research, data-model, contracts, quickstart)

**Tests**: REQUIRED (test-first — Constitution IV).

**Organization**: by user story. **US1 (the pilot gate) is the MVP + go/no-go** — US2/US3 proceed only if it passes.

## Review Gates (auto-run)

After every code-producing phase, run the two-reviewer gate in
[`../003-inference-search/reviewers.md`](../003-inference-search/reviewers.md) (Reviewer A —
senior RL game-dev; Reviewer B — senior game-dev SWE), adapted to ExIt (distillation
correctness: target/label fidelity, warm-start, value/forgetting; isolation/reuse/types),
in parallel on the phase diff. Resolve every BLOCKER + SHOULD-FIX, re-green pytest + ruff +
mypy --strict, commit. **RG-US1 is hard-blocking before the pilot run T016.**

---

## Phase 1: Setup

- [ ] T001 Create `src/catan_rl/expert_iteration/__init__.py` (module name avoids the `exit` builtin) + `tests/unit/expert_iteration/__init__.py`. Confirm nothing imports it (additive gate, FR-011).

## Phase 2: Foundational (blocking — shared by all stories)

- [ ] T002 [P] Write `tests/unit/bc/test_train_warmstart.py`: `train_bc(init_ckpt=PATH)` loads the checkpoint's policy weights before training (param shifts from the loaded init, not random); `init_ckpt=None` path unchanged.
- [ ] T003 Add `init_ckpt: Path | None = None` to `train_bc` in `src/catan_rl/bc/train.py` — when set, strict-load that checkpoint into the policy after `set_board_geometry`, before the loop (warm-start, contract C2; default None byte-identical). Passes T002.
- [ ] T004 [P] Add `SearchLabelConfig` + `DistillConfig` dataclasses (+ `__post_init__` validation per data-model) in `src/catan_rl/expert_iteration/config.py`; `tests/unit/expert_iteration/test_config.py`.

- [ ] **🔬 REVIEW GATE RG-Foundational** — reviewers.md (A+B) on the Phase 1+2 diff. Focus: warm-start strict-load + v2-lineage + `init_ckpt=None` byte-identical; config validation; additivity.

---

## Phase 3: User Story 1 — Pilot gate (P1) 🎯 MVP / GO-NO-GO

**Goal**: one search-distilled fine-tune of v6 beats raw v6 search-free — the single decision that gates the whole effort.

**Independent test**: `run_exit_pilot.py` → `gate.json` with Wilson LB > 0.50 at n≥200→500 (or a documented FAIL).

- [ ] T005 [P] [US1] Write `tests/unit/expert_iteration/test_labeler.py`: generated shards are `BcDataset`-loadable; recorded `action` equals the `SearchAgent` choice at that state; forced (mask_sum==1) decisions skipped; `z_disc` filled; reproducible at a fixed seed; tiny n.
- [ ] T006 [US1] Implement `src/catan_rl/expert_iteration/labeler.py` — `generate_search_labels(cfg)` plays `SearchAgent` vs opponent, records non-forced agent decisions as BcDataset-compatible NPZ shards + manifest (reuse `bc.dataset` `_DecisionRecord`/`_flatten_records`/shard writer) (contract C1). Passes T005.
- [ ] T007 [P] [US1] Write `tests/unit/expert_iteration/test_distill.py`: `distill(cfg)` warm-starts from the base ckpt, trains on a tiny labeled shard dir, and returns a v2-lineage checkpoint loadable by the existing manager.
- [ ] T008 [US1] Implement `src/catan_rl/expert_iteration/distill.py` — `distill(cfg)` calls `train_bc(init_ckpt=base, data_dir=labels, peak_lr/max_epochs per D8)` -> distilled ckpt (contract C2). Passes T007.
- [ ] T009 [P] [US1] Write `tests/unit/expert_iteration/test_gate.py`: `run_gate` is search-free (reuses `evaluate_policy_vs_policy`), seat-symmetrized, PASS iff Wilson LB > 0.50; reports WR vs heuristic + a prior rung; tiny n.
- [ ] T010 [US1] Implement `src/catan_rl/expert_iteration/gate.py` — `run_gate(distilled, v6, ...) -> GateResult` (contract C3, encodes SC-001 + the forgetting guard). Passes T009.
- [ ] T011 [US1] Implement `scripts/run_exit_pilot.py` — label → distill → gate; writes `data/exit/round_0/` + `runs/exit/round_0/{distill,gate.json}` (contract C5; detached-friendly; CPU search/eval, MPS distill; no GUI import).
- [ ] T012 [US1] Add `tests/integration/test_exit_smoke.py`: a tiny pilot (few games, sims=4, max_turns small) runs label→distill→gate end-to-end producing a GateResult with zero ruleset violations.
- [ ] **🔬 REVIEW GATE RG-US1 (HARD-BLOCKING)** — reviewers.md (A+B) on the Phase 3 diff BEFORE T016. Focus: label fidelity (action == search choice, obs/mask consistency, forced-skip, z_disc), warm-start distillation correctness (no v6-clobber; LR sane), search-free gate (no MCTS at eval), forgetting guard, reproducibility, test strength. Resolve all BLOCKER/SHOULD-FIX before the run.
- [ ] T016 [US1] 🚦 **RUN THE GATE**: `run_exit_pilot.py --sims 50 --n-positions 5000 --seed 0` (detached + watcher). **If PASS (Wilson LB>0.50 at n≥500) → US2/US3. If FAIL → STOP; document the pivot (soft visit-distribution targets / more positions / more sims via batched leaf eval / LR+epoch tuning) and DO NOT build US2/US3.**

**Checkpoint**: US1 alone answers "does distilling search beat the base policy?" with a Wilson-bounded number — a decisive MVP.

---

## Phase 4: User Story 2 — Compounding flywheel (P2)  *(only if T016 PASSED)*

- [ ] T013 [US2] Implement `src/catan_rl/expert_iteration/round.py` — `run_round(...)` (generate→distill→gate; keep best-so-far as next base; don't seed from a regressed round) (contract C4); test the orchestration + best-so-far logic.
- [ ] T014 [US2] Run ≥2 ExIt rounds; assert per-round Elo over raw v6 monotone non-decreasing within CI (SC-003); write `runs/exit/flywheel.json`.
- [ ] **🔬 REVIEW GATE RG-US2** — reviewers.md (A+B) on the Phase 4 diff (round orchestration, best-so-far, regression handling, reproducibility).
- [ ] T015 [US2] *(only if throughput-bound)* Batched leaf evaluation for the search (FR-012) — many MCTS leaves per forward; assert identical action vs unbatched at a fixed seed; the flywheel's throughput lever.

---

## Phase 5: User Story 3 — Banked fast agent on the Elo ladder (P3)  *(only if T016 PASSED)*

- [ ] T017 [US3] Add the best distilled checkpoint as a (search-free) rung on `scripts/elo_ladder.py`; report its Elo delta over raw v6 with CI to `runs/exit/elo_uplift.json` (SC-004); confirm positive + CI excludes 0.

## Phase 6: Polish & Cross-Cutting

- [ ] T018 [P] No-regression: `pytest tests/unit/bc tests/unit/search tests/unit/eval tests/integration/test_selfplay_smoke.py` — confirm BC/search/eval byte-identical with `train_bc(init_ckpt=None)` + ExIt present-but-unused (SC-005).
- [ ] T019 [P] `ruff check` + `mypy --strict` over `src/catan_rl/expert_iteration/` + `src/catan_rl/bc/train.py` + `scripts/run_exit_pilot.py`; fix nits.
- [ ] T020 Update `MEMORY.md` + a memory note with the pilot outcome (Elo uplift or the documented pivot); refresh `specs/004-expert-iteration/` notes + tasks.md checkboxes. No new docs.

## Dependencies & MVP

- Setup (T001) → Foundational (T002-T004) → US1 / gate (T005-T012,T016) → *[gate PASS]* → US2 (T013-T015) + US3 (T017) → Polish (T018-T020).
- **US1 is the MVP + a hard gate**: T016 PASS required before any US2/US3 task.
- Within US1: warm-start (T003) → labeler (T006) → distill (T008) → gate (T010) → runner (T011) → smoke (T012) → **RG-US1** → **run gate (T016)**. Tests precede implementations (test-first).
