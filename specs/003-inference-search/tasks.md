# Tasks: Inference-Time Search (Determinized MCTS)

**Input**: Design documents from `specs/003-inference-search/` (plan.md, spec.md, research.md, data-model.md, contracts/internal-interfaces.md, quickstart.md)

**Tests**: REQUIRED (test-first — Constitution Principle IV: every behavioral change ships with tests).

**Organization**: by user story. **US1 (the bake-off) is the MVP and the go/no-go gate** — US2/US3 proceed only if it passes.

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: parallelizable (different files, no incomplete-task dependency)
- **[US#]**: user story (US1=P1 bake-off, US2=P2 thinking opponent, US3=P3 Elo)

---

## Review Gates (auto-run)

After **every** phase that produces code, before advancing, run the two-reviewer
gate in [`reviewers.md`](reviewers.md) on that phase's diff
(`git diff <phase-start-sha>..HEAD`): **Reviewer A — senior RL game-dev** (search
correctness) + **Reviewer B — senior game-dev SWE** (quality/architecture), both
as `general-purpose` subagents, in parallel. Resolve every **BLOCKER** and
**SHOULD-FIX**, re-green tests + `mypy --strict`/`ruff`, then commit and proceed.
The build workflow triggers these automatically at the `🔬 REVIEW GATE` markers
below — they are mandatory, not optional. **RG-US1 is hard-blocking**: it must
pass *before* the expensive bake-off T016 runs.

---

## Phase 1: Setup

- [x] T001 Create the isolated package `src/catan_rl/search/__init__.py` (empty public exports) and test package `tests/unit/search/__init__.py`. Confirm no existing module imports `search` (additive/isolated gate, FR-009).

---

## Phase 2: Foundational (blocking — shared by all stories)

- [x] T002 [P] Add `SearchConfig` dataclass + `__post_init__` validation in `src/catan_rl/search/config.py` (per data-model: `sims_per_move`/`time_budget_s` mutually exclusive, `n_determinizations>=1`, `c_puct>0`, `value_squash_a/b`, `pw_c/pw_alpha`, `max_depth`, `seed`).
- [x] T003 [P] Write `tests/unit/search/test_value.py`: `squash_value` returns strictly (0,1) and is monotone; raw-V inputs across [-1.6, 1.8] map into (0,1); perspective sign flips for the to-move seat; a terminal node uses the true 1/0 outcome, not the leaf.
- [x] T004 Implement `src/catan_rl/search/value.py` — `squash_value(v,a=3.22,b=-1.14)=sigmoid(a*v+b)` and `leaf_value(policy, env, *, perspective_seat)` -> squashed win-prob — to pass T003 (contract C1).
- [x] T005 [P] Write `tests/unit/search/test_priors.py`: `action_priors` keys are exactly the legal actions from `env.get_action_masks()`, probabilities sum to 1, no illegal action has nonzero prior, built as type-head x conditional sub-head priors.
- [x] T006 Implement `src/catan_rl/search/priors.py` — `action_priors(policy, env)` over the 6 autoregressive heads, legal-only + normalized — to pass T005 (contract C2).

- [x] **🔬 REVIEW GATE RG-Foundational** — run [`reviewers.md`](reviewers.md) (A+B) on the Phase 1+2 diff. Focus: value-squash bounds/perspective sign (C1), priors mask-consistency + normalization (C2), SearchConfig isolation/validation, additivity. Resolve BLOCKER/SHOULD-FIX before T007.

---

## Phase 3: User Story 1 — Bake-off gate (P1) 🎯 MVP / GO-NO-GO

**Goal**: a minimal determinized search beats the raw policy it wraps — the single decision that gates the whole multi-session effort.

**Independent test**: build only the minimal search + the env-access eval loop; `evaluate_search_vs_policy(search, raw_v6_1499)` over >=200 seat-symmetrized games yields a Wilson lower bound > 0.50 (or a documented FAIL).

- [x] T007 [P] [US1] Write `tests/unit/search/test_mcts.py`: PUCT select/expand/backup on a small node; deterministic under fixed seed; backup sign is correct across an EndTurn transition (the opponent's turn is folded into the agent's EndTurn step — value perspective must flip correctly).
- [x] T008 [US1] Implement `src/catan_rl/search/node.py` — `SearchNode` (cloned `CatanEnv`, `legal_types`, `priors`, `children`, `N`/`W`, `is_terminal`/`outcome`) per data-model.
- [x] T009 [US1] Implement the MINIMAL determinized PUCT in `src/catan_rl/search/mcts.py`: select by PUCT on the SQUASHED leaf value; expand one prior-sampled child per simulation; clone the env per simulation to fix that line's dice future (1 determinization); backup; terminal uses the true outcome. Fixed small sim budget; NO progressive widening yet. Passes T007.
- [x] T010 [P] [US1] Write `tests/unit/search/test_agent.py`: a forced move (1 legal action) short-circuits without spending budget; the returned action is always legal; `choose_action` does NOT mutate the passed env; identical seed+budget -> identical action sequence.
- [x] T011 [US1] Implement `src/catan_rl/search/agent.py` — `SearchAgent(policy, cfg).choose_action(env) -> np.ndarray` + `last_diagnostics` — to pass T010 (contract C3).
- [x] T012 [P] [US1] Write `tests/unit/search/test_eval_search.py`: `evaluate_search_vs_policy` is seat-symmetrized and returns a Wilson CI, mirroring `evaluate_policy_vs_policy` semantics; tiny n.
- [x] T013 [US1] Implement `src/catan_rl/search/eval_search.py` — `evaluate_search_vs_policy(search_cfg, search_ckpt, opponent_ckpt, *, n_games, seed, device="cpu", max_turns=400)`: a search-aware loop that hands `SearchAgent.choose_action(env)` the LIVE env, opponent = `FrozenSnapshotOpponent(opponent_ckpt)`, seat-symmetrized, CPU-pinned, torch RNG saved/restored — to pass T012 (contract C4).
- [x] T014 [US1] Add `tests/integration/test_search_smoke.py`: a tiny-budget search vs the heuristic plays a full game to completion with zero ruleset violations (assert via `eval/rules_invariants.py`).
- [x] T015 [US1] Implement `src/catan_rl/search/bakeoff.py` — `run_bakeoff(ckpt)`: minimal search vs the raw `ckpt`, PASS iff Wilson lower bound > 0.50 at n>=200 then re-confirmed at n>=500; returns `{passed, wr, ci, failure_mode}` (contract C6, encodes SC-001).
- [x] **🔬 REVIEW GATE RG-US1 (HARD-BLOCKING)** — run [`reviewers.md`](reviewers.md) (A+B) on the Phase 3 diff BEFORE T016. Focus: backup/perspective-sign across EndTurn (the silent killer), determinization/env-clone of the dice future, no env mutation in `choose_action`, legality, eval-loop RNG save/restore + seat symmetry, test strength (do they actually catch a sign flip / noise-not-lookahead?). T016 does NOT run until every BLOCKER/SHOULD-FIX is resolved.
- [x] T016 [US1] 🚦 **RUN THE GATE** ✅ **PASS** (quick n=200 WR 0.575/LB 0.506; confirm n=500 WR 0.578/LB 0.534; 0 violations; `runs/search/bakeoff_gate.json`): execute `run_bakeoff(runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt)`; write results to `runs/search/bakeoff_n200.json` + `bakeoff_n500.json`. **If PASS -> proceed to Phase 4/5. If FAIL (WR~0.50 or LB<=0.50) -> STOP; document the pivot (priors-weighted search / bounded-rollout-to-a-late-state / fix the leaf) in the spec notes and DO NOT build US2/US3.**

**Checkpoint**: US1 alone is a complete, decisive MVP — it answers "does lookahead beat the policy?" with a Wilson-bounded number.

---

## Phase 4: User Story 2 — Reproducible "thinking" opponent, strength scales (P2)  *(only if T016 PASSED)*

**Goal**: a configurable, deterministic agent that plays measurably stronger as its budget grows.

- [x] T017 [US2] Harden `src/catan_rl/search/mcts.py`: progressive widening on the type head (`ceil(pw_c * N^pw_alpha)` legal types), N-determinization aggregation, anytime `time_budget_s` (return best action found so far), optional `max_depth` leaf cut.
- [x] T018 [P] [US2] Extend `tests/unit/search/test_mcts.py`: progressive widening expands more types as visits grow; anytime returns within budget; N-determinization averages across cloned futures.
- [x] T019 [US2] Implement `src/catan_rl/cli/search_eval.py` and add the `catan-rl-search-eval` console-script to `pyproject.toml` (contract C5; additive entry; no GUI import).
- [x] T020 [P] [US2] Add `tests/unit/search/test_determinism.py`: same seed + budget reproduces an identical action sequence over a short game (SC-003).
- [~] T021 [US2] (RUNNING — sims ladder, runs/search/ladder.pid) Run the strength-budget ladder (quickstart Scenario 2): search vs raw v6 at 0.25/1/5 s per move; assert WR is monotone in budget; write `runs/search/ladder_{0.25,1,5}s.json` (SC-002).
- [x] **🔬 REVIEW GATE RG-US2** — run [`reviewers.md`](reviewers.md) (A+B) on the Phase 4 diff. Focus: progressive-widening schedule, N-determinization aggregation, anytime/time-budget returns best-so-far, CLI off training path + no GUI import. Resolve before Phase 5.

---

## Phase 5: User Story 3 — Elo uplift on the strength ladder (P3)  *(only if T016 PASSED)*

- [ ] T022 [US3] Promote the ladder harness to a committed `scripts/elo_ladder.py` (from `/tmp/elo_ladder.py`) that can include a `SearchAgent` rung; run a round-robin with `search@1s` + `v6_u1499` (+ a couple of v6 rungs).
- [ ] T023 [US3] Report `search@1s` Elo delta over raw `v6_u1499` with CI to `runs/search/elo_uplift.json`; confirm positive + non-overlapping CI when the gate passed (SC-004).

---

## Phase 6: Polish & Cross-Cutting

- [ ] T024 [P] No-regression: run `pytest tests/unit/eval tests/integration/test_selfplay_smoke.py` — confirm existing eval/training behavior is byte-identical with search present-but-unused (SC-005).
- [ ] T025 [P] Green CI on the new module: `ruff check` + `mypy --strict` over `src/catan_rl/search/` and `src/catan_rl/cli/search_eval.py`; fix nits.
- [ ] T026 Update `MEMORY.md` + `project_inference_search_viability.md` with the bake-off outcome (Elo uplift, or the documented pivot if it failed); refresh `specs/003-inference-search/` notes. No new docs.

---

## Dependencies & Story Completion Order

- **Setup (T001)** → **Foundational (T002–T006)** → **US1 / gate (T007–T016)** → *[gate PASS]* → **US2 (T017–T021)** + **US3 (T022–T023)** → **Polish (T024–T026)**.
- **US1 is the MVP and a hard gate**: T016 PASS is required before any US2/US3 task.
- Within US1: node (T008) → minimal MCTS (T009) → agent (T011) → eval loop (T013) → bakeoff (T015) → **run gate (T016)**. Test tasks (T007, T010, T012, T014) precede their implementations (test-first).
- US2 and US3 are independent of each other (both depend only on the gate).

## Parallel Execution Examples

- **Foundational**: T002, T003, T005 together (different files); then T004 (after T003), T006 (after T005).
- **US1 tests**: T007, T010, T012 together (different test files), each before its implementation.
- **Polish**: T024, T025 together.

## Independent Test Criteria (per story)

- **US1**: `evaluate_search_vs_policy(search, raw_v6_1499)` over ≥200 games → Wilson LB > 0.50 (go) or a documented FAIL — testable with only Phases 1–3 built.
- **US2**: a budget ladder (0.25/1/5 s) shows monotone WR vs raw v6, and same-seed runs reproduce identical actions.
- **US3**: the search rung's Elo on the ladder is positive over raw `v6_u1499` with a non-overlapping CI.

## MVP Scope

**US1 (T001–T016)** is the MVP and the go/no-go. If the bake-off passes, US2 (usable thinking opponent) and US3 (the Elo number) follow; if it fails, the feature stops at US1 with a documented pivot — still a decisive, valuable outcome that cost only the minimal prototype.
