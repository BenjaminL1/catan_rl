# Feature Specification: Scale Inference Search (v8 base)

**Feature Branch**: `006-scale-inference-search`

**Created**: 2026-06-20

**Status**: Draft

**Input**: Scale determinized PUCT-MCTS inference search on the v8 champion toward superhuman 1v1 Catan, after the raw-policy population/exploiter program concluded null. Re-baseline the banked +54.6-Elo search uplift (measured on v6_u1499) onto v8, find how far the budget→strength curve rises before it plateaus, make higher budgets tractable if they keep helping, and bank a new deployed Elo ceiling over v8.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-baseline search on v8 + budget→strength curve (Priority: P1)

The operator measures whether determinized search still buys strength on the **stronger v8 base** (not the v6 it was tuned on) and how far the **simulation-budget → win-rate curve** rises before it plateaus. This is the gate-first probe and the core measurement; on its own it answers "is scaling search on v8 worth anything, and at what budget."

**Why this priority**: Everything else (the speed lever, the knob scan, the banked ceiling) is wasted if search does not beat v8 or if the curve is already flat at the budgets we can afford. This story is the MVP — it produces the go/no-go gate and the budget the project should deploy at.

**Independent Test**: Run search@{50,100,200,400} (base v8) self-match vs raw v8, seat-symmetric, CPU, fixed `sims_per_move` seed; report each budget's WR + Wilson CI + Elo uplift over raw v8; confirm the gate (Wilson LB > 0.50 at ≥1 budget) and identify the plateau/best budget. Delivers a decision without any new code.

**Acceptance Scenarios**:

1. **Given** frozen v8 (`runs/anchors/v8_promobar_u243.pt`), **When** search@50 and search@100 (base v8) play seat-symmetric self-matches vs raw v8 at n≥200, **Then** the run records each WR + Wilson CI and a GATE verdict (search beats raw v8 iff Wilson LB > 0.50); a failing gate halts the build and is reported as such.
2. **Given** the gate passes, **When** the budget curve {50,100,200,400} (base v8) is run, **Then** a monotone-or-plateauing WR/Elo-uplift curve with Wilson CIs is produced, the plateau budget is identified, and the chosen best budget's uplift over raw v8 is measured at n≥500 (promotion-grade).
3. **Given** any budget eval, **When** it runs, **Then** it is CPU-pinned, seat-symmetric, reproducible at a fixed `sims_per_move` seed, and reports rules-violation count (must be 0).

---

### User Story 2 - Batched leaf-value forward (tractable higher budgets) (Priority: P2)

If the budget curve still rises at 200/400 sims but those budgets are too slow at the current ~120 sims/sec (NN-forward-bound), the operator enables an **additive, default-off** speed optimization that evaluates several pending MCTS leaves' value/priors in one network forward pass instead of one-at-a-time — a pure speed change that never alters search decisions.

**Why this priority**: Only needed if US1 shows headroom above ~100 sims. It unlocks the strongest deployable budget within a practical per-move time, but it is an enabler, not the result.

**Independent Test**: With the flag off, search is byte-identical to today. With it on, at a fixed seed the search produces **identical root visit counts and chosen actions** as the unbatched path (proven by a unit/integration test), while wall-clock sims/sec improves measurably.

**Acceptance Scenarios**:

1. **Given** a fixed seed and a position, **When** search runs with batched-leaf-forward ON vs OFF, **Then** the root visit-count vector and the selected action 6-tuple are identical (bit-for-bit), proven by an automated test.
2. **Given** the flag absent/off, **When** any existing search/eval runs, **Then** behavior and outputs are byte-identical to the pre-feature baseline (default-off additivity).
3. **Given** the flag on at a higher budget, **When** throughput is measured, **Then** sims/sec is higher than the unbatched path at the same budget (the optimization actually pays off).

---

### User Story 3 - Knob scan + bank the re-baselined ceiling (Priority: P3)

The operator runs a small ablation over already-exposed `SearchConfig` knobs (`c_puct`, `n_determinizations`, `fpu_mode`, `root_dirichlet_alpha`) at the chosen best budget, picks the best config under a no-regression gate, re-baselines the `elo_ladder.py` search rung on the v8 base, and **banks the new search-vs-v8 Elo uplift as the deployed ceiling** (the v6_u1499 +54.6 number stays on record).

**Why this priority**: Polish + the durable artifact (the new deployed number). Depends on US1's best budget and benefits from US2's speed.

**Independent Test**: Each knob change is accepted only if a no-regression eval vs raw v8 (Wilson LB not worse) passes; the final ladder run adds a v8-base search rung and writes a new banked uplift JSON without renaming any existing scalar/file.

**Acceptance Scenarios**:

1. **Given** the best budget, **When** a knob is changed and evaluated vs raw v8, **Then** it is banked only if it does not regress (Wilson LB ≥ the prior config's), else reverted.
2. **Given** the best config + budget, **When** the ladder is re-run with a v8-base search rung, **Then** a new `search@N(v8)` Elo uplift is recorded as the deployed ceiling and the prior v6_u1499 +54.6 reference remains intact in the record.

---

### Edge Cases

- **Gate fails** (search does not beat raw v8, Wilson LB ≤ 0.50): halt — do not build US2/US3; report that search does not scale on v8 and reconsider direction.
- **Curve plateaus at/below 100 sims**: deploy the best sub-plateau budget; do NOT build the batched-leaf speed lever (US2) — there is no higher budget worth enabling.
- **Batched-leaf forward is NOT bit-identical** at a fixed seed: it is a correctness bug, not a tuning choice — block until identical (the optimization must never change decisions).
- **Higher budget increases rules violations** (>0): treat as a search/engine bug; block the budget.
- **Determinization interacts with budget** (e.g., `n_determinizations>1` changes the best budget): report the (sims × determinizations) trade-off honestly rather than only the raw sims axis.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST evaluate determinized search on the **v8 base** (`runs/anchors/v8_promobar_u243.pt`) as a seat-symmetric self-match vs raw v8, reusing the shipped `evaluate_search_vs_policy` path and `catan-rl-search-eval` CLI — no new evaluation algorithm.
- **FR-002**: The system MUST record a **gate verdict** (search beats raw v8 iff Wilson lower bound > 0.50, n≥200) BEFORE any new code is built, and MUST halt the build on a failing gate.
- **FR-003**: The system MUST produce a **budget→strength curve** over `sims_per_move ∈ {50,100,200,400}` (base v8) with per-budget WR + Wilson CI + Elo uplift over raw v8, and MUST identify the plateau / best budget.
- **FR-004**: The system MUST measure the chosen best budget's uplift over raw v8 at **n≥500** (promotion-grade), distinct from the n=100 monitoring cadence.
- **FR-005**: IF higher budgets retain headroom but are too slow, the system MAY add a **batched leaf-value forward** path that MUST be opt-in via config (default-off) and MUST yield **bit-identical** root visit counts and chosen actions to the unbatched path at a fixed seed.
- **FR-006**: All search evals MUST be **CPU-pinned**, **seat-symmetric**, and reproducible via fixed `sims_per_move` (NOT `time_budget_s`, which is not bit-reproducible).
- **FR-007**: The system MUST preserve the shipped search design — **max-only tree, no per-ply value sign-flip, open-loop determinization** — introducing no opponent or chance nodes.
- **FR-008**: Any knob change (`c_puct`, `n_determinizations`, `fpu_mode`, `root_dirichlet_alpha`) MUST be gated by a **no-regression eval** vs raw v8 before being banked, else reverted.
- **FR-009**: The system MUST re-baseline the `scripts/elo_ladder.py` search rung on the v8 base at the chosen config/budget and **bank** the new search-vs-v8 Elo uplift as the deployed ceiling, **append-only** (the v6_u1499 +54.6 reference stays recorded; no existing scalar/file renamed).
- **FR-010**: With the new flags unused, existing search, training, and eval behavior MUST be **byte-identical** to the pre-feature baseline (additive, default-off).
- **FR-011**: All checkpoints MUST load `strict=True` (v2-lineage); the feature MUST NOT change the policy state-dict shape, the engine ruleset, the 6-head action space, or the observation schema.
- **FR-012**: No GUI import on any search/eval path (headless-safe).

### Key Entities *(include if feature involves data)*

- **Search budget point**: a `(base_ckpt, sims_per_move, n_determinizations, config)` → `(WR, Wilson CI, Elo uplift, rules_violations, n)` measurement vs raw v8.
- **Budget→strength curve**: the ordered set of budget points over {50,100,200,400} on the v8 base, plus the identified plateau/best budget.
- **Banked ceiling**: the deployed `search@N(v8)` Elo uplift recorded in the ladder/JSON, alongside (not replacing) the historical v6_u1499 +54.6 reference.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A recorded v8 re-baseline **gate result** exists (search@50 or @100 vs raw v8, Wilson LB vs 0.50, n≥200) and is evaluated before any new code lands.
- **SC-002**: A **budget→strength curve** {50,100,200,400} on the v8 base is produced with Wilson CIs, the plateau budget identified, and the best budget's uplift over raw v8 measured at n≥500 with its Elo delta.
- **SC-003**: IF built, the batched-leaf-forward path is proven **bit-identical** to unbatched search at a fixed seed by an automated test, is default-off (existing search byte-identical), and demonstrably raises sims/sec at a higher budget.
- **SC-004**: A new **search@N(v8) deployed-ceiling** Elo uplift is banked (append-only), with the historical v6_u1499 +54.6 reference still on record.
- **SC-005**: Across all banked evals, **0 rules violations** and full seat-symmetry/CPU/fixed-seed reproducibility hold.
- **SC-006**: With the feature's flags unused, a no-op smoke on existing search/eval produces **identical** outputs to the pre-feature baseline.

## Assumptions

- Search still helps on v8 (the value head ranks states well; the +54.6 Elo on v6 and the monotone v6 budget ladder make a v8 uplift the expected, not guaranteed, outcome — hence the FR-002 gate).
- `sims_per_move` (fixed budget) is the banked axis for reproducibility; `time_budget_s` is a secondary convenience only.
- CPU is the eval device per the project device policy; throughput is ~120 sims/sec (NN-forward-bound), so @200/@400 are feasible-but-slow without batching.
- The Rust engine is NOT the lever (search is NN-forward-bound, not engine-bound) and is out of scope.
- piKL-Hedge search is scaffolded (`src/catan_rl/algorithms/pikl/`) but **out of scope** here — a separate later feature.
- No training changes (value-target sharpening, self-play, exploiters) and no website/product work are in scope.
- v8 (`runs/anchors/v8_promobar_u243.pt`) is the frozen base; the v8-cont ckpt_524 and v6/v7 anchors remain available for the ladder but are not the search base.
