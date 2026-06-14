# Feature Specification: Inference-Time Search (Determinized MCTS)

**Feature Branch**: `003-inference-search`

**Created**: 2026-06-14

**Status**: Draft

**Input**: User description: "Inference-time SEARCH (determinized MCTS) layered on the trained PPO policy for 1v1 Settlers of Catan, to add the lookahead the reactive single-forward-pass policy lacks. v6 plateaued (Elo-confirmed); search is the human-level lever. De-risk probes passed (value head ranks + calibrates, engine ~120 sims/sec, clones faithful). Offline-only first; bake-off gate is the first deliverable."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Bake-off gate: does lookahead beat the reactive policy? (Priority: P1)

A researcher runs a minimal search agent at a small fixed budget head-to-head against the raw policy it wraps, over a statistically meaningful number of seat-symmetrized games, to make a go/no-go decision **before** any larger investment. This is the MVP: if a tiny-budget search beats the raw policy, lookahead demonstrably adds value and the build proceeds; if it does not, the build stops here and the failure mode + recommended pivot are documented.

**Why this priority**: It is the single decision that gates the entire (multi-session) effort. It converts the search plan from a blind bet into an evidence-based one for the cost of a minimal prototype, and directly retires the one residual risk (the value/priors are validated only on-policy, but search visits off-distribution states).

**Independent Test**: Build only the minimal search + the search-aware head-to-head loop; verifiable by the win-rate + its Wilson confidence interval alone — no other component (budget ladder, Elo rung) is required.

**Acceptance Scenarios**:

1. **Given** the v6 frontier checkpoint loaded as both the search agent's policy and the raw opponent, **When** the search agent at a small fixed budget plays ≥200 seat-symmetrized games vs the raw policy, **Then** it wins with a Wilson lower bound strictly above 0.50 — or the run halts and the failure mode is documented.
2. **Given** the gate passes at n≥200, **When** it is re-run at n≥500, **Then** the Wilson lower bound remains above 0.50.

---

### User Story 2 - A reproducible "thinking" opponent whose strength scales with compute (Priority: P2)

A user configures the search budget (simulations per move or wall-clock per move) and gets a deterministic, reproducible agent that plays measurably stronger as the budget grows — a usable stronger opponent for analysis and evaluation.

**Why this priority**: This is the usable artifact, and the budget-vs-strength curve is the proof that the gain comes from search rather than from incidental noise.

**Independent Test**: Run a budget ladder (e.g., 0.25 / 1 / 5 s per move) vs the raw policy and confirm monotonically increasing win-rate plus exact reproducibility under a fixed seed.

**Acceptance Scenarios**:

1. **Given** a fixed seed and budget, **When** the agent selects actions twice, **Then** the two action sequences are identical.
2. **Given** an increasing compute budget across at least three settings, **When** each is measured vs the raw policy, **Then** win-rate increases monotonically (within confidence intervals).
3. **Given** any reachable game state, **When** the agent acts, **Then** the chosen action is legal under the 1v1 ruleset.

---

### User Story 3 - Search uplift quantified on the strength ladder (Priority: P3)

A researcher places the search agent on the existing Elo ladder to report its uplift, in Elo, over the raw policy it wraps.

**Why this priority**: Turns the result into a single comparable number on the project's scoreboard, alongside the existing checkpoint lineage.

**Independent Test**: Add the search agent as a ladder rung and run the round-robin; verifiable by its Elo delta over the raw policy with a confidence interval.

**Acceptance Scenarios**:

1. **Given** the search agent and the raw policy both on the ladder, **When** the round-robin completes, **Then** the search agent's Elo is reported with a confidence interval and is positive (non-overlapping with the raw policy) exactly when the bake-off gate passed.

---

### Edge Cases

- **Forced moves** (setup phase, or exactly one legal action): search must short-circuit — return the forced action without spending budget.
- **Non-build decision nodes** (discard on a 7, robber placement, dev-card play, bank trade): handled by the same legal-action expansion, never special-cased away.
- **Off-distribution / poorly-calibrated leaf states**: search must not end up *weaker* than the raw policy; the bake-off gate is the explicit catch, and a pivot (lean on priors over value, or bounded-rollout to a better-calibrated late state) is the documented fallback.
- **Determinization variance**: a single sampled future must not dominate the decision — the value of a line is aggregated over multiple sampled futures.
- **Budget exhausted mid-search**: the agent returns the best action found so far (anytime behavior).
- **Terminal reached inside a simulated line** (a player hits 15 VP): the true game outcome is used at that node, not a leaf estimate.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a search agent that, at an agent decision point, selects a legal action by simulating possible futures from the LIVE game state (lookahead), rather than by a single reactive forward pass.
- **FR-002**: The search agent MUST be evaluatable head-to-head against any policy checkpoint over seat-symmetrized games using the existing evaluation machinery, and MUST be placeable as a rung on the existing strength ladder.
- **FR-003**: The search MUST handle game stochasticity (dice, dev-card draws, robber) by sampling concrete futures and aggregating over them, WITHOUT sampling a hidden opponent hand (perfect 1v1 hand-tracking makes the opponent's hand known).
- **FR-004**: The search MUST use the trained policy's action priors to focus exploration and a leaf evaluator derived from the trained value head, with the leaf value mapped to a bounded win-probability before it is aggregated.
- **FR-005**: The search MUST scale its strength with its compute budget (more budget → measurably stronger play), with the budget exposed as a configurable simulations-per-move or wall-clock-per-move setting.
- **FR-006**: The search MUST be reproducible — a fixed seed and budget yield identical action choices.
- **FR-007**: The search MUST only ever select actions that are legal under the 1v1 ruleset, and MUST NOT modify or violate the engine's rules (it simulates the existing engine).
- **FR-008**: The system MUST provide a go/no-go bake-off — a minimal search measured against the raw policy with an explicit statistical decision rule (Wilson lower bound > 0.50) — and MUST document the failure mode and recommended pivot if the gate fails.
- **FR-009**: The search components MUST be additive and isolated: the training path, policy network architecture, observation/action space, and checkpoint format remain UNTOUCHED, and default (non-search) evaluation/training behavior remains byte-identical.
- **FR-010**: The search MUST run on the CPU-pinned evaluation device and use existing v2-lineage checkpoints as-is (no state-dict change, no migration).
- **FR-011**: Search behavior MUST be observable — for the chosen action, the estimated value / visit distribution and the budget consumed are reported for analysis.

### Key Entities *(include if feature involves data)*

- **Search Node**: a clonable game state at a decision point, holding per-candidate-action statistics (visit counts, aggregated value) and the legal-action set at that state.
- **Search Agent**: wraps a frozen policy (action priors + leaf value), a compute budget, and a seed; maps a live game state to a chosen legal action.
- **Determinization**: one concrete sampled future (dice / dev-card draws) along which a line is evaluated; the line's value is aggregated across determinizations.
- **Bake-off / Search-aware Eval Loop**: a play loop that hands the search agent the live environment to clone and simulate, runs search-vs-policy games, and reports win-rate, confidence interval, and Elo.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: At a small fixed search budget, the search agent beats the raw policy it wraps with a Wilson lower bound above 0.50 over at least 500 seat-symmetrized games (the go signal).
- **SC-002**: Win-rate versus the raw policy increases monotonically across at least three increasing compute budgets (strength scales with search, not noise).
- **SC-003**: An identical seed and budget reproduce the exact action sequence (100% determinism) across repeated runs.
- **SC-004**: When the gate passes, the search agent's Elo on the existing ladder is positive over the raw policy with a non-overlapping confidence interval.
- **SC-005**: With search absent/disabled, existing evaluation and training behavior is byte-identical (no regression).
- **SC-006**: Zero 1v1-ruleset violations are recorded across all bake-off games (the engine and rules remain untouched).

## Assumptions

- The trained value head is a usable and calibratable leaf evaluator — validated: Spearman 0.69 vs outcome on peer games, with a monotone squash to win-probability fitted at ECE 0.039. The search aggregates the **squashed** value, because raw value is unbounded ([-1.6, +1.8], ~27% outside [-1, 1]).
- The pure-Python engine is fast enough for an offline budget (~120 simulations/sec measured); the Rust engine is **not** required because search is bounded by neural-network evaluation, not by the engine. Batched leaf evaluation is the throughput lever if more simulations are needed.
- A deep copy of the game state is a faithful, independent search node (validated: clones reproduce the same result under the same action and do not mutate the original).
- The large autoregressive action space is expanded via the small legal branching factor of the action-type choice (~2-6 mid-game), with conditional sub-choices drawn from the policy priors — a planning decision that avoids combinatorial blow-up.
- The specific search algorithm (e.g., determinized PUCT-MCTS vs depth-limited expectimax) is a **planning** decision resolved in `research.md`; this spec requires only the *behavior* — lookahead, value-based leaf evaluation, and strength that scales with budget.
- Scope is **offline only**. In-the-loop expert iteration (search as a league teacher), piKL-Hedge regularized search, the Rust engine, and any retraining of the policy or value head are explicitly out of scope (later phases).
- Constitution alignment: the 1v1 ruleset stays sacred (engine untouched, only legal actions proposed); self-play stays 2-player zero-sum (search models the symmetric game and the known opponent hand); artifacts stay additive (new isolated module, no checkpoint/obs/action changes, append-only TensorBoard scalars); evaluation stays CPU-pinned; and the bake-off, determinism, and no-regression checks are the test-first gates.
