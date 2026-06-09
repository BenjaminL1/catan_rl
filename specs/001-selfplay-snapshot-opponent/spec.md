# Feature Specification: Frozen-policy self-play opponent + policy-vs-policy evaluation

**Feature Branch**: `001-selfplay-snapshot-opponent`

**Created**: 2026-06-07

**Status**: Draft

**Input**: User description: "Frozen-policy self-play opponent + policy-vs-policy evaluation — wire the single primitive that lets the agent play against frozen snapshots of its own past selves (unblocking self-play) and lets the champion be measured head-to-head against any loaded policy."

## Clarifications

### Session 2026-06-08 (after senior-RL review)

- **Opponent turn-driver scope**: build the **full opponent sub-turn state
  machine** — the snapshot opponent plays a complete Catan turn (roll → knight/
  robber → road-builder → dev-card plays → bank trades → build → EndTurn),
  identical to the agent. No constrained subset (a crippled opponent voids the
  self-play signal).
- **Engine-apply path**: extract a **shared `_apply_action(player, action)`**
  helper from `step()`, used by both agent and opponent — one code path, no
  rules drift (Constitution II). Behavior-identical regression test on the
  agent path required.
- **Opponent action sampling (rollout)**: **stochastic with an isolated
  `torch.Generator`** seeded from the env/league seed — faithful to the policy's
  distribution AND does not perturb the learner's rollout RNG (so before/after
  reproducibility holds; satisfies FR-006).
- **Turn completion contract**: the opponent's main turn **runs to completion
  inside one `env.step`** with a **hard per-turn action cap** that force-ends the
  turn on exceed (logged as an anomaly). The 7-roll / agent-discard interleave
  continues to use the existing `_opp_pending` suspension mechanism.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train against a frozen past self (Priority: P1)

A frozen snapshot of a previously-trained policy can act as the in-env
opponent. When the league assigns an environment a snapshot opponent, that
environment's opponent moves are produced by the loaded frozen policy — not the
fixed heuristic — and a self-play rollout runs end to end without error.

**Why this priority**: This is the keystone. Without a learning opponent, the
agent cannot improve past "beats a weak heuristic." This single capability also
underpins best-response probing and human-vs-policy play. It is the MVP — if
only this ships, self-play becomes possible.

**Independent Test**: With `snapshot_weight = 0.5` and a one-entry league whose
snapshot is a deterministic stub policy that always ends its turn, run a short
rollout; confirm it completes and the opponent's observed actions come from the
stub (the heuristic is bypassed).

**Acceptance Scenarios**:

1. **Given** a non-empty league and `snapshot_weight > 0`, **When** a rollout
   runs, **Then** it completes with no `NotImplementedError`.
2. **Given** an env assigned a known stub snapshot, **When** the opponent takes
   its turn, **Then** the action taken matches the stub policy's output, not the
   heuristic's.
3. **Given** the same seed and the same snapshot id, **When** the rollout is
   repeated, **Then** the opponent's behavior is identical.

---

### User Story 2 - Measure the champion against any policy (Priority: P2)

The champion can be evaluated head-to-head against any loaded opponent policy —
a league snapshot or a saved checkpoint — over a set of seat-symmetrized games,
returning a win rate with a confidence interval. This is the only way to tell
whether the agent is improving once it has saturated the heuristic.

**Why this priority**: Self-play without a strength signal is flying blind.
Policy-vs-policy eval turns the league into a measurable strength ladder and
provides the best-response / champion-gate machinery later phases depend on.

**Independent Test**: Evaluate the champion against a frozen checkpoint over
N≥100 symmetrized games; confirm a finite win rate + Wilson 95% CI is returned
and is reproducible at the same seed.

**Acceptance Scenarios**:

1. **Given** two loaded policies, **When** policy-vs-policy eval runs, **Then**
   it returns a win rate and a Wilson 95% CI over N symmetrized games.
2. **Given** the same seed, **When** the eval is repeated, **Then** the result
   is bit-for-bit identical.

---

### User Story 3 - Bring fresh snapshots into play (Priority: P3)

As training proceeds and new snapshots are added to the league, subsequent
rollouts can be told the new opponent assignment so freshly-added snapshots
actually enter play, under a configurable heuristic-vs-snapshot opponent mix.

**Why this priority**: Today the opponent mix is frozen when the run is
constructed, so new snapshots can never be played. Without this, the league
pool grows but the agent keeps training against the same opponents.

**Independent Test**: Add a snapshot mid-run, refresh the opponent assignment,
and confirm the next rollout's opponent is the newly added snapshot.

**Acceptance Scenarios**:

1. **Given** a snapshot added after construction, **When** the opponent
   assignment is refreshed, **Then** the next rollout uses the new snapshot.
2. **Given** a configured heuristic:snapshot mix, **When** opponents are
   assigned, **Then** the observed mix matches the configured ratio in
   expectation.

---

### Edge Cases

- **Empty league but `snapshot_weight > 0`**: the system MUST fall back to a
  non-snapshot opponent (heuristic/random) rather than erroring — no snapshot
  exists to load yet (true at the very start of self-play).
- **Requested snapshot id evicted from the pool**: the assignment MUST be
  resolved against the current pool (skip/replace) rather than loading a stale
  or missing snapshot.
- **An incompatibly-shaped policy checkpoint loaded for eval (US2)**: MUST be
  rejected with a clear error referencing checkpoint back-compat, never silently
  mis-loaded. (In-league snapshots from US1 are shape-correct by construction;
  this guard applies to externally-loaded checkpoints.)
- **Opponent inference device differs from the learner**: opponent inference
  follows the learner device; eval inference runs on CPU.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A league snapshot MUST be usable as an in-env opponent; when
  selected, the opponent's moves are produced by a frozen policy loaded from the
  league pool, not by the heuristic.
- **FR-002**: With a non-empty league and `snapshot_weight > 0`, a rollout MUST
  complete without raising `NotImplementedError` (both existing guards removed).
- **FR-003**: Snapshot-opponent inference MUST be batched across environments in
  the main process (not per-environment, batch-of-one).
- **FR-004**: The opponent assignment for each environment MUST be refreshable
  between rollouts so newly-added snapshots can enter play.
- **FR-005**: The system MUST provide policy-vs-policy evaluation: a champion
  policy vs any loaded opponent policy (snapshot or checkpoint) over N
  seat-symmetrized games, returning a win rate and a Wilson confidence interval.
- **FR-006**: The snapshot opponent MUST sample actions stochastically using an
  **isolated `torch.Generator`** (seeded from the env/league seed), so it does
  NOT advance the learner's rollout RNG stream. Behavior MUST be reproducible
  given the same seed, snapshot id, and device — the same action *sequence* is
  produced. (Bit-for-bit numerical identity is only guaranteed on CPU; batched
  MPS/GPU inference may differ in low-order bits across batch groupings without
  changing the sampled actions.)
- **FR-007**: The opponent mix MUST be configurable as a static heuristic:snapshot
  ratio (a scheduled/annealed mix is out of scope).
- **FR-008**: The feature MUST NOT change observation or action-head shapes; the
  existing opponent-id embedding MUST be fed real values without being resized.
- **FR-009**: The 1v1 ruleset, reward function, and action space MUST be
  unchanged.
- **FR-010**: Existing policy checkpoints MUST remain loadable (back-compat).
- **FR-011**: When the league is empty or a requested snapshot is unavailable,
  the system MUST gracefully fall back to a non-snapshot opponent.
- **FR-012** (correctness-critical): The opponent's policy input MUST be built
  from the **opponent's** point of view — it sees its OWN hidden dev cards and
  only the agent's *played* cards, and the hand-tracker perspective is the
  opponent's. The agent's hidden information (hidden dev cards, exact hand)
  MUST NOT appear in the opponent's observation. A test MUST assert no leakage.
- **FR-013**: The opponent's main turn MUST terminate — a hard per-turn action
  cap force-ends the turn if exceeded (logged as an anomaly), preventing
  livelock from a policy that never samples EndTurn.

### Key Entities

- **League snapshot**: a frozen past policy — its parameters, a stable id, and
  the training step at which it was captured.
- **Opponent assignment**: a per-environment mapping to an opponent kind
  (heuristic / random / snapshot) and, for snapshots, a concrete snapshot id.
- **Frozen opponent policy**: a policy instantiated from a snapshot and run
  inference-only (never trained) to produce opponent moves.
- **Eval matchup result**: the win rate and Wilson confidence interval for a
  champion vs a loaded opponent over a set of symmetrized games.

## Success Criteria *(mandatory)*

### Measurable Outcomes (this feature)

- **SC-001**: A self-play run (`snapshot_weight = 0.5`, non-empty league) trains
  for ≥1M steps with zero `NotImplementedError` or device errors.
- **SC-002**: In a controlled stub-opponent test, 100% of the snapshot
  opponent's actions originate from the loaded snapshot policy (the heuristic is
  never invoked for a snapshot opponent).
- **SC-004**: Policy-vs-policy eval returns a finite win rate with a Wilson 95%
  CI for any two loaded policies over N≥100 seat-symmetrized games, and is
  reproducible bit-for-bit at a fixed seed **when run on CPU** (the eval device).

### Downstream phase gates (OUT OF SCOPE for this feature)

These are validated by later phases and full training runs, NOT by this
plumbing PR. They depend on machinery this feature explicitly defers (a complete
self-play run, plus the PFSP / best-response / diversity work of a later phase).
Listed only so planning does not pull them into scope.

- **PG-1**: Over a self-play run, the agent's win rate against its own recent
  snapshots stays within 40–60% (healthy zero-sum equilibrium) while its win
  rate against a frozen early baseline rises by ≥10 percentage points.
- **PG-2**: Symmetrized win rate ≥ 0.90 vs the heuristic (the bootstrap /
  graduate-to-self-play bar — v1 only ever reached ~0.55, so this also subsumes
  "stronger than v1" with no v1 policy loaded), and a fresh 1M-step
  best-response adversary cannot exceed 0.65 win rate against the champion.

## Assumptions

- The in-flight `bootstrap_v1` checkpoint seeds the first league snapshot; the
  no-shape-change constraint exists precisely so that checkpoint stays loadable.
- When the league pool holds more than one snapshot, selection is
  uniform-random (priority-based selection / PFSP is a separate later phase).
- The opponent mix is a single static heuristic:snapshot ratio (e.g. 60:40);
  any schedule/annealing is deferred.
- Opponent inference during rollout runs on the learner device, batched;
  evaluation inference runs on CPU (consistent with the existing eval policy).
- The existing perfect 1v1 hand-tracker and observation encoder are reused
  unchanged — no new observation surface is introduced.

## Known gaps (US1 MVP — tracked, not bugs)

- **Opponent's own 7-roll discard stays heuristic** (`opp.discardResources`), not
  policy-driven. In 1v1 with perfect hand-tracking this is a minor strength
  asymmetry, not a correctness or info-leak issue — but Phase 4 eval must not
  read it as a policy-strength signal. Low-leverage; revisit if discards matter.
- **Opponent inference is per-env-sequential, not batched across envs** (T019).
  The in-env run-to-completion driver (D11) plays the opponent's whole turn
  inside one `env.step`, so the D1 "batch across envs" optimisation does not
  apply as written — reframed as a **performance follow-up**, not a correctness
  item. The MVP is correct; at n_envs=128 the per-env forward is the cost.
