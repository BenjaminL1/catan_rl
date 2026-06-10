# Feature Specification: Prioritized Fictitious Self-Play (PFSP) Opponent Sampling

**Feature Branch**: `002-pfsp-opponent-sampling`

**Created**: 2026-06-10

**Status**: Draft

**Input**: User description: PFSP opponent sampling for the 1v1 Catan self-play league — sample past-snapshot opponents weighted by how hard they currently are for the learner, to arrest the catastrophic-forgetting drift (v2/v3 strength-vs-baseline peaked ~0.77 then collapsed ~0.43 under uniform sampling). Complements the shipped frozen-baseline anchor.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Win-rate-weighted opponent selection that arrests self-play drift (Priority: P1)

Instead of sampling past-snapshot opponents uniformly, the training run maintains how often the learner currently beats each snapshot and preferentially trains against the snapshots it is NOT reliably beating. Learning focuses on the agent's current weaknesses, countering the forgetting that made v2/v3 peak then collapse.

**Why this priority**: This *is* the feature — the prioritization mechanism that fixes the observed regression. Nothing else matters without it.

**Independent Test**: With a synthetic league where the learner's win rate vs snapshot A is low and vs snapshot B is high, draw a rollout's opponent assignment under the "hard" curve and verify A is assigned to strictly more envs (in expectation) than B; and that recording game outcomes moves each snapshot's estimate toward the observed win rate.

**Acceptance Scenarios**:

1. **Given** a non-empty snapshot pool with known per-opponent win rates and the "hard" curve, **When** the per-rollout opponent assignment is drawn, **Then** a snapshot with a lower win rate is assigned to more envs (in expectation) than one with a higher win rate.
2. **Given** a rollout completes games against assigned snapshot opponents, **When** outcomes are recorded, **Then** each snapshot's win-rate estimate moves toward the observed agent win rate against that specific snapshot.
3. **Given** PFSP is disabled (default), **When** assignments are drawn, **Then** the opponent assignment is identical to the current uniform-pool behaviour.

---

### User Story 2 - Cold-start for fresh snapshots (Priority: P2)

A newly added snapshot has no game history; it must still be sampled (not starved by snapshots with established win rates) so it accrues data and enters the prioritization.

**Why this priority**: Without it, the agent's latest selves might never be played under a pure win-rate weighting — but the P1 mechanism is demonstrable without it, so it's a required refinement, not the core.

**Independent Test**: Add a snapshot with zero games to a pool of established snapshots; draw a rollout and verify the new snapshot receives a non-zero share of envs.

**Acceptance Scenarios**:

1. **Given** a snapshot with fewer than the minimum games for a reliable estimate, **When** assignments are drawn, **Then** it receives at least the cold-start sampling weight.
2. **Given** a snapshot accumulates games past the cold-start threshold, **When** assignments are drawn, **Then** its weight transitions to the win-rate-based weight.

---

### User Story 3 - Resumable + reproducible prioritization (Priority: P3)

The per-opponent win-rate state is part of the run's saved state, so a resumed run keeps prioritizing correctly, and seeded sampling makes the opponent sequence reproducible.

**Why this priority**: Important for long runs and reproducibility, but the prioritization works within a single run without it.

**Independent Test**: Run N updates with PFSP, checkpoint, restore, and verify the win-rate estimates match exactly; a same-seed continuation reproduces the opponent-assignment sequence.

**Acceptance Scenarios**:

1. **Given** a run with accumulated win-rate estimates, **When** the checkpoint is saved and reloaded, **Then** the restored estimates equal the saved ones.
2. **Given** two runs with the same seed and league state, **When** assignments are drawn, **Then** the opponent-assignment sequences are identical.

---

### Edge Cases

- **Empty / fully-evicted pool**: fall back to the non-snapshot kinds (existing FR-011 behaviour); PFSP contributes nothing.
- **Snapshot evicted while holding win-rate state**: its record is dropped without affecting other snapshots' estimates or weights.
- **All snapshots have equal win rates**: PFSP reduces to ~uniform — no division-by-zero or degenerate weights.
- **Extreme win rates (p=0 or p=1)**: the curve stays finite — a never-beaten opponent is not weighted to infinity (monopolizing all envs), a never-lost-to opponent is not weighted to zero (starved).
- **Anchor + heuristic-floor envs**: excluded from PFSP weighting (fixed reserved weights). Anchor game outcomes MAY update the anchor's own win-rate record, but the anchor's sampling weight stays fixed at `anchor_weight` (not PFSP-driven).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The league MUST maintain, per snapshot (keyed by stable `snapshot_id`), a running estimate of the learner's win rate against that snapshot, updated from completed-game outcomes.
- **FR-002**: Each completed game MUST attribute its win/loss to the specific snapshot the env faced **when that game started** — correct under vec-env auto-reset (the just-ended game belongs to the opponent assigned at its start, not any newly-swapped one).
- **FR-003**: When PFSP is enabled, the per-rollout snapshot-pool opponent assignment MUST be drawn with each snapshot weighted by a configurable function of its win-rate estimate, including at minimum a "hard" curve (up-weights low win rate) and a "uniform" option.
- **FR-004**: A snapshot with fewer than a configurable minimum number of recorded games MUST receive a cold-start sampling weight rather than a weight derived from an unreliable estimate, and MUST NOT be starved.
- **FR-005**: PFSP MUST be off by default; with it off, opponent assignment MUST be byte-identical (per seed) to the current uniform-pool behaviour.
- **FR-006**: PFSP MUST govern only the snapshot-pool portion of the opponent mix; the frozen `anchor_weight` and `heuristic_weight` reserved fractions MUST be unchanged by PFSP.
- **FR-007**: The win-rate estimates MUST be saved with and restored from the run checkpoint (alongside the league), and PFSP sampling MUST be seeded so a resumed run reproduces the opponent-assignment sequence.
- **FR-008**: The win-rate weighting MUST remain finite and well-defined at the extremes (p=0, p=1) and when all snapshots share a win rate (no NaN/degenerate weights, no starvation, no monopolization).
- **FR-009**: PFSP MUST add no extra policy inference and only O(n_envs) bookkeeping per rollout (no meaningful training slowdown).

### Key Entities *(include if feature involves data)*

- **Opponent win-rate record**: per `snapshot_id` — accumulated wins + games (or an equivalent running statistic) sufficient to produce a win-rate estimate and a games-count for the cold-start gate. Lives with the league; evicted with its snapshot.
- **PFSP configuration**: enable flag; curve selection (hard / uniform); curve sharpness `k`; cold-start min-games threshold and cold-start weight.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Under the "hard" curve, an opponent at 30% agent-win-rate is sampled at least 2× as often as one at 70%, over a large draw.
- **SC-002**: A newly added snapshot receives a non-zero env share in the first rollout after it is added.
- **SC-003**: With PFSP disabled, opponent assignments are identical (bit-for-bit per seed) to the pre-PFSP behaviour.
- **SC-004**: Win-rate estimates round-trip through checkpoint save/load with zero difference, and a same-seed resumed run reproduces the opponent-assignment sequence exactly.
- **SC-005**: A PFSP-enabled self-play run's strength-vs-frozen-baseline does not exhibit the v2/v3 peak-then-collapse (no ≥0.20 drop from peak) — it holds within a tighter band or keeps climbing. *(Outcome metric, validated by a training run.)*
- **SC-006**: Per-update wall-clock with PFSP on is within ~2% of PFSP off.

## Assumptions

- Win-rate estimation uses a simple, resumable running statistic (e.g., Beta-Bernoulli wins/games with a small prior, or an EMA). The exact estimator is a design choice for `/speckit-plan`; cold-start is handled by the min-games gate + default weight.
- The "hard" curve default is proportional to `(1 − p_win)^k` with a default `k` (≈1–2), configurable; "uniform" is the fallback that reproduces today's behaviour.
- Cold-start min-games default is small (≈5–10 games); cold-start weight is the maximum pool weight so new snapshots are eagerly tried.
- Win rate is computed over rollout games as the env assigns seats (seat handling unchanged); evaluation seat-symmetry is handled separately by the eval harness, not PFSP.
- Anchor outcomes may update the anchor's own win-rate record, but the anchor's sampling weight stays fixed at `anchor_weight`.
- The existing snapshot-opponent driver, frozen anchor, heuristic floor, and league FIFO eviction are reused unchanged.

## Dependencies

- Builds on the shipped self-play league (snapshot pool, `OpponentAssignment`, `build_env_opponent_assignments`), the frozen-baseline anchor (`LeagueConfig.anchor_weight`, `League.set_anchor`), and the checkpoint manager's league capture/restore.
