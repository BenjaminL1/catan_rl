# Feature Specification: Expert Iteration (search-as-teacher distillation)

**Feature Branch**: `004-expert-iteration`

**Created**: 2026-06-15

**Status**: Draft

**Input**: User description: "Expert iteration (ExIt / search-as-teacher) — convert the offline inference-time determinized-MCTS search lever (003) into a permanent, compounding improvement of the 1v1 Catan policy by distilling search's targets back into the network, gated by a cheap pilot before scaling the flywheel."

## User Scenarios & Testing *(mandatory)*

The agent (v6) has plateaued. Inference-time search (003) plays measurably stronger
(+~55 Elo at 50 sims/move) but only *offline* — it never improves the base policy and
costs ~32 s/move. Expert iteration uses search as a *teacher*: search is a stronger
operator than the raw network, so the action it most-visits and the value it backs up are
*improved targets* the policy can be trained toward. Distilling them yields a **faster,
stronger search-free policy**; repeating on the stronger base compounds. The risk is that
the teacher is only modestly stronger and that, on one machine, generating search-labeled
data is slow — so the work is **gated by a cheap pilot** before the full loop is built.

### User Story 1 - Pilot gate: does distilling search beat the raw policy? (Priority: P1)

A practitioner generates a **small** search-labeled dataset from the current best policy
(v6), runs a **single** distillation fine-tune toward those targets, and evaluates the
resulting policy **head-to-head against raw v6** as a fast, search-free agent. This one
decision — *does search-distillation produce a stronger base policy at all?* — gates the
entire multi-round effort.

**Why this priority**: It is the cheapest decisive test of the whole premise. If a single
distillation round can't beat the policy it came from, the flywheel won't either, and the
build should stop or pivot before investing in throughput engineering and multi-round
orchestration. It mirrors the 003 bake-off's go/no-go discipline.

**Independent Test**: Build only the labeler + a single-round distillation + the existing
eval harness; generate a few-thousand-position dataset, fine-tune v6, and confirm the
distilled policy beats raw v6 with a Wilson lower bound > 0.50 over ≥200 then ≥500
seat-symmetrized games (or a documented FAIL + pivot).

**Acceptance Scenarios**:

1. **Given** the v6 policy and the 003 search, **When** a few-thousand-position
   search-labeled dataset is generated and v6 is fine-tuned toward it, **Then** the
   distilled policy (search-free, single forward pass) wins against raw v6 with a Wilson
   lower bound > 0.50 at n≥200, re-confirmed at n≥500.
2. **Given** the gate evaluation completes, **When** the win-rate is ≈0.50 or its lower
   bound ≤ 0.50, **Then** the build STOPS and records the failure mode + recommended pivot
   (stronger/more search per label, more positions, target weighting, value-only vs
   policy-only distillation) — and does NOT build the flywheel.
3. **Given** any distilled checkpoint, **When** it is loaded for evaluation, **Then** it
   loads through the existing checkpoint manager unchanged (v2-lineage, no shape change)
   and produces zero 1v1-ruleset violations across the evaluated games.

---

### User Story 2 - The flywheel: compounding improvement across rounds (Priority: P2)  *(only if US1 passes)*

The practitioner runs a small number of expert-iteration **rounds** — each round
generates fresh search-labeled data using the *previous round's* distilled policy as the
search base, distills, and evaluates — and observes the agent's strength over raw v6
grow (or hold) round over round.

**Why this priority**: This is where ExIt's value compounds (the AlphaZero flywheel), but
it is only worth the orchestration + throughput cost if the pilot (US1) proves a single
round works.

**Independent Test**: Run ≥2 rounds and confirm the distilled policy's Elo over raw v6 is
monotone non-decreasing round-over-round (each round at least as strong as the last,
within CI), with zero ruleset violations.

**Acceptance Scenarios**:

1. **Given** US1 passed, **When** ≥2 ExIt rounds run (each seeded from the prior round's
   distilled policy), **Then** the per-round Elo-over-raw-v6 is monotone non-decreasing
   within confidence intervals.
2. **Given** a round's search-labeled generation, **When** the same round is re-run with
   the same seed, **Then** it produces identical labeled data and an identically-evaluated
   distilled checkpoint (reproducible).

---

### User Story 3 - A banked, faster, stronger agent (Priority: P3)  *(only if US1 passes)*

The best distilled checkpoint is placed on the existing strength (Elo) ladder and
established as a new v2-lineage best — a **search-free** agent (single forward pass, no
~32 s/move search at play time) that is measurably stronger than raw v6.

**Why this priority**: This is the end-value — the offline search gain *banked* into the
cheap policy that can actually be deployed.

**Independent Test**: Evaluate the best distilled checkpoint on the Elo ladder; confirm a
positive Elo delta over raw v6 with a non-overlapping confidence interval, achieved
without any inference-time search.

**Acceptance Scenarios**:

1. **Given** the best distilled checkpoint, **When** it is rated on the ladder as a
   search-free agent, **Then** its Elo over raw v6 is positive with a CI that excludes
   zero.
2. **Given** the distilled agent at play time, **When** it selects a move, **Then** it
   uses a single network forward pass (no MCTS), at raw-policy speed.

### Edge Cases

- **Teacher barely stronger / distillation captures little**: the gate (US1) catches this —
  WR ≈ 0.50 → documented FAIL + pivot, no flywheel.
- **Catastrophic forgetting**: a distillation fine-tune that over-fits search targets could
  regress vs the heuristic / earlier opponents even while beating raw v6 — evaluation must
  include the heuristic + a prior-lineage rung, not only raw v6.
- **Throughput**: at ~32 s/move, a full flywheel round's data generation may be infeasible
  on one machine — the pilot is sized to current throughput; scaling is explicitly gated on
  a batched-leaf-evaluation throughput multiplier.
- **Degenerate targets**: positions where search short-circuits (a single legal action) carry
  no learning signal — the labeler must skip or down-weight them so the dataset isn't
  dominated by forced moves.
- **Round divergence**: a round that produces a *weaker* policy must not silently seed the
  next round — the flywheel keeps the best-so-far as the next base.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST generate, per sampled game position, a POLICY TARGET derived
  from the 003 search at that position (the search root's normalized visit distribution over
  the legal action set) and a VALUE TARGET (the search-backed root value and/or the realized
  game outcome).
- **FR-002**: The system MUST sample positions from games played by the current best policy
  (search-driven self-play and/or search-vs-heuristic), and MUST skip or down-weight forced
  (single-legal-action) positions.
- **FR-003**: The system MUST produce a distilled policy by fine-tuning the current best
  policy toward the search-labeled targets, reusing the existing behavior-cloning training
  path (teacher = search instead of replays/heuristic), with the network architecture,
  observation space, action space, and head set unchanged.
- **FR-004**: Every distilled checkpoint MUST remain v2-lineage and load through the existing
  checkpoint manager with no state-dict shape change (no migration unless explicitly
  documented).
- **FR-005**: The system MUST evaluate any distilled policy head-to-head against raw v6 as a
  SEARCH-FREE agent (single forward pass) via the existing eval harness, returning a
  seat-symmetrized win-rate with a Wilson confidence interval.
- **FR-006**: The pilot gate MUST declare PASS iff the distilled policy beats raw v6 with a
  Wilson lower bound > 0.50 at n≥200, re-confirmed on a disjoint sample at n≥500; otherwise
  FAIL with a recorded failure mode + recommended pivot, and the flywheel MUST NOT be built.
- **FR-007**: *(only if the gate passes)* The system MUST run ≥2 expert-iteration rounds,
  each seeding its search teacher from the previous round's distilled policy and keeping the
  best-so-far as the base if a round regresses.
- **FR-008**: The system MUST place the best distilled checkpoint on the existing Elo ladder
  (search-free rung) and report its Elo delta over raw v6 with a confidence interval.
- **FR-009**: Data generation and distillation MUST be deterministic + reproducible at a
  fixed seed.
- **FR-010**: Distilled policies MUST NOT violate the 1v1 Colonist ruleset (zero violations
  reported by the rules-invariant audit across evaluated games); the engine, ruleset, and
  003 search behavior MUST be unchanged (the labeler consumes the search as-is).
- **FR-011**: The feature MUST be additive: a new module hosts the labeling + round
  orchestration; existing training, search, and evaluation behavior is byte-identical when
  the feature is unused; any new diagnostic/TensorBoard scalars use new names.
- **FR-012**: *(scaling, only after the gate passes)* The system SHOULD provide a batched
  leaf-evaluation path for the search so the flywheel's data generation is tractable on a
  single machine; this is the throughput lever, not a pilot prerequisite.

### Key Entities

- **Search-labeled position**: a game state + its search-derived policy target (visit
  distribution over legal actions) + value target (search/outcome) + a weight (down-weighting
  forced/low-information positions).
- **Search-labeled dataset**: a collection of search-labeled positions for one round, sized to
  the available compute (pilot: a few thousand positions).
- **Distilled checkpoint**: a v2-lineage policy checkpoint produced by fine-tuning the base
  policy toward a search-labeled dataset; evaluated search-free.
- **Expert-iteration round**: one {generate → distill → evaluate} cycle; the best distilled
  checkpoint becomes the next round's base.
- **Gate result**: the PASS/FAIL verdict + win-rate + CI + (on FAIL) the failure mode and
  recommended pivot.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (pilot gate / go-no-go)**: A single distillation fine-tune of the current best
  policy, trained only on search-derived targets, beats raw v6 as a search-free agent with a
  **Wilson lower bound > 0.50** at n≥200, re-confirmed at n≥500.
- **SC-002 (meaningful uplift target)**: The distilled policy's win-rate over raw v6 reaches
  **≥ 0.55** (≈ +35 Elo) — evidence that distillation captured a meaningful fraction of the
  search teacher's strength (the teacher is ~0.578 / +55 Elo). (Aspiration above the SC-001
  significance bar; informs whether to keep iterating.)
- **SC-003 (compounding)**: Across ≥2 expert-iteration rounds, the distilled policy's Elo over
  raw v6 is monotone non-decreasing within confidence intervals.
- **SC-004 (banked, fast, deployable)**: The best distilled checkpoint is a single-forward-pass
  agent (no inference-time search) with a positive Elo over raw v6 and a CI excluding zero, at
  raw-policy move speed.
- **SC-005 (safety + compatibility)**: Zero 1v1-ruleset violations across all evaluated games;
  every distilled checkpoint loads via the existing checkpoint manager unchanged.
- **SC-006 (reproducibility)**: The same seed reproduces the same labeled dataset and the same
  evaluated distilled checkpoint.

## Assumptions

- **Gate calibration**: SC-001 uses Wilson **LB > 0.50** (statistically-significant improvement,
  mirroring the 003 bake-off rigor) as the go/no-go; the user's "~+35 Elo / WR 0.55" target is
  captured as SC-002 (an aspiration, not the gate) because a student distilled from a +55-Elo
  teacher typically *approaches* rather than exceeds it, so a strict LB > 0.55 gate could reject
  a genuinely-working first round.
- **Reuse, not rebuild**: the 003 search provides the teacher (visit counts + value already
  available from its diagnostics); the existing behavior-cloning trainer provides distillation
  (soft targets instead of hard replay actions); the eval harness + Elo ladder provide
  measurement. The new module is orchestration + labeling only.
- **Offline ExIt**: this is generate-then-distill (search OUTSIDE the training loop); search
  inside the PPO rollout, piKL, the Rust engine, architecture/obs/action changes, and
  search-at-deployment are explicitly out of scope.
- **Compute is the binding constraint**: a single Apple M1 Pro (CPU search ~120 sims/sec; MPS
  for distillation SGD). The pilot is sized to current throughput; the full flywheel's
  feasibility depends on the batched-leaf-evaluation throughput multiplier (FR-012), in scope
  only after the gate passes.
- **Base + opponents**: the base policy is v6 `ckpt_000001499`; evaluation includes raw v6, the
  heuristic, and at least one prior-lineage rung to catch catastrophic forgetting (not only raw
  v6).
- **Device policy**: MPS for distillation training, CPU-pinned for search + evaluation; no GUI
  import on any training/search/eval path.
