# Specification Quality Checklist: Gumbel Search-Decision Upgrade

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-06-20
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- **Revised after senior-RL review (verdict was NOT-READY; all 4 blockers + should-fix resolved):**
  - BLOCKER 1 (central bet unvalidated; US0b can't discriminate collapse-bound from near-optimal-root) → US0(c)/FR-004 **oracle root-headroom pre-check + pre-registered GO rule** (headroom >+15 Elo AND high depth-0 collapse AND root-child Spearman ≥0.60) now GATES the Gumbel build; US0(b) reframed as a diagnostic readout.
  - BLOCKER 2 (completed-Q guarantee weak on a chance-folded tree at Spearman ~0.68) → formula written (FR-005), ~30% regret inflation noted, **+7–15 Elo band pre-registered**, **Gumbel-max-Q fallback if Spearman<0.60**.
  - BLOCKER 3 (SPRT unbuilt, elo1 undefined) → FR-001 SPRT built FIRST, elo1=+10 Elo default, vetted pentanomial formula, tests.
  - BLOCKER 4 (fixed-budget split absent — K=4 silently runs 4× sims) → FR-003 budget-split (sims//K) + parity assertion.
  - should-fix resolved: per-depth collapse metric → root-only-vs-extend decision; LCB fully specified (z=1.96, in-memory sum_Q2, eps floor, ties, no state-dict change); Gumbel params (m/temp/rounds/degeneration/forced-no-op); matched-budget operationalized (assert+raise+equal n_det); all-flags-off byte-identity regression test; STAGE-A/STAGE-B split so the kill-gate precedes the build; config enums; verdict JSON schema; fpu_parent framing corrected.
- The kill-gate (STAGE-A, ~<1 day) decides GO/NO-GO cheaply before the multi-day Gumbel build is funded — the highest-value structural change the review produced.
- Grounded in this session's evidence (spec 007 US0/US0.5 value-ceiling probes; the search@50→@100 flat measurement; the documented `fpu_mode='zero'` visit-collapse) + a cited literature survey (Gumbel ICLR 2022, MiniZero) — code refs are context/assumptions; the WHAT (better decision rule on the frozen net, SPRT-confirmed at matched budget) is implementation-agnostic.
- **The load-bearing control (FR-005/SC-006): matched sim budget on every comparison** — the claim is "better allocation of the same sims," so any sim-count difference invalidates it.
- Gate-first chain (FR-009): US0 precursors + collapse-bound diagnostic → US1 Gumbel → all SPRT-confirmed. US0(a) LCB is the independently-shippable near-free win; US1 Gumbel is the main lever; US2 SPRT is the confirmation infra used by both.
- Inference-only (FR-006): no retrain, no checkpoint change — the cheapest, lowest-risk lever consistent with the value-ceiling finding.
- Deferred as future specs (non-goals): piKL, Stochastic-MuZero chance nodes (only if US0(b) = variance-bound), non-root Gumbel, expert iteration.
