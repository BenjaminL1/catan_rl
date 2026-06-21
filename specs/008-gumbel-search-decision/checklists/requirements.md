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

- Grounded in this session's evidence (spec 007 US0/US0.5 value-ceiling probes; the search@50→@100 flat measurement; the documented `fpu_mode='zero'` visit-collapse) + a cited literature survey (Gumbel ICLR 2022, MiniZero) — code refs are context/assumptions; the WHAT (better decision rule on the frozen net, SPRT-confirmed at matched budget) is implementation-agnostic.
- **The load-bearing control (FR-005/SC-006): matched sim budget on every comparison** — the claim is "better allocation of the same sims," so any sim-count difference invalidates it.
- Gate-first chain (FR-009): US0 precursors + collapse-bound diagnostic → US1 Gumbel → all SPRT-confirmed. US0(a) LCB is the independently-shippable near-free win; US1 Gumbel is the main lever; US2 SPRT is the confirmation infra used by both.
- Inference-only (FR-006): no retrain, no checkpoint change — the cheapest, lowest-risk lever consistent with the value-ceiling finding.
- Deferred as future specs (non-goals): piKL, Stochastic-MuZero chance nodes (only if US0(b) = variance-bound), non-root Gumbel, expert iteration.
