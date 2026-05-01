# ADR 0004: 6-Head Autoregressive Action Space

**Status:** Accepted
**Date:** 2026-04-30

## Context

Catan actions are **composite**: "Build Settlement at vertex 12" requires both a type choice (BuildSettlement) and a target (vertex 12). A flat `Discrete(N)` action space would have on the order of `13 * 54 * 72 * 19 * 5 * 5 ≈ 24M` possible actions — most invalid in any given state.

## Decision

Use a 6-head autoregressive action space `MultiDiscrete([13, 54, 72, 19, 5, 5])` corresponding to `[type, corner, edge, tile, resource1, resource2]`. Per the chosen action type, only relevant heads contribute to the joint log-probability and entropy via registered relevance buffer masks.

## Consequences

- Action masking is per-head (9 mask keys); see `docs/action_schema.md`.
- All 6 heads always forward-pass; relevance is applied at log-prob/entropy aggregation. Phase 2.4 of the roadmap will tighten this with AdaLN/FiLM head conditioning so each head is more strongly conditioned on action type.
- Joint entropy can mask silent collapse on individual heads. Phase 0 adds **per-head** entropy logging to detect this.
- The resource heads (4, 5) are reused across YoP, Monopoly, BankTrade, Discard via context one-hot. This is space-efficient but couples four different operational meanings into one head's gradient signal.

## Alternatives Considered

- **Flat Discrete(24M)** with masking. Rejected: most actions are always invalid, gradient signal is sparse, and the type-vs-target structure is real and worth modeling.
- **Per-action-type heads.** Would have ~13 separate sub-policies. More expressive but more parameters and harder to share representations across related actions.

## Related

- `src/catan_rl/models/action_heads_module.py`
- `src/catan_rl/models/distributions.py` (MaskedCategorical)
- `docs/action_schema.md`
- `docs/plans/superhuman_roadmap.md` §5.1.4 (AdaLN head conditioning)
