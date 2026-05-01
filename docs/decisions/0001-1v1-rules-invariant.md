# ADR 0001: 1v1 Rules Are Invariant

**Status:** Accepted
**Date:** 2026-04-30

## Context

Catan has 4-player and 1v1 (Colonist.io) variants. The 1v1 ruleset differs from the standard rules in win VP (15 vs 10), discard threshold (9 vs 7), Friendly Robber, StackedDice, **disabled P2P trade**, and snake-draft setup specifics.

## Decision

This project implements **1v1 only**. The full rule table is in [`docs/1v1_rules.md`](../1v1_rules.md). The rules are invariants enforced by `src/catan_rl/eval/rules_invariants.py` (Phase 0) and validated by every eval-harness invocation.

## Consequences

- The action space has no propose/accept/counter trade actions (13-type discrete space, BankTrade-only).
- The observation models exactly one opponent (no list-of-opponents abstraction).
- Hand tracking is deterministic-perfect (relies on no P2P trade — see ADR 0002).
- Self-play machinery is 2-player symmetric zero-sum (PFSP, Nash pruning, exploitability).
- D6 board symmetry × Z_2 player swap = 24-fold data augmentation (Phase 1.5).

## Alternatives Considered

- **Generalize to N-player.** Requires belief-state hand tracking, multi-opponent obs encoder, propose/accept trade actions, n-player population game machinery. Out of scope.

## Related

- ADR 0002 (perfect hand tracking)
- `docs/1v1_rules.md`
- `docs/plans/superhuman_roadmap.md` §2 (cross-cutting 1v1 invariants)
