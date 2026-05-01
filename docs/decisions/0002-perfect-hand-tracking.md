# ADR 0002: Perfect Opponent Hand Tracking via GameBroadcast

**Status:** Accepted
**Date:** 2026-04-30

## Context

In the standard 4-player ruleset with player-to-player trading, the opponent's resource hand is partially observable: trades obscure resource flow. Most Catan agents therefore maintain a belief state over opponent hands.

In **1v1 with no P2P trading**, every resource change is observable: dice rolls grant resources visibly to known settlements; bank trades, builds, robber steals, discards, and Year-of-Plenty all emit observable state changes. There is no obscurement channel.

## Decision

Use a `GameBroadcast` event bus (`src/catan_rl/engine/broadcast.py`) and a `BroadcastHandTracker` (`src/catan_rl/env/hand_tracker.py`) to maintain a **deterministic-perfect** record of every player's resource hand at every step. The tracker subscribes to `RESOURCE_CHANGE` events and updates its internal state.

The agent's observation uses the tracked hand for the opponent's resources rather than reading `opponent.resources` directly (so the architecture is conceptually correct: the agent only sees what's broadcast-observable).

## Consequences

- No belief state needed for opponent resources — the tracker is deterministic.
- Every action that mutates resources **must** emit a `RESOURCE_CHANGE` event. Failure to emit is a correctness bug, not a missing optimization.
- Phase 2 of the roadmap can add a belief head for opponent **dev-card types** specifically — the only remaining hidden state in 1v1.
- Phase 4 ISMCTS can sample dev-card determinizations from the belief head; the resource side is exact.
- This is **1v1-only**. Generalizing to 4-player or P2P trade re-introduces hidden flows and breaks tracker correctness.

## Alternatives Considered

- **Belief-state network for resources.** Costlier and unnecessary in 1v1 — and would mask correctness bugs in broadcast emissions.
- **Bypass the broadcast and read `opponent.resources` directly.** Tempting but incorrect: the agent's obs would contain information not available through the formal game-event channel, defeating the design.

## Related

- ADR 0001 (1v1 invariant)
- `docs/plans/archive/BROADCAST_HAND_TRACKING_PLAN.md` (original plan, now implemented)
- `src/catan_rl/env/hand_tracker.py`, `src/catan_rl/engine/broadcast.py`
