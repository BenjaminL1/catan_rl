# Phase 1 Data Model: Finite Per-Resource Bank

## Entities

### ResourceBank

The finite public supply of resource cards, held as engine state.

| Attribute | Type | Notes |
|---|---|---|
| supply | 5 counts (one per resource) | Python: `dict[str, int]` keyed by engine `RESOURCES`. Rust: `[u8; N_RESOURCES]` indexed by `IDX_*`. |
| initial value | 19 per resource | `INITIAL_RESOURCE_BANK` in TS; literal `19` in both oracles. |
| visibility | public | Both players may observe it in principle; **not** added to the RL obs (D5). |
| location | on the cloned game state | Python: `catanBoard` (so `deepcopy` clones it). Rust: `GameState` (Clone derive clones it for free; used by determinized MCSearch). |

**Invariant (the central correctness gate)**: for every resource R, at all times,
`bank[R] + player0.hand[R] + player1.hand[R] == 19`, equivalently `Σ all hands + Σ bank == 95`.

**Bounds**: each `bank[R] ∈ [0, 19]`. Never negative (draws are gated/`checked_sub`); never exceeds 19 (conservation guarantees it given a legal start).

### ResourceDemand (transient, per production roll)

Computed during dice resolution; never persisted.

| Attribute | Type | Notes |
|---|---|---|
| per (player, resource) owed | int | settlement = 1, city = 2, summed over the player's buildings on hexes whose number token matches the roll and that are not robbed. |
| total per resource | int | `d0 + d1`; the input to the depletion rule. |

**Lifecycle**: built in pass 1 (tally) of `update_playerResources`/`distribute_resources`, consumed in pass 2 (apply depletion), then discarded.

### ConformanceFixture (unchanged schema)

A recorded reference game used for cross-engine parity. **Schema is NOT modified by this feature** (D4).

| Attribute | Type | Notes |
|---|---|---|
| schema_version | int | unchanged (no bump). |
| seed | int | RNG seed. |
| board | object | tiles, ports, robber. |
| steps[] | array | each step = `{action, outcome, state_after}`. |
| state_after | object (7 keys) | `robber_hex, settlements, cities, roads, players, longest_road_holder, largest_army_holder` — **no bank field**. |
| players[] in snapshot | object | `{vp, resources (Charlesworth), dev_hand, vp_cards, knights_played}` — hands here are how depletion is observed. |

**New instance**: one additional `game-seed-<N>.json` whose line of play forces a bank-short payout (auto-globbed by the TS suite).

## Resource-movement table (the behavioral contract)

| Action | Bank effect | Direction | Notes |
|---|---|---|---|
| Dice production | per-resource depletion rule | from-bank | bank decremented by what is actually granted (D3, D8). |
| Build paid road | WOOD+BRICK | to-bank | free Road-Builder road: **no** bank change. |
| Build settlement | WOOD+BRICK+WHEAT+SHEEP | to-bank | |
| Build city | 3 ORE + 2 WHEAT | to-bank | |
| Buy dev card | 1 ORE + 1 WHEAT + 1 SHEEP | to-bank | dev card itself drawn from the separate dev-card supply. |
| Bank/port trade | give × ratio in; receive × 1 out | to-bank **and** from-bank | ratio ∈ {2,3,4}; only path that both credits and debits. |
| Discard on a 7 | all discarded cards | to-bank | |
| Year of Plenty | 2 cards | from-bank | gated on availability; rejects if bank can't supply. |
| Setup 2nd settlement | 1 per adjacent non-desert hex | from-bank | **flat** decrement, NOT routed through the depletion rule. |
| Monopoly | none | n/a | player↔player only; bank untouched. |
| Robber / Knight steal | none | n/a | player↔player only; bank untouched. |
| First setup settlement | none | n/a | grants nothing. |

## Legal-move gating predicates

- **BankTrade(give, receive)** is offered only if `bank[receive] > 0` (and the player can afford `ratio` of `give`).
- **YearOfPlenty(first, second)** is offered only if the bank can supply it: `bank[first] >= (2 if first == second else 1)` and (if `first != second`) `bank[second] >= 1`.

## State transitions

The bank only changes via the movements above. There is no independent bank lifecycle; it is a derived ledger kept consistent with hands by the conservation invariant. Every action that changes any player's hand must apply the matching bank delta in the same atomic step (this is what the centralized helpers enforce — D1).
