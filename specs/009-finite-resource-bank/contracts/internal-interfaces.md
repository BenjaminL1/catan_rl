# Internal Interface Contract: Finite Resource Bank

This is the parity contract both engines (Python `catan_rl.engine`, Rust `crates/catan_engine`) MUST honor, and which the Torevan TS engine already implements. It is an *internal* engine contract (no public API change). Where this and the TS code disagree, **the TS code wins**.

## C1 — Bank state

- A finite per-resource bank, 5 resources, initialized to **19 each**, held on the cloned game state.
- Conservation invariant (MUST hold after every applied action): `∀R: bank[R] + Σ_players hand[R] == 19`  ⇔  `Σ hands + Σ bank == 95`.

## C2 — Production depletion resolver

Signature (conceptual): `resolve(avail, d0, d1) -> (g0, g1, remaining)` applied per resource.

```
total = d0 + d1
if total == 0:           return (0, 0, avail)          # no-op
if total <= avail:       return (d0, d1, avail - total) # both paid in full
if d0 > 0 and d1 > 0:    return (0, 0, avail)           # both owed, short -> NEITHER
if d0 > 0:               return (avail, 0, 0)           # sole claimant seat 0 takes remainder
                         return (0, avail, 0)           # implicit else: sole claimant seat 1
```

Rules for the mirror:
- Preserve the `total == 0` short-circuit and the **implicit final else** (do NOT add a redundant `d1 > 0` test).
- Preserve seat-0-before-seat-1 ordering.
- `g0`/`g1` are added to the respective hands; `remaining` is written back to `bank[R]` (decrement equals what was granted).
- The driver builds `d0`/`d1` per resource from a **two-pass** tally (settlement = 1, city = 2, summed over matching unrobbed hexes), then calls `resolve` once per resource. No incremental grant inside the hex loop.

## C3 — Recirculate-to-bank (cost return)

A helper that adds spent costs back to the bank, used by:
- paid road `{WOOD:1, BRICK:1}` (NOT free Road-Builder roads),
- settlement `{WOOD:1, BRICK:1, WHEAT:1, SHEEP:1}`,
- city `{ORE:3, WHEAT:2}`,
- buy dev card `{ORE:1, WHEAT:1, SHEEP:1}`,
- bank-trade give side `{give: ratio}`,
- discard `{each discarded resource: amount}`.

## C4 — Draw-from-bank (supply debit)

A helper that subtracts from the bank, used by:
- dice production (via C2),
- setup 2nd-settlement grant — **flat** `-1` per adjacent non-desert hex, NOT via C2,
- Year of Plenty — `-1` per chosen card (2 total),
- bank-trade receive side — `-1`.

Underflow handling: on the gated action paths (YOP, bank-trade), the draw MUST be guarded (Python: pre-check availability; Rust: `checked_sub` → reject the action), never silently saturated. Production (C2) is infallible and clamps by construction via the tally.

## C5 — Bank-neutral actions

Monopoly and robber/Knight steal MUST NOT read or write the bank.

## C6 — Legal-move gating

The legal-action generator MUST consult the bank:
- offer `BankTrade(give, receive)` only if `bank[receive] > 0`,
- offer `YearOfPlenty(first, second)` only if `bank[first] >= (2 if first==second else 1)` and (`first==second` or `bank[second] >= 1`).

## C7 — Single-path requirement (anti-drift)

All three Python resource-mutation paths — engine (`player.py`/`game.py`), env handlers (`catan_env.py`), and the conformance recorder (`recorder.py`) — MUST go through the C2–C4 helpers. No path may re-implement bank math. (A test MUST drive each path through a depleting line and assert identical bank + hands.)

## C8 — Boundary ordering

Internally each engine uses its native resource order; the conformance snapshot emits hands in **Charlesworth** order (unchanged from today). The bank is not serialized into the snapshot (C1 is verified by tests, not by the fixture).
