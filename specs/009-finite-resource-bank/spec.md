# Feature Specification: Finite Per-Resource Bank (Oracle ↔ TS Parity)

**Feature Branch**: `feat/resource-bank-oracle` (intended canonical name `feat/finite-resource-bank`; temporarily renamed to avoid a live cross-session branch collision — see Assumptions)

**Created**: 2026-06-23

**Status**: Draft

**Input**: Task #57 / Torevan decision D12 — mirror the Torevan TypeScript engine's finite resource bank (`feat/finite-resource-bank` branch) in the `catan_rl_v2` reference oracle (Python `src/catan_rl/engine/` + Rust `crates/catan_engine/`) so the cross-engine Python↔TS conformance harness stays green.

## Why This Feature

Standard Catan / Colonist.io 1v1 uses a **finite supply of 19 cards per resource** with an **official depletion rule**. The `catan_rl_v2` reference engines currently model an **infinite** resource supply — which is itself a deviation from Colonist. The Torevan product engine has already shipped the finite bank (`GameState.resourceBank`, `INITIAL_RESOURCE_BANK` 19 each, `resolveBankProduction`, `bank.test.ts`); this oracle must mirror it so the conformance guard does not drift.

This is a **Colonist-faithfulness fix** (drift *toward* the reference ruleset, closing an existing TS↔Python drift), surfaced per `CLAUDE.md` rule #2 and constitution Principle II ("the game engine MUST match Colonist.io exactly"), and owner-approved via Torevan decision D11.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Python oracle matches the TS finite bank (Priority: P1)

The Python reference engine — which the conformance recorder and RL training both use — enforces the finite 19-per-resource bank, the official depletion rule, and full cost recirculation, so cross-engine conformance with the Torevan TS engine is restored and the documented "two engines never drift" guarantee becomes true.

**Why this priority**: This is the parity-critical path and the entire point of the task. The recorder drives the **Python** engine; without the Python mirror, any game that depletes diverges from the TS engine and the conformance guard is silently false. Delivered alone, this closes the drift and is a viable, demonstrable MVP.

**Independent Test**: Make the Python engine bank-aware; run `pytest` (new bank coverage) green; re-record seeds 7/8/15 and confirm byte-identical fixtures; record a new depletion fixture and confirm the Torevan conformance suite passes including it.

**Acceptance Scenarios**:

1. **Given** a fresh game, **When** any sequence of legal actions is applied, **Then** for every resource R the invariant `bank[R] + player0.hand[R] + player1.hand[R] == 19` holds after every action (equivalently Σ all hands + Σ bank == 95).
2. **Given** a production roll whose combined demand for a resource exceeds the remaining bank **and** both players are owed that resource, **When** the roll is resolved, **Then** neither player receives any of that resource and the bank for it is unchanged.
3. **Given** a production roll whose combined demand for a resource exceeds the remaining bank **and** exactly one player is owed it, **When** the roll is resolved, **Then** that sole player receives the entire remaining bank for that resource and the bank for it becomes 0.
4. **Given** seeds 7, 8, 15 (which never deplete), **When** they are re-recorded with the bank-aware oracle, **Then** the produced fixtures are byte-identical to the committed ones (`git diff` empty).
5. **Given** a build / dev-card purchase / bank trade / discard, **When** it is applied, **Then** the spent cards are returned to the bank such that conservation holds (a free Road-Builder road recirculates nothing).

---

### User Story 2 - Rust engine mirrors the same finite bank (Priority: P2)

The Rust engine (`crates/catan_engine/`) enforces the identical finite bank, depletion rule, and recirculation, so the two reference engines remain byte-for-byte consistent with each other and with the TS source of truth.

**Why this priority**: The Rust engine is not yet wired into training and is not on the conformance-recording path, so it is not parity-blocking today — but leaving it on the infinite-supply model would re-open the very drift this feature closes. It follows immediately after the Python mirror.

**Independent Test**: Implement the bank in `crates/catan_engine/src/state.rs`; run `cargo test` green, including a conservation-invariant assertion across full self-play games and the three depletion-branch unit tests; verify the `distribute_resources` restructure does not break event-stream consumers (`hand_tracker.rs` / `obs.rs` / `events.rs`).

**Acceptance Scenarios**:

1. **Given** a full self-play game driven to terminal, **When** the conservation invariant is checked after every applied action, **Then** Σ all hands + Σ bank == 95 holds at every step.
2. **Given** the three depletion branches (production short with both owed, production short with one owed, draw-from-empty for Year of Plenty / bank-trade-receive), **When** each is exercised, **Then** the outcome matches the TS `resolveBankProduction` / gating semantics exactly.
3. **Given** the restructured `distribute_resources`, **When** existing event-stream consumers parse its output, **Then** they continue to function (the per-hex → per-resource grant-event shape change is absorbed without breakage).

---

### User Story 3 - Documentation and lineage are truthful (Priority: P3)

Every existing artifact that describes the resource supply reflects the finite bank, and no stale "never drift" / "infinite supply" claim remains in either repo.

**Why this priority**: Required for honesty and to satisfy the documentation-sync rule, but it does not change engine behavior. It lands with the implementation.

**Independent Test**: Grep both repos for the old supply description and confirm each reference is updated; confirm the Torevan `state.ts` "mirrored exactly … never drift" comment is now true (or softened until the Python mirror lands); confirm the D11 note is flipped and a D12 row appended.

**Acceptance Scenarios**:

1. **Given** the `catan_rl_v2` `CLAUDE.md` ruleset table, **When** read after the change, **Then** the resource row states the finite 19-bank + depletion rule (not just "standard counts").
2. **Given** the Torevan `docs/plans/mvp-build-plan.md`, **When** read after the change, **Then** the D11 note is reconciled and a D12 (#57) row records the oracle mirror.
3. **Given** the lineage notes for banked WR/Elo results (v8 +121 Elo, the search ladder), **When** read after the change, **Then** they record that pre-bank numbers were measured on the infinite-bank engine.

### Edge Cases

- **Both players owed, bank short**: neither player receives that resource; bank unchanged (official "not enough to go around").
- **One player owed, bank short**: sole claimant takes the entire remaining bank; bank → 0 (partial fulfillment).
- **Nobody owed (total demand 0)**: no-op; bank unchanged.
- **Year of Plenty for a doubled pick** (`first == second`) when the bank holds fewer than 2 of that type: the pick is not offered (legal-move gating); apply-time rejects it if forced.
- **Bank trade requesting a resource the bank has 0 of**: not offered; apply-time rejects.
- **Free Road-Builder road**: costs nothing, recirculates nothing (distinct from a paid road).
- **First setup settlement**: grants nothing, does not touch the bank; only the second settlement grants (and decrements the bank by a flat 1 per adjacent non-desert hex, *not* routed through the depletion rule).
- **Monopoly / robber-or-Knight steal**: player↔player transfer only; bank untouched.
- **Discard on a 7**: discarded cards return to the bank.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The engine MUST maintain a finite per-resource bank for all five resources, initialized to 19 each, public to both players.
- **FR-002 (Conservation)**: After every action, `bank[R] + Σ players' hands[R] == 19` MUST hold for each resource R (equivalently Σ all hands + Σ bank == 95), for the entire duration of any game.
- **FR-003 (Production depletion)**: Dice production MUST be resolved per resource by the official rule: let `total = demand0 + demand1`, `avail = bank[R]`. If `total ≤ avail`, both players are paid in full and `bank[R] -= total`. Else if both players are owed (`demand0 > 0 and demand1 > 0`), neither is paid and `bank[R]` is unchanged. Else if only player 0 is owed, player 0 receives `avail` and `bank[R] → 0`. Else player 1 receives `avail` and `bank[R] → 0`. (Semantics MUST match the TS `resolveBankProduction` verbatim, including the `total == 0` no-op and the implicit final branch; no redundant re-test, seat-0-before-seat-1 ordering preserved.)
- **FR-004 (Recirculation to bank)**: Spent resources MUST return to the bank: paid road → WOOD+BRICK; settlement → WOOD+BRICK+WHEAT+SHEEP; city → 3 ORE+2 WHEAT; buy dev card → 1 ORE+1 WHEAT+1 SHEEP; bank/port trade → give×ratio; discard on a 7 → all discarded cards. A **free** Road-Builder road MUST recirculate nothing.
- **FR-005 (Draws from bank)**: Resources MUST be drawn from (decrement) the bank for: dice production (per FR-003), the setup second-settlement grant, Year of Plenty (2 cards), and the bank/port-trade receive side (1 card).
- **FR-006 (Setup grant)**: The setup second-settlement starting grant MUST decrement the bank by a flat 1 per adjacent non-desert hex and MUST NOT be routed through the production-depletion rule. The first setup settlement grants nothing.
- **FR-007 (Bank-neutral actions)**: Monopoly and robber/Knight steal MUST be player↔player transfers only and MUST NOT touch the bank.
- **FR-008 (Legal-move gating)**: The legal-action generator MUST only offer a bank trade whose received resource has `bank > 0`, and MUST only offer a Year-of-Plenty pick the bank can supply (≥2 of one type if both picks are the same resource, else ≥1 of each).
- **FR-009 (Cross-engine parity)**: The Python and Rust engines MUST produce identical bank and hand outcomes for the same line of play, and both MUST match the Torevan TS engine's semantics exactly. Neither engine may clamp, floor, or reorder anything the TS engine does not.
- **FR-010 (Observation unchanged)**: The resource bank MUST NOT be added to the observation schema. The observation vector shape MUST be byte-identical to before this change so that all existing v2 policy checkpoints remain loadable with no state-dict migration.
- **FR-011 (Always on)**: The finite bank MUST be active in every engine path (training, evaluation, conformance recording). No feature flag or toggle is introduced.
- **FR-012 (Conformance proof)**: Re-recording the existing conformance fixtures (seeds 7/8/15) MUST yield byte-identical files with no snapshot-schema change. A NEW conformance fixture that forces a bank-short payout MUST be added and the Torevan conformance suite MUST pass including it. (Depletion is proven via the resulting player hands; the bank value itself is not serialized.)
- **FR-013 (RNG keystream stability)**: The bank refactor MUST NOT alter the dice / steal / dev-card-draw random keystream on non-depleting games, so that the seed-7/8/15 re-records stay byte-identical.
- **FR-014 (Documentation truthfulness)**: Every existing artifact describing the resource supply MUST be updated: the `catan_rl_v2` `CLAUDE.md` ruleset table + engine-implications, the Torevan `docs/plans/mvp-build-plan.md` (flip D11, append D12), and the Torevan `state.ts` "mirrored exactly … never drift" comment.
- **FR-015 (Change governance)**: The commit MUST flag the game-rule change per `CLAUDE.md` rule #2 / constitution Principle II and MUST NOT include any AI co-author / "Generated with" trailers. The new depletion fixture is written into the Torevan working tree but NOT committed or pushed there by this work (the Torevan session commits it on `feat/finite-resource-bank`).

### Key Entities

- **Resource Bank**: a finite per-resource supply (5 resources, 19 each at game start), public to both players, conserved against player hands.
- **Resource Demand**: the per-roll, per-player amount owed for a resource (settlement = 1, city = 2, summed over the player's buildings on hexes matching the roll), the input to the production-depletion rule.
- **Conformance Fixture**: a recorded reference game (seed → ordered steps → per-step canonical snapshot of player hands and board state) used to assert byte-for-byte cross-engine agreement; auto-discovered by the TS suite via a `game-seed-*.json` glob.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The conservation invariant (Σ all hands + Σ bank == 95) holds after 100% of applied actions across a suite of full self-play games, in both the Python and Rust engines.
- **SC-002**: Re-recording seeds 7, 8, 15 produces byte-identical fixtures (`git diff` reports zero changes).
- **SC-003**: A new depletion fixture exists that exercises every reachable depletion branch, and the Torevan conformance suite (`npm run test -w @torevan/engine`) passes with it included.
- **SC-004**: 100% of existing v2 checkpoints (e.g. `bootstrap_v1`, `v6_u1399`, the v8 champion) load with no migration — the observation vector shape is unchanged.
- **SC-005**: The Python and Rust engines produce identical bank and hand values at every step of the new depletion fixture's line of play.
- **SC-006**: `pytest` and `cargo test` are green, and CI (ruff + mypy strict + pytest) is green.

## Assumptions

- **Parity source of truth**: the Torevan `feat/finite-resource-bank` branch (`state.ts`, `actions.ts`, `setup.ts`, `legal-moves.ts`, `bank.test.ts`). Where this spec's prose and the TS code disagree, the TS code wins.
- **Depletion is rare in 1v1**: with only two hands drawing from 19, the bank is approached only in unusual late-game lines, so the resulting MDP change is negligible. Banked pre-bank WR/Elo (v8 +121 Elo, the search ladder) were measured on the infinite-bank engine; they remain valid as historical measurements and are annotated as such.
- **Recorder engine**: the conformance recorder drives the Python `catanGame` engine; making that engine bank-aware is what restores parity.
- **Resource ordering**: the canonical cross-engine order is Charlesworth (WOOD, BRICK, WHEAT, ORE, SHEEP); each engine may key its bank in its own native internal order and re-key only at the conformance boundary.
- **Branch name**: this work is on `feat/resource-bank-oracle` in an isolated git worktree because a concurrent session held the intended `feat/finite-resource-bank` name in the shared tree; the name can be reconciled before merge.

## Out of Scope

- Adding the resource bank to the observation schema (would resize the obs vector → policy state-dict migration + retrain; a deliberate, separately-gated follow-up).
- Serializing/asserting the bank *value* in the conformance snapshot (a `schema_version` bump regenerating all fixtures + extending the TS normaliser).
- Any feature flag / toggle to disable the finite bank.
- Changes to the dev-card bank (a separate finite supply that already exists) beyond the resource cost that recirculates on purchase.

## Dependencies

- The Torevan `feat/finite-resource-bank` branch must remain the authoritative spec.
- The cross-engine conformance harness (`src/catan_rl/conformance/` recorder + the Torevan `conformance.test.ts` suite) must be present and runnable.
