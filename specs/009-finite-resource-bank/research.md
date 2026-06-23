# Phase 0 Research: Finite Per-Resource Bank

All "unknowns" were resolved by a 5-way parallel recon across `catan_rl_v2` and `../Torevan` earlier this session (TS source-of-truth extraction, conformance-schema analysis, Python + Rust change-site mapping, constitution/obs impact). This file records the decisions; there are **no open `NEEDS CLARIFICATION` items**.

## D1 — Centralize the bank on the engine (vs. patch each call-site)

- **Decision**: Hold the bank as engine state (`catanBoard.resourceBank` in Python, `GameState.bank` in Rust) and expose a small set of engine methods (a production/depletion resolver, a recirculate-to-bank helper, a draw-from-bank helper). Every resource mutation routes through them.
- **Rationale**: Python mutates resources across **three bypassing paths** — `player.py`/`game.py`, the env's own YOP/Discard/BankTrade handlers in `catan_env.py`, and the recorder's inline arithmetic in `recorder.py`. If each re-implements bank math, they drift silently and the conformance guarantee becomes false without any test failing on 7/8/15. One code path is the only way to guarantee they agree.
- **Alternatives considered**: Patch each of the three paths independently (rejected — guaranteed drift, triples the bug surface, violates the spirit of the conformance guard). A free function taking the bank by reference (viable, but a method on the board/state is more idiomatic and clones for free).

## D2 — Strict constant-sum conservation (Σ hands + Σ bank == 95)

- **Decision**: Recirculate **every** spent resource back to the bank (build, dev-card purchase, bank-trade give side, discard), matching the TS engine; the invariant is the constant sum 5 × 19 = 95.
- **Rationale**: The TS engine recirculates all costs; to match it byte-for-byte the oracle must too. A constant-sum invariant is the strongest, cheapest single correctness gate — assert it after every action in a full-game test.
- **Alternatives considered**: A "minimal" bank that only meters production/YOP/bank-trade and lets builds destroy resources (rejected — diverges from TS, makes the bank monotonically deplete, and weakens the invariant to a one-sided "never negative").

## D3 — Depletion algorithm: port `resolveBankProduction` verbatim

- **Decision**: Per resource R, with `total = d0 + d1`, `avail = bank[R]`: `total == 0` → no-op; `total <= avail` → both paid, `bank -= total`; `d0 > 0 and d1 > 0` → neither paid, bank unchanged; `d0 > 0` → seat 0 takes `avail`, bank → 0; else → seat 1 takes `avail`, bank → 0.
- **Rationale**: This is the exact TS function. The final branch is an **implicit else** (reachable only when `d0 == 0, d1 > 0`); mirroring must NOT add a redundant `d1 > 0` test and must preserve seat-0-before-seat-1 ordering, or subtle divergences appear in depletion fixtures.
- **Alternatives considered**: "Proportional" or "first-come" shortage splits (rejected — not the official Catan rule and not what TS does).

## D4 — Conformance proof via hands, not a serialized bank field

- **Decision**: Do NOT serialize the bank into the recorded snapshot. Re-record seeds 7/8/15 (must be byte-identical) and add ONE new depletion fixture that proves the rule through the resulting player **hands**.
- **Rationale**: Verified — the committed fixtures' `state_after` has exactly 7 keys (no bank), and the TS `normalise()` never reads `resourceBank`; so adding a bank to engine state is invisible to the assertion. Seeds 7/8/15 top out at ~6–8 combined held per resource (far below 19), so depletion never fires and the finite bank is a provable no-op on them. Serializing the bank would be a `schema_version` bump regenerating all fixtures — explicitly out of scope.
- **Alternatives considered**: Schema bump to assert the bank value (rejected — churns all fixtures, extends the TS normaliser, contradicts the byte-identical requirement, and is unnecessary since hands already encode the depletion effect).

## D5 — Observation schema unchanged (bank out of obs)

- **Decision**: The bank is engine state only; it is not added to any obs field.
- **Rationale**: Verified the bank is absent from `obs_schema.py`/`obs_encoder.py`. Keeping it out leaves every obs dim and the policy state-dict byte-identical → no migration, all v2 checkpoints loadable (constitution Principle III satisfied trivially). Depletion is rare, so the agent playing "unaware" of it costs little. Exposing the public bank is a deliberate, separately-gated follow-up.
- **Alternatives considered**: Add a 5-dim bank-remaining feature (rejected for this feature — resizes `CURR/NEXT_PLAYER_DIM`, triggers Principle III migration + retrain).

## D6 — Always-on (no feature flag)

- **Decision**: The finite bank is active in every engine path (training, eval, recorder). No toggle.
- **Rationale**: Owner-confirmed. It is the correct Colonist rule and depletion is rare in 1v1, so the MDP change is negligible; a flag is unnecessary ceremony. Pre-bank WR/Elo (v8 +121 Elo, the search ladder) get a one-line lineage note that they were measured on the infinite-bank engine.
- **Alternatives considered**: A default-OFF `finite_bank` flag preserving exact lineage comparability (considered and rejected by the owner as unnecessary given rarity).

## D7 — Native internal order, Charlesworth at the boundary

- **Decision**: Each engine keys/indexes its bank in its own native order (Python dict by engine `RESOURCES` name; Rust `[u8;5]` by `IDX_*`). Re-key to Charlesworth only where the conformance snapshot is emitted.
- **Rationale**: Respects the standing two-orderings constraint and avoids the resource-transposition bug class; the existing hand serialization already re-keys at the boundary, so the bank follows the same seam.

## D8 — Two-pass `distribute_resources` / `update_playerResources`

- **Decision**: Replace the per-vertex incremental grant with a two-pass structure: pass 1 tallies per-(player, resource) demand without mutating; pass 2 applies the depletion rule per resource and decrements the bank.
- **Rationale**: The all-or-nothing-per-resource rule is order-dependent and **cannot** be expressed by granting incrementally inside the hex loop (the first-iterated player would wrongly receive what the rule denies). The refactor must be exactly identity-preserving whenever `bank >= demand` so non-depleting games (incl. 7/8/15) are unchanged.
- **Rust note**: This changes the emitted event shape from per-(player, hex) to per-(player, resource). Consumers (`hand_tracker.rs`, `obs.rs`, `events.rs`) must be checked to still parse `ResourceChange` correctly — this is the main additivity hazard on the Rust side.

## D9 — RNG keystream preservation

- **Decision**: The refactor must not add/remove/reorder any `np.random` (or Rust RNG) consumption on non-depleting paths.
- **Rationale**: `steal_resource`/`draw_devCard` draw from the RNG in hand-size-dependent ways; any keystream shift makes seeds 7/8/15 drift for reasons unrelated to the bank. Verification: re-record on a clean checkout and `git diff` must be empty. Bank arithmetic itself consumes no randomness, so this is achievable.

## D10 — Crafting the depletion fixture

- **Decision**: Prefer a **scripted line of play via the recorder's dice/outcome replay seam** to deterministically drive one resource's combined demand above its remaining bank, rather than brute-force seed search. Capture both depletion sub-cases if practical (both-owed → neither; one-owed → partial).
- **Rationale**: 7/8/15 never approach 19, and random play rarely depletes within a short game; scripting the dice through the seam pins the exact sequence without RNG luck and keeps the fixture short and debuggable. Requires the Python oracle to be bank-aware first (otherwise the recorded hands disagree with the TS engine). The TS suite auto-globs `game-seed-*.json`, so the new file is picked up with no test-code change.
- **Alternatives considered**: Brute-force seed/step search over a longer `--max-main-turns` (viable fallback if the seam can't pin a clean depletion, but slower and less surgical).

## Open risks carried into implementation

- The two-pass distribution rewrite is the highest bug-density edit in both engines (D8). Mitigation: identity tests on non-depleting rolls + the byte-identical 7/8/15 re-record gate.
- The three-path centralization (D1) must capture the env handlers and recorder, not just `player.py` — a missed path passes 7/8/15 but breaks the depletion fixture. Mitigation: a test that drives each path (engine, env, recorder) through a depleting line and asserts identical bank/hands.
