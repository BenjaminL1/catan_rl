# Implementation Plan: Finite Per-Resource Bank (Oracle ↔ TS Parity)

**Branch**: `feat/resource-bank-oracle` | **Date**: 2026-06-23 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/009-finite-resource-bank/spec.md`

## Summary

Mirror the Torevan TypeScript engine's finite 19-per-resource bank (with the official depletion rule and full cost recirculation) in the `catan_rl_v2` reference engines so the cross-engine Python↔TS conformance harness stays green. The technical approach: **centralize the bank as engine state and route every resource mutation through one production/recirculation path** (eliminating the current three duplicated paths), reproduce the TS `resolveBankProduction` depletion semantics verbatim, keep the bank out of the observation schema, and prove parity via byte-identical re-records of seeds 7/8/15 plus one new depletion fixture. Python + conformance land first (parity-critical path), then the Rust mirror, then documentation.

### Implementation Strategy (high level — granular tasks in `tasks.md`)

- **Phase A — Python engine + conformance (P1, parity-critical).** Add `resourceBank` to `catanBoard` (so deepcopy clones it); restructure `update_playerResources` to a two-pass tally→apply with the depletion rule; centralize recirculation/draw helpers on the engine; route the env's own YOP/Discard/BankTrade handlers and the recorder's inline arithmetic through those helpers; gate YOP/BankTrade legal moves on availability. Test-first (`tests/unit/engine/test_resource_bank.py` + conservation walk). Re-record seeds 7/8/15 → prove byte-identical; craft + record the depletion fixture into the Torevan tree; run the TS conformance suite.
- **Phase B — Rust mirror (P2).** Add `bank: [u8; N_RESOURCES]` to `GameState`; two-pass `distribute_resources`; recirculate build/dev/discard/trade costs; `checked_sub` draws on the YOP/bank-trade reject paths; verify the per-hex→per-resource event-shape change against `hand_tracker.rs`/`obs.rs`/`events.rs`. Conservation + depletion-branch tests; `cargo test`.
- **Phase C — Docs + lineage (P3).** `CLAUDE.md` ruleset table + engine-implications; Torevan `mvp-build-plan.md` D11 flip + D12 append; Torevan `state.ts` "never drift" comment; lineage note that pre-bank Elo/WR was measured on the infinite-bank engine.

Gate-first: Phase A's conformance proof (byte-identical 7/8/15 + green TS suite incl. depletion fixture) is the go/no-go before Phase B is committed.

## Technical Context

**Language/Version**: Python 3.11+ (engine + env + conformance recorder); Rust (edition per `crates/catan_engine/Cargo.toml`, PyO3 extension).

**Primary Dependencies**: NumPy (engine RNG / arrays), PyO3 + maturin (Rust↔Python), the Torevan `@torevan/engine` TS package (parity counterpart, read-only). No new runtime dependencies.

**Storage**: JSON conformance replay-logs (`game-seed-*.json`) in `../Torevan/packages/engine/src/conformance/`.

**Testing**: `pytest` (unit + integration), `cargo test` (in-file `#[cfg(test)]` + `tests/state_integration.rs`), and the Torevan Vitest conformance suite (`npm run test -w @torevan/engine`). CI = ruff + mypy --strict + pytest on Python 3.11+.

**Target Platform**: Apple M1 Pro (MPS for training, CPU for eval/conformance); engine logic is platform-agnostic pure Python / Rust.

**Project Type**: Reinforcement-learning game engine (library) with a cross-engine parity harness — single repo, two engine implementations.

**Performance Goals**: No regression to the current engine step rate (~120 sims/sec value-forward-bound). The two-pass distribution is O(active hexes + 5 resources) per roll — negligible. Rust `[u8; 5]` bank adds 5 bytes per cloned search node — negligible for determinized MCTS.

**Constraints**: Byte-for-byte parity with the TS engine (clamp nothing TS does not; Charlesworth order at the conformance boundary, native engine order internally); RNG keystream must be undisturbed on non-depleting games; observation vector shape must be unchanged (no checkpoint migration); no AI co-author trailers; do not commit/push into the Torevan repo.

**Scale/Scope**: ~2 engine implementations, ~12 Python change-sites + ~9 Rust change-sites, 1 new conformance fixture, 3 existing fixtures re-recorded, docs in 2 repos.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Verdict | Justification |
|---|---|---|
| **I. The 1v1 Ruleset Is Sacred** | ✅ PASS (with required statement) | This touches a game-rule constant (resource supply), so per Principle I the change MUST state how it preserves the ruleset: it **adds** the finite 19-per-resource bank + official depletion rule, which is *part of* the Colonist.io 1v1 ruleset — the current infinite supply is the deviation. All other invariants (15 VP, 2 players, P2P-trade disabled, 9-card discard, Friendly Robber, StackedDice) are untouched. The commit message will carry this statement. |
| **II. Engine Integrity** | ✅ PASS | The change makes the engine match Colonist.io *more* exactly (closes an existing TS↔Python drift), with explicit surfaced justification. Eval comparability impact is limited to the rare depletion case and is annotated in the lineage (pre-bank Elo measured on the infinite-bank engine). |
| **III. Backward-Compatible, Additive Artifacts** | ✅ PASS | The bank is deliberately kept OUT of the observation schema → no policy state-dict shape change → no migration; all existing v2 checkpoints stay loadable. No TensorBoard scalar renames. |
| **IV. Test-First & Green CI** | ✅ PASS | Tests are written alongside (conservation invariant, depletion branches, recirculation, gating). ruff + mypy --strict + pytest + cargo test kept green; CI per-check conclusions verified explicitly. |
| **V. Self-Play Is 2-Player Zero-Sum** | ✅ PASS | No self-play machinery is touched; the bank is symmetric 2-player state. |

**Additional-constraints check**: the two resource orderings (`RESOURCES` engine vs `RESOURCES_CW`) are respected — the bank is keyed/indexed in each engine's native order and re-keyed to Charlesworth only at the conformance boundary. Obs/action-head shapes unchanged.

**Result**: No violations. Complexity Tracking not required.

## Project Structure

### Documentation (this feature)

```text
specs/009-finite-resource-bank/
├── plan.md              # This file
├── research.md          # Phase 0 — decisions consolidated from this session's recon
├── data-model.md        # Phase 1 — ResourceBank / ResourceDemand / ConformanceFixture
├── quickstart.md        # Phase 1 — how to validate parity end-to-end
├── contracts/
│   └── internal-interfaces.md   # the bank invariant + depletion fn + recirculation table both engines honor
├── checklists/
│   └── requirements.md  # spec quality checklist (already created)
└── tasks.md             # Phase 2 — /speckit-tasks (NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/catan_rl/engine/
├── board.py        # catanBoard.__init__ — add resourceBank (native order, 19 each) so deepcopy clones it; bank helper methods
├── game.py         # update_playerResources — TWO-PASS tally→apply depletion; setup grant flat decrement
└── player.py       # build_road/settlement/city, draw_devCard, trade_with_bank, play_devCard (YOP/Monopoly), discardResources, steal_resource → recirc/draw via engine helpers

src/catan_rl/env/
└── catan_env.py    # route its OWN YOP/Discard/BankTrade handlers through the engine bank helpers; YOP/BankTrade legal-move gating

src/catan_rl/conformance/
└── recorder.py     # route inline grant/discard/YOP arithmetic through the same engine helpers (no divergent math)

crates/catan_engine/src/
└── state.rs        # GameState.bank; distribute_resources two-pass; recirc in build/dev/discard/trade; YOP/bank-trade draw; setup decrement

crates/catan_engine/tests/
└── state_integration.rs   # full-game conservation walk (template: resource_conservation_across_setup_grants)

tests/unit/engine/
└── test_resource_bank.py  # NEW — init, depletion branches, recirculation, setup, gating, monopoly/steal-neutral, conservation

../Torevan/packages/engine/src/conformance/
└── game-seed-<N>.json     # NEW depletion fixture (written into Torevan tree; committed by the Torevan session)
```

**Structure Decision**: Single repo, two engine implementations sharing one parity contract. The decisive structural choice is **centralizing the resource bank on the engine** (`catanBoard` in Python, `GameState` in Rust) with a single production/recirculation code path, so the env handlers and the conformance recorder consume the engine's bank logic instead of each re-implementing it — this is what eliminates the three-path drift hazard.

## Complexity Tracking

> No Constitution Check violations — section intentionally empty.
