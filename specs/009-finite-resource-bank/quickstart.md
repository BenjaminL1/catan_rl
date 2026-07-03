# Quickstart: Validate the Finite Resource Bank

End-to-end validation that the oracle's finite bank matches the Torevan TS engine. Run from the feature worktree root unless noted. See [contracts/internal-interfaces.md](./contracts/internal-interfaces.md) for the rules being verified and [data-model.md](./data-model.md) for the movement table.

## Prerequisites

- Feature worktree checked out (`feat/resource-bank-oracle`), Python env installed (`pip install -e .`), Rust toolchain for `cargo`.
- The Torevan repo at `../Torevan` on branch `feat/finite-resource-bank` with `npm` deps installed.

## 1. Python unit + conservation tests (Phase A gate)

```bash
pytest tests/unit/engine/test_resource_bank.py -q
```

**Expected**: green. Covers bank init (19 each), all depletion branches (C2: both-owed-short → neither; one-owed-short → partial; no-op), recirculation per build/dev/trade/discard (C3), draws (C4: production/setup/YOP/bank-trade), bank-neutral monopoly/steal (C5), legal-move gating (C6), the single-path requirement (C7), and the conservation invariant `Σ hands + Σ bank == 95` after every action across full self-play games.

## 2. Byte-identical re-record of seeds 7/8/15 (parity no-op proof)

```bash
PYTHONPATH=src python scripts/record_conformance.py --seeds 7 8 15 \
  --out-dir ../Torevan/packages/engine/src/conformance
git -C ../Torevan diff --stat -- packages/engine/src/conformance/game-seed-7.json \
  packages/engine/src/conformance/game-seed-8.json packages/engine/src/conformance/game-seed-15.json
```

**Expected**: the `git diff` is **empty** — proving the finite bank is a no-op on these non-depleting games and the RNG keystream is undisturbed (D9). Any diff means the bank logic perturbed the non-depleting path or the keystream — investigate before proceeding.

## 3. Record the new depletion fixture

```bash
PYTHONPATH=src python scripts/record_conformance.py --seeds <N> \
  --out-dir ../Torevan/packages/engine/src/conformance
```

(`<N>` = the crafted depletion seed/scripted line — see research D10.) **Expected**: a new `game-seed-<N>.json` whose hands reflect a bank-short payout (one resource's combined demand exceeded its remaining supply, so a player received fewer cards than nominally owed).

## 4. Cross-engine parity: Torevan conformance suite

```bash
cd ../Torevan && npm run test -w @torevan/engine
```

**Expected**: green, including the new depletion fixture (the suite auto-globs `game-seed-*.json`). This is the **parity guarantee**: the TS engine replays every fixture — including the depleting one — to byte-identical player-hand state.

## 5. Rust mirror (Phase B gate)

```bash
cargo test -p catan_engine
```

**Expected**: green, including the full-game conservation walk (`Σ hands + Σ bank == 95` after every action) and the three depletion-branch unit tests, and confirming the `distribute_resources` event-shape change didn't break `hand_tracker.rs`/`obs.rs`/`events.rs` consumers (D8).

## 6. Regression gates

```bash
ruff check . && mypy --strict src/catan_rl && pytest -q
```

**Expected**: green. Confirms no obs-shape change (existing v2 checkpoints still load) and no broader regression.

## Definition of done

- Steps 1–6 green.
- Seeds 7/8/15 re-record byte-identical (step 2).
- The depletion fixture exists and the TS suite passes with it (steps 3–4).
- Docs updated (CLAUDE.md ruleset table; Torevan `mvp-build-plan.md` D11 flip + D12; Torevan `state.ts` comment) and the game-rule-change statement is in the commit message.
- The new fixture is written into the Torevan tree but NOT committed/pushed there (the Torevan session commits it).
