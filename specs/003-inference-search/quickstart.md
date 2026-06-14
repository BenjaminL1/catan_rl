# Quickstart / Validation: Inference-Time Search

Runnable scenarios that prove the feature works end-to-end. References [contracts](contracts/internal-interfaces.md) + [data-model](data-model.md); no implementation code here.

## Prerequisites

- A v6 frontier checkpoint, e.g. `runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt` (the "raw policy").
- The Elo ladder baseline: `runs/elo_ladder_full.json` (raw v6 ≈ 1100 Elo).
- CPU-pinned; no training running.

## Scenario 1 — Bake-off gate (US1, the go/no-go) — DO THIS FIRST

```bash
# minimal search vs the raw policy it wraps, n=200 seat-symmetrized
catan-rl-search-eval --ckpt <v6_1499> --opponent policy:<v6_1499> \
    --sims 50 --n-games 200 --seed 0 --out runs/search/bakeoff_n200.json
```
**Expected (PASS)**: `wr` with **Wilson lower bound > 0.50**. Then re-confirm:
```bash
catan-rl-search-eval --ckpt <v6_1499> --opponent policy:<v6_1499> --sims 50 --n-games 500 --seed 0 --out runs/search/bakeoff_n500.json
```
**PASS** ⇒ proceed to the full build. **FAIL** (`wr≈0.50` or LB≤0.50) ⇒ STOP; record the failure mode and pivot per research D4 (lean on priors / bounded-rollout-to-late-state / fix the leaf).

## Scenario 2 — Strength scales with compute (US2)

```bash
for B in 0.25 1 5; do
  catan-rl-search-eval --ckpt <v6_1499> --opponent policy:<v6_1499> \
      --time-budget $B --n-games 200 --seed 0 --out runs/search/ladder_${B}s.json
done
```
**Expected**: `wr` vs the raw policy **monotonically increases** with budget (0.25s < 1s < 5s, within CI). Flat/non-monotone ⇒ likely a bug (the search adds noise, not lookahead).

## Scenario 3 — Determinism (SC-003)

```bash
catan-rl-search-eval --ckpt <v6_1499> --opponent heuristic --sims 100 --n-games 10 --seed 7 --out runs/search/det_a.json
catan-rl-search-eval --ckpt <v6_1499> --opponent heuristic --sims 100 --n-games 10 --seed 7 --out runs/search/det_b.json
diff runs/search/det_a.json runs/search/det_b.json   # expect: identical
```

## Scenario 4 — Elo uplift on the ladder (US3)

Add the search agent as a rung (search@1s) alongside the raw v6 rung and re-run the ladder fitter (`/tmp/elo_ladder.py`-style). **Expected**: `search@1s` Elo > raw `v6_u1499`, non-overlapping CI, when the gate passed. A 0.60 WR ≈ +70 Elo.

## Scenario 5 — No-regression + legality (SC-005/006)

```bash
pytest tests/unit/search tests/unit/eval tests/integration/test_selfplay_smoke.py -q
```
**Expected**: search unit tests pass; existing eval/smoke tests unchanged (byte-identical behavior with search absent); the rules-invariant check (`eval/rules_invariants.py`) reports zero violations across bake-off games.

## Done

Feature is validated when Scenario 1 PASSES (gate), Scenario 2 shows monotone scaling, Scenario 3 is deterministic, Scenario 4 reports a positive Elo delta, and Scenario 5 is green — OR Scenario 1 FAILS and the failure + pivot are documented (still a valid, decisive outcome).
