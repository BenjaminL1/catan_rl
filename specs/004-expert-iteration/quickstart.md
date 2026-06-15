# Quickstart / Validation: Expert Iteration

Runnable scenarios proving the feature works. References [contracts](contracts/internal-interfaces.md) + [data-model](data-model.md).

## Prerequisites
- v6 base: `runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt`.
- The 003 search (`catan_rl.search`) + the BC pipeline (`catan_rl.bc`) present.
- CPU for search labeling + eval; MPS for the distillation SGD.

## Scenario 1 — Pilot gate (US1, the go/no-go) — DO THIS FIRST
```bash
# label (~5k positions, search@50 vs heuristic) -> distill (warm-start v6) -> gate
python scripts/run_exit_pilot.py --sims 50 --n-positions 5000 --seed 0
```
**Expected (PASS)**: `runs/exit/round_0/gate.json` with the distilled policy beating raw v6,
**Wilson lower bound > 0.50** at n≥200 then n≥500, as a SEARCH-FREE agent; 0 ruleset
violations; no regression vs the heuristic / prior rung.
**FAIL** (WR≈0.50 or LB≤0.50) ⇒ STOP; record the pivot (soft visit-distribution targets /
more positions / more sims via batched leaf eval / LR+epoch tuning) and DO NOT build US2/US3.

## Scenario 2 — Compounding flywheel (US2, only if gate passed)
```bash
python scripts/run_exit_pilot.py --rounds 2   # each round seeds from the prior distilled best
```
**Expected**: per-round Elo over raw v6 monotone non-decreasing within CI (SC-003).

## Scenario 3 — Banked fast agent on the Elo ladder (US3, only if gate passed)
```bash
python scripts/elo_ladder.py --nps 100   # add the distilled ckpt as a (search-free) rung
```
**Expected**: the distilled rung's Elo over raw v6 is positive with a non-overlapping CI
(SC-004), at raw-policy move speed (no MCTS).

## Scenario 4 — Reproducibility (SC-006)
```bash
python scripts/run_exit_pilot.py --seed 0 --out-suffix a
python scripts/run_exit_pilot.py --seed 0 --out-suffix b
# expect: identical labeled shards + identical gate verdict
```

## Scenario 5 — No-regression + warm-start (SC-005)
```bash
pytest tests/unit/expert_iteration tests/unit/bc -q
```
**Expected**: ExIt unit tests pass; `train_bc(init_ckpt=None)` behavior byte-identical to
before (existing BC tests green); distilled checkpoints load via the existing manager.

## Done
Validated when Scenario 1 PASSES (gate) — then 2/3 show compounding + a banked search-free
agent — OR Scenario 1 FAILS and the failure + pivot are documented (still decisive).
