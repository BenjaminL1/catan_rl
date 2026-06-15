# Phase 1 Data Model: Expert Iteration

Reuses existing types where possible; new types are config + result records.

## SearchLabelConfig (new dataclass, isolated — not on TrainConfig)

| Field | Type | Default | Notes |
|---|---|---|---|
| `base_ckpt` | str | v6 `ckpt_000001499.pt` | the search teacher's policy (round n's base) |
| `sims_per_move` | int | 50 | search budget per labeled decision (the 003 gated budget) |
| `opponent` | str | `"heuristic"` | label-game opponent (`heuristic` pilot; `policy:<ckpt>` self-play later) |
| `n_positions` | int | 5000 | target non-forced labeled positions (stop after a game crosses it) |
| `discount` | float | 0.998 | γ for `z_disc` (matches BC) |
| `seed` | int | 0 | reproducibility |
| `out_dir` | str | `data/exit/round_0` | BcDataset-compatible shard dir |

**Validation**: `sims_per_move>0`; `n_positions>0`; `0<discount<=1`; opponent in {heuristic, random, policy:PATH}.

## Search-labeled shard (REUSED format — `catan_rl.bc.dataset`)

Per non-forced agent decision: `obs_*` (the policy obs keys), `action` (6,) int64 = the
**search's chosen action**, `mask_*` (the 9 mask keys), `belief_target` (5,) float32,
`z_disc` float32 (discounted game outcome). NPZ shards + `manifest.json` — read by the
existing `BcDataset`. (Forced/`mask_sum==1` decisions skipped at write time, as in BC.)

## DistillConfig (new — distillation = warm-started BC fine-tune)

| Field | Type | Default | Notes |
|---|---|---|---|
| `init_ckpt` | str | the round base (v6) | warm-start weights (new `train_bc(init_ckpt=)`) |
| `data_dir` | str | the labeled shard dir | |
| `out_dir` | str | `runs/exit/round_0/distill` | distilled checkpoint dir |
| `peak_lr` | float | 5e-5 | low — nudging a warm-started net, not retraining (D8) |
| `max_epochs` | int | 5 | short; early-stop on val (D8) |
| `value_weight` / `belief_weight` | float | 0.10 / 0.05 | BC defaults |
| `seed` | int | 0 | |

## GateResult / RoundResult (new — JSON)

- **GateResult**: `{passed: bool, wr_vs_v6, ci:[lo,hi], n, wr_vs_heuristic, wr_vs_prior_rung, failure_mode: str|None}` — PASS iff `ci.lo > 0.50` at n≥200 then n≥500 (SC-001); the extra opponents guard catastrophic forgetting (SC-005 edge case).
- **RoundResult** (US2): `{round_idx, base_ckpt, distilled_ckpt, gate: GateResult, elo_delta_vs_v6}`; the flywheel keeps the best distilled ckpt as the next round's base.

## Reused entities

`SearchAgent`/`SearchConfig` (003 teacher), `BcDataset` + `bc_loss` + `train_bc`
(distillation), `EvalMatchupResult` + `wilson_interval` (gate), `CatanPolicy` + the
checkpoint manager (v2-lineage I/O), `scripts/elo_ladder.py` (US3 placement).
