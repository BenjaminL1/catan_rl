# Architecture Overview

One-page description of the training loop. For details, see the linked source files (paths are relative to repo root).

## Layers

```
┌─────────────────────────────────────────────────────┐
│ scripts/train.py                                    │
│   └─ resolve_config(yaml_path)                      │
│      └─ CatanPPO(config).train()                    │
└─────────────────────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌─────────────────┐            ┌──────────────────┐
│ algorithms/ppo  │            │ selfplay         │
│   trainer.py    │            │   league.py      │
│   arguments.py  │            │   game_manager   │
└─────────────────┘            └──────────────────┘
       │                               │
       │                               ▼
       │                       ┌──────────────────┐
       │                       │ env              │
       │                       │   catan_env.py   │
       │                       │   hand_tracker   │
       │                       └──────────────────┘
       ▼                               │
┌─────────────────┐                    ▼
│ models          │            ┌──────────────────┐
│   policy.py     │            │ engine           │
│   action_heads  │            │   game.py        │
│   observation/  │            │   board.py       │
│     tile_enc    │            │   player.py      │
│     player_mod  │            │   dice.py        │
│   distributions │            │   broadcast.py   │
└─────────────────┘            └──────────────────┘
       │
       ▼
┌─────────────────┐
│ algorithms/     │
│   common/       │
│     gae.py      │
│     rollout_buf │
└─────────────────┘
```

## Training step

```
1. GameManager.reset_all()
   ├─ each env samples opponent from League (PFSP / random / heuristic)
   └─ env.reset(opponent_options)

2. CatanPPO.collect_rollouts():
   └─ while steps < n_steps:
        ├─ batched policy.act()
        ├─ env.step() per env
        ├─ if opp_turn_pending: batched opponent NN inference
        └─ buffer.add(obs, action, reward, terminated, truncated, value, log_prob, masks)

3. Compute GAE with terminated/truncated split (Phase 0).
4. Per-batch advantage normalization.
5. PPO update for n_epochs (or until KL > target_kl).
6. League.maybe_add(policy snapshot) every N updates.
7. Periodic eval against heuristic / random / champion.
```

## Key entry points

| Concern | File |
|---|---|
| Train | `scripts/train.py` |
| Evaluate vs heuristic/random | `scripts/evaluate.py` |
| Play vs trained model (GUI) | `scripts/play_vs_model.py` |
| Setup-phase trainer | `scripts/train_setup.py` |
| Eval harness (rules-invariant / champion-bench / exploitability / league-rating) | `scripts/eval_harness.py` |
| Migrate pre-Phase-0 checkpoint | `scripts/migrate_checkpoint.py` |

## Phase 1 sample-efficiency upgrades

All five sub-features are config-flagged so leave-one-out ablations work.
Defaults preserve back-compat with `checkpoint_07390040.pt`; opt in via
`configs/phase1_full.yaml`.

| Sub-feature | Config key | Default | Effect |
|---|---|---|---|
| 1.1 PPO2 value clipping | `use_value_clipping`, `clip_range_vf` | `True`, `0.2` | Pessimistic value loss `max(MSE_unclipped, MSE_clipped)` |
| 1.2 Per-rollout adv. norm | `advantage_norm` | `"rollout"` | Standardize over the whole buffer; alt: `"batch"`, `"none"` |
| 1.3 Compact obs schema | `use_thermometer_encoding` | `True` (legacy) | Drop bucket8: 166/173 → 54/61 |
| 1.4 Dev-card count enc. | `use_devcard_mha` | `True` (legacy) | Replace MHA over padded sequence with bincount + MLP (~36k fewer params) |
| 1.5 D6 symmetry aug. | `symmetry_aug_prob` | `0.0` | Per-minibatch, with this probability sample one of 11 non-identity D6 elements and permute tile/corner/edge slots |

Phase 1 lineage is **not** state-dict-compatible with `checkpoint_07390040.pt`
(compact obs changes the model's first-layer input dim). Phase 1 training
runs from scratch; the frozen champion stays on the legacy 166/173 lineage
for benchmark purposes.

## Phase 0 diagnostics

The trainer's TensorBoard scalars now include per-head entropy diagnostics
that detect silent collapse on individual action heads:

  - `train/entropy_head_<name>` — unconditional mean entropy of head `<name>`
    (one of: `type, corner, edge, tile, resource1, resource2`).
  - `train/entropy_head_<name>_cond` — relevance-weighted mean: only averages
    over samples where this head's output actually contributed to the
    chosen action.
  - `train/entropy_collapse_flag` — set to `1.0` if any head's unconditional
    entropy stayed below `entropy_collapse_threshold` (config) for
    `entropy_collapse_consecutive_updates` consecutive updates.

The legacy `train/entropy` joint-entropy scalar is preserved.

GAE handling now distinguishes terminated (real game-over) from truncated
(`max_turns` cutoff): terminations zero the bootstrap value; truncations keep
the bootstrap (`V(s_T)`) but reset the GAE accumulator at the boundary. See
`catan_rl.algorithms.common.gae` for the recurrence and
`tests/unit/algorithms/test_gae.py` for the contract tests.
