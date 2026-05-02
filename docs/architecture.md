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

## Phase 2 architecture upgrades

Four config-flagged additions on top of Phase 1. Defaults preserve Phase 1
behavior; opt in via `configs/phase2_full.yaml` (or the four leave-one-outs).

| Sub-feature | Config key | Default | Effect |
|---|---|---|---|
| 2.1 Axial pos. embedding | `use_axial_pos_emb`, `axial_pos_dim` | `False`, `24` | Learned 2D embedding indexed by hex axial coords `(q, r)`; concatenated to per-tile features before the transformer projection. Breaks the encoder's permutation-equivariance over tiles |
| 2.2 Modern transformer recipe | `transformer_dropout`, `transformer_activation` | `None`, `"relu"` | Pre-norm was already on; adds optional dropout (0.05 in `phase2_full`) and GELU FFN activation |
| 2.4 AdaLN action heads | `action_head_film` | `False` | Replaces concat-MLP conditioning on context-using heads (corner / resource1 / resource2) with FiLM modulation: `(1+γ) ⊙ LN(x) + β` where `(γ, β)` are per-sample, generated from the head's context. γ-init=0 → identity at construction |
| 2.5 Decoupled value tower | `value_head_mode` | `"shared"` | `"decoupled"` builds a second `ObservationModule` exclusively for the value head, breaking gradient interference between policy loss and value loss. ~+0.7M params |

Phase 2 lineage stays on the Phase 1 obs schema (compact 54/61). The full
phase2_full policy is ~2.22M params (vs ~1.54M for phase1_full); the bulk of
the increase is the second observation encoder for the value head.

`build_agent_model.DEFAULT_MODEL_CONFIG`, `arguments.MODEL_CONFIG`, and the
trainer's `model_kwargs` whitelist all carry the four new keys. The four
leave-one-out configs live next to `phase2_full.yaml`:
`phase2_no_axial_pos`, `phase2_no_transformer_recipe`, `phase2_no_film`,
`phase2_no_decoupled_value`.

## Phase 3 self-play diversity

Five config-flagged additions on top of Phase 2; defaults preserve Phase 2
behavior. Opt in via `configs/phase3_full.yaml` (or one of five leave-one-outs).

| Sub-feature | Config key | Default | Effect |
|---|---|---|---|
| 3.1 PFSP-hard | `pfsp_mode`, `pfsp_p`, `pfsp_window` | `'linear'`, `2.0`, `32` | `'hard'` switches to AlphaStar `(1-w)^p` priorities biased toward losses; sliding 32-game window per opponent |
| 3.2 Latest-policy reg. | `latest_policy_weight` | `0.0` | With this prob, league emits `('current_self', None, -2)`; trainer fills with a fresh in-place snapshot |
| 3.3 Duo exploiter cycle | `exploiter_mode` | `'off'` | **Scaffolded only** — config keys parse but the interleaved exploiter trainer ships in a follow-up PR. `'duo'` emits a deferred-feature notice in train.log |
| 3.4 TrueSkill ratings | `use_trueskill`, `trueskill_decay` | `False`, `1.001` | TrueSkill (or Glicko-2 fallback) per-policy ratings; σ inflated each PPO update so non-stationarity is visible |
| 3.5 Nash pruning | `prune_strategy`, `prune_every`, `nash_top_k`, `nash_prune_round_games` | `'fifo'`, `20`, `32`, `50` | `'nash'` runs replicator dynamics on a top-K round-robin payoff matrix every `prune_every` adds; drops the lowest-Nash-mass entry |
| 3.6 Opponent ID emb. | `use_opponent_id_emb`, `opp_id_emb_dim`, `opp_id_mask_prob` | `False`, `16`, `0.40` | Two embeddings (kind ∈ {unknown, random, heuristic, self_latest, league, main_exploiter}; policy_id ∈ {0..league_maxlen}) concatenated to the fusion input. With `opp_id_mask_prob` the kind is force-set to `unknown` to keep the policy robust at eval time |

Phase 3 builds on the Phase 2 architecture (and inherits its compact-obs /
fresh-training lineage). `phase3_full` policy is ~2.24M params (vs ~2.22M
phase2_full) — the only delta is the small embedding pair on the obs encoder.

Files modified:
- `src/catan_rl/selfplay/league.py`: PFSP mode dispatch, sliding-window WR,
  `current_self` emission, `prune_nash` math.
- `src/catan_rl/selfplay/game_manager.py`: `current_policy_state_fn` hook,
  `record_match_fn` hook, opponent-id options plumbed to env reset.
- `src/catan_rl/selfplay/ratings.py`: unchanged (built in Phase 0; activated here).
- `src/catan_rl/env/catan_env.py`: opponent-kind enum, `_opponent_id_obs`
  with random masking, optional dict-obs keys.
- `src/catan_rl/models/observation_module.py`: opponent-id embedding pair,
  fusion-input width grows by `opp_id_emb_dim`.
- `src/catan_rl/algorithms/common/rollout_buffer.py`: opt-in
  `store_opponent_id` per-step storage.
- `src/catan_rl/algorithms/ppo/trainer.py`: rating table + decay + scalar
  logging, Nash-pruning round-robin, exploiter-cycle stub.

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
