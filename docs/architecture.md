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
| Eval harness (Phase 0+) | `scripts/eval_harness.py` (planned) |
