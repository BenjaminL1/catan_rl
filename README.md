# Catan RL: 1v1 Settlers of Catan with Custom PPO

A reinforcement learning agent trained to play 1v1 Settlers of Catan (Colonist.io ruleset) using a custom PPO implementation. The agent uses a Transformer-based observation encoder and a 6-head autoregressive action representation to handle Catan's complex, multi-phase action space.

Trained via self-play league on an Apple M1 Pro (CPU-only). Reached ~7.4M steps before archiving.

---

## Credits

Game engine adapted from [Catan-AI](https://github.com/kvombatkere/Catan-AI) by Karan Vombatkere.

RL architecture inspired by [settlers_of_catan_RL](https://github.com/henrycharlesworth/settlers_of_catan_RL) by Henry Charlesworth.

---

## 1v1 Rules (Colonist.io Variant)

| Rule | Value |
|------|-------|
| Win condition | **15 VP** (vs 10 in standard 4-player) |
| Player trading | **Disabled** — bank/port only |
| Discard threshold | **9 cards** (vs standard 7) |
| Friendly Robber | Cannot place robber on hex adjacent to player with <3 visible VP |
| Dice | **StackedDice** — shuffled bag of 36 outcomes with noise + Karma mechanic |
| Setup | Snake draft (1→2→2→1); 2nd settlement yields starting resources |

---

## Architecture

### Observation (Dict, Charlesworth-Style)

| Component | Shape | Encoder | Output |
|-----------|-------|---------|--------|
| `tile_representations` | 19 × 79 | Transformer (2 layers, 4 heads) → projection | 475-dim |
| `current_player_main` | 166-dim | MLP | 128-dim |
| `current_player_hidden_dev` / `_played_dev` | seq × 6 | Embedding + MHA | 25-dim each |
| `next_player_main` | 173-dim | MLP | 128-dim |

All components concatenated → Linear → LayerNorm → ReLU → **512-dim state vector**

Total parameters: **1.54M**

### Action Space (6 Autoregressive Heads)

| Head | Size | Purpose |
|------|------|---------|
| 0 `action_type` | 13 | Which action (BuildSettlement, BuildCity, BuildRoad, EndTurn, MoveRobber, BuyDevCard, PlayKnight, PlayYoP, PlayMonopoly, PlayRoadBuilder, BankTrade, Discard, RollDice) |
| 1 `corner` | 54 | Settlement / city vertex |
| 2 `edge` | 72 | Road edge |
| 3 `tile` | 19 | Robber placement hex |
| 4 `resource_1` | 5 | Primary resource (YoP / Monopoly / Trade-give / Discard) |
| 5 `resource_2` | 5 | Secondary resource (YoP 2nd / Trade-receive) |

Each head is a 2-layer MLP → MaskedCategorical. Joint log-prob is the sum of relevant head log-probs (irrelevant heads zero-weighted via registered buffers).

### PPO Hyperparameters

| Param | Value |
|-------|-------|
| n_steps | 2048 |
| n_envs | 4 |
| batch_size | 512 |
| n_epochs | 10 |
| γ (discount) | 0.995 |
| λ (GAE) | 0.95 |
| clip ε | 0.2 |
| LR | 3e-4 → 1e-5 (linear decay) |
| Entropy coef | 0.04 → 0.005 (annealed, updates 50–200) |
| Value normalization | Running normalizer |
| Total target steps | 100M |

### Self-Play (League)

- In-memory deque of past policy state dicts (maxlen=100)
- New policy added every 4 PPO updates
- Sampling: uniform base + linear recency bias toward most recent 25 policies
- Pure self-play (no random opponents after warm-up)

---

## Project Structure

```
catan/
  engine/               Game logic (board, player, geometry, StackedDice)
  agents/               Heuristic and random AI opponents
  rl/
    env.py              Gymnasium environment wrapper (obs, action masking)
    distributions.py    MaskedCategorical distribution
    debug_wrapper.py    Diagnostic wrapper for env debugging
    models/
      observation_module.py     Transformer TileEncoder + player encoders
      action_heads_module.py    6-head autoregressive action selection
      policy.py                 CatanPolicy (encoder + heads + value net)
      build_agent_model.py      Factory and constants
      utils.py                  Weight init, running value normalizer
    ppo/
      ppo.py            Custom PPO training loop (CatanPPO)
      rollout_buffer.py Composite-action rollout buffer with tensor finalization
      game_manager.py   Multi-env rollout with league opponent sampling
      league.py         Past-policy pool for self-play
      arguments.py      Hyperparameter configs (TRAIN_CONFIG, MODEL_CONFIG)
      utils.py          GAE, explained variance, LR/entropy schedules
scripts/
  train.py              Main training entry point
  evaluate.py           Evaluation vs heuristic / random opponents
  play_vs_model.py      Interactive play against a trained checkpoint
  test_integration.py   Environment integration tests
```

---

## Getting Started

### Install

```bash
pip install gymnasium torch numpy pygame tqdm tensorboard
```

### Train

```bash
# Fresh run
python scripts/train.py --verbose

# Resume from checkpoint
python scripts/train.py --resume checkpoints/train/checkpoint_XXXXXXXX.pt --verbose
```

### Monitor

```bash
tensorboard --logdir runs/train/
```

### Evaluate

```bash
python scripts/evaluate.py --model checkpoints/train/checkpoint_XXXXXXXX.pt
```

---

## Training Notes

- Hardware: Apple M1 Pro, CPU-only (MPS too slow at batch=1 inference)
- Real-world throughput: ~30–35 steps/s during active compute; ~2–2.5M steps/day accounting for sleep/idle
- Checkpoints saved every 500k steps; eval vs heuristic every 100k steps
- Bottleneck profiling (n_envs=4, n_steps=2048): NN inference ~53% of rollout time, obs-building ~33%
- Obs-building optimized: cached per-tile static features and corner geometry at episode reset → ~1.8× speedup on `_build_tile_features`
