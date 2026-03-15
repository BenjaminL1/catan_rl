# Catan RL: Mastering 1v1 Settlers of Catan with PPO

This project implements a high-performance Reinforcement Learning (RL) agent trained to play one-vs-one Settlers of Catan. Built upon a refactored version of the classic Catan-AI engine, this environment uses a custom PPO implementation with multi-head autoregressive action selection to handle the complex, multi-phase action space of the game.

## Credits & Citations

The core game logic and hex-grid geometry are adapted from the Catan-AI repository by Karan Vombatkere.

> Vombatkere, K. (2018). Catan-AI: A Python Implementation of Settlers of Catan with Heuristic Agents. GitHub. https://github.com/kvombatkere/Catan-AI

The RL architecture is inspired by Henry Charlesworth's Settlers of Catan RL project:

> Charlesworth, H. (2023). settlers_of_catan_RL. GitHub. https://github.com/henrycharlesworth/settlers_of_catan_RL

## RL Architecture

### Action Space (Multi-Head Composite)

The agent uses 6 autoregressive action heads:

| Head | Output Size | Purpose |
|------|------------|---------|
| ACTION_TYPE | 12 | Which action to take |
| CORNER | 54 | Settlement / city vertex |
| EDGE | 72 | Road edge |
| TILE | 19 | Robber placement hex |
| RESOURCE_1 | 5 | Primary resource (YoP / Monopoly / Trade) |
| RESOURCE_2 | 5 | Secondary resource (YoP 2nd / Trade receive) |

### Observation Space (1,258 Features)

The agent receives a detailed belief state including:
- **Hex features**: resource types, number tokens, robber location (19 tiles × 8 features)
- **Vertex features**: ownership, building types, port access (54 vertices × 14 features)
- **Edge features**: road occupancy (72 edges × 4 features)
- **Scalar features**: resources, dev cards, VP counts, trade rates, game phase (62 features)

### Training (Charlesworth-Style)

Single self-play phase from start—no curriculum. Trains against random initially;
league of past policies is populated every 4 updates for future policy-opponent
support. Entropy annealing, linear LR decay, M1-optimized (CPU, in-process multi-env).

## Project Structure

```
catan/
  engine/         Game logic (board, player, geometry, dice)
  agents/         Heuristic and Random AI opponents
  rl/
    env.py        Gymnasium environment wrapper
    distributions.py   Masked categorical distribution
    debug_wrapper.py   Diagnostic wrapper for env debugging
    league.py     Ghost opponent pool management
    models/
      observation_module.py   Transformer-based entity encoders
      action_heads_module.py  Multi-head autoregressive action selection
      policy.py               Top-level CatanPolicy (obs + heads + value)
      build_agent_model.py    Factory function and constants
      utils.py                Weight init, value normalizer
    ppo/
      ppo.py           Custom PPO training loop
      rollout_buffer.py Composite action rollout storage
      arguments.py     Hyperparameter configs per phase
      utils.py         GAE, explained variance, scheduling
scripts/
  train.py         Main training entry point
  evaluate.py      Evaluation with statistics
  play_vs_model.py Interactive play against trained agent
  test_integration.py  Environment integration tests
```

## Getting Started

### Prerequisites

```bash
pip install gymnasium torch numpy pygame tqdm tensorboard
```

### Training

```bash
python scripts/train.py --verbose
python scripts/train.py --resume checkpoints/train/final_model.pt --verbose
```

### Monitoring

```bash
tensorboard --logdir runs/train/
```
