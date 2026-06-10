# Catan RL — 1v1 Settlers of Catan with Custom PPO

A reinforcement-learning agent for **1v1 Settlers of Catan** under the
Colonist.io ruleset, trained with a custom PPO implementation. The policy
pairs a tile transformer + GNN board encoder with a **6-head autoregressive
action space** that handles Catan's multi-phase, structured action set.

Training path: **heuristic bootstrap → league self-play**. The agent trains
on MPS (Apple M1 Pro); eval is pinned to CPU. This is the **v2** codebase
(`src/catan_rl/`); the old v1 `catan/` tree is deprecated and gone.

---

## Credits

- Game engine adapted from [Catan-AI](https://github.com/kvombatkere/Catan-AI) by Karan Vombatkere.
- RL architecture inspired by [settlers_of_catan_RL](https://github.com/henrycharlesworth/settlers_of_catan_RL) by Henry Charlesworth (Charlesworth resource order).

---

## 1v1 Rules (Colonist.io Variant)

| Rule | Value |
|------|-------|
| Win condition | **15 VP** (vs 10 in standard 4-player) |
| Players | **2** |
| Player-to-player trading | **Disabled** — bank/port only |
| Discard threshold on a 7 | **9 cards** (vs standard 7) |
| Friendly Robber | No robber on a hex adjacent to a player with `< 3` visible VP |
| Dice | **StackedDice** — shuffled bag of 36 outcomes + 1 noise swap + 20% Karma forced-7 |
| Setup | Snake draft (1→2→2→1); 2nd settlement yields starting resources |

See [`docs/1v1_rules.md`](docs/1v1_rules.md) for the authoritative table.

---

## Architecture

### Observation (dict, Charlesworth order)

Built by `src/catan_rl/policy/obs_encoder.py`; dims live in
`src/catan_rl/policy/obs_schema.py` (do not hardcode). The obs is **honest**:
the opponent's hidden dev-card *types* and hidden-VP count are **not** in the
obs — only the opponent's hidden-card *count* and *played* cards are. The
belief head predicts the hidden types as an auxiliary target.

| Key | Shape | Notes |
|-----|-------|-------|
| `tile_representations` | `(19, 79)` | per-tile features (resource/token/robber/ownership) |
| `current_player_main` | `(54,)` | agent scalars |
| `next_player_main` | `(61,)` | opponent scalars (54 + hidden-count one-hot + total-res) |
| `current_dev_counts` | `(5,)` | agent's held dev-card counts |
| `next_played_dev_counts` | `(5,)` | opponent's *played* dev cards (observable) |
| `hex_features` / `vertex_features` / `edge_features` | `(19,19)` / `(54,16)` / `(72,16)` | GNN node inputs |
| `opponent_kind` / `opponent_policy_id` | scalar `int64` | opp-id embedding inputs |

Trunk: TileEncoder + GraphEncoder + player/dev encoders + opp-id embedding →
concat → Linear → LayerNorm → GELU → **512-dim trunk** (`CatanPolicy`,
~1.5M params). Full I/O contract in [`docs/io_schema.md`](docs/io_schema.md).

### Action space (6 autoregressive heads)

`MultiDiscrete([13, 54, 72, 19, 5, 5])` = `[type, corner, edge, tile, res1, res2]`.

| Head | Size | Purpose |
|------|------|---------|
| 0 `type` | 13 | action type (see below) |
| 1 `corner` | 54 | settlement / city vertex |
| 2 `edge` | 72 | road edge |
| 3 `tile` | 19 | robber hex |
| 4 `resource1` | 5 | YoP / Monopoly / Trade-give / Discard |
| 5 `resource2` | 5 | YoP-2nd / Trade-receive |

Types: `0 BuildSettlement, 1 BuildCity, 2 BuildRoad, 3 EndTurn, 4 MoveRobber,
5 BuyDevCard, 6 PlayKnight, 7 PlayYoP, 8 PlayMonopoly, 9 PlayRoadBuilder,
10 BankTrade, 11 Discard, 12 RollDice`. There are **no P2P-trade actions**.
Each head is masked; the joint log-prob sums only the relevant heads
(per-type relevance buffers). Context-using heads (corner/res1/res2) use
FiLM/AdaLN conditioning. A value head and a 5-way belief head share the trunk.

### PPO hyperparameters (defaults — `src/catan_rl/ppo/arguments.py` is the source of truth)

| Param | Value |
|-------|-------|
| n_envs | 128 |
| n_steps | 256 (→ 32,768 transitions/rollout) |
| batch_size | 512 |
| n_epochs | 4 |
| γ (discount) | 0.995 |
| λ (GAE) | 0.95 |
| clip ε / clip_vf | 0.2 / 0.2 |
| target_kl (k3 estimator) | 0.02 |
| LR | 3e-4 → 1e-5 (linear decay) |
| entropy coef | 0.04 → 0.005 (annealed updates 50–200) |
| advantage norm | per-rollout (no value/return normalizer) |
| belief_coef / opp_action_coef | 0.05 / 0.03 |
| total_steps | 50,003,968 (~50M) |
| device | `auto` → MPS on M1 (eval pinned to CPU) |

### Self-play (league)

`src/catan_rl/selfplay/league.py` keeps a FIFO snapshot pool (maxlen 100); a
snapshot is added every 4 PPO updates. Opponent draws mix
`random_weight` / `heuristic_weight` / `snapshot_weight`. Defaults are
heuristic-only (`snapshot_weight=0`); setting `snapshot_weight>0` turns on
self-play and the snapshot-opponent driver seats frozen snapshots in-env.
A heuristic-weight floor (`require_heuristic_floor`) keeps the agent from
forgetting how to beat the heuristic during self-play.

---

## Project layout

```
src/catan_rl/
  engine/     pure-Python game (game, board, player, dice, broadcast, geometry)
  env/        Gymnasium env (catan_env.py), action masks, hand tracker
  policy/     CatanPolicy: encoders, obs_encoder, obs_schema, heads, network
  ppo/        trainer, buffer, gae, vec_env, game_manager, arguments (config SoT)
  selfplay/   league.py (snapshot pool), snapshot_opponent.py
  eval/       harness.py, wilson.py, rules_invariants.py
  bc/ setup_phase/ replay/ agents/ augmentation/ checkpoint/ cli/
  gui/        optional pygame (never import on training/eval paths)
src/catan_engine/   Rust engine crate (scaffolding; not the default backend)
scripts/    thin CLI shims (train.py, train_bc.py, generate_bc_dataset.py, …)
configs/    ppo_default.yaml, bc.yaml
docs/       architecture.md, io_schema.md, 1v1_rules.md, decisions/, plans/
```

---

## Getting started

### Install

```bash
pip install -e ".[dev]"        # editable + dev tools (pytest, ruff, mypy)
pip install -e ".[dev,gui]"    # add optional pygame GUI / replay viewer
```

### Train

```bash
make train                                  # defaults
catan-rl-train --config configs/ppo_default.yaml --run-name my_run
python scripts/train.py --dry-run           # validate config without training
```

`scripts/train.py` is a thin shim for the `catan-rl-train` console script
(`catan_rl.cli.train:main`). Common flags: `--config`, `--run-name`,
`--total-steps`, `--n-envs`, `--device`, `--dry-run`, `--max-updates`.

### Monitor / test

```bash
tensorboard --logdir runs/train/
make test          # pytest tests/unit
make lint          # ruff
make typecheck     # mypy
```

---

## Notes

- Training resolves `device: auto` → MPS on M1 (batched SGD is ~2.6–3.3× faster
  than CPU at batch 512); eval runs on CPU. Launch long runs detached (`nohup`).
- The training bottleneck is SGD (~80% of each update at n_envs=128), not the
  engine — the Rust backend barely changes wall-clock.
- `arguments.py` is the config source of truth; this README may lag.
