# Architecture Overview

One page on the v2 system: obs → trunk → heads, and the PPO training loop.
Source of truth is `src/catan_rl/`; this doc may lag.

## Modules

```
catan-rl-train  (catan_rl.cli.train:main; scripts/train.py is a shim)
  └─ TrainConfig.load(yaml)            # catan_rl/ppo/arguments.py (config SoT)
     └─ trainer / training_loop        # catan_rl/ppo/{trainer,training_loop}.py

catan_rl/
  ppo/        trainer, training_loop, buffer, gae, vec_env, game_manager,
              losses, schedules, arguments
  selfplay/   league.py (snapshot pool), snapshot_opponent.py
  env/        catan_env.py (dict obs, action masks, opponent dispatch), hand_tracker
  policy/     network.py (CatanPolicy), encoders.py, obs_encoder.py,
              obs_schema.py (shape SoT), heads.py
  engine/     game.py, board.py, player.py, dice.py, broadcast.py, geometry
  eval/       harness.py (symmetric-seat WR), wilson.py, rules_invariants.py
```

## Policy pipeline (`catan_rl/policy/`)

The env emits a dict obs (keys/shapes in `obs_schema.py`, built by
`obs_encoder.py`). `CatanPolicy._encode` runs:

1. **TileEncoder** over `tile_representations` `(19, 79)` → per-tile vectors,
   flattened.
2. **GraphEncoder** (GNN) over `hex_features` / `vertex_features` /
   `edge_features` — tripartite message passing → pooled vector.
3. **Player/dev encoders** over `current_player_main` `(67,)`,
   `next_player_main` `(69,)`, the `global_features` `(14,)` block, and the two
   `(5,)` dev-count vectors.
4. **Opp-id embedding** over `opponent_kind` + `opponent_policy_id`.

All parts are concatenated → `Linear → LayerNorm → GELU` → **512-d trunk**
(`CatanPolicy`, ~1.5M params). The trunk feeds:

- **CatanActionHeads** — six masked autoregressive heads (`heads.py`).
  `type/edge/tile` are plain MLPs; `corner/resource1/resource2` use FiLM/AdaLN
  context conditioning. Joint log-prob/entropy sum only the heads relevant to
  the sampled action type (per-type relevance buffer).
- **ValueHead** — `512 → 256 → 128 → 1`.
- **BeliefHead** — 5-way logits over the opponent's hidden dev-card types
  (`DEV_CARD_ORDER`), trained via soft cross-entropy against the engine's
  ground-truth distribution. This is the only model component that consumes
  hidden opponent state; the obs itself stays honest.

Full I/O contract: [`io_schema.md`](io_schema.md).

## Training loop (`catan_rl/ppo/`)

```
1. Reset the vec env (catan_rl/ppo/vec_env.py); each env is seated with an
   opponent per the league weights (heuristic by default; frozen snapshots
   when snapshot_weight > 0, driven by selfplay/snapshot_opponent.py).

2. Collect a rollout of n_envs * n_steps (= 128 * 256 = 32,768) transitions:
     - batched policy.sample() → action (6 heads) + value + belief + log_prob
     - env.step(); the opponent's full turn is driven inside the env
     - buffer.add(obs, action, reward, terminated, truncated, value, log_prob,
       masks)

3. Compute GAE (gae.py) with a terminated/truncated split: a real game-over
   zeros the bootstrap; a max_turns truncation keeps V(s_T) but resets the
   accumulator at the boundary.

4. Standardise advantages per rollout (advantage_norm="rollout"). There is
   NO value/return normalizer.

5. PPO update for n_epochs=4, early-stopping when the k3-KL estimate exceeds
   target_kl=0.02. Total loss = policy + value_coef*value
   - entropy_coef(t)*entropy + belief_coef*belief_CE
   + opp_action_coef*opp_action_CE (the opp-action term fires only on
   rollouts seated against a historical league policy).

6. Every league.add_snapshot_every_n_updates (=4) updates, push the current
   weights into the snapshot pool (FIFO, maxlen 100).

7. Every eval.eval_every_updates (=40) updates, run the eval harness
   (CPU-pinned, symmetric seats) vs the heuristic.
```

## Key entry points

| Concern | Entry point |
|---|---|
| Train | `catan-rl-train` → `catan_rl.cli.train:main` |
| Eval harness (symmetric-seat WR, Wilson CI, rule invariants) | `catan_rl.eval.harness` |
| Behavioral cloning | `catan-rl-bc-generate`, `catan-rl-bc-train` |
| Migrate a v2 checkpoint | `catan-rl-migrate-ckpt` |
| Record / replay a game | `catan-rl-record`, `catan-rl-replay` |
| Setup-phase labeling | `catan-rl-label-setup` |

`catan-rl-*` console scripts are wired by `[project.scripts]` in
`pyproject.toml`; the `scripts/*.py` files are thin shims.

## Honesty & 1v1 invariants baked into the obs/heads

- Obs models exactly **one** opponent; no opponent-set encoder.
- No P2P-trade actions; `BankTrade` is the only trade type (1v1 ruleset).
- The opponent's hidden dev-card *types* and hidden VP never enter the obs —
  the belief head predicts the types. Played dev cards and hidden-card *count*
  are observable and encoded.
- Self-play assumes a symmetric 2-player zero-sum game.
