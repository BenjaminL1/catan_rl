# Briefing — Catanatron

Source: https://github.com/bcollazo/catanatron (master, fetched 2026-05-14) + https://docs.catanatron.com + the Collazo postmortem (separate doc).

## What it is

A high-performance Settlers of Catan simulator + AI bot collection in Python. The repo's **strongest bot is `AlphaBetaPlayer` at depth 2** (`catanatron/catanatron/players/minimax.py:13`). All ML/RL attempts in the project sit in `catanatron_experimental/catanatron_experimental/machine_learning/` and *underperform* the handcrafted alpha-beta player. Catanatron is the de-facto open-source baseline for Catan AI.

## Bot players (relative strength)

From the Medium postmortem + leaderboard configs:

| Player | Method | Strength |
|---|---|---|
| `AlphaBetaPlayer(n=2)` | Expectimax alpha-beta + handcrafted value fn, depth 2 | **Top** (used on leaderboard) |
| `AlphaBetaPlayer(n=3)` | Same, depth 3 | **Weaker than depth 2** — author's "depth-3 paradox" |
| `ValueFunctionPlayer` | Greedy 1-step over handcrafted value fn | Strong; backbone of AlphaBeta's leaf |
| `MCTSPlayer` | Standard MCTS, 25-100 sims | Better than WeightedRandom, worse than ValueFunction |
| `WeightedRandomPlayer` | Weighted-random policy | Beats `RandomPlayer` 61% |
| `RandomPlayer` | Uniform random | Floor |

## Value function (the load-bearing piece)

`catanatron/catanatron/players/value.py:10-24` — `DEFAULT_WEIGHTS`:

```python
DEFAULT_WEIGHTS = {
    "public_vps": 3e14,          # dominates — everything else is tiebreaker
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,       # applied when hand > 7 (4-player rule)
    "hand_devs": 10,
    "army_size": 10.1,
}
```

`AlphaBetaPlayer.decide()` (`minimax.py:63-75`) → `alphabeta()` (`minimax.py:85-168`) recurses with `MAX_SEARCH_TIME_SECS = 20` (`minimax.py:14`) and uses `list_prunned_actions()` when `prunning=True` (`minimax.py:44-46`).

## Chance nodes (dice)

`minimax.py:111-126` — `expand_spectrum(game, actions)` returns `action → [(game, proba)]`. The expectimax accumulates `expected_value += proba * value` over all dice outcomes — full 11-way 2d6 expansion, not bucketed.

## Action space

- **289 discrete actions** in the post-postmortem refactor (initially 5000+); per Medium postmortem.
- 2-player BASE map dynamic size **[0, 327]** per `catanatron_env.py:238` docstring.
- Action masking: `action_masks()` (`catanatron_env.py:106-112`) returns a bool list aligned to `action_space_size`, SB3-compatible: `[action_int in valid for action_int in range(self.action_space_size)]`.

## Gym environment

`catanatron/catanatron/gym/envs/catanatron_env.py`:
- **Default reward** (`catanatron_env.py:24-30`): `simple_reward()` → `+1 win / -1 loss / 0 ongoing`. **Sparse, terminal-only.**
- **Configurable**: `self.reward_function = self.config.get("reward_function", simple_reward)` (`catanatron_env.py:49`). The Medium postmortem mentions `VP_DISCOUNTED_RETURN = 0.9999^turns × victory_points` was tried in supervised-learning experiments but the default in-repo is binary terminal.
- **Vector obs**: 194·N + 226 features (`catanatron_env.py:331`); for 2 players = **614-dim flat vector**.
- **Mixed obs (board tensor)**: `(channels, 21, 11)` — a 2D-CNN-friendly hex-grid representation (`catanatron_env.py:62-66`).
- **VP target**: `self.vps_to_win = self.config.get("vps_to_win", 10)` (`catanatron_env.py:53`) — **defaults to 10 (4-player rules)**.

## What's NOT in the repo

- No deep search beyond depth 3 (compute infeasible at expectimax cost).
- No imperfect-information handling; the value function reads the *true* opp resources (3-4 players, no hidden hand modelling). For a 4-player game with P2P trade, this is approximate.
- No belief modelling for opp dev cards.
- No published superhuman result — the value-function weights are hand-tuned.

## Citable failure modes (from Medium postmortem)

1. **5000+ → 289 action-space compression** was necessary; the original space had so many no-op-like actions that learning collapsed.
2. **END_TURN dominance**: ~30% of game states give the bot no productive action; classifier collapses to "always END_TURN".
3. **MCTS @ 100 sims/turn**: ~3 seconds/decision; worse than ValueFunctionPlayer in raw strength.
4. **TensorForce DQN**: plateaued at 80% WR vs RandomPlayer after 8 hours — never broke RL ceiling.
5. **Cross-Entropy Method + supervised w/ VP_DISCOUNTED_RETURN**: still lost to RandomPlayer.
6. **Author's conclusion**: handcrafted value + alpha-beta is the SOTA for Catan in this codebase, mirroring chess (Stockfish) rather than Go (AlphaZero).

## What this means for our analysis

- **Catanatron is a 3-4-player benchmark with a 10-VP win condition.** Direct WR comparison to v2 is invalid (1v1, 15-VP, no P2P trade).
- **The depth-3 regression is a real, reproduced result** — the author claims it's not a bug but a property of the value function being miscalibrated at deeper horizons (myopic-tactical bias).
- **Catanatron's value function is a target to beat** as a 1-step greedy heuristic baseline. Our v2 heuristic should at minimum match it on 1v1 to be considered competitive.
- **The 289-action flat space + masked-categorical** is one design point our 6-head autoregressive [13, 54, 72, 19, 5, 5] space replaces.
