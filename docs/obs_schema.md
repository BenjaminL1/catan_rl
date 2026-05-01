# Observation Schema

Source: `src/catan_rl/env/catan_env.py`. All shapes/ranges below are checked by Phase 0's rules-invariant test.

## Resource ordering

There are **two** orders in this codebase. Use the right one.

| Constant | Order | Used by |
|---|---|---|
| `RESOURCES` | `BRICK, ORE, SHEEP, WHEAT, WOOD` | engine code (`catan_rl.engine.player`, etc.) |
| `RESOURCES_CW` | `WOOD, BRICK, WHEAT, ORE, SHEEP` | RL code (Charlesworth order; observation, hand tracker, action heads) |

Mixing the two silently produces wrong feature vectors. When in doubt, import explicitly:

```python
from catan_rl.env.catan_env import RESOURCES, RESOURCES_CW
```

## Observation dict

The env returns a dict with these keys:

| Key | Shape | Dtype | Range | Notes |
|---|---|---|---|---|
| `tile_representations` | `(19, 79)` | `float32` | `[0, 1]` | Per-tile features, see breakdown below |
| `current_player_main` | `(166,)` | `float32` | `[-1, 2]` | Acting player scalars |
| `next_player_main` | `(173,)` | `float32` | `[-1, 2]` | Opponent scalars (uses tracked hand) |
| `current_player_hidden_dev` | `(15,)` | `int32` | `[0, 5]` | Padded sequence; 0 = pad, 1-5 = card type |
| `current_player_played_dev` | `(15,)` | `int32` | `[0, 5]` | Same encoding |
| `next_player_played_dev` | `(15,)` | `int32` | `[0, 5]` | Same encoding |

Constants are also exported as `OBS_TILE_DIM`, `CURR_PLAYER_DIM`, `NEXT_PLAYER_DIM`, `MAX_DEV_SEQ`.

## Tile feature breakdown (79 dims)

| Dims | Field |
|---|---|
| 0-5 | resource one-hot (BRICK, ORE, SHEEP, WHEAT, WOOD, DESERT) |
| 6-16 | number-token one-hot (None, 2-6, 8-12; 11 slots) |
| 17 | has_robber (dynamic) |
| 18 | dot count / 5 |
| 19-54 | 6 vertices × 6 dims (none/self/other ownership + none/settle/city building type) |
| 55-78 | 6 edges × 4 dims (no-road / self-road / other-road / has-road) |

## Dev card IDs

```
0 = pad
1 = KNIGHT
2 = VP
3 = ROADBUILDER
4 = YEAROFPLENTY
5 = MONOPOLY
```

## Hand tracking

`next_player_main` is built using the **broadcast-based tracked hand** from `BroadcastHandTracker` rather than reading `opponent.resources` directly. In 1v1 with no P2P trading, every resource change is broadcast-observable, so the tracker is **deterministic-perfect** — no belief state required. This is documented in [ADR 0002](decisions/0002-perfect-hand-tracking.md).
