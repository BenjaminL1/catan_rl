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

The env returns a dict with these keys. Player-feature dims depend on the
`use_thermometer_encoding` flag (Phase 1.3): legacy thermometer mode keeps
the original 166/173 dims; compact mode drops them to 54/61 by replacing the
eight `bucket8` thermometer encodings with single normalized scalars (same
information content, smaller obs).

| Key | Shape (legacy / compact) | Dtype | Range | Notes |
|---|---|---|---|---|
| `tile_representations` | `(19, 79)` | `float32` | `[0, 1]` | Per-tile features, see breakdown below |
| `current_player_main` | `(166,)` / `(54,)` | `float32` | `[-1, 2]` | Acting player scalars |
| `next_player_main` | `(173,)` / `(61,)` | `float32` | `[-1, 2]` | Opponent scalars (uses tracked hand) |
| `current_player_hidden_dev` | `(15,)` | `int32` | `[0, 5]` | Padded sequence; 0 = pad, 1-5 = card type |
| `current_player_played_dev` | `(15,)` | `int32` | `[0, 5]` | Same encoding |
| `next_player_played_dev` | `(15,)` | `int32` | `[0, 5]` | Same encoding |

Constants live in `catan_rl.models.utils` and are the single source of truth:
`N_TILES`, `OBS_TILE_DIM`, `CURR_PLAYER_DIM` / `CURR_PLAYER_DIM_COMPACT`,
`NEXT_PLAYER_DIM` / `NEXT_PLAYER_DIM_COMPACT`, `MAX_DEV_SEQ`, `DEV_CARD_VOCAB`.
Helper functions `curr_player_dim(use_thermometer_encoding)` and
`next_player_dim(use_thermometer_encoding)` return the right dim for a given
flag. Import these rather than hardcoding the literals.

### Phase 1.3 thermometer-vs-compact

Set `use_thermometer_encoding=True` (default) on `CatanEnv` to keep the
legacy 166/173 schema, which is what `checkpoint_07390040.pt` was trained on.
Set it to `False` (or use `configs/phase1_full.yaml`) to switch to the
compact 54/61 schema. The eight legacy `bucket8` sites — 5 resource counts,
VP, road length, knights played, settlements left, cities left, 5 dev-card
counts, and deck remaining — collapse from 113 dims of 8-threshold
thermometers down to 16 dims of single normalized scalars.

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
