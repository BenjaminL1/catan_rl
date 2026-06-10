# Model I/O Schema (observation + action space)

The contract between the env and `CatanPolicy`. Shape/name source of truth is
`src/catan_rl/policy/obs_schema.py`; the obs is built in
`src/catan_rl/policy/obs_encoder.py`. **Import the constants — don't hardcode
the literals.** The schema is compact-only: there is no thermometer mode and
no 166/173 variant.

## Resource ordering

There are **two** orders. Only `RESOURCES_CW` is exported from the RL stack.

| Constant | Order | Used by |
|---|---|---|
| engine `RESOURCES` | `BRICK, ORE, SHEEP, WHEAT, WOOD` | engine code (`catan_rl.engine.*`); *not* exported to the RL stack |
| `RESOURCES_CW` | `WOOD, BRICK, WHEAT, ORE, SHEEP` | obs, hand tracker, action heads (Charlesworth order) |

```python
from catan_rl.policy.obs_schema import RESOURCES_CW
```

`obs_encoder.py` owns the translation between the two; a unit test pins it.

---

## Observation dict

Every key is always present (no opt-in/legacy keys). Dims come from
`obs_schema.py`.

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `tile_representations` | `(19, 79)` | `float32` | per-tile features (`TILE_DIM=79`); breakdown below |
| `current_player_main` | `(54,)` | `float32` | agent scalars (`CURR_PLAYER_DIM=54`) |
| `next_player_main` | `(61,)` | `float32` | opponent scalars (`NEXT_PLAYER_DIM=61` = 54 + 6-bin hidden-count one-hot + total-res scalar) |
| `current_dev_counts` | `(5,)` | `float32` | agent's *held* dev counts over `DEV_CARD_ORDER` |
| `next_played_dev_counts` | `(5,)` | `float32` | opponent's *played* dev counts (observable) |
| `hex_features` | `(19, 19)` | `float32` | GNN per-hex node input |
| `vertex_features` | `(54, 16)` | `float32` | GNN per-vertex node input |
| `edge_features` | `(72, 16)` | `float32` | GNN per-edge node input |
| `opponent_kind` | scalar | `int64` | opp-id kind in `[0, N_OPP_KINDS=6)`; UNKNOWN=0 |
| `opponent_policy_id` | scalar | `int64` | league slot in `[0, N_OPP_POLICY_SLOTS=101)`; last index = unknown |

### Honesty (no opponent-secret leak)

The obs encodes the agent's own state in full but exposes only **observable**
opponent state. The opponent's hidden dev-card **types** and hidden **VP**
count are deliberately absent:

- The agent's hidden hand → `current_dev_counts` (it knows its own cards).
- Opponent's hidden dev-card *types* → **not in the obs**; they are the
  belief head's prediction target (soft CE, `belief_coef=0.05`).
- Opponent's hidden-card *count* → the 6-bin one-hot inside `next_player_main`.
- Opponent's *played* dev cards → `next_played_dev_counts`.
- Opponent VP → only `visibleVictoryPoints` enters `next_player_main`.

### `tile_representations` breakdown (79 dims)

| Dims | Field |
|---|---|
| 0–5 | resource one-hot (BRICK, ORE, SHEEP, WHEAT, WOOD, DESERT) |
| 6–16 | number-token one-hot (None, 2–6, 8–12; 11 slots) |
| 17 | `has_robber` (dynamic) |
| 18 | dot count / 5 |
| 19–54 | 6 vertices × 6 dims (none/self/other ownership + none/settle/city) |
| 55–78 | 6 edges × 4 dims (no-road / self-road / other-road / has-road) |

### Dev-card order

`DEV_CARD_ORDER = (KNIGHT, VP, ROADBUILDER, YEAROFPLENTY, MONOPOLY)`,
`N_DEV_TYPES = 5`. This is the order of the `(5,)` count vectors and the
belief head's logits.

### Hand tracking

Opponent resource counts in `next_player_main` come from
`BroadcastHandTracker`, which reconstructs the opponent's hand from the engine
event bus. In 1v1 with no P2P trading every resource change is broadcast, so
tracking is **deterministic-perfect** — no belief state needed. See
[ADR 0002](decisions/0002-perfect-hand-tracking.md). (Without a tracker the
encoder falls back to `opponent.resources` — valid only under these same 1v1
assumptions.)

---

## Action space

`gym.spaces.MultiDiscrete([13, 54, 72, 19, 5, 5])` — 6 autoregressive heads
(`HEAD_DIMS` in `obs_schema.py`, implemented in `policy/heads.py`).

| Idx | Head | Size | Used by action types |
|---|---|---|---|
| 0 | `type` | 13 | always |
| 1 | `corner` | 54 | 0 BuildSettlement, 1 BuildCity |
| 2 | `edge` | 72 | 2 BuildRoad |
| 3 | `tile` | 19 | 4 MoveRobber, 6 PlayKnight |
| 4 | `resource1` | 5 | 7 PlayYoP, 8 PlayMonopoly, 10 BankTrade (give), 11 Discard |
| 5 | `resource2` | 5 | 7 PlayYoP (2nd), 10 BankTrade (receive) |

> Note: `PlayKnight` (6) also drives the `tile` head — playing a knight moves
> the robber in the same step.

### Action types (head 0)

| ID | Name | Sub-heads |
|---|---|---|
| 0 | BuildSettlement | corner |
| 1 | BuildCity | corner |
| 2 | BuildRoad | edge |
| 3 | EndTurn | — |
| 4 | MoveRobber | tile |
| 5 | BuyDevCard | — |
| 6 | PlayKnight | tile |
| 7 | PlayYoP | resource1 + resource2 |
| 8 | PlayMonopoly | resource1 |
| 9 | PlayRoadBuilder | — |
| 10 | BankTrade | resource1 (give) + resource2 (receive) |
| 11 | Discard | resource1 |
| 12 | RollDice | — |

**1v1 invariant:** no propose/accept/counter trade actions. P2P trading is
hard-disabled; `BankTrade` is the only trade ([ADR 0001](decisions/0001-1v1-rules-invariant.md)).

### Action masks

`CatanEnv.get_action_masks()` returns these 9 bool tensors (`MASK_KEYS`):

| Key | Shape | Notes |
|---|---|---|
| `type` | `(13,)` | legal action types in the current phase |
| `corner_settlement` | `(54,)` | valid settlement vertices |
| `corner_city` | `(54,)` | own settlements upgradeable to cities |
| `edge` | `(72,)` | valid road edges |
| `tile` | `(19,)` | valid robber hexes (post Friendly-Robber filter) |
| `resource1_trade` | `(5,)` | resources you hold enough of to bank-trade |
| `resource1_discard` | `(5,)` | resources you hold `> 0` of |
| `resource1_default` | `(5,)` | YoP / Monopoly (any resource) |
| `resource2_default` | `(5,)` | YoP-2nd, BankTrade-receive |

For `BankTrade`, the `resource2` mask additionally forbids `r2 == r1` (the
engine would otherwise accept a strictly-losing same-resource trade).

### Autoregressive structure

All six heads are evaluated each step; per-head context (corner settle/city,
resource-action one-hot) is derived from the *sampled* upstream heads. Only
the heads relevant to the sampled action type contribute to the joint
log-prob and entropy, via the per-type relevance buffer in
`CatanActionHeads`:

```
log_prob(a) = lp_type(type)
            + relevance[type, corner]    * lp_corner(corner)
            + relevance[type, edge]      * lp_edge(edge)
            + relevance[type, tile]      * lp_tile(tile)
            + relevance[type, resource1] * lp_res1(res1)
            + relevance[type, resource2] * lp_res2(res2)
```

Context-using heads (`corner`, `resource1`, `resource2`) apply FiLM/AdaLN
conditioning: `(1 + γ) ⊙ LN(x) + β`, with `γ` init 0 (identity at start).
`type`, `edge`, `tile` are plain 2-layer MLPs.
