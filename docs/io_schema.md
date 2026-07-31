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
| `current_player_main` | `(67,)` | `float32` | agent scalars (`CURR_PLAYER_DIM=67` = 54 legacy base + 6 own-extras (hand-total, discard-pressure, own played YoP/Mono/RB, remaining-owed discard count `min(1, n/8)`) + 7 reserved strict-0.0 slots (`CURR_RESERVED_SLOTS`)). The 6th extra consumed one reserved slot, so the total is unchanged and no checkpoint forks; `RESERVED_PLAYER_SLOTS` stays 8 because it also feeds `next_player_main`. |
| `next_player_main` | `(69,)` | `float32` | opponent scalars (`NEXT_PLAYER_DIM=69` = 54 legacy base + 7 opp-extras (6-bin hidden-count one-hot + total-res scalar) + 8 reserved strict-0.0 slots) |
| `global_features` | `(14,)` | `float32` | Shared block outside the POV player pair (`GLOBAL_DIM=14` = 5 finite-bank-remaining + 5 public-reveal-derived dev-deck-remaining + 4 reserved). The bank subvector is seat-invariant; the dev-deck subvector is each seat's honest per-POV view of the unseen pool and **differs across seats by design** (the seat-swap pin asserts only the bank subvector). pointer-arch fork D3.3 |
| `is_setup` | `(1,)` | `float32` | snake-draft-setup flag, threaded to the corner pointer head's FiLM context (D2) |
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
>
> **Serve-time caveat.** `env/catan_env.py` IGNORES `action[3]` for
> `PlayKnight`: it sets `robber_placement_pending` and asks for a separate
> `MoveRobber` step, and `compute_action_masks` returns early in the robber
> branch (only `MoveRobber` in the `type` mask). So on a live `PlayKnight` step
> the `tile` mask is all-False. The BC recorder
> (`bc/dataset.py:patched_play_knight`) deliberately diverges: it splices the
> robber-branch `tile` mask onto the main-branch `type` mask and labels the row
> with the hex the teacher is about to rob, so the tile head gets a real
> gradient instead of a fabricated `tile_idx = 0`. That mask combination is
> emitted for no env state — a BC mask is therefore NOT reproducible from the
> stored obs on knight rows, and the joint log-prob composition for
> `PlayKnight` differs between BC and PPO rollouts. Accepted; see
> `docs/plans/v2/step3_bc.md` (this is what `RULESET_VERSION = 3` stamps).

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

`CatanEnv.get_action_masks()` returns these 12 bool tensors (`MASK_KEYS`):

| Key | Shape | Notes |
|---|---|---|
| `type` | `(13,)` | legal action types in the current phase |
| `corner_settlement` | `(54,)` | valid settlement vertices |
| `corner_city` | `(54,)` | own settlements upgradeable to cities |
| `edge` | `(72,)` | valid road edges |
| `tile` | `(19,)` | valid robber hexes (post Friendly-Robber filter) |
| `resource1_trade` | `(5,)` | resources you hold enough of to bank-trade |
| `resource1_discard` | `(5,)` | resources you hold `> 0` of |
| `resource1_default` | `(5,)` | Monopoly (any resource — Monopoly is bank-independent) |
| `resource1_yop` | `(5,)` | YoP-1st — a bank-supplyable first pick that leaves a legal partner |
| `resource2_yop` | `(5,)` | YoP-2nd, DIFFERENT from the first pick (`bank[r] >= 1`) |
| `resource2_yop_same` | `(5,)` | YoP-2nd, DOUBLING the first pick (`bank[r] >= 2`) |
| `resource2_trade` | `(5,)` | BankTrade-receive — gated on finite-bank supply (`bank[r] >= 1`) |

For `BankTrade`, the `resource2` mask additionally forbids `r2 == r1` (the
engine would otherwise accept a strictly-losing same-resource trade). For
`PLAY_YOP` the `resource2` mask swaps `resource2_yop_same` in at index `r1`,
since doubling a pick draws two of that resource from the bank.

Writers of offline corpora (BC, expert iteration) must check a candidate action
against these masks with `policy/obs_schema.action_masked_legal(masks, action)`
— the torch-free twin of the masking `CatanActionHeads` applies at sample time.
It checks every head the action type actually uses (including the
autoregressive head-5 rules above), not `masks["type"][a[0]]` alone; a
type-legal row whose corner / edge / tile / resource index is masked OFF is an
action the policy can never sample, i.e. exactly the train/serve skew BC exists
to avoid.

The single `resource1_default` / `resource2_default` keys were split so the
finite bank (spec 009) can gate each action independently. `PLAY_YOP` is legal
for a pair `(first, second)` iff `bank[first] >= 2` when `first == second`,
else `bank[first] >= 1` and `bank[second] >= 1`; `BankTrade` needs
`bank[r2] >= 1` with `r2 != r1`. The gate lives at the LEGALITY layer because
that is where the rule lives: the reference engine
(`Torevan/packages/engine/src/legal-moves.ts`) enumerates only fully-supplyable
`(give, receive)` pairs, so a mask that offers more has drifted from the
ruleset. When no legal `(give, receive)` pair remains, `type[BANK_TRADE]` is
withheld; when no legal YoP pair remains, `type[PLAY_YOP]` is withheld.

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
