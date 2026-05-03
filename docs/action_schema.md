# Action Schema

Source: `src/catan_rl/env/catan_env.py`, `src/catan_rl/models/action_heads_module.py`.

## Action space

`gym.spaces.MultiDiscrete([13, 54, 72, 19, 5, 5])` — 6 autoregressive heads:

| Idx | Name | Size | Used by action types |
|---|---|---|---|
| 0 | `type` | 13 | always |
| 1 | `corner` | 54 | type 0 (BuildSettlement), 1 (BuildCity) |
| 2 | `edge` | 72 | type 2 (BuildRoad) |
| 3 | `tile` | 19 | type 4 (MoveRobber) |
| 4 | `resource1` | 5 | type 7 (YoP), 8 (Monopoly), 10 (BankTrade-give), 11 (Discard) |
| 5 | `resource2` | 5 | type 7 (YoP-2nd), 10 (BankTrade-receive) |

## Action types (head 0)

| ID | Name | Notes |
|---|---|---|
| 0 | BuildSettlement | uses corner |
| 1 | BuildCity | uses corner |
| 2 | BuildRoad | uses edge |
| 3 | EndTurn | no sub-heads |
| 4 | MoveRobber | uses tile |
| 5 | BuyDevCard | no sub-heads |
| 6 | PlayKnight | no sub-heads (robber placement triggered next step) |
| 7 | PlayYoP | uses resource1 + resource2 |
| 8 | PlayMonopoly | uses resource1 |
| 9 | PlayRoadBuilder | no sub-heads (road-build phase triggered) |
| 10 | BankTrade | uses resource1 (give) + resource2 (receive) |
| 11 | Discard | uses resource1 |
| 12 | RollDice | no sub-heads |

**1v1 invariant:** there are NO propose/accept/counter trade actions. P2P trading is hard-disabled (see [ADR 0001](decisions/0001-1v1-rules-invariant.md)).

## Action masks

`env.get_action_masks()` returns a dict with these 9 keys:

| Key | Shape | Notes |
|---|---|---|
| `type` | `(13,)` bool | Which action types are legal in current phase |
| `corner_settlement` | `(54,)` bool | Valid settlement vertices |
| `corner_city` | `(54,)` bool | Valid city vertices (i.e. existing settlements you own) |
| `edge` | `(72,)` bool | Valid road edges |
| `tile` | `(19,)` bool | Valid robber hexes (after Friendly Robber filter) |
| `resource1_trade` | `(5,)` bool | Resources you have ≥ N of (depends on best port ratio) |
| `resource1_discard` | `(5,)` bool | Resources you have > 0 of |
| `resource1_default` | `(5,)` bool | Used by YoP/Monopoly (any resource) |
| `resource2_default` | `(5,)` bool | Used by YoP-2nd, BankTrade-receive |

## Autoregressive structure

All 6 heads are evaluated every step. Per the chosen action type, **only relevant heads contribute** to the joint log-probability and entropy via registered relevance buffers:

```
log_prob(action) = lp_type(action_type)
                 + lp_mask_corner[action_type]    * lp_corner(corner)
                 + lp_mask_edge[action_type]      * lp_edge(edge)
                 + lp_mask_tile[action_type]      * lp_tile(tile)
                 + lp_mask_resource1[action_type] * lp_res1(res1)
                 + lp_mask_resource2[action_type] * lp_res2(res2)
```

## Phase-specific additions affecting action heads

- **Phase 1.5 D6 symmetry augmentation (landed):** with probability
  `symmetry_aug_prob` per minibatch, `MultiActionHeads`'s training-time
  inputs are permuted by one of 11 non-identity D6 elements via tables
  in `catan_rl.augmentation.symmetry_tables`. Permutes the corner/edge/tile
  action axes plus the within-tile vertex/edge feature blocks.
- **Phase 2.4 AdaLN/FiLM heads (landed):** when `action_head_film=True`,
  the context-using heads (`corner`, `resource1`, `resource2`) replace
  concat-MLP conditioning with FiLM modulation. Layer norm is followed
  by `(1 + γ) ⊙ x + β` where `(γ, β)` come from a small generator MLP
  over the head's context. γ-init=0 so the modulation is identity at
  construction. `type/edge/tile` heads have no context and stay on the
  legacy path.
- **Phase 0 per-head entropy logging:** each head's unconditional and
  conditional (relevance-weighted) entropy is logged per update under
  `train/entropy_head_<name>` and `train/entropy_head_<name>_cond`. A
  `train/entropy_collapse_flag` fires when any head's unconditional
  entropy stays below `entropy_collapse_threshold` for
  `entropy_collapse_consecutive_updates` consecutive updates.
- **Phase 4.1 ISMCTS visit-count target:** when wired into the rollout
  loop (deferred follow-up), `ISMCTS.search` returns a visit-count
  distribution over the 13 action types. Convert via
  `visits_to_distribution(visits, temperature)` and use as a
  cross-entropy target on the **type head only** during PPO update.
  Sub-heads (corner/edge/tile/resource1/resource2) are not currently
  searched — the search tree is fixed at depth 1 over action types.
