# v2 Catan RL Agent — Technical Briefing

## 1. State Representation

The observation dict contains 10 keys per `src/catan_rl/bc/loader.py:94-104` (Phase 3 additions at lines 45-50 in `docs/obs_schema.md`):

| Key | Shape | Dtype | Encoding |
|---|---|---|---|
| `tile_representations` | (19, 79) | float32 | Charlesworth tile features: resource one-hot (6), number-token one-hot (11), robber flag, dot count, 6 vertices × 6 dims (ownership + building type), 6 edges × 4 dims (road status) |
| `current_player_main` | (54,) | float32 | Agent scalars (resources, VP, road length, knights, settlements, cities, dev counts, normalized via compact schema) |
| `next_player_main` | (61,) | float32 | Opponent scalars from `BroadcastHandTracker` perfect resource tracking + played dev cards |
| `current_dev_counts` | (5,) | float32 | Agent hidden dev-card count vector over KNIGHT/VP/ROADBUILDER/YOP/MONOPOLY |
| `next_played_dev_counts` | (5,) | float32 | Opponent played dev-card counts (observable; hidden hand inferred by belief head) |
| `hex_features` | (19, 19) | float32 | GNN node features for hexes per `src/catan_rl/policy/encoders.py:150-160` (graph encoder input) |
| `vertex_features` | (54, 16) | float32 | GNN node features for vertices |
| `edge_features` | (72, 16) | float32 | GNN node features for edges |
| `opponent_kind` | scalar | int64 | Phase 3.6: discrete value in [0,5] = (UNKNOWN, RANDOM, HEURISTIC, SELF_LATEST, LEAGUE, EXPLOITER); 40% masking per `CLAUDE.md` line 80 |
| `opponent_policy_id` | scalar | int64 | Phase 3.6: league slot [0, 100] or unknown sentinel |

**Hex grid encoding**: Tiles use a fixed 19-element hex lattice indexed canonically. Axial coordinates (q, r) are embedded in the `TileEncoder` via learned 2D positional embeddings (`_AxialPositionalEmbedding`, `src/catan_rl/policy/encoders.py:36-74`). The encoder applies a 2-layer transformer (pre-norm, GeLU) over the 19 tiles with concatenated axial features (24 dims total, split 12/12 across q/r embeddings at init std=0.02).

**GNN**: The tripartite `GraphEncoder` (`src/catan_rl/policy/encoders.py:134-180` approx) runs 2 rounds of mean-pool message passing over hex/vertex/edge nodes with precomputed adjacency tables. Input dims are 19 (hex), 16 (vertex), 16 (edge); output is a single 64-d pooled vector per game.

## 2. Action Space

6 autoregressive heads: `MultiDiscrete([13, 54, 72, 19, 5, 5])` per `docs/action_schema.md:7`.

| Head | Size | Used by action types | Notes |
|---|---|---|---|
| 0. `type` | 13 | all | Action type ID (see below) |
| 1. `corner` | 54 | 0, 1 | Settlement (0) or city (1) vertex placement |
| 2. `edge` | 72 | 2 | Road edge placement |
| 3. `tile` | 19 | 4 | Robber hex destination |
| 4. `resource1` | 5 | 7, 8, 10, 11 | YoP/Monopoly/BankTrade-give/Discard resource |
| 5. `resource2` | 5 | 7, 10 | YoP-2nd / BankTrade-receive resource |

**Action types 0–12** (per `docs/action_schema.md:18-35` and `src/catan_rl/policy/obs_schema.py:78-93`):
- 0: `BUILD_SETTLEMENT` (corner)
- 1: `BUILD_CITY` (corner)
- 2: `BUILD_ROAD` (edge)
- 3: `END_TURN` (no sub-heads)
- 4: `MOVE_ROBBER` (tile)
- 5: `BUY_DEV_CARD` (no sub-heads)
- 6: `PLAY_KNIGHT` (no sub-heads)
- 7: `PLAY_YOP` (resource1 + resource2)
- 8: `PLAY_MONOPOLY` (resource1)
- 9: `PLAY_ROAD_BUILDER` (no sub-heads)
- 10: `BANK_TRADE` (resource1 give + resource2 receive)
- 11: `DISCARD` (resource1)
- 12: `ROLL_DICE` (no sub-heads)

**Total flat size**: 13 + 54 + 72 + 19 + 5 + 5 = **168** discrete choices. Autoregressive composition per action type via relevance masking in `src/catan_rl/policy/heads.py:129-180` (registered buffer `head_relevance` tracks which heads matter).

## 3. Action Masking

The 9-key mask dict is computed by `src/catan_rl/env/masks.py:55-101` and applied at inference and training. Per `docs/action_schema.md:40-52`:

| Mask key | Shape | Notes |
|---|---|---|
| `type` | (13,) | Phase-dependent action types (e.g., no roll if not pending, no move-robber if setup) |
| `corner_settlement` | (54,) | Vertices where the agent can build settlements (no adjacent settlements, not the robber's starting spot) |
| `corner_city` | (54,) | Agent's existing settlements eligible for upgrade |
| `edge` | (72,) | Free edges not adjacent to other roads of the same player |
| `tile` | (19,) | Robber placement post-7, excluding friendly-robber restriction (Friendly Robber: cannot place adjacent to player with < 3 visible VP per `CLAUDE.md` line 26) |
| `resource1_trade` | (5,) | Resources the agent has ≥ N of (where N depends on best port ratio) |
| `resource1_discard` | (5,) | Resources the agent has > 0 of |
| `resource1_default` | (5,) | Any resource (YoP/Monopoly context) |
| `resource2_default` | (5,) | Any resource (YoP-2nd / BankTrade-receive context) |

Masked categories use hard masking via `masked_log_softmax` in `src/catan_rl/policy/heads.py:51-61`: illegal slots are set to -∞ before softmax, yielding zero probability. Rows with no valid entries return uniform log-prob to avoid NaN.

## 4. Hidden Information Handling

The opponent's hidden dev-card *type* distribution is modeled via the **belief head**, a 5-way softmax over `{KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY}` per `src/catan_rl/policy/heads.py:407-426`.

**`BroadcastHandTracker`** (`src/catan_rl/env/hand_tracker.py:53-100`) maintains *perfect* opponent resource counts by subscribing to the engine's `RESOURCE_CHANGE` broadcast events. In 1v1 with no player-to-player trading, every resource mutation is observable (build, buy, trade, discard, rob, dev-card effects, dice yield). This is rule-correct per `CLAUDE.md` lines 28-29 and `docs/decisions/0002-perfect-hand-tracking.md` (referenced in `docs/obs_schema.md:93`). The tracker maintains hands in Charlesworth order and feeds directly into `next_player_main` construction.

The belief head is trained with soft cross-entropy on the env's ground-truth normalized opponent dev-card type count vector (`obs['belief_target']`, shape (5,)). Loss weight = 0.05 in both BC (Step 3, `docs/plans/v2_step3_bc.md:102-174`) and PPO (Step 4, `docs/plans/v2_step4_ppo.md:45-70`). In Step 5 MCTS, the belief head is used as a sampler for belief-state determinization per `docs/plans/v2_step5_mcts.md:145-160`.

## 5. Learning Algorithm

**Step 3 (BC warm-start)** per `docs/plans/v2_step3_bc.md`:
- Generate 30k games of heuristic-vs-heuristic + 30% epsilon-variant rollouts
- D6 augmentation (prob = 0.5 pending preflight E0.3/E0.4)
- Loss: `L_policy (per-head CE) + 0.1 * L_value (MSE on discounted terminal z) + 0.05 * L_belief (soft CE)`
- AdamW, LR=3e-4, batch=1024, constant schedule after 500-step warmup, early-stop on val NLL (patience=3)

**Step 4 (PPO + piKL anchor)** per `docs/plans/v2_step4_ppo.md`:
- Load BC checkpoint as frozen anchor
- PPO: `loss = policy_loss + value_loss + kl_penalty + belief_loss`
- Value clipping (clip_range_vf=0.2), GAE λ=0.95, γ=0.998
- LR: 3e-4 → 1e-5 linear decay over total budget
- Entropy coef: 0.04 → 0.005 from updates 200–1500
- **piKL anchor**: `λ_initial=0.2`, linear decay to 0 over 2M steps per `docs/plans/v2_step4_ppo.md:71-81`
- League: maxlen=100, FIFO eviction, PFSP-hard `(1-w)^2` with 32-game sliding window
- Curriculum: 60% heuristic / 25% league / 10% self-latest / 5% random (first 2M steps); then 25% heuristic / 50% league / 20% self-latest / 5% random
- Duo exploiter cycles every 1M main steps (32 updates per cycle)
- TrueSkill ratings with σ-decay=1.001
- Opp-id embedding 8-dim with 40% mask prob

**Step 5 (MCTS — planned, not built)** per `docs/plans/v2_step5_mcts.md`:
- PUCT search: `argmax_a [Q(s,a) + 1.5 * P(s,a) * √(Σ_b N(s,b)) / (1 + N(s,a))]`
- Policy prior from policy head (type only; sub-heads at expansion)
- Value bootstraps from value head at leaf (non-terminal)
- 6 chance-node buckets (low/low-mid/six/seven/mid-high/high) weighted by remaining StackedDice bag distribution
- Belief-determinization: sample 2 opponent dev-card realisations from belief head per game root
- Dirichlet noise at root (α=0.3, ε=0.25, training only)
- AlphaZero-style visit-count target training on policy type head

## 6. Policy + Value Architecture

`CatanPolicy` (`src/catan_rl/policy/network.py:61-128`) assembles:

1. **TileEncoder** (transformer + axial positional embedding): (B, 19, 79) → (B, 19, 25) via 2-layer transformer (d_model=96, 4 heads, FFN=192, GeLU pre-norm, dropout=0.05) with learned axial embeddings (24 dims, init std=0.02)
2. **GraphEncoder** (tripartite GNN): 19 hex + 54 vertex + 72 edge nodes, 2 message-passing rounds, outputs 64-d pooled vector
3. **CountDevEncoder** (bincount): converts dev-card counts to 16-d vectors (×2 for agent + opponent)
4. **PlayerEncoder** (2-layer MLP): (54,) / (61,) → learned vectors
5. **OppIdEmbedding** (Phase 3.6): kind + policy-id embeddings, 8-d total

**Fusion** (lines 117–121): concatenate all encoder outputs → linear(fused_dim, 512) → LayerNorm → GeLU → 512-d trunk

**Action heads** (`CatanActionHeads`, `src/catan_rl/policy/heads.py:129-200` approx):
- `type`, `edge`, `tile`: 2-layer SimpleHeads (no context)
- `corner`, `resource1`, `resource2`: FiLM heads with learned modulation `(1 + γ) ⊙ LN(x) + β` (γ-init=0 → identity at construction)

**Value head** (`ValueHead`): 2-layer MLP on trunk → scalar

**Belief head** (`BeliefHead`): 2-layer MLP on trunk → 5-way logits (init gain=0.01 for near-uniform start)

**Total params**: ~1.4M (Step 3), scales to ~2.24M with Phase 3 opp-id embedding per `CLAUDE.md` line 82.

## 7. Auxiliary Heads

**Belief head** (5-way over opponent dev-card type):
- Weight: 0.05 in BC and PPO
- Target: normalized count vector from env (`obs['belief_target']`)
- Training-only (never enters policy input)
- Sampled for determinization in MCTS (Step 5)
- KL to env-GT must be ≤ 0.35 for Step 5 gate (preflight 0.4, `docs/plans/v2_step5_mcts.md:75-89`)

**Opponent-action head** (13-way over action type):
- Trained in Step 3 BC on heuristic action distribution
- Loss weight: 0.03
- **Disabled in Step 4 PPO** (weight 0.0) because supervision shifts under self-play; parameters frozen at BC value per `docs/plans/v2_step4_ppo.md:69`
- Trainer checks `obs['opp_action_target_valid']` to exclude random/heuristic/current_self rows

## 8. Training Regime

**D6 symmetry augmentation**: applies to both obs and action via precomputed tables in `src/catan_rl/augmentation/symmetry_tables.py`. Probability set post-preflight E0.3/E0.4 (default 0.5, may increase to 1.0 if determinism audit < 5% and equivariance probe passes). Applied in dataset loader `__getitem__` so on-disk shards stay canonical.

**Self-play League**:
- Maxlen=100, FIFO eviction (Nash pruning too slow on CPU)
- PFSP-hard `(1-w)^p` with p=2.0 and 32-game sliding window
- Curriculum warmup (60→25% heuristic), self-latest regularization, duo exploiter cycles, TrueSkill ratings, opp-id embedding

**Opponent mix** (post-2M steps): 25% heuristic, 50% league, 20% self-latest, 5% random

**TrueSkill**: continuous rating updates per match; μ/σ stored in TensorBoard under `eval/trueskill_{main,top1..5}_{mu,sigma,conservative}`

**Opponent-id embedding**: 8-d total (4-d kind embedding + 4-d policy-id embedding); 40% masking at inference to keep policy robust to hidden-opponent scenarios

## 9. Evaluation Methodology

Per `docs/plans/v2_step4_ppo.md:116-130` and `v2_step5_mcts.md:26-37`:

- **Heuristic bench**: 100 games every 100k steps, symmetrised WR (P1 / P2 / sym) with N≥3 seeds
- **Champion bench**: frozen **v2** checkpoints from the run's own lineage (bootstrap + earlier self-play snapshots), 200 games at milestones. No v1 checkpoints.
- **AlphaBeta bench**: d=2 every 500k steps, d=4 every 5M steps
- **PPO-BR-gap** (exploitability proxy): 1M-step and 5M-step best-response adversaries; sensitivity probe Δ_BR = b_5M − b_1M as suboptimality lower bound
- **TrueSkill within league**: continuous; Nash-weighted checkpoint pruning (planned Phase 3.5 but CPU-prohibitive, using FIFO instead)
- **Drift probes**: `bc/v_drift_l1` (post-BC value head redirection under PPO, every 100k steps) and `belief/v_drift_kl` (belief KL to env-GT as Step-5 gate predictor)

## 10. 1v1-Specific Rule Choices

Per `CLAUDE.md` lines 7–21 and `docs/1v1_rules.md`, hardcoded invariants:

| Rule | 1v1 value | Code location |
|---|---|---|
| Win condition | 15 VP | `src/catan_rl/engine/game.py`: `catanGame.maxPoints = 15` |
| Player count | 2 | `catanGame.numPlayers = 2` |
| P2P trading | **DISABLED** — bank only | `src/catan_rl/engine/player.py`: `initiate_trade` early-returns on non-`'BANK'` |
| Discard threshold | 9 cards | `player.discardResources`: `maxCards = 9` |
| Friendly Robber | cannot place adjacent to player with < 3 visible VP | `catanBoard.get_robber_spots` filtering |
| Dice mechanic | StackedDice (36-bag + 1 non-7 swap + **persistent 20% Karma forced-7 buff** — `last_player_to_roll_7` updated only on a 7, never reset by turn change; see `docs/1v1_rules.md` "Karma mechanic note") | `src/catan_rl/engine/dice.py` |
| Setup | snake draft (1→2→2→1); 2nd settlement yields resources | `catanGame.build_initial_settlements` + env grant |
| Largest Army / Longest Road | 3 / 5 threshold | `catanGame.check_largest_army` / `check_longest_road` |

**No P2P trade implications**: action space has 13 types (BankTrade only), single-opponent obs, perfect hand tracking, zero-sum 2-player symmetry, reward sign-flip for player-swap augmentation.

---

**Document generated**: 2026-05-13. Sources: CLAUDE.md conventions + docs/plans/ (v2_step3_bc.md, v2_step4_ppo.md, v2_step5_mcts.md) + source files under src/catan_rl/policy/, src/catan_rl/env/, src/catan_rl/bc/.
