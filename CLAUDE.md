# Catan RL — Project Conventions

## Project goal — **1v1 ONLY, NOT 4-PLAYER**

This project solves **1v1 Settlers of Catan** under the Colonist.io ruleset. It is **not** a 4-player Catan agent and should never be generalized to 4-player without explicit user instruction. Many of the choices below only make sense in the 1v1 context — assuming 4-player rules will silently break correctness.

**The 1v1 Colonist.io ruleset (must be preserved across all changes):**

| Rule | 1v1 value | Standard 4p value |
|---|---|---|
| Win condition | **15 VP** | 10 VP |
| Player count | **2** | 3–4 |
| **Player-to-player trading** | **DISABLED** — bank/port only | enabled |
| Discard threshold on 7-roll | **9 cards** | 7 cards |
| **Friendly Robber** | cannot place robber on a hex adjacent to a player with `< 3` visible VP | none |
| Dice | **`StackedDice`** — shuffled bag of 36 outcomes + 1 noise swap + 20% Karma forced-7 if opponent rolled the previous 7 | independent 2d6 each roll |
| Setup | snake draft (1 → 2 → 2 → 1); 2nd settlement yields starting resources | same draft, same starting-resource rule |
| Board / resources / ports | standard 19 tiles, 54 vertices, 72 edges; standard resource and number-token counts; 5×2:1 + 4×3:1 ports | identical |
| Largest Army threshold | 3 knights | 3 knights |
| Longest Road threshold | 5 roads | 5 roads |

**Implications baked into the code (do not undo any of these without flagging):**
- `catanGame.maxPoints = 15` and `catanGame.numPlayers = 2` are hardcoded in `catan/engine/game.py`.
- `player.discardResources` uses `maxCards = 9`, not 7 (`catan/engine/player.py`, also `RandomAIPlayer.discardResources`, also `heuristicAIPlayer.discardResources`).
- `player.initiate_trade` early-returns on any non-`'BANK'` mode — **player-to-player trading is hard-disabled** (`catan/engine/player.py`).
- `catanBoard.get_robber_spots` filters out hexes whose corners include any player with `victoryPoints - devCards['VP'] < 3` (Friendly Robber).
- `StackedDice` (`catan/engine/dice.py`) replaces independent 2d6: pre-shuffled bag of 36 sums + one random non-7 swap + 20% Karma forced-7 mechanic.
- `BroadcastHandTracker` does **perfect** opponent hand tracking — this is only valid in 1v1 with no P2P trading; every resource change is observable. In 4-player or with P2P trading, hand tracking becomes a belief-state problem and this assumption breaks.
- The action space has **no propose-trade / accept-trade / counter-trade actions**. `BankTrade` is the only trade action type. Adding P2P trading would require expanding the 13-type action space and the corresponding masks/heads.
- The observation models exactly **one opponent** (`next_player_main`, `next_player_played_dev`). There is no opponent-list / opponent-set encoder.
- Self-play league assumes a symmetric 2-player zero-sum game (PFSP, Nash pruning, exploitability — all defined for 2-player only).

Any PR that touches game-rule constants, the action space, the obs schema, or the trading API must explicitly state how it preserves the 1v1 ruleset, or the change should be rejected.

## Goal target
Train a superhuman 1v1 Catan agent using custom PPO + League self-play. Hardware target: Apple M1 Pro CPU-only baseline. Latest checkpoint `checkpoints/train/checkpoint_07390040.pt` reached ~7.4M steps.

## Layout
- `catan/engine/` — pure-Python game engine (board, player, dice, broadcast event bus)
- `catan/rl/env.py` — Gymnasium wrapper, dict obs, action masking, deferred opponent NN inference
- `catan/rl/models/` — Charlesworth-style net: TileEncoder transformer + dev-card MHA + player MLPs → 512-dim → 6 autoregressive action heads + value
- `catan/rl/ppo/` — PPO trainer, rollout buffer, GAE, league, evaluation manager
- `catan/rl/setup/` — separate setup-phase trainer (Monte Carlo heuristic rollouts)
- `scripts/` — entry points (`train.py`, `evaluate.py`, `play_vs_model.py`, `train_setup.py`)
- `docs/plans/` — roadmaps and design notes

## Action space (6 autoregressive heads)
`MultiDiscrete([13, 54, 72, 19, 5, 5])` = `[type, corner, edge, tile, res1, res2]`.
Action types: `0 BuildSettlement, 1 BuildCity, 2 BuildRoad, 3 EndTurn, 4 MoveRobber, 5 BuyDevCard, 6 PlayKnight, 7 PlayYoP, 8 PlayMonopoly, 9 PlayRoadBuilder, 10 BankTrade, 11 Discard, 12 RollDice`.

## Observation
Dict with keys `tile_representations (19,79)`, `current_player_main (166,)`, `next_player_main (173,)`, plus three padded dev-card sequences. Resource order is **Charlesworth** (`WOOD, BRICK, WHEAT, ORE, SHEEP`), not the engine's `RESOURCES` order — do not confuse them.

## Rules to follow

1. **Never edit the engine to change game rules** without flagging it. The engine matches Colonist.io 1v1 and any drift breaks evaluation comparability.
2. **Keep checkpoint backward-compatibility.** `CatanPPO.load()` must read `checkpoint_07390040.pt`. If a change breaks the policy state-dict shape, ship a one-shot migration in `scripts/migrate_checkpoint.py` and document the new lineage.
3. **TensorBoard layout is additive.** Existing scalars under `runs/train/` must keep their names; new diagnostics are new scalars, never renames.
4. **No new docs unless asked.** Update `README.md` and `MEMORY.md` when conventions change; do not create explanatory `.md` files speculatively.
5. **Two resource orderings exist** — `RESOURCES` (engine) vs `RESOURCES_CW` (RL). Always import the right one.
6. **PPO config drift is real.** `arguments.py` is source-of-truth; `README.md` and `MEMORY.md` may be stale. Verify against `arguments.py` before quoting hyperparameters.
7. **Default to CPU.** MPS is documented as 9× slower at batch=1 inference; CUDA path is opt-in only.
8. **Don't import from `catan/gui/`** in any RL path — pygame is optional and breaks headless training.
9. **Symmetry-aware code paths.** When touching obs/action heads, remember the board has D6 symmetry; precomputed tables live in `catan/rl/ppo/symmetry_tables.py` (when added).

## Testing & smoke
- Quick env sanity: `python scripts/play_vs_model.py <ckpt> --smoke-test`
- Eval against heuristic: `python scripts/evaluate.py <ckpt> --opponent heuristic --n-games 100`
- TensorBoard: `tensorboard --logdir runs/train/`

## Active roadmap
See `docs/plans/superhuman_roadmap.md` (Phase 0 = correctness/eval-harness, Phase 1 = sample efficiency, Phase 2 = architecture, Phase 3 = self-play diversity, Phase 4 = optional MCTS/GRU/belief).

## Phase 3 (landed)
- **3.1 PFSP-hard**: `pfsp_mode='hard'` returns priorities `(1-w)^pfsp_p` (AlphaStar default), biased toward currently-losing opponents. Per-opponent WR uses a sliding `pfsp_window` of recent outcomes (default 32). Modes: `'linear'` (legacy), `'hard'`, `'var'`. Asserts 1v1 zero-sum at construction.
- **3.2 Latest-policy regularization**: `latest_policy_weight=0.10` (default 0.0) makes the league emit a special `('current_self', None, -2)` opponent with that probability; trainer wires a `current_policy_state_fn` callback so the env opponent loads a fresh snapshot of `self.policy` every reset. Damps drift between league updates.
- **3.3 Duo exploiter cycles**: `exploiter_mode='duo'` periodically pauses main and runs an interleaved exploiter cycle. Cycle structure: snapshot main → load main weights into the trainer's policy + fresh AdamW → swap `self.league` for a `League.frozen_main_for_exploiter(main_state_dict)` (single-opponent league, no random/heuristic) → run `exploiter_n_updates` PPO updates with TB scalars under `exploiter/<key>` → restore main from snapshot → `League.add_with_boost(exploiter_state, multiplier=exploiter_priority_multiplier, boost_games=exploiter_priority_games)`. Cycle scheduling: every `exploiter_cycle_steps` of *main* progress (not exploiter steps); exploiter rollouts don't move main's `global_step`. The priority boost lives in `League._priority_boost: dict[id → (multiplier, games_remaining)]`, applied as a multiplicative scalar on PFSP priorities and decremented in `update_result`. Tests live in `tests/unit/selfplay/test_exploiter_primitives.py` and `tests/unit/algorithms/test_exploiter_snapshot.py`.
- **3.4 TrueSkill league rating**: `use_trueskill=True` builds a `RatingTable`. Trainer hooks `_record_rating_match` into `GameManager` so every concluded match updates ratings (skips opponent_id < 0 — random/heuristic/current_self never enter the table). `trueskill_decay=1.001` per update inflates σ to handle non-stationarity. Logs `eval/trueskill_main_{mu,sigma,conservative}` and `eval/trueskill_top{1..5}_conservative`.
- **3.5 Nash-weighted checkpoint pruning**: `prune_strategy='nash'` replaces FIFO eviction at capacity. Every `prune_every=20` league adds, runs an internal h2h round-robin among the most recent `nash_top_k=32` entries (`nash_prune_round_games=50` games per pair, deterministic seeds via `EvaluationManager.evaluate_h2h`), builds a (k, k) payoff matrix, runs 100 multiplicative-weights replicator iterations on the centered antisymmetric form, and evicts the entry with the lowest Nash support. `phase3_full.yaml` caps `nash_top_k=16, nash_prune_round_games=16` to keep M1 Pro round-robin cost reasonable.
- **3.6 Opponent ID embedding**: `use_opponent_id_emb=True` adds two `nn.Embedding`s (`opp_kind ∈ {unknown, random, heuristic, self_latest, league, main_exploiter}` and `opp_policy_id ∈ {0..league_maxlen}`) summed in halves and concatenated to the fusion input. Env applies random masking with `opp_id_mask_prob=0.40` (forces both to "unknown") so the policy stays robust to the eval-time "unknown opponent" distribution. Buffer's opt-in `store_opponent_id=True` carries kind/id per timestep.

Phase 3 YAML configs: `configs/phase3_full.yaml` plus 5 leave-one-outs (`phase3_no_{pfsp_hard, latest_policy, trueskill, nash_pruning, opp_id_emb}.yaml`). Param count: ~2.22M (phase2_full) → ~2.24M (phase3_full); the only delta is the embedding pair on the obs encoder.

## Phase 4 (landed — modules; activation deferred to follow-up)
- **4.1 ISMCTS** (1v1-only): `src/catan_rl/algorithms/search/ismcts.py` provides single-step PUCT search with belief-determinization. `ISMCTS(config).search(policy, obs, masks, belief_logits)` returns visit-count distribution over the 13 action types. Uses the policy head's prior + the value head's leaf evaluation; does NOT roll out the env (would require `CatanGame.copy()` which doesn't exist yet — search is policy improvement, not multi-step lookahead). Defaults: `c_puct=1.5`, `n_sims_per_det=50`, `n_determinizations=4`. The module ships *available* but is not wired into the rollout loop yet — gating it on belief-head quality is left for a follow-up. `visits_to_distribution(visits, temperature)` converts to a CE target.
- **4.2 GRU recurrent value head**: `use_recurrent_value=True` replaces `value_net` with `RecurrentValueHead` (`nn.GRUCell` + value MLP). Per-env hidden state maintained by the trainer across rollout steps; resets to zeros on `terminated=True` (preserved on `truncated=True`). Buffer's `gru_hidden_dim>0` triggers per-step `value_hidden_in` storage so PPO update can replay the cell. No BPTT past the single step — gradient flows through one GRU step per sample. `policy.act(..., value_hidden_in=h)` returns 4-tuple including new hidden; `policy.get_value(obs, value_hidden_in=h)` returns `(value, hidden_out)`. Param cost: ~+150k.

## Phase 2 (landed)
- **2.1 Axial positional embedding**: `use_axial_pos_emb=True` adds a learned 2D embedding indexed by hex axial coords `(q, r)` in `TileEncoder`, concatenated to per-tile features before the transformer projection. Breaks the encoder's permutation-equivariance over tiles. `axial_pos_dim=24` (split half/half across q and r); init std=0.02 so it starts as a small perturbation.
- **2.2 Modern transformer recipe**: `transformer_dropout` (None = inherit `dropout`) and `transformer_activation` (`'relu'` | `'gelu'`) override the encoder layer's recipe without disturbing other dropout sites. `phase2_full` uses `0.05 / gelu`. Pre-norm was already on by default.
- **2.3 GNN encoder** over hex/vertex/edge graph: `use_graph_encoder=True` adds a tripartite message-passing encoder (`src/catan_rl/models/graph_encoder.py`) with 19 hex + 54 vertex + 72 edge nodes. Adjacency tables (`hex_to_vertex`, `vertex_to_hex`, `edge_to_vertex`, `vertex_to_edge`) precomputed once via cached `_board_adjacency_tables()` from a single `catanBoard()` build. 2 rounds of mean-pool message passing in both directions; pooled output concatenated to the fusion input alongside the legacy tile-encoder output. Defaults: `graph_hidden_dim=64`, `graph_n_rounds=2`, `graph_out_dim=64`; ~30k params.
- **2.4 AdaLN/FiLM action heads**: `action_head_film=True` flips every context-using head (`corner`, `resource1`, `resource2`) from concat-MLP conditioning to FiLM modulation `(1+γ) ⊙ LN(x) + β` where `(γ, β)` are produced from the head's context. `γ`-init=0 so the modulation is identity at construction. `type/edge/tile` heads have no context and stay on the legacy path regardless of the flag.
- **2.5 Decoupled value tower**: `value_head_mode='decoupled'` builds a second `ObservationModule` exclusively for the value head. Breaks gradient interference between policy and value losses. Adds ~+0.7M params (~1.54M → 2.22M with all Phase 2 flags on).
- **2.5b Belief head** (1v1-only): `use_belief_head=True` adds a 2-layer MLP on the policy encoder's 512-dim output predicting the opponent's hidden dev-card type distribution (5-way over `KNIGHT/VP/ROADBUILDER/YOP/MONOPOLY`). Trained with soft cross-entropy against the *true* normalized count vector that the env exposes via `obs['belief_target']` (training-only — never enters the policy's input). Loss weight `belief_loss_weight=0.05` (default; trainer adds `belief_loss_weight * soft_ce` to the PPO total). Final layer init gain=0.01 → starts near uniform → loss starts at log(5)≈1.609 at init. Buffer's opt-in `store_belief_target=True` carries the (B, 5) target per timestep. Trainer logs `train/belief_loss` per update. **Why 1v1-only**: with no P2P trade and a single opponent, the broadcast tracker reveals everything except dev-card *type*; that's what the head models. With P2P trade the supervision is stale; with 4 players the joint-distribution output dim explodes.
- **2.5c Opponent-action auxiliary head** (1v1-only): `use_opponent_action_head=True` predicts the opponent's *next* action TYPE (13-way) from the policy encoder. Trained with cross-entropy on rollouts where the opponent is a historical league policy (filter excludes random/heuristic/current_self/no-opp — degenerate fixed-point dynamics if trained against ourselves). The trainer captures the opponent's first action of each deferred-opp turn from `_run_batched_opponent_turns` and writes it into `obs['opp_next_action_type']` + `obs['opp_action_target_valid']` before `buffer.add`. Loss weight `opp_action_loss_weight=0.03` (smaller than belief because supervision is noisier — opponent action depends on hidden hand). `OpponentActionHead.masked_cross_entropy` returns `None` when no batch row qualifies; trainer skips the loss term. Logs `train/opp_action_loss`.

Phase 2 YAML configs: `configs/phase2_full.yaml` plus 4 leave-one-outs (`phase2_no_{axial_pos, transformer_recipe, film, decoupled_value}.yaml`).

## Phase 1 (landed)
- **1.1 Value clipping** (PPO2-style): `loss_v = max(MSE_unclipped, MSE_clipped)` with `clip_range_vf=0.2`. Config: `use_value_clipping` (default `True`).
- **1.2 Per-rollout advantage normalization**: `advantage_norm: 'rollout' | 'batch' | 'none'`. Default `'rollout'` standardizes globally over the buffer; the trainer's per-batch step is a no-op in that mode.
- **1.3 Compact obs schema**: `use_thermometer_encoding=False` (Phase 1 default in `configs/phase1_full.yaml`) drops bucket8 thermometers. Player feature dims: 166/173 (legacy) → 54/61 (compact). `_maybe_apply_compact_obs_dims` in `arguments.py` auto-aligns model input dims when only the flag is flipped. **Phase 1 lineage is incompatible with `checkpoint_07390040.pt`** — fresh training only.
- **1.4 Dev-card count encoding**: `use_devcard_mha=False` swaps MHA + sum-pool for `DevCardCountEncoder` (bincount → 2-layer MLP). Saves ~36k params; permutation-invariant by construction.
- **1.5 D6 dihedral symmetry augmentation**: `symmetry_aug_prob=0.5` in `phase1_full.yaml`. `catan_rl.augmentation.apply_symmetry` permutes the tile axis, the within-tile corner/edge feature blocks, and the corner/edge/tile action axes through precomputed tables in `catan_rl.augmentation.symmetry_tables`.

Phase 1 YAML configs: `configs/phase1_full.yaml` plus 5 leave-one-outs (`phase1_no_{value_clip, advantage_norm, thermometer_drop, devcard_count, symmetry_aug}.yaml`).

## Phase 0 (landed)
- `compute_gae` / `compute_gae_vectorized` accept separate `terminated` and `truncated` arrays. Truncations bootstrap with `V(s_T)`; the GAE accumulator resets at any episode boundary. Legacy single-`dones` calling convention still works for back-compat.
- `CompositeRolloutBuffer` stores `terminated` and `truncated` separately; `buffer.dones` is a derived OR-mask for any old code that still reads it.
- `MultiActionHeads` and `CatanPolicy.evaluate_actions` accept `return_per_head=True` and return per-head entropy/log-prob/weight. Trainer logs `train/entropy_head_<name>` (unconditional) and `_cond` (relevance-weighted), plus `train/entropy_collapse_flag`.
- `catan_rl.eval.rules_invariants.run()` is a runtime-callable 1v1 invariant gate (8 default checks; opt-in 9th drift check).
- `catan_rl.selfplay.ratings.RatingSystem/Rating/RatingTable` — TrueSkill if available, in-house Glicko-2 fallback otherwise.
- `scripts/eval_harness.py` runs `--mode {rules-invariant, champion-bench, exploitability, league-rating, all}` and writes JSON to `runs/eval_harness/<run_name>/`.
- `scripts/migrate_checkpoint.py` upgrades pre-Phase-0 checkpoints (config-only patch — no policy state change).
- Observation dim constants `OBS_TILE_DIM, CURR_PLAYER_DIM, NEXT_PLAYER_DIM, MAX_DEV_SEQ, N_TILES, DEV_CARD_VOCAB` are exported from `catan_rl.models.utils`. New code should import them rather than hardcoding `79`/`166`/etc.

## Branch & commit conventions
- Branches: `<type>/<kebab-slug>` (`feat/`, `fix/`, `refactor/`, `chore/`, `docs/`, `test/`).
- Conventional commits, lowercase, under 72 chars.
- No `Co-Authored-By` AI trailers (per global rules).
- One PR per phase; no big-bang merges that span multiple roadmap phases.
