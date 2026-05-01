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

## Branch & commit conventions
- Branches: `<type>/<kebab-slug>` (`feat/`, `fix/`, `refactor/`, `chore/`, `docs/`, `test/`).
- Conventional commits, lowercase, under 72 chars.
- No `Co-Authored-By` AI trailers (per global rules).
- One PR per phase; no big-bang merges that span multiple roadmap phases.
