# Catan RL — Project Conventions (v2)

This is the **v2** codebase (`src/catan_rl/`, custom PPO, 6-head autoregressive
action space). The old **v1** codebase (`catan/`, SB3 MaskablePPO, flat
`Discrete(248)` action space) is **deprecated and fully superseded** — see the
**No v1 artifacts** rule below.

Project governance lives in `.specify/memory/constitution.md` (the ratified
constitution); this file is the operational quick-reference. Where they overlap,
the constitution governs.

## Project goal — **1v1 ONLY, NOT 4-PLAYER**

Solve **1v1 Settlers of Catan** under the Colonist.io ruleset with a superhuman
agent (custom PPO → heuristic bootstrap → league self-play). It is **not** a
4-player agent and must never be generalized to 4-player without explicit
instruction — many choices below only make sense in 1v1 and silently break under
4-player assumptions. Hardware target: Apple M1 Pro (MPS for training, CPU for
eval).

**The 1v1 Colonist.io ruleset (must be preserved across all changes):**

| Rule | 1v1 value | Standard 4p value |
|---|---|---|
| Win condition | **15 VP** | 10 VP |
| Player count | **2** | 3–4 |
| **Player-to-player trading** | **DISABLED** — bank/port only | enabled |
| Discard threshold on 7-roll | **9 cards** | 7 cards |
| **Friendly Robber** | no robber on a hex adjacent to a player with `< 3` visible VP | none |
| Dice | **`StackedDice`** — shuffled bag of 36 outcomes + 1 noise swap + 20% Karma forced-7 if opponent rolled the previous 7 | independent 2d6 |
| Setup | snake draft (1 → 2 → 2 → 1); 2nd settlement yields starting resources | same |
| Board / resources / ports | standard 19 tiles, 54 vertices, 72 edges; standard counts; 5×2:1 + 4×3:1 ports | identical |
| Largest Army / Longest Road | 3 knights / 5 roads | same |

**Implications baked into the engine (`src/catan_rl/engine/`) — do not undo without flagging:**
- `catanGame.maxPoints = 15`, `numPlayers = 2` hardcoded (`engine/game.py`).
- `player.discardResources` uses `maxCards = 9` (`engine/player.py`; also Random/heuristic players).
- `player.initiate_trade` early-returns on any non-`'BANK'` mode — **P2P trading hard-disabled** (`engine/player.py`).
- `catanBoard.get_robber_spots` filters Friendly-Robber-protected hexes (`engine/board.py`).
- `StackedDice` (`engine/dice.py`) replaces independent 2d6.
- `BroadcastHandTracker` (`engine/broadcast.py` / `tracker.py`) does **perfect** opponent hand-tracking — valid only in 1v1 with no P2P trade.
- Action space has **no P2P-trade actions**; `BankTrade` is the only trade type.
- Obs models exactly **one opponent**; no opponent-set encoder.
- Self-play assumes a symmetric 2-player zero-sum game.

Any PR touching game-rule constants, the action space, the obs schema, or the
trading API must state how it preserves the 1v1 ruleset, or be rejected.

## Layout (v2)

- `src/catan_rl/engine/` — pure-Python game engine (game, board, player, dice, broadcast/tracker, geometry).
- `src/catan_engine/` — Rust engine crate (scaffolding; **not** the default backend — `engine_backend: python`).
- `src/catan_rl/env/` — Gymnasium env (`catan_env.py`): dict obs, action masking, opponent dispatch.
- `src/catan_rl/policy/` — `CatanPolicy` (`network.py`): TileEncoder + GNN + opp-id embedding → fusion → 6 action heads + value (`heads.py`, `encoders.py`, `obs_encoder.py`).
- `src/catan_rl/ppo/` — trainer, buffer, GAE, `arguments.py` (config SoT), `training_loop.py`, `vec_env.py`, `game_manager.py`.
- `src/catan_rl/selfplay/` — `league.py` (snapshot pool).
- `src/catan_rl/eval/` — `harness.py` (symmetric-seat WR), `wilson.py`, `rules_invariants.py`.
- `src/catan_rl/{bc,setup_phase,replay,agents,augmentation,checkpoint,cli}/` — BC, setup pretrain, replay/player_factory, heuristic agent, symmetry aug, checkpoint mgr, CLI entry.
- `scripts/` — `train.py` (→ `catan_rl.cli.train`), `train_bc.py`, `generate_bc_dataset.py`, `migrate_checkpoint.py`, replay/record tools. (No v1 `evaluate.py`/`play_vs_model.py`.)
- `configs/` — `ppo_default.yaml`, `bc.yaml`. `docs/plans/v2/` — current roadmap.

## Action space (6 autoregressive heads)

`MultiDiscrete([13, 54, 72, 19, 5, 5])` = `[type, corner, edge, tile, res1, res2]`.
Types: `0 BuildSettlement, 1 BuildCity, 2 BuildRoad, 3 EndTurn, 4 MoveRobber,
5 BuyDevCard, 6 PlayKnight, 7 PlayYoP, 8 PlayMonopoly, 9 PlayRoadBuilder,
10 BankTrade, 11 Discard, 12 RollDice`.

## Observation

Dict obs built by `src/catan_rl/policy/obs_encoder.py` (per-tile features +
current/next player scalars + padded dev-card sequences). **Do not hardcode obs
dims** — they vary by config (thermometer/compact); use the exported `OBS_*`
constants / `obs_schema.py`. Resource order in the RL stack is **Charlesworth**
(`WOOD, BRICK, WHEAT, ORE, SHEEP`), distinct from the engine's `RESOURCES`.

## Rules to follow

1. **No v1 artifacts.** v1 = the deprecated `catan/` codebase. Do **NOT** load,
   train against, evaluate against, or benchmark against any v1 policy,
   checkpoint, or champion (`checkpoint_07390040.pt`, `checkpoint_16162816.pt`,
   etc. — gone from this tree). All policies / checkpoints / league snapshots /
   eval baselines are **v2-only**. The v2 checkpoint lineage starts from the
   heuristic-bootstrap run (`bootstrap_v1`).
2. **Never change engine game rules** without flagging — the engine matches
   Colonist.io 1v1 and drift breaks eval comparability.
3. **Checkpoint compatibility is within v2.** A change that alters the policy
   state-dict shape needs a one-shot migration + documented v2 lineage; prefer
   keeping existing **v2** checkpoints loadable (e.g. `bootstrap_v1`).
4. **TensorBoard scalars are additive** — existing names never renamed.
5. **`arguments.py` is the config source of truth** — README/MEMORY may lag.
6. **Two resource orderings** — `RESOURCES` (engine) vs `RESOURCES_CW` (RL).
7. **Device policy.** Training resolves `auto`→**MPS** on M1 (batched SGD ~3×
   faster at batch 512); **eval is pinned to CPU** (batch=1 faster there); CUDA
   opt-in. Launch long runs detached (`nohup`) so a session restart can't kill
   them.
8. **Don't import `src/catan_rl/gui/`** in any RL/training path — pygame is
   optional and breaks headless runs.
9. **No new docs unless asked.** Update `README.md`/`MEMORY.md` when conventions
   change.

## Testing & smoke

- Train (MPS, full run): `make train` (or `python scripts/train.py --config configs/ppo_default.yaml --run-name <name>`).
- Tests: `pytest` (CI runs ruff + mypy strict + pytest on Python 3.11+; GUI pixel tests skip off-darwin).
- TensorBoard: `tensorboard --logdir runs/train/`.

## Roadmap & governance

- **Active roadmap**: `docs/plans/v2/` (`design.md` is the locked design; `step3_bc.md`, `step4_ppo.md`, `step5_mcts.md`, `setup_strength_roadmap.md`). Spec-driven work flows through Spec Kit (`.specify/`, `specs/`); see `docs/plans/v2/speckit-playbook.md`.
- **Constitution**: `.specify/memory/constitution.md` (authoritative principles).
- **Feature state is not "Phase X landed".** Older docs claimed many advanced
  league/search/aux-head features as "landed"; the 2026-06 gap audit found most
  were **scaffolded-but-unwired in v2**. As of 2026-06-08 the **self-play
  snapshot-opponent keystone IS wired & merged** (in-env full-game opponent
  driver, league assignment consumer, `vec_env.set_opponents` mid-rollout swap,
  `evaluate_policy_vs_policy`); self-play *training* has not been RUN yet. Still
  unwired: PFSP, TrueSkill, Nash pruning, exploiters, MCTS, piKL, belief/
  opp-action aux heads. **Verify actual feature state against `src/catan_rl/`**,
  not against feature-claim lists.

## Commit & workflow conventions

- **Solo project — NO pull requests.** Commit and push directly to
  `origin/main` (`git push origin main`). Short-lived *local* branches are fine
  for keeping risky/in-progress work off main until it's green, then merge to
  main and push — but no PR review gate.
- Conventional commits, lowercase, under 72 chars.
- No `Co-Authored-By` AI trailers.
- CI still runs on push to main (ruff + mypy + pytest, Python 3.11+); keep it
  green, but it's a safety net, not a merge gate.

<!-- SPECKIT START -->
Active Spec Kit feature: **self-play snapshot opponent** — see
`specs/001-selfplay-snapshot-opponent/plan.md` (plus `spec.md`, `research.md`,
`data-model.md`, `contracts/`, `quickstart.md`) for the current plan.
<!-- SPECKIT END -->
