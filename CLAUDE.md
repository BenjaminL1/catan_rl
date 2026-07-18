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
| **Resource bank** | **finite 19 per resource** + official depletion (bank short & one player owed → sole claimant takes the remainder; bank short & both owed → neither receives) | same (finite 19) |
| Largest Army / Longest Road | 3 knights / 5 roads | same |

**Implications baked into the engine (`src/catan_rl/engine/`) — do not undo without flagging:**
- `catanGame.maxPoints = 15`, `numPlayers = 2` hardcoded (`engine/game.py`).
- `player.discardResources` uses `maxCards = 9` (`engine/player.py`; also Random/heuristic players).
- `player.initiate_trade` early-returns on any non-`'BANK'` mode — **P2P trading hard-disabled** (`engine/player.py`).
- `catanBoard.get_robber_spots` filters Friendly-Robber-protected hexes (`engine/board.py`).
- `StackedDice` (`engine/dice.py`) replaces independent 2d6.
- **Finite resource bank** (`catanBoard.resourceBank`, 19 each; spec 009): dice
  production is metered by `resolve_bank_production` (the official depletion
  rule), and build / dev-buy / bank-trade-give / discard costs **recirculate**
  into the bank while setup-grant / YoP / bank-trade-receive **draw** from it.
  Conservation invariant: `resourceBank[R] + Σ hands[R] == 19` for every R. The
  bank is **engine state only — NOT in the obs** (no policy-shape change). All
  mutation paths (engine, env, recorder, heuristic, random_ai, bc.dataset,
  labeling) route through `bank_recirculate` / `bank_draw`. Mirrors the Torevan
  TS `resourceBank`; the conformance harness pins parity (seeds 7/8/15 are a
  no-op; depletion is exercised by a dedicated fixture + cross-engine tests).
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
  were **scaffolded-but-unwired in v2**. As of 2026-06-09: the **self-play
  snapshot-opponent keystone is wired & merged** (in-env full-game opponent
  driver, league consumer, `vec_env.set_opponents` mid-rollout swap,
  `evaluate_policy_vs_policy`); **self-play training is RUNNING** (honest-obs
  lineage seeded from `bootstrap_v1` u799); the **belief head is wired** (aux
  soft-CE on opponent hidden dev-card types, `belief_coef=0.05`); the obs is
  **honest** (opponent hidden dev types + hidden VP no longer leak). Still
  unwired: PFSP, TrueSkill, Nash pruning, exploiters, MCTS, piKL, opp-action aux
  head. **Verify actual feature state against `src/catan_rl/`**, not against
  feature-claim lists.

## Commit & workflow conventions

- **Solo project — NO pull requests.** Commit and push directly to
  `origin/main` (`git push origin main`). Short-lived *local* branches are fine
  for keeping risky/in-progress work off main until it's green, then merge to
  main and push — but no PR review gate.
- Conventional commits, lowercase, under 72 chars.
- **Never add Claude (or any AI) as a contributor.** No `Co-Authored-By: Claude`
  / `Co-Authored-By: <AI>` trailers, no "Generated with Claude" lines, no AI
  attribution anywhere in commit messages, PR bodies, or authorship — even when
  a tool's default template suggests one. Commits are authored solely by the
  human account.
- CI still runs on push to main (ruff + mypy + pytest, Python 3.11+); keep it
  green, but it's a safety net, not a merge gate.

## Review-and-resolve loop (the default workflow for feature work)

When the user says **"the review-and-resolve loop"** (or asks to "review and
resolve until ready"), run this loop per feature/slice — it is the standing
expectation for substantive implementation work:

1. **Implement** the feature (or a coherent slice), test-first where practical.
   Re-green `ruff` + `mypy --strict` + `pytest` and commit.
2. **Review** — run a senior-RL-game-dev review of the diff via the **Workflow**
   tool: independent lenses (typically RL-experiment-correctness + SWE/additivity,
   adapted to the feature) reading the actual files, → a synthesis that returns a
   **severity-tagged issue list** (`BLOCKER` / `SHOULD-FIX` / `NIT`) + a verdict.
   This is the same "senior RL engineer" reviewer used throughout the project
   (the `specs/003-inference-search/reviewers.md` A+B personas generalized).
3. **Resolve** every `BLOCKER` and `SHOULD-FIX` (NITs at discretion). Re-green
   ruff + mypy + pytest; commit each resolution.
4. **Loop** — re-review (or spot-verify the fixes) until the verdict is **READY**
   (no open BLOCKER/SHOULD-FIX). Only then move on.
5. **Next feature** — repeat from (1) for the next item in the plan.

**With a long-running training run in the loop:** launch the run as early as a
*correct* config allows (it is usually the long pole), then do the review/resolve
work while it trains. A review BLOCKER that **invalidates the running config**
(wrong opponent, broken warm-start, wrong objective) → **kill + relaunch**;
non-invalidating issues (gate script, tests, docs, NITs) are fixed in-flight.
Gate-first still governs: never commit the expensive *next* stage before the
current stage's go/no-go gate result is in.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current feature plan
at specs/009-finite-resource-bank/plan.md
<!-- SPECKIT END -->
