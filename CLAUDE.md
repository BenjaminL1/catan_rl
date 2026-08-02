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
| **Pre-roll dev card** | **allowed** — one dev card (Knight / YoP / Monopoly) may be played BEFORE the dice, both seats (ruleset epoch **R1**) | same |

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
  bank remaining (`bank[R]/19`, all 5 resources) **IS now in the obs** — the
  POV-neutral `global_features` block added by the pointer-arch fork (ratified
  2026-07-19, `.claude/veriloop/specs/pointer-arch-fork.md` D3.3). This reverses
  the earlier "engine state only — NOT in the obs" rule, whose justification (no
  policy-shape change) evaporated in a fork whose purpose is a shape change; the
  bank is still read via **read-only accessors** (no mutation path near
  `bank_recirculate`/`bank_draw`, conservation untouched). All mutation paths
  (engine, env, recorder, heuristic, random_ai, bc.dataset,
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
- `scripts/` — `train.py` (→ `catan_rl.cli.train`), `train_bc.py`, `generate_bc_dataset.py`, `migrate_checkpoint.py`, replay/record tools, `play_vs_model.py` (the v2 human-vs-policy GUI harness — bot panel is BLIND unless `--reveal-bot`, but always shows the
  PUBLIC facts: knights played + longest-road length, plus an on-screen move log,
  the public-reveal-derived dev-deck remaining, and a BUY DEV CARD button greyed
  on the bot's own mask predicate; games log `"hud": 3` for that information
  regime, `"rules_epoch": 2` for the human's legal option set (which now matches
  the bot's — see `.claude/veriloop/specs/human-path-conformance-fixes.md`), and
  `"rules_ok"` / `"rules_violations"` from an interactive `rules_invariants`
  audit that RECORDS and never asserts. The harness runs **ruleset
  `R1`**, so **BOTH seats get the one-card pre-roll dev window** (Knight / YoP /
  Monopoly, never Road Builder). The human's menu is narrowed by
  `_pre_roll_dev_options()`, which reads the very `compute_action_masks` type
  mask the policy's own pre-roll node is built from (through the new optional
  `gui/view.catanGameView.dev_card_filter`, default `None` = unrestricted), so
  seat symmetry is structural rather than a parallel card list; the auto-played
  human seat (headless `--self-test`) reaches the base env's
  `heuristic_pre_roll` unchanged — the window in KIND, but NOT in extent
  (Knight only, by a scripted rule), so the stand-in stays a weaker model of the
  human's seat. MCTS clones never take that branch (a clone always carries a
  snapshot opponent and the heuristic pre-roll is gated on there being none) but
  are not untouched by `R1`: `_opponent_pre_roll`'s snapshot branch no longer
  early-returns, so clones now sample the modelled seat's pre-roll node. Games
  log `"preroll": true` **derived from `env.ruleset`**, plus `"ckpt_ruleset"`
  (an absent checkpoint stamp ⇒ `R0`); an epoch mismatch warns loudly but does
  not refuse — `DEFAULT_CKPT` is itself an R0 checkpoint worth playing and an
  unreadable stamp must not stop a game, so the DEFAULT invocation mismatches
  by construction (pass `--ckpt <R1-ckpt>`). The check runs in
  `play_interactive` only; `--self-test` returns before it. That pre-roll
  window is itself an option-set change, so it is part of **`rules_epoch` 2**
  alongside the conformance fixes above — epoch 2 = R1 pre-roll + a dev-card
  picker that no longer strands a public played-counter + a robber victim
  picker that filters the human out.
  The human resource picker (`gui/view.get_resource_selection`) routes
  through the finite bank, the driver asserts `bank[R] + Σ hands[R] == 19` after
  every step and records `"bank_ok"`, an interrupted game still writes both
  artifacts with `"partial": true` (including the real window-close path, which
  calls `pygame.quit()` before `sys.exit(0)` — the post-loop redraw is guarded),
  and every record carries `"git_sha"` (`-dirty`-suffixed on a modified tree) +
  `"ckpt_sha256"`. `DEFAULT_CKPT` is the BANKED `runs/anchors/ptr_v1_u500.pt`,
  never a live `runs/train/**` path under `keep_last_n` rotation).
  (No v1 `evaluate.py`.)
- `configs/` — `ppo_default.yaml`, `bc.yaml`. `docs/plans/v2/` — current roadmap.

## Action space (6 autoregressive heads)

`MultiDiscrete([13, 54, 72, 19, 5, 5])` = `[type, corner, edge, tile, res1, res2]`.
Types: `0 BuildSettlement, 1 BuildCity, 2 BuildRoad, 3 EndTurn, 4 MoveRobber,
5 BuyDevCard, 6 PlayKnight, 7 PlayYoP, 8 PlayMonopoly, 9 PlayRoadBuilder,
10 BankTrade, 11 Discard, 12 RollDice`.

## Ruleset epochs — R0 vs R1 (`src/catan_rl/env/ruleset.py`)

The env carries a `ruleset` constructor arg. **Numbers from the two epochs are
NOT comparable**; every banked figure (v8 `0.668` WR, search `+54.6` Elo, the
accept-gate clause-(a) `n=600` result) is an **R0** number.

- **`R0`** — the shipped rules up to 2026-07. At a `roll_pending` node the mask
  offers exactly `ROLL_DICE` to **every policy seat**. It is the **default**
  everywhere in code (`compute_action_masks`, `CatanEnv`, `EvalHarness`), so no
  caller re-rules a banked checkpoint merely by upgrading. The scripted
  heuristic's pre-roll Knight is shipped R0 behaviour and runs under **both**
  epochs — it never goes through the mask, and every banked number was measured
  with it on, so gating it would stop `R0` reproducing the epoch it names.
- **`R1`** — Colonist-faithful: **both policy seats** may play ONE dev card
  before rolling. Whitelist Knight / YoP / Monopoly; **Road Builder excluded**
  (free roads must not straddle the roll). Selected explicitly by
  `RolloutConfig.ruleset` (`configs/ppo_default.yaml` → `rollout.ruleset: R1`),
  which is the config source of truth. **`RolloutConfig.ruleset` itself defaults
  to `R0`** and `ppo_default.yaml` is the only config that opts in — the other
  train configs (notably `selfplay_v8_cont_resume.yaml`, which exists to RESUME
  a banked R0 lineage) carry no key and stay R0, and `("rollout", "ruleset")` is
  in `training_loop._RESUME_CRITICAL_FIELDS` so a resume-time epoch flip warns
  loudly instead of silently re-ruling a lineage mid-run. Action space, obs
  schema, mask keys and checkpoint shapes are unchanged — `roll_pending` is
  already an obs phase flag and types 6-9 already exist.
- **Every run stamps its epoch.** `TrainConfig.to_dict()` lands in the
  checkpoint payload, so `eval.harness.checkpoint_ruleset(path)` reads a
  checkpoint's epoch back; **an absent stamp means R0** (all pre-slice
  checkpoints). `cross_arch_h2h` and `evaluate_search_vs_policy` take BOTH
  checkpoint paths, so they read **both** seats' stamps automatically and an
  R1-vs-R1 rung there needs no flag. `evaluate_policy_vs_policy` / `EvalHarness`
  auto-read the **opponent** stamp only — their champion arrives as a live
  policy object, not a path — so the champion seat falls back to the `ruleset=`
  default (`R0`), and a caller holding the champion's checkpoint path must pass
  `ruleset=checkpoint_ruleset(path)` (as `scripts/elo_ladder.py`,
  `scripts/run_exploiter_gate.py` and `expert_iteration/gate.py` do). Omit it
  with an R1 champion and the matchup fails **closed** —
  `CrossRulesetEvalError` on a legitimate same-epoch pairing — never a silently
  mixed number.
- A cross-epoch head-to-head is **refused** (`CrossRulesetEvalError`), never
  annotated. The single sanctioned override is `allow_mixed_ruleset=True`,
  which does not relax the rules — it gives each seat its own epoch
  (`CatanEnv(ruleset=..., opponent_ruleset=...)`) so both policies play what
  they were trained for. That is D9's replacement accept gate.
- The post-game rule audit's event-stream checks (one-dev-card-per-turn, no
  P2P trade) need `CatanEnv(audit_events=True)` — the engine's broadcast keeps
  no history. Every audited path sets it; training does not. The one-card check
  infers Knight plays from per-turn `MOVE_ROBBER` events (minus the one owed to
  an own rolled 7), since Knight emits no card event of its own.
- The conformance recorder (`record_game(..., ruleset=...)`,
  `catan-rl-conformance --ruleset`) also defaults to **R0**: the Torevan TS
  reference engine still refuses pre-roll actions, so an R1 fixture is not
  replayable there until the TS half of D10 lands. Logs carry a `"ruleset"`
  field; `CONFORMANCE_SCHEMA_VERSION` is 2.
  **Owed, cross-repo:** the Torevan oracle still pins the OLD version —
  `Torevan/packages/engine/src/conformance/conformance.test.ts` asserts
  `expect(log.schema_version).toBe(1)`. This repo commits no fixtures, so
  `make test-unit` cannot see it; the oracle goes RED the first time ANY
  fixture is regenerated (even an R0 one, whose steps are otherwise identical
  to v1). Bump that assertion to `2` in the same change that regenerates
  fixtures.
- Spec: `.claude/veriloop/specs/preroll-dev-cards-r1.md`.

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

## /dev-loop (the default workflow for feature work)

Feature work runs through the **veriloop-generated `/dev-loop`**
(`.claude/workflows/catan-rl-v2-dev-loop.js`; installed 2026-07-17, superseding the
hand-run "review-and-resolve loop" below): spec detection (`/dev-plan` for non-trivial
specs) → plan-review against `.claude/veriloop/constitution.md` → worktree implement →
risk-tiered GO/NO-GO gate on **real command exit codes** (`make typecheck` · `make lint`
· `cargo fmt --check` · `make test-unit`) plus review lenses
(`.claude/veriloop/experts/`: baseline + drift) → bounded auto-fix → docs sync →
preview branch, stopping before merge for owner sign-off. The old loop's
RL-experiment-correctness + SWE/additivity review intent lives on in the lens
personas — extend them via the `.overrides.md` siblings, which are never overwritten.

*(Legacy phrase: if the user says "the review-and-resolve loop", run `/dev-loop`.)*

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
