# Superhuman 1v1 Catan Agent — Implementation Roadmap

**Document version:** 1.0
**Created:** 2026-04-30
**Status:** Awaiting file-layout restructuring; implementation begins after restructure
**Scope:** Full 5-phase plan to take the current ~7.4M-step PPO + League system to documented superhuman 1v1 Catan play
**Hardware target:** Apple M1 Pro, CPU-only baseline; optional CUDA path additive only
**Prior art consulted:** PPG (Cobbe et al. 2020), AlphaStar (Vinyals et al. 2019), AlphaZero (Silver et al. 2018), Charlesworth Catan-RL (2018), MuZero (Schrittwieser et al. 2020)

---

## 0. How to Use This Document

This document is the single source of truth for the upgrade. It is written to be unambiguous so that future-you (or another engineer) can pick up any phase without re-reading the conversation that produced it.

**File-path reconciliation.** The pre-Phase-0 file restructure is **complete** (see [`file_layout_restructure.md`](file_layout_restructure.md) and [ADR 0006](../decisions/0006-src-layout-restructure.md)). All file paths in this document use the **new src-layout** (`src/catan_rl/...`). The repo is now installable via `pip install -e ".[dev]"`, has YAML configs under `configs/`, and a test suite under `tests/`. If a path here doesn't resolve, treat it as a documentation bug.

**Reading order.**
1. Section 1 (success criteria) and Section 2 (1v1 invariants) are mandatory before touching code.
2. Sections 3–7 are the phases, in implementation order. Each phase has identical structure: scope, files, code changes, ablation, rollback, compute, decision gate, risks.
3. Section 8 (DAG) and Section 12 (suggested ordering) tell you what can run in parallel.
4. Section 13 (file list) is the master index.

**Phase atomicity.** Each phase is independently mergeable. No phase may be merged without:
1. Passing its ablation metric (Section 2.3 of that phase's subsection).
2. Passing the rules-invariant test (Section 2 below).
3. Producing a TensorBoard run + checkpoint that the next phase can resume from.

**No big-bang merges.** Every merge is one phase. Sub-features within a phase may be bundled if their ablation can prove the bundle helps without leave-one-out, but they must be individually feature-flagged so a leave-one-out is possible if the bundle fails.

---

## 1. Project Goal & Success Criteria

### 1.1 Goal
Train a superhuman 1v1 Settlers of Catan agent under the Colonist.io ruleset using custom PPO + League self-play, on a single Apple M1 Pro CPU.

### 1.2 Success criteria (all three must be met to declare success)

| # | Criterion | Threshold | Measurement |
|---|-----------|-----------|-------------|
| C1 | Beats heuristic AI | ≥99% over 200 deterministic eval games | `scripts/eval_harness.py --mode champion-bench --opponent heuristic` |
| C2 | Beats previous best champion | ≥70% over 200 deterministic eval games | H2H vs `checkpoints/train/checkpoint_07390040.pt`, both first-mover orderings |
| C3 | Exploitability | <5% WR for a fresh 5M-step adversary | `scripts/eval_harness.py --mode exploitability` |

### 1.3 Non-goals (explicit)
- 4-player Catan (see Section 2 invariants — this is a hard scope boundary).
- Player-to-player trading (hard-disabled in engine).
- Browser / Colonist.io integration plumbing.
- GPU-only kernels. CUDA path is opt-in; CPU baseline must never regress.
- Custom maps, expansions, scenarios.

### 1.4 Frozen artifacts
- **Frozen champion:** `checkpoints/train/checkpoint_07390040.pt` (~7.4M steps; current best). Must remain loadable through the entire roadmap. It is the C2 benchmark and the C3 starting point.
- **Eval seeds:** integers `0..199`, frozen at Phase 0. Same seed list is used for every champion-bench run for reproducibility.

---

## 2. Cross-Cutting 1v1 Invariants

These are **hard constraints** enforced by the rules-invariant test (Phase 0). Any phase that violates them is rejected regardless of metric improvement.

### 2.1 Rule table

| Rule | 1v1 Colonist.io value | Standard 4p value | Code enforcement point |
|---|---|---|---|
| Win condition | **15 VP** | 10 VP | `catanGame.maxPoints = 15` (`src/catan_rl/engine/game.py:34`) |
| Player count | **2** | 3–4 | `catanGame.numPlayers = 2` (`src/catan_rl/engine/game.py:35`) |
| Player-to-player trading | **DISABLED** | enabled | `player.initiate_trade` early-returns on non-`'BANK'` (`src/catan_rl/engine/player.py:520-522`) |
| Discard threshold on 7 | **9 cards** | 7 | `player.discardResources` `maxCards=9` (`src/catan_rl/engine/player.py:538`); also `RandomAIPlayer.discardResources`, `heuristicAIPlayer.discardResources` |
| Friendly Robber | cannot place on hex adjacent to player with `<3` visible VP | none | `catanBoard.get_robber_spots` (`src/catan_rl/engine/board.py:407-428`) |
| Dice mechanic | `StackedDice` (36-bag + noise + 20% Karma) | independent 2d6 | `src/catan_rl/engine/dice.py` |
| Setup | snake draft, 2nd settlement yields starting resources | same | `catanGame.build_initial_settlements` + `_grant_setup_resources` in env |
| Largest Army threshold | 3 knights | 3 | `catanGame.check_largest_army` |
| Longest Road threshold | 5 roads | 5 | `catanGame.check_longest_road` |

### 2.2 Architectural assumptions baked into the code

- **Perfect opponent hand tracking** (`src/catan_rl/env/hand_tracker.py`): valid only because 1v1 + no P2P trade ⇒ every resource delta is broadcast-observable. Any new action that mutates resources without emitting a `RESOURCE_CHANGE` event breaks this.
- **No P2P trade actions** in the action space. The 13-type action space has BankTrade only; no propose/accept/counter actions.
- **Single-opponent observation**: `next_player_main` and `next_player_played_dev` model exactly one opponent. No "list of opponents" abstraction.
- **2-player symmetric zero-sum self-play**: PFSP, Nash pruning, exploitability metric all assume this.
- **Reward sign-flip symmetry**: useful for value-target augmentation under player swap (Phase 1.5).

### 2.3 Rules-invariant test (created in Phase 0)

**File:** `src/catan_rl/eval/rules_invariants.py` (new module). See `tests/unit/eval/test_rules_invariants.py` for the unit tests already shipped during the restructure — Phase 0 promotes these into a runtime-callable harness step.

**Function:** `run() -> List[InvariantFailure]`

**Checks (each is its own assertion):**
1. `catanGame().maxPoints == 15`.
2. `catanGame().numPlayers == 2`.
3. Discard threshold: construct a player with 10 cards, run a sandboxed 7-roll, assert discard count == 5.
4. P2P trade hard-disabled: `player.initiate_trade(game, 'PLAYER')` and `'OPEN_TRADE'` both return immediately with no state mutation.
5. Friendly Robber: construct a board state where one player has visible_vp == 2 on hex H, assert `get_robber_spots()` does not contain H.
6. `StackedDice` produces 36-bag draws across 100 rolls (sum-distribution check) and Karma triggers ≥10/50 times when last_7_roller is opponent.
7. Action space: `CatanEnv.action_space` is `MultiDiscrete([13, 54, 72, 19, 5, 5])` exactly.
8. Mask keys: `env.get_action_masks().keys()` == frozen set of 9 documented keys.
9. Hand tracker drift: run 50 random games with `verify=True`; tracker must equal actual on every step.

**Wiring:** `scripts/eval_harness.py` calls `rules_invariants.run()` as a precondition. Harness fails fast (exit 1) if any invariant fails.

**Phase gating:** Decision gate of every phase requires `rules_invariants.run()` to return zero failures.

---

## 3. Phase 0 — Eval Harness + Correctness Bugs (BLOCKING)

**Why first.** Without a deterministic harness, no later phase can prove progress. The GAE/`dones` truncation bug silently miscredits every truncated episode (with `max_turns=500` truncations are non-trivial), so any later "improvement" is partially confounded by it. The rules-invariant test prevents 4-player drift through the entire roadmap.

### 3.1 Files

**Modify:**
- `src/catan_rl/env/catan_env.py` — return `(obs, reward, terminated, truncated, info)` (already does, but `step` aggregates them inconsistently).
- `src/catan_rl/algorithms/common/gae.py` — `compute_gae` and `compute_gae_vectorized` accept separate `terminated` and `truncated` arrays.
- `src/catan_rl/algorithms/common/rollout_buffer.py` — split `self.dones` into `self.terminated` and `self.truncated`; update `add()` signature.
- `src/catan_rl/algorithms/ppo/trainer.py` — pass `terminated, truncated` through `collect_rollouts`; update episode-end branch; add per-head entropy logging.
- `src/catan_rl/models/policy.py` — `evaluate_actions(..., return_per_head=False)` flag.
- `src/catan_rl/models/action_heads_module.py` — surface per-head log-prob/entropy tuples when `return_per_head=True`.
- `src/catan_rl/eval/evaluation_manager.py` — add `evaluate(..., seeds=None)`, `evaluate_h2h(...)`, `compute_exploitability(...)`.
- `src/catan_rl/algorithms/ppo/arguments.py` — add config keys (Section 3.5).
- `README.md` — reconcile obs schema (current code mixes 78/79 references, see Section 3.4).

**Create:**
- `src/catan_rl/eval/__init__.py`, `src/catan_rl/eval/rules_invariants.py`.
- `src/catan_rl/selfplay/ratings.py` — TrueSkill / Glicko-2 wrapper (used heavily in Phase 3, but the wrapper is created here).
- `scripts/eval_harness.py` — main harness CLI.
- `scripts/migrate_checkpoint.py` — one-shot translator for pre-Phase-0 checkpoints.
- `src/catan_rl/models/utils.py` add `OBS_TILE_DIM`, `CURR_PLAYER_DIM`, `NEXT_PLAYER_DIM` constants.

### 3.2 GAE truncation fix (specifics)

**Problem.** Current `compute_gae`:

```python
next_non_terminal = 1.0 - dones[t]
delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
```

`dones[t]` aggregates `terminated | truncated`. On truncation, this zeroes the bootstrap (treats truncation as terminal), which is wrong: truncated trajectories should bootstrap with `V(s_T)`.

**Fix.** New signature:

```python
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_value: float,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        # Bootstrap on truncation (use V(s_{t+1})), zero on termination.
        non_terminal = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        # Reset GAE accumulator at any episode boundary (terminated OR truncated).
        last_gae = delta + gamma * gae_lambda * non_terminal * (1.0 - truncated[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
```

Same shape change in `compute_gae_vectorized` (round-robin per-env stripes).

### 3.3 Per-head entropy logging

**Problem.** Current `update()` logs only joint entropy. If e.g. the type head converges, the conditional heads can collapse silently while joint entropy still looks healthy.

**Fix.** Modify `MultiActionHeads.forward(...)` to return a per-head entropy tuple when `return_per_head=True`. New TB scalars:
- `train/entropy_head_type`
- `train/entropy_head_corner`
- `train/entropy_head_edge`
- `train/entropy_head_tile`
- `train/entropy_head_resource1`
- `train/entropy_head_resource2`

Plus a derived flag `train/entropy_collapse_flag` set to 1 when any head's entropy < `entropy_collapse_threshold` for ≥ 3 consecutive updates.

### 3.4 Observation-schema reconciliation

**Problem.** `src/catan_rl/algorithms/common/rollout_buffer.py:6` docstring claims `(n_steps, 19, 78)`; line 55 allocates `(n_steps, 19, 79)`. `env.py:69` declares `TILE_DIM = 79`. README claims 79.

**Fix.** Lock `OBS_TILE_DIM = 79` in `src/catan_rl/models/utils.py`. Replace all hardcoded `79` literals across `observation_module.py`, `rollout_buffer.py`, `ppo.py` with the constant. Same for `CURR_PLAYER_DIM = 166`, `NEXT_PLAYER_DIM = 173`. Update `rollout_buffer.py` docstring. Update `README.md` to confirm 79.

### 3.5 Eval harness CLI

**File:** `scripts/eval_harness.py`

**Modes:**
- `--mode champion-bench --champion <ckpt> --candidate <ckpt> --opponent {heuristic,random,champion} --n-seeds 200` → H2H over `range(0, n_seeds)`, swap first-player, output JSON to `runs/eval_harness/<run_name>/champion_bench.json`. Both directions averaged → reported WR.
- `--mode exploitability --champion <ckpt> --n-steps 5_000_000` → train fresh adversary against frozen champion, return final WR. Output to `exploitability.json`.
- `--mode league-rating --policies <ckpt_glob>` → round-robin TrueSkill across all snapshots. Output to `ratings.json`.
- `--mode rules-invariant` → run only the invariant test, exit 0/1.
- `--mode all` → invariant + champion-bench + exploitability + league-rating.

**Determinism.** Seed propagates: `seed → np.random.seed(seed) + torch.manual_seed(seed) + random.seed(seed) + os.environ['PYTHONHASHSEED'] = str(seed)`. The seed determines board layout, dice bag, port shuffle, and any tie-breaks in setup heuristic. Verify by running the same seed twice and asserting identical game logs.

**H2H method:** for each seed `s`:
1. Run game with policy A first, policy B second, seed `s`. Record `winner`.
2. Run game with policy B first, policy A second, seed `s`. Record `winner`.
Total: `2 × n_seeds` games. WR(A) = (games where A won) / (2 × n_seeds).

### 3.6 Checkpoint migration

**File:** `scripts/migrate_checkpoint.py`

**Behavior.** Loads any pre-Phase-0 checkpoint (e.g. `checkpoint_07390040.pt`), patches the `config` dict to include all new Phase-0 keys with defaults, re-saves under the new schema. Does not modify policy state-dict (Phase 0 makes no architectural changes).

**Acceptance test:**
```bash
python scripts/migrate_checkpoint.py checkpoints/train/checkpoint_07390040.pt
python -c "from catan.rl.ppo.ppo import CatanPPO; t = CatanPPO.load('checkpoints/train/checkpoint_07390040.pt'); print(t.global_step)"
# Must print 7390040 without error.
```

### 3.7 Config keys added in Phase 0

```python
# arguments.py — additions
"eval_harness_seeds": list(range(0, 200)),
"eval_harness_swap_first_player": True,
"frozen_champion_path": "checkpoints/train/checkpoint_07390040.pt",
"entropy_collapse_threshold": 0.0005,
"entropy_collapse_consecutive_updates": 3,
```

### 3.8 Ablation

Train two new runs **from scratch**, 5M steps each, identical seed list:
- `phase0_baseline` — current code, GAE bug present.
- `phase0_fixed` — Phase 0 changes applied.

**Metric.** WR vs heuristic at 5M; expected absolute improvement +1–3 pp. EV improvement on long-truncated episodes ≥ +0.05.

**Per-head entropy:** `phase0_fixed` must not show silent collapse (entropy < 0.001 on any head before update 200).

### 3.9 Rollback condition

- WR vs heuristic at 5M is *worse* by > 2 pp **and** EV not improved → revert (likely a regression elsewhere in the patch).
- Rules-invariant test fails → block merge until fixed.

### 3.10 Compute budget

| Item | Steps | Wall-clock (M1 Pro @ 2.5M steps/day) |
|---|---|---|
| `phase0_baseline` 5M run | 5M | 2 d |
| `phase0_fixed` 5M run | 5M | 2 d |
| Exploitability harness validation | 5M | 2 d |
| Engineering (harness, migration, docs, invariants) | — | 1.5 d |
| **Total Phase 0** | **15M** | **~7–8 d** |

### 3.11 Decision gate

All must pass:
1. `phase0_fixed` ≥ `phase0_baseline` WR (within ±2 pp).
2. Frozen `checkpoint_07390040.pt` produces self-consistent TrueSkill rating across 2 harness runs (variance < 50 TS units).
3. Exploitability harness completes one full cycle, outputs a number, and the resulting JSON loads cleanly.
4. `rules_invariants.run()` returns zero failures.

If any fails: do not proceed to Phase 1.

### 3.12 Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | GAE fix interacts with value normalizer running stats | Reset `value_normalizer` in `phase0_fixed`; assert EV ≥ 0 on first update post-reset |
| MEDIUM | `terminated`/`truncated` API change breaks every caller | Grep for `done` in `src/catan_rl/`; add typed `TransitionResult` namedtuple at the boundary |
| MEDIUM | Determinism seeding incomplete; H2H games not reproducible | Five-way seed (`np`, `torch`, `random`, `PYTHONHASHSEED`, `torch.cuda.manual_seed_all`); add seed-dump assertion in harness |
| LOW | TrueSkill library unpinned | Pin in `requirements.txt`; vendor 200 LOC if dependency proves fragile |

---

## 4. Phase 1 — Sample Efficiency from Current Architecture

**Why second.** Cheap wins that compound. Value clipping reduces value-loss variance; advantage normalization stabilizes early updates; symmetry augmentation gives ~24× effective data multiplier on board-positional features (`D6 × Z_2` due to 1v1 player-swap symmetry); replacing `bucket8` thermometer encodings shrinks the input dimensionality on dozens of channels with no information loss; replacing dev-card MHA with count encoding eliminates ~30k wasted params.

### 4.1 Sub-features

#### 4.1.1 Value clipping (PPO2-style)

**File:** `src/catan_rl/algorithms/ppo/trainer.py:425-430` (the `value_loss` line).

**Before:**
```python
value_loss = nn.functional.mse_loss(values, norm_returns)
```

**After:**
```python
if self.use_value_clipping:
    v_clipped = batch['old_values'] + (values - batch['old_values']).clamp(
        -self.clip_range_vf, self.clip_range_vf
    )
    v_loss_unclipped = (values - norm_returns).pow(2)
    v_loss_clipped = (v_clipped - norm_returns).pow(2)
    value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
else:
    value_loss = nn.functional.mse_loss(values, norm_returns)
```

**Buffer:** `CompositeRolloutBuffer.get_batches` already exposes `_t_values`; rename to `'old_values'` in the yielded dict.

**Config:** `clip_range_vf: 0.2`, `use_value_clipping: True`.

#### 4.1.2 Per-rollout advantage normalization

**File:** `src/catan_rl/algorithms/ppo/trainer.py:411-412` (per-batch norm) and `src/catan_rl/algorithms/common/rollout_buffer.py` `compute_returns_and_advantages`.

**Change.** Add `advantage_norm: "batch" | "rollout" | "none"` config key, default `"rollout"`. When `"rollout"`, normalize at end of `compute_returns_and_advantages` using global mean/std over the buffer; per-batch norm becomes identity.

**Config:** `advantage_norm: "rollout"`.

#### 4.1.3 Drop `bucket8` thermometer encodings

**File:** `src/catan_rl/env/catan_env.py:81-89` (`bucket8` definition) and all call sites in `_build_current_player_main`, `_build_next_player_main`.

**Change.** Replace each `bucket8(x, max_v)` (8 dims) with `np.array([x / max_v], dtype=np.float32)` (1 dim) followed by a LayerNorm at the encoder input. `current_player_main` shrinks from 166 → ~50; `next_player_main` shrinks from 173 → ~57. Update `OBS_TILE_DIM`, `CURR_PLAYER_DIM`, `NEXT_PLAYER_DIM` constants.

**[1v1] Migration impact:** Phase-1 policies are **not** state-dict-compatible with `checkpoint_07390040.pt`. The frozen champion stays on the old shape for benchmark purposes; new lineage starts at Phase 1.

**Config:** `use_thermometer_encoding: False` (default off after Phase 1).

#### 4.1.4 Dev-card MHA → count encoding

**File:** `src/catan_rl/models/player_modules.py` (`CurrentPlayerModule`, `OtherPlayersModule`).

**Change.** Replace MHA over padded (15-token) dev-card sequences with:
```python
counts = torch.zeros(B, 6, device=x.device)
counts.scatter_add_(1, dev_card_ids.long(), torch.ones_like(dev_card_ids, dtype=torch.float))
counts = counts[:, 1:]  # drop padding token
counts = counts / 16.0
emb = self.dev_count_mlp(counts)  # Linear(5, 32) -> ReLU -> Linear(32, 25)
```

Order doesn't matter for dev cards (it's a multiset). Saves ~30k params.

**Config:** `use_devcard_mha: False`.

#### 4.1.5 Hex dihedral symmetry data augmentation (`D6 × Z_2`)

**Files:**
- `src/catan_rl/augmentation/symmetry_tables.py` (new) — precomputed permutations for the 12 D6 elements + the Z_2 player-swap.
- `src/catan_rl/augmentation/dihedral.py` (new) — `apply_symmetry(obs, action, masks, g)` and `apply_player_swap(obs, action, reward, masks)`.
- `src/catan_rl/algorithms/common/rollout_buffer.py` — `get_batches` calls `apply_symmetry` per minibatch with `aug_prob` probability.
- `src/catan_rl/augmentation/player_swap.py` (new) — Z_2 swap utilities.

**D6 group (12 elements):**
- 6 rotations (0°, 60°, 120°, 180°, 240°, 300°).
- 6 reflections (mirror through 6 axes).

**Per-element permutations (precomputed, static):**
- `tile_perm[g]: (19,) int array` — maps tile index `i` to its image under `g`.
- `corner_perm[g]: (54,)` — same for vertices.
- `edge_perm[g]: (72,)` — same for edges.

**Action permutation under D6:**
- Head 0 (type): unchanged.
- Head 1 (corner): `corner_perm[g]`.
- Head 2 (edge): `edge_perm[g]`.
- Head 3 (tile): `tile_perm[g]`.
- Head 4 (resource1): unchanged.
- Head 5 (resource2): unchanged.

**Mask permutation under D6:** same axis as action.

**Z_2 player-swap (1v1 ONLY):**
- `current_player_main ↔ next_player_main`
- `current_player_played_dev ↔ next_player_played_dev`
- `current_player_hidden_dev`: dropped from acting-player obs (becomes opponent's hidden state, which we don't observe). **Limitation:** this means Z_2 swap creates an obs that lacks the new "current" player's hidden cards. This is unavoidable; only use Z_2 in **value-target augmentation** (negate reward target), never in policy gradient.
- Tile-feature ownership bits flip (`is_self ↔ is_opponent`).
- Reward sign flips.
- Acting-player ID embedding flips.

**Integration:**
- D6: applied in `CompositeRolloutBuffer.get_batches` with `aug_prob` (default 0.5) per minibatch. Sample one `g ≠ identity`. Old log-probs are invariant under symmetry permutation (the policy gradient is well-defined since the symmetric action is sampled from the same distribution after permutation).
- Z_2: applied **only** in PPG aux phase (Phase 2.5 Option B), gated by `player_swap_aug_prob` (default 0.25). Used for value-distillation loss with negated target; never for policy gradient.

**[1v1] Validation tests** in `src/catan_rl/ppo/symmetry_tables_test.py`:
1. Apply `g` then `g^-1` to a constructed obs → identity (per-field, including Friendly-Robber mask).
2. Apply 60° rotation visually (render before and after) and confirm the board looks rotated.
3. Apply Z_2 swap to a state with known agent VP=5, opp VP=3 → after swap, agent VP=3, opp VP=5, reward sign flipped.

**Config:** `symmetry_aug_prob: 0.5`, `use_symmetry_aug: True`, `player_swap_aug_prob: 0.25` (active only in PPG aux phase).

### 4.2 Ablation

Run `phase1_full` (all sub-features bundled) vs `phase0_fixed` champion. 5M steps from scratch (Phase 1 obs schema change is incompatible with `phase0_fixed` lineage; new lineage starts here).

If `phase1_full` fails (< 50% WR vs `phase0_fixed`), run leave-one-out 3M-step ablations:
- `phase1_no_value_clip`
- `phase1_no_advantage_norm`
- `phase1_no_thermometer_drop`
- `phase1_no_devcard_count`
- `phase1_no_symmetry_aug`
- `phase1_no_z2_swap` (only meaningful if PPG is also enabled)

### 4.3 Rollback condition

- WR vs `phase0_fixed` at 5M < 50% → leave-one-out, revert offending feature(s).
- Per-head entropy collapses on any head before update 100 → revert advantage-normalization mode change.
- Symmetry-tables unit tests fail → block merge; do not run training until tables fixed.

### 4.4 Compute budget

| Item | Steps | Wall-clock |
|---|---|---|
| `phase1_full` 5M | 5M | 2 d |
| Up to 5 leave-one-out 3M runs (only if needed) | 15M | 6 d |
| Engineering (symmetry, augmentation, obs cleanup) | — | 3 d |
| **Total Phase 1 (expected)** | **5M** | **~5 d** |
| **Total Phase 1 (with leave-one-out)** | **20M** | **~12 d** |

### 4.5 Decision gate

`phase1_full` beats `phase0_fixed` ≥ 55% over 200 deterministic games. Else investigate and fix before Phase 2.

### 4.6 Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | Symmetry tables wrong → silent perf degradation | Unit tests in §4.1.5; visual board render before/after rotation |
| MEDIUM | `bucket8` removal under-fits because input MLPs assumed thermometer width | Re-tune first MLP layer; rely on LayerNorm |
| MEDIUM | Z_2 swap introduces incorrect targets in PPG aux | Use only for value distillation (negated reward); add unit test on a constructed state with known VPs |
| MEDIUM | `aug_prob=0.5` washes out true-board signal early in training | Schedule `aug_prob: 0.0 → 0.5` linearly over first 1M steps |
| LOW | Value clipping with mis-tuned `clip_range_vf` collapses value learning | Standard 0.2 from PPO2 well-tested |

---

## 5. Phase 2 — Architecture Upgrades

**Status (2026-05-02):** all of Phase 2 landed across multiple PRs.
2.1 + 2.2 + 2.4 + 2.5 (Option A) on `feat/phase-2-architecture`; 2.5b
belief head on `feat/phase-2-5b-belief-head`; 2.3 GNN encoder + 2.5c
opp-action auxiliary loss on `feat/phase-2-3-and-phase-4` (alongside
Phase 4 work). See ADRs
[`0007-phase-2-architecture-upgrades.md`](../decisions/0007-phase-2-architecture-upgrades.md)
and [`0009-phase-4-search-and-recurrence.md`](../decisions/0009-phase-4-search-and-recurrence.md).
Configs: `configs/phase2_full.yaml` plus
`phase2_no_{axial_pos, transformer_recipe, film, decoupled_value}.yaml`,
plus `phase2_belief_head.yaml`. Param count:
~1.54M → ~2.22M (phase2_full) → ~2.29M (phase2_belief_head) → see
phase4_full (~2.74M) for the full architectural stack.

**Why third.** With trainer correct (Phase 0) and sample-efficient on existing arch (Phase 1), any architecture gain is real and not eaten by training noise. Adds inductive biases the network lacks: 2D positional structure, explicit GNN over hex/vertex/edge incidence, decoupled value/policy, AdaLN-conditioned heads. Includes 1v1-specific cheap wins (belief head, opponent-action aux loss).

### 5.1 Sub-features

#### 5.1.1 Axial-coordinate positional embedding

**File:** `src/catan_rl/models/observation_module.py`.

**Change.** Each tile has axial coordinates `(q, r)` known at board construction. Add a learned 2D embedding:
```python
self.axial_q_emb = nn.Embedding(11, 25)  # q in [-5, 5] -> shifted [0, 10]
self.axial_r_emb = nn.Embedding(11, 25)
# in forward:
pos = self.axial_q_emb(q_idx) + self.axial_r_emb(r_idx)  # (B, 19, 25)
tile_features_with_pos = torch.cat([tile_features, pos], dim=-1)
```

**Config:** `use_axial_pos_emb: True`, `axial_pos_dim: 25`.

#### 5.1.2 Pre-norm + GELU + dropout in tile transformer

**File:** `src/catan_rl/models/tile_encoder.py`.

**Change.** `nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, activation='gelu', dropout=0.05, norm_first=True, batch_first=True)`. Already has `norm_first=True` per current code; explicitly add `activation='gelu'` and `dropout=0.05`.

**Config:** `transformer_dropout: 0.05`, `transformer_activation: "gelu"`.

#### 5.1.3 Bipartite hex/vertex/edge GNN

**File:** `src/catan_rl/models/graph_encoder.py` (new).

**Architecture.**
- 19 hex nodes + 54 vertex nodes + 72 edge nodes.
- Static incidence relations:
  - hex ↔ vertex (each hex has 6 vertices, each vertex has 1–3 hexes).
  - vertex ↔ edge (each vertex has 2–3 edges, each edge has 2 vertices).
- 2 layers of message passing: each layer does `hex → vertex → edge → vertex → hex` aggregations using simple `Linear + sum + LayerNorm + GELU`.
- Output: enriched 128-dim per-vertex and per-edge embeddings; corner/edge action heads consume these directly (instead of replicating from tile features).

**[1v1]** Constraints:
- Vertex-node features include 3-way ownership (`is_self, is_opponent, none`). Do not generalize to `is_player_k for k in range(N)`.
- Edge-node features carry single-opponent road ownership.
- Do NOT add a "player" node type to the bipartite graph. Single-opponent state stays in `current_player_main`/`next_player_main`, fed to a global-readout MLP.

**Implementation.** Plain matmul-based message passing (no PyG dependency to keep CPU lean).

**Config:** `use_graph_encoder: True`, `graph_n_layers: 2`, `graph_hidden_dim: 128`.

#### 5.1.4 AdaLN-conditioned action heads

**File:** `src/catan_rl/models/action_heads_module.py`.

**Change.** Currently each head's MLP processes `concat(state, type_one_hot_context)`. Replace with AdaLN: `(γ, β) = MLP(action_type_one_hot)`, applied as `h = γ * LayerNorm(x) + β` at each layer of the head's MLP.

**Init.** `γ` initialized close to 1 (gain=0.1 for the γ output); `β` initialized to 0. Prevents instability.

**Config:** `action_head_film: True`.

#### 5.1.5 Decoupled value tower (Option A) OR PPG (Option B)

**Decision.** Try Option A first (1 day to implement, lower risk). If after 5M steps it doesn't improve EV by ≥ 0.05, switch to Option B.

**Option A — Decoupled value tower.**
- File: `src/catan_rl/models/policy.py`.
- Build a second observation encoder + value MLP that does not share weights with the policy encoder. Doubles params (~1.5M → ~3M).
- Forward cost: one extra encoder pass per minibatch.

**Option B — PPG (Phasic Policy Gradient).**
- File: `src/catan_rl/algorithms/ppg/aux_phase.py` (new).
- Every `n_pi=32` PPO updates, run `n_aux=6` epochs of joint (policy distillation + value) loss on a buffer of recent `n_pi * n_steps` obs.
- Distillation loss = KL(π_old || π_new) + MSE(V_aux, V_target).
- Z_2 player-swap augmentation (from Phase 1.5) is active here.

**Config:**
- `value_head_mode: "shared" | "decoupled" | "ppg"`, default `"decoupled"`.
- `ppg_n_pi: 32`, `ppg_n_aux: 6`, `ppg_aux_buffer_size: 32 * n_steps`.

#### 5.1.6 Belief head (1v1-specific, promoted from original Phase 4)

**File:** `src/catan_rl/models/belief_head.py` (new).

**Architecture.** Auxiliary head from the policy encoder's 512-dim output, predicting the opponent's hidden dev-card type distribution (5-way softmax over `KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY`, normalized by count).

**Training.** Cross-entropy loss against the *true* hidden distribution from `opponent_player.devCards + opponent_player.newDevCards` (env knows it; cheating is fine during training). Loss weight 0.05.

**[1v1] Why this works only in 1v1:** with no P2P trade and a single opponent, the broadcast tracker gives perfect resource info, leaving only dev-card type as hidden. With P2P trade, the supervision target is stale (cards may have been traded). With 4 players, the supervision becomes a joint distribution and the head's output dim explodes.

**Config:** `use_belief_head: True`, `belief_loss_weight: 0.05`.

#### 5.1.7 Opponent-action prediction aux loss (1v1-specific cheap leverage)

**File:** `src/catan_rl/models/opponent_action_head.py` (new).

**Architecture.** Auxiliary head from encoder output, predicting the opponent's *next action distribution* (over 13 action types).

**Training.** Cross-entropy against the action actually taken by the opponent on its next turn. Only trained on rollouts where the opponent is a **historical league policy** (different `policy_id` from current main) — never against the current policy itself, to avoid degenerate fixed-point dynamics. Loss weight 0.03.

**Config:** `use_opponent_action_head: True`, `opp_action_loss_weight: 0.03`.

### 5.2 Ablation

`phase2_full` (all of 5.1.1–5.1.7 with Option A for 5.1.5) vs `phase1_full`. 5M steps from scratch.

If `phase2_full` improves: try Option B (`phase2_ppg`) as a follow-up 5M run; keep whichever is better.

If `phase2_full` fails (< 50% WR vs `phase1_full`), leave-one-out 3M runs:
- `phase2_no_axial`, `phase2_no_pre_norm`, `phase2_no_gnn`, `phase2_no_adaln`, `phase2_no_decoupled`, `phase2_no_belief`, `phase2_no_opp_action`.

### 5.3 Rollback condition

- WR vs `phase1_full` < 50% → leave-one-out, revert offending feature(s).
- Param count > 5M → revert decoupled tower to shared encoder + small auxiliary head.
- CPU FPS regresses by > 25% → revert GNN (most likely culprit).
- Belief head training loss > 1.5 nats per token after 1M steps → belief head signal not being captured, investigate.

### 5.4 Compute budget

| Item | Steps | Wall-clock |
|---|---|---|
| `phase2_full` (Option A) 5M | 5M | 2 d |
| Optional `phase2_ppg` (Option B) 5M | 5M | 2 d |
| Up to 7 leave-one-out 3M runs | 21M | 8 d |
| Engineering (graph encoder, AdaLN, decoupled tower, belief, opp-action) | — | 6 d |
| **Total Phase 2 (expected)** | **5–10M** | **~9–13 d** |
| **Total Phase 2 (with leave-one-out)** | **31M** | **~16 d** |

### 5.5 Decision gate

All must pass:
1. WR vs heuristic ≥ 99% on 200 seeds.
2. WR vs `phase1_full` ≥ 55%.
3. WR vs `checkpoint_07390040.pt` ≥ 60% (intermediate target; final 70% gated to Phase 3).
4. EV ≥ 0.85.
5. Belief head loss < 1.5 nats per token by 1M steps.

If WR vs `phase1_full` < 52% after both Option A and Option B: **stop and reassess** — likely architecture is not the bottleneck and Phase 3 (more diverse opponents) is more valuable.

### 5.6 Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | GNN implementation error silently degrades policy | Visualize per-vertex attention weights / message magnitudes after 100k steps; assert non-degenerate distribution |
| HIGH | Decoupled tower doubles forward cost → CPU FPS drops | 100k-step micro-benchmark before committing; if FPS < 25, fall back to PPG |
| MEDIUM | AdaLN unstable at init | γ init gain=0.1; β init=0; LR warm-up first 50k steps |
| MEDIUM | Belief head training overwhelms encoder if loss weight wrong | Ablate belief weight ∈ {0.01, 0.05, 0.1}; pick smallest that produces useful loss curve |
| MEDIUM | Opp-action aux head creates degenerate fixed point if trained vs current policy | Hard exclusion: only train on rollouts with `opponent_policy_id != current_policy_id` |
| LOW | Pre-norm conversion changes effective LR | Re-tune LR if EV regresses |

---

## 6. Phase 3 — Self-Play Upgrades

**Status (2026-05-02):** all 6 sub-features landed (initial PR plus a 3.3
follow-up). 3.1 PFSP-hard, 3.2 latest-policy regularization, 3.4
TrueSkill ratings, 3.5 Nash-weighted pruning, and 3.6 opponent-ID
embedding shipped on `feat/phase-3-self-play-diversity`. 3.3 duo
exploiter cycles shipped on `feat/phase-3-3-duo-exploiter` (snapshot/restore
the trainer in-place, `League.frozen_main_for_exploiter` for the
single-opponent league, `League.add_with_boost` for amplified PFSP
priority on freshly-injected exploiters). See ADR
[`0008-phase-3-self-play-diversity.md`](../decisions/0008-phase-3-self-play-diversity.md).
Configs: `configs/phase3_full.yaml` plus
`phase3_no_{pfsp_hard, latest_policy, trueskill, nash_pruning, opp_id_emb}.yaml`,
plus `phase3_full_with_exploiter.yaml` (turns 3.3 on). Param count:
~2.22M (phase2_full) → ~2.24M (phase3_full).

**Why fourth.** Architectural gains saturate when opponent distribution is too narrow. AlphaStar showed that diverse opponent populations give the largest late-game gains. Phase 3 also delivers the bulk of the **exploitability < 5%** improvement (exploitability *is* a self-play diversity signal).

### 6.1 Sub-features

#### 6.1.1 PFSP-hard

**File:** `src/catan_rl/selfplay/league.py`.

**Change.** Currently `_pfsp_weights` returns `priorities = w(1-w) + ε`. Add `pfsp_mode: "linear" | "hard" | "var"` config; `"hard"` returns `(1 - w)^p` with `p=2`.

This concentrates training on opponents the agent is currently losing to. Per-opponent WR uses a 32-game sliding window per league entry.

**[1v1]** PFSP-hard `(1 - w)^p` is mathematically defined for 2-player zero-sum only. Document this constraint in `league.py` docstring. Add assertion.

**Config:** `pfsp_mode: "hard"`, `pfsp_p: 2.0`, `pfsp_window: 32`.

#### 6.1.2 Latest-policy regularization

**File:** `src/catan_rl/selfplay/league.py`.

**Change.** With probability `latest_policy_weight` (default 0.10), opponent is the *current* policy snapshot (zero-shot copy of `_raw_policy_state_dict()` taken at start of rollout). Prevents drift between league updates.

Implementation: extend `League.sample()` return to include `("current_self", state_dict, -2)` as a special policy_id (-2 = self, -1 = random/heuristic).

**Config:** `latest_policy_weight: 0.10`.

#### 6.1.3 Duo exploiter (1v1-tailored 2-population)

**File:** `src/catan_rl/algorithms/ppo/trainer.py` (interleave directly, no separate `populations.py` module needed).

**[1v1]** AlphaStar's 3-population design (main + main_exploiter + league_exploiter) targets a much larger compute budget. In 2-player zero-sum, the league_exploiter mostly diversifies the historical pool, which we already get from PFSP. Drop league_exploiter; keep main + main_exploiter.

**Cycle structure:**
- Every `T = 1_000_000` main steps, pause main and run a 1M-step exploiter cycle.
- Exploiter is reset to a snapshot of current main, then trained for 1M steps to beat current main (game ratio 1:0.5 — main gets 2/3 of compute).
- After cycle ends, exploiter's final snapshot is added to the regular league as a high-priority entry (PFSP weight × 1.5 for its first 64 games).
- Resume main.

**Config:** `exploiter_mode: "duo" | "off"`, default `"duo"`. `exploiter_cycle_steps: 1_000_000`. `exploiter_priority_multiplier: 1.5`.

#### 6.1.4 TrueSkill league rating

**File:** `src/catan_rl/selfplay/ratings.py` (created in Phase 0; activate full integration here).

**Change.** After every rollout, push observed match outcomes (main vs each sampled opponent) into the rating system. Log `eval/trueskill_main_mu`, `eval/trueskill_main_sigma`, top-K league entries' ratings.

**Decay.** Half-life decay on σ to handle non-stationary policies: `σ ← σ * 1.001` per update.

**Config:** `use_trueskill: True`, `trueskill_decay: 1.001`.

#### 6.1.5 Nash-weighted checkpoint pruning

**File:** `src/catan_rl/selfplay/league.py`.

**Change.** When league hits `maxlen=100`, replace FIFO eviction with Nash pruning:
- Every `prune_every=20` adds, run internal round-robin among top-32 league members (50 games per pair, fast eval, deterministic seeds).
- Compute approximate Nash mixture via replicator dynamics (100 iterations, well-defined for 2-player symmetric zero-sum).
- Drop the entry with lowest Nash support.

**[1v1]** Add assertion in `League.prune_nash()` that population count == 2 (the symmetric pair: main vs opponent slot).

**Config:** `prune_strategy: "fifo" | "nash"`, default `"nash"`. `nash_prune_round_games: 50`. `prune_every: 20`. `nash_top_k: 32`.

#### 6.1.6 Opponent ID embedding

**File:** `src/catan_rl/models/observation_module.py` + `src/catan_rl/env/catan_env.py`.

**Change.** Include 16-dim learned embedding `(opponent_kind_emb + opponent_policy_id_emb)` in `current_player_main`.
- `opponent_kind ∈ {self_latest, league, heuristic, random, main_exploiter, unknown}` — 6-way embedding.
- `opponent_policy_id ∈ {0, ..., league_maxlen}` — embedding lookup.

**Random masking.** During training, with probability `opp_id_mask_prob` (default 0.40), set both to `unknown` token. **[1v1]** Higher than the typical 25% because in 1v1 the eval-time benchmark uses an "unknown" opponent (frozen champion); we need robustness to this distribution shift.

**Deployment:** at eval/champion-bench time, always use `unknown` token.

**Config:** `use_opponent_id_emb: True`, `opp_id_emb_dim: 16`, `opp_id_mask_prob: 0.40`.

### 6.2 Ablation

`phase3_full` vs `phase2_full` champion. 8M steps from scratch (Phase 3 needs longer compute due to exploiter cycles eating throughput).

**Metrics** (all required):
- WR vs `phase2_full` ≥ 55%.
- WR vs `checkpoint_07390040.pt` ≥ 70% (final target).
- WR vs heuristic ≥ 99%.
- Exploitability < 5% (5M-step fresh adversary).
- TrueSkill mean μ vs full league: top-1.

### 6.3 Rollback condition

- Exploitability ≥ 8% after 8M steps → revert exploiters or change cycle ratio (try 1:1 with longer cycles).
- WR vs `phase2_full` < 50% → revert PFSP-hard back to PFSP-linear; retain only latest-policy regularization.

### 6.4 Compute budget

| Item | Steps | Wall-clock |
|---|---|---|
| `phase3_full` 8M (with exploiter overhead) | 8M | 4 d |
| Exploitability harness validation | 5M | 2 d |
| Engineering (PFSP-hard, latest reg, duo exploiter, ratings, Nash, opp-id emb) | — | 4 d |
| **Total Phase 3** | **13M** | **~9 d** |

### 6.5 Decision gate

All success criteria (C1, C2, C3) met → declare project success, freeze, write up.

Else: proceed to Phase 4 selectively based on which gap remains:
- Exploitability ≥ 5%: prioritize Phase 4.3 (belief-conditioned ISMCTS).
- WR vs champion < 70%: prioritize Phase 4.1 (ISMCTS) and 4.2 (GRU value).

### 6.6 Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | Exploiters destabilize main if cycle ratio is wrong | Log main's WR vs `checkpoint_07390040.pt` before/after each exploiter cycle; if drops by > 5 pp, increase main:exp ratio to 3:1 |
| HIGH | Nash pruning round-robin expensive (50 games × 32² / 2 = 25.6k games per prune) | Bound to top-32 by recency; use 50 games per pair; run prune asynchronously off the main rollout |
| MEDIUM | Opponent ID embedding leaks info that doesn't generalize | 40% masking rate; `unknown` token used at eval time |
| MEDIUM | Duo exploiter snapshot pollutes league | Limit to `exploiter_priority_multiplier: 1.5` for first 64 games only; revert to standard PFSP weight thereafter |
| LOW | TrueSkill drift due to non-stationary policies | Half-life decay on σ |

---

## 7. Phase 4 — Optional / Compute-Permitting

**Status (2026-05-02):** 4.1 ISMCTS module + 4.2 GRU recurrent value head
landed on `feat/phase-2-3-and-phase-4` (alongside the Phase 2.3 GNN and
2.5c opp-action items). 4.1's rollout-loop integration is gated on
empirical belief loss < 1.0 nats and is a follow-up. 4.3 PBT is
hardware-gated (requires multi-machine compute) and remains future
work. See ADR
[`0009-phase-4-search-and-recurrence.md`](../decisions/0009-phase-4-search-and-recurrence.md).
Configs: `configs/phase4_full.yaml` plus
`phase4_no_{graph, recurrent_value, opp_action}.yaml`.

**Only run if Phase 3 misses success criteria.** Each item is high-cost-to-implement and may not pay off on CPU-only hardware. Items prioritized by which gap they close.

### 7.1 Sub-features

#### 7.1.1 Belief-conditioned ISMCTS

**File:** `src/catan_rl/algorithms/search/ismcts.py` (new).

**[1v1]** Information-Set MCTS — necessary for hidden information (opponent's hand). In 1v1 with broadcast tracker, the only hidden state is opponent's dev-card types. Specific design:

- Opponent dev-card count is known via tracker; only the *type distribution* is uncertain.
- For each MCTS rollout, sample `D=4` **determinizations** of opponent's hidden dev cards from the **belief head** (Phase 5.1.6).
- Run perfect-information MCTS for `n_sims_per_det=50` per determinization → effective `n_sims=200` total via Cazenave's averaging.
- AlphaZero-style PUCT with policy as prior, value head as leaf evaluator.

**[1v1]** This is **only viable in 1v1**. With P2P trade or 4 players, determinization branching explodes.

**Cost.** ~50ms per MCTS-replaced step on CPU. With `mcts_prob=0.25`, throughput drops to ~10–15 steps/sec. Hard limit; only viable if Phase 3 is close but not closing.

**Action target during PPO update:** visit-count distribution from MCTS, used as cross-entropy target against the policy's stored old log-probs.

**Config:** `use_ismcts: True`, `ismcts_determinizations: 4`, `ismcts_sims_per_det: 50`, `ismcts_prob: 0.25`, `ismcts_c_puct: 1.5`.

**Decision gate within Phase 4:** ISMCTS only run if belief head loss < 1.0 nats per token; otherwise determinization quality too low.

#### 7.1.2 GRU recurrent value head

**File:** `src/catan_rl/models/recurrent_value.py` (new).

**Change.** Add 64-dim GRU on the value head only (policy stays Markovian). Hidden state reset on `terminated`. Buffer stores hidden states + reset markers.

**[1v1]** Especially valuable for stalemate detection: with no P2P trade, resource locks are common; long stalemates often hit `max_turns=500`. A recurrent value head can recognize "this position has been static for K turns" and predict draw → truncation more accurately.

**Config:** `use_recurrent_value: True`, `gru_hidden_dim: 64`.

#### 7.1.3 Population-Based Training (PBT)

**File:** `scripts/pbt.py` (new).

**Hardware-gated.** Requires 4× compute; M1 Pro alone is insufficient. Only run if user has access to additional hardware.

**Design.** 4 concurrent trainers with perturbed hyperparameters (LR, entropy_coef, clip_range). Every 1M steps: exploit (copy weights from best worker) + explore (perturb by ±20%).

**Config:** `pbt_n_workers: 4`, `pbt_perturb_pct: 0.20`, `pbt_exploit_every: 1_000_000`.

### 7.2 Compute budget

| Item | Steps | Wall-clock |
|---|---|---|
| ISMCTS 5M (at 12 steps/sec) | 5M | 5 d |
| GRU value 5M | 5M | 2.5 d |
| PBT 4 workers (hardware-gated) | 4 × 5M = 20M | 5 d serial / 2 d parallel |
| **Total Phase 4 (best 2 of 3)** | **~10M** | **~10–15 d** |

In practice: do at most 2 items based on Phase 3's gap.

### 7.3 Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | ISMCTS inference cost makes training intractable on M1 Pro | Cap `mcts_prob` to 0.10 if FPS < 10 |
| HIGH | GRU breaks rollout-buffer abstraction | Keep recurrent state buffer additive; do not modify existing fields |
| MEDIUM | PBT requires multi-machine orchestration | Hardware-gated; document as "future work" if not available |
| LOW | Determinization sampling biased by belief head errors | Validate belief head accuracy on held-out games before enabling ISMCTS |

---

## 8. Dependency DAG

```
Phase 0 (eval harness + correctness + RULES-INVARIANT TEST)  [BLOCKING]
  │
  ├── REQUIRED for ──> Phase 1, 2, 3, 4
  │
Phase 1 (sample efficiency)         [internal sub-features parallelizable]
  │     1.1 value clip
  │     1.2 advantage norm
  │     1.3 drop bucket8
  │     1.4 dev-card count
  │     1.5 D6 + Z_2 player-swap symmetry
  │
  ├── REQUIRED for ──> Phase 2 (architecture work assumes clean obs)
  │
Phase 2 (architecture)              [2.1, 2.2, 2.4 parallelizable; 2.3 depends on 2.1; 2.5 independent;
  │                                   2.5b/c depend on 2.5 plumbing]
  │     2.1 axial pos-emb
  │     2.2 pre-norm/GELU/dropout
  │     2.3 GNN
  │     2.4 AdaLN heads
  │     2.5 decoupled value / PPG
  │     2.5b BELIEF HEAD (1v1-specific cheap leverage)
  │     2.5c opponent action prediction aux loss (1v1-specific)
  │
  ├── REQUIRED for ──> Phase 3 (exploiters need stable arch)
  │
Phase 3 (self-play)                 [3.1+3.2 first, then 3.3+3.4+3.5 parallel, then 3.6]
  │     3.1 PFSP-hard
  │     3.2 latest regularization
  │     3.3 duo exploiter (1v1-tailored 2-population)
  │     3.4 TrueSkill ratings
  │     3.5 Nash pruning
  │     3.6 opponent ID embed (40% masking)
  │
  ├── (gate) success criteria met? ──> STOP, ship.
  │           else ──> Phase 4 (selective)
  │
Phase 4 (optional)                  [each independent]
        4.1 ISMCTS (gated on belief head quality from 2.5b)
        4.2 GRU value
        4.3 PBT (hardware-gated)
```

**Cross-phase parallelism (engineering-only):**
- Phase 1.5 symmetry tables can be built while Phase 0 runs (engineering, no training).
- Phase 2 engineering (graph encoder, AdaLN, belief head, opp-action head) can begin while Phase 1 ablations train.
- Phase 3 engineering (populations, ratings) can begin while Phase 2 ablations train.
- Phase 4 engineering can begin during Phase 3 if confident Phase 3 will miss target.

---

## 9. Compute Budget Summary

Given M1 Pro at ~30–35 steps/sec ≈ 2–2.5M steps/day:

| Phase | Engineering | Compute (steps) | Wall-clock |
|---|---|---|---|
| 0 | ~1.5–2 d | ~15M | **7–8 d** |
| 1 | ~3.5 d | ~5M (expected) / 20M (with leave-one-out) | **5–12 d** |
| 2 | ~6 d | ~10M (expected) / 31M (with leave-one-out) | **9–16 d** |
| 3 | ~4 d | ~13M | **9 d** |
| 4 | ~7 d (best 2 of 3) | ~10M | **11–15 d if needed** |
| **Total to ship (no Phase 4)** | **~15.5 d** | **~43M** | **~30–45 d** |
| **+ Phase 4 if needed** | **+7 d** | **+10M** | **+11–15 d** |

**Optimistic path:** no leave-one-outs, Phase 4 not triggered → ~30 days.
**Pessimistic path:** all leave-one-outs needed, Phase 4 (best 2) triggered → ~60 days.

---

## 10. Decision Gates

Each phase has a decision gate that must pass before proceeding to the next. Stop-and-reassess points listed below:

- **After Phase 0:** If `phase0_fixed` ≠ `phase0_baseline` within ±2 pp **and** EV did not improve, the bug fix is wrong — debug before proceeding. If `rules_invariants.run()` returns ≥ 1 failure, block all work until fixed.
- **After Phase 1:** If `phase1_full` < 50% WR vs `phase0_fixed` after leave-one-out, **stop and reassess**: probably regression in 1.3 (obs schema) or 1.5 (symmetry tables). Do not proceed to Phase 2.
- **After Phase 2:** If WR vs heuristic < 99% **or** EV < 0.85 **or** belief head loss > 1.5 nats per token, stop and rerun leave-one-out. Do not proceed to Phase 3 — exploiters will silently hide architectural weakness.
- **After Phase 3 (success):** If all three success criteria (C1, C2, C3) met, **declare project success, freeze, write up**. Do not run Phase 4 speculatively.
- **After Phase 3 (failure, 1):** If exploitability ≥ 8% with `1:1` exploiter ratio and 2× cycle length, reconsider whether a fundamentally different approach (MuZero-style learned model) is needed before committing more compute.
- **After Phase 3 (failure, 2):** If WR vs champion < 70% but exploitability < 5%, prioritize Phase 4.1 (ISMCTS) and 4.2 (GRU). If exploitability ≥ 5%, prioritize Phase 4.1 (belief-conditioned ISMCTS) only.

---

## 11. Risks Summary

Highlighted risks across all phases (full per-phase tables in Sections 3.12, 4.6, 5.6, 6.6, 7.3):

| Severity | Phase | Risk |
|---|---|---|
| HIGH | 0 | GAE fix interacts with value normalizer; reset normalizer in `phase0_fixed` |
| HIGH | 1 | Symmetry tables wrong → silent perf degradation; unit tests mandatory |
| HIGH | 2 | GNN implementation error degrades policy silently |
| HIGH | 2 | Decoupled tower halves CPU FPS → fall back to PPG |
| HIGH | 3 | Exploiters destabilize main; monitor WR vs frozen champion before/after every exploiter cycle |
| HIGH | 4 | ISMCTS makes training intractable on M1 Pro; cap `mcts_prob` |
| HIGH | All | Future refactor accidentally generalizes to N-player; rules-invariant test catches this |
| MEDIUM | 0 | Determinism seeding incomplete |
| MEDIUM | 1 | Z_2 swap introduces subtle 1v1 asymmetry bugs |
| MEDIUM | 2 | Opp-action aux head creates degenerate fixed point with current policy |
| MEDIUM | 3 | Nash pruning expensive; bound to top-32 |
| LOW | All | Library version drift; pin in `requirements.txt` |

---

## 12. Suggested Ordering (Maximum Expected Value per Compute-Day)

Priors derived from the literature: PPG ~10% sample-efficiency gain; AlphaStar ~30% Elo gain from leagues; AlphaZero most gain from search at deployment, not training; Charlesworth Catan confirmed value of dict obs + MHA.

1. **Phase 0** — non-negotiable; do first.
2. **Phase 1.5 D6 augmentation only** — ship before Z_2 to isolate the standard piece.
3. **Phase 1.3, 1.4 (obs cleanup, dev-card count)** — straightforward.
4. **Phase 1.5 Z_2 player-swap** — added to PPG aux phase only, low risk.
5. **Phase 1.1, 1.2 (value clip, advantage norm)** — bundle into `phase1_full`.
6. **Phase 2.5b belief head** — fastest 1v1-specific architectural win.
7. **Phase 2.1, 2.5 (Option A) (axial pos-emb, decoupled tower)** — biggest single-shot wins.
8. **Phase 2.5c opp-action aux** — once 2.5b validates plumbing.
9. **Phase 2.3 GNN, 2.4 AdaLN, 2.2 pre-norm/dropout** — refinements.
10. **Phase 3.1, 3.2 (PFSP-hard, latest reg)** — surprisingly cheap and proven impactful.
11. **Phase 3.3 (duo exploiter)** — heavy lift but highest exploitability win.
12. **Phase 3.4, 3.5, 3.6 (TrueSkill, Nash pruning, opp-id embed)** — incremental.
13. **Phase 4** — only if Phase 3 misses targets. Belief-conditioned ISMCTS first, GRU second.

---

## 13. File List

> Paths reflect the **post-restructure** src-layout (see [`file_layout_restructure.md`](file_layout_restructure.md) and [ADR 0006](../decisions/0006-src-layout-restructure.md)).

### 13.1 Files to modify (existing)

Listed by phase that first touches them. Phase 0 touches the most because Phase 0 is the structural foundation.

| Phase | File |
|---|---|
| 0 | `src/catan_rl/env/catan_env.py` (terminated/truncated split, deterministic seeding) |
| 0 | `src/catan_rl/algorithms/common/gae.py` (GAE truncation fix) |
| 0 | `src/catan_rl/algorithms/common/rollout_buffer.py` (terminated/truncated arrays) |
| 0 | `src/catan_rl/algorithms/ppo/trainer.py` (per-head entropy logging, truncated handling) |
| 0 | `src/catan_rl/eval/evaluation_manager.py` (deterministic seeds, H2H, exploitability) |
| 0 | `src/catan_rl/algorithms/ppo/arguments.py` (Phase 0 config keys) |
| 0 | `configs/_base.yaml`, `configs/phase0_*.yaml` (Phase 0 config keys mirrored in YAML) |
| 0 | `src/catan_rl/models/policy.py` (return_per_head flag) |
| 0 | `src/catan_rl/models/action_heads_module.py` (per-head log_prob/entropy tuples) |
| 0 | `src/catan_rl/models/utils.py` (OBS_TILE_DIM, CURR_PLAYER_DIM, NEXT_PLAYER_DIM constants) |
| 0 | `README.md`, `docs/obs_schema.md` (reconcile obs schema) |
| 1 | `src/catan_rl/env/catan_env.py` (drop bucket8 in `_build_*_main`) |
| 1 | `src/catan_rl/models/observation_module.py` (input dim changes from bucket8 removal) |
| 1 | `src/catan_rl/models/player_modules.py` (replace dev-card MHA with count encoding) |
| 1 | `src/catan_rl/algorithms/common/rollout_buffer.py` (per-rollout adv norm; symmetry aug in `get_batches`) |
| 1 | `src/catan_rl/algorithms/ppo/trainer.py` (value clipping; per-rollout adv norm wiring) |
| 1 | `src/catan_rl/algorithms/ppo/arguments.py` + `configs/phase1_*.yaml` (Phase 1 keys) |
| 2 | `src/catan_rl/models/observation_module.py` (axial pos-emb) |
| 2 | `src/catan_rl/models/tile_encoder.py` (pre-norm/GELU/dropout) |
| 2 | `src/catan_rl/models/policy.py` (decoupled value or PPG aux phase) |
| 2 | `src/catan_rl/models/action_heads_module.py` (AdaLN conditioning) |
| 2 | `src/catan_rl/algorithms/ppo/trainer.py` (PPG aux phase if Option B; aux loss wiring for belief and opp-action heads) |
| 2 | `src/catan_rl/algorithms/ppo/arguments.py` + `configs/phase2_*.yaml` (Phase 2 keys) |
| 3 | `src/catan_rl/selfplay/league.py` (PFSP-hard, latest reg, Nash pruning) |
| 3 | `src/catan_rl/algorithms/ppo/trainer.py` (duo exploiter interleaving) |
| 3 | `src/catan_rl/selfplay/game_manager.py` (opponent kind/ID tracking for embedding) |
| 3 | `src/catan_rl/env/catan_env.py` (opponent ID embedding in obs) |
| 3 | `src/catan_rl/models/observation_module.py` (opponent ID embedding integration) |
| 3 | `src/catan_rl/algorithms/ppo/arguments.py` + `configs/phase3_*.yaml` (Phase 3 keys) |

### 13.2 Files to create

| Phase | File | Purpose |
|---|---|---|
| 0 | `src/catan_rl/eval/rules_invariants.py` | 1v1 rules-invariant test (runtime callable; complement to `tests/unit/eval/test_rules_invariants.py` already shipped) |
| 0 | `src/catan_rl/selfplay/ratings.py` | TrueSkill / Glicko-2 wrapper |
| 0 | `src/catan_rl/eval/champion_bench.py` | Frozen-champion H2H benchmark |
| 0 | `src/catan_rl/eval/exploitability.py` | Fresh-adversary exploitability runner |
| 0 | `scripts/eval_harness.py` | Main eval harness CLI (calls into `catan_rl.eval.*`) |
| 0 | `scripts/migrate_checkpoint.py` | One-shot pre-Phase-0 checkpoint translator |
| 1 | `src/catan_rl/augmentation/symmetry_tables.py` | Precomputed D6 permutations |
| 1 | `src/catan_rl/augmentation/dihedral.py` | D6 group ops; `apply_symmetry` |
| 1 | `src/catan_rl/augmentation/player_swap.py` | Z_2 player-swap utilities (1v1-specific) |
| 1 | `src/catan_rl/augmentation/__init__.py` | Module init |
| 1 | `tests/unit/augmentation/test_symmetry_tables.py` | Unit tests for D6 + Z_2 invariance |
| 2 | `src/catan_rl/models/graph_encoder.py` | Bipartite hex/vertex/edge GNN |
| 2 | `src/catan_rl/models/belief_head.py` | Opponent dev-card belief head (1v1-specific) |
| 2 | `src/catan_rl/models/opponent_action_head.py` | Opponent action prediction aux head (1v1-specific) |
| 2 | `src/catan_rl/algorithms/ppg/aux_phase.py` | PPG aux phase trainer (Option B) |
| 2 | `src/catan_rl/algorithms/ppg/__init__.py` | Module init |
| 4 | `src/catan_rl/algorithms/search/__init__.py` | Module init |
| 4 | `src/catan_rl/algorithms/search/ismcts.py` | Belief-conditioned ISMCTS |
| 4 | `src/catan_rl/models/recurrent_value.py` | GRU value head |
| 4 | `scripts/pbt.py` | Population-based training orchestrator |

### 13.3 Files NOT to create

- ~~`src/catan_rl/selfplay/populations.py`~~ — original plan had this for AlphaStar 3-population; **removed** in favor of in-`trainer.py` interleaving for the simpler 2-population duo exploiter.
- ~~`src/catan_rl/models/belief_head.py` in Phase 4~~ — promoted to Phase 2.5b; not in Phase 4.

### 13.4 Configs

Each ablation gets its own YAML config under `configs/`. Naming convention: `phaseN_<feature>.yaml`. All inherit from `_base.yaml`.

| Phase | Config |
|---|---|
| 0 | `configs/phase0_baseline.yaml`, `configs/phase0_fixed.yaml`, `configs/eval_harness.yaml` (already shipped during restructure) |
| 1 | `configs/phase1_full.yaml` and leave-one-out variants `phase1_no_value_clip.yaml`, `phase1_no_advantage_norm.yaml`, `phase1_no_thermometer_drop.yaml`, `phase1_no_devcard_count.yaml`, `phase1_no_symmetry_aug.yaml`, `phase1_no_z2_swap.yaml` |
| 2 | `configs/phase2_full.yaml`, `phase2_ppg.yaml`, plus 7 leave-one-out variants |
| 3 | `configs/phase3_full.yaml` |
| 4 | `configs/phase4_ismcts.yaml`, `phase4_gru.yaml`, `phase4_pbt.yaml` |

### 13.5 Tests

Each new module gets a unit test under `tests/unit/<module>/`, mirroring the source path. Already-shipped Phase 0 tests:

- `tests/unit/engine/test_dice.py`
- `tests/unit/algorithms/test_gae.py` (includes xfail for the post-Phase-0 truncation contract)
- `tests/unit/eval/test_rules_invariants.py`

Phase 0 promotes the rules-invariants tests to also run as a runtime gate via `scripts/eval_harness.py --mode rules-invariant`.

---

## 14. Glossary

- **C1, C2, C3** — Success criteria (Section 1.2).
- **D6** — Dihedral group of order 12; symmetry group of the regular hexagon.
- **Z_2** — Cyclic group of order 2; the 1v1 player-swap symmetry.
- **PPG** — Phasic Policy Gradient (Cobbe et al. 2020). Auxiliary value-distillation phase.
- **PFSP** — Prioritized Fictitious Self-Play (AlphaStar). Sample opponents weighted by `w(1-w)` (linear) or `(1-w)^p` (hard).
- **GAE** — Generalized Advantage Estimation (Schulman et al. 2015).
- **AdaLN** — Adaptive LayerNorm (LLaMA-style conditioning).
- **FiLM** — Feature-wise Linear Modulation.
- **EV** — Explained Variance, value-network diagnostic.
- **KL** — Kullback-Leibler divergence between old and new policy distributions.
- **ISMCTS** — Information-Set Monte Carlo Tree Search (handles hidden information).
- **PUCT** — Predictor + Upper Confidence Bound for Trees (AlphaZero's MCTS variant).
- **TrueSkill** — Microsoft's Bayesian skill rating system.
- **Nash mixture** — equilibrium mixed strategy in 2-player zero-sum.
- **Phase X.Y.Z** — Phase X, sub-feature Y, sub-sub-feature Z (e.g., 5.1.6 = belief head).
- **Champion bench** — H2H eval vs `checkpoint_07390040.pt` over 200 deterministic seeds.
- **Exploitability** — final WR of a fresh 5M-step adversary trained against the frozen champion.

---

## 15. Implementation Notes

### 15.1 Branching strategy

One feature branch per phase. Branch names follow project convention (`<type>/<kebab-slug>`):
- `feat/phase-0-eval-harness`
- `feat/phase-1-sample-efficiency`
- `feat/phase-2-architecture`
- `feat/phase-3-self-play`
- `feat/phase-4-mcts` (only if needed)

Each phase merges to `main` only after passing its decision gate.

### 15.2 Documentation updates

Update with each phase merge:
- `README.md` — section on whichever subsystem changed.
- `~/.claude/projects/-Users-benjaminli-my-projects-catan-rl/memory/MEMORY.md` — current hyperparameters, current architecture summary.
- `CLAUDE.md` (project-level) — new file references, new conventions, new constraints.

Do **not** create new top-level `.md` files speculatively. Update this roadmap in place when implementation diverges from the plan.

### 15.3 Test discipline

Every new module gets a unit test before merge:
- `tests/test_symmetry_tables.py` (Phase 1.5)
- `tests/test_belief_head.py` (Phase 2.5b)
- `tests/test_opponent_action_head.py` (Phase 2.5c)
- `tests/test_ratings.py` (Phase 0/3)
- `tests/test_rules_invariants.py` (Phase 0)
- `tests/test_eval_harness.py` (Phase 0)
- `tests/test_ismcts.py` (Phase 4)

### 15.4 Compute observability

Every training run logs to `runs/<phase>/<run_name>/` with:
- TensorBoard scalars (cumulative, additive only).
- `config.json` snapshot of the full config dict.
- `git_sha.txt` of the commit used.
- `eval_seeds.json` snapshot of the deterministic seed list used by harness.

This makes each ablation reproducible from artifacts.

### 15.5 Handover

If implementation is paused mid-phase, leave behind:
- A short "WHERE I LEFT OFF" entry at the top of the relevant phase section.
- Latest checkpoint and TB log.
- Most recent `git status` output if the branch is mid-edit.

---

## 16. Out-of-Scope Reminders

- **4-player Catan** — see Section 2 invariants. Hard scope boundary.
- **P2P trading** — hard-disabled in engine; do not re-enable.
- **Custom maps / expansions** — out of scope.
- **Browser integration** — out of scope.
- **GPU-only kernels** — CUDA path opt-in only; CPU baseline must never regress.
- **Big-bang merges** — no PR may span more than one phase.

---

**End of roadmap.** Reconcile file paths with the planned restructure before beginning Phase 0.
