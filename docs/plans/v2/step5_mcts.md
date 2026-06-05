# v2 Step 5 — AlphaZero-style PUCT MCTS with chance + belief determinization (Phase B)

**Status**: design locked (panel + faculty carry-forward 2026-05-13); implementation gated on the §0 preflight gates and on Step 4's anchor checkpoint clearing Gates 1-3.

**Revision history**:
- 2026-05-13 — Original draft. Phase B specification per `v2_design.md` §3.5 + §5 Step 5 + faculty re-review framing as ISMCTS-style policy improvement (Cowling, Powley, Whitehouse 2012) on top of a strong learned prior, **not** Nash convergence (which would require CFR-family algorithms — explicitly out of v2 scope per the faculty re-review).

**Preflight gate** (per `v2_design.md` §0 + carry-forward from Step 4): Step 5 implementation does **not** start until:
  - Step 4 PPO has converged with Gates 1 + 2 + 3 green (per `v2_step4_ppo.md` §6). The Step-4 anchor checkpoint becomes the MCTS prior (policy head) + leaf evaluator (value head) + determinization prior (belief head).
  - Belief-head calibration probe (§0.4 below) passes: `KL(belief_pred || env_GT) ≤ 0.35` averaged over a 1024-state held-out batch. Without this, determinization sampling is misleading and the Cicero-panelist KEY FLIP (sample from belief, not uniform) flips in the wrong direction — see §8 Risk register.
  - `CatanGame.copy()` performance probe passes: end-to-end copy cost < 5 ms (M1 Pro CPU baseline). Established Phase 1.2 in v2 ships 14 unit tests pinning state independence + cycle preservation, but wall-clock cost was not measured.

This doc is the planning equivalent of `v2_step3_bc.md` and `v2_step4_ppo.md` — it specifies what gets built, how it's tested, what numbers count as success, and where the risks are. Pulls from `v2_design.md` §3.5, §3.7 (panel votes D1/D12/D13), and §5 Step 5.

## Inputs

- `checkpoints/ppo/<run_id>/checkpoint_*.pt` — the Step-4 anchor. Must clear `v2_step4_ppo.md` §6 Gates 1 + 2 + 3.
- `CatanPolicy` from Step 2/3/4 with three heads in use here:
  - **policy head** → MCTS prior `P(a|s)` per action type (head 0).
  - **value head** → leaf evaluator `V(s)` at non-terminal expansion.
  - **belief head** → 5-way distribution over opponent's hidden dev-card type, used as the determinization sampler.
- `catanGame.copy()` from Phase 1.2 — wraps `deepcopy` with state-independence and cycle-preservation tests already in place.
- `CatanEnv` (Phase 1.5) with `compute_action_masks(state) → dict[str, Tensor]` already standalone since the BC refactor.
- `BroadcastHandTracker` (perfect 1v1 opp resource tracking — visible state is fully observable except dev-card *type*; this is the only hidden axis the belief head models).
- `StackedDice` (Phase 1.1) with current bag state accessible — used for chance-node weighting. Phase 1.1 verified the dice bag is a plain list that `deepcopy` rounds through cleanly.

## Outputs

- `checkpoints/mcts/<run_id>/checkpoint_*.pt` — periodic policy snapshots, AlphaZero-style trained against MCTS visit-count targets.
- `runs/mcts/<run_id>/` — TensorBoard logs (PUCT statistics, visit entropy, tree depth, sims/sec, league composition under MCTS, WR vs Step-4-policy-alone baseline).
- `runs/mcts/<run_id>/league.json` — final league state (MCTS-trained).
- `runs/mcts/<run_id>/eval/`:
  - `mcts_vs_policy.json` — A/B head-to-head, 200 games × N=3 seeds.
  - `mcts_vs_heuristic.json` — symmetrized WR per seat.
  - `mcts_vs_step4.json` — champion regression.
  - `mcts_ppo_br_gap.json` — fresh 1M-step BR adversary.
- Acceptance criterion (§6) gates handoff to the eventual paper/release artefact.

---

## 0. Preflight gates (block Step 5 start)

Four checks. **All four pass or Step 5 does not start.** This is the analogue of `v2_design.md` §0 for Phase B; it calibrates the plan against measurement, not rhetoric.

### 0.1 — Step-4 anchor passed Gates 1 + 2 + 3

Per `v2_step4_ppo.md` §6:
- Symmetrised WR ≥ 0.70 vs heuristic at 200-game eval (Gate 1).
- 1M-step PPO-BR-gap ≤ 0.65 (Gate 2).
- Symmetrised WR ≥ 0.60 vs v1 champion `checkpoint_16162816.pt` (Gate 3).

If any of these failed at Step 4, the Step-4 risk register + diagnosis ladder applies; Step 5 does not start.

### 0.2 — `CatanGame.copy()` performance probe

**Question**: is `CatanGame.copy()` fast enough for sustained MCTS training? 50 sims × ~depth-30 average × ~200 decisions/game means ≈ 300k copy calls per game. At 5 ms/copy = 25 min wall-clock per game, which is infeasible; at 1 ms/copy = 5 min, marginal; at 0.2 ms/copy = 60 s, fine.

**Method**: profile 1,000 mid-game `copy()` calls (sampled from heuristic-vs-heuristic rollouts at varying game stages: setup, early-build, mid-game, late-game with dev-cards in hand). Report mean + p95 wall-clock per copy.

**Decision rule**:
- **mean ≤ 1 ms, p95 ≤ 3 ms** → green; proceed at the full `n_sims_per_det = 25, n_determinizations = 2` budget.
- **mean 1–3 ms, p95 ≤ 5 ms** → yellow; halve the sims budget to `n_sims_per_det = 12, n_determinizations = 2` (24 effective sims) and document the perf gap as a Phase B v2.1 follow-up.
- **mean > 3 ms or p95 > 10 ms** → red; **STOP** and either port the copy hot path to Cython / Rust (1-2 weeks of dev) or shelve Phase B until that ships.

The probe runs in `tests/perf/test_catan_copy_perf.py` (new) and prints to TensorBoard `eval/copy_ms_mean` + `eval/copy_ms_p95`.

### 0.3 — `StackedDice` bag-state accessibility

**Question**: can we read the *remaining-bag* distribution from a copied `CatanGame` without mutating the bag? Chance-node weights come from this distribution (panel D13: NOT the i.i.d. 2d6 prior).

**Method**: copy a `CatanGame` mid-rollout (post-roll-7, with bag partly consumed). Read the bag's remaining count of each sum. Verify the read does not advance the bag's RNG cursor. Roll forward on both the original and the copy; verify identical sequences.

**Decision rule**: pass iff bag-state read is side-effect-free AND identical sequences on copy. Phase 1.1 already verified `deepcopy` round-trips the bag cleanly; this probe is the search-side check.

### 0.4 — Belief-head calibration probe

**Question**: is the Step-4 policy's belief head well-calibrated enough that sampling from it gives a *better* determinization than the uniform-prior baseline? Cicero-panelist's KEY FLIP correction turns on this answer.

**Method**: this is the Phase B analogue of E0.4 in the BC plan. Sample 1,024 mid-game states from a heuristic-vs-step4-anchor mixed-distribution rollout (NOT pure-Step-4-vs-Step-4 — that would measure on the training distribution; NOT pure-heuristic — that would measure on the wrong distribution). For each:
- Compute the env's ground-truth opp dev-card type counts (the same `obs['belief_target']` the BC + PPO heads were trained on).
- Compute the policy's predicted distribution.
- Compute `KL(belief_pred || env_GT)` and `KL(uniform || env_GT)` per state; report mean + the difference.

**Decision rule**:
- `mean KL(belief_pred || env_GT) ≤ 0.35` AND the belief KL is at least 0.10 better than uniform → pass. Sample from belief head during determinization.
- `0.35 < mean KL ≤ 0.70` → yellow; ship `belief_kl_warning=True` through to the search log, but proceed with belief-head sampling.
- `mean KL > 0.70` OR belief KL no better than uniform → **fail**. Either retrain the Step-4 belief head (raise `belief_loss_weight` in PPO config and run one more seed) OR fall back to uniform-prior determinization for Phase B with a TensorBoard scalar `mcts/belief_fallback=1`. Do NOT mix the two within a single run.

The probe runs in `tests/integration/test_belief_calibration.py` (new) and writes a JSON summary alongside the Step-4 evaluation artefacts.

---

## 1. Search algorithm (per `v2_design.md` §3.5, panel-revised)

### 1.1 PUCT formula

For decision nodes (player-to-move):
```
a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]
```
- `c_puct = 1.5` (AlphaZero default; panel D12).
- `P(s,a)` is the policy head's prior over **action type** (the 13-way head only — sub-head priors for corner/edge/tile/res1/res2 are computed at expansion time via the same masked-categorical factorisation the rollout policy uses).
- `Q(s,a)` is the mean backed-up value, in `[-1, +1]` (terminal signed margin) clamped at non-terminal value-head bootstraps.
- `N(s,a)` is the visit count for child `a`.

**Dirichlet noise at the root** (panel D12, AlphaZero default):
```
P(s_root, a) ← (1 − ε) · P(s_root, a) + ε · Dir(α)_a
```
- `α = 0.3`, `ε = 0.25`. Applied **only** at root, only at training time; eval disables noise (deterministic search).

### 1.2 Chance node fan-out

Per panel D13 (5-0 vote, faculty-confirmed): use **6 compressed buckets**, NOT the naive 11-way 2d6 fan-out and NOT the 36-way bag expansion. Each bucket's weight comes from the **remaining-bag distribution** of `StackedDice` at the chance node's parent state.

| Bucket | Sums | Why this grouping |
|---|---|---|
| `low` | 2, 3 | Low-production, similar resource impact |
| `low-mid` | 4, 5 | Moderate production, both common |
| `six` | 6 | Singleton — 6/8 are the dominant pips |
| `seven` | 7 | **Singleton** — triggers discard + robber + Karma 20% override; cannot be merged |
| `mid-high` | 8, 9 | Symmetric to 4-5 |
| `high` | 10, 11, 12 | Low-production, similar resource impact |

Bucket weight at expansion:
```
w_b = Σ_{s ∈ b} count_remaining(s) / total_remaining
```
where `count_remaining(s)` is read from the live bag (preflight 0.3 verifies side-effect-freeness). The 7-bucket weight is **further adjusted** for the persistent Karma 20%-forced-7 buff when `karma_buff_active(current_player) = True`, where:

```
karma_buff_active(p) = (game.last_player_to_roll_7 is not None) AND (game.last_player_to_roll_7 != p)
```

i.e., the most recent 7 in the game was rolled by some player *other than* `p`. The buff persists across turns until `p` themselves rolls a 7 (which flips the predicate). The chance-node weight adjustment:
```
w_seven ← w_seven + 0.20 · (1 − w_seven)   # mass shifted from non-7
w_other ← w_other · (1 − 0.20)              # for other ∈ {low, low-mid, six, mid-high, high}
```
The bag-noise swap (1 random non-7 swap per game) is **not** modelled separately — its effect is absorbed into the remaining-bag count at the moment of read.

**Common-error guard**: do **not** read `karma_buff_active` as "was the previous roll a 7." The engine's `last_player_to_roll_7` is only updated when a 7 is rolled — it does NOT reset between turns. A non-7 roll leaves it unchanged. The buff therefore covers as many turns as needed for the buffed player to actually roll a 7. See `docs/1v1_rules.md` "Karma mechanic note" for the canonical definition.

**Expected-value backup** at chance nodes:
```
V(chance) = Σ_b w_b · V(child_b)
```
Visit counts are updated proportionally:
```
N(chance, b) ← floor(N(chance) · w_b)   # rounding handled by residual to highest-weight bucket
```

### 1.3 Belief-determinization

Per panel D13 + Cicero KEY FLIP correction: at the **root** (the player-to-move's belief over the opponent's hidden hand), sample `n_determinizations = 2` opponent dev-card hand realisations.

**Sampler**:
1. Read the policy's belief head output for the current obs: `b ∈ R^5` over `{KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY}`.
2. Compute the opponent's *known* unplayed dev-card count `K` from `BroadcastHandTracker` (total drawn − total played-and-revealed).
3. Sample `K` dev-card types i.i.d. from `b` (multinomial). This yields a concrete `(count_K, count_VP, count_RB, count_YOP, count_MONO)` tuple with `sum = K`.
4. Hydrate a `catanGame` copy with this concrete opp hand.

**Fallback to uniform** (gated by preflight 0.4 result): if the belief KL gate flagged yellow/red, replace `b` with `uniform(5)` and tag the run with `mcts/belief_fallback=1`.

**Aggregation across determinizations**: run the full PUCT tree independently per determinization (sharing no tree state — visit counts do not transfer). Final visit-count distribution is the **mean** across the `n_determinizations` trees:
```
π_mcts(a | s_root) ∝ (1 / D) · Σ_d N_d(s_root, a)
```
Effective sims per decision = `n_sims_per_det · n_determinizations` = `25 · 2 = 50`.

### 1.4 Per-decision tree allocation

**No tree reuse across decisions.** Chance node expansions and belief-determinization sampling invalidate the tree at each true state transition (the realised dice + the realised opp hand were both random variables in the prior tree). Fresh tree per decision keeps the implementation simple and matches the 50-sim budget cap.

Tree memory budget per decision: 50 sims × average branching ~30 × mean depth ~10 = ~15k nodes per tree. At ~256 bytes per node (Q, N, P, child-pointer-list, parent-pointer), ~4 MB per tree. With 16 parallel envs at training time → 64 MB peak. Comfortably within M1 Pro / A100 RAM. Depth-limited at 40 plies to bound worst case.

### 1.5 Terminal value

- **True terminal** (`game.is_done()`): the player-to-move's signed margin in `[-1, +1]`:
  ```
  z = (vp_self - vp_opp) / 15   # clipped to [-1, +1]
  ```
  At a 15-VP target, `vp_self == 15 AND vp_opp ≤ 11` is the typical win; `(15 - 11) / 15 ≈ 0.27`. Win-only signal would be `±1`; the margin scaling keeps the reward shape consistent with Step-4's faculty-corrected terminal reward (`±1 + (vp_diff)/15`).
- **Non-terminal leaf** (depth limit reached or new node expansion): `V(s) ← value_head(obs)`. Clamped to `[-1, +1]`.

### 1.6 Action masking inside search

Reuse `catan_rl.env.masks.compute_action_masks(state) → dict[str, Tensor]` — already standalone since the BC refactor. Inside the search, **action masks must be recomputed at every expansion**; caching them across copies would silently break under late-game state transitions (the failure mode that bit `bc/dataset.py`). Tested in `tests/unit/search/test_engine_simulator.py`.

The prior `P(s, a)` is the policy head's **masked** distribution — illegal action types receive zero mass and no PUCT visits.

---

## 2. Training under MCTS

### 2.1 AlphaZero-style loss

```
L_AZ = CE(π_mcts(· | s) || π_logits(· | s)) + 0.5 · MSE(z_terminal || V(s))
```
- `π_mcts` is the visit-count distribution from §1.3, with temperature applied per §2.3.
- `π_logits` is the (current, online) policy's type-head logits.
- `z_terminal` is the realised terminal value of the trajectory (using the same `±1 + (vp_diff)/15` shape as Step 4).
- `V(s)` is the value head's prediction at `s`.

This **replaces** PPO's clipped policy-ratio loss during Phase B. The piKL anchor (Step 4) is **dropped** at Phase B start — by construction the Step-4 checkpoint is already strong enough to be the prior; layering piKL on top of MCTS targets would double-anchor the policy. Belief and opp-action auxiliary heads continue to train with the same weights (`0.05` and `0.03` respectively).

### 2.2 Self-play loop

Sample actions from `π_mcts` (the visit-count distribution), NOT from `π_logits` directly. This is the AlphaZero policy improvement mechanism — the network's prior is updated *toward* the search-improved distribution.

PFSP-hard league from Step 4 continues, but **both sides use MCTS** at decision time. Opponent inference cost roughly doubles per decision under MCTS; budget accordingly.

### 2.3 Temperature schedule

| Phase | Temperature τ | Effect |
|---|---|---|
| Training (full game) | 1.0 | `π_mcts(a) ∝ N(a)^(1/τ) = N(a)` — sample proportional to visits |
| Eval | 0.1 | Near-argmax over visits |
| Final 10 plies of training games | 0.1 | Switch to near-argmax once a game is close to terminal — matches AlphaZero (Silver et al. 2018) |

### 2.4 Compute budget warning (key block)

M1 Pro CPU-only is **almost certainly infeasible** for sustained MCTS training: 50 sims × N envs × policy forward-pass-per-sim × tree-traversal-overhead → ~10-50× slower than policy-alone PPO. Even at v1's measured 25 FPS (policy-alone), MCTS training brings the effective FPS to ~0.5-2.5 FPS, which means ~30 days for the same 10M-env-step run that Step 4 fit in 3-7 days.

**Explicit requirement**: ≥ 1 week of **A100** cloud time. Without A100 access, Phase B is queued.

**Fallback if A100 is unavailable**:
1. Ship the Step-4 policy-alone result as the v2 deliverable.
2. Queue Phase B (this plan) for whenever compute lands.
3. Run only the §0 preflight gates + the §6 `mcts_vs_policy.json` A/B test in §STOP/RESUME below — this requires ~6 hours of M1 Pro and answers "does MCTS earn its cost in this codebase, in principle" without paying the full training tax.

---

## 3. File layout (new code)

```
src/catan_rl/search/
├── __init__.py
├── puct_node.py             Node class:
│                              - statistics (Q, N, P, parent, children)
│                              - select_child(c_puct) — UCB1+P argmax
│                              - expand(prior, mask) — child instantiation
│                              - backup(value) — incremental Q/N update
├── chance_node.py           ChanceNode subclass:
│                              - 6-bucket fan-out per §1.2
│                              - weighted expected-value backup
│                              - bucket-weight recomputation from
│                                StackedDice bag at parent state
├── belief_determinization.py
│                            sample_opp_hand(policy, obs, K) → dev-card
│                              type counts; fallback-to-uniform path.
├── engine_simulator.py      EngineSimulator(catanGame) — search-side
│                              wrapper around CatanGame.copy() + step:
│                              - copy() → independent sim state
│                              - step(action) → (next_state, reward,
│                                terminated, dice_was_rolled,
│                                opp_hand_was_drawn)
│                              - get_action_masks() — recomputes via
│                                catan_rl.env.masks
│                              - get_dice_bag_dist() — side-effect-free
│                                read of StackedDice bag
├── tree_search.py           run_search(root_state, root_obs, root_masks,
│                              policy, n_sims, n_dets) → visit_counts
│                              (13,):
│                              - sample n_dets belief realisations
│                              - run PUCT tree per realisation
│                              - average visit counts
│                              - apply Dirichlet noise at root (training)
└── callbacks.py             prior_from_policy(policy, obs, masks) →
                              (priors_dict, value);
                              value_from_policy(policy, obs) → V.
                              Both batched-friendly.

src/catan_rl/training/
├── mcts_trainer.py          MctsTrainer (analogue of CatanPPO):
│                              - replaces PPO policy loss with AZ CE
│                              - reuses CompositeRolloutBuffer + GAE
│                                paths only for value targets at non-
│                                terminal leaves
│                              - league + PFSP-hard from Step 4 carry
│                                forward; both sides invoke run_search()
└── mcts_self_play.py        Per-env decision loop:
                              - generate obs + masks
                              - run_search() → visit counts
                              - sample action ~ visits^(1/τ)
                              - apply, log (obs, visits, value_target)

scripts/
├── train_mcts.py            CLI over mcts_trainer.train(...).
├── evaluate_mcts.py         CLI for §6 acceptance gates.
└── profile_search.py        Preflight 0.2 + 0.3 perf probes.

configs/
├── mcts.yaml                Main MCTS config (per §1, §2).
└── ablations/
    ├── phase_b_no_search.yaml        Policy-only baseline (§5).
    ├── phase_b_full_chance.yaml      11-way chance fan-out (§5).
    ├── phase_b_more_sims.yaml        200 effective sims (§5).
    └── phase_b_uniform_belief.yaml   Uniform-prior determinization (§5).

tests/
├── unit/search/
│   ├── test_puct_node.py
│   ├── test_chance_node.py
│   ├── test_belief_determinization.py
│   ├── test_tree_search.py
│   └── test_engine_simulator.py
├── perf/
│   └── test_catan_copy_perf.py        Preflight 0.2.
└── integration/
    ├── test_az_smoke.py               Random-init policy + MCTS →
    │                                    legal actions over 100 sims.
    ├── test_belief_calibration.py     Preflight 0.4.
    └── test_dice_bag_side_effect.py   Preflight 0.3.
```

---

## 4. Testing (TDD-first, per Step 3 discipline)

Tests are written **before** implementation, per the user's established preference in Step 3. The patterns below target the failure modes that bit `bc/dataset.py` (silent action filtering) and the Phase 1.2 `copy()` invariants.

### 4.1 `tests/unit/search/test_puct_node.py`

- **UCB1+P math**: build a Node with known `Q`, `N`, `P` over 3 children; assert `select_child(c_puct=1.5)` returns the analytic argmax. Vary `c_puct ∈ {0.0, 1.5, 10.0}` to verify monotonic exploration-exploitation behaviour.
- **Backup arithmetic**: 10 sequential backups of values `[0.2, -0.4, 0.6, ...]`; assert `Q == mean(values)` to floating-point precision.
- **Expand-then-select roundtrip**: expand with a prior distribution over a masked action set; assert no masked-out child is ever selectable.
- **Zero-visit child preference**: when one child has `N=0` and others have `N>0`, the zero-visit child must be selected first (UCB1+P is infinite at N=0; confirms numerical handling).

### 4.2 `tests/unit/search/test_chance_node.py`

- **Bucket weight correctness**: hand-construct a StackedDice with a known remaining-bag; assert the 6 bucket weights match the analytic sum-of-counts / total.
- **Karma 7-override math**: with `karma_buff_active(current_player) = True` (i.e. `last_player_to_roll_7 != current_player` and not `None`), verify `w_seven` is shifted by the analytic 20% rule and other buckets are correspondingly down-weighted; weights still sum to 1. Specifically add `test_karma_buff_persistence`: roll a 7 from player A, advance two non-7 turns, then verify the chance-node weighting for player B still applies the Karma adjustment — the buff must not decay or reset on turn change.
- **Expected-value backup**: assign known `V(child_b)` per bucket; assert `V(chance_node) == Σ w_b · V(child_b)` to floating-point precision.
- **Visit count residual handling**: ensure floor-then-residual allocation never produces negative visit counts and conserves total `N` exactly.
- **Singleton-bucket sanity**: assert `seven` bucket cannot be collapsed into another bucket under any code path (regression guard for the panel D13 invariant).

### 4.3 `tests/unit/search/test_belief_determinization.py`

- **Sample shape**: sample 1000 hands from a known belief distribution + known `K`; assert all samples have `sum == K` and per-type frequencies match the input belief to χ² confidence.
- **Normalisation**: pass an un-normalised logits vector; sampler must normalise internally.
- **Fallback-to-uniform path**: with `belief_fallback=True`, sampler must use uniform regardless of input belief; tag is propagated.
- **Edge case K=0**: opponent has no unplayed dev cards; sampler returns the zero vector.
- **Edge case K_max=25**: stress the upper bound (theoretical max dev-cards-in-deck = 25); shape correctness.

### 4.4 `tests/unit/search/test_tree_search.py`

- **Visit-count consistency over deterministic seed**: with `n_sims=10, n_dets=2`, seed fixed, expect bit-for-bit identical visit counts across two runs.
- **Convergence on toy 2-action MDP**: hand-construct a state with two legal actions where one is analytically optimal; assert visits concentrate on the optimal action as `n_sims` grows from 10 → 100 → 1000.
- **Dirichlet noise effect**: at root with noise on, verify visits are *not* deterministic across reseeds (noise injects variance); at root with noise off (eval), verify they ARE deterministic.
- **Temperature application**: from a known visit-count vector, assert `π_mcts ∝ visits^(1/τ)` at τ ∈ {1.0, 0.1}.
- **Mask propagation**: assert no illegal action receives non-zero visit count under any tree path.

### 4.5 `tests/unit/search/test_engine_simulator.py`

- **State independence under copy + step**: copy a mid-game state, step both copies with different actions; assert resource counts / building counts / dev-card hands diverge correctly with no cross-contamination. Patterns from Phase 1.2's 14 copy tests carry over.
- **Round-trip through random play**: 200 random-play games on a copy; assert no exceptions and the original state is unchanged.
- **Mid-discard copy**: copy the state inside the 9-card discard subphase (the edge case that bit Phase 1.1); assert the discard subphase progresses identically on the copy.
- **Mid-dev-card-draw copy**: copy after `BuyDevCard` is dispatched but before the dev-card type is revealed to the player; assert the copy resolves to the same dev-card type as the original.
- **Action mask recomputation**: `get_action_masks()` after a state-changing step must return new masks consistent with the new state — never the cached pre-step masks (regression guard for the `bc/dataset.py` silent-filtering bug).
- **Dice bag side-effect-free read**: 1000 reads of `get_dice_bag_dist()` between rolls; assert bag state unchanged.

### 4.6 `tests/integration/test_az_smoke.py`

End-to-end smoke: random-init `CatanPolicy` + MCTS at `n_sims_per_det=25, n_determinizations=2` on a real env. Run 100 decisions across a full game; assert:
- All sampled actions are legal (zero mask violations).
- Tree node count per decision stays under the 15k budget.
- No exceptions in the chance-node + belief-determinization paths.
- Wall-clock per decision < 30 s (M1 Pro CPU baseline; A100 target < 2 s).

### 4.7 `tests/integration/test_belief_calibration.py`

Implements preflight 0.4: loads the Step-4 checkpoint + a 1024-state held-out batch + the env's GT belief targets. Computes mean KL (belief_pred || env_GT) and mean KL (uniform || env_GT). Prints pass/fail per the §0.4 decision rule + JSON summary.

### 4.8 Test-budget commentary

Targets the patterns that bit `bc/dataset.py` and Phase 1.2:
- **Silent action-filtering after state transition**: `test_engine_simulator.py` action-mask-recomputation test.
- **`copy()` correctness under edge cases**: `test_engine_simulator.py` mid-discard + mid-dev-card-draw tests.
- **deepcopy edge cases for policy state**: `CatanPolicy.set_board_geometry` registered buffers — covered in `test_engine_simulator.py` round-trip test (the policy is *not* deep-copied inside search; only the env state is — but the prior_from_policy callback must not mutate the policy's registered buffers; this is asserted by running 1000 prior_from_policy calls on a fixed obs and comparing the policy state-dict hash before/after).

---

## 5. Ablations (per `v2_design.md` §3.8 tiered budget)

Four ablations, each a leave-one-out vs the full Phase B config. N=2 seeds each at the §6 acceptance-gate scale, then full-N if a result needs paper-grade confidence.

### 5.1 `phase_b_no_search.yaml` (policy-only baseline)

The Step-4 checkpoint, no MCTS at decision time. The control: **does MCTS earn its cost?** If `phase_b_full` does not beat this by ≥ 0.05 symmetrised WR vs heuristic, Phase B is **not** shipped as a v2 deliverable — the policy alone is the artefact.

### 5.2 `phase_b_full_chance.yaml` (11-way chance fan-out)

Replaces the 6-bucket compression with the naive 2d6 distribution (sums 2-12 as 11 independent buckets). Tests panel D13's compression decision. Expected outcome: roughly equivalent WR but ~2× slower per decision (more chance children → more leaf evaluations per sim). If WR is *higher* under 11-way, D13 was miscalibrated and we should revisit.

### 5.3 `phase_b_more_sims.yaml` (200 effective sims)

`n_sims_per_det = 100, n_determinizations = 2 = 200 effective sims`. Tests the AlphaZero-veteran-panelist's KEY FLIP (the panel compromised at 50; the AZ vet pushed for higher). Compute cost: ~4× per decision. If WR improves by ≥ 0.05 over 50-sim, raise the scale-up gate; if not, 50 stays as the Phase B baseline.

### 5.4 `phase_b_uniform_belief.yaml` (uniform-prior determinization)

Replaces belief-head sampling with uniform 5-way at the same `K`. Tests Cicero-panelist's KEY FLIP correction. Expected outcome: **meaningfully lower WR if the Step-4 belief head is well-calibrated (preflight 0.4 green)**. If WR is *equivalent* or higher under uniform, the belief head is not actually informative — that's an upstream Step-4 issue, not an MCTS issue.

### 5.5 Tiered budget

- **Tier 1** (must-run for any Phase B claim): `no_search`, `uniform_belief`. These directly test the load-bearing claims ("MCTS earns its cost", "belief-head sampling beats uniform").
- **Tier 2** (panel-split decisions): `full_chance`, `more_sims`. Run if Tier 1 is green AND compute remains.
- **Tier 3** (opt-in, not budgeted by default): vary `c_puct ∈ {1.0, 1.5, 2.0}`, vary Dirichlet `α ∈ {0.1, 0.3, 1.0}`, vary temperature schedule (constant 0.5 throughout vs the §2.3 schedule). Only run if a result needs paper-grade defence.

---

## 6. Acceptance gate

Compound gate, all four must pass with **N ≥ 3 seeds** (per faculty-corrected `v2_design.md` §6).

### Gate 1 — heuristic WR

Symmetrised WR ≥ **0.90** vs heuristic, 200 games per seat × N=3 seeds (1200 total per seed). Reported with P1-seat / P2-seat / symmetrised columns. Bootstrap CI 99% must clear 0.90 from below.

### Gate 2 — MCTS-vs-policy-alone (MCTS earns its cost)

WR ≥ **0.55** vs the Step-4 policy-alone baseline, head-to-head, 200 games per seat × N=3 seeds. Bootstrap CI must clear 0.55 from below. This is the **load-bearing claim** of Phase B; if it fails, Phase B is not shipped.

### Gate 3 — PPO-BR-gap held

Fresh 1M-step PPO-BR adversary against the final Phase B checkpoint (per `v2_step4_ppo.md` §4.4 + `v2_design.md` §3.6). PPO-BR-gap ≤ Step-4's BR-gap (i.e., MCTS does not make the policy more exploitable than it already was). Sensitivity probe: also report `Δ_BR = b_5M − b_1M` for context.

### Gate 4 — champion regression vs Step-4

Symmetrised WR ≥ **0.55** vs the Step-4 checkpoint over 200 games per seat × N=3 seeds. Stricter than Gate 2 (which only tests the *same* network with vs without search at decision time) — Gate 4 tests that the MCTS-trained network is strictly stronger than the PPO-trained network, even when both play policy-alone at eval.

### Diagnosis ladder when a gate fails

- **Gate 1 fails (< 0.90)**: heuristic-WR was 0.70 at Step 4; if MCTS doesn't lift it to 0.90, the search is not adding enough signal. Check Gate 2 first — if Gate 2 also fails, the search itself is the problem (audit per the §8 risk register).
- **Gate 2 fails (≤ 0.55)**: MCTS adds nothing. Check preflight 0.4 — if belief calibration was yellow/red, retrain Step-4 belief head and re-run.
- **Gate 3 fails**: MCTS-trained network is *more* exploitable than Step-4. Likely cause: visit-count targets at training are noisy (low `n_sims`) and the network overfits to MCTS artefacts. Try `more_sims` ablation.
- **Gate 4 fails but Gate 2 passes**: the *network* hasn't learned much; only the *combo network + search* is strong. Run more training-time MCTS data (extend budget) before shipping.

---

## 7. Compute budget

- **Phase B (this plan)**: ≤ **1 week of A100** cloud time (per `v2_design.md` §6, faculty-corrected wall-clock cap).
- **A100 baseline FPS estimate**: ~50-100 FPS under MCTS at 50 effective sims (10× M1 Pro CPU MCTS rate; 2-4× slower than A100 policy-alone PPO at ~200 FPS). At 75 FPS sustained, 1 week = ~45M MCTS-decisions, plenty for AZ-style training on top of a warm-started network.
- **Preflight gates** (this plan §0): 6 hours on M1 Pro for all four checks.
- **Ablations Tier 1**: 2 configs × 3-5 days on A100 = 6-10 days. Tier 2 adds ~6 days if Tier 1 green. Tier 3 opt-in.
- **PPO-BR-gap eval** (Gate 3): 1M BR per final checkpoint × N=3 seeds = 3 BR runs × ~6 hours on A100 = 18 hours.

**If A100 is unavailable**:
- Run preflight gates 0.1-0.4 only.
- Run Gate 2 A/B (200-game MCTS-vs-policy-alone) on M1 Pro — at ~30 s/decision × 200 actions/game × 200 games = ~333 hours. Still infeasible. Subsample to 50 games × N=1 seed for a ~83-hour smoke (3-4 days wall-clock); report as preliminary and queue full eval for whenever A100 lands.
- Document the gap; Phase B is then **queued**, not delivered.

---

## 8. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| `CatanGame.copy()` perf above budget — 50 sims × 200 actions × N games is wall-clock-prohibitive if mean copy > 5 ms | **High** | **High** | Preflight 0.2 measures this before Step 5 starts. If yellow, halve sims budget. If red, port hot path to Cython/Rust (1-2 wk) OR shelve Phase B. |
| Belief head from Step 4 is biased early/poorly-calibrated → determinization sampling misleads the search | Medium | High | Preflight 0.4 measures this with a comparator-anchored threshold (`KL ≤ 0.35` AND ≥ 0.10 better than uniform). Yellow path: proceed with warning scalar. Red path: retrain belief head OR fall back to uniform determinization. |
| Search tree memory blow-up — 50 sims × ~30 branching × ~50 depth (worst case) = ~2.5M nodes/decision | Low | High | Depth-limited at 40 plies; mean depth ~10 keeps actual usage at ~15k nodes (4 MB / tree). 16 parallel envs × 4 MB = 64 MB peak, well within RAM. Transposition tables NOT added (1v1 with chance + imperfect-info — transpositions are rare and the bookkeeping cost outweighs the deduplication gain). |
| MCTS during training amplifies value-head noise (low `n_sims` → noisy visit-count targets → policy overfits) | Medium | Medium | `more_sims` ablation (§5.3) measures sensitivity. If `phase_b_more_sims` materially beats `phase_b_full`, raise scale-up gate. |
| Chance-node 6-bucket compression loses signal vs 11-way native fan-out | Low | Low | `phase_b_full_chance` ablation (§5.2) measures this. Compute-cost analysis says 11-way is 2× slower for marginal WR change; if surprising, revisit. |
| Karma 20%-7 override math is wrong | Medium | Medium | `test_chance_node.py` has the analytic check + the 7-singleton invariant. Engine round-trip test in `test_engine_simulator.py` exercises Karma-7 forced rolls. |
| Policy-state mutation through `prior_from_policy` (registered-buffer corruption à la `CatanPolicy.set_board_geometry`) | Low | High | `test_engine_simulator.py` state-dict-hash before/after 1000 callback calls. |
| 1v1 imperfect-info: ISMCTS aggregation across determinizations is heuristic, not Nash-convergent | High | Medium | **By design** — Phase B is framed as ISMCTS-style policy improvement (Cowling, Powley, Whitehouse 2012), NOT Nash convergence. Faculty review confirmed CFR-family algorithms are out of v2 scope. Gate 3 (PPO-BR-gap held) is the empirical bound. |
| Phase B compute infeasible on M1 Pro — A100 not available | **High** | **High** | §7 fallback path: ship Step-4 result + queue Phase B. The plan is explicit that Phase B *requires* A100; without it, Phase B is queued, not delivered. |
| Visit-count distribution at low `n_sims` is degenerate (most actions get 0 visits) → AZ CE loss has near-infinite gradient | Medium | Medium | Sample temperature τ=1.0 (proportional-to-visits, not argmax) during training keeps the target distribution smooth. `test_tree_search.py` verifies no visit count is < 0. Floor visit count at 1e-4 in the CE target if needed. |

---

## 9. STOP/RESUME points

| Where | What to verify | Human decision |
|---|---|---|
| **Pre-Step-5** (after Step 4 + this plan + preflight gates) | Preflight 0.1-0.4 all green; A100 access confirmed (or fallback path acknowledged) | Approve Step 5 kickoff |
| **First MCTS-vs-policy-alone A/B** (~100 games, BEFORE full training) | MCTS WR > policy-alone WR + 0.03 (variance-aware paired test). Runs on whatever compute is available — 6 hours on A100 or ~3-4 days on M1 Pro. | **PASS → approve full training run.** **FAIL → STOP and audit before paying the training-time MCTS tax.** |
| **First 1M MCTS-decisions** | Value loss not diverging; entropy of visit-count distribution stays > 0.5 (search not collapsed); tree-node-allocation profiler within budget; sims/sec stable | Approve continuation OR adjust temperature schedule + sims budget for next seed |
| **First eval at 100k MCTS-decisions** | Heuristic WR ≥ Step-4-policy-alone WR (MCTS shouldn't immediately regress) | OK if within band; STOP if WR collapses |
| **Mid-training (~5M MCTS-decisions per seed)** | WR trending up; Gate 2 (MCTS-vs-policy-alone) on 100-game smoke clears 0.55 | Approve continuation OR stop seed and re-spec |
| **Soft-fail at Gate 1** (WR 0.80–0.89 at 1-week wall-clock) | Per §8 risk register | Pick a mitigation (more sims, larger compute window), run another seed |
| **Hard-fail at Gate 2** | MCTS doesn't earn its cost. STOP and audit. | Either ship Step-4 policy-alone as v2 deliverable (Phase B not shipped) or open a follow-up investigation |
| **Final gate evaluation** | All four gates §6 green with N=3 seeds | Approve v2 paper / release; archive checkpoints; close Phase B |

---

## 10. Panel + faculty-review carry-forward

Decisions inherited from `v2_design.md` round-2 (commit `9d34138`) and `v2_step4_ppo.md`:

- **D1** (3-2 panel + faculty): Phase B (MCTS) kept; budgeted at ≤ 1 week A100. Step 5 must produce a policy strong enough that MCTS sees a real lift (Gate 2 ≥ 0.55); if Step-4 alone hit Gate 1 ≥ 0.85 in Step 4, the marginal value of MCTS is small and we should re-evaluate before paying the compute tax.
- **D12** (4-1 panel + faculty): MCTS sims **shrunk** from 50×4 = 200 effective sims (AZ vet's preference) to **25×2 = 50 effective sims** (M1 Pro panelist + compute-budget realism). `more_sims` ablation (§5.3) is the panel-split test.
- **D13** (5-0 panel): chance-node fan-out compressed from naive 11-way to **6 buckets** weighted by StackedDice remaining-bag. `full_chance` ablation (§5.2) is the empirical sanity check.
- **Cicero KEY FLIP correction**: sample opponent's hidden dev-card hand from the **policy's belief head**, not uniform. `uniform_belief` ablation (§5.4) is the empirical test; preflight 0.4 is the calibration gate that decides whether to use belief sampling vs the fallback uniform.
- **Faculty re-review framing**: Phase B is **ISMCTS-style policy improvement** on top of a strong learned prior (Cowling, Powley, Whitehouse 2012), **NOT** Nash convergence. AlphaZero's convergence guarantees are for perfect-information deterministic games and do NOT transfer to imperfect-info stochastic games. CFR-family algorithms (Zinkevich et al. 2007) are the principled 2p-zero-sum imperfect-info analogues but are explicitly out of v2 scope per the faculty re-review.
- **Faculty re-review** seat-stratification: all WR metrics in §6 report P1 / P2 / symmetrised, N ≥ 3 seeds, bootstrap CI.
- **Faculty re-review** wall-clock budget: Phase B ≤ 1 week A100; fallback path documented in §7 if A100 unavailable.

### References cited (mirrored from `v2_design.md` §7)

- **Cowling, Powley, Whitehouse 2012** — *Information Set Monte Carlo Tree Search*. The actually-cited prior art for ISMCTS; Phase B's framing comes from this paper, not from AlphaZero.
- **Silver et al. 2018** — *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Source for the PUCT formula, Dirichlet noise α=0.3 ε=0.25 at root, temperature schedule (τ=1.0 training, τ=0.1 eval after move N).
- **Zinkevich, Johanson, Bowling, Piccione 2007** — *Regret Minimization in Games with Incomplete Information*. CFR; cited ONLY to scope what we're explicitly NOT doing here. Phase B produces a policy-improvement step on top of a learned prior; it does not provide game-theoretic Nash convergence bounds. CFR-on-Catan-late-game-abstraction is the principled future work, scoped out of v2.

---

## Provenance

- Base design: `docs/plans/v2_design.md` §3.5, §3.7 (panel votes D1/D12/D13), §5 Step 5, §7 references.
- BC plan that Phase B is downstream of: `docs/plans/v2_step3_bc.md`.
- PPO plan that Phase B is directly downstream of: `docs/plans/v2_step4_ppo.md`. Gates 1-3 of that plan are the preflight 0.1 gate for this plan.
- v1 search prototype (informational, not load-bearing for v2): `/Users/benjaminli/my_projects/catan_rl/src/catan_rl/algorithms/search/ismcts.py` shipped in v1 Phase 4.1 as a 1v1-only single-step PUCT module. The v2 Phase B design supersedes it (chance nodes + belief determinization + AZ-style training were never wired in v1).
- Preserved v1 champion for the Gate 4 regression check: `/Users/benjaminli/my_projects/catan_rl/checkpoints/train/checkpoint_16162816.pt` (only v1 checkpoint preserved; all other v1 checkpoints deleted per user direction).

---

**WAITING FOR CONFIRMATION**
