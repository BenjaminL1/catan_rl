# Round 2 — Disagreement Surfacing

Five real disagreements after reading all four Round-1 papers. None manufactured. Each names the agents holding incompatible positions, the load-bearing stake, and what a decisive test would look like.

---

## D1. Does MCTS (Step 5) earn its compute, or does it inherit Catanatron's depth-3 pathology?

- **Agent B** (learning-algo): MCTS earns its 30-50× cost. Catanatron's MCTS failed because it had no learned prior and no learned value — random rollouts in long episodes carry no signal. v2's PUCT search uses the Step-4 policy as prior + value-head at leaves; this is the AlphaZero recipe, not vanilla MCTS, and Catanatron's failure does **not** generalise. Belief-KL gating (`KL ≤ 0.35`) handles the imperfect-info axis (`round_1_learning_algo.md:30-32`).
- **Agent D** (skeptic): the depth-3 paradox is a **value-function pathology**, not a search-quality issue. Catanatron's value weights (`public_vps=3e14` dominating `production=1e8` by 6 orders) cause depth-3 to find VP-maximizing paths that are stochastic traps. v2 inherits this directly via the `±1 + (vp_diff)/15` reward coupling policy + value gradients to VP-leaning shaping. MCTS amplifies that bias. **Step 5 has no value-calibration gate, only belief-KL.** Catanatron's measured outcome is the prior; v2 has produced no evidence to override it (`round_1_skeptic.md:7-13`).
- **Agents A, C**: silent on this question (out of scope).
- **Why it matters for final performance**: Step 5 commits ≥ 1 week A100 cloud time. If D is right and the value head is VP-biased, MCTS regresses below policy-alone; Step 5 is a compute write-off.
- **Decisive test**: Step-5 plan §STOP/RESUME's "first MCTS-vs-policy-alone A/B" — 100 games, MCTS WR > policy-alone WR + 0.03 with variance-aware paired test. **D withdraws at ≥ 3pp absolute symmetrised WR with N≥3 seeds; B's position is confirmed.** Already in the plan as a gate but worth raising the seeds to 3 from "whatever compute available."

---

## D2. Is `±1 + (vp_diff)/15` the right reward, or does it VP-couple gradients into the depth-3 pathology?

- **Agent C** (reward/eval): keep terminal-only `±1 + (vp_diff)/15`. The faculty review was correct: per-step ΔVP shaping amplifies END_TURN dominance (Catanatron's 30%-of-states finding); plain binary leaves blowout-vs-squeaker margin information on the table; VP-discounted (`0.9999^turns × VP`) was tried by Collazo and lost to RandomPlayer. The `/15` normalisation bounds the margin term so it cannot dominate the win signal (`round_1_reward_eval.md:5-9`).
- **Agent D**: the VP-margin term *is* the depth-3 trigger. Coupling gradients to visible-VP leads positively rewards states that invite the Friendly Robber. The reward and the value head co-evolve a VP-leaning bias. Catanatron's documented depth-3 regression maps directly to "more search of a VP-biased value" (`round_1_skeptic.md:11`, `round_1_skeptic.md:25`).
- **Agents A, B**: silent on shape; A only notes the obs/action factorization is independent of reward; B notes piKL anchors policy not value, so reward bias propagates to value undetected (`round_1_learning_algo.md:25`).
- **Why it matters**: reward shape is the single hyperparameter that touches every downstream module — value head, advantages, MCTS leaf bootstrap, belief head's training signal. A 0.5pp WR difference in the shape decision compounds over 10M-step training.
- **Decisive test**: A/B at Step 4 between `±1 + (vp_diff)/15` and pure `±1` over N=3 seeds × 10M steps. If pure binary's PPO-BR-gap is within 0.02 of the margined version, the margin term carries no signal and D's critique stands. If margined ≥ 0.05 better, C's position is confirmed.

---

## D3. Is piKL decay over 2M steps adequate, or shorter than the learning horizon?

- **Agent B**: 2M is adequate; fallback path documented (λ=0.4 over 5M if Step 4 plateaus). PPO is on-policy, trust-region-bounded; the piKL anchor is a prior not a constraint (`round_1_learning_algo.md:23-25`).
- **Agent D**: 2M steps ≈ 8000 games (1v1 15-VP episodes ≈ 250 steps). With terminal-only reward and self-play variance, 8000 games is the *variance horizon*, not the *learning horizon*. piKL decays before the value head has stabilised → PPO collapses onto an undertrained value head, the exact pathology attack 1 describes (`round_1_skeptic.md:33`).
- **Agents A, C**: silent.
- **Why it matters**: piKL is the only mechanism preventing PPO from drifting away from the BC anchor before its own value head is reliable. If D is right, Gate 1 fails before λ has decayed and the diagnosis ladder (B's fallback) is invoked anyway — we'd burn one 7-day seed before knowing.
- **Decisive test**: log `bc/v_drift_l1` (already in the plan) over the first 2M steps of seed #1. If `v_drift_l1` is still trending up at 2M, D is right and decay needs to extend before seeds #2-3 launch. If it's trending down or flat, B is right.

---

## D4. Is the obs schema Markovian, or missing dice-bag state?

> **Erratum (2026-05-15)**: two corrections to this section.
> 1. The Karma mechanic was misread as a one-roll signal; it is in fact a *persistent* buff (`last_player_to_roll_7` updates only when a 7 rolls).
> 2. **Karma state is already in the obs** — `src/catan_rl/policy/obs_encoder.py:545-560` ships two persistent Karma flags inside `current_player_main`. The originally-claimed missing-from-obs predicate is not actually missing.
>
> The Markov-violation attack therefore reduces to a single legitimate concern: `bag_remaining` is not in the obs. The A2 ablation collapses to a single new field (`bag_remaining: (11,)`) rather than two.

- **Agent A** (state/action): the 10-key obs is the right factored representation. Belief head handles dev-card hidden axis. D6 augmentation is cheap and correct (`round_1_state_action.md:35-41`).
- **Agent D**: StackedDice's next-roll distribution depends on (a) bag remaining (36 outcomes minus consumed), (b) persistent Karma buff state (`last_player_to_roll_7 != current_player` triggers a 20% forced-7 override on every roll until the current player rolls a 7). **Neither is in the obs dict.** `value_head(obs)` is therefore not a state-value function in the MDP sense. GAE bootstrap is biased by an unknown amount. MCTS chance nodes need bag state — the plan never states whether the search has access (`round_1_skeptic.md:27`).
- **Agents B, C**: B mentions MCTS reads `StackedDice` remaining-bag for chance-node weighting per the Step-5 plan; this is search-time, not obs-time. C silent.
- **Why it matters**: if the value head is conditioned only on board state, it cannot distinguish "early-game roll-7 likely" from "late-game roll-7 unlikely." Advantages are systematically miscalibrated. The miscalibration interacts with D1 and D2 — if value is also dice-biased, MCTS amplifies it twice.
- **Decisive test**: ablation `phase_a_with_dice_obs.yaml` — add `bag_remaining: (11,)` to the obs schema, retrain Step 4. (Karma state was the other half of the originally-proposed extension but is already encoded; see erratum above.) If WR vs heuristic improves ≥ 2pp, D is right; the original schema was incomplete. If indistinguishable, A is right; bag state was already implicitly captured by board state.

---

## D5. Does the 8-d opp-id embedding with 40% mask generalise, or memorise the 100-slot league?

- **Agent A**: keep current opp-id design at 8-d with 40% mask (`round_1_state_action.md:7-13`'s implicit endorsement of the v2 design).
- **Agent C**: opp-id + TrueSkill are "cheap relative to PPO step cost"; the question is "would dropping any of them recover meaningful compute" and the answer is no (`round_1_reward_eval.md:18-20`). Implicit endorsement.
- **Agent D**: league maxlen=100 + 8-d embedding has more capacity than the support set. Mask is training-time only — no eval-time test confirms generalisation. Against held-out opponents the embedding is forced to "unknown" and the policy silently loses learned conditioning (`round_1_skeptic.md:31`).
- **Why it matters**: if the embedding has memorised league IDs, the agent's strength against the heuristic/v1 champion / AlphaBeta benches (all eval-time "unknown") is a *lower bound* on its in-league strength — but it's a lower bound that excludes the conditioning signal. The "superhuman" claim relies on benches being representative.
- **Decisive test**: at any milestone, run two parallel 200-game evals: (a) opp-id provided correctly, (b) opp-id forced to "unknown" for the same opponent. If `Δ_WR > 2pp`, D is right. This costs an extra 200 games per eval — small.

---

## Out-of-scope-by-consensus

The following appeared in Round 1 but **all four agents agree** (or D is silent and the others agree):
- **DQN is not the fallback** (B explicit, others silent / supportive).
- **Catanatron's flat 289-action space is a downgrade** (A explicit; B, C don't push back).
- **QSettlers contributes no transferable benchmark** (D's attack 2 is the strongest statement; A, B, C accept it implicitly).
- **D6 symmetry augmentation should stay at 0.5** until E0.3/E0.4 probes confirm (A explicit; D's attack 3.3 is a verification request not a disagreement).
- **Bootstrap CI + 1200-game gate floor** (C explicit; A, B silent; D doesn't contest).
- **AlphaBeta-d4 is the first eval cut under compute pressure** (C explicit; nobody contests).

These do not need Round 3.

---

## Disagreements to take into Round 3

D1, D2, D3, D4, D5 — five focused exchanges, each with a decisive experiment.
