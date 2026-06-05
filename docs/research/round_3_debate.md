# Round 3 — Targeted Debate

Five disagreements from Round 2. For each: Turn 1 (strongest argument + evidence), Turn 2 (direct response), Turn 3 (change-mind condition + status). Citations to file paths required; no vibes.

---

## D1 — Does MCTS earn its compute, or inherit Catanatron's depth-3 pathology?

**Turn 1 — B**: AlphaZero's policy-iteration loop (policy → MCTS → improved policy → improved value) has *measured* superhuman precedent in chess/Go/shogi (Silver et al. 2018). The Catanatron MCTS attempt (`catanatron.md` §"Bot players") used 25-100 random rollouts with **no learned prior, no learned value** — a textbook vanilla MCTS in a 200-turn-episode game where Monte Carlo returns carry near-zero signal. v2's `tree_search.py` (per Step-5 plan §3) uses `prior_from_policy` + `value_from_policy` callbacks — Catanatron's failure mode is upstream of the algorithm v2 ships.

**Turn 1 — D**: B's claim is true *for the search algorithm*. The depth-3 paradox isn't about which algorithm you use to traverse the tree — it's about what you traverse. Catanatron's value weights at `value.py:10-24` are *measurably* dominated by `public_vps=3e14`; the value function gives a 6-orders-of-magnitude signal to "have more VP" and a 4-orders-smaller signal to "have more production." Depth-3 *correctly* finds VP-greedy paths — those paths just happen to be high-variance under stochastic dice. v2's Step-4 reward `±1 + (vp_diff)/15` (`v2_step4_ppo.md:64`) couples PPO's value gradient to the same VP-leaning direction. The mechanism transfers. Step-5 plan §0.4 gates on belief-KL only — no `value_calibration_l1` probe (Step-5 plan §8 lists this risk but doesn't gate on it).

**Turn 2 — B responds to D**: granted that a biased value head + deeper search is worse than a biased value head + shallow search. But the v2 value head is *learned* from on-policy returns, not hand-tuned. PPO's value loss against terminal returns is unbiased in expectation; the value head should *correct* the reward's VP coupling over training. Concrete check: at end of Step 4, if `value/explained_variance > 0.7` on a holdout of late-game states with non-trivial VP differentials, the value head has learned the *strategic* value, not the reward shape. Catanatron's hand-tuned weights have no such learning step.

**Turn 2 — D responds to B**: explained-variance is computed against the *same* return distribution PPO trained on. If the training distribution is VP-coupled, EV is high and the head is *still* biased. EV is a self-consistency check, not a calibration check. The right probe is `value(s) − empirical_WR(s)` on **held-out** evaluation games against a non-PPO opponent (AlphaBeta-d2 or heuristic). v2 has no such probe in Step 4 or Step 5.

**Turn 3 — B**: would change my mind if Step-5's "first MCTS-vs-policy-alone A/B" returns MCTS WR < policy-alone WR + 0.00 (no lift) at N≥3 seeds. **Has not changed.** Adds: agrees a `value_calibration_l1` probe vs heuristic empirical WR should be added to Step-4 milestone evals (this is a concession on the probe, not the algorithm).

**Turn 3 — D**: would change my mind if (a) the new probe lands within 5% absolute calibration AND (b) the Step-5 A/B shows ≥ 3pp MCTS lift. **Has not changed.** The depth-3 prior stands until measured.

**Status**: COMPROMISE-LIKELY — both agents agree the value-calibration probe should be added; the *interpretation* of its result is what differs. Defer to Round 4.

---

## D2 — Is `±1 + (vp_diff)/15` the right reward, or the VP-coupling trigger?

**Turn 1 — C**: pure binary `±1` (Catanatron's `simple_reward()` at `catanatron_env.py:24-30`) leaves blowout-vs-squeaker information on the table. In a 15-VP game where matches typically end 15-8 to 15-14, the margin term distinguishes states the policy must learn to evaluate differently. The Medium postmortem rules out per-step ΔVP (`catanatron.md` §"Citable failure modes" #2, END_TURN dominance) and time-discounted VP_DISCOUNTED_RETURN ("still lost to RandomPlayer"). Among the surviving options, `±1 + (vp_diff)/15` is dense enough to teach margin without amplifying END_TURN.

**Turn 1 — D**: C's argument assumes margin is *information* about strategic value. It is not — it is information about *how the heuristic plays*. v2's BC anchor learned from heuristic-vs-heuristic + perturbed games; the heuristic systematically races VP. Conditioning PPO's reward on `vp_diff` re-teaches the policy what it already knew from BC, while *also* pushing it toward exactly the Friendly Robber-attracting states the rules penalise (`my_agent.md` §10 + `CLAUDE.md:26`). Pure `±1` removes this coupling; the policy learns purely from WR. Catanatron's value function with `public_vps=3e14` is the canonical VP-coupled value — and Catanatron's depth-3 result is the canonical failure mode.

**Turn 2 — C responds**: D conflates the *reward signal* with the *value function*. PPO's value head fits returns. Under terminal `±1 + (vp_diff)/15`, the value learns "win is +1 baseline, +1.0 for a 15-0 blowout, +1.06 for a 15-1 win, asymptotically +2 for 15+VP" — a smoothly-graded surface. Under pure `±1`, the value head sees a step function and credit assignment over 250-step episodes becomes harder. The right test is BR-gap, not "does reward shape resemble Catanatron's value weights."

**Turn 2 — D responds**: C's "smoothly-graded surface" is exactly the issue. A smoothly-graded VP-correlated surface rewards leading-by-3-VP states equally to leading-by-3-VP-and-robber-immune states. The Friendly Robber rule explicitly punishes leading. The reward fails to model the rule. Pure `±1` is shape-agnostic — the policy must learn the Robber's coupling from WR alone, but at least it isn't being *trained* to ignore it.

**Turn 3 — C**: would change my mind if a 1M-step Step-4 ablation between `±1` and `±1 + (vp_diff)/15` shows pure binary with WR within 0.01 of margined. The cost of the experiment is the same; if it's a wash, drop the margin term. **Has not changed yet — wants the experiment.**

**Turn 3 — D**: would change my mind if the same ablation shows margined ≥ 0.05 better on PPO-BR-gap (the only WR-agnostic measure). **Has not changed.**

**Status**: DECIDABLE BY EXPERIMENT — both accept the same A/B (Step-4 ablation, 1M steps, 3 seeds). Defer to Round 4 to log as a launch-blocking probe.

---

## D3 — piKL decay 2M vs ≥ 10M?

**Turn 1 — B**: 2M PPO steps is the panel-revised window (`v2_step4_ppo.md:78`). The original 5M was deemed too long ("anchor doesn't drag past peak"). The fallback to λ=0.4 over 5M is documented if Step 4 plateaus (`v2_step4_ppo.md`'s diagnosis ladder). Most relevant: the BC anchor is *frozen at BC training time* — its strength is fixed. After 2M PPO steps, the anchor has either pulled PPO into a useful basin or it's irrelevant; extending decay accomplishes nothing.

**Turn 1 — D**: 2M PPO env-steps ÷ ~250 steps/episode = 8000 episodes. The value-head variance over 8000 self-play episodes with terminal-only reward is *substantial* — Catanatron's TensorForce DQN plateaued at 80% WR vs Random after 8 hours and never broke through, and that's an *easier* learning problem with denser feedback. piKL decaying to 0 at 2M leaves the value head sitting on at most 8000 trajectories' worth of return information — below the threshold at which value-based policy improvement is reliable. B's "anchor has either pulled or is irrelevant" assumes the policy is already strong at 2M; that's the conclusion, not a premise.

**Turn 2 — B**: D's variance argument is over the *full* state space. PPO doesn't need to explore all states — it needs the BC anchor to keep it on the productive subspace, then exploit. 8000 self-play episodes against PFSP-hard league opponents is enough exploration if the BC anchor is reasonable. The probe `bc/v_drift_l1` (`v2_step4_ppo.md:92`) catches the failure mode if it materialises.

**Turn 2 — D**: `bc/v_drift_l1` measures L1 between BC value head and current value head. If both are biased the same direction, L1 is small and the probe is silent. D2 makes the same point about EV.

**Turn 3 — B**: would change my mind if `bc/v_drift_l1` is still trending up at 2M in seed #1. **Has not changed but agrees to log this.**

**Turn 3 — D**: would change my mind if `bc/v_drift_l1` *is* informative and stable at 2M. **Has not changed.**

**Status**: DECIDABLE BY MEASUREMENT — first seed of Step 4 provides the answer. Round 4: ship 2M, gate seeds #2-3 on the seed-1 drift curve.

---

## D4 — Is the obs schema Markovian, or missing bag state?

> **Erratum (2026-05-15)**: two corrections.
> 1. The Karma mechanic is *persistent*, not a previous-turn flag (`last_player_to_roll_7` updates only when a 7 rolls).
> 2. The Karma half of the originally-proposed extension is already in the obs at `src/catan_rl/policy/obs_encoder.py:545-560`.
>
> Net effect on this debate: Agent D's bag-state attack is unaffected and still load-bearing. The Karma-state portion of the attack is retracted as moot — already addressed by the existing obs encoder. A2 reduces to a `bag_remaining`-only ablation.

**Turn 1 — A**: the obs schema is the *factored* state. `StackedDice` is part of the env, not the agent's input — same reason chess engines don't have "moves-played-so-far" as an input. The policy sees the board, hands, dev-cards, ports; the env emits transitions consistent with its dice mechanism. PPO converges to the policy that's optimal for the *induced* MDP, dice noise included.

**Turn 1 — D**: chess is deterministic. Catan is not. Under StackedDice + Karma 20%-7, P(next_roll | game_history) ≠ P(next_roll | board_state). The bag's remaining 11 outcomes is hidden state — the agent's value function `V(obs)` integrates over the *unconditional* dice prior, not the *conditional-on-bag* posterior. For late-game states where the bag is partly consumed, this gap is potentially several VP of advantage misvalued.

**Turn 2 — A**: in 1v1 with no P2P trade, both players observe every dice roll. The bag posterior is *common knowledge* — both players have the same belief. Whatever bias `V(obs)` has is *symmetric* across self-play opponents. In zero-sum 2-player self-play, symmetric bias cancels: PPO optimises the same policy under both views, and the symmetric error doesn't affect the gradient direction.

**Turn 2 — D**: cancels in the gradient *direction*, not magnitude. A symmetric overestimate of late-game state value still affects advantage scale → still affects entropy + PPO clipping behaviour. More importantly: at *eval* time against the heuristic or AlphaBeta, the opponent isn't symmetric. v2's evaluator may pick suboptimal late-game actions because its value head over-trusts low-bag-residual states.

**Turn 3 — A**: would change my mind if the ablation `phase_a_with_dice_obs.yaml` (D2's proposed test) shows ≥ 2pp WR improvement against heuristic. **Has not changed.**

**Turn 3 — D**: same. **Has not changed.**

**Status**: DECIDABLE BY EXPERIMENT — cheap ablation (add 11 + 1 features to obs schema, retrain Step 3 BC, evaluate Gate-1).

---

## D5 — Does the opp-id embedding generalise or memorise?

**Turn 1 — A**: 8-d embedding × 100 league entries = 800 params total in this subspace. Mask probability 40% at training (`my_agent.md` §8) forces the policy to *also* function with `opp_id=unknown`. This is dropout-regularization-as-generalization; the policy must produce reasonable actions without the embedding. Whatever the embedding contributes when present is bonus signal.

**Turn 1 — C**: the cost of the embedding is negligible (`round_1_reward_eval.md` §3). Even if it overfits, the 40% mask means the policy *always* has a working zero-embedding fallback. Worst case: the embedding adds nothing; best case: it conditions league play. Asymmetric upside.

**Turn 1 — D**: the 40% mask is training-time. At eval against `heuristic` or `AlphaBeta` or `v1 champion`, the opp_id is `unknown` — the policy operates on its fallback only. v2 has *never measured* whether the embedding-on vs embedding-off policy is the same against held-out opponents. If embedding-on adds ε to in-league WR and embedding-off loses 2ε at the bench, the headline "superhuman" claim is a lower bound.

**Turn 2 — A+C**: D's concern is real but cheap to test. Two parallel 200-game evals at each milestone — embedding-on vs embedding-off against the same opponent. ΔWR is the diagnostic. This eval surface already exists; the marginal cost is 200 games per milestone.

**Turn 2 — D**: accepts the test. If `Δ_WR > 2pp`, the embedding is conditioning in-distribution opponents differently than out-of-distribution ones — a partial-memorisation finding. If `Δ_WR < 2pp`, generalisation is fine.

**Turn 3 — All**: agree on the test. Status: RESOLVED via instrumentation, not algorithm change.

**Status**: COMPROMISE — keep the embedding; add the `Δ_WR_emb_on_vs_off` probe to milestone evals.

---

## Aggregate

- D1: COMPROMISE (add value-calibration probe; defer interpretation to data).
- D2: DECIDABLE BY EXPERIMENT (Step-4 ablation, 1M × 3 seeds).
- D3: DECIDABLE BY MEASUREMENT (seed-1 `v_drift_l1` curve).
- D4: DECIDABLE BY EXPERIMENT (`phase_a_with_dice_obs` ablation).
- D5: RESOLVED-with-instrumentation (add embedding-on/off ΔWR probe).

Two compromises, two experiment-decidable, one resolved. Zero genuine deadlocks. Move to Round 4 for the formal classification.
