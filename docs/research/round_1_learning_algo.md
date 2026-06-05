# Round 1 — Learning Algorithm Position Paper (Agent B)

**Scope**: core learning/search method for v2. Sample efficiency, suitability for stochastic + imperfect-information 1v1 Catan, failure modes from prior work, and which approach has a principled path to superhuman play.

## 1. Handcrafted heuristic + alpha-beta-depth-2 (Catanatron SOTA)

Catanatron's `AlphaBetaPlayer(n=2)` tops the leaderboard. `catanatron/catanatron/players/value.py:10-24` defines `DEFAULT_WEIGHTS` where `public_vps = 3e14` outweighs every other term by six-plus orders of magnitude, with `production=1e8`, `enemy_production=-1e8`, and small tiebreakers (`hand_synergy=1e2`, `army_size=10.1`). The leaf evaluator is "VP first, productive expansion second"; alpha-beta with expectimax chance nodes (full 11-way 2d6, `minimax.py:111-126`) provides 2-ply lookahead under `MAX_SEARCH_TIME_SECS=20` (`catanatron.md:44`).

The **depth-3 paradox** (`catanatron.md:84`): Collazo reports depth-3 *regresses*. The postmortem attributes this to value miscalibration at deeper horizons — myopic-tactical bias. When the leaf evaluator is a sum of greedy proxies, deeper search amplifies their noise along sequences the heuristic was never designed to score. Critically, this is a **4-player** result: at depth 3 the branching factor explodes across three opponents, and friendly-fire dynamics (Player A's optimal move hands Player B the win) are opaque to a sum-of-features evaluator.

Does this apply to 1v1? **Partially.** Branching is dramatically smaller (one opponent, no P2P trade per `glossary.md:8`, 13-type action space per `my_agent.md:37-50`). Depth-3 expectimax becomes feasible, and friendly-fire vanishes in zero-sum 2-player. But the *root cause* — leaf-evaluator miscalibration — persists. A hand-tuned 1v1 value function still risks depth-3-optimal myopic lines a learned value head would reject. Handcrafted+search is **bounded above by the quality of its hand-tuned weights**.

## 2. DQN (QSettlers) — would modern DQN succeed?

QSettlers failed on three structural axes (`qsettlers.md:36-67`): (a) replay buffer of **100** is roughly 5 orders below modern minimum (`qsettlers.md:42`), (b) the settlement-placement network was **never trained** — only the binary trade-accept head was (`qsettlers.md:10-11`), and (c) JSettlers **discarded ~35% of games** to crashes, biasing the surviving sample (`qsettlers.md:55-60`). Catanatron's TensorForce DQN attempt independently plateaued at 80% WR vs RandomPlayer (`catanatron.md:77`).

Would a *modern* DQN (Rainbow, prioritised replay, 1M+ buffer) succeed? **In principle yes, in practice no.** DQN learns Q(s,a) over the *flat* action space. Catan's autoregressive 6-head structure (`my_agent.md:24-52`) forces either factorising the heads (Rainbow doesn't natively do this) or treating 168 head-positions as independent Q-values — losing conditional structure. Under sparse terminal reward (`catanatron.md:60`, `my_agent.md:188`), DQN also suffers from the deadly triad (off-policy + bootstrapping + function approximation). PPO sidesteps this by being on-policy and trust-region-bounded.

## 3. PPO + piKL anchor (v2 Step 4)

PPO is provably better-suited than DQN here: on-policy (no deadly triad), native MultiDiscrete autoregressive heads via composite log-probs, robust to high-variance terminal reward `±1 + vp_diff/15` (`glossary.md:57`).

v2 Step 4 stacks four anti-collapse mechanisms (`my_agent.md:88-99`): (1) BC warm-start from heuristic+ε games (`docs/plans/v2_step4_ppo.md:9`), (2) **piKL anchor** `λ_initial=0.2`, linear decay to 0 over **2M steps** (`v2_step4_ppo.md:77-78`), (3) **target_kl=0.03** (raised from 0.01 to avoid undertraining; `v2_step4_ppo.md:56`), (4) **entropy floor=0.003 with rebound** (`v2_step4_ppo.md:61`).

**Residual risk piKL does not address**: piKL anchors the *policy*, not the *value head*. A value-head collapse surviving policy-KL bounds (explained-variance negative for >5M steps, `v2_step4_ppo.md:337`) poisons advantages without tripping alarms. The `bc/v_drift_l1` probe (`my_agent.md:172`) is the safeguard. Secondary risk: piKL is a *prior*, not a *constraint* on exploitation — a strong league opponent the BC anchor never saw will pull π toward BC and away from the correct response.

## 4. AlphaZero / ISMCTS on policy + value (v2 Step 5)

MCTS earns its 30-50× compute cost when (a) the policy prior is decent (PUCT exploration is directed), (b) leaf values are calibrated (backups carry signal), and (c) the search horizon contains tactically-decisive nodes the policy misses. Step 5 commits to PUCT with `c_puct=1.5`, `n_sims_per_det=25`, 6 compressed dice buckets weighted by `StackedDice` remaining-bag (`v2_step5_mcts.md:114`), and Cicero-style belief determinization with `n_determinizations=2` sampled from the **trained belief head**, gated on `KL(belief_pred || env_GT) ≤ 0.35` (`v2_step5_mcts.md:85`).

Catanatron's MCTS at 100 sims/turn was worse than ValueFunctionPlayer because it had **no learned prior and no learned value** — every leaf was a Monte Carlo random-playout, signal-free in Catan's long episodes. v2's MCTS uses the Step-4 policy + value heads at leaves — this is the AlphaZero recipe, not vanilla MCTS, and Catanatron's failure does **not** generalise. The gating condition that matters: belief calibration (`v2_step5_mcts.md:77-88`). If miscalibrated, determinization is worse than uniform — Cicero KEY FLIP risk.

## 5. Path to superhuman

| Approach | Ceiling | Path to superhuman |
|---|---|---|
| Handcrafted + α-β | Bounded by weight tuning | None — Stockfish-analog (`catanatron.md:79`) |
| Modern DQN | Bounded by flat Q on factored actions | Weak — no Catan-shape precedent |
| PPO + piKL | Bounded by policy class + league diversity | Reaches strong-heuristic; ambiguous beyond |
| **PPO + AlphaZero-style MCTS** | Bounded by belief-head quality | **AlphaZero (chess/Go/shogi) + Cicero (imperfect-info)** |

The Catanatron author's "Stockfish not AlphaZero" conclusion (`catanatron.md:79`) is correct *for 4-player Catan with P2P trade* (multi-agent friendly-fire, large action space, weak learned-value precedent) but is **not binding on 1v1 Catan**. 1v1 is zero-sum, perfect-information *modulo* opponent dev-cards — the regime where AlphaZero's policy-iteration loop (policy → MCTS → improved policy → improved value) provably converges, with Cicero-style belief sampling for the hidden axis.

## Recommendation

**Commit v2 to PPO + piKL (Step 4) as the trunk, with AlphaZero-style PUCT MCTS + belief determinization (Step 5) as the superhuman extension.** This is the only approach with a principled, precedented path beyond strong heuristics.

**Fallback if Step 4 plateaus before Gate 1 (sym WR ≥ 0.70 vs heuristic)**: do **not** abandon PPO. Diagnose in order: (1) `bc/v_drift_l1` — if value head drifts freely, lower `value_loss_coef` and re-anchor; (2) raise piKL `λ_initial` to 0.4 and extend decay to 5M (revert the recent panel change per `v2_step4_ppo.md:368`); (3) redo BC with stronger heuristic-ε mixing if BC val NLL stagnated; (4) last resort, *value-distill* into a Catanatron-style hand-tuned-weights baseline as a competitive floor while diagnosing. Crucially **do not pivot to DQN** — none of QSettlers' or Catanatron's DQN failure modes have been refuted by recent Catan-shape literature, and the action-space structure punishes off-policy flat-Q methods.
