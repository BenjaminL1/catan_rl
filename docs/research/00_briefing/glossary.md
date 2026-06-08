# Briefing — Glossary + Cross-Reference

Used by all sub-agents. Pins terminology + the 1v1-vs-4-player asymmetries that matter.

## Rule asymmetry table — 1v1 (v2) vs 4-player (Catanatron + QSettlers)

| Rule | v2 (Colonist.io 1v1) | Catanatron 4-player | QSettlers 4-player |
|---|---|---|---|
| Players | 2 | 3-4 | 4 |
| Win condition | **15 VP** | 10 VP (default) | 10 VP |
| P2P trading | **disabled** (bank only) | enabled | enabled (separate trade network) |
| Discard threshold | **9 cards** | 7 cards | 7 cards |
| Friendly Robber | yes (`<3 visible VP` exempt) | no | no |
| Dice | StackedDice (bag of 36 + 1 non-7 swap + **persistent Karma 20% buff**) | i.i.d. 2d6 | i.i.d. 2d6 |
| Belief state for opp hidden info | yes (5-way over dev type) | no (uses true opp resources) | no |

**Direct WR comparison is invalid.** Any "X beats Y by Z%" claim across these systems must specify which ruleset.

## Algorithm short-codes

| Code | Meaning |
|---|---|
| BC | Behavior Cloning — supervised loss on heuristic decisions. v2 Step 3. |
| PPO | Proximal Policy Optimization (Schulman et al. 2017). v2 Step 4. |
| piKL | Policy-KL anchor loss → keeps PPO policy close to BC anchor; decay over 2M steps. |
| MCTS | Monte Carlo Tree Search. PUCT variant. v2 Step 5 (planned). |
| ISMCTS | Information Set MCTS (Cowling, Powley, Whitehouse 2012) — for imperfect-info games. |
| PUCT | Predictor + UCB applied to Trees (Silver et al. 2018, AlphaZero). |
| PFSP | Prioritized Fictitious Self-Play (AlphaStar 2019) — opponent sampling biased toward losses. |
| GAE | Generalized Advantage Estimation (Schulman et al. 2016). |
| CFR | Counterfactual Regret Minimization (Zinkevich et al. 2007) — Nash-convergent for 2p-zero-sum imperfect-info. Out-of-scope for v2. |
| BR-gap | PPO-best-response gap — train a fresh PPO adversary against the frozen agent for N steps; the BR's WR minus 0.5 is a lower bound on exploitability. |
| Δ_BR | `b_5M − b_1M` sensitivity probe — bounds the gap between 1M-BR (our budget) and true BR. |
| TrueSkill | Microsoft skill rating (μ, σ) with conservative `μ - 3σ`. Used in v2 league. |
| D6 symmetry | The hex board has dihedral group D6 symmetry (12 rotations + reflections). v2 augments training with this. |

## Architecture short-codes

| Term | Meaning |
|---|---|
| Charlesworth-style | Per the 4-player Catan paper (Charlesworth 2018?) — per-tile transformer + dev-card MHA + autoregressive heads. v2's policy borrows the *structure* but not the rules. |
| FiLM | Feature-wise Linear Modulation (Perez et al. 2018) — conditioning a layer's activations as `(1+γ) ⊙ LN(x) + β`. v2 uses for action heads that need context. |
| GNN (tripartite) | Hex/vertex/edge message-passing graph encoder. v2 Phase 2.3, 30k params. |
| Belief head | Auxiliary head predicting opponent's hidden dev-card type distribution. Trained against env GT. |
| Autoregressive heads | Each successive action head conditions on the previous head's choice. v2 has 6 heads in order: type → corner → edge → tile → res1 → res2. |
| Compact obs | Phase 1.3 — dropped bucket-8 thermometer encoding. Saved param count. |
| Karma | StackedDice's 20% forced-7 mechanic. **Persistent** — once a player rolls a 7, the *other* player has a 20% Karma chance per roll until they themselves roll a 7 (at which point the buff swaps). Engine state: `last_player_to_roll_7`, updated only on a 7, never reset by turn change. Correct obs-time predicate: `karma_buff_active(p) = (last_player_to_roll_7 is not None) AND (last_player_to_roll_7 != p)`. NOT "was the previous roll a 7." **Already encoded** in obs by `src/catan_rl/policy/obs_encoder.py:545-560` as two flags inside `current_player_main`. Colonist.io-specific. |

## Numerical anchors

- **v2 policy params**: ~1.4M (Step 3 BC config); grows to ~2.24M with all Phase 2/3 flags on.
- **v2 action space**: MultiDiscrete([13, 54, 72, 19, 5, 5]) = 6 heads, 168 total head positions; jointly factorable.
- **v2 obs**: 10 keys, mixed shapes; flat policy input is the 512-dim fused obs vector.
- **Catanatron action space**: 289 (post-refactor), up to 328 dynamic for 2-player BASE map.
- **Catanatron value-fn dominant weight**: `public_vps = 3e14` — VP outweighs everything else by ≥6 orders of magnitude.
- **QSettlers reward**: ranked 10/7/4/0; sparse, terminal-only.
- **v2 reward (Step 4 plan)**: terminal-only `±1 + (vp_diff)/15`. Faculty review correction from per-step ΔVP shaping.

## Strength benchmarks (where they live)

- **Catanatron leaderboard**: AlphaBeta(n=2) is the published top.
- **Self-regression baseline (v2-only)**: a frozen *earlier v2* checkpoint from the run's own lineage (bootstrap / early self-play snapshot), used for the Step-4 Gate-3 self-regression check. **No v1 champion is loaded** — "stronger than v1" is captured by the WR-vs-heuristic bar (v1 peaked ≈ 0.55), not by running a v1 policy.
- **v2 heuristic**: `src/catan_rl/agents/heuristic.py` — greedy 1-step. The BC target distribution and the eval anchor.
- **QSettlers**: no transferable benchmark.
