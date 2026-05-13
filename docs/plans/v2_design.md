# Catan RL v2 — Superhuman 1v1 Catan Agent Design

**Status**: design locked, implementation begins next.

**Target**: superhuman 1v1 Catan (Colonist.io ruleset) — defined as ≥ 0.95 WR
against the engine heuristic, ≥ 0.70 WR against a strong AlphaBeta-d2
baseline (per Catanatron precedent), exploitability < 5% from a fresh
5M-step adversary.

**Starting point**: Charlesworth's 4-player Catan deep-RL bot
(settlers-rl.github.io, 2021-2022), the closest published prior art. This
doc reconstructs his design, identifies the 4-player-specific bloat, drops
or modernizes every component, and pins down the v2 build.

---

## 1. Charlesworth's design — one-page summary

Network (~1.2-1.4M params total):

```
Per-tile features (60d)         dev-card sequence (variable)
        │                              │
TileEncoder transformer       Dev-card embedding (vocab=6, d=16)
2 layers × 4 heads × d=64                │
        │                       MultiHeadedAttention 4 heads
   per-tile proj 25                      │
        │                          sum-pool + proj 25
   19 × 25 = 475 flat                    │
        │                                │
        └──────┬─────────────────────────┘
               │
   ┌────── concat 987 ──────┐
   │           │            │
CurrentPlayer  Other × 3    (with min/max bounded
MLP (152→128)  MLP (159→128) opponent hand bounds)
   │           │            │
   └────── Fusion Linear 987 → 512 + LN + ReLU ──────┐
                                                     │
                                              ┌──────┴──────┐
                                       Value 512→256→128→1   12 action heads
                                       (norm mean=150 std=150) autoregressive
```

Action space: **13 action types** × multiple resource/corner/edge/tile/player
sub-heads, **including a recurrent trade head that calls up to 4 times per
decision** to assemble a multi-resource trade offer.

Algorithm:
- PPO with γ=0.999, λ=0.95, clip=0.2, value clipping, PPO epochs=10
- Adam LR 3e-4 linear decay
- Entropy 0.04 → 0.005 annealed updates 500-1500
- Value normalizer running mean/std init (150, 150)
- 128 processes × 5 envs × 200 steps = **128k decisions / update**
- ~3,500 updates ≈ 450M decisions total
- 32-core + 3090 + 128GB RAM, ~1 month wall clock

Self-play: **league of 500 FIFO-evicted past policies**, add every 4 updates,
opponent sampled with **recency bias only** (no PFSP, no exploiter, no
TrueSkill).

Inference: optional **root-only PUCB-style forward search** with 10s thinking
time, random determinization of opponent hidden state. **+90% relative win
rate over bare RL policy.**

Author-reported results:
- Game length trended down throughout training (still improving at end)
- Bare RL "definitely learned some stuff" but "not yet superhuman"
- Forward search: 47/100 vs 3 copies of final policy (vs 25% random baseline)
- **Documented failure modes**: entropy collapse on action-type head, broken
  trading (impossible-trade proposals + self-favoring trade ratios), weak
  opening placements without search.

---

## 2. 4-player → 1v1 simplifications (free wins)

| Charlesworth 4p | v2 1v1 Colonist.io | What we save |
|---|---|---|
| 10 VP win | **15 VP** | Engine already correct. Adjust value normalizer init mean ~250 instead of 150. |
| P2P trading enabled | **DISABLED** (bank only) | **Drop 4 of 12 action heads** (accept/reject, player_head, propose_give, propose_receive). Eliminates the entire "dead head" problem that plagued his agent. |
| 3 opponents | **1 opponent** | Drop 2/3 of OtherPlayers modules. Drop player_head (no opponent to target with steals/trades). |
| Min/max bounded opp resources | **Exact opp resources** (BroadcastHandTracker) | Drop 2×40 dims of obs encoding. Only opp dev-card *type* remains hidden — that's what the belief head targets. |
| 7-discard threshold | **9-discard** | Engine handles. |
| No Friendly Robber | **Friendly Robber** (no robber on hex adjacent to player with <3 visible VP) | Engine handles. Reduces tile mask. |
| Independent 2d6 | **StackedDice** + Karma | Engine handles. Lower variance → may permit slightly lower γ. |
| 4-player snake setup | **2-player snake** (1-2-2-1) | Setup phase has half as many decisions; opening matters proportionally more. |
| 12-head action structure | **6-head** (type, corner, edge, tile, res1, res2) | Done. |

**Charlesworth's biggest design mistake** — unmasked impossible trades that
broke the trade head — **does not exist for us** because the ruleset removes
P2P trading entirely.

---

## 3. Modern improvements over Charlesworth (per-component verdicts)

### 3.1 Architecture

| Component | Charlesworth | v2 decision | Rationale |
|---|---|---|---|
| TileEncoder transformer | 2 layers × 4 heads × d=64, ReLU, no pos-emb, no dropout | **2 layers × 4 heads × d=96, GeLU, dropout 0.05, axial pos-emb** | GeLU + LayerNorm pre-norm is the modern recipe; dropout 0.05 is fine at small d; axial pos-emb breaks tile-permutation equivariance (Catan board has a fixed canonical layout). |
| Dev-card encoder | MHA + sum-pool | **Count encoder (bincount → MLP)** | Permutation-invariant, ~30k fewer params, simpler. Cards in hand are a multiset, MHA was over-engineering. |
| Graph encoder (hex↔vertex↔edge) | NONE | **Lightweight GNN, 2 rounds of message passing, d=64** | Catan's topology is a tripartite graph; passing it implicitly via per-tile concatenation throws away spatial structure. Modern boardgame RL (chess GNN, hex DeepRL) confirms gains. ~30k params. |
| Player modules | 152→128, 3 separate copies | **One 61→128 module (1 opponent)** | 1v1 simplification. |
| Opponent resource encoding | Min/max bounds (2×40 dims) | **Exact counts (8 dims)** | BroadcastHandTracker gives ground truth. |
| Fusion | 987 → 512 → LN → ReLU | **475 + 128 + 128 + 64 + 16 = 811 → 512 → LN → ReLU** | Same target bottleneck, smaller input due to dropped opponents. |
| Action head conditioning | concat one-hot into MLP | **FiLM/AdaLN** `(1+γ)·LN(x)+β` with γ-init=0 | Identity at init (safe); strictly dominates concat at same parameter count (StyleGAN/AdaLN-DiT precedent). |
| Value head | 512→256→128→1, shared encoder | **512→256→128→1, shared encoder** | Phase 2.5 decoupled tower added ~700k params and was never shown to help. Drop it; the policy and value losses sharing the trunk is the AlphaZero default. |
| Recurrent value GRU | none | **none** | Phase 4.2 single-step BPTT was academically unusual and added 150k params for unclear value. Catan state is observable enough not to need recurrence. |
| Belief head (opp dev-card type) | NONE | **Keep, 5-way soft-CE on env ground truth** | The only hidden info in 1v1; auxiliary supervision is free. Phase 2.5b kept. |
| Opp-action head | NONE | **DROP** | Phase 2.5c added complexity for weak supervision; opp action is hard to predict from obs alone. |
| Opp-id embedding | NONE | **Keep, 16-dim** | Phase 3.6; lets the policy switch personas for different opponent kinds. |
| Symmetry augmentation (D6) | NONE | **Keep, prob=1.0** | Phase 1.5; free 1.5-2× effective data, correctness-preserving. |

**Net architecture target: ~1.5M parameters.** Smaller than the archived
v1's 2.74M, larger than Charlesworth's 1.2M, right-sized for the problem.

### 3.2 Training algorithm

**The single biggest change vs both Charlesworth and v1**: shift from pure
PPO to **AlphaZero-style policy iteration with MCTS at decision time**, with
PPO as the warm-start.

#### Phase A — Warm-start (target: heuristic-WR ≥ 0.50 in ≤ 24h wall)

1. **Behavior-clone the heuristic** (Cicero piKL pattern).
   - Generate 50,000 heuristic-vs-heuristic games. Extract (state, action) pairs.
   - Supervised pretrain on these labels with cross-entropy. ~30 min on M1 Pro.
   - Result: a starting policy at ~50% vs heuristic (because BC of heuristic
     IS approximately the heuristic).
2. **PPO with piKL anchor** for ~5M steps:
   - Loss: `PPO_loss + λ · KL(π || π_BC)` with λ = 0.1 → 0 linearly.
   - Keeps the policy close to the BC anchor while RL diverges to higher quality.
3. **League** with PFSP-hard sampling, latest-policy reg, duo exploiter cycles.
   Drop Nash pruning (too slow on CPU; FIFO eviction is fine at this scale).

#### Phase B — AlphaZero-style refinement (target: heuristic-WR ≥ 0.90)

1. **Add `CatanGame.copy()` to the engine.** Required for multi-step search.
   ~200 LOC; deepcopy of board + player + tracker state + RNG state.
2. **PUCT-MCTS with chance nodes:**
   - Action nodes have chance children for dice rolls (11 possible 2d6 sums
     given StackedDice's bag) and for dev-card draws.
   - 64-200 sims per decision. Connect 4 succeeds at 64; Catan's branching
     is similar after action masking.
   - PUCT formula: `Q + c_puct · P · √N / (1+n)`, c_puct=1.5.
   - **Use the trained policy as PUCT prior** and **trained value head as
     leaf evaluator** — no rollouts, AlphaZero-style.
   - **Belief-determinization** for opponent dev cards: sample from belief
     head's predicted distribution per determinization (4 determinizations,
     50 sims each → 200 effective sims).
3. **Continue training under MCTS:**
   - Sample actions from MCTS visit count distribution (not policy logits).
   - AlphaZero loss: `CE(MCTS_visits || π) + MSE(MCTS_root_value || V)`.
   - PFSP-hard league continues but MCTS plays both sides.

#### Phase C — Final polish (target: heuristic-WR ≥ 0.95, AlphaBeta-d2 ≥ 0.70)

1. **Search-budget exploiter**: periodically run an exploiter with 2× MCTS
   sims as the main agent. Pluribus insight: real-time search ≥ training-time search.
2. **Auxiliary GRP** (Suphx Global Reward Prediction): 2-layer GRU on
   `(state_t, state_{t-1}) → final outcome`, used as a per-step shaping
   `Φ(s_t) - Φ(s_{t-1})`. Replaces hand-tuned ΔVP shaping.

### 3.3 PPO recipe details (Phase A)

| Param | Value | Source |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.999), eps=1e-5 | Loshchilov & Hutter 2019 |
| LR | 3e-4 → 1e-5 linear decay over total budget | Charlesworth + Andrychowicz 2020 |
| Weight decay | 1e-4 | Standard |
| γ | **0.998** | 15-VP games are ~30% longer than 10-VP; Charlesworth used 0.999, our v1 used 0.995, 0.998 is the right middle. |
| GAE λ | 0.95 | Standard |
| Clip ε | 0.2 | Standard |
| PPO epochs | 6 (with KL early-stop) | Lower than Charlesworth's 10 for wall-time; early-stop catches over-update. |
| target_kl | 0.01 | Andrychowicz 2020: early-stop is one of the top robustness levers. |
| Minibatch size | 512 | |
| n_envs | 16 | |
| n_steps | 4096 | |
| Entropy coef | 0.04 → 0.005 over updates 200-1500 | Charlesworth's schedule + 1v1 episodes are shorter |
| Entropy floor | 0.003 with rebound | Phase 0 collapse prevention |
| Value loss coef | 1.0 | |
| Value clipping | True, clip_range_vf=0.2 | Phase 1.1 PPO2-style |
| Value normalization | Running mean/std, init mean=250 std=150 | Adjusted for 15-VP |
| Max grad norm | 0.5 | |
| Recompute returns | True (each epoch) | Charlesworth default |
| Reward shaping | +0.05 × ΔVP_agent, -0.025 × ΔVP_opp, terminal ±1 + (vp_diff)/15 | Verified-correct from v1 env |

### 3.4 Self-play

- League maxlen=100, **FIFO eviction** (not Nash — too slow on CPU).
- Add to league every 4 updates.
- Sampling mix: 60% heuristic anchor, 5% random, 10% latest-self, 25% league.
  - **Charlesworth had ZERO heuristic in the opponent mix** — direct
    gradient on the metric we care about is more valuable.
- PFSP-hard with `(1-w)^p`, p=2.0, sliding 32-game window.
- Duo exploiter cycles every 1M main steps, 32 PPO updates per cycle.
- TrueSkill ratings + σ-decay 1.001 per update.
- Opp-id embedding: kind ∈ {unknown, random, heuristic, self_latest, league, exploiter}, 40% mask prob.

### 3.5 Search (Phase B+)

| Param | Value | Source |
|---|---|---|
| c_puct | 1.5 | AlphaZero |
| n_sims_per_det | 50 | |
| n_determinizations | 4 | (effective 200 sims/decision) |
| Temperature on visit count | 1.0 during training, 0.1 (near-argmax) at eval | AlphaZero |
| Dirichlet noise at root | α=0.3, ε=0.25 | AlphaZero |
| Chance node sampling | Per StackedDice bag (11 outcomes) | Engine-aware |

### 3.6 Eval

- **Heuristic bench**: 100 games every 100k steps, eval_games=100 (binomial CI ~±0.05 at p=0.5).
- **Champion bench**: 5 historic checkpoints + 200 games each at major milestones.
- **AlphaBeta-d2 bench**: 100 games every 500k steps. Port from Catanatron.
- **Exploitability**: train a fresh 5M-step adversary against the frozen final policy.
- **TrueSkill within league**: continuous.
- Async eval via subprocess (Phase 4.2 v1 design — proven to work).

---

## 4. v2 codebase structure

```
catan_rl_v2/
├── src/catan_rl/
│   ├── engine/             ← KEPT (1v1 Colonist.io rules verified)
│   ├── agents/             ← KEPT (heuristic + random)
│   ├── viz/, gui/          ← KEPT (debugging + play-vs-model)
│   │
│   ├── env/                ← NEW: clean Gym wrapper
│   ├── policy/             ← NEW: ~1.5M-param net (no decoupled, no GRU)
│   │   ├── network.py        TileEncoder + GNN + heads
│   │   ├── obs_encoder.py    Obs dict → tensor (count-encoded dev cards)
│   │   ├── action_heads.py   6 autoregressive heads w/ FiLM conditioning
│   │   ├── belief_head.py    Phase 2.5b aux loss
│   │   └── value_head.py     Shared encoder + simple MLP value
│   ├── bc/                 ← NEW Phase A: behavior-clone heuristic
│   │   ├── dataset.py        50k heuristic-vs-heuristic games
│   │   └── pretrain.py       Supervised CE on heuristic actions
│   ├── ppo/                ← NEW Phase A: PPO + piKL anchor + KL early-stop
│   │   ├── trainer.py
│   │   ├── rollout_buffer.py
│   │   └── gae.py
│   ├── search/             ← NEW Phase B: AlphaZero MCTS w/ chance nodes
│   │   ├── puct_mcts.py
│   │   ├── chance_node.py
│   │   └── belief_determinization.py
│   ├── selfplay/           ← NEW Phase A: league + PFSP-hard + exploiter
│   │   ├── league.py
│   │   ├── pfsp.py
│   │   ├── exploiter.py
│   │   └── ratings.py
│   └── eval/               ← NEW: heuristic + champion + AlphaBeta-d2 + exploitability
│       ├── evaluation_manager.py
│       ├── alphabeta_bench.py
│       ├── champion_bench.py
│       └── exploitability.py
├── scripts/
│   ├── train_bc.py         Phase A.1: behavior-clone heuristic
│   ├── train_ppo.py        Phase A.2: PPO + piKL anchor
│   ├── train_az.py         Phase B: AlphaZero refinement
│   ├── evaluate.py         All bench harnesses
│   └── play_vs_model.py    Human play (GUI)
├── configs/
│   ├── bc.yaml
│   ├── ppo.yaml
│   └── az.yaml
├── tests/
│   ├── unit/
│   │   ├── engine/         (port from v1, light)
│   │   ├── policy/         (param counts, gradient flow, mask correctness)
│   │   ├── search/         (PUCT formula, chance node expansion)
│   │   ├── selfplay/       (PFSP weighting, exploiter cycle, ratings)
│   │   └── eval/           (regression on a checkpoint)
│   └── integration/
│       ├── test_bc_pretrain.py
│       ├── test_ppo_smoke.py
│       └── test_az_smoke.py
└── docs/plans/v2_design.md  ← this file
```

---

## 5. Implementation order

Each phase has a falsifiable gate before moving on.

### Step 1 — Engine `copy()` + clean Gym env (1-2 days)

- Add `CatanGame.copy()` (deepcopy board, players, dice bag, tracker, RNG).
- Build `src/catan_rl/env/catan_env.py`: Gym wrapper, dict obs, 6-head action.
- Smoke gate: random-vs-random plays 1000 games to termination without errors.

### Step 2 — Policy network (1-2 days)

- `src/catan_rl/policy/network.py`: TileEncoder + GNN + count dev card encoder
  + shared trunk + 6 FiLM action heads + value MLP + belief head.
- Parameter count test: ~1.5M params.
- Forward pass test: produces valid log-probs over masked action space.
- Gradient flow test: loss.backward() updates every parameter.

### Step 3 — Behavior clone (Phase A.1, 1 day + ~1h compute)

- `scripts/train_bc.py`: generate 50k heuristic games, train policy on (state, action) pairs.
- **Gate**: BC policy reaches WR ≥ 0.45 vs heuristic in 100-game eval.

### Step 4 — PPO + piKL anchor + KL early-stop (Phase A.2, 2-3 days + ~3-7 days compute)

- Custom PPO trainer with all the recipe items from §3.3.
- piKL anchor with decaying λ.
- League + PFSP-hard + duo exploiter.
- Async eval via subprocess.
- **Gate**: WR ≥ 0.70 vs heuristic at 10M steps OR 3-day wall.

### Step 5 — AlphaZero MCTS (Phase B, 2-3 days + 1 week compute)

- PUCT-MCTS with chance nodes for dice + dev-card draws.
- Belief-determinization for opponent dev cards.
- Loss: CE(MCTS_visits || π) + MSE(MCTS_value || V).
- **Gate**: WR ≥ 0.90 vs heuristic within 5M more steps.

### Step 6 — Final polish (Phase C, 1-2 weeks)

- Search-budget exploiter.
- Optional: Suphx GRP.
- Champion bench + AlphaBeta-d2 bench + exploitability test all green.
- **Gate**: WR ≥ 0.95 vs heuristic AND AlphaBeta-d2 ≥ 0.70 AND exploitability < 0.55.

---

## 6. Concrete success criteria (v2 = success when ALL pass)

- [ ] WR ≥ 0.95 vs engine heuristic over 200 deterministic games
- [ ] WR ≥ 0.70 vs Catanatron AlphaBeta-d2 over 100 games
- [ ] Exploitability < 0.55 from a fresh 5M-step adversary trained against the
      frozen final policy
- [ ] Beats the v1 archive checkpoint (`archive/phase4-may2026`) ≥ 0.70 in head-to-head
- [ ] Compute budget: ≤ 4 weeks on M1 Pro CPU + 1 week on a cloud A100 (Phase B)

---

## 7. Key references

- [Charlesworth, settlers-rl.github.io](https://settlers-rl.github.io/) — the
  baseline this design extends.
- [Charlesworth et al., Application of Self-Play RL to a Four-Player Game of
  Imperfect Information, arXiv 1808.10442](https://arxiv.org/abs/1808.10442) —
  Big 2 PPO precedent.
- [Gendre & Kaneko, Playing Catan with Cross-dimensional NN, arXiv 2008.07079](https://arxiv.org/abs/2008.07079) —
  first to beat jSettlers, used L2 policy activity loss.
- [Silver et al., AlphaZero (Science 2018)](https://www.science.org/doi/10.1126/science.aar6404) —
  PUCT-MCTS + value bootstrap.
- [Stochastic MuZero (Antonoglou 2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1) —
  chance nodes / afterstates for stochastic games.
- [Vinyals et al., AlphaStar (Nature 2019)](https://www.nature.com/articles/s41586-019-1724-z) —
  PFSP-hard, exploiter cycles, league self-play.
- [Bakhtin et al., Cicero / piKL (Science 2022)](https://www.science.org/doi/10.1126/science.ade9097) —
  KL-anchored RL for human-similar play; relevant for BC warm-start.
- [Li et al., Suphx (arXiv 2003.13590)](https://arxiv.org/abs/2003.13590) —
  Global Reward Prediction for sparse-reward games.
- [Andrychowicz et al., What Matters in On-Policy RL? (arXiv 2006.05990)](https://arxiv.org/abs/2006.05990) —
  PPO ablations; basis for KL early-stop adoption.
- [Catanatron](https://github.com/bcollazo/catanatron) — source for AlphaBeta-d2
  port for evaluation.

---

**Open this doc before each phase to validate against the gates above.**
