# Catan RL v2 — Superhuman 1v1 Catan Agent Design

**Status**: design locked (post 5-expert panel revisions, 2026-05-13);
implementation in flight (Step 1-2 ✅, Step 3 next).

**Revision history**:
- 2026-05-13 — Original draft.
- 2026-05-13 — 5-expert panel (AlphaZero veteran, AlphaStar/PPO engineer,
  Cicero/Pluribus researcher, Catanatron domain expert, M1 Pro efficiency
  engineer) voted on 15 contested decisions. Consensus revisions: drop
  per-step ΔVP shaping (D9, 4-1), cut heuristic mix from 60% to 25%
  post-warmup (D8, 5-0), raise PPO target_kl 0.01 → 0.03 (D11, 3-2),
  shrink opp-id embedding 16-d → 8-d (D7, compromise), shrink MCTS to
  25×2 = 50 effective sims (D12, 4-1), compress chance-node fan-out
  (D13, 5-0), shorten routine exploitability eval 5M → 1M with a final
  5M run for paper-grade numbers (D15, 3-2). Phase B (MCTS) kept by
  3-2; budgeted heavily. See §3.7 for the full vote table.
- 2026-05-13 — **ML/game-theory faculty review** flagged
  game-theoretic imprecision and unprincipled gates. Corrections applied:
  (a) added §0 with four preflight experiments (v1-ablation, heuristic
  action distribution audit, heuristic determinism audit, network
  equivariance probe) that gate further work;
  (b) reframed Phase B as ISMCTS-style policy improvement (not Nash
  convergence) with cited prior art;
  (c) flagged PFSP-hard as empirical (not theoretically grounded) and
  cited fictitious self-play + double oracle as the principled 2p-zero-sum
  analogues;
  (d) renamed "Exploitability" → **"PPO-BR-gap"** throughout — a PPO-
  trained BR is a *lower* bound on true exploitability, not an upper
  bound (Cicero/Pluribus use CFR for this reason; CFR-on-late-game-
  abstraction added as future work);
  (e) all WR metrics now require **N ≥ 3 training seeds** and are reported
  **seat-stratified** (P1, P2, symmetrized average) — the residual 1v1
  seat asymmetry from snake-draft + 2nd-settlement-resource is real;
  (f) AlphaBeta benchmark split: cheap d=2 sanity at every milestone +
  expensive d=4 at the final gate (d=2 alone is too weak a baseline);
  (g) added §3.8 ablation budget with leave-one-out configs for the
  ~10 v2 features (without ablations no claim is defensible);
  (h) dropped the param-count rationale in §3.1 (parameter count is a
  poor capacity proxy; the real capacity gate is the BC generalization
  gap measurement);
  (i) Step 3 BC plan: replaced guessed gate thresholds (top-1 ≥ 0.60,
  WR ≥ 0.40) with **NLL-gap-vs-measured-baseline** and **WR-relative-
  to-heuristic-self-WR**; symmetry-aug prob choice deferred to E0.3+E0.4
  result; added empirical KL-drift probe for the BC value head.

**Target**: superhuman 1v1 Catan (Colonist.io ruleset) — defined as ≥ 0.95 WR
against the engine heuristic, ≥ 0.70 WR against a strong AlphaBeta-d2
baseline (per Catanatron precedent), exploitability < 5% from a fresh
5M-step adversary.

**Starting point**: Charlesworth's 4-player Catan deep-RL bot
(settlers-rl.github.io, 2021-2022), the closest published prior art. This
doc reconstructs his design, identifies the 4-player-specific bloat, drops
or modernizes every component, and pins down the v2 build.

---

## 0. Preflight experiments (block further implementation past Step 2)

Faculty review (2026-05-13) flagged that several design decisions rely on
*post-hoc* rationales (the "47% plateau caused by ΔVP shaping") and on
*unmeasured* baselines (the BC gate at top-1 acc ≥ 0.60). These four
experiments calibrate the plan against measurement, not rhetoric. Each is
< 1 day of work. They run **concurrently** with Steps 1-2 finalisation;
their results gate the Step 3 acceptance thresholds and the Step 4 reward
shaping decision.

### E0.1 — v1-checkpoint plateau ablation (priority)

**Question**: is the v1 47%-WR plateau caused by (a) heuristic over-mix +
ΔVP shaping (current panel hypothesis), or by something deeper that v2
inherits?

**Method**: from the latest v1 checkpoint (`catan_rl/checkpoints/train/`),
resume one PPO run with `heuristic_opp_weight=0.25` and `vp_shaping=off`,
everything else held constant. Train 5M further steps. Eval every 100k
steps.

**Decision rule**:
- If heuristic-WR climbs past 0.52 within 3M steps → v1's plateau was
  the panel-diagnosed mix + shaping. v2's D8 + D9 decisions are
  vindicated and the existing plan continues.
- If WR stays ≤ 0.49 after 5M → the panel diagnosis was wrong. Open
  a follow-up investigation before committing more v2 work.

The existing v1 training process (running unsupervised since 2026-05-12)
is *not* this experiment — its config doesn't isolate the two variables.
A clean controlled resume is required.

### E0.2 — Heuristic action distribution audit

**Question**: what are the actual baseline numbers for the BC gate?

**Method**: run 1,000 `heuristicAIPlayer` self-play games. For each
(state, action) pair, record per-head:
- The action's **marginal frequency** (e.g., `P(type = EndTurn)`,
  `P(corner = vertex_17)`).
- The action's **conditional entropy** given the mask
  (`H(action | legal action set)`).
- The action's **top-1 baseline accuracy** under "always predict the
  most frequent legal action" and under "always predict the heuristic's
  modal action class."

**Decision rule**: Step 3's BC gate threshold of "top-1 type-head
accuracy ≥ 0.60" is replaced by **"top-1 type-head NLL gap ≥ 0.3 nats
better than the measured frequency-baseline policy."** Number 0.3 nats
corresponds to ≈ 26% relative likelihood improvement, large enough to
exceed sampling noise on a 10%-holdout val set at our data scale.
Other head thresholds are calibrated analogously.

### E0.3 — Heuristic determinism audit

**Question**: does the heuristic's action selection have D6-symmetric
behaviour, or does it have deterministic tiebreakers (Python dict
iteration order, fixed argmax over ties, etc.) that would make
state-only D6 augmentation produce inconsistent labels?

**Method**: instrument `heuristicAIPlayer.move()` to log every
decision's *candidate set* (the set of actions evaluated as "tied
best"). Over 200 games:
- Fraction of decisions where the candidate set has size > 1.
- Fraction of those where the chosen action is the lexicographically
  first (by Python iteration order) — i.e. a deterministic tiebreaker.

**Decision rule**:
- If deterministic-tiebreaker fraction < 5% → state+action D6
  augmentation at prob=1.0 is safe (the panel majority's position).
- If 5-20% → fall back to **prob=0.5** (the OAI/M1 minority position).
- If > 20% → consider replacing tiebreakers with random tiebreakers
  in the heuristic before BC data generation. Otherwise the BC anchor
  inherits Python-dict-order artefacts.

### E0.4 — Network symmetry-equivariance probe

**Question**: at initialisation and after BC, is the v2 network
approximately D6-equivariant?

**Method**: instantiate `CatanPolicy()` at random init. Build a single
obs `s` and its D6-rotation `T(s)`. Forward both:
- `||π(s) - T⁻¹(π(T(s)))||₁` over the type head.
- Same for the value head and belief head.

Repeat at the end of BC training (Step 3 deliverable). Compare.

**Decision rule**: the *axial positional embedding* (which we added to
the TileEncoder, §3.1) is by construction *not* equivariant — it
attaches a learned vector to each axial coord. The probe quantifies
how much equivariance the architecture *loses* vs the GNN component
(which is approximately equivariant). The result is used to set the
symmetry-aug prob at Step 4 PPO training: high equivariance → prob=0.5
is enough; low equivariance → prob=1.0 to force the policy to learn
the invariance from data.

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
| Opp-id embedding | NONE | **Keep, 8-dim** (panel compromise — was 16-d) | Phase 3.6; lets the policy switch personas. Panel 2-3 split on keep vs drop — 8-d cuts the parameter cost in half while preserving the signal, with `opp_id_mask_prob=0.40` providing the regularization. |
| Symmetry augmentation (D6) | NONE | **Keep, prob=1.0** | Phase 1.5; free 1.5-2× effective data, correctness-preserving. |

**Net architecture: ~1.4M parameters as a starting point.** This is *not*
a principled capacity target — parameter count is a poor proxy for
effective capacity (faculty review). The real capacity decision is
deferred to a **generalization-gap audit** at the BC gate (Step 3): if
train-vs-val NLL gap is small, the network can absorb more capacity;
if large, we're already at the capacity ceiling for the data scale.
The current 1.4M number bounds the M1 Pro CPU forward-pass latency for
MCTS at the planned sim budget, which is the *hard* upper constraint;
within that, the per-component widths are calibrated by ablation
(§3.8), not by matching Charlesworth.

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
   - Loss: `PPO_loss + λ · KL(π || π_BC)` with **λ = 0.2 → 0 linearly over 2M steps**
     (panel revision; was 0.1→0 over the full 5M).
   - Keeps the policy close to the BC anchor for the first ~40% of training,
     then lets RL diverge. Higher initial λ buys more stability against
     early collapse off the BC prior; the shorter decay window means the
     anchor doesn't drag the policy past its peak.
3. **League** with PFSP-hard sampling, latest-policy reg, duo exploiter cycles.
   Drop Nash pruning (too slow on CPU; FIFO eviction is fine at this scale).

#### Phase B — Search-augmented policy improvement (target: heuristic-WR ≥ 0.90)

**Theoretical framing** (faculty review, 2026-05-13): Catan is an
extensive-form game with **imperfect information** (hidden dev cards) and
**chance nodes** (dice + dev-card deck). AlphaZero's convergence guarantees
are for perfect-information, deterministic-move games — they do *not*
transfer here. The closest cited prior art is **Information Set Monte
Carlo Tree Search** (Cowling, Powley, Whitehouse, *Information Set Monte
Carlo Tree Search*, IEEE TCIAIG 2012), which is a heuristic policy-
improvement method, not a Nash-convergent algorithm. The CFR-family
methods (Counterfactual Regret Minimization, Zinkevich et al. 2007;
DeepStack; Libratus; Pluribus) are the actually-Nash-convergent
imperfect-info game algorithms.

What Phase B claims: **MCTS used as a policy-improvement operator on top
of a strong learned prior** (the post-piKL PPO policy). It is *not*
claimed to converge to Nash, and it is *not* a substitute for CFR. The
empirical hypothesis is that ISMCTS + the trained value/policy gives a
meaningful WR boost vs the policy alone — same hypothesis Charlesworth's
optional root-only forward search verified (his run-time search gained
~90% relative WR over the bare RL policy).

1. **Add `CatanGame.copy()` to the engine.** Required for multi-step search.
   ~200 LOC; deepcopy of board + player + tracker state + RNG state.
2. **PUCT-MCTS with chance nodes** (panel revised — much smaller budget):
   - Action nodes have chance children for dice rolls; **compressed to 6 buckets**
     `{2-3, 4-5, 6, 7, 8-9, 10-12}` weighted by StackedDice's actual remaining-bag
     distribution (not the i.i.d. 2d6 prior). The 7 stays a singleton because of
     the Karma 20% override and the discard/robber subphase that follows.
     11-way fan-out at every chance node was a tree-budget catastrophe on CPU.
   - **25 sims × 2 determinizations = 50 effective sims per decision** (was 200).
     The panel split 4-1 on this — only the AlphaZero veteran wanted more, the
     other four pointed out that 200 effective sims on M1 Pro CPU collapses FPS
     by 10×. Start at 50; profile; scale up *only if* an A/B shows ≥ 5% WR gain
     over policy-alone.
   - PUCT formula: `Q + c_puct · P · √N / (1+n)`, c_puct=1.5.
   - **Use the trained policy as PUCT prior** and **trained value head as
     leaf evaluator** — no rollouts, AlphaZero-style.
   - **Belief-determinization** for opponent dev cards: sample from belief
     head's predicted distribution per determinization. The Cicero panelist
     noted: feed the belief-head posterior in directly, not just as an aux loss.
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
| target_kl | **0.03** (panel revision; was 0.01) | Andrychowicz 2020 + panel 3-2 vote: 0.01 is too conservative for a 13-head autoregressive policy with a piKL anchor already doing extra regularization; early-stop would fire on update 2 every time. 0.03 lets PPO actually move; piKL prevents drift. |
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
| Reward shaping | **Terminal only: ±1 + (vp_diff)/15** (panel revision — dropped per-step ΔVP) | 4-1 panel consensus (Catanatron the lone dissenter). Per-step ΔVP shaping biases the policy toward greedy early VP accumulation and away from late longest-road/largest-army timing plays that win 1v1 Catan. The previous 47%-WR plateau showed this exact pathology. γ=0.998 with GAE λ=0.95 handles credit assignment across long episodes; let it do its job. |

### 3.4 Self-play

- League maxlen=100, **FIFO eviction** (not Nash — too slow on CPU).
- Add to league every 4 updates.
- **Curriculum opponent mix** (panel revision — unanimous 5-0 that the original
  60% heuristic was the single biggest plateau risk; all five experts flagged
  it as either their KEY FLIP or a clear consensus item):
  - **First 2M steps (warmup)**: 60% heuristic / 25% league / 10% self-latest / 5% random.
    Heavy heuristic exposure while the BC anchor is still meaningful and the
    league is small.
  - **After 2M steps (steady state)**: 25% heuristic / 50% league / 20% self-latest / 5% random.
    Heuristic capped low enough that the policy isn't optimizing "beat one
    fixed style"; the bulk of gradient comes from league + latest-self diversity.
  - **Charlesworth had ZERO heuristic in the opponent mix.** v1 used 20% the
    whole run and plateaued at 47%; v2 keeps a low-mix anchor so the metric we
    eval against is also in the training distribution, but doesn't dominate.
- PFSP-hard with `(1-w)^p`, p=2.0, sliding 32-game window.
  - **Theoretical caveat** (faculty review): PFSP was developed for
    AlphaStar (asymmetric multi-player StarCraft II) and is an
    *empirical* opponent-sampling heuristic. The theoretically-grounded
    2p-zero-sum analogue is fictitious self-play (Heinrich & Silver 2015)
    or double oracle (McMahan, Gordon, Greenwald 2003). Both converge to
    Nash in the limit; PFSP does not. We use PFSP because it's
    well-tooled and empirically effective, not because it has the right
    convergence theorem.
- Duo exploiter cycles every 1M main steps, 32 PPO updates per cycle.
- TrueSkill ratings + σ-decay 1.001 per update.
- Opp-id embedding (8-dim, panel-revised): kind ∈ {unknown, random, heuristic, self_latest, league, exploiter}, 40% mask prob.

### 3.5 Search (Phase B+)

| Param | Value | Source |
|---|---|---|
| c_puct | 1.5 | AlphaZero |
| n_sims_per_det | **25** (panel revision; was 50) | M1 Pro wall-clock budget |
| n_determinizations | **2** (panel revision; was 4) | **50 effective sims/decision** total |
| Temperature on visit count | 1.0 during training, 0.1 (near-argmax) at eval | AlphaZero |
| Dirichlet noise at root | α=0.3, ε=0.25 | AlphaZero |
| Chance node sampling | **Compressed 6-bucket fan-out** (panel revision; was 11-way) | `{2-3, 4-5, 6, 7, 8-9, 10-12}` weighted by StackedDice's remaining-bag distribution. The 7 stays separate (Karma override + robber subphase). |
| Belief-determinization source | Belief head posterior at the root | Cicero panelist: feed the trained 5-way belief logits directly as the sampling prior for opp dev cards. Use the trained head, not the uniform prior. |
| Scale-up trigger | A/B WR delta ≥ 5% vs policy-alone | Only raise n_sims if search clearly earns it. |

### 3.6 Eval

**Variance discipline** (faculty review): every reported WR runs with
**N ≥ 3 training seeds** (config: `seed_offset=0,1,2`). Single-seed
numbers below ±0.05 of any threshold are not distinguishable from
sampling noise on 100-game evals (binomial 95% CI ~ ±0.10 at p=0.5).
A 200-game eval with N=3 seeds gives effective ±0.04 CI on the seed-
averaged WR.

**Seat-stratification** (faculty review): 1v1 Catan is not exactly
symmetric — Player 2 places the 2nd settlement second (gets to "react")
*and* receives starting resources from it. Report WR separately for
**P1-seat** and **P2-seat** plus the symmetrized two-game average. The
Step 1 smoke gate showed 647-vs-39 wins between seats under uniform
random play; that residual is partly P-seat advantage, not all noise.

- **Heuristic bench**: 100 games every 100k steps, eval_games=100,
  N=3 seeds. Report P1-seat, P2-seat, and symmetrized average.
- **Champion bench**: 5 historic checkpoints + 200 games each at major
  milestones. N=3 seeds.
- **AlphaBeta bench**:
  - **AlphaBeta-d2** (CI-friendly): 100 games every 500k steps. Cheap,
    fixed-strength sanity check. Beating d=2 is **necessary but not
    sufficient** for "strong agent" — faculty review correctly notes
    that classical Catanatron-style engines run at d=4-6 with α-β
    pruning + opening books.
  - **AlphaBeta-d4** (milestone-only): 100 games every 5M steps and at
    final-checkpoint. This is the actually-strong classical baseline.
- **PPO-BR-gap** (faculty-corrected from "Exploitability"):
  - **What it measures**: train a fresh 1M-step PPO best-response
    adversary against a frozen policy snapshot; report the gap between
    the adversary's WR and 0.5.
  - **What it does NOT measure**: true exploitability. A PPO-trained BR
    is an *approximate* BR; if PPO under-trains, the gap is a **lower
    bound** on true exploitability, not an upper bound (Cicero / Pluribus
    use CFR-family algorithms specifically because they yield true BRs
    in imperfect-info games). Calling this "exploitability" is an
    overclaim the prior plan made; the corrected name is used everywhere.
  - **Routine**: 1M-step PPO-BR against snapshots every 5M main steps.
  - **Final**: 5M-step PPO-BR against the frozen final checkpoint.
  - **Future work**: implement a depth-limited CFR pass on a tabular
    abstraction of the late-game state for a real exploitability bound.
    Tractable in 1v1 Catan because the branching factor is small after
    action masking. Deferred to Phase C+.
- **TrueSkill within league**: continuous.
- Async eval via subprocess (Phase 4.2 v1 design — proven to work).

### 3.7 Expert-panel consensus (2026-05-13)

Five experts were polled in parallel on 15 contested decisions. Each took a
distinct worldview and voted independently. The table records each vote
plus the consensus pick; the design above reflects the consensus.

| # | Decision | AZ vet | AlphaStar/PPO | Cicero/Pluribus | Catanatron | M1 Pro eff | **Consensus** |
|---|---|---|---|---|---|---|---|
| D1  | Phase B MCTS exists at all | KEEP | DROP | KEEP | KEEP | DROP | **KEEP** (3-2; budget heavily) |
| D2  | BC warm-start | BC | BC | BC | BC | BC | **BC** (unanimous) |
| D3  | piKL anchor on PPO loss | KEEP | KEEP | KEEP | KEEP | DROP | **KEEP** (4-1) |
| D4  | Network size 1.38M | KEEP | KEEP | KEEP | KEEP | SMALLER | **KEEP** (4-1) |
| D5  | Tripartite GNN | KEEP | DROP | KEEP | KEEP | DROP | **KEEP** (3-2; ablate later) |
| D6  | Belief head | KEEP | KEEP | KEEP | KEEP | KEEP | **KEEP** (unanimous) |
| D7  | Opp-id 16-d emb | DROP | CUT-8 | KEEP | KEEP | DROP | **CUT TO 8-DIM** (compromise) |
| D8  | 60% heuristic mix | LESS | LESS | LESS | LESS | LESS | **LESS** (unanimous; curriculum 60→25%) |
| D9  | Per-step ΔVP reward shaping | TERMINAL | TERMINAL | TERMINAL | KEEP | TERMINAL | **TERMINAL ONLY** (4-1) |
| D10 | D6 symmetry aug prob=1.0 | KEEP | KEEP | KEEP | KEEP | KEEP | **KEEP** (unanimous) |
| D11 | target_kl=0.01 | HIGHER | HIGHER | KEEP | KEEP | HIGHER | **HIGHER (0.03)** (3-2) |
| D12 | 200 effective MCTS sims | MORE | FEWER | FEWER | FEWER | FEWER | **FEWER (50 effective)** (4-1) |
| D13 | 11-way chance fan-out | COMPR | COMPR | COMPR | COMPR | COMPR | **COMPRESSED 6-bucket** (unanimous) |
| D14 | AlphaBeta-d2 bench | KEEP | KEEP | KEEP | KEEP | KEEP | **KEEP** (unanimous) |
| D15 | 5M-step exploitability eval | KEEP | SHORTER | KEEP | SHORTER | SHORTER | **SHORTER (1M routine, 5M paper-final)** (3-2) |

**Cross-panel KEY FLIPs** (each expert named the one decision they'd fight
hardest to change vs the original plan):

- D9 (drop ΔVP shaping) — 2 votes (AlphaZero veteran, Cicero/Pluribus).
- D1 (drop Phase B entirely) — 2 votes (AlphaStar/PPO, M1 Pro eff).
- D8 (cut heuristic mix) — 1 vote (Catanatron).

D9 carried because Cicero and AZ converged from opposite worldviews:
shaping biases the value head into a myopic VP-rate predictor, and the
previous v1 47%-plateau showed exactly that pathology (racing to 10 VP
and losing to a Largest Army timing flip).

D1 didn't flip — Phase B kept by 3-2 — but the dissent earned the
massive budget cut on D12 (50 effective sims, not 200) and the
"scale-up only if A/B justifies" gate.

**HIGH CONFIDENCE picks** (each expert's most-certain vote):

- D1 keep-B (AZ vet): "every 2-player zero-sum game solved at superhuman
  level in the last decade used decision-time search."
- D10 keep-symm-at-1.0 (AlphaStar/PPO, Catanatron): "D6 is a correctness
  property of the hex board, not a hyperparameter."
- D2 BC (Cicero): "no universe where 50k heuristic games of supervised
  pretraining is worse than another cold PPO run."
- D9 terminal-only (M1 Pro eff): "per-step ΔVP shaping in Catan is a known
  footgun."

### 3.8 Ablation budget (faculty review)

v2 ships ~12 architecture / training features (axial pos emb, GNN, FiLM
heads, belief head, opp-id emb, value-head-BC, symmetry aug, piKL anchor,
duo exploiter, MCTS chance buckets, …). Without per-feature ablations
we cannot:
  (a) diagnose failure (which feature is the culprit when the gate fails?),
  (b) defend the design (which features are *load-bearing* vs cosmetic?),
  (c) publish (no claim is supportable without ablation evidence).

**Leave-one-out config matrix.** Each row is a YAML config that ablates
exactly one v2 feature, holding everything else constant. Run after
Phase A converges; budget ~3-7 days compute per config on M1 Pro at
5M steps each.

| Config | Feature ablated | Hypothesis under test |
|---|---|---|
| `phase_a_no_bc.yaml` | BC warm-start (cold start PPO) | BC is load-bearing for the gate |
| `phase_a_no_piKL.yaml` | piKL anchor (λ=0) | piKL prevents early collapse off BC anchor |
| `phase_a_no_belief.yaml` | Belief head (weight 0) | Aux supervision contributes to ceiling |
| `phase_a_no_oppid.yaml` | Opp-id embedding | Opp-id ≠ cosmetic at 8-dim |
| `phase_a_no_gnn.yaml` | Tripartite GNN | GNN is load-bearing vs transformer-only |
| `phase_a_no_axial.yaml` | Axial pos emb in TileEncoder | Pos emb is load-bearing vs permutation-equivariant transformer |
| `phase_a_no_film.yaml` | FiLM heads (revert to concat) | FiLM ≠ concat at same param count |
| `phase_a_no_symm.yaml` | D6 symmetry aug | Aug is a real data multiplier |
| `phase_a_no_heur_anchor.yaml` | 25% heuristic in opp mix (0% instead) | Anchor helps the metric we eval against |
| `phase_a_no_exploiter.yaml` | Duo exploiter cycles | Exploiter improves vs PFSP-only league |

**Acceptance rule for "feature is load-bearing"**: ablated config WR is
≥ 0.05 below full config WR on the heuristic bench, with N ≥ 3 seeds,
across at least the last 1M training steps of each run. Features that
fail this bar are flagged as **candidates for removal** in v2.1.

**Phase B ablations** (Phase B onward):
- `phase_b_no_search.yaml` — policy-only baseline (no MCTS at decision time).
- `phase_b_full_chance.yaml` — 11-way chance fan-out vs 6-bucket (D13).
- `phase_b_more_sims.yaml` — 200 effective sims vs 50 (D12 dissent).

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

Full plan in `docs/plans/v2_step3_bc.md` (post-panel revision 2026-05-13).
Headline picks: **30k games** (70% canonical / 30% perturbed heuristic), skip
forced moves, hard CE with per-head relevance weighting, value head BC at
weight 0.1, belief head BC at weight 0.05, D6 augmentation at prob=1.0 with
both-state-and-action transform, AdamW + constant 3e-4 + 500-step warmup,
batch=1024, train to convergence (patience=3).

**Compound gate** (5-0 rejected the original `WR ≥ 0.45 vs heuristic`):
val NLL plateau + held-out top-1 type-head accuracy ≥ 0.60 + WR ≥ 0.40 vs
heuristic in 200-game eval. The BC anchor's job is to be a useful prior for
piKL, not to win outright.

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
- Champion bench + AlphaBeta-d2 + AlphaBeta-d4 bench + PPO-BR-gap test all green.
- **Gate**: symmetrized WR ≥ 0.95 vs heuristic AND AlphaBeta-d4 ≥ 0.55 AND
  PPO-BR-gap < 0.55 (all with N≥3 seeds).

---

## 6. Concrete success criteria (v2 = success when ALL pass)

All metrics reported as the **symmetrized average across P1 and P2 seats**,
with **N ≥ 3 training seeds**. Single-seat or single-seed numbers do not
count for the gate (faculty review: 100-game CI ~ ±0.10 is too wide to
trust at single-seed).

- [ ] **Symmetrized WR ≥ 0.95** vs engine heuristic over 200 games (per seed).
- [ ] **Symmetrized WR ≥ 0.55** vs Catanatron AlphaBeta-**d4** over 100 games.
      Lowered from the prior 0.70-vs-d2 target — d=4 is a genuinely strong
      classical baseline (Catanatron's measured d=4 beats d=2 by ~15-20pp);
      0.55 vs d4 is a more honest superhuman gate than 0.70 vs d2.
- [ ] **AlphaBeta-d2 reference**: symmetrized WR ≥ 0.85 (sanity that we didn't
      regress on the cheap baseline while pushing on the hard one).
- [ ] **PPO-BR-gap < 0.55** from a fresh 5M-step PPO adversary against the
      frozen final policy. Reminder (§3.6): this is a lower bound on
      exploitability, not a true bound. Reported as "PPO-BR-gap," not
      "exploitability."
- [ ] Beats the v1 archive checkpoint (`archive/phase4-may2026`) ≥ 0.70
      symmetrized in head-to-head, N=3 seeds.
- [ ] Compute budget: ≤ 4 weeks on M1 Pro CPU + 1 week on a cloud A100 (Phase B).
- [ ] **Optional / stretch**: depth-limited tabular CFR on a late-game
      abstraction shows ≤ X% true exploitability. Deferred to Phase C+.

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
- [Cowling, Powley, Whitehouse, *Information Set Monte Carlo Tree Search*,
  IEEE TCIAIG 2012](https://ieeexplore.ieee.org/document/6203567) — the
  honest theoretical framing for what Phase B is (heuristic policy
  improvement in imperfect-info EFGs, not Nash convergence).
- [Zinkevich et al., *Regret Minimization in Games with Incomplete
  Information*, NeurIPS 2007](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) —
  CFR; the actually-Nash-convergent imperfect-info game algorithm. Cited
  for the "future work" CFR-on-late-game-abstraction direction.
- [Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games*, arXiv 1603.01121](https://arxiv.org/abs/1603.01121) —
  fictitious self-play; the 2p-zero-sum theoretical analogue of PFSP.
- [Hilton, Cobbe, Schulman, *Batch size-invariance for policy optimization*,
  arXiv 2110.00641](https://arxiv.org/abs/2110.00641) — actual RL scaling-law
  reference (vs the misapplied Chinchilla framing the BC plan originally
  used).
- [Cobbe et al., *Phasic Policy Gradient* (procgen scaling appendix)](https://arxiv.org/abs/2009.04416) —
  same.
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
