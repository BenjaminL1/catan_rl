# SYNTHESIS — v2 Catan AI Design after Multi-Agent Comparative Analysis

Date: 2026-05-14
Inputs: `analysis/00_briefing/{my_agent,catanatron,qsettlers,glossary}.md`, `round_{1,2,3,4}_*.md`.

---

## 1. Executive summary

The proposed v2 architecture is **substantially correct**: PPO + piKL anchor (Step 4) → AlphaZero-style PUCT MCTS with belief determinization (Step 5), built on a 6-head autoregressive policy with a tripartite GNN + per-tile transformer obs encoder. The four-agent debate converged on **keep** for every load-bearing v2 design choice while surfacing **two empirical deadlocks** that require launch-blocking ablations before Step 4 begins.

How v2 differs from the two prior efforts:

- **vs Catanatron** — v2 is 1v1 / 15-VP / Friendly-Robber / no-P2P-trade; Catanatron is 3-4 player / 10-VP / open trading. v2 *learns* its value head from on-policy returns; Catanatron *hand-tunes* `DEFAULT_WEIGHTS` (`catanatron/catanatron/players/value.py:10-24`). v2's belief head is unavailable to Catanatron because 4-player + P2P trade breaks perfect resource tracking. v2's factored 6-head action space shares parameters across corner/edge decisions; Catanatron's flat 289-action space does not. v2 augments with D6 symmetry; Catanatron does not. v2 uses MCTS *only* as policy improvement on top of a learned prior + value (AlphaZero recipe); Catanatron's MCTS was vanilla random-rollout and failed.
- **vs QSettlers** — v2 has one unified policy + value + belief network across all decisions; QSettlers fragmented into a trade-only DQN and an unbuilt settlement DQN. v2 uses PPO (on-policy, autoregressive-native); QSettlers used DQN with a 100-sample replay buffer in a 35%-discard-rate framework. None of QSettlers' reported numbers transfer.
- **vs the v1 attempt** — v2 adds the GNN encoder, FiLM heads, axial pos emb, belief head, opp-action aux head (Step-3 only), D6 augmentation, PFSP-hard league, duo exploiter cycles, TrueSkill ratings, opp-id embedding, and a faculty-corrected terminal-only reward. v1 plateaued at ~0.56 WR vs heuristic; v2 targets `≥ 0.70` (Gate 1) → `≥ 0.90` (Gate 1 after Step 5).

**Decisions that survived all four agents' scrutiny**: 6-head autoregressive action space, 9-key contextual masks, tripartite GNN + axial pos emb, BroadcastHandTracker + 5-way belief head, BC warm-start, curriculum opponent mix (60% → 25% heuristic), PFSP-hard sampling, terminal-only reward.

**Decisions still empirically open after Round 4**: (D2) reward `±1 + (vp_diff)/15` vs pure `±1` — launch-blocking A1 ablation needed; (D4) obs schema completeness re: dice-bag + persistent Karma buff state — launch-blocking A2 ablation needed before Step 4.

**Decisions changed by the debate**: three new probes added — value-calibration vs empirical WR (Step-4 milestone), `v_drift_l1` gating rule (after 1M of seed #1), embedding-on-vs-off ΔWR (every milestone eval).

---

## 2. Component-by-component design

### State representation
- **Keep** v2's 10-key dict obs (`bc/loader.py:89-99` schema) — tile_representations + current/next_player_main + dev counts + hex/vertex/edge graph features + opp-id ints.
- **Keep** TileEncoder (per-tile transformer + axial pos emb, Charlesworth-style) + tripartite GNN (19 hex + 54 vertex + 72 edge nodes, 2 message-passing rounds).
- **Borrow from Catanatron**: nothing in this scope. The `(channels, 21, 11)` 2D-CNN board tensor (`catanatron_env.py:62-66`) misaligns Cartesian kernels with hex topology.
- **Borrow from QSettlers**: nothing. The 83-input flat MLP has no spatial prior.
- **Open**: D4 — extend obs with `bag_remaining: (11,) float32` if the A2 ablation shows ≥ 2pp WR improvement. Defer the decision until A2 runs. **Correction (2026-05-15)**: the Karma half of the originally-proposed A2 extension is already in the obs — `current_player_main` includes two persistent Karma flags computed from `last_player_to_roll_7` (`src/catan_rl/policy/obs_encoder.py:545-560`). A2 therefore reduces to a single new field: `bag_remaining`. Field-name precision matters here — the original Round-3 framing used `opp_last_rolled_7` which both misread the Karma rule and missed that the persistent equivalent was already shipped.

### Action space
- **Keep** v2's `MultiDiscrete([13, 54, 72, 19, 5, 5])` factored 6-head autoregressive — type → corner → edge → tile → res1 → res2.
- **Keep** the 9-key contextual mask dict (`env/masks.py`): type, corner_settlement, corner_city, edge, tile, resource1_trade, resource1_discard, resource1_default, resource2_default.
- **Reject from Catanatron**: the flat 289-action space (`catanatron.md` §"Action space"). The original 5000+ space had to be compressed to be learnable; the compressed version still cannot share parameters across corner decisions.
- **Reject from QSettlers**: fragmented per-decision networks (`qsettlers.md` §"Action space + masking"). No shared value head, no transfer.

### Hidden-information modelling
- **Keep** `BroadcastHandTracker` (perfect resource tracking; legal in 1v1 with no P2P trade per `CLAUDE.md:28-29`).
- **Keep** the 5-way belief head over hidden dev-card types, trained with soft CE at weight 0.05 against env GT (`my_agent.md` §4 + §7). Required by Step-5 preflight 0.4.
- **Reject from Catanatron**: reading true opp resources in the value function (`catanatron.md` §"What's NOT in the repo"). Pragmatic for 4-player; unnecessary for 1v1 because the tracker is rule-legal.
- **Reject from QSettlers**: no opp modelling.

### Learning algorithm
- **Keep** the full pipeline: Step 3 BC warm-start → Step 4 PPO + piKL anchor (`λ=0.2`, 2M-step decay, `target_kl=0.03`, terminal-only reward) → Step 5 AlphaZero-style PUCT MCTS with `c_puct=1.5`, 25 sims × 2 determinizations, 6-bucket dice fan-out, belief-determinization sampling (`v2_step5_mcts.md` §1).
- **Borrow from Catanatron**: nothing in the algorithm itself, but **borrow the depth-3 paradox as a calibration concern**. Add a `value/calibration_l1_vs_heuristic_wr` probe to Step-4 milestone evals (per D1 compromise).
- **Reject from Catanatron**: hand-tuned value weights. v2's learned value has a path to superhuman; hand-tuned weights are bounded by tuning quality (`round_1_learning_algo.md` table).
- **Reject from Catanatron**: vanilla MCTS with random rollouts. v2 uses the AlphaZero recipe (learned prior + value at leaves), which is a different algorithm class.
- **Reject from QSettlers**: DQN. Not because modern DQN couldn't work in principle, but because (a) flat Q over the 6-head autoregressive space loses conditional structure, (b) deadly triad under sparse terminal reward, (c) no Catan-shape precedent. PPO is on-policy and trust-region-bounded — directly addresses both failures (`round_1_learning_algo.md` §2).

### Reward + evaluation
- **Keep**, pending A1: `±1 + (vp_diff)/15` terminal-only.
- **Keep**: 200 games × 2 seats × N=3 seeds = 1200-game gate floor; bootstrap 95% CI on every WR claim.
- **Keep**: PPO-BR-gap (1M BR) + Δ_BR sensitivity (5M BR comparison) as exploitability proxy.
- **Reject from Catanatron**: pure binary `+1/-1/0` reward (`catanatron_env.py:24-30`). Round 3 debate D2 didn't settle; A1 will. Default ships with margined.
- **Reject from Catanatron**: `VP_DISCOUNTED_RETURN = 0.9999^turns × VP` (per Medium postmortem, lost to RandomPlayer in supervised experiments).
- **Reject from QSettlers**: ranked 10/7/4/0 (collapses to binary in 1v1; no margin information).
- **Reject from v2's original spec**: per-step ΔVP shaping (faculty-corrected per `v2_step4_ppo.md:64`; Medium postmortem's END_TURN dominance argument is the citation).

### Training regime
- **Keep** all of: BC warm-start → curriculum 60% → 25% heuristic opponent mix → PFSP-hard league with 32-game sliding window → duo exploiter cycles every 1M main steps → TrueSkill ratings with σ-decay 1.001 → 8-d opp-id embedding with 40% training-time mask.
- **Add**: D5 ΔWR-emb-on-vs-off probe at every milestone eval.

### Compute discipline
- Step-3 BC: ~3-4 hours (after the loader perf fix; smoke verified at 95s for 50 games).
- Step-4 PPO per seed: 3-7 days M1 Pro CPU at v1's 25-FPS estimate (needs v2 calibration; flagged in `v2_step4_ppo.md` §7).
- Step-5 MCTS per seed: ≥ 1 week A100 cloud. Step-5 plan §7 explicit fallback if A100 unavailable.

---

## 3. What we're rejecting and why

| Rejected idea | Source | Why |
|---|---|---|
| Flat 289-action discrete space | Catanatron post-compression | Cannot share parameters across factored decisions; v2's 6-head autoregressive does. |
| `(channels, 21, 11)` CNN board tensor | Catanatron Gym mixed-rep | Misaligns Cartesian kernels with hex topology; v2's GNN encodes adjacency correctly. |
| Hand-tuned value function weights | Catanatron `value.py:10-24` | Bounded by tuning quality; cannot improve via play. The 6-orders-of-magnitude `public_vps` dominance is the *cause* of the depth-3 paradox. |
| Vanilla MCTS with random rollouts | Catanatron MCTSPlayer | No learned prior + no learned value = no signal in 200-turn episodes. v2's AlphaZero recipe is a different algorithm. |
| Pure binary `+1/-1/0` reward | Catanatron `catanatron_env.py:24-30` | Pending A1 result; if margined ties or wins, default stays. |
| `VP_DISCOUNTED_RETURN` | Collazo postmortem | Tried; lost to RandomPlayer. Time-discounted VP does not fix learnability. |
| Per-step ΔVP shaping | v2 pre-revision | Amplifies Catanatron's END_TURN dominance failure mode. Faculty-corrected. |
| Modern DQN | QSettlers + Catanatron's TensorForce | Flat Q over factored actions + deadly triad + sparse reward = wrong tool. |
| Ranked 10/7/4/0 reward | QSettlers | Collapses to binary in 1v1; no margin signal. |
| Fragmented per-decision networks | QSettlers | No shared value head, no transfer; load-bearing in why QSettlers couldn't beat its baselines. |
| QSettlers as a quantitative benchmark | QSettlers `qsettlers.md:56-60` | 35% game discard biases the surviving sample; no number is signal. |
| Removing duo exploiter / opp-id / TrueSkill for compute savings | Hypothetical reaction to budget pressure | All three are O(1) cost per PPO step; dropping recovers no meaningful compute. |

---

## 4. Open questions (the DEADLOCKs)

Ranked by impact on final strength:

### A1 — Reward margin: `±1 + (vp_diff)/15` vs pure `±1`
- **Impact**: every downstream gradient. Affects value head, MCTS leaf values, belief head's coupling to state.
- **Experiment**: Step-4 ablation, two arms with single-flag toggle on reward function. N=3 seeds × 5M PPO steps each.
- **Decision metric**: PPO-BR-gap delta. Threshold 0.05 either direction.
- **Cost**: ~7 days × 1 extra-seed-equivalent. Parallel with canonical seed.

### A2 — Obs schema: extend with `bag_remaining` (11,)

*(Karma state was originally part of A2 but is already encoded in `current_player_main` via two persistent flags computed from `last_player_to_roll_7` — see correction in §2.)*
- **Impact**: value head Markovianity. Affects PPO advantages, MCTS chance nodes, all eval calibration.
- **Experiment**: Step-3 BC ablation. Canonical obs vs extended (10 → 12 keys). N=3 seeds × 30k-game BC training each.
- **Decision metric**: WR vs heuristic at 600-game eval. Threshold 0.02.
- **Cost**: ~30 hours total (parallelisable). **Launch-blocking**: should run before Step 4 begins because changing the obs schema invalidates Step-3 checkpoints.

### Lower-priority post-Step-4 instrumentation
- The D1 `value_calibration_l1` probe — costs ~200 extra games per milestone eval. Adds Step-5 preflight 0.5.
- The D3 `v_drift_l1` gating rule — already-planned probe, costs nothing; just a STOP/RESUME branch.
- The D5 `Δ_WR_emb_on_vs_off` probe — 200 extra games per milestone. Already-existing eval surface.

---

## 5. Implementation order (start tomorrow)

1. **(today, ~1 hr)** Generate the 30k-game BC dataset using current schema → `data/bc/v1/`. Loader perf fix already in place.
2. **(today/tomorrow, ~4 hrs)** Run Step-3 BC training to convergence on the canonical schema. Save best.pt. Run Step-3 gates (NLL, WR-equivalence-via-TOST). This is the baseline against A2.
3. **(launch-blocking, ~30 hrs)** Run A2 ablation: a second BC training run with the extended obs schema (`bag_remaining: (11,) float32` added to the env, then to the dataset, then to the policy's encoder). Karma state is *already* encoded — only `bag_remaining` is the new field. 600-game eval. Decision per §4.
4. **(after A2)** Patch `v2_step4_ppo.md` §0 with A2's outcome — either the canonical schema stays, or v2.1 adopts extended obs and Step 3 is re-run.
5. **(blocks Step 4 launch)** Implement Step-4 PPO modules per the patched plan from the audit (`docs/plans/v2_step4_ppo.md` after the audit deltas landed earlier today). TDD-first per the plan's §5 preamble.
6. **(parallel to Step 4)** Add the three new probes to the trainer: `value_calibration_l1`, `Δ_WR_emb_on_vs_off`, the `v_drift_l1` gating decision at 1M steps.
7. **(Step 4 launch)** Run canonical Step 4 + A1 reward ablation in parallel. N=3 seeds for canonical; 1 seed for A1 first arm at 5M steps, then second arm.
8. **(after Step-4 Gate 1)** Decide: do Step-5 preflights pass (incl. the new 0.5 value-calibration gate)? If yes → kick off Step 5 on A100 cloud. If no → run the Step-4 diagnosis ladder before committing Step-5 compute.

---

## Provenance

- Briefing pack: `analysis/00_briefing/{my_agent,catanatron,qsettlers,glossary}.md`.
- Round 1 position papers: `analysis/round_1_{state_action,learning_algo,reward_eval,skeptic}.md`.
- Round 2 disagreement list: `analysis/round_2_disagreements.md`.
- Round 3 targeted debate: `analysis/round_3_debate.md`.
- Round 4 resolution: `analysis/round_4_resolution.md`.
- External sources: Catanatron repo (master, fetched 2026-05-14), Collazo Medium postmortem, QSettlers writeup at akrishna77.github.io.
- v2 codebase: `/Users/benjaminli/my_projects/catan_rl_v2/` (Step 3 BC verified; Steps 4, 5 planned not yet built).

No code changes recommended in this document; the deliverable is the design.
