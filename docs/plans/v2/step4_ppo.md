# v2 Step 4 — PPO + piKL anchor (Phase A.2)

**Status**: design locked (post-panel + faculty review, 2026-05-13);
implementation queued behind Step 3 BC.

**Preflight gate** (per `v2_design.md` §0): Step 4 implementation does
**not** start until:
  - Step 3 BC has converged with Gate 1 + 2 + 3 green (per
    `v2_step3_bc.md` §6). The BC checkpoint becomes the piKL anchor.
  - E0.1 v1-checkpoint 2×2 ablation has returned a result (or been
    explicitly waived). If E0.1 shows the v1 plateau was caused by
    something **other** than `heuristic_opp_weight` + `vp_shaping`, the
    Step 4 design needs revision before continuing.

This doc is the planning equivalent of `v2_step3_bc.md` — it specifies
what gets built, how it's tested, what numbers count as success, and
where the risks are. Pulls from `v2_design.md` §3.2, §3.3, §3.4.

## Inputs

- `checkpoints/bc/best.pt` — the BC anchor from Step 3.
- v2 policy network (`CatanPolicy`) loaded from that checkpoint.
- `CatanEnv` (Phase 1.5) with obs schema + masks already plumbed.
- `BroadcastHandTracker` (perfect 1v1 opp resource tracking).
- D6 symmetry tables (Phase 1.4) + augmentation entry point.

## Outputs

- `checkpoints/ppo/<run_id>/checkpoint_*.pt` — periodic policy snapshots.
- `runs/ppo/<run_id>/` — TensorBoard logs (PPO scalars, league
  composition, eval WR by seat, exploiter cycle markers).
- `runs/ppo/<run_id>/league.json` — final league state.
- Acceptance criterion (§6) gates handoff to Step 5 (MCTS).

---

## 1. PPO recipe (per `v2_design.md` §3.3, with faculty revisions)

| Param | Value | Provenance |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.999), eps=1e-5 | Loshchilov & Hutter 2019 |
| LR | 3e-4 → 1e-5 linear decay over total budget | Charlesworth + Andrychowicz |
| Weight decay | 1e-4 | Standard |
| γ | 0.998 | Mid-point between 4p (0.999) and v1 (0.995) for 15-VP games |
| GAE λ | 0.95 | Standard |
| Clip ε | 0.2 | Standard |
| PPO epochs | 6 (with KL early-stop) | Charlesworth=10; lower for wall-time |
| **target_kl** | **0.03** (faculty re-review, was 0.01) | Andrychowicz; 0.01 was too conservative for 13-head policy with piKL already regularising |
| Minibatch size | 512 | |
| n_envs | 16 | |
| n_steps | 4096 | |
| Entropy coef | 0.04 → 0.005 over updates 200-1500 | Charlesworth + 1v1 shorter episodes |
| Entropy floor | 0.003 with rebound | Phase 0 collapse prevention |
| Value loss coef | 1.0 | |
| **Value clipping** | True, clip_range_vf=0.2 | Phase 1.1 PPO2-style |
| Value normalization | Running mean/std, init mean=250 std=150 | Adjusted for 15-VP |
| Max grad norm | 0.5 | |
| Recompute returns | True (each epoch) | Charlesworth default |
| **Reward shaping** | **Terminal only ±1 + (vp_diff)/15** (faculty review, was per-step ΔVP) | 4-1 panel + faculty: ΔVP shaping biases value head toward greedy VP-racing |

## 2. piKL anchor (per `v2_design.md` §3.2)

`PPO_loss + λ · KL(π || π_BC)` where `π_BC` is the frozen BC anchor.

| Param | Value | Provenance |
|---|---|---|
| λ_initial | **0.2** (faculty re-review, was 0.1) | Stronger early anchor against collapse |
| λ schedule | Linear decay to 0 over first **2M steps** (was 5M) | Tighter window so anchor doesn't drag past peak |
| KL implementation | Per-head sum, relevance-weighted | Matches BC loss for consistency |
| Frozen anchor | `checkpoints/bc/best.pt`, eval-mode, no grad | Standard |

**Empirical safeguard** (faculty re-review): during the first 1M PPO
steps, log `bc/v_drift_l1_per_1m_steps` — L1 difference between the
BC value head and the current value head on a rolling 1024-obs batch
sampled from current PPO-policy rollouts. If drift does *not* shrink,
**disable BC value training for the next seed** (set BC value-loss
weight to 0); see §3.2 of `v2_step3_bc.md` for the full safeguard.

## 3. League + self-play (per `v2_design.md` §3.4)

- **League maxlen=100**, **FIFO eviction** (not Nash — too slow on CPU).
- **Add to league every 4 updates**.
- **Curriculum opponent mix** (5-0 panel vote, the single biggest plateau
  risk identified):
  - **First 2M steps (warmup)**: 60% heuristic / 25% league / 10%
    self-latest / 5% random.
  - **After 2M steps (steady state)**: **25% heuristic / 50% league /
    20% self-latest / 5% random**.
- **PFSP-hard** with `(1−w)^p`, p=2.0, sliding 32-game window.
  - Flag as empirical (not Nash-convergent — see §3.4 of `v2_design.md`).
- **Duo exploiter cycles** every 1M main steps, 32 PPO updates per cycle.
- **TrueSkill ratings** + σ-decay 1.001 per update.
- **Opp-id embedding** at **8-dim** (was 16; faculty re-review): kind ∈
  {unknown, random, heuristic, self_latest, league, exploiter}, 40%
  mask prob.

## 4. Eval (per `v2_design.md` §3.6, faculty-corrected)

- **Heuristic bench**: 100 games every 100k steps, eval_games=100,
  **N=3 seeds**, **report P1 / P2 / symmetrised**.
- **Champion bench**: 5 historic checkpoints (including the preserved
  v1 `checkpoint_16162816.pt` at 0.56 WR) + 200 games each at major
  milestones.
- **AlphaBeta bench**:
  - **d=2** every 500k steps (CI-friendly).
  - **d=4** at milestones every 5M steps.
- **PPO-BR-gap** (renamed from "exploitability"): 1M-step BR adversary
  against snapshots every 5M main steps. Plus the **sensitivity probe**:
  also run 5M-step BR and report `Δ_BR = b_5M − b_1M` as a
  BR-suboptimality lower bound.
- **TrueSkill within league**: continuous.

## 5. File layout (new code under `src/catan_rl/ppo/`)

```
src/catan_rl/ppo/
├── __init__.py
├── gae.py                  GAE compute_gae(rewards, values, dones,
│                             gamma, lam) -> advantages, returns.
├── rollout_buffer.py       CompositeRolloutBuffer pre-allocated for
│                             the v2 obs dict + 6-head actions + masks
│                             + opp-id + belief target.
├── pikl_anchor.py          PikLAnchor(bc_checkpoint, lambda_init,
│                             lambda_decay_steps) — frozen BC policy +
│                             per-head KL computation +
│                             linear-decayed lambda.
├── trainer.py              CatanPPO(config, policy, env_factory,
│                             league, anchor) — outer loop, value norm,
│                             entropy floor, KL early-stop, grad clip.
├── value_normaliser.py     Running mean/std with init mean=250 std=150.
├── losses.py               compute_policy_loss(ratio, adv, clip_eps),
│                             compute_value_loss(values, returns,
│                             clip_range_vf), compute_entropy(per-head).

src/catan_rl/selfplay/
├── __init__.py
├── league.py               League(maxlen=100, fifo, pfsp_hard_p=2.0).
├── pfsp.py                 sample_opponent(league, p_self, ratings,
│                             window=32).
├── exploiter.py            duo_exploiter_cycle(main_policy, league,
│                             n_updates=32).
├── ratings.py              TrueSkill RatingTable with σ-decay 1.001.
└── game_manager.py         Multi-env opponent dispatcher + batched
                            opponent NN inference (port from v1).

src/catan_rl/eval/
├── __init__.py
├── evaluation_manager.py   Async eval via subprocess pool (port v1
│                             Phase 4.2 design).
├── heuristic_bench.py      P1/P2 stratified WR per seed.
├── champion_bench.py       Head-to-head vs preserved checkpoints,
│                             N=3 seeds × 200 games.
├── alphabeta_bench.py      Port d=2 + d=4 from Catanatron.
└── ppo_br_gap.py           Train fresh BR adversary (1M or 5M
                            steps); also compute Δ_BR sensitivity.

scripts/
├── train_ppo.py            CLI over ppo.trainer.train(...).
├── run_ppo_br_gap.py       PPO-BR adversary trainer (uses same trainer
│                             but with the main policy as frozen opp).
└── evaluate.py             All bench harnesses, JSON output.

configs/
├── ppo.yaml                Main PPO config (per §1, §2, §3).
├── ppo_br_adversary.yaml   PPO-BR adversary config (smaller league,
│                             no heuristic, no piKL).
└── ablations/              §3.8 leave-one-out matrix (Tier 1+2 first).
    ├── phase_a_no_bc.yaml          BC warm-start ablated.
    ├── phase_a_no_pikl.yaml        piKL anchor ablated (λ=0).
    ├── phase_a_heur_at_0.yaml      0% heuristic in opp mix.
    ├── phase_a_no_gnn.yaml         GNN encoder disabled.
    ├── phase_a_no_oppid.yaml       Opp-id embedding disabled.
    └── phase_a_no_bc_value.yaml    BC value loss weight = 0.

tests/
├── unit/ppo/
│   ├── test_gae.py
│   ├── test_rollout_buffer.py
│   ├── test_pikl_anchor.py
│   ├── test_losses.py
│   └── test_value_normaliser.py
├── unit/selfplay/
│   ├── test_league.py
│   ├── test_pfsp.py
│   ├── test_exploiter.py
│   └── test_ratings.py
├── unit/eval/
│   ├── test_alphabeta_bench.py
│   └── test_ppo_br_gap.py
└── integration/
    ├── test_ppo_smoke.py        1k-step run; loss goes down + no NaN
    └── test_pikl_drift_probe.py Value-drift logging works end-to-end
```

## 6. Acceptance gate

Per `v2_design.md` §5 Step 4 + faculty-corrected §6:

1. **Gate 1 — convergence**:
   - At 10M PPO steps OR 3-day wall-clock (whichever first), report
     symmetrised heuristic WR with N=3 seeds.
   - **Pass**: symmetrised WR ≥ **0.70** vs heuristic at 200-game eval.
   - **Soft-fail (≥0.55 but <0.70)**: investigate per the §8 risk
     register; consider one of:
       - Re-enable BC value loss if `v_drift_l1` indicated divergence.
       - Reduce piKL `λ_initial` if anchor is over-constraining.
       - Add heuristic-mix curriculum sub-step.
   - **Hard-fail (<0.55)**: stop, audit league composition + reward
     trajectory + value-drift probe before continuing.

2. **Gate 2 — exploitability**:
   - 1M-step PPO-BR-gap against the final Step-4 checkpoint.
   - **Pass**: BR-gap ≤ 0.65 (the BR cannot push the policy below 35%
     WR within 1M steps).
   - Reminder: PPO-BR-gap is a *lower bound* on true exploitability
     — `Δ_BR = b_5M − b_1M` sensitivity probe quantifies the gap to
     true BR.

3. **Gate 3 — champion regression** (sanity):
   - Symmetrised WR vs the preserved v1 champion
     (`/Users/benjaminli/my_projects/catan_rl/checkpoints/train/checkpoint_16162816.pt`,
     v1 peak 0.56 WR) over 200 games per seat.
   - **Pass**: WR ≥ **0.60** — Step-4 policy is at least as strong as
     v1's peak.

## 7. Compute budget

- **Phase A (BC + PPO + piKL)**: ≤ **10 days per seed**, ≤ **30 days
  across N=3 seeds** (cap from `v2_design.md` §6).
- **Step 4 itself**: ~3-7 days per seed at ~25 FPS on M1 Pro.
- **PPO-BR-gap routine eval**: 1M BR per snapshot every 5M main steps
  → 20% overhead. Reduce eval frequency to every 10M if it exceeds 25%.
- **Final 5M BR**: ~5% on the back end.
- **Tier 1 ablations** (`no_bc`, `no_pikl`, `heur_at_0`): 3 configs × 5M
  steps each = ~15-day wall-clock budget allocated post-Step-4.
- **Phase B (Step 5) compute**: deferred to that plan; only assume here
  that ≥ 1 week of A100 time will be available.

## 8. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| BC anchor mis-calibrated → piKL pulls policy away from useful states | Medium | High | The empirical value-drift safeguard in §2 turns BC value training off if KL doesn't shrink. piKL `λ_initial=0.2` decay over 2M is short enough that PPO can recover. |
| Heuristic-mix curriculum mis-tuned → plateau at the heuristic's WR | Medium | High | Tier-1 ablation `heur_at_0` directly tests this. The 25% steady-state was the 5-0 panel vote. |
| Value normalisation init wrong for 15-VP terminal-only rewards | Low | Medium | `init mean=250 std=150` matches v1 Phase 0 numbers; tweak if `eval/value_explained_variance` < 0 for >5M steps. |
| KL early-stop fires too aggressively at target_kl=0.03 → undertraining | Medium | Medium | Raised from 0.01 specifically to address this. If still firing within 1 epoch consistently, raise further to 0.05. |
| Duo exploiter cycle corrupts main policy state | Medium | High | Snapshot main before each cycle; restore after. Test in `test_exploiter.py` with a 2-cycle integration. |
| Subproc env workers crash on engine edge cases | Medium | Medium | The Step-1 random-vs-random smoke (1000 games, 0 exceptions) suggests the engine is robust; per-env exception handling in `game_manager.py` should isolate. |
| Compute overrun (>10 days/seed) | Medium | Medium | Hard cap per §6; if exceeded, plan revised before continuing. Most likely cause is PPO-BR routine overhead — drop to every-10M cadence. |
| KL between BC anchor and PPO policy underflows after 2M steps when λ→0 | Low | Low | After 2M, anchor is no longer in the loss; underflow is irrelevant. |
| TrueSkill σ-decay 1.001 too slow / too fast for 1v1 self-play | Low | Low | Tune empirically once Step 4 is running. Not a blocker for Gate 1. |
| Champion regression Gate 3 fails (we lose to v1) | Low | High | Indicates BC anchor + PPO produced a policy strictly worse than v1's peak. Hard-stop; either BC failed silently or PPO is collapsing. |

## 9. STOP/RESUME points

| Where | What to verify | Human decision |
|---|---|---|
| **Pre-Step-4** (after Step 3 + this plan) | BC Gates 1+2+3 green, BC checkpoint saved, E0.1 result interpreted, ablation Tier-1 configs reviewed | Approve PPO kickoff; commit budget |
| **First 1M PPO steps** | `v_drift_l1` probe trending down; entropy floor not engaged; KL early-stop not firing every update | Approve continuation OR adjust BC-value weight + piKL λ for next seed |
| **First eval at 100k steps** | Heuristic-WR ≥ BC-anchor WR within ±0.05 (PPO shouldn't immediately regress) | OK if within band; STOP if WR collapses |
| **Soft-fail at Gate 1** (WR 0.55–0.69 at 10M) | Per §8 risk register | Pick a mitigation, run another seed |
| **Hard-fail at Gate 1 or 3** | Stop. Audit. | Open follow-up investigation; do not proceed to Step 5 |
| **Tier-1 ablation results** | Compare to full Step 4 result; any feature that doesn't lose ≥ 0.05 WR is a candidate for removal in v2.1 | Document; decide whether to drop |

## 10. Expert-panel + faculty-review carry-forward

Decisions inherited from the round-3 design (commit `9d34138`):

- **D1**: Phase B (MCTS) kept by 3-2 panel split, budgeted heavily. Step 4
  must produce a policy strong enough that MCTS sees a real lift; if Step 4
  hits Gate 1 ≥ 0.85, consider whether MCTS is even needed.
- **D3**: piKL anchor kept (4-1 panel + decay schedule revised to
  `λ=0.2 → 0 over 2M steps`).
- **D7**: opp-id embedding cut from 16-d to 8-d (3-2 compromise).
- **D8**: curriculum heuristic mix (unanimous panel: 60% → 25%).
- **D9**: terminal-only reward (4-1 panel + faculty).
- **D11**: target_kl raised from 0.01 to 0.03 (3-2 panel + faculty).
- **D14**: AlphaBeta-d4 added as the milestone bench (faculty-corrected;
  d=2 alone is too weak a baseline).
- **D15**: PPO-BR-gap renamed from "exploitability"; sensitivity probe
  (`Δ_BR = b_5M − b_1M`) added.

---

## Provenance

- Base design: `docs/plans/v2_design.md` §3.2, §3.3, §3.4, §5 Step 4.
- BC plan that this hands off from: `docs/plans/v2_step3_bc.md`.
- v1 ablation experiment (may inform reward shaping): `scripts/run_e01_ablation.sh`
  (in the v1 worktree at `/Users/benjaminli/my_projects/catan_rl/scripts/`).
- Preserved v1 champion for the Gate-3 regression check:
  `/Users/benjaminli/my_projects/catan_rl/checkpoints/train/checkpoint_16162816.pt`
  (0.56 symmetrised WR vs heuristic at v1 step 16,109,568).
