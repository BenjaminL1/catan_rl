# v2 Step 3 — Behavior Clone (Phase A.1)

**Status**: design locked (5-expert panel 2026-05-13, faculty review
2026-05-13); implementation gated on the §0 preflight experiments.

**Preflight gate** (faculty review): Step 3 implementation does **not**
start until `v2_design.md` §0 experiments E0.2 (heuristic action
distribution audit) and E0.3 (heuristic determinism audit) have run.
Their measured numbers feed the §6 acceptance gates and the §2
augmentation prob choice. Without those measurements the gates are
guessed thresholds and the augmentation choice is theoretical
guesswork — exactly the failure mode the review flagged.

**Goal**: pretrain the v2 policy network on heuristic gameplay so PPO + piKL has
a useful prior. The BC anchor's job is *not* to be a strong standalone player —
it's to be a non-uniform prior that piKL can leash and PPO can build on.

## Inputs

- Network: `CatanPolicy` from Step 2 (1.38M params), random init.
- Engine + env: `catanGame`, `CatanEnv` from Step 1.
- Heuristic agent: `heuristicAIPlayer` in `src/catan_rl/agents/heuristic.py`.
- Obs schema: per `src/catan_rl/policy/obs_schema.py`.

## Outputs

- `checkpoints/bc/best.pt` — policy weights + optimizer state.
- `runs/bc/<run_id>/` — TensorBoard logs (NLL, head accuracies, WR-vs-heuristic).
- `data/bc/v1.tar` — generated dataset (sharded NPZ).
- Acceptance criterion (see §6) gates handoff to Step 4 (PPO + piKL).

---

## 1. Data generation

| Parameter | Value | Rationale |
|---|---|---|
| Total games | **30,000 as starting point** (Tier-3 sweep if gate fails) | Panel 3-2 split (3 want 10-15k, 2 want 200k); 30k is the middle. Faculty review note: "10× tokens-per-param ≈ Chinchilla" is a category error — that scaling law is for LM tokens, not structured (s,a) pairs. **No theoretical claim is made on 30k.** If Gate 2 of §6 fails with `Δ_NLL` CI not clearing zero, run the **Tier-3 sweep {10k, 30k, 100k}** per the §3.8 budget — this is the only ablation that touches data-scale. If Gate 1 plateaus before 2 epochs, the dataset is too small *for the gates set*, not necessarily too small overall — read the per-head NLL contributions before declaring failure. |
| Source mix | **70% canonical heuristic vs heuristic, 30% perturbed vs heuristic** | Unanimous panel vote: pure heur-vs-heur is mode-collapsed on a single trajectory distribution. Variant mix broadens state coverage without injecting random-policy noise. |
| Perturbation recipe (variant side) | ε-greedy with ε=0.10 over the heuristic's top-K candidate actions, OR ±15% noise on the heuristic's scoring weights | Cheap implementation, preserves "heuristic-like" plays but reaches different states. |
| Player side filter | **Both players** | 3-2 vote: winner-only halves the data and is survivorship bias; the heuristic is symmetric so both sides are valid behavioral targets. Catanatron's "training on losing trajectories trains blind spots" concern is addressed by the variant mix above, not by filtering. |
| Storage format | Sharded NPZ (~320 MB / shard), one shard per 2.5k games | Parallelizes generation. NB: `np.savez_compressed` archives are **NOT** mmap-able (`np.load(mmap_mode=...)` is silently ignored for `.npz`, and reading any member decompresses the whole ~5.5 GB array). The loader therefore chunk-streams bounded row-ranges into a small LRU rather than mmapping — see `src/catan_rl/bc/loader.py`. |
| Compute target | **≤ 1 hour on M1 Pro**, 8-way concurrent rollouts | 17k env-steps/sec measured in Step 1 random-vs-random; heuristic adds ~2× per-step cost. 30k games × ~50 actions = 1.5M env-steps. Target wall-clock ~30 min single-process, less with parallelism. |

**Filter at write time**: drop (s, a) pairs where `mask_sum(type) == 1` — i.e., the
state had only one legal action type. Forced moves contribute zero gradient and
inflate accuracy metrics (unanimous panel vote on D4).

Record per-pair:
- The full obs dict (per `obs_schema.py`).
- The 6-head action `[type, corner, edge, tile, res1, res2]`.
- The 9-key mask dict.
- The opponent's true dev-card type counts at this state (for the belief target).
- The terminal outcome `z ∈ {−1, +1}` and discounted return `γ^(T−t) · z` (for
  the BC value loss — see §3.2).
- Game id + step index (for debugging / per-game stratification).

---

## 2. Augmentation

**D6 dihedral symmetry, prob = TBD pending preflight E0.3 + E0.4 results.**

Default before preflight: **prob = 0.5** (the conservative position, held by
the OAI and M1 panelists). Re-set to **prob = 1.0** only if both preflight
checks pass:

  - **E0.3 (heuristic determinism audit)**: deterministic-tiebreaker fraction
    < 5%. Catanatron's panel argument was that if the heuristic's choice is
    not equivariant under D6 (e.g. always-pick-lex-first-corner-in-tie),
    rotating the state but copying the action creates inconsistent labels.
  - **E0.4 (network equivariance probe)**: the random-init network is
    approximately D6-equivariant. **Faculty review correction**: the axial
    positional embedding in the TileEncoder is by construction *non*-
    equivariant. If the probe shows low equivariance, prob=0.5 is the
    correct hedge — it gives the network *some* augmentation without forcing
    it to learn a property the architecture cannot represent natively.

Implementation invariant (independent of prob): augmentation transforms
**both state and action** through the precomputed D6 tables in
`catan_rl/augmentation/symmetry_tables.py` (matching v1 Phase 1.5). Under
this transform every (T(s), T(a)) pair is a correct sample of the policy
under that group element. State-only augmentation is a correctness bug.

**Correction to a prior claim**: the original draft said "Catan boards are
randomized per game (no canonical orientation), so there's no canonical-view
risk." Faculty review: this conflates the *resource/number-token shuffle*
(random per game, orthogonal to D6) with the *board geometry* (fixed 19-tile
hex lattice, the actual domain of the D6 action). The D6 augmentation
covers the geometry symmetry only; it does not multiply effective data
over the resource shuffle (which is already sampled by the dataset
generation).

Augmentation is applied in `__getitem__` of the dataset loader so the
on-disk shards stay canonical (one copy of the data, 6× or 12× effective
at train time depending on prob).

---

## 3. Loss

```
L_total = L_policy + 0.1 · L_value + 0.05 · L_belief
```

### 3.1 Policy CE — per-head relevance-weighted

For each head h ∈ {type, corner, edge, tile, res1, res2}:
```
L_h = sum_{i: relevance_h(action_type_i) = 1} CE(logits_h_i, action_h_i)
     / max(1, batch_relevance_count_h)
L_policy = sum_h L_h
```
Each head's loss is averaged over the batch rows where it is relevant
(unanimous panel vote on D5). The relevance buffer is already shipped on
`CatanActionHeads.head_relevance` from Step 2; reuse it here.

**Hard labels** (unanimous on D6) — no label smoothing, no soft distributions.
The heuristic is near-deterministic; soft labels invent supervision we don't
have.

### 3.2 Value MSE on discounted terminal outcomes — weight 0.1

3-2 panel vote KEEP (AZ vet + Cicero + OAI Five vs Catanatron + M1 Pro).
Train V(s) to predict the discounted terminal outcome `γ^(T−t) · z` (with
γ = 0.998 to match the PPO config). Weight 0.1 — the M1/Catanatron concern
("V_heuristic ≠ V_PPO eventually") is real but addressed by keeping the
weight small enough that PPO can correct it within the first ~500k steps.

**Faculty-review correction**: the "PPO will redirect V within 500k steps"
argument is **hand-waved**. Mathematically, $V_\text{BC}(s) \approx
\mathbb{E}_\text{heuristic}[z|s]$, while PPO needs $V_\pi(s) =
\mathbb{E}_\pi[z|s]$; the two are equal only at $\pi \approx \text{heur}$.
As PPO improves, $V_\text{BC}$ becomes a stale baseline. Whether the value
loss can catch up depends on the rate of policy improvement.

**Empirical safeguard** (faculty review + re-review): during Step 4 PPO
training, log `bc/v_drift_l1_per_1m_steps` = the L1 difference between
the BC value head's predictions at step 0 vs at step N, on a **rolling
eval-state batch sampled from the *current* PPO policy's rollouts**
(not from heuristic-vs-heuristic — that would measure drift on the
*training* distribution, which is uninformative since V_BC was fit on
it).

Implementation: every 100k PPO steps, snapshot 1024 (s, a, z) tuples
from the most recent rollout window. Compute `V_BC(s)` (using the
frozen post-BC value-head weights) and `V_now(s)` (using the current
PPO value-head). Report `mean |V_BC(s) − V_now(s)|` and the trend
across the first 1M PPO steps.

**Decision rule**: if `v_drift_l1` does *not* decrease over the first
1M steps — i.e. the value head fails to redirect under PPO's value
loss — **disable BC value training for the next seed** (weight 0). If
it decreases but remains > 0.3 (i.e. predictions differ by ~30% of
the value range), enable the **detached-trunk variant** for the next
seed: the value head trains during BC, but its gradients do not flow
back into the obs encoder. This protects the encoder from contamination
while preserving the head warm-start.

The probe runs without slowing PPO (1024 forward passes every 100k
steps is ~0.1% overhead). It produces a single scalar per measurement
window, easy to read on TensorBoard.

The AZ vet's KEY FLIP carried this: V(s) bootstraps slowest under PPO and
we have free `z` labels. Better to start PPO with V ≈ V_heuristic than
V ≈ 0 (random init); PPO's value loss + GAE will redirect V toward V_π as
the policy diverges from the heuristic anchor — **subject to the empirical
probe above**.

### 3.3 Belief soft-CE on env GT — weight 0.05

Unanimous panel vote KEEP. Same weight as in the main PPO config so the
loss landscape is consistent across BC and PPO. Target is the normalized
opponent dev-card type count vector exposed by the env (`obs['belief_target']`).

---

## 4. Optimizer / schedule

| Parameter | Value | Source |
|---|---|---|
| Optimizer | **AdamW**, β=(0.9, 0.999), eps=1e-5, weight_decay=1e-4 | Unanimous panel vote (D10); matches PPO so optimizer-state transfer is meaningful. |
| Peak LR | **3e-4** | Matches PPO peak. |
| LR schedule | **Constant after a 500-step linear warmup** | 3-2 panel vote (D11): cosine decay over a 1-hour BC run is theater; constant LR lets early-stop pick the right termination point. |
| Batch size | **1024** | 3-1-1 panel vote (D12): M1 Pro is throughput-bound at BC time (not latency-bound like at PPO rollout time); larger batches = fewer optimizer steps = faster wall-clock with stabler gradients. |
| Max epochs | 10 (cap) | Bound on the to-convergence loop. |
| Termination | **Held-out NLL early-stop, patience = 3 evals** | Unanimous panel vote (D13); evals every 500 steps. |
| Gradient clipping | max_norm=1.0 | Standard. |
| AMP | Off (CPU only) | M1 Pro CPU; bf16 not worth the kernel-cost. |

---

## 5. Validation

Hold out **10% of games** (stratified by game-id, not by (s, a) pair — to avoid
leakage of within-game states between train and val).

Track, every 500 training steps:
- **Val NLL** (sum of relevance-weighted per-head CE).
- **Val top-1 type-head accuracy** (the heuristic's action-type matches the
  policy's argmax).
- **Val top-1 corner accuracy** (conditional on type ∈ {settle, city}).
- **Val belief soft-CE**.
- **Val value MSE**.

Track every epoch:
- **WR vs heuristic** in a 200-game eval (cheap at ~5s/100 games measured in
  Step 1, so 200 games ≈ 10s).

This satisfies the unanimous D14 vote (both NLL and WR).

---

## 6. Acceptance gate

5-0 panel rejected the original `WR ≥ 0.45 vs heuristic in 100-game eval`.
Faculty review additionally flagged that the panel's *replacement* thresholds
(top-1 ≥ 0.60, WR ≥ 0.40) are **also unprincipled guesses**. Both rounds
of feedback agreed on the shape (compound gate) but disagreed on the
calibration. The corrected gate uses **measured baselines from preflight
E0.2**, not numbers picked from intuition.

**Required preflight measurements** (E0.2, before this gate is set):
  - `BASE_NLL_FREQ` — per-head NLL of an always-predict-the-marginal-mode
    policy on 1000 heuristic-vs-heuristic games.
  - `BASE_TOP1_FREQ` — top-1 accuracy of the same trivial baseline.
  - `BASE_WR_HEUR_SELF` — heuristic vs heuristic self-play WR, P1-seat and
    P2-seat, both seats over 1000 games. Should be near 0.50 + a small
    seat advantage.

**Compound gate** (all three pass).

**Faculty re-review correction**: the prior formulation used fixed-margin
thresholds (NLL gap ≥ 0.30 nats; WR ≥ measured-self − 0.10) that were
still guesses dressed up with rhetoric. The corrected formulation
replaces both with **proper statistical tests** that produce p-values,
not verdicts.

#### Gate 1 — convergence

Val NLL has plateaued for 3 consecutive eval ticks (early-stop trigger).
Mechanical, not a hyperparameter.

#### Gate 2 — non-triviality (paired-bootstrap NLL test)

For each head h ∈ {type, corner, edge}, the BC policy must be a
statistically significant improvement over the trivial frequency-
baseline policy `freq` (predicts the marginal mode within the mask).
Paired-bootstrap on the held-out 10%-of-games val split:

```
For each bootstrap sample b ∈ 1..10⁴:
    resample (s, a) val pairs with replacement
    compute Δ_h^b = NLL_freq(b) - NLL_BC(b)   # positive = BC better
Report mean(Δ_h) and the 99% bootstrap CI.

Gate passes for head h iff CI_lower(Δ_h) > 0.
Compound gate 2 passes iff this holds for type, corner, AND edge heads.
```

Concrete implementation: `tests/integration/test_bc_gate2.py` loads the
val split + the BC checkpoint, computes per-pair NLL for both
policies, runs the paired bootstrap, and prints pass/fail per head
with the CI bounds.

The reason this is more honest than "NLL gap ≥ 0.30 nats": the gap
needed to declare significance depends on the variance of the val
distribution, not on a chosen number. If the heuristic is very
deterministic, even a 0.05-nat gap may clear significance; if noisy,
0.5 nats may not. The test reads the data, not the rhetoric.

#### Gate 3 — standalone sanity (power-calibrated WR test)

The natural null is "BC policy ≡ heuristic teacher." Run 600 games
per seat (200 × N=3 seeds), pool. Test:

```
H₀: WR_BC = WR_heur_self          (BC indistinguishable from teacher)
H₁: |WR_BC − WR_heur_self| > 0    (two-sided)

At α = 0.05 and n = 600 games per seat, the Wald-CI half-width is
~ 1.96 · √(0.25/600) ≈ 0.040.

Gate 3 passes iff symmetrized |WR_BC − WR_heur_self| ≤ 0.04
i.e. we can't statistically reject equivalence with the teacher.
```

This is a **TOST-style** (two one-sided tests) equivalence test, not a
superiority test. The BC clone's job is to *match* the teacher, not
beat it; a clone that beats the teacher by 0.10 WR is also a failure
(usually a sign of policy collapse onto a heuristic-exploiting mode
that won't survive PPO + piKL).

If the symmetrized BC-vs-heur WR distribution is centered at the
teacher self-WR with CI ≤ 0.04, the BC anchor is doing exactly what
piKL needs: faithfully representing the teacher policy with small
calibration error.

#### Diagnosis ladder when a gate fails

If gate 2 fails but gate 1 passes → loss weights mis-balanced
  (probably value/belief swamping policy). Audit per-head NLL
  contributions individually.
If gate 3 fails *because BC WR > teacher self-WR + 0.04* → BC has
  collapsed onto a heuristic-exploiting mode. Increase data,
  decrease epochs.
If gate 3 fails *because BC WR < teacher self-WR − 0.04* → BC didn't
  finish learning the teacher. Increase data or epochs.
If gate 1 plateaus very early (< 2 epochs at the 30k-game scale) →
  dataset too small. Sweep upward (Tier-3 ablation: 10k → 30k → 100k
  game-count sweep, per §3.8).

---

## 7. File layout (new code)

```
src/catan_rl/bc/
├── __init__.py
├── dataset.py        # generate_dataset(...) — parallel heuristic rollouts → NPZ shards
├── perturbed_heuristic.py  # ε-greedy + weight-noise wrappers around heuristicAIPlayer
├── loader.py         # BcDataset (torch.utils.data.Dataset) + D6 aug in __getitem__
├── loss.py           # bc_loss(policy_out, batch) → dict[str, Tensor]
└── train.py          # train(...) — outer loop, val callbacks, early stop

scripts/
├── generate_bc_dataset.py  # CLI wrapper over bc.dataset
└── train_bc.py             # CLI wrapper over bc.train

configs/
└── bc.yaml                 # all hyperparameters from §1, §4

tests/
├── unit/bc/test_dataset.py
├── unit/bc/test_loader.py
├── unit/bc/test_loss.py
└── integration/test_bc_smoke.py  # 100-game tiny pretrain, asserts loss goes down
```

---

## 8. Risks

- **Data leakage via D6 aug** if the augmentation tables are wrong → forward/
  backward smoke test before any real training.
- **Heuristic is too narrow** → variant-mix is the mitigation; if still narrow,
  fall back to LARGER game count (D1's 200k option) — the 30k middle is
  reversible.
- **Value head contaminates encoder** (Catanatron's concern) → weight 0.1 is
  the throttle; if downstream PPO sees V-divergence in first 500k steps, drop
  weight to 0.05 or detach the value head from the trunk gradient.
- **Top-1 acc gate is unreachable** → the heuristic may have actions that are
  hard to predict from obs alone (e.g., near-ties broken by Python dict
  iteration order). Audit the data — if the heuristic's near-tie rate is >40%,
  switch the gate to "top-3 within mask ≥ 0.85".

---

## 9. Expert-panel consensus (2026-05-13)

| # | Decision | AZ vet | OAI Five | Cicero | Catanatron | M1 Pro | **Pick** |
|---|---|---|---|---|---|---|---|
| D1 | Game count | FEWER 10-15k | MORE 200k | MORE 200k | FEWER 10-15k | FEWER 15k | **30k** (compromise) |
| D2 | Source mix | VARIANTS | VARIANTS | VARIANTS | VARIANTS | VARIANTS | **70/30 canonical/variant** (5-0) |
| D3 | Winner filter | WINNER | BOTH | BOTH | WINNER | BOTH | **BOTH** (3-2) |
| D4 | Forced moves | SKIP | SKIP | SKIP | SKIP | SKIP | **SKIP** (5-0) |
| D5 | Head weights | RELEV | RELEV | RELEV | RELEV | RELEV | **Relevance-weighted** (5-0) |
| D6 | Soft labels | HARD | HARD | HARD | HARD | HARD | **Hard CE** (5-0) |
| D7 | Value head BC | YES | YES | YES | NO | NO | **YES @ weight 0.1** (3-2) |
| D8 | Belief head BC | YES | YES | YES | YES | YES | **YES @ weight 0.05** (5-0) |
| D9 | D6 aug prob | 1.0 | 0.5 | 1.0 | DROP | 0.5 | **1.0 with state+action transform** (3-1-1) |
| D10 | AdamW | YES | YES | YES | YES | YES | **AdamW** (5-0) |
| D11 | LR schedule | CONST | KEEP | CONST | CONST | OTHER | **Constant 3e-4 + 500-step warmup** (3-2) |
| D12 | Batch size | 1024-2048 | 1024-2048 | 512 | 256 | 1024 | **1024** (3-1-1) |
| D13 | Epochs | TO-CONV | TO-CONV | TO-CONV | TO-CONV | TO-CONV | **To-convergence, patience=3** (5-0) |
| D14 | Val metrics | BOTH | BOTH | BOTH | BOTH | BOTH | **Both NLL and WR** (5-0) |
| D15 | Gate | DIFFERENT | HIGHER 0.48 | DIFFERENT | DIFFERENT | HIGHER 0.50 | **Compound: NLL+top1≥0.60+WR≥0.40** (5-0 rejected 0.45) |

**Cross-panel KEY FLIPs:**
- D7 — split AZ vet vs M1 Pro. Carried for the value head, weight throttled to 0.1.
- D15 — both Cicero and OAI flipped it for opposite reasons; compound gate satisfies both.
- D3 — Catanatron's winner-only argument outvoted but motivates D2 variant mix.

**HIGH CONFIDENCE picks:**
- D6 Hard CE (AZ vet): "soft labels are cargo-cult AlphaZero applied to the wrong distribution."
- D10 AdamW (OAI): "no scenario where Adam beats AdamW in 2026."
- D7+D8 aux heads ON (Cicero): "free supervision is malpractice to leave on the floor."
- D9 drop symmetry aug (Catanatron): outvoted but the inconsistent-labels risk drove the "transform both state and action" implementation requirement.
- D4 skip forced moves (M1 Pro): "zero gradient signal, inflates accuracy by 10-20pts."
