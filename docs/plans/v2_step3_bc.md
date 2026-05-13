# v2 Step 3 — Behavior Clone (Phase A.1)

**Status**: design locked (5-expert panel, 2026-05-13); implementation next.

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
| Total games | **30,000** | Panel 3-2 split (3 want 10-15k, 2 want 200k); 30k is the ~10× tokens-per-param middle ground (~1.5M (s,a) pairs at ~50 actions/game). Cheap to regenerate (~30 min compute) if scaling laws bite. |
| Source mix | **70% canonical heuristic vs heuristic, 30% perturbed vs heuristic** | Unanimous panel vote: pure heur-vs-heur is mode-collapsed on a single trajectory distribution. Variant mix broadens state coverage without injecting random-policy noise. |
| Perturbation recipe (variant side) | ε-greedy with ε=0.10 over the heuristic's top-K candidate actions, OR ±15% noise on the heuristic's scoring weights | Cheap implementation, preserves "heuristic-like" plays but reaches different states. |
| Player side filter | **Both players** | 3-2 vote: winner-only halves the data and is survivorship bias; the heuristic is symmetric so both sides are valid behavioral targets. Catanatron's "training on losing trajectories trains blind spots" concern is addressed by the variant mix above, not by filtering. |
| Storage format | Sharded NPZ (~256 MB / shard), one shard per 5k games | Easy to mmap and to parallelize generation. |
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

**D6 dihedral symmetry, prob = 1.0** (3-2 vote KEEP, with 2 of 3 KEEP votes
specifying 1.0; the Catanatron DROP vote raised an important correctness flag):

- Catanatron's argument: the heuristic has a deterministic tiebreak on corner
  index, so D6-rotating the state but not the action would teach inconsistent
  labels.
- Resolution: augmentation transforms **both state and action** through the
  precomputed D6 tables in `catan_rl/augmentation/symmetry_tables.py` (matching
  v1 Phase 1.5's implementation). Under this transform every (T(s), T(a)) pair
  is a *correct* sample of the same underlying policy — no inconsistency.
- At prob=1.0 each minibatch is a uniform sample over the 12-element D6 group.
  Catan boards are randomized per game (no canonical orientation at eval time),
  so there's no "lose the canonical view" risk.

Augmentation is applied in `__getitem__` of the dataset loader so the on-disk
shards stay canonical (one copy of the data, 12× effective at train time).

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

The AZ vet's KEY FLIP carried this: V(s) bootstraps slowest under PPO and
we have free `z` labels. Better to start PPO with V ≈ V_heuristic than
V ≈ 0 (random init); PPO's value loss + GAE will redirect V toward V_π as
the policy diverges from the heuristic anchor.

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

5-0 panel rejected the original `WR ≥ 0.45 vs heuristic in 100-game eval`. The
dispersion of dissent (2 said "raise the WR floor", 3 said "use a different
metric") meets in the middle as a **compound gate** — all three sub-gates must
pass:

1. **Val NLL has plateaued** for 3 consecutive evals (early-stop trigger).
2. **Held-out top-1 type-head accuracy ≥ 0.60**. This is the "we actually
   cloned the policy" signal — Cicero & Catanatron's KEY FLIPs both argued
   that this is the BC-as-prior metric that matters, not raw WR.
3. **WR ≥ 0.40 vs heuristic in a 200-game eval** (binomial 95% CI ~ ±0.068
   at p=0.5, so 0.40 implies the lower bound clears 1/3). Generous floor —
   piKL will pull us up from here; we are not gating on standalone strength.

If gate 2 fails but gate 1 passes, the loss weights are likely mis-balanced
(probably value/belief swamping policy). If gate 3 fails but gate 2 passes,
the policy is matching the heuristic's argmax but losing on tie-break or low-
mask-entropy states — investigate per-head accuracy by head.

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
