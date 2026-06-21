# Feature Specification: Value-Head Sharpening

**Feature Branch**: `007-value-head-sharpening`

**Created**: 2026-06-20

**Status**: Draft

**Input**: Break the v8 strength plateau by sharpening the net's value evaluation — the evidence-confirmed root cause. The value head only *ranks* states at Spearman 0.69 vs outcome and is trained on a point-margin-inflated target (`±1 + vp_diff/15`, `catan_env.py:1024`), not win-probability — so raw V spans [−1.6,+1.8] (27% outside [−1,1]) and inference search must bolt on a fitted `sigmoid(3.22V−1.14)` squash. Both raw self-play and search hit this same ceiling. Make the value head a calibrated win-probability and then let the bot's own search teach it (value-distillation expert iteration), reusing the shipped value loss and the spec-004 distillation apparatus.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Win-probability value target (Priority: P1)

The bot learns to predict its **probability of winning** directly, instead of a point-margin-inflated outcome. This fixes the mis-specified target the inference-search squash is patching and raises the value head's state-ranking quality — which lifts *both* raw-policy and search strength, because every decision (PPO advantage, MCTS backup) reads this evaluation.

**Why this priority**: This is the cheapest, most direct attack on the confirmed root cause (a 0.69-rank, mis-targeted value head). It needs no self-play campaign and is independently shippable: if it raises rank *and* doesn't regress strength, it is a win on its own and the foundation US2 builds on.

**Independent Test**: On a held-out set of peer-strength self-play states, measure the new value output's rank (Spearman vs eventual win/loss), calibration (Brier, fraction of raw output outside [0,1]), and then play strength (raw and search) vs v8 at n≥500. Delivers a sharper, squash-free evaluator if rank > 0.69 and strength does not regress.

**Acceptance Scenarios**:

1. **Given** the v8 base, **When** the net is trained/fine-tuned with a win/loss (probability) value target, **Then** the resulting value output ranks held-out states at **Spearman > 0.69** and is **calibrated without the squash** (Brier < the current 0.149-after-squash; raw output within [0,1]).
2. **Given** that checkpoint, **When** it plays vs v8 (raw and with search) at n≥500 seat-symmetric, **Then** playing strength does **not regress** (Wilson LB ≥ v8's), and ideally shows an uplift.
3. **Given** the win-prob value flag is **off**, **When** training/search/eval run, **Then** behavior is byte-identical to the pre-feature baseline (additive, default-off).

---

### User Story 2 - Value-distillation expert iteration (Priority: P2)

The bot improves its own evaluator by **learning from its deep thinking**: run determinized search to get a better (bootstrapped best-child) value at many states, distill those targets back into the value head, re-evaluate, and iterate — a compounding train-time loop (sharper value → sharper search → re-label).

**Why this priority**: This is the compounding lever, but it builds on US1's calibrated target and is heavier (a label→distill→gate loop). Spec 004 closed *policy*-distillation correctly (action overrides were win-neutral), but its own Probe B found the search-backed **value** carries live, undistilled signal (mean |best_q − root_value| 0.056, a systematic +0.030 optimism correction, 33% of states corrected >0.05).

**Independent Test**: Reuse the spec-004 labeler/distill/gate apparatus to label states with search-backed value, distill into the value head, and gate the result on the hardened ladder (n≥500, ≥2 seeds) vs the prior champion. One iteration that beats the champion (or a recorded "exhausted" verdict) proves or closes the lever.

**Acceptance Scenarios**:

1. **Given** a champion checkpoint, **When** states are labeled with search-backed value targets and distilled into the value head, **Then** the distilled checkpoint's value ranks/calibrates better on held-out states than the input checkpoint.
2. **Given** the distilled checkpoint, **When** it is gated on the hardened ladder (un-gameable min-anchor delta, n≥500, ≥2 seeds) vs the prior champion, **Then** it is banked as the new champion only if it wins, else reverted.
3. **Given** repeated iterations, **When** each is gated, **Then** the loop continues while iterations keep winning and stops (recorded as exhausted) when an iteration fails to beat the prior champion.

---

### Edge Cases

- **Rank up but strength down** (the win-prob value ranks better yet loses WR vs v8): do NOT bank — investigate whether the margin signal carried useful shaping gradient; treat as a real result, not a tuning failure.
- **v2-lineage break**: adding a value signal changes the policy state-dict — if the one-shot migration (load v8, init the new head) cannot keep existing v2 checkpoints loadable, block until it can.
- **Distillation amplifies search's optimism bias** (the +0.030 bias becomes a systematic over-estimate): the hardened-ladder strength gate must catch it; a calibration-only win that loses strength is rejected.
- **Squash interaction**: with the win-prob head the squash is unnecessary; the legacy squashed margin-value path must remain unchanged for old checkpoints (no silent behavior change for v6/v7/v8 search).
- **Held-out leakage**: the calibration set must be disjoint from training states, else Spearman/Brier are optimistic.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a value signal trained on the **terminal win/loss outcome as a probability** (binary cross-entropy, win=1/loss=0), additive and **default-off**, reusing the shipped value-loss path (`compute_value_loss`, `trainer.py:292`).
- **FR-002**: The system MUST measure, on a **held-out** peer-strength state set, the value output's **rank (Spearman vs outcome)**, **calibration (Brier, ECE)**, and **fraction of raw output outside [0,1]**; the US1 gate is **Spearman > 0.69** and calibrated **without** the squash.
- **FR-003**: The system MUST verify the win-prob checkpoint does **not regress** raw-policy or search playing strength vs v8 (Wilson LB ≥ v8's, n≥500, seat-symmetric) before banking.
- **FR-004**: The system MUST label states with **search-backed value targets** (the determinized-MCTS bootstrapped best-child Q) and **distill** them into the value head, reusing the spec-004 apparatus (`src/catan_rl/expert_iteration/`: `labeler.py`, `distill.py`, `gate.py`).
- **FR-005**: The system MUST **iterate** value-distillation, gating each iteration on a **hardened-ladder strength win** (un-gameable min-anchor delta, n≥500, ≥2 seeds) vs the prior champion, reverting iterations that do not win, and recording when the lever is exhausted.
- **FR-006**: The system MUST preserve **v2-lineage** via a documented **one-shot migration** (load v8, initialize the new head) that keeps existing v2 checkpoints loadable; produced checkpoints MUST themselves be v2-lineage-loadable (`strict=True` for their own architecture).
- **FR-007**: With the new value flags unused, training/search/eval MUST be **byte-identical** to the pre-feature baseline; the feature MUST add only **new** TensorBoard scalars (e.g. `value/win_brier`, `value/rank_spearman`), renaming none.
- **FR-008**: The inference-search squash MUST become **optional** when the win-prob head is used (raw output already in [0,1]); the **legacy squashed margin-value path MUST remain unchanged** for existing checkpoints.
- **FR-009**: The feature MUST NOT change the engine ruleset, the 6-head action space, or the observation schema.
- **FR-010**: All strength evals MUST be **CPU-pinned**, seat-symmetric, fixed-seed; training MUST use MPS per the device policy; no GUI import on any path.
- **FR-011**: **Gate-first** — US1's calibration + no-regression gate MUST precede US2; each US2 distillation iteration MUST be gated before banking.

### Key Entities *(include if feature involves data)*

- **Win-probability value target**: the terminal outcome (win=1/loss=0) as a BCE target for the value output — a calibrated P(win) in [0,1].
- **Value-calibration measurement**: `(Spearman rank, Brier, ECE, fraction-outside-[0,1])` of the value output vs eventual outcome on a held-out peer-game set.
- **Search-backed value target**: the determinized-MCTS bootstrapped best-child Q at a labeled state — the sharper, distillable evaluation.
- **Distillation iteration**: one `label → distill → hardened-ladder gate` cycle producing a candidate checkpoint (banked iff it beats the prior champion).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The win-probability value output ranks held-out states at **Spearman > 0.69** (vs v8's 0.69) and is calibrated (**Brier < 0.149**) **without** the squash.
- **SC-002**: The win-prob checkpoint shows **no strength regression** vs v8 (Wilson LB ≥ v8's, n≥500) for both raw and search play; an uplift is the target.
- **SC-003**: At least one value-distillation iteration **beats the prior champion** on the hardened ladder (min-anchor gate, n≥500, ≥2 seeds) and is banked as the new champion — **or** the lever is shown exhausted and that verdict is recorded.
- **SC-004**: With the feature's flags unused, a no-op smoke reproduces the pre-feature training/search/eval outputs **byte-identically**.
- **SC-005**: Existing v2 checkpoints (v8) remain loadable through the one-shot migration; produced checkpoints are v2-lineage-loadable.
- **SC-006**: Zero engine-rule / action-space / observation-schema changes; all banked numbers from CPU-pinned, seat-symmetric, fixed-seed evals.

## Assumptions

- The value head's 0.69 rank has real headroom — a win-probability target plus search-backed distillation can plausibly reach ~0.8+, the lever with the most slack for a pure-strength mandate.
- The spec-004 ExIt apparatus (labeler / distill / gate), built for the (closed) policy-distillation, generalizes to **value** targets with modest changes.
- The search-backed value carries distillable signal (spec-004 Probe B: systematic +0.030 optimism correction, 33% of states corrected >0.05) — never previously distilled.
- A held-out peer-strength state set for calibration can be generated from self-play games disjoint from any distillation training set.
- "Superhuman" is **unverifiable** here (no human baseline); the operational target is *strongest by in-engine proxies* — ladder-Elo (n≥500, ≥2 seeds), exploiter-resistance, search-parity.
- The cheap search-config fix (`fpu_mode=parent` / `n_determinizations`) and raw-sims/batched-leaf scaling belong to spec 006, not here; capacity/width changes and piKL are deferred to later specs.
- v8 (`runs/anchors/v8_promobar_u243.pt`) is the starting champion/base.
