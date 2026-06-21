# Feature Specification: Value-Head Sharpening

**Feature Branch**: `007-value-head-sharpening`

**Created**: 2026-06-20

**Status**: Draft (revised after senior-RL review — adds a de-risk probe gate, second-head design, rescoped distillation)

**Input**: Break the v8 strength plateau by sharpening the net's value evaluation — the confirmed root cause. The value head ranks states at only Spearman 0.69 vs outcome and is trained on a point-margin-inflated target (`±1 + vp_diff/15`, `catan_env.py:1024`), not win-probability, so raw V spans [−1.6,+1.8] (27% outside [−1,1]) and inference search bolts on a fitted `sigmoid(3.22V−1.14)` squash. **But the rank deficit may be a trunk/representation limit, not a value-target limit** (late-game Spearman is already ~0.86 — the drag is early/near-0.5 states), so this spec **de-risks that first** with a cheap probe before any training campaign.

## Prior Work *(context for the priority call)*

- A prior senior-RL council was **lukewarm** on value-sharpening ("the critic is healthy after the squash — this may fix a non-problem"). This spec is a **minority call** that must be justified by the **frozen-trunk probe (US0)** evidence, not assumed.
- Spec **004 closed *policy*-distillation** correctly: the search action-overrides were win-neutral. Its Probe B's value signal (+0.030 optimism, 33% corrected >0.05) was shown **near-tautological / win-neutral out-of-model** (+0.025, CI [−0.038,+0.088]). So US2 must **prove the value target beats the squashed leaf out-of-model before any distillation loop** — it is NOT assumed to carry signal.
- The shipped spec-004 apparatus (`expert_iteration/labeler.py`, `distill.py`, `gate.py`) records `action` + `z_disc` (game outcome) and gates a single-forward policy-WR vs v6 — it does **NOT** emit search best-child Q, has **no value-only distill path**, and is **not** the hardened ladder. US2 reuses only the **BC shard format + train loop**; the best_q labeler, value-only distill mode, and `ladder_gate.py` gating are **new**.
- Relationship to **spec 006** (scale inference search): **parallel, independent.** 006 rents inference-time strength (capped by this value head); 007 raises the value head itself (lifts raw + search). Neither blocks the other. The cheap search-config fix (`fpu_mode=parent`/`n_determinizations`) is 006's, not here.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - Frozen-trunk rank-probe: is the rank deficit even target-fixable? (Priority: P0, gates everything)

Before any training campaign, decide cheaply (<1h) whether the value head's 0.69 rank is fixable by changing the **target/last-layer** at all, or whether it is a **trunk/representation ceiling** that a retarget cannot move.

**Why this priority**: This retires the single load-bearing risk for the whole spec. If rank is trunk-limited, US1-as-written is the wrong fix and the path forward becomes a capacity/representation spec — knowing that costs <1h vs a multi-day dead end.

**Independent Test**: Freeze the v8 trunk; retrain **only** the value head on disjoint held-out v8 self-play data under two targets — (a) win/loss BCE, (b) the current per-VP margin (control). Report Spearman **overall and by game phase (early/mid/late)** and raw Brier for each.

**Acceptance Scenarios**:

1. **Given** a frozen v8 trunk, **When** the value head is retrained on win/loss vs the margin control on held-out data, **Then** both runs' Spearman (overall + per-phase) and raw Brier are reported.
2. **Given** those numbers, **When** win/loss rank **exceeds both the 0.69 baseline AND the frozen-trunk margin control by a bootstrap-CI-clean delta** → PROCEED to US1. **When** both controls land ≈0.69 → conclude **trunk-limited, STOP**, record the verdict, and do not build US1 (escalate to a capacity/representation spec instead).

---

### User Story 1 - Second (parallel) win-probability value head (Priority: P1, gated on US0)

Add a **second** value output trained on terminal win/loss (BCE) **alongside** the existing margin value head — *not* a replacement. The net keeps the dense per-game margin gradient (and the legacy squash path for old checkpoints) while gaining a directly-calibrated P(win) the search can read raw.

**Why this priority**: Keeping the margin head preserves the dense shaping signal the review warned sparse BCE would lose, makes the v2-lineage migration a clean additive load, and lets search opt into the calibrated head. Independently shippable if it raises rank *and* doesn't regress strength.

**Independent Test**: Train the dual-head net (margin + win-prob); on a precisely-defined disjoint held-out set, measure the win-prob output's rank (Spearman + bootstrap CI), raw Brier (vs the 0.149 post-squash baseline), and fraction outside [0,1]; then play strength (raw + search, search reading the win-prob head raw) vs v8 at n≥500 (n≥1500 confirm if the LB straddles 0.50).

**Acceptance Scenarios**:

1. **Given** the dual-head net, **When** the win-prob output is measured on held-out states, **Then** Spearman has a **bootstrap-CI lower bound > 0.69** (target point ≥ 0.73) and raw output lies in [0,1] needing **no squash**; raw Brier is reported alongside the 0.149 baseline.
2. **Given** that checkpoint, **When** it plays vs v8 (raw and search-reading-win-prob-head) at n≥500, **Then** strength does **not regress** (Wilson LB ≥ v8's), confirmed at n≥1500 before banking if n=500 straddles 0.50.
3. **Given** the win-prob flag **off**, **When** training/search/eval run, **Then** behavior is byte-identical to baseline (the margin head's params load untouched).

---

### User Story 2 - Value-distillation expert iteration (Priority: P2, gated on US1 + an out-of-model value check)

Sharpen the value head by distilling the **search-backed best-child Q** into it — but only after proving that target is genuinely better-calibrated than the current squashed leaf (the 004 win-neutral lesson), via a **new** value-distill path.

**Why this priority**: The compounding train-time lever, but heaviest and the most at-risk of re-imprinting search's optimism bias. It is gated twice: an out-of-model value-validation pre-check, then per-iteration hardened-ladder strength.

**Independent Test**: (pre-gate) on ~50–200 held-out states, compare corrected-Brier(best_q) vs Brier(squashed-leaf) against eventual outcomes; only if best_q is **genuinely better-calibrated** does the loop run. (loop) a new labeler emits best_q; a value-only distill mode (policy/belief heads frozen) trains the value head on **MSE(V, best_q)**; the result is gated on the hardened ladder.

**Acceptance Scenarios**:

1. **Given** ~50–200 held-out states, **When** corrected-Brier(best_q) is compared to Brier(squashed-leaf) vs eventual outcomes, **Then** the loop proceeds **only if** best_q is better-calibrated (not just re-imprinting the +0.030 optimism bias); else US2 stops, recorded.
2. **Given** the pre-gate passes, **When** best_q targets are distilled into the value head (value-only, MSE) and the result is gated on the hardened ladder (min-anchor delta, n≥500, ≥2 eval seeds) vs the prior champion, **Then** it is banked only if it wins, else reverted.
3. **Given** repeated iterations, **When** each is gated, **Then** the loop stops at **N_FAIL=2 consecutive non-winning gates OR N_ITER=5 total**, whichever first, recording the stop reason.

---

### Edge Cases

- **Trunk-limited (US0 fails):** both targets land ≈0.69 → STOP, escalate to a capacity spec; do not build US1.
- **Rank up but strength down (US1):** do not bank — record as a real result (the margin signal may carry useful shaping).
- **Double-squash (US1):** if the win-prob head outputs [0,1] and search still applies `sigmoid(3.22P−1.14)`, calibration is destroyed → a `use_value_squash` toggle + head selector must prevent it (tested).
- **best_q re-imprints optimism (US2):** the out-of-model pre-gate + the hardened-ladder strength gate must catch a calibration-only or win-neutral "gain."
- **Held-out leakage:** the calibration set must be disjoint from any train/distill set, else rank/Brier are optimistic.
- **Target-type mismatch:** US1 = BCE(sigmoid(v), 0/1 outcome); US2 = MSE(v, best_q soft target in [0,1]). Never BCE on soft best_q.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-000** (US0 gate): Provide a frozen-trunk value-head retrain probe (win/loss BCE vs margin control) on disjoint held-out v8 data, reporting Spearman overall + per-phase + raw Brier; the build proceeds to US1 only on a CI-clean rank win over both 0.69 and the margin control.
- **FR-001**: Add a **second, parallel** win-probability value output (BCE on terminal win/loss) to `ValueHead` **without removing** the margin output; additive and default-off behind `TrainConfig.use_value_win_prob` (default False).
- **FR-002**: Measure on a **precisely-defined disjoint held-out set** (source, ≥5–10k states, seat + phase balance, seed/step range, train-calib vs test split): rank (Spearman + bootstrap CI), raw Brier (reported vs the 0.149 post-squash baseline), and fraction outside [0,1]. Gate: **bootstrap-CI LB > 0.69**, raw output in [0,1] with **no squash**.
- **FR-003**: Verify no strength regression vs v8 (Wilson LB ≥ v8's) at **n≥500**, with an **n≥1500 confirmation before banking** when the n=500 LB straddles 0.50; raw and search (search reading the win-prob head).
- **FR-004**: Provide a **new** value-distill path: (a) a labeler that extracts and emits the search **best-child Q scalar** per state, (b) a **value-only distill mode** (policy + belief heads frozen) training the value head on **MSE(V, best_q)**, (c) the **hardened-ladder gate** (`scripts/ladder_gate.py`). Reuse only the BC shard format + train loop; do NOT claim the 004 labeler/distill/gate are reusable as-is.
- **FR-005**: Gate US2 first on an **out-of-model value-validation** (corrected-Brier(best_q) vs Brier(squashed-leaf) vs eventual outcomes on ~50–200 held-out states); then iterate, gating each iteration on the hardened ladder (n≥500, ≥2 eval seeds), reverting losers; stop at **N_FAIL=2 or N_ITER=5**, recording the reason.
- **FR-006**: Preserve **v2-lineage** via a one-shot migration: load v8 `strict=False`, init the new win-prob head (final-layer gain ~0.01, mirroring `BeliefHead` at `heads.py:452`), re-save as `strict=True` for the new architecture; v8's margin head loads untouched. Tests: (a) v8→new-arch migration, (b) flag-off forward matches baseline, (c) migrated ckpt round-trips `strict=True`.
- **FR-007**: With `use_value_win_prob`/`use_value_distillation` (both default False) unused, training/search/eval are **byte-identical** to baseline; add `test_value_sharpening_disabled_is_identical` (2 epochs flag-off, state-dict + optimizer match). Only **new** TB scalars (`value/win_brier`, `value/rank_spearman`); none renamed.
- **FR-008**: Add `SearchConfig.use_value_squash` (default True, backward-compat) + a head selector so search reads the win-prob output **raw** when active; test that search@50 values stay in [0,1] under both settings.
- **FR-009**: No change to the engine ruleset, 6-head action space, or observation schema.
- **FR-010**: Strength evals CPU-pinned, seat-symmetric, fixed-seed; training MPS; no GUI import on any path.
- **FR-011**: **Gate-first** — US0 gates US1; US1 gates US2; US2's out-of-model check gates its loop; each iteration gated before banking.

### Key Entities *(include if feature involves data)*

- **Frozen-trunk probe result**: per-target (win/loss, margin) `(Spearman overall + per-phase, raw Brier)` on held-out v8 data — the US0 go/no-go.
- **Win-probability value output**: a second `ValueHead` output, BCE on terminal win/loss, calibrated P(win) in [0,1].
- **Value-calibration measurement**: `(Spearman + bootstrap CI, raw Brier, ECE, fraction-outside-[0,1])` on a disjoint held-out set.
- **Search best-child Q target**: the determinized-MCTS bootstrapped best-child Q at a labeled state — the US2 distill target (soft, MSE).
- **Distillation iteration**: one `label(best_q) → value-only distill → hardened-ladder gate` cycle.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-000**: The frozen-trunk probe produces a recorded verdict (target-fixable vs trunk-limited) in <~1h before any training campaign.
- **SC-001**: IF US1 proceeds, the win-prob output ranks held-out states at **Spearman bootstrap-CI LB > 0.69** (point ≥ 0.73) and is calibrated (**raw Brier < 0.149**) **without** the squash.
- **SC-002**: The win-prob checkpoint shows **no strength regression** vs v8 (Wilson LB ≥ v8's, n≥500; n≥1500 confirm before banking) for raw and search; uplift is the target.
- **SC-003**: US2 runs only if best_q beats the squashed leaf out-of-model; then ≥1 iteration **beats the prior champion** on the hardened ladder (min-anchor delta, n≥500, ≥2 seeds) and is banked — **or** the lever is recorded exhausted (per the stop rule).
- **SC-004**: With the feature flags unused, a no-op smoke reproduces baseline training/search/eval **byte-identically** (named test).
- **SC-005**: Existing v2 checkpoints (v8) remain loadable via the one-shot migration; produced checkpoints are v2-lineage-loadable.
- **SC-006**: Zero engine-rule / action / obs changes; all banked numbers from CPU-pinned, seat-symmetric, fixed-seed evals.

## Design Hypotheses (tested by gates, not assumed)

- **H-rank-fixable**: the 0.69 rank has headroom reachable by a better target/head — **tested by US0**; if false (trunk-limited), the spec stops and escalates to capacity.
- **H-dense-signal**: keeping the margin head alongside the win-prob head preserves the dense gradient — realized by the second-head design (not a separate gate).
- **H-bestq-signal**: search best_q is better-calibrated than the squashed leaf out-of-model — **tested by FR-005's pre-gate**; if false, US2 stops (the 004 win-neutral lesson).
- **H-capacity-adequate**: 1.38M params suffice for a sharper value — held until US0/US2 evidence says otherwise; capacity is a deferred later spec.

## Assumptions

- A disjoint held-out peer-strength state set (with phase/seat balance) can be generated from v8 self-play games.
- "Superhuman" is unverifiable here (no human baseline); the operational target is *strongest by in-engine proxies* — ladder-Elo (n≥500, ≥2 seeds), exploiter-resistance, search-parity.
- v8 (`runs/anchors/v8_promobar_u243.pt`) is the starting champion/base; capacity/width and piKL are out of scope (later specs).
