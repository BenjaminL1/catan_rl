# Feature Specification: Gumbel Search-Decision Upgrade

**Feature Branch**: `008-gumbel-search-decision`

**Created**: 2026-06-20

**Status**: Draft

**Input**: Extract more playing strength from the FROZEN v8 net by fixing the search's decision rule (the value head is at its ceiling, so the live lever is better decisions, not a better value). Replace the visit-collapsing PUCT root selection + max-visit final-move with Gumbel root selection (Sequential Halving + Gumbel-top-k) + completed-Q, inference-only, with cheap LCB/n-determinization precursors and an SPRT confirmation gate. Search-side only — engine, net, action space, obs schema, and the v8 checkpoint are unchanged.

## Prior Work / Why This Spec *(context)*

- **Spec 007 closed value-target / capacity work as low-EV.** US0 (frozen-trunk probe) showed retargeting the value head does not improve rank; US0.5 (MC-ground-truth probe) showed the value leaves only ~0.08 Spearman vs the best-possible estimator (net_accuracy 0.72–0.86), the residual being irreducible StackedDice variance. **Better value is tapped.**
- **Search's raw-sim scaling is flat for a documented, structural reason.** Determinized PUCT-MCTS adds +~90 Elo over raw v8, but search@50→@100 is within-noise flat (0.615→0.630). The cause is a **visit-collapse**: `config.py` defaults `fpu_mode='zero'` (its own docstring: "collapses visits to one action"; 51% of states put all sims on one action) and the final-move rule is raw max-visit (`mcts.py:268-275`). `fpu_mode='parent'`@100 gave only +4 Elo — the collapse is **structural, not parametric**. The search is buying more sims it cannot use.
- **A literature survey ranked Gumbel #1** (3 of 4 threads): provable low-sim policy improvement (Danihelka et al., "Policy Improvement by Planning with Gumbel", ICLR 2022) that *cannot* visit-collapse, with an external low-sim precedent (MiniZero, arXiv:2310.11305 / IEEE ToG 2024: Gumbel n=16 ≈ standard n=200 on 9×9 Go). It is **inference-only on the frozen net** — matching "better decisions, not better value" — and a near-ceiling value is a *precondition* it exploits (completed-Q only needs correctly-evaluated values), not a blocker.
- **Honest magnitude**: stacked on an already-AlphaZero-style search, realistic per-lever gains are **+10 to +40 Elo** (smaller than the +90 that *adding* search bought), so each claim must be confirmed at matched sim budget. After Gumbel + LCB the curve likely flattens toward a genuine ceiling.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - Cheap precursors that run first and de-risk the build (Priority: P0)

Before the Gumbel build, run two near-free changes that share the eval harness and final-move code path: a better final-move *selection* rule, and a diagnostic that tells us *why* sims are wasted.

**Why this priority**: Both are hours-not-days, default-off, and inform/de-risk US1. The diagnostic decides whether the plateau is collapse-bound (→ Gumbel is the fix) or dice-variance-bound (→ a future chance-node spec). The LCB rule may itself be a small free win and exercises the matched-budget A/B harness US1 needs.

**Independent Test**: (a) A/B the LCB final-move rule vs max-visit at matched sim budget; (b) run the fixed-total-budget n-determinization sweep and read its collapse-vs-variance verdict.

**Acceptance Scenarios**:

1. **Given** the frozen v8 + current search, **When** the final-move rule is switched to LCB `argmax(mean_Q − z·stderr)` (a per-child variance accumulator added to the search node), **Then** at MATCHED sim budget it does not regress vs max-visit (ideally a small gain), and with the flag off, search is byte-identical to today.
2. **Given** the determinization aggregation, **When** K∈{2,4,8} worlds are run at FIXED TOTAL budget (sims/K per world), **Then** a recorded verdict states whether fixed-budget averaging un-flattens @50→@100 (variance-bound) or not (collapse-bound).

---

### User Story 1 - Gumbel root selection + completed-Q (Priority: P1, gated on US0)

Replace the root decision rule: sample a Gumbel-top-k subset of legal root actions and allocate sims by Sequential Halving (which cannot collapse onto one action), and pick the final move by completed-Q (a value-interpolated improved policy) instead of max-visit. Inference-only on the frozen v8 net; root-first (PUCT unchanged at non-root nodes initially).

**Why this priority**: This is the main strength lever — the only candidate that fixes the *measured* defect with a provable low-sim guarantee, on the frozen net. Gated on US0(b) confirming the plateau is collapse-bound.

**Independent Test**: Gumbel-root+completed-Q search@N vs [frozen v8 + current PUCT search@N] at the SAME N, decided by the SPRT gate.

**Acceptance Scenarios**:

1. **Given** the frozen v8 net, **When** Gumbel-root+completed-Q search plays at sims=N, **Then** it allocates sims across the sampled root actions (no visit-collapse) and selects via completed-Q, using only the existing priors + value (no retrain, no net change).
2. **Given** matched sim budget N, **When** Gumbel-search@N plays [frozen v8 + current PUCT search@N] on the SPRT gate, **Then** it reaches a PROMOTE (LLR > +2.94) or REJECT (LLR < −2.94) verdict with 0 rules violations, fixed-seed reproducible.
3. **Given** the Gumbel search-mode flag is off, **When** search runs, **Then** behavior is byte-identical to the current PUCT search.

---

### User Story 2 - SPRT confirmation gate (Priority: P2, infrastructure used by US0/US1)

A sequential, paired, seat-swapped, common-seed test (pentanomial scoring) that decides PROMOTE/REJECT in materially fewer games than the project's fixed n≥600 eval.

**Why this priority**: It is the confirmation protocol every other story uses; on scarce M1 compute it makes each candidate ~2–5× cheaper to confirm. Built early so US0(a)/US1 are gated by it.

**Independent Test**: Run the gate on a known-equal pairing (expect no early PROMOTE) and on US1 (expect a decision); confirm it stops earlier than fixed-n for equal error rates.

**Acceptance Scenarios**:

1. **Given** two search configs, **When** the SPRT gate plays paired seat-swapped common-seed games (pentanomial), **Then** it stops at LLR crossing ±2.94 (α=β=0.05) or a max-games cap, returning PROMOTE/REJECT/INCONCLUSIVE.
2. **Given** a true small positive Elo, **When** the gate runs, **Then** it reaches PROMOTE using fewer games than a fixed n≥600 eval at equal error rates.

---

### Edge Cases

- **Sim-budget mismatch** (the load-bearing failure): any comparison where the two configs use different total sims is INVALID — the claim is "better allocation of the *same* sims." Every gate must assert matched budget.
- **US0(b) says variance-bound** (averaging un-flattens): Gumbel may help less; record it and flag a future chance-node spec (do NOT build chance nodes here — the StackedDice bag is path-dependent, an iid-2d6 model would be wrong).
- **Gumbel ties / very few legal actions**: at 2–6 legal root types, m may cover all of them (Sequential Halving degenerates gracefully to evaluating each); the rule must handle k ≥ #legal.
- **completed-Q with a noisy value**: the value's ~0.68 rank makes the guarantee approximate; root-first limits exposure. If Gumbel regresses vs PUCT at matched budget, it is recorded as a real result, not retried blindly.
- **LCB z too large**: over-penalizes low-visit good moves; z is a tunable, default chosen conservatively.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Add an LCB final-move selection mode — `argmax(mean_Q − z·stderr)` — requiring a per-child sum-of-squares (variance) accumulator in the search node; additive, default-off behind a config flag.
- **FR-002**: Add a fixed-total-budget n-determinization sweep (K worlds at sims/K each) and report a recorded collapse-bound-vs-variance-bound verdict; diagnostic only, no behavior change to the shipped path.
- **FR-003**: Add a Gumbel root-selection mode — Sequential Halving over a Gumbel-top-k sample of legal root actions + completed-Q final-move — inference-only on the frozen net, reusing existing priors + value squash; PUCT retained at non-root nodes; additive, default-off behind a search-mode flag.
- **FR-004**: Add an SPRT gate — paired, seat-swapped, common-seed (determinized-dice-reproducible) games with pentanomial scoring, H0 elo≤0 / H1 elo≥elo1, α=β=0.05, LLR bounds ±2.94, max-games cap — wrapping the eval harness; returns PROMOTE/REJECT/INCONCLUSIVE.
- **FR-005**: Every strength comparison MUST be at MATCHED total sim budget; the gate MUST assert/record the budget of both sides and refuse mismatched comparisons.
- **FR-006**: All search modes MUST be **inference-only** on the frozen v8 net — no retrain, no policy state-dict change, v2-lineage untouched.
- **FR-007**: With all new flags off, search/eval MUST be byte-identical to today (a no-op smoke matches the pre-change baseline); new TensorBoard/JSON scalars only, none renamed.
- **FR-008**: The shipped search design MUST be preserved — max-only tree, no per-ply value sign-flip, open-loop determinization (no opponent/chance nodes introduced).
- **FR-009**: Gate-first ordering — US0 precursors + the collapse-bound diagnostic precede US1; each strength claim is SPRT-confirmed at matched budget before being banked.
- **FR-010**: Reproducibility — fixed `sims_per_move` (not `time_budget_s`), fixed seeds; CPU-pinned eval per the device policy; no GUI import on any search/eval path.

### Key Entities *(include if feature involves data)*

- **LCB selection stat**: per-child `(N, sum_Q, sum_Q²)` → `mean_Q`, `stderr`, used by `argmax(mean_Q − z·stderr)`.
- **Determinization-budget point**: `(sims_per_move, K worlds, sims/K each)` → WR vs raw v8 at fixed total budget — the collapse-vs-variance diagnostic.
- **Gumbel root plan**: the Gumbel-top-k sampled action set + the Sequential-Halving visit schedule + the completed-Q final selection.
- **SPRT match**: a paired seat-swapped common-seed game pair → pentanomial outcome; accumulated LLR vs the ±2.94 bounds.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A recorded US0(b) verdict (collapse-bound vs variance-bound) exists before US1 is built.
- **SC-002**: The LCB final-move rule, at matched sim budget, does not regress vs max-visit (no-regression A/B), and is byte-identical when off.
- **SC-003**: Gumbel-root+completed-Q search@N beats [frozen v8 + current PUCT search@N] at the SAME N, SPRT-PROMOTE (LLR > +2.94), 0 rules violations — or is recorded as not beating it.
- **SC-004**: The SPRT gate reaches a PROMOTE/REJECT decision in materially fewer games than a fixed n≥600 eval at equal error rates.
- **SC-005**: With all new flags off, a no-op smoke reproduces the current search/eval outputs byte-identically; the v8 checkpoint and v2-lineage are unchanged.
- **SC-006**: Every banked comparison is at matched total sim budget, CPU-pinned, seat-symmetric, fixed-seed; 0 engine/action/obs changes.

## Assumptions

- The plateau is collapse-bound (US0(b) will confirm); if variance-bound, Gumbel is down-weighted and a chance-node spec is flagged.
- 2–6 legal root types is a near-ideal regime for Gumbel-top-k at the root (m can cover all legal types).
- "Superhuman" is unverifiable (no human baseline); the operational target is *strongest by in-engine proxies* — Elo/SPRT vs the v-lineage, exploiter-resistance, search-parity.
- Reference algorithm: Danihelka et al. (ICLR 2022); a vetted SPRT/LLR implementation (e.g. the Stockfish-fishtest pentanomial model) is copied, not hand-derived.
- v8 (`runs/anchors/v8_promobar_u243.pt`) is the frozen base; piKL, chance nodes, non-root Gumbel, capacity/value work, and expert iteration are explicitly out of scope (future specs).
