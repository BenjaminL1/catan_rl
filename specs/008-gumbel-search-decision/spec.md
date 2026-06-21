# Feature Specification: Gumbel Search-Decision Upgrade

**Feature Branch**: `008-gumbel-search-decision`

**Created**: 2026-06-20

**Status**: Draft (revised after senior-RL review — kill-gate-first restructure: an oracle root-headroom pre-check + SPRT infra gate the Gumbel build, since the core premise is not yet validated)

**Input**: Extract more playing strength from the FROZEN v8 net by fixing the search's decision rule (value is at its ceiling, so the live lever is better decisions, not a better value). BUT the premise "the flat sim-scaling is a fixable visit-collapse that Gumbel cures" is **not yet validated** — so this spec is **two-stage**: a cheap STAGE-A kill-gate (LCB + SPRT infra + a fixed-budget n-det diagnostic + an oracle root-headroom pre-check with a pre-registered GO rule) that decides GO/NO-GO, and STAGE-B (Gumbel root + completed-Q) built **only on GO**. Inference-only, search-side only — engine, net, action space, obs schema, and the v8 checkpoint are unchanged.

## Prior Work / Why This Spec *(context)*

- **Spec 007 closed value/capacity work as low-EV.** US0/US0.5 probes: value leaves only ~0.08 Spearman vs the best-possible estimator (net_accuracy 0.72–0.86); residual is irreducible StackedDice variance. Better value is tapped.
- **Search raw-sim scaling is flat — but WHY is unproven.** Determinized PUCT-MCTS adds +~90 Elo over raw v8; search@50→@100 is within-noise flat (0.615→0.630). `config.py` defaults `fpu_mode='zero'` (docstring: "collapses visits to one action"; 51% all-sims-on-one-action), final-move is raw max-visit (`mcts.py:268-275`).
- **The central tension (the review's #1 finding, now built into the design):** `fpu_mode='parent'`@100 gave only +4 Elo over `zero` — but that smoke is **n=200 (CI ±~50 Elo)**, statistically inert. It is consistent with BOTH "collapse is structural, Gumbel will fix it" AND "the v8 root is already near-optimal, so no anti-collapse mechanism (FPU *or* Gumbel) raises strength" (the latter consistent with the value ceiling). **A flat n-det curve looks identical under both hypotheses**, so the diagnostic alone cannot discriminate them — hence STAGE-A's **oracle root-headroom pre-check** must establish there is real strength to gain at the root *before* the Gumbel build is funded.
- **If GO**: Gumbel (Danihelka et al., ICLR 2022) cannot visit-collapse and carries a provable low-sim policy-improvement guarantee (MiniZero, arXiv:2310.11305 / IEEE ToG 2024: Gumbel n=16 ≈ standard n=200 on 9×9 Go). Honest magnitude, stacked on existing search, is **+7 to +15 Elo** (about half the generic +10–40, discounted ~30% for the value's Spearman ~0.68 noise on a chance-folded tree), SPRT-confirmed.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - STAGE-A kill-gate: does the root have real headroom? (Priority: P0, gates US1)

Before building Gumbel, cheaply establish whether there is strength to be gained at the root at all, build the confirmation infra everything else uses, and ship the near-free LCB win. This stage produces a recorded GO/NO-GO verdict.

**Why this priority**: The premise is half-falsified by the fpu_parent null and is indistinguishable from the value ceiling. If a near-perfect root chooser does NOT beat current PUCT-root, no decision-rule change (Gumbel included) will help — and the project saves the multi-day US1 build. This is the kill-gate.

**Independent Test**: Land STAGE-A; run it to a recorded `runs/search/008a_verdict.json`; apply the pre-registered GO rule.

**Acceptance Scenarios**:

1. **(SPRT infra — built first)** **Given** two search configs at MATCHED total sim budget, **When** the SPRT gate plays paired seat-swapped common-seed games (pentanomial), **Then** it returns PROMOTE (LLR > +2.94) / REJECT (LLR < −2.94) / INCONCLUSIVE (max-games cap), with `elo1` defined (default ~+10 Elo) and ~2–5× fewer games than a fixed n≥600 eval at equal error rates.
2. **(LCB final-move)** **Given** a per-child in-memory second-moment accumulator, **When** the final move is chosen by `argmax(mean_Q − z·stderr)` (default z=1.96), **Then** at matched budget it does not regress vs max-visit (SPRT not REJECT), and with the flag off it is byte-identical to today.
3. **(Fixed-budget n-det diagnostic)** **Given** the budget-split mechanism (sims/K per world, K worlds, total matched), **When** K∈{2,4,8} is swept, **Then** a recorded readout reports per-K WR + the **per-depth visit-concentration** (% of nodes with >50% visits on one action, depths 0,1,2) + **Spearman(root-child value, ex-post terminal outcome)** on a 50-game both-seats sample.
4. **(Oracle root-headroom pre-check — the kill probe)** **Given** the frozen v8, **When** a near-perfect root chooser (strong-rollout / high-budget oracle at the root, PUCT below) plays current PUCT-root at MATCHED total budget on the SPRT gate, **Then** the root headroom is measured. **GO RULE (pre-registered): build US1 ONLY IF (oracle root headroom > +15 Elo) AND (depth-0 visit-collapse is high) AND (root-child-value Spearman ≥ 0.60). Otherwise STOP, record NO-GO, and flag a chance-node/belief spec instead.**

---

### User Story 1 - STAGE-B: Gumbel root selection + completed-Q (Priority: P1, gated on US0 GO)

Only if STAGE-A returns GO: replace the root decision rule with Sequential Halving over a Gumbel-top-k sample of legal root actions (cannot collapse), and the final move with completed-Q. Inference-only on the frozen net.

**Why this priority**: The main lever — but funded only after the kill-gate proves root headroom exists. If STAGE-A is NO-GO, this story is NOT built.

**Independent Test**: Gumbel-search@N vs [frozen v8 + current PUCT search@N] at the SAME N, decided by the SPRT gate; measured uplift compared to the pre-registered +7–15 Elo band.

**Acceptance Scenarios**:

1. **Given** STAGE-A GO and root-child Spearman ≥ 0.60, **When** Gumbel-root runs, **Then** it samples m=min(gumbel_k, #legal) actions (gumbel_k default ~8, temperature 1.0), allocates by Sequential Halving (rounds=ceil(log2 m), standard per-round allocation), and selects by completed-Q `(N(a)·Qmean(a) + V_root) / (N(a)+1)` using the existing squashed root value; degenerates gracefully when m≥#legal and is a no-op at forced (1-legal) roots.
2. **Given** STAGE-A showed root-child Spearman < 0.60, **When** US1 is built, **Then** the **Gumbel-max-Q** variant (no completed-Q interpolation) is used first (completed-Q's guarantee is too weak under that noise).
3. **Given** the per-depth diagnostic showed depth-1+ collapse wastes >~20% of sims, **When** US1 is built, **Then** Gumbel/Sequential-Halving (or `fpu_mode='parent'`) is extended below the root; else root-only is kept with the per-depth justification recorded.
4. **Given** matched sim budget N, **When** Gumbel-search@N plays current PUCT search@N on the SPRT gate, **Then** it reaches PROMOTE/REJECT with 0 rules violations, fixed-seed reproducible; the uplift is compared to the pre-registered band.
5. **Given** the Gumbel flag off, **When** search runs, **Then** behavior is byte-identical to the current PUCT search.

---

### Edge Cases

- **Sim-budget mismatch (the load-bearing failure):** "matched total sim budget" = sum of leaf-evaluations across ALL determinizations; both sides MUST use equal `n_determinizations` (default 1 for fair A/Bs). The gate MUST assert `total_sims_run` equal on both sides and **raise** on mismatch.
- **STAGE-A NO-GO:** record it; do NOT build US1; flag a chance-node/belief spec (the StackedDice bag is path-dependent — an iid-2d6 chance model would be wrong).
- **LCB degenerate stderr / ties:** a many-times-visited high-Q action has stderr≈0 (LCB→max-visit); apply an eps floor on stderr; unvisited children are non-competitive; tie-break by (lcb_value, visits, prior).
- **Gumbel at forced/2-legal roots:** m≥#legal → Sequential Halving degenerates to "evaluate each, pick by completed-Q"; forced (1-legal) root → no-op.
- **completed-Q under noisy value:** root children are EndTurn nodes whose value already averages over the opponent turn + a dice draw (chance-folded), and value Spearman ~0.68 inflates regret ~30% — hence the Spearman≥0.60 gate and the max-Q fallback.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (SPRT, built first)**: Add an SPRT gate (e.g. `search/sprt.py`): paired seat-swapped common-seed games, pentanomial 5-outcome aggregation, incremental LLR vs ±2.94, H0 elo≤0 / H1 elo≥`elo1` (default +10 Elo), α=β=0.05, max-games cap → PROMOTE/REJECT/INCONCLUSIVE. Copy a vetted formula (Stockfish-fishtest pentanomial), do not hand-derive. Tests: known-equal→INCONCLUSIVE; mock +30-Elo→PROMOTE in ≪ fixed-n.
- **FR-002 (LCB)**: Add `final_move_mode='lcb'` (vs `'max_visit'` default) — `argmax(mean_Q − z·stderr)`, default z=1.96, via an IN-MEMORY per-child second-moment accumulator (`sum_Q2 += value²` in the node's record(); NOT persisted → no state-dict change), `stderr=sqrt(max(eps, sum_Q2/n − (sum_W/n)²)/n)`, eps floor, unvisited non-competitive, tie-break (lcb, visits, prior). Additive, default-off.
- **FR-003 (fixed-budget n-det diagnostic + budget split)**: Add a budget-split so `n_determinizations=K` runs `sims_per_move//K` per tree at MATCHED total (the current code runs full sims per tree — `mcts.py:331-332` — so this is new); sweep K∈{2,4,8}; record per-K WR + per-depth visit-concentration + root-child-value Spearman. Diagnostic only.
- **FR-004 (oracle root-headroom pre-check + GO rule)**: Add a root-only oracle/strong-rollout chooser; measure its Elo over current PUCT-root at matched budget via the SPRT gate. **Pre-registered GO rule gating US1: headroom > +15 Elo AND high depth-0 collapse AND root-child Spearman ≥ 0.60; else NO-GO.** Record the verdict to `runs/search/008a_verdict.json` (schema: `{configs, sprt_results, headroom_elo, per_depth_collapse, root_child_spearman, go:bool, reason}`).
- **FR-005 (Gumbel, STAGE-B, gated)**: Add `root_select_mode='gumbel'` — Sequential Halving over Gumbel-top-k (m=min(gumbel_k,#legal), gumbel_k default 8, temp 1.0, rounds=ceil(log2 m), standard allocation) + completed-Q `(N·Qmean+V_root)/(N+1)` (V_root = existing squashed root value); Gumbel-max-Q fallback if root-child Spearman<0.60; root-first unless the per-depth metric mandates extending below; default-off; inference-only.
- **FR-006 (matched budget, operational)**: Define matched total sim budget = Σ leaf-evals across determinizations; require equal `n_determinizations` on both sides; the gate asserts equality and **raises** on mismatch; a budget-parity smoke test.
- **FR-007 (inference-only)**: All modes inference-only on the frozen v8 net — no retrain, no policy state-dict change, v2-lineage untouched (the LCB/Gumbel stats are in-memory only).
- **FR-008 (additive/default-off + tested)**: With all new flags off (`final_move_mode='max_visit'`, `root_select_mode='puct'`, no budget split), search/eval are byte-identical to today; add an all-flags-off byte-identity regression test over the full search output (visit counts + chosen action) at a fixed seed. New TB/JSON scalars only.
- **FR-009 (design preserved)**: max-only tree, no per-ply value sign-flip, open-loop determinization — no opponent/chance nodes introduced.
- **FR-010 (gate-first + reproducibility)**: STAGE-A precedes STAGE-B; US1 built only on GO; every strength claim SPRT-confirmed at matched budget; fixed `sims_per_move` (not time-budget), fixed seeds, CPU-pinned eval, no GUI import. Config uses enums (`final_move_mode`, `root_select_mode`) not boolean sprawl.

### Key Entities *(include if feature involves data)*

- **SPRT match**: paired seat-swapped common-seed game-pair → pentanomial outcome; accumulated LLR vs ±2.94; (elo1, α, β, max-games).
- **LCB stat**: in-memory per-child `(N, sum_W, sum_Q2)` → `mean_Q`, `stderr`, `lcb = mean_Q − z·stderr`.
- **008a verdict**: `{per-K WR, per-depth visit-concentration, root-child-value Spearman, oracle headroom Elo, go, reason}` → the kill-gate decision.
- **Gumbel root plan**: Gumbel-top-k sampled set + Sequential-Halving schedule + completed-Q (or max-Q) selection.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: STAGE-A produces a recorded `008a_verdict.json` with the oracle root-headroom Elo, per-depth collapse, root-child Spearman, and a GO/NO-GO per the pre-registered rule — before any US1 build.
- **SC-002**: The SPRT gate reaches PROMOTE/REJECT using materially fewer games than a fixed n≥600 eval at equal error rates (verified on a mock +30-Elo pairing).
- **SC-003**: The LCB final-move rule does not regress vs max-visit at matched budget (SPRT not REJECT) and is byte-identical when off.
- **SC-004 (only if GO)**: Gumbel(-completed-Q or -max-Q) search@N beats current PUCT search@N at the SAME N, SPRT-PROMOTE, 0 rules violations; the uplift is reported against the pre-registered +7–15 Elo band — or recorded as not beating it.
- **SC-005**: With all new flags off, an all-flags-off regression test reproduces current search output (visit counts + chosen action) byte-identically; v8 checkpoint + v2-lineage unchanged.
- **SC-006**: Every banked comparison is at matched total sim budget (asserted), CPU-pinned, seat-symmetric, fixed-seed; 0 engine/action/obs changes.

## Assumptions

- The oracle pre-check (FR-004) is a faithful upper bound on root-decision headroom (a strong-rollout/high-budget chooser approximates "best possible root move on the frozen net"); if it shows <+15 Elo, decision-rule work (incl. Gumbel) is NO-GO and the bottleneck is elsewhere (chance/belief or a true ceiling).
- 2–6 legal root types means Gumbel-top-k often covers all legal actions (leverage is inherently limited; recorded honestly).
- "Superhuman" is unverifiable (no human baseline); target = strongest by in-engine proxies (SPRT/Elo vs the v-lineage, exploiter-resistance, search-parity).
- A vetted SPRT/pentanomial implementation is copied (Stockfish-fishtest), not hand-derived.
- v8 (`runs/anchors/v8_promobar_u243.pt`) is the frozen base; piKL, chance nodes, non-root Gumbel (unless the per-depth metric mandates it), capacity/value work, and expert iteration are out of scope (future specs).
