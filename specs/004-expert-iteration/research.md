# Phase 0 Research: Expert Iteration — resolved decisions

All forks resolved with sensible defaults (single-M1, reuse-first). Decisions marked
**[REVIEW]** are flagged for the user (they trade off rigor vs cost/scope).

## D1 — Distillation target: HARD action (pilot) vs SOFT visit distribution

**Decision**: the pilot distills the search's **chosen action** (hard label), reusing the
BC loss's per-head CE exactly as-is. **[REVIEW]**
**Rationale**: the BC pipeline already does hard-action per-head CE + value MSE + belief
soft-CE; recording search's argmax action into the existing shard format = zero new loss
code, maximal reuse, fastest path to the gate. The search's *root visit distribution*
(FR-001) is lower-variance and the "purer" ExIt target, but consuming it needs a BC-loss
extension (soft policy CE) — deferred to a refinement if the pilot is marginal.
**Alternative**: soft visit-distribution targets (AlphaZero-style) — better signal, more
code; revisit if the hard-action pilot lands at WR ~0.52-0.55 (capturing only part of the
teacher).

## D2 — Distillation = fine-tune v6 (warm-start), not train-from-scratch

**Decision**: add a backward-compatible `init_ckpt: Path | None = None` to `train_bc`; when
set, load that checkpoint's weights into the policy before training. ExIt fine-tunes v6 on
the search shards. **[REVIEW — touches the shared BC trainer]**
**Rationale**: ExIt improves the *current* policy; training from scratch on a few-thousand
search positions would underperform v6. Warm-start is the minimal, generally-useful change
(default None = today's from-scratch behavior, byte-identical).

## D3 — Labeler: new search-game recorder emitting BC-compatible shards

**Decision**: a new `labeler.py` plays `SearchAgent` (003) vs an opponent, recording at each
AGENT decision `(obs, search_action, mask, belief_target)` and filling `z_disc` (discounted
game outcome) at game end — written as BcDataset-compatible NPZ shards + manifest (reusing
`_DecisionRecord` + `_flatten_records` + the shard writer).
**Rationale**: the BC dataset generator instruments the *heuristic* player's `move()` (a
context manager) — not reusable for a search agent that drives via the env. But the on-disk
shard FORMAT is exactly what BcDataset reads, so we reuse the format + flatten/writer and
only add the search-driven play loop (mirrors `eval_search` / the search replay recorder).

## D4 — Value target: discounted game outcome (z_disc), same as BC

**Decision**: `z_disc = γ^(T-t)·z` from the realized game outcome (γ=0.998, BC default), so
the value head distills toward search-game outcomes alongside the policy head.
**Alternative**: the search root value as the value target — closer to AlphaZero, but the
003 search value is the squashed leaf, not a clean game-outcome estimate; outcome is simpler
and matches the BC value head's existing target semantics. Revisit in US2.

## D5 — Label-game opponent: search-vs-heuristic (pilot)

**Decision**: pilot label games are `search@50` vs the heuristic. **[REVIEW]**
**Rationale**: cheap, diverse, and the heuristic gives varied non-degenerate positions; the
agent (search) plays both seats across the seat-symmetric set. Search self-play (search vs a
frozen v6 snapshot) yields on-distribution positions but doubles cost (both seats search) —
deferred to US2 as a diversity/quality lever.

## D6 — Pilot gate: search-FREE eval, Wilson LB > 0.50, n≥200→500

**Decision**: `evaluate_policy_vs_policy(distilled_ckpt, v6_ckpt)` (single-forward, no MCTS),
seat-symmetrized, PASS iff Wilson lower bound > 0.50 at n≥200 then a disjoint n≥500 AND the
distilled policy still beats the heuristic above a floor (default 0.60 — v6 is ~0.9, so a
healthy distill stays well above; a distill that overfit to v6 and collapsed elsewhere fails
as catastrophic forgetting). The prior-lineage rung from the original sketch is DEFERRED to
US2 — for the one-round pilot, v6 is the only prior lineage and is already the head-to-head
opponent, so heuristic-floor forgetting detection suffices.
**Rationale**: the whole point is a *fast* stronger policy; the gate must be search-free. The
eval is cheap (no search), so n=500 is minutes. LB>0.50 = significant improvement (the spec's
SC-001; WR≥0.55/+35 Elo is the SC-002 aspiration).

## D7 — Pilot dataset size: ~5,000 non-forced positions (~60-80 search games)

**Decision**: target ~5k non-forced labeled positions for the pilot (~60-80 search games @
50 sims, ~45-60 min CPU). **[REVIEW — size vs signal]**
**Rationale**: enough to fine-tune the ~1.5M-param policy a few epochs without overfitting,
small enough to generate in ~1h on one M1. If the gate is marginal, the documented pivot is
"more positions / stronger search per label (more sims, via batched leaf eval)".

## D8 — Distillation hyperparameters (fine-tune, not from-scratch)

**Decision**: low peak LR (~5e-5, an order below from-scratch BC's 3e-4 — we're nudging, not
retraining), short schedule (~3-5 epochs, early-stop on val), keep `value_weight=0.10` /
`belief_weight=0.05`, trunk unfrozen. **[REVIEW — LR/epochs/freeze are the main knobs]**
**Rationale**: warm-started fine-tuning needs a small LR to avoid clobbering v6; few epochs on
5k samples avoids overfitting. These are the most likely things to tune if the pilot is
marginal.

## Throughput note (FR-012, US2 only)

Search labeling is NN-forward-bound (~120 sims/sec). For the full flywheel (tens of thousands
of positions/round × multiple rounds), batched leaf evaluation (many MCTS leaves per forward)
is the 5-10× multiplier. In scope ONLY after the pilot passes — the pilot is sized to current
throughput so the gate is reached without it.

## Pilot outcome — US1 GATE FAILED (2026-06-15)

`runs/exit/round_0/gate.json`: **quick n=200 WR 0.49 vs raw v6, Wilson LB 0.422 ≤ 0.50** →
FAIL (confirm not run). 0 ruleset violations. Run was clean end-to-end (5,055 search@50 labels
in 38 min → 5-epoch warm-started MPS distill in 22 min → search-free gate in 5 min). The
distilled policy is **statistically indistinguishable from v6** (CI [0.422, 0.558] straddles
0.50) — distillation did not move it.

**Root cause (the finding): hard-action distillation captured too little signal.** D1's hard
label = the search's *chosen action*. But on most positions a 50-sim search's argmax AGREES
with v6's own argmax (search refines, rarely overturns the top move of an already-decent
policy), so distilling "search's action" ≈ distilling "v6's own argmax" ≈ an identity map →
WR ≈ 0.50. The real ExIt signal lives in the *visit distribution* (where search is uncertain,
or strongly prefers a NON-argmax move) and the *margin* — both discarded by hard labels.

**Recommended pivots (ranked; USER DECISION — not auto-pursued):**
1. **Soft visit-distribution targets** (KL/soft-CE toward the search root's normalized visit
   counts over the type head + conditional sub-heads), replacing hard-action CE. The principled
   ExIt target; carries the full search preference. Needs a `bc_loss` soft-policy extension.
   *(Top recommendation — directly attacks the dilution.)*
2. **Train only on DISAGREEMENT positions** (search argmax ≠ v6 argmax), or up-weight them —
   concentrates the learning signal; cheap; composes with (1) or hard labels.
3. **Stronger teacher** (higher sims-per-label → more positions where search overturns v6),
   which needs batched leaf eval (FR-012) to stay tractable; composes with (1).
4. **LR/epoch sweep** (D8) — least likely to fix the *dilution* root cause, but a cheap check
   that the fine-tune isn't simply too gentle (the distilled landed ≈ v6, not worse, so it did
   move — just toward v6's own argmax).

The gate-first discipline worked: this retired a weak approach in ~1 h + the minimal prototype,
before building the flywheel. Decision deferred to the user; US2/US3 NOT built.

## Round-2 de-risk — SOFT + SUB-ACTION distillation REJECTED (2026-06-16)

User chose pivot (1) + "go full" (let search explore sub-actions, then distill the full soft
visit distribution). Built the apparatus (additive multi-representative priors,
`sub_actions_per_type` threaded SearchConfig→build_node→MCTS; `MCTS.run` diagnostics now expose
`visit_counts` + `priors`; commit `b1a0e7f`, 64 search tests green) and **re-probed BEFORE
building the soft pipeline** (`/tmp/search_where_probe.py`, `/tmp/value_lookahead_probe.py`).

**Probe A — full-action gap (k=4 sub-action search @200 sims, 80 where-rich on-distribution
placement states, mean 6.3 candidate actions):**
- `full_tv(search visits, v6 prior)` = **0.025** (97.5% overlap) — vs type-only `type_tv` 0.011.
  Sub-action search ~doubled the divergence over k=1 (0.012) but the absolute level stays tiny.
- search overturns v6's greedy FULL action **1 / 80 = 1.25%** (the one override was a where-move);
  placement-only where-override = 1 / 71 = 1.4%.
- **Conclusion: REJECTED.** v6 is confident about *where* to build, not just *which type*. A
  whole-distribution soft target (type + sub-action) is ~98% identical to v6's own prior →
  distilling it ≈ identity → would reproduce the ~0.49 pilot. Reconciles with "search@50 beats
  v6 0.578 in games": search's edge is ~1-2% rare PIVOTAL overrides, too sparse for
  whole-distribution distillation to capture (the signal is averaged to ~zero across the set).

**Probe B — value lookahead (same setting, 70 states):** unlike the action distribution, the
search-backed value DOES diverge from v6's leaf value: mean `|best_q − root_value|` = **0.056**,
p90 = **0.155**, **systematic +0.030 optimism bias** (v6 under-rates its own win-prob), 33% of
states corrected by >0.05 (13% by >0.10), corr 0.98. → Value carries a live, non-identity signal.

**Implication / refined fork (USER DECISION):** policy-action distillation (hard OR soft, type OR
sub-action) is **empirically dead** for producing a stronger SEARCH-FREE agent — v6 already
imitates search's actions ~98%. The remaining levers:
- **A. Accept search as inference-only** (shipped +55 Elo) and close ExIt. The action-distillation
  thesis is retired by two probes.
- **B. Value-targeted distillation** — distill `best_q` (search-backed value) into the value head.
  Non-identity (Probe B). Does NOT change search-free move selection (the value head isn't
  consulted at action time), so it does not directly satisfy the 004 "fast agent stronger than
  v6" goal; it sharpens leaf values (modestly helps the inference-search lever) and could give a
  PPO/self-play restart better-calibrated advantages (fixes the +3pp pessimism). Cheap (reuse
  pilot pipeline, swap z_disc→best_q).
- **C. Disagreement-targeted policy distillation** — the ONLY path to a stronger search-free
  policy: filter/up-weight the ~1-2% pivotal override states instead of averaging across all.
  Expensive: ~50-100× more searched positions to collect enough disagreement labels → needs
  batched leaf eval (FR-012, was scoped post-gate); highest effort + highest uncertainty.

## Settling experiment — the "distillation is dead" verdict was an ARTIFACT, REVERSED (2026-06-16)

A senior-RL panel review (4 lenses + adversarial synthesis) found the round-2 probes measured a
search that **mechanically cannot explore**: `node.py` hardcodes **FPU = 0**, and there is **no
root Dirichlet noise / temperature** — so with squashed leaf values ~0.5-0.9 every unvisited
sibling sits at Q=0 below the prior argmax and PUCT never revisits it. Verified in the value-probe
JSON: **median chosen-action visit-share = 1.000, 51% of states put ALL sims on one action.** So
"search agrees with v6 ~98%" was substantially circular. Second confound (pragmatist/skeptic):
the probes sampled v6's OWN self-play distribution — the easy states v6 already navigates — not
the states search actually visits.

Built additive, default-off exploration knobs (`root_dirichlet_alpha`/`_fraction`, `fpu_mode`
`zero`|`parent`; `MCTS.run` also exposes per-action `action_q`; commit `badd0b3`, 73 search tests
green, shipped search byte-identical when off) and ran the settling experiment
(`/tmp/settling_probe.py`): step games with the EXPLORING search (FPU=parent + Dir(0.5)@0.25),
n=150 search-stepped non-forced states, compare each search's best action to v6's clean prior
argmax + the per-override value margin.

**Result — the null is REFUTED:**
- visit collapse broken: median visit-share 1.000 (baseline FPU=0) → **0.795** (exploring), mean
  3.17 distinct actions visited.
- **override rate = 7.3%** (Wilson 95% CI **[4.1%, 12.7%]**) — vs the 1.25% the flawed probe
  reported. Lower bound 4.1% is ABOVE the "signal real" threshold, far above the 2% "dead" line.
- **every override is value-positive**, mean margin **+0.154 win-probability** — search's overrides
  are large, genuine improvements, not noise.
- Nuance: override rate is 7.3% for BOTH baseline and exploring configs on this distribution → the
  DOMINANT fix was the SAMPLING DISTRIBUTION (DAgger: evaluate the teacher on the states it
  visits). FPU+noise fixed a DIFFERENT thing — the collapsed visit distribution (one-hot → spread),
  which is what makes a *soft* visit-count target usable.

**Reconciliation + why the pilot still failed (corrected — the pilot was NOT mis-sampled):** the
WHERE-PROBE that produced "1.25% / dead" stepped with raw v6 (easy distribution) — that was the
artifact. The PILOT, however, ALREADY stepped with the SearchAgent (`labeler.py:54`
`agent.choose_action(env)`), so its data was on the 7.3%-override search distribution — it had real
disagreement signal and STILL returned WR 0.49. So the pilot's failure is NOT distribution. The two
real causes: (1) it used HARD-action labels from a COLLAPSED (FPU=0) teacher whose visit
distribution is one-hot — unusable as a soft target, and hard labels under-extract a 7%-sparse
signal; (2) it stopped at an UNDERPOWERED n=200 quick gate — a 7%-override distill plausibly yields
only ~+2-4% WR, and at n=200 the Wilson half-width is ~7%, so LB 0.422 is fully CONSISTENT with a
small true improvement the gate couldn't resolve (n=500 was never run). 7.3% × ~15pp margin is
exactly consistent with search@50's 0.578 WR.

**CORRECTED PATH (recommended, gate-first):** a corrected ExIt PILOT before the full flywheel. Keep
the (already search-stepped) labeler; change three things: (a) teacher = EXPLORING search
(FPU=parent + root noise, sims ~50-100) so the visit distribution is non-degenerate and usable as a
SOFT target; (b) targets = completed-Q soft targets or value-margin-weighted labels (amplify the
~7% high-margin overrides over the 93% identity labels); (c) a PROPERLY-POWERED gate — n≥500, not
the n=200 quick gate the first pilot stopped at. Cost ≈ the first pilot (reuses the pipeline; no
batched-eval needed at pilot scale). Escalate to the full Gumbel-AlphaZero + batched-leaf-eval
flywheel only if the corrected pilot's Wilson LB > 0.50 at n≥500. **Options A (inference-only) / B
(value-distill) survive as fallbacks, but the "policy distillation is dead" framing is withdrawn —
the signal is real; the open question is whether the policy can absorb it (a capacity ablation at
~1.5M params is a cheap parallel check the panel flagged).**
