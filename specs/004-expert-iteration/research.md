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
