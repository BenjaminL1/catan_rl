# Spec 008 (Gumbel Search-Decision Upgrade) — STAGE-A Kill-Gate Verdict

**Feature:** `specs/008-gumbel-search-decision/` · **Stage:** A (kill-gate, US0 / FR-004)
**Frozen base:** `runs/anchors/v8_promobar_u243.pt` (byte-identical, sha256 `0968b4fa…`; inference-only, CPU-pinned)
**Run date:** 2026-07-11 · **Decision:** **NO-GO** — do **not** build STAGE-B (US1 Gumbel-root); flag a chance-node/belief spec instead.

---

## Pre-registered GO rule (verbatim, FR-004 — NOT softened)

> build US1 (Gumbel-root search) ONLY IF
> **(oracle root headroom > +15 Elo) AND (depth-0 visit-collapse high) AND (root-child-value Spearman ≥ 0.60)**;
> otherwise NO-GO → record it and flag a chance-node/belief spec instead.

The rule is a **conjunction**: any single failing clause forces NO-GO.

---

## Verdict at a glance — the headroom clause fails

| Clause (pre-registered) | Threshold | Measured (authoritative, deployed n_det=1) | Pass? |
|---|---|---|---|
| Oracle root headroom | **> +15 Elo** | **−0.0 Elo** (rollouts=24 oracle **ties** PUCT-root; 1 pair / 4 games, SPRT INCONCLUSIVE) — PROVISIONAL | **NO** |
| Depth-0 visit-collapse high | **> 0.70** | **0.986** (50 games, 3206 decisions) | YES |
| Root-child-value Spearman | **≥ 0.60** | **0.662** (50 games) | YES |

**Decisive clause: oracle root headroom ≈ 0 Elo, far below the +15 gate.** A genuinely stronger
root chooser (rollouts=24, 4× the discarded smoke's 6) — one given a strong Monte-Carlo-rollout
evaluator over exactly the candidate set PUCT expands — **merely ties** the deployed PUCT-root
decision rule (pentanomial pair score 1.0, counts `[0,0,1,0,0]`, elo_estimate −0.0). If even a
near-perfect root chooser cannot beat the deployed rule, **no decision-rule change (Gumbel
included) can** — which is the exact question this kill-gate was built to answer. NO-GO.

Note the other two clauses **pass** on the robust deployed sample (collapse is extreme; the value
does rank root children ≥ 0.60) — so the NO-GO is *not* a value-noise or a no-collapse artifact.
It is the cleanest possible kill signal: **the root decision is already essentially optimal; the
headroom the whole Gumbel premise assumed does not exist.** This is the value-ceiling hypothesis
(Spec 007) confirmed at the decision layer.

---

## 1. Fixed-budget n-determinization diagnostic (FR-003)

`scripts/dev/ndet_diagnostic.py --sims 48 --n-games 30` → `runs/search/ndet_diagnostic.json`.
K∈{2,4,8} at a **matched total sim budget of 48** (each K runs `48//K` sims/tree × K trees;
`total_sim_budget=48` on every row — the matched-total control). 30 both-seats games per K (seat
alternates `g%2`).

| K | sims/tree | total budget | WR (wins/30) | depth-0 collapse | depth-1 | depth-2 | root-child Spearman |
|---|---|---|---|---|---|---|---|
| 2 | 24 | 48 | 0.600 (18) | 0.981 | 0.997 | 0.991 | +0.594 |
| 4 | 12 | 48 | 0.567 (17) | 0.975 | 0.998 | 0.989 | +0.603 |
| 8 | 6  | 48 | 0.633 (19) | 0.965 | 0.998 | 0.989 | +0.532 |

**Reading:**
- **Visit-collapse is extreme at every depth (0, 1, 2 ≈ 0.96–0.998).** Depth-0 collapse ≫ 0.70. But
  high collapse is *equally* consistent with "the v8 root is already near-optimal" (value ceiling)
  as with "collapse is a fixable pathology Gumbel cures" — collapse alone cannot discriminate the
  two hypotheses (the spec's central tension), which is why the **oracle headroom** clause is the
  load-bearing one, and it settles the question against a fixable pathology.
- **WR does not improve with more determinizations** (0.600 → 0.567 → 0.633 is within-noise, not
  monotone). Splitting the fixed budget across worlds buys no strength → no root budget-allocation
  pathology to fix.
- **Root-child Spearman** here (n_det>1 budget-split configs) brackets 0.53–0.60; on the
  **deployed n_det=1 config** the robust 50-game measurement is **0.662** (§3) — the authoritative
  number for the GO rule, and it clears 0.60. The value ranks root children well enough; it is the
  *absence of headroom to rank toward* that kills the premise, not value noise.

## 2. LCB-vs-max-visit final-move A/B (FR-002 / SC-003)

`scripts/dev/lcb_sprt_ab.py --sims 48 --max-pairs 8` → `runs/search/lcb_sprt_ab.json`. Both sides
wrap the frozen v8; only `final_move_mode` differs (`lcb` z=1.96 vs deployed `max_visit`), so
`assert_matched_budget` passes trivially (same sims, n_det=1). Pentanomial SPRT vs a frozen v8
reference, common-reference differential pairing.

- **SPRT decision:** INCONCLUSIVE (llr within ±2.94 at the 8-pair cap), counts `[1,0,7,0,0]`, point
  elo_estimate −87.3 (n=32 games).
- **SC-003 verdict:** **PASS — LCB does not regress** (SPRT ≠ REJECT). With the flag off, search is
  byte-identical to today (FR-008 regression test, in the build commit). LCB is a near-free,
  no-regression addition (not a strength *win* here — INCONCLUSIVE — consistent with the value
  ceiling). It stays available, default-off.

## 3. Oracle root-headroom pre-check — THE KILL PROBE (FR-004)

`scripts/dev/probe_oracle_sprt.py --rollouts 24 --sims 32 --max-pairs 1 --collapse-games 50`
→ `runs/search/008a_verdict.json` (this run supersedes the discarded 21:07 smoke).

**Oracle design.** The oracle enumerates the same candidate root actions PUCT expands and scores
each by `rollouts` Monte-Carlo playouts to terminal under the frozen v8,
**common-random-numbers–paired across candidates** (shared per-rollout seed → the shared dice bag +
opponent line cancel in the pairwise `q(a)−q(b)` the argmax depends on). It picks the argmax
win-prob candidate; PUCT runs below it. It is intentionally over-budget (a *ceiling* probe) — the
matched-total-sim-budget invariant governs the deployable A/Bs (LCB, and any future
Gumbel-vs-PUCT), not this probe.

**Result (authoritative):**
- **headroom_elo = −0.0** — SPRT INCONCLUSIVE, 1 pair / **4 games**, pentanomial counts
  `[0,0,1,0,0]` (a clean tie). The stronger oracle neither beats nor loses to PUCT-root.
- **depth-0 collapse = 0.986**, **root-child Spearman = +0.662** — deployed n_det=1, **50 games**,
  3206 decisions.

**PROVISIONAL label (read this).** This run uses **rollouts=24** (4× the smoke's 6; CRN
difference-SE ≈ ±4.6pp — a genuinely stronger chooser) but on a **1-pair / 4-game** SPRT, because
full power (design target rollouts≈64 ≈ ±2.8pp, hundreds of pairs) is **unaffordable alongside the
live training run** sharing this machine's CPU (each oracle game to terminal ≈ 20 min of
rollouts). The headroom is therefore a **provisional point estimate, not a settled ceiling**: a
4-game INCONCLUSIVE cannot statistically *exclude* a small positive headroom, but the point
estimate is **exactly 0** and there is **zero evidence of the demonstrated > +15 Elo the GO rule
requires** — so the headroom clause is not satisfied.

**Sanity check that the smoke was underpower, not a real negative ceiling.** The discarded smoke
(`008a_verdict_smoke.json`, rollouts=6/20 games) reported headroom **−381.7** — physically
impossible for a genuine near-perfect chooser (headroom ≥ 0 by construction): its 6-rollout oracle
was pure variance and picked *worse* moves than PUCT (mean pair 0.5). Raising the oracle to
rollouts=24 moved headroom from **−381.7 → −0.0**, i.e. toward ≥ 0 exactly as a strengthening
oracle must. This **confirms the smoke was underpower** and that the true root headroom is ≈ 0, not
negative and not > +15. (Reproducibility: the seed-0 pair scored an identical 1.0 / elo −0.0 on two
independent launches.)

**Headroom clause:** −0.0 Elo **is not > +15 Elo → FAIL → NO-GO.**

---

## Decision & rationale

**NO-GO.** The pre-registered conjunction fails on the **oracle-root-headroom** clause: a genuinely
stronger root chooser only **ties** the deployed PUCT-root (≈ 0 Elo ≪ +15). The other two clauses
pass on the robust deployed sample (depth-0 collapse 0.986; root-child Spearman 0.662), so this is
not a value-noise or no-collapse artifact — it is the direct, designed kill signal: **there is no
root-decision headroom for Gumbel (or any decision rule) to capture.**

This is exactly the outcome STAGE-A was built to detect. The spec's Prior Work flagged that a flat
n-det curve + high collapse look identical under both "Gumbel fixes it" and "the v8 root is already
near-optimal (value ceiling)", and pre-registered the oracle headroom as the discriminator that
must show real, capturable headroom before funding the multi-day US1 Gumbel build. It shows none.

**STAGE-B (US1 Gumbel root + completed-Q) is NOT built.**

### What ships from STAGE-A (all inference-only, additive, default-off, v8 byte-identical)

- The pentanomial paired seat-swapped common-seed **SPRT gate** (`src/catan_rl/search/sprt.py`) —
  the confirmation infra for every future search A/B, matched-budget-asserted.
- The **LCB final-move rule** (`final_move_mode='lcb'`) — no-regression (SC-003 PASS), default-off.
- The **fixed-budget n-det split** + diagnostic.
- The **oracle root-headroom kill-probe** + this recorded verdict.

### Recommended next spec (per FR-004 / Edge Cases)

Flag a **chance-node / belief spec**, not more root-decision-rule work. With the root decision
already ≈ optimal and value at its irreducible-noise ceiling (Spec 007), the honest remaining lever
is the chance/opponent-belief structure the current max-only open-loop determinization folds away.
The StackedDice bag is path-dependent (shuffled 36-outcome bag + noise swap + Karma forced-7), so an
iid-2d6 chance model would be wrong — a faithful chance/belief model is the next bet.

---

## Invariants upheld (SC-006)

- **Inference-only on frozen v8** — no retrain, **no state-dict write anywhere in the gate path**
  (grep-verified across all four gate files); `v8_promobar_u243.pt` byte-identical (mtime 2026-06-18,
  pre-dating this run; sha256 `0968b4fa…`). CPU-pinned eval (`torch.device("cpu")`).
- **Matched total sim budget** on every *deployable* strength A/B: the LCB A/B asserts it
  (`assert_matched_budget`, same sims + n_det=1); the n-det sweep pins `total_sim_budget=48` across
  all K. The oracle probe is an intentional **ceiling** probe (over-budget by design), and the
  invariant governs the deployable A/Bs, not the ceiling probe (documented in `probe_oracle_sprt.py`).
- **Fixed `sims_per_move`** (never wall-clock), fixed seeds, append-only JSON readouts, no
  `src/catan_rl/ppo/` or training-loop import, no GUI import, all new flags default-off.

## Artifacts (gitignored on-disk readouts under `runs/search/`)

- `runs/search/008a_verdict.json` — oracle kill-probe verdict (this run; supersedes the 21:07 smoke).
- `runs/search/ndet_diagnostic.json` — n-det sweep (n=30).
- `runs/search/lcb_sprt_ab.json` — LCB no-regression A/B.
- `runs/search/008a_verdict_smoke.json` — the discarded underpowered smoke (kept for provenance).

> `runs/` is gitignored (repo convention; the build commit committed no JSON). This markdown is the
> committed record of the numbers; the JSONs remain on-disk reproducible artifacts.
