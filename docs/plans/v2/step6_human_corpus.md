# Step 6 — Human-Corpus Use: Opening Scoreboard + Opening-Diversity Training

**Status:** **v5.1 — RATIFIED** (2026-07-05). Council: **unanimous ACCEPT 4/4**
(round 4). Expert: **READY** (round 5) — every round-4 item verified landed; the
READY verdict's pre-freeze pin pack (1 SHOULD-FIX + 6 NITs) is folded into this
v5.1. Full history: v1–v5 per §8/git.
**Depends on:** the human_data harvest completing. The corpus does **not exist yet**;
the pre-corpus lane (§2) runs first. **One item is harvest-blocking (§3.1): the
settlement placement-order contract must land in `record.py` before the harvest runs.**
**Invariants preserved:** 1v1 ruleset, obs schema, action space, checkpoint lineage,
natural-start-only evaluation/promotion, additive TB.

---

## 0. Inputs, scale, population honesty

Corpus: `GameRecord` schema-v2 JSONL. **Ports unrecorded**; the 9 port **slot
positions** are fixed topology (slot adjacency computable); type assignment is not.

**Gating cell, defined as a predicate (not prose):**
`gating cell = {r : r.is_scoreboard_eligible() ∧ r.video_id ∈ EVAL-A}`, one
observation per game, oriented to ThePhantom's seat (he plays in every game — the
"within-ThePhantom-seat" framing is an orientation, not a filter, and does not halve
the sample; v4's 74–240 estimate contained an erroneous ×0.5).
Scale: 204 videos × ~2–5 games × ~60–80% acceptance ≈ 245–800 eligible; × ~60%
EVAL-A ≈ **147–480 gating-cell games in ~60–120 videos**; winner-null/cutoff
attrition is already inside the acceptance range.

**Population statement (declared deviation)** — unchanged from v4 in substance:
tournament-source is the only opponent-controlled bucket (~15 videos; cannot power
any endpoint), so the primary population is ThePhantom's own games with opponent
strength uncontrolled and acknowledged. Binding consequences:

1. GATE-A licenses **"v8 misjudges the openings ThePhantom wins with."** The report
   headlines this framing **and prints ThePhantom's eligible-game WR with a cluster
   CI beside M2** (base-rate disclosure for a winners-heavy channel).
2. **Tournament sign-guard:** binds any GO only if the tournament subset has ≥10
   eligible games; below that, waived and reported. Sign disagreement → INCONCLUSIVE.
3. `opponent_strength.source` joins the permutation strata.
4. **Doc-sync, same-commit rule (council):** the `record.py` guidance docstrings
   (module header AND `is_strong_opponent_scoreboard_eligible`) are rewritten to the
   superseding contract **in the same commit that freezes `opening_scoreboard.py`** —
   the frozen manifest must never point at code whose docstrings teach the old
   headline.

**Terminology:** external opening calibration (OPE caveat; bounded by M0 and the
basin-mass + board-ore robustness rows).

---

## 1. Pre-registration discipline

- **Freeze set (all hashes recorded by the CLI, refuse-to-run on mismatch):** this
  doc, `opening_archetypes.py`, `opening_scoreboard.py`, **`engine_bridge.py` + the
  `engine/board.py` injection surface** (council: bridged-state drift between
  PRE-GATE-0 and GATE-A must break the manifest), the corpus JSONL, the checkpoint,
  the split file, and the PRE-GATE-0 mass table (§2.2).
- **Floor:** gating cell < **100 games** ⇒ GATE-A cannot return NO-GO.
- **Split BY VIDEO**; EVAL-A ~60% / HOLDOUT-B ~40% of videos; seeded RNG; committed.
  Seeds from EVAL-A videos only; GATE-B3 re-measures on HOLDOUT-B only.
- **Cluster statistics:** bootstrap resamples videos; permutation unit =
  **within-stratum video-block** (a video's games within one stratum move as a unit;
  its games in different strata permute independently — mixed-draft videos are the
  norm, so whole-video permutation is not executable and game-i.i.d. is banned).
- **CI flavor pinned:** video-cluster **percentile** bootstrap, B = 10,000,
  one-sided where a sign is pre-registered.
- **Determinism:** per-record-per-layout seed =
  `int.from_bytes(hashlib.sha256(f"{video_id}:{game_index}:{k}".encode()).digest()[:8], "big")`,
  k ∈ 0..7. Python `hash()` is banned (PYTHONHASHSEED-salted).
- **Compute note (council):** the K=8 per-layout `v̂` are computed once per game and
  **cached**; every bootstrap/permutation resamples cached values — no re-bridging
  inside resampling loops.

---

## 2. PRE-CORPUS LANE (run now; ~1 week including the sweep)

### 2.1 The archetype featurizer (frozen)

`src/catan_rl/human_data/opening_archetypes.py` (named to avoid colliding with the
existing `labeling/archetypes.py` — council), committed before PRE-GATE-0 executes.
Spec identical to v4 §2.1 (pair-share semantics, 5 buckets with precedence
ORE_ENGINE ≥0.45 / WOOD_BRICK ≥0.45 / PORT_LED / BALANCED_HIGH ≥26 pips /
BALANCED_LOW; zero-pip ⇒ BALANCED_LOW; 26-pip boundary sanity-checked against
PRE-GATE-0 with any amendment committed pre-corpus; collinearity note: buckets never
enter M2 as covariates).

### 2.2 PRE-GATE-0 measurements

- **v8 opening distribution:** n≥400 natural games. **All downstream "PRE-GATE-0
  mass" quantities — GATE-B0's qualifying buckets AND the basin-mass covariate — are
  computed from the v8-vs-anchor subset only, and the per-bucket mass table is
  written to a committed artifact that both consumers read** (expert: the pooled
  histogram is a second, unfrozen fork — closed). Collapse verdict: ≥70% one-bucket
  mass. PRE-GATE-0 pins the measurement condition and collapse verdict only; it is
  never a GATE-B3 baseline.
- **M0 — in-distribution anchor, identity pinned:** M0 = the M1 statistic (AUC) plus
  the M2 partial Spearman, computed on v8's own eval games vs their outcomes
  (true-port now; port-marginal re-run when the injection API lands — the true-port
  M0 is binding). **M0's permutation strata = `(draft_position)` only**
  (`opponent_strength.source` does not exist for self-play games — fork closed).
- **`setup_entropy_coef`:** wire the setup-phase-only loss term (absent from
  `arguments.py` today; additive, default 0.0), then the calibration sweep. Honest
  budget: readable anchor-WR resolution is ~1–2 days/point ⇒ **pre-registered
  shortened readout**: fixed 2M-step window per candidate, WR-drop cluster CI vs
  coef=0; pick the largest coef whose CI includes 0 drop.

### 2.3 Corpus-free Workstream-C parts (now)

Engine injection API (rule-2 flagged; explicit-RNG port assignment + board
overwrite), v8-state serialization/re-bridge, bridge null test (i) with true-port
out-of-band injection and a port-vertex-settlement fixture.

---

## 3. Workstream C — bridge + env injection (corpus-dependent, ~2d)

### 3.1 Placement-order contract (HARVEST-BLOCKING — lands in `record.py` first)

Colonist grants starting resources from the **second-placed** settlement; the
openings CV reads an order-blind post-setup frame, and order is recoverable only
from the log's setup-event sequence. **Pinned now, before any harvest:**
`PlayerOpening.settlements` (and `.roads`) tuples are in **log placement order**
(`settlements[0]` = first-placed, `settlements[1]` = second-placed/granting),
sourced from the setup-event sequence; the `record.py` docstring is amended in the
same doc-sync commit; the harvest driver must establish order from the log or mark
the game **order-unestablished ⇒ EVAL-excluded, seed-eligible only**. The bridge
grants from `settlements[1]`; **an order-unestablished record used as a seed samples
the grant hypothesis per episode** (never a fixed arbitrary pick — a
Colonist-unreachable start state must not enter training deterministically).
(Fallback if order proves broadly unrecoverable: grant-marginalize — K layouts × 2
grant hypotheses — pre-registered here, used only if >20% of games are
order-unestablished.)

### 3.2 Bridge

As v4: mask-based legality with post-condition assertions (engine `build_*` silently
no-op), spec-009 `bank_draw` + conservation + `rules_invariants`, tracker parity,
K=8 full re-bridges (portList/encoder frozen at construction). Null test (ii):
human-record idempotence + random-agent rollout to terminal.

**Spiral-consistency gate (expert round-5):** engine number placement is
deterministic (`SPIRAL_CHIP_SEQUENCE` given orientation + desert), so a harvested
board's 18 tokens must match one of the **12 spiral walks** for its desert — a free
check the multiset gate cannot make (it is blind to multiset-preserving swaps).
Bridge-time: non-matching boards → `rejection_reason="non_spiral_board"`; the
corpus pass-rate is a line in the PRE-GATE-0/GATE-A report.

---

## 4. Workstream A — the scoreboard (~3d after C)

Per EVAL-A game: K=8 re-bridges → to-move-seat `v̂` per layout → cached.

### Pinned formulas

- **Win-probability and residual (seat-complement pin — expert SF):**
  `p̂ = σ(3.22·v̂ − 1.14)` **if ThePhantom is the to-move seat, else
  `p̂ = 1 − σ(3.22·v̂ − 1.14)`** (probability-space complement; V-negation-then-squash
  is invalid — the squash is calibrated for the to-move POV). `r_i = win_i − p̂_i`,
  `win_i` = ThePhantom won. Port-marginal `p̂` = mean over the 8 layouts.
- **M2 (PRIMARY, α=0.05, predicted sign POSITIVE):** partial Spearman of `r_i` on
  ThePhantom's ORE+WHEAT pip-share, partialling port-slot adjacency; within-stratum
  video-block permutation on strata `(draft_position, opponent_strength.source)`.
- **Luck proxy (units pinned):** realized = cards produced by ThePhantom's two
  opening settlements over `dice_log`; expected = Σ(adjacent pips)/36 per roll ×
  #rolls; proxy = realized − expected. Primary is luck-unadjusted; luck-adjusted =
  robustness row #2; significance disagreement ⇒ INCONCLUSIVE.
- **Robustness whitelist (complete, nothing else computed):** #1 logistic-coefficient
  variant; #2 luck-adjusted; #3 basin-mass competing covariate (from the §2.2
  committed table; ore losing its partial effect to basin-mass ⇒ report attributes
  the result to off-basin miscalibration and flags B-alt); **#4 board-total ore pips
  as competing covariate** (separates "misvalues ore-rich boards" from "misvalues
  his ore openings" — expert pin pack); + the {60,65,70}-percentile / ORE-only
  reporting grid.
- **M1:** **AUC** (pinned door statistic; Spearman reported descriptively) of the
  to-move-seat rank statistic vs outcome; against **M1_pip** (ThePhantom's
  two-settlement pip total − opponent's; desert 0; no ports; same orientation) and
  v8+search@50.
- **M3 (supporting only, never gates anything incl. B3):** stratified quartile WR
  gap; honest cell width quoted: quartile cells are ~18–60 games ⇒ ±12–23pp.

**Power:** clustered effective gating-cell n ≈ 105–420 ⇒ one-sided MDE partial r ≈
**0.14–0.24** at power 0.8. `r = 0.20` is the pre-registered minimum effect of
interest. NO-GO remains structurally demanding; INCONCLUSIVE is first-class and
routes to "extend, not fail" (disclosure unchanged from v4).

### GATE-A (verdicts exhaustive; INCONCLUSIVE = else; doors now disjoint)

- **GO** ⟺ either:
  - **Primary door:** M2 significant (video-block permutation, positive sign) AND
    per-layout **sign-agreement ≥7/8** (pooled M2 carries significance) AND the
    tournament guard passes-or-is-waived **AND the one-sided CI does not exclude
    r ≥ 0.20** (a demonstrated-below-minimum effect is never GO — closes the
    GO/NO-GO overlap at extended-corpus n); **or**
  - **Secondary door** (only on a well-powered M2 null): paired video-cluster
    bootstrap CI on **ΔAUC(M1_v8 − M1_pip)** entirely below 0 (α=0.05), same
    per-layout sign-agreement analog + tournament guard.
- **NO-GO** ⟺ gating cell ≥100 AND M2's one-sided CI excludes r=0.20 **AND the
  luck-adjusted row is also a powered null** (guard-beats-NO-GO precedence — a
  luck-masked signal must not be read as absence) AND the secondary door fails.
- **INCONCLUSIVE** ⟺ anything else.

Deliverables/spike as v4.

---

## 5. Workstream B — training response (gated on GATE-A GO + PRE-GATE-0)

Mechanism, B-alt (EVAL-A-only logistic fit), B1 seed loader (incl. the
two-lowest-buckets fallback and the first-class "corpus adds no unexplored mass"
finding; **seed draws are uniform-over-qualifying-buckets, never at empirical corpus
frequency** — the D6-dedup removes almost nothing on all-distinct boards, and
frequency-weighting would concentrate `seed_prob` in ThePhantom's modal archetype),
B2 wiring (`reset_source`; `anchor_window_stats()` unit test; harness assertion; TB
incl. per-archetype `seeded/return`) — all as v4, plus:

**B-launch precondition (expert round-5 — moved from B3 time):** immediately after
GATE-A GO, compute frozen v8's HOLDOUT-B M2 effect (contaminates nothing — v8 is
frozen; the result is cached and reused at B3). If its cluster-bootstrap CI does not
exclude 0 in the predicted sign, B3(2) is **VOID before any arm launches**: the GO
is flagged failed-replication and routes to INCONCLUSIVE/extend — a strength-only
launch (B3(1)/(3)/(4) gates) remains a conscious option, but the calibration
endpoint of a 1–2-week run is never spent on a license that didn't replicate.

**Squash-refit pin (expert round-5 SHOULD-FIX):** σ(3.22·v̂−1.14) was fit to v8's
value distribution; after weeks of PPO with value normalization, v8′'s scale drifts,
and the seat-complement mixes σ/1−σ branches, so a stale squash changes ranks.
**Any non-v8 policy entering the calibration metrics gets its own squash refit by
the spec-003 procedure on its own natural eval games (disjoint from HOLDOUT-B)**,
and B3(2) PASS additionally requires v8′'s HOLDOUT-B M1 AUC ≥ v8's − 0.02 (the
attenuation guard — a noisier p̂ must not shrink |effect| into a spurious PASS).

### The in-run gate statistic (expert round-4 BLOCKER — the moving-anchor confound)

Each arm's promotion ratchet fires at endogenous, arm-specific times; a promotion
resets that arm's anchor WR toward the bar (~0.63). Gating on arm-local anchor WR
therefore **kills the seeded arm for succeeding** (it promotes first → its WR resets
→ "gap > 3pp" → halve/kill). Fix, binding:

- **GATE-B1's statistic and B3(3)'s `openings/entropy_natural` are computed from a
  fixed shared reference:** periodic natural-start `evaluate_policy_vs_policy`
  matches **vs the frozen v8 checkpoint**, identical eval seeds for both arms,
  scheduled by step count — fully independent of each arm's ratchet state.
  Promotions stay enabled; they simply never define a gate statistic.
- **Noise-calibrated trip:** each rolling 2M-step window must contain ≥300 fixed-
  reference games per arm; GATE-B1 trips only if the seeded−control gap exceeds 3pp
  AND clears a two-proportion test (α=0.05), or persists across 2 consecutive
  non-overlapping windows. First trip → halve `seed_prob`; second → kill.
  **Operative-branch note (expert round-5):** at n=300/arm, SE(gap) ≈ 4.1pp, so the
  significance branch binds only above ~8pp — **the 2-window persistence branch is
  the operative protection at 3pp** (per-pair false-trip ≈ 0.05; the shared eval
  seeds make the unpaired test conservative). Optional power upgrade if needed:
  paired shared-seed McNemar.

### B4 gates (deltas from v4)

- **GATE-B3(2) — replication precondition (expert SF):** scored **only if** frozen
  v8's HOLDOUT-B M2 effect has a cluster-bootstrap CI excluding 0 in the predicted
  sign. Otherwise B3(2) = **VOID**, the GATE-A GO is flagged **failed-replication**,
  and the outcome routes to INCONCLUSIVE/extend — never to the value-side-gap pivot.
  Construct unchanged (paired Δ = |effect_v8| − |effect_v8′|, CI LB > 0.5·|effect_v8|,
  same-family degrade path), with the **single shared deterministic resample stream**
  for both measurements pinned (council: two loops would decorrelate and inflate
  CI(Δ)).
- **GATE-B3(4) — reworked (expert NIT):** M3 dropped from B3 entirely (it is
  supporting-only by §4). For M1/M2: paired video-cluster bootstrap CI on the deltas
  (v8′ vs v8, HOLDOUT-B); fail only if the CI **excludes** the tolerance
  (0.02 AUC / 0.03 partial-Spearman) in the regression direction — a point-estimate
  check at eff-n 35–140 would fail an unchanged champion ~20% of the time.
- B3(1) (Wilson LB ≥0.55), B3(3) (fixed-reference entropy vs contemporaneous
  control), GATE-B0/B1/B2, control-arm branch, kill disambiguator: as v4/above.

## 6. Non-goals — inline, unchanged from v4 (no BC/piKL; no move-agreement metrics;
no engine/obs/action changes; no 4-player; no harvest-v1 port extraction).

## 7. Sequencing

```
NOW (pre-corpus, ~1wk):  opening_archetypes.py → PRE-GATE-0 (+ committed mass table)
                         + M0 (true-port) + setup-entropy wiring + shortened sweep
                         + injection API + re-bridge + null test (i)
                         + record.py placement-order contract  ◄── HARVEST-BLOCKING
corpus lands ──► C (~2d) → null test (ii)
  ├─► A: spike → scoreboard (~3d) → GATE-A (exhaustive 3-way, disjoint doors)
  │     GO → B-alt + seeded A/B (fixed-reference gates; ~1-2wk/arm)
  │     INCONCLUSIVE → extend / decide on §2+M0 grounds
  │     NO-GO → B-alt iff collapsed; standing eval
  └─► BY-VIDEO split committed BEFORE any metric
```

## 8. Review traceability

v1→v4: git history (every finding → fix verified by both tracks in the following
round). **v4→v5 (expert round 4):** fixed-shared-reference GATE-B1/B3(3) +
noise-calibrated trip (BLOCKER); probability-space seat complement pin (SF);
placement-order contract, harvest-blocking (SF); B3(2) replication precondition +
VOID routing (SF); PRE-GATE-0 mass provenance + committed table (SF); within-stratum
video-block permutation; GO/NO-GO disjointness clause; luck-guard/NO-GO precedence;
gating-cell predicate + corrected arithmetic (147–480, MDE 0.14–0.24); ΔAUC pin +
CI flavor pin; B3(4) rework, M3 degated; luck units, M0 identity, M3 cell width,
base-rate disclosure, board-ore covariate row, per-layout seed formula, sweep
re-budget. **v4→v5 (council round 4, all non-blocking):** same-commit doc-sync;
bridge hashes in the freeze set; cached layouts; shared resample stream;
`opening_archetypes.py` rename. **Council verdict: 4/4 ACCEPT (round 4).**
**v5→v5.1 (expert round 5, READY — pre-freeze pin pack):** squash-refit pin +
attenuation guard for B3 metrics on non-v8 policies (SHOULD-FIX); spiral-consistency
bridge gate; M0 strata = draft_position only; B3(2) replication precondition moved
to a B-launch precondition with cached reuse; uniform-over-qualifying-buckets seed
draw; per-episode grant-hypothesis sampling for order-unestablished seeds;
GATE-B1 operative-branch (persistence) note. **Expert verdict: READY (round 5).**

## 9. Risks — as v4, plus:

| Risk | Mitigation |
|---|---|
| Seeded arm killed because seeding worked | fixed-frozen-v8 reference for all in-run gate statistics; promotions never define a gate |
| Noise-tripped GATE-B1 | min-n per window + two-proportion test or 2-window persistence |
| Wrong-settlement resource grant | placement-order contract in record.py BEFORE harvest; order-unestablished ⇒ EVAL-excluded |
| GATE-A effect fails to replicate on holdout | B3(2) VOID + failed-replication routing (never misread as a value-side gap) |
| Seat-complement formula fork | probability-space complement pinned in doc + frozen code |
