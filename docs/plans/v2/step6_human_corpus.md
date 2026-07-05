# Step 6 — Human-Corpus Use: Opening Scoreboard + Opening-Diversity Training

**Status:** DRAFT v2 (2026-07-05) — revision after expert review (NOT-READY, 6 BLOCKERs)
and a 4-member council review (unanimous NOT-READY). All v1 blockers are addressed
in-line; the v1→v2 changelog is §8.
**Depends on:** the human_data harvest completing (`docs/plans/human_data_pipeline.md`).
The corpus does **not exist yet**; PRE-GATE-0 (§2) runs before it and costs nothing.
**Scope:** what we DO with the corpus: **A — external opening-calibration scoreboard**
(measurement), **B — opening-diversity training response** (gated on A), joined by
**C — the engine bridge + env injection path**. Hard invariants preserved throughout:
1v1 ruleset, obs schema, action space, checkpoint lineage, natural-start-only
evaluation, additive TB.

---

## 0. Inputs, scale, and two structural facts

Corpus: `GameRecord` schema-v2 JSONL — `{hexes (19× resource+number), openings
(2 settlements + 2 roads per player), draft_order, winner|null, dice_log,
opponent_strength, episode_source (parse provenance), passed_crosscheck,
rejection_reason, provenance}`. **Ports are NOT in the record** (schema v1 decision,
`record.py` "ports OMITTED") — this drives §3's port-marginalization design.

Expected scale: 204 high-rank videos × ~2–5 games × ~60–80% acceptance ≈ **~350–800
scoreboard-eligible games**, plus seed-only records from `unknown`-tier videos
(**included in the seed pool at launch** — seeds need diversity, not verified
strength; resolves the v1 §0/§B1 inconsistency).

Two structural facts shape the statistics:

1. **One-human corpus.** ThePhantom occupies a seat in ~every game, and per
   `record.py` the ranked (`rank_badge`) games are **opponent-uncontrolled** (Colonist
   matched by *his* rating). The only opponent-controlled bucket is
   tournament-source (~15 videos). Consequently every gating statistic below is
   **stratum-pinned** (within-ThePhantom-seat, or tournament-only robustness) — a
   pooled "vs humans" number is never computed as evidence (contract:
   `is_strong_opponent_scoreboard_eligible`, pipeline brief §5.4).
2. **Winner-null games** (resign/cutoff) are seed-eligible, never outcome data.

**Terminology (v2):** the measured quantity is **external opening calibration** — how
well v8's post-setup evaluation predicts real elite-game outcomes. It is NOT claimed
to isolate "the value head is wrong" from off-policy/dynamics divergence (the OPE
caveat): a failed calibration is a deployment-relevant defect whatever its
decomposition, and the in-distribution anchor (§3.M0) bounds the OOD part.

---

## 1. Pre-registration discipline (applies to every gate)

- All thresholds, strata, splits, and archetype definitions in this document are
  **frozen by committing this file before any scoreboard number is computed**; the
  scoreboard CLI refuses to run unless the doc's git hash + corpus JSONL hash +
  checkpoint hash are recorded into its output.
- **Holdout split, created before any metric:** scoreboard-eligible games are split
  **by game** into EVAL-A (~60%) and HOLDOUT-B (~40%), seeded RNG, committed as a
  file. GATE-A runs on EVAL-A. GATE-B3's re-measurement runs on **HOLDOUT-B only**,
  and **seeds are drawn from EVAL-A games only** — the champion is never re-scored on
  states it trained from (kills the v1 train-on-test blocker and blunts
  regression-to-the-mean).
- Unit of analysis is the **game** (game-level bootstrap for every CI; per-seat rows
  never pooled as independent).

---

## 2. PRE-GATE-0 — test the premise before building anything (zero corpus, run NOW)

The diversity-collapse premise must itself be measured (council: the plateau already
had a competing explanation — the promotion bar). From **existing** eval machinery:

- Play n≥400 natural v8-vs-v8 (and v8-vs-anchor) games; bucket the resulting openings
  with §3's archetype featurizer; report **v8's opening entropy and archetype support**
  (deterministic given seeds; CPU).
- Also compute `openings/setup_head_entropy` (the policy's setup-head entropy at the
  placement decisions) — the leading indicator B would monitor anyway.

Outcomes: (i) *collapsed* (≥70% of mass in one archetype bucket) → the diversity
premise stands, B stays as designed; (ii) *not collapsed* → B's rationale shifts from
"restore diversity" to "recalibrate opening evaluation"; the seeded arm is retained
only if GATE-A shows a calibration failure, and the diversity criterion GATE-B3(3)
is re-pointed at *calibration* (B3(2)) as primary. Either way the measurement is a
baseline number B3(3) needs.

---

## 3. Workstream C — engine bridge + env injection path (~2–3d, build first)

`src/catan_rl/human_data/engine_bridge.py` + a seeded-reset branch in `CatanEnv`.

**Bridge (`build_post_setup_game(record, port_layout_rng) -> catanGame | BridgeReject`):**

- Board from `hexes` (resource-literal → engine-order conversion at this boundary,
  CLAUDE.md rule 6); robber on the desert.
- **Ports:** the record has none and `catanBoard` randomizes them
  (`updatePorts`/permutation). The bridge takes an explicit port-layout RNG; consumers
  choose the policy (§4 marginalizes; §5 randomizes per episode). Port-dependent
  archetypes are **not computable** and do not exist in this plan.
- Placements replayed in `draft_order` **with explicit legality checks against the
  env's own action masks / geometry** (v1 wrongly claimed the engine build APIs
  re-check the distance rule — they don't (`player.py` checks resources/pieces only);
  `is_seed_eligible` likewise doesn't). Distance rule, road incidence, id ranges are
  asserted here; violation → `BridgeReject(reason)`, counted, never repaired.
- Starting resources granted through the **spec-009 bank** (`bank_draw`), then
  `assert_conservation` (`resourceBank[R] + Σ hands[R] == 19` ∀R) +
  `rules_invariants` run on every bridged state.
- Hand tracker / broadcast subscription order matches a natural game (council-caught
  bug class): bridged tracker state must equal `player.resources` — asserted.

**Env injection (`CatanEnv.reset(seed_game=...)` branch):** `reset()` normally builds
its own game and runs setup interactively; the injection path accepts a bridged game,
skips `initial_placement_phase`, rebuilds obs-encoder index maps for the injected
board, seats the snapshot opponent, and sets the to-move player. This is a real
reset/vec_env fork (v1 under-scoped it) — it is C's second deliverable because both
A and B consume it.

**The bridge null test (replaces v1's GUI spot-check):** serialize N=50 post-setup
states from ordinary v8 self-play into `GameRecord`s, bridge them back, and assert
`v_hat` parity (|Δ|<ε) with the value computed in the live env, plus state-hash
equality of engine internals. One headless test catching the whole bridge-artifact
class (grants, robber, tracker order, index maps, bank). No `gui/` import anywhere.

---

## 4. Workstream A — the external opening-calibration scoreboard (~3d, CPU)

For each EVAL-A game: bridge → post-setup state → evaluate **the to-move seat's**
`v_hat` (the other seat is the zero-sum complement — v1's both-seats evaluation
fabricated an OOD obs for the never-to-move seat) → record
`(v_hat, outcome, seat_is_thephantom, draft_position, archetype features, luck proxy)`.

**Port marginalization (expert BLOCKER):** `v_hat` is the mean over **K=8 sampled
port layouts** per record (obs feeds a port one-hot; a single random layout would
systematically garble port-anchored openings, biasing M2 in H1's direction). The
estimand is declared **port-marginal opening value**; the per-record spread across
layouts is reported, and any effect smaller than the median port spread is
pre-registered as *inconclusive*.

### The metric family (all stratum-pinned, game-level bootstrap)

- **M0 — in-distribution anchor (new in v2):** the same estimator (port-marginal
  `v_hat` at post-setup, to-move seat) computed on **v8's own self-play eval games**
  (from PRE-GATE-0's runs) vs their outcomes. This is the honest reference for "how
  predictive can a post-setup value be under this dice engine" — replacing v1's
  uncomputable "midgame external" clause and anchoring the OPE caveat.
- **M1 — external discrimination:** game-level AUC/Spearman of
  `v_hat(P1) − v_hat(P2)` vs winner on EVAL-A, **within-ThePhantom-seat stratum**
  (his games, his seat's sign convention; opponent-strength as covariate). Reported
  against two baselines on the same games: the **pip-count heuristic** and
  **v8+search@50**. (Expected CI: Spearman SE ≈ 1/√(n−3) ≈ 0.045–0.06 at n≈250–500.)
- **M2 — the ore-engine hypothesis (PRIMARY endpoint, α = 0.05):** continuous form —
  partial Spearman (or logistic coefficient) of the calibration residual
  (outcome − rank-normalized `v_hat`) on **ORE+WHEAT pip-share** of the opening,
  within an exactly stratified permutation test on
  `(seat_is_thephantom, draft_position)`. The pip-share is a *covariate*, not a
  binary bucket; for reporting, the binary split is fixed **now** at the corpus's
  outcome-blind 65th percentile of ORE+WHEAT pip-share, with a {60,65,70}th-percentile
  sensitivity grid. (Note: ORE+WHEAT deliberately widens the playtest's ORE-specific
  hypothesis to the ore→city→dev engine; an ORE-only split is in the sensitivity
  grid — council request.)
- **M3 — inversion check (supporting):** bottom-quartile minus top-quartile (by
  `v_hat`, quartiles stratified by draft position) realized-WR gap, within-stratum,
  game-level bootstrap CI. Fires only on a CI excluding 0 in the *inverted*
  direction — v1's "≳0.5" point rule was ~coin-flip under the null.

**Confound controls:** single-human (all gating metrics within-ThePhantom-seat;
tournament-only subset reported as directional robustness), draft-position
stratification everywhere, and the **luck proxy** defined exactly as: Σ over
`dice_log` of pip-production matched to the player's two opening settlements only
(ignores robber/bank/later builds — the approximation is named in the report).
The pipeline's per-archetype acceptance table is reproduced next to M2 (CV rejection
bias visibility).

### GATE-A (decision rule, no OR-soup)

- **GO** ⟺ **M2 is significant in the predicted direction** (primary, α=0.05,
  stratified permutation) **AND** the effect exceeds the port-marginalization
  inconclusive band. M1-vs-baselines and M3 are reported as supporting context;
  additionally, **M1 below the pip-count baseline** on the same games is an
  independent GO trigger (a trivial heuristic out-ranking the champion is
  opening-specific failure that irreducible dice variance cannot explain).
- **NO-GO** → the external evidence does not support the blind spot; Workstream B's
  seeded arm is shelved; the corpus still feeds the **setup-prior calibration** use
  (§5.B-alt) and the scoreboard remains a standing eval for future champions.
- Deliverables: `scripts/opening_scoreboard.py`, `archetypes.py`, report at
  `data/human/scoreboard_report.md` with every number CI'd and stratum-labeled.
- **Pragmatist spike (day 1 of A):** before the full apparatus — bridge 20 tournament
  games, dump raw `v_hat` vs outcomes, eyeball. If the bridge or values look broken,
  fix before building the statistics layer.

---

## 5. Workstream B — training response (gated on GATE-A + PRE-GATE-0)

### The mechanism, stated honestly (v1 gap, expert + council convergent)

Seeded episodes begin post-setup ⇒ **the setup heads receive zero policy gradient
from them**. The pathways by which seeds can help are: (1) **value recalibration on
human-opening states** → better advantages when natural exploration visits similar
placements; (2) **the search pathway** — deployed play uses PUCT search whose leaf
evaluations consume the value head, so better opening values improve *searched*
opening play even with an unchanged prior. Pathway (1) is rate-limited by the setup
heads' residual entropy in a converged v8 warm start; therefore the seeded arm
**co-deploys a setup-phase-only entropy bonus** (flagged config delta,
`setup_entropy_coef`, additive & default-off) so natural episodes actually explore
placements the recalibrated value can now rank. This is consistent with spec 007
(which tested rank-*retargeting* on v8's own state distribution; seeds change the
state distribution itself) — stated here so the plan doesn't appear to contradict a
closed spec.

### B-alt — the setup-prior lever (promoted from v1's fallback; Skeptic/Pragmatist)

Independent of seeding, the corpus calibrates `setup_strength_roadmap.md` Phase A's
analytic placement prior (its resource weights — including the ore weight — fit
against elite-human openings instead of heuristic guesses). Cheap, CPU-only,
policy-shape-free. **Decision point after GATE-A:** GO → run **both** B-alt and the
seeded arm (they compose; B-alt is days-cheap); NO-GO → B-alt only if PRE-GATE-0
showed collapse.

### B1. Seed loader

`src/catan_rl/selfplay/human_seeds.py`: `is_seed_eligible()` records from **EVAL-A
games only** (holdout discipline, §1) + `unknown`-tier records; every seed passes the
bridge's mask-based legality gate; **D6-canonicalized** before dedup with key
`(board, both openings)`; the learner's inherited seat is **randomized per episode**;
ports randomized per episode (stated B limitation). **Pool diversity gate at load
(GATE-B0):** archetype histogram + entropy logged; require ≥3 archetype buckets with
≥10% mass each, else stratified sampling over buckets — an entropy-poor pool
(one more basin) fails loudly instead of shipping.

### B2. Wiring (additive, default-off)

- `seed_prob` in `arguments.py` (default 0.0), runtime tag named **`reset_source`**
  (NOT `episode_source` — that name is the record's parse-provenance field; the
  collision was a v1 NIT).
- **Natural-only discipline, named call sites:** (i) `League.record_outcome()` is
  called **only for natural-start games** — at `seed_prob=0.25` a quarter of anchor
  games are seeded, and the EMA+streak promotion ratchet (the *actual* wired
  mechanism, `league.py` — v1 misdescribed it as an n≥600 Wilson gate) must never see
  them; unit test: a seeded episode's outcome leaves `opponent_win_rate(anchor)`
  untouched. (ii) the eval harness asserts no seed pool is attached
  (`EvalHarness._evaluate_matchup`). GATE-B2 below is the *offline* n≥600 Wilson
  eval, distinct from the in-run ratchet — both stay natural-only.
- TB (additive): `seeded/frac`, `seeded/return`, `seeds/legality_reject_rate`,
  `seeds/pool_entropy`, `openings/setup_head_entropy`,
  `openings/entropy_natural` (defined once: archetype-bucketed entropy of openings in
  the periodic natural-start anchor-eval games).

### B3. Run design

- Warm-start v8, the proven anchor + promotion-bar machinery, run
  `selfplay_v9_humanseed`, `seed_prob=0.25`, `setup_entropy_coef` per §5-mechanism.
- **Mandatory contemporaneous A/B** (v1 made this budget-contingent — both reviews
  called it a blocker): the control arm (same warm start, same code, `seed_prob=0`,
  same `setup_entropy_coef` so the seeds are the only delta) runs interleaved on the
  same machine. GATE-B1's tripwire is defined against *this* control, never a
  historical trajectory.

### B4. Gates

- **GATE-B0 (pre-launch):** loader/wiring tests green; pool diversity gate passed;
  1k-episode smoke (0 crashes, legality-reject <5%, obs schema byte-identical,
  bridged-tracker == player.resources assertion, `reset_source` accounting correct).
- **GATE-B1 (in-run):** seeded arm's natural-start anchor WR must stay within 3pp of
  the contemporaneous control for rolling 2M-step windows; first trip → halve
  `seed_prob`; second → kill.
- **GATE-B2 (offline promotion eval):** champion vs v8, natural-start, n≥600
  symmetrised Wilson — unchanged from project convention.
- **GATE-B3 (success claim, all on HOLDOUT-B):**
  1. head-to-head ≥0.55 vs v8 (n=600, natural-start), AND
  2. **calibration improves on HOLDOUT-B by an equivalence-style criterion:** the M2
     residual effect shrinks by a pre-registered margin (point estimate ≤ half the
     GATE-A effect, TOST-style) — *not* "CI includes 0", which rewards noisier
     measurement (v1 blocker), AND
  3. `openings/entropy_natural` exceeds the PRE-GATE-0 baseline by a pre-registered
     margin (+0.3 nats or +1 archetype bucket at ≥10% mass) — skipped as a criterion
     if PRE-GATE-0 already showed no collapse (§2), AND
  4. (deployed-config check) M1–M3 under **v8'+search** do not regress vs v8+search.
- **Kill/pivot:** GATE-B3 fails twice → seeds insufficient; B-alt (if not already
  run) and/or a setup-phase-only fine-tune become the lever. **BC/piKL stay off the
  table** at this corpus scale.

Effort: loader+wiring ≈ 2–3d (the env fork is real work, v1 underestimated); run =
usual multi-day M1 training × 2 arms (A/B), review-and-resolve loop alongside.

---

## 6. Non-goals (unchanged, restated so they stay decided)

No BC/piKL on this corpus; no move-agreement metrics anywhere; no engine-rule,
obs-schema, or action-space changes; no 4-player anything; no port extraction in v1
of the harvest (if a future harvest adds ports to the schema, §4's marginalization
becomes exact conditioning — flagged as the §5.9 pipeline change it would be).

## 7. Sequencing

```
NOW (no corpus):     PRE-GATE-0 (v8 opening entropy + setup-head entropy)
corpus lands ──►     C: bridge + env injection (~2-3d) ──► bridge null test
                     ├─► A: day-1 spike ──► full scoreboard (~3d) ──► GATE-A
                     │        GO ──► B-alt + seeded A/B run (B0→B1→B2→B3)
                     │        NO-GO ─► B-alt iff PRE-GATE-0 collapsed; scoreboard
                     │                 becomes a standing eval
                     └─► holdout split committed BEFORE any metric (§1)
```

Spec-driven execution: A and B become Spec Kit features
(`specs/010-opening-scoreboard`, `specs/011-human-seeds`) citing this document.

## 8. v1 → v2 changelog (review traceability)

- Expert B1 (uncomputable midgame clause) → deleted; replaced by M0 in-distribution
  anchor + pip-baseline-relative M1 trigger.
- Expert B2 (ports) → port-marginalized estimand, K=8 layouts, inconclusive band,
  port-led archetype removed, per-episode port randomization in B, risk row.
- Expert B3 (unanchored stats) → M2 primary with α; M3 → stratified inversion CI;
  no OR-of-three; CI widths stated.
- Expert B4 (M2 stratum) → within-ThePhantom-seat pinning + exact stratified
  permutation + outcome-blind threshold committed now + sensitivity grid.
- Expert B5 (train-on-test B3) → §1 holdout split; seeds from EVAL-A only; B3(2) on
  HOLDOUT-B with TOST-style margin.
- Expert B6 (bridge legality myth) → mask-based legality in the bridge; engine APIs
  documented as not checking placement rules.
- Expert SFs → league EMA call-site + natural-only unit test; mechanism statement +
  setup-entropy co-deploy + search-pathway B3(4); to-move-seat-only v_hat + bridge
  null test; game-level unit of analysis; luck proxy defined; pool diversity gate +
  seat randomization; C rescoped with bank conservation; NITs (reset_source rename,
  unknown-tier resolution, dedup key, entropy_natural definition, assertion call
  sites, acceptance table beside M2).
- Council (Architect) → contract-legal primary statistic; mandatory A/B; env-fork
  scoping; bank invariant; co-deployed setup lever; hash pre-commitment; D6 dedup.
- Council (Skeptic/Pragmatist) → B-alt promoted; day-1 spike; ORE-only sensitivity;
  TOST for B3(2); corpus-doesn't-exist status note.
- Council (Researcher) → PRE-GATE-0; claim renamed to external calibration with the
  OPE caveat + M0 anchor; M2 primary-endpoint α discipline; pool-entropy gate.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Ports absent → biased v_hat | port-marginal estimand (K=8) + inconclusive band (§4) |
| Single-human confound | every gating stat within-ThePhantom-seat; tournament robustness split |
| GATE-A false GO | single primary endpoint (M2, α=0.05, stratified permutation) + port band |
| GATE-B3 self-certification | holdout split §1; seeds from EVAL-A only; TOST margin |
| Seeds interfere with base play | mandatory contemporaneous A/B + GATE-B1 tripwire |
| Eval/promotion contamination | named call sites + unit tests (League.record_outcome, harness) |
| Bridge artifacts | bridge null test (v8-state round-trip, ε-parity) + bank conservation + rules_invariants |
| Setup heads can't move (mechanism) | setup-entropy co-deploy + `setup_head_entropy` leading indicator + search-pathway B3(4) |
| Premise wrong (no collapse) | PRE-GATE-0 measures it for free before anything is built |
