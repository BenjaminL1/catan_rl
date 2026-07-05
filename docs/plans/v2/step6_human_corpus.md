# Step 6 — Human-Corpus Use: Opening Scoreboard + Opening-Diversity Training

**Status:** DRAFT v3 (2026-07-05) — third revision. v1: expert NOT-READY (6 BLOCKERs) +
council unanimous NOT-READY. v2: all v1 blockers verified structurally resolved by both
tracks; expert NOT-READY (6 residual/new, two introduced by v2's own fixes) + council
3-REJECT/1-ACCEPT. §8 carries the full v1→v2→v3 traceability.
**Depends on:** the human_data harvest completing. The corpus does **not exist yet**;
§2 (PRE-GATE-0) is corpus-free and runs first.
**Scope:** **A — external opening-calibration scoreboard** (measurement),
**B — opening-diversity training response** (gated on A), **C — engine bridge + env
injection** (shared foundation). Invariants preserved: 1v1 ruleset, obs schema, action
space, checkpoint lineage, natural-start-only evaluation, additive TB.

---

## 0. Inputs, scale, and the population honesty statement

Corpus: `GameRecord` schema-v2 JSONL (`hexes`, `openings`, `draft_order`, `winner|null`,
`dice_log`, `opponent_strength{tier,source}`, `passed_crosscheck`, provenance). **Ports
are NOT recorded** (schema v1) — §3/§4 handle this explicitly. Port **slot positions**
are fixed board geometry (9 slots, committed topology); only their **type assignment**
is unrecorded — so *port-slot adjacency of a settlement IS computable* from the record
and is used as a covariate (§4).

Corrected scale arithmetic (v2 overstated the low end): 204 high videos × ~2–5
games × ~60–80% acceptance ≈ **~245–800 scoreboard-eligible games**; after the
EVAL-A/HOLDOUT-B split and stratum pinning, the **gating cell is ~74–240 games** —
the power section (§4) is written against this honest number, not the headline.

**Population statement (resolves the contract-inversion finding, both tracks, both
rounds).** `record.py`'s guidance designates tournament-source games as the only
opponent-controlled bucket, with ranked (`rank_badge`) games opponent-uncontrolled
(matched by ThePhantom's rating; opponent strength unverified, and the
opponent-strength covariate is near-constant, carrying almost no adversary
information). The tournament bucket is ~15 videos — **incapable of powering any
endpoint** (per-cell n≈7 for a 2-way split). This plan therefore makes a **declared,
power-justified deviation**: the primary population is the **within-ThePhantom-seat
stratum** (his skill held constant across his own games; opponent strength
uncontrolled and acknowledged). Consequences, all binding:

1. What GATE-A licenses is the claim **"v8 misjudges the openings ThePhantom wins
   with"** — NOT "elite-vs-elite opening truth." The report states this framing.
2. **Tournament sign-agreement guard:** GO additionally requires the tournament-only
   subset's M2 point estimate to agree in sign (pre-registered: opposite sign →
   INCONCLUSIVE, not GO). Caveat carried: tournament labels are title-regex,
   frame-unverified.
3. `opponent_strength.source` joins M2's exact strata.
4. Workstream A includes updating `record.py`'s guidance docstring to name this
   power-justified primary in the same change (doc-sync rule).

Structural facts: one-human corpus (every gating statistic stratum-pinned; pooled
"vs humans" numbers never computed as evidence); winner-null games are seed-only.

**Terminology:** the measured quantity is **external opening calibration**; no claim
to isolate value-head error from off-policy/dynamics divergence (OPE caveat), bounded
by the in-distribution anchor M0.

---

## 1. Pre-registration discipline

- Thresholds, strata, splits, **and the archetype featurizer** (§2.1 — spec in this
  document, implementation committed before PRE-GATE-0 runs) are frozen by commit;
  the scoreboard CLI records and refuses to run without: this doc's git hash, the
  **`archetypes.py` git hash**, corpus JSONL hash, checkpoint hash.
- **Corpus-size floor (new in v3):** if scoreboard-eligible games < **200**, GATE-A
  cannot return NO-GO — only GO or INCONCLUSIVE(extend) (§4) — an underpowered null
  is never read as evidence of absence.
- **Holdout split before any metric:** by game, EVAL-A (~60%) / HOLDOUT-B (~40%),
  seeded RNG, committed. GATE-A runs on EVAL-A; seeds come from EVAL-A only;
  GATE-B3 re-measures on HOLDOUT-B only.
- Unit of analysis = the **game** (game-level bootstrap everywhere; per-seat rows
  never pooled as independent).
- **Determinism pins:** the K port layouts per record are seeded by
  `hash(video_id:game_index)` (bit-stable reruns under the hash manifest).

---

## 2. PRE-GATE-0 — corpus-free measurements (run NOW; ~1 day + short runs)

### 2.1 The archetype featurizer (frozen here — it is load-bearing in four gates)

`src/catan_rl/human_data/archetypes.py`, committed **before** PRE-GATE-0 executes.
Spec (fixed; never changes after PRE-GATE-0's numbers exist):

- **Feature basis** per (seat, opening): the 5-vector of **pip-share** by resource
  over the seat's two settlements (pips of adjacent hexes, desert=0, normalized to
  sum 1), plus total pips, plus **port-slot adjacency** (either settlement on one of
  the 9 fixed port-slot vertex pairs — boolean).
- **Buckets (fixed count = 5, numeric boundaries):**
  `ORE_ENGINE` (ORE+WHEAT share ≥ 0.45), `WOOD_BRICK` (WOOD+BRICK share ≥ 0.45),
  `PORT_LED` (port-adjacent AND no share ≥ 0.45), `BALANCED_HIGH` (no share ≥ 0.45,
  total pips ≥ 26), `BALANCED_LOW` (otherwise). Deterministic precedence in that
  order. (PORT_LED uses slot adjacency only — no port *types*, which are unrecorded.)
- Entropy metrics are over these 5 buckets; "+1 bucket at ≥10% mass" is well-defined
  because the count is fixed.

### 2.2 Measurements

- **v8 opening distribution:** n≥400 natural v8-vs-v8 + v8-vs-anchor games →
  archetype histogram, entropy, `openings/setup_head_entropy`. Collapse verdict
  pre-registered: ≥70% mass in one bucket = collapsed. The **v8-vs-anchor** histogram
  is the pinned baseline for GATE-B3(3) (it matches what the v9 run's periodic anchor
  evals produce).
- **M0 — in-distribution anchor (moved pre-corpus, Skeptic):** port-marginal-style
  `v_hat` at post-setup (to-move seat) on v8's own eval games vs their outcomes —
  the honest ceiling for "how predictive can a post-setup value be under StackedDice."
- **`setup_entropy_coef` pre-calibration (Researcher):** a short control-only run
  (no seeds) sweeping 2–3 values; pick the largest that does not degrade anchor WR —
  fixed before the multi-week commit so an uncalibrated HP cannot sink both arms.

Branch: not-collapsed → B's rationale becomes calibration-repair; B3(3) demoted from
criterion to report (§5.B4).

---

## 3. Workstream C — engine bridge + env injection (~2–3d)

`engine_bridge.py` + a `CatanEnv.reset(seed_game=...)` branch, **plus a small
additive engine surface that does not exist yet** (flagged per CLAUDE.md rule 2;
`engine/board.py` joins C's file list): a board/port **injection API** —
`catanBoard.__init__` randomizes resources/spiral numbers and `updatePorts` draws
from *global* `np.random` with no injectable RNG, and human boards need arbitrary
number placement the spiral generator cannot produce. Injection = post-construction
overwrite of `hexTileDict`/`resourcesList`/robber + deterministic port assignment
from an explicit RNG. Additive, default-unused by training.

Bridge (`build_post_setup_game(record, port_rng) -> catanGame | BridgeReject`):

- Resource-literal → engine-order conversion at this boundary (rule 6).
- **Placement legality via the env's action masks / geometry, asserted explicitly**
  (engine `build_*` APIs check resources/pieces only — verified; they also
  **silently no-op on failure**, so the bridge asserts post-conditions: piece
  counts, vertex/edge occupancy — never relies on exceptions).
- Starting resources through the spec-009 bank (`bank_draw`) then
  `assert_conservation` + `rules_invariants`; hand-tracker state == `player.resources`
  asserted (council bug class).
- **Per-layout ordering (expert):** `player.portList` is captured inside
  `build_settlement` and the encoder's port one-hot is frozen at construction — so
  each of the K port layouts is a **full re-bridge** (set ports → replay placements →
  grant → build encoder), 8 re-bridged games per record, never 8 obs tweaks.

**Bridge null test (two parts, expert-corrected):** (i) serialize N=50 v8 self-play
post-setup states **with their true port layouts carried out-of-band** and injected
via the port parameter (RNG bypass) → assert `v_hat` ε-parity vs the live env +
engine state-hash equality; the fixture set must include a port-vertex opening
settlement (catches a `portList` weld). K-marginalization is explicitly NOT
exercised by this test — it has its own determinism test. (ii) human-side: bridge N
corpus records → re-serialize → re-bridge → state-hash idempotence, then a short
random-agent rollout to terminal with invariants green.

---

## 4. Workstream A — the scoreboard (~3d after C; CPU)

Per EVAL-A game: K=8 full re-bridges (per-record seeded port layouts) → **to-move
seat** `v_hat` per layout (the other seat is the zero-sum complement; never a second
forward pass — v2's `v_hat(P1)−v_hat(P2)` notation is retired for a single-seat rank
statistic sign-oriented to ThePhantom's seat).

- **Port-marginal estimand:** `v_hat` = mean over the 8 layouts; per-record spread
  reported as a diagnostic (not a gate quantity — v2's incommensurable band is
  replaced by the per-layout rule below).

### Metrics (game-level bootstrap; within-ThePhantom-seat unless labeled)

- **M0** — from §2 (in-distribution reference).
- **M1** — external discrimination: AUC/Spearman of the to-move-seat rank statistic
  vs outcome on EVAL-A; reported against the pip-count baseline and v8+search@50.
- **M2 — PRIMARY (α = 0.05):** partial Spearman / logistic coefficient of the
  calibration residual on **ORE+WHEAT pip-share**, exact stratified permutation on
  `(seat_is_thephantom, draft_position, opponent_strength.source)`, with
  **port-slot adjacency as a covariate** (the Skeptic's selection-bias control:
  elite players choose port-adjacent spots; marginal `v_hat` undervalues them,
  correlated with ore share). **Primary is luck-UNADJUSTED; the luck-adjusted M2 and
  the non-port-adjacent-subset M2 are the two named robustness rows** (no other
  variants may be computed — forking-paths clause). Binary reporting split fixed at
  the outcome-blind 65th percentile (sensitivity {60,65,70}; ORE-only in the grid).
- **M3** — supporting inversion check (unchanged from v2: stratified quartile
  WR-gap CI).

**Power, stated honestly (expert):** gating cell ≈ 74–240 games ⇒ M2's minimum
detectable partial correlation at α=0.05 / power 0.8 is **r ≈ 0.19–0.27** (a large
effect). This is why INCONCLUSIVE exists as a first-class verdict.

### GATE-A — three-way verdict (v3 replaces v2's binary rule)

- **GO** ⟺ **M2 significant in the predicted direction** (α=0.05, stratified
  permutation) **AND per-layout robustness:** the per-layout M2 estimate clears sign
  + significance in **≥7 of 8** port layouts (the computable replacement for v2's
  band) **AND** the tournament sign-agreement guard (§0.2) holds.
  *Hierarchical secondary door (tested only if M2 fails, its own α=0.05):* the
  pip-count baseline beats v8's M1 with a **paired game-level bootstrap CI on
  Δ(M1_v8 − M1_pip) excluding 0** — a trivial heuristic significantly out-ranking
  the champion is opening-specific failure dice variance can't explain.
- **INCONCLUSIVE** ⟺ any of: corpus below the §1 floor; M2 null with the CI not
  excluding the pre-registered MDE (r=0.20) — an underpowered null; per-layout
  estimates straddling significance (port-driven instability); tournament sign
  disagreement. Action: extend the corpus (harvest more videos) or decide B on
  PRE-GATE-0 + M0 grounds alone — pre-registered as "extend, not fail."
- **NO-GO** ⟺ a **well-powered** null: M2 CI excludes r=0.20 in both directions'
  favor of ~0 AND the secondary door also fails. Action: seeded arm shelved; B-alt
  only if PRE-GATE-0 collapsed; scoreboard becomes a standing eval.

Deliverables: `scripts/opening_scoreboard.py`, report with every number CI'd,
stratum-labeled, with the pipeline's per-archetype acceptance table beside M2.
Day-1 spike unchanged (bridge 20 tournament games, eyeball raw values first).

---

## 5. Workstream B — training response (gated on GATE-A GO + PRE-GATE-0)

Mechanism statement unchanged from v2 (zero setup-head gradient on seeded episodes;
pathways = value recalibration + the search pathway; `setup_entropy_coef` co-deployed
in **both** arms at the §2-calibrated value). **All B3 attribution is
seeded-vs-control** — the entropy bonus can never be credited to seeds (council).

**B-alt** (runs on GO regardless): calibrate the setup-prior's resource weights
against the corpus via a **win-prediction fit** (logistic regression of winner on
opening pip-share features) — NOT Phase A's existing NNLS-on-terminal-shares path,
which fits a different target (Architect).

### B1. Seed loader
As v2 (EVAL-A + `unknown` tier, mask-based legality, D6-canonical dedup on
`(board, both openings)`, per-episode seat randomization + port randomization),
plus **GATE-B0 coverage requirement vs v8** (expert): ≥25% of pool mass must lie in
archetype buckets where v8's PRE-GATE-0 natural mass is <5% — a pool that only
revisits v8's own basins fails loudly. Pool histogram printed beside the
per-archetype acceptance table (CV-rejection-bias visibility; stratified-sampling
fallback noted as amplifying CV-noisy records — accepted, visible).

### B2. Wiring
As v2 (`seed_prob` default 0.0; `reset_source` runtime tag; natural-only harness
assertion), with the league correction (all reviewers): the promotion ratchet is the
**sliding-window mean + streak** (`_anchor_window` / `anchor_window_stats` /
`maybe_promote_anchor`; the EMA was explicitly rejected in the 2026-07 audit). The
natural-only unit test asserts **`anchor_window_stats()` is unchanged** after a
seeded anchor-game outcome (EMA `opponent_win_rate` secondarily). TB additions as
v2 **plus `seeded/return` split by archetype bucket** (the B3-failure disambiguator).

### B3. Run design
As v2: warm-start v8, window+streak ratchet, `selfplay_v9_humanseed`,
`seed_prob=0.25`, **mandatory contemporaneous A/B** (control = same everything,
`seed_prob=0`, same `setup_entropy_coef`), interleaved on the same machine.

### B4. Gates
- **GATE-B0:** as v2 + the coverage requirement (B1) + bridged-tracker assertion.
- **GATE-B1:** unchanged (±3pp vs contemporaneous control, rolling 2M-step windows).
- **GATE-B2:** offline n≥600 symmetrised Wilson vs v8, natural-start (distinct from
  the in-run ratchet; both natural-only).
- **GATE-B3 (all on HOLDOUT-B, all paired seeded-vs-control):**
  1. head-to-head ≥0.55 vs v8 (n=600, natural-start), AND
  2. **calibration:** comparator = **frozen v8's M2 effect re-measured on HOLDOUT-B
     at B3 time** on identical games/strata/port draws (v8 never trained on corpus
     states — uncontaminated and free; the GATE-A number is context only, being
     winner's-curse-inflated). Criterion: the paired per-game residual-effect
     difference (v8′ vs v8) shows improvement with a **CI-bound equivalence test**
     at margin = 50% of v8's own HOLDOUT-B effect. If HOLDOUT-B is too small for
     the TOST (power check pre-registered at B3 time), the criterion degrades
     explicitly to "significant paired improvement + no worsening," labeled
     underpowered in the report. AND
  3. `openings/entropy_natural` (v8-vs-anchor condition) exceeds the
     **contemporaneous control's** by +0.3 nats or +1 bucket ≥10% — never compared
     to PRE-GATE-0 (both arms carry the bonus; council). Skipped as a criterion if
     PRE-GATE-0 showed no collapse. AND
  4. M1–M3 under v8′+search do not regress vs v8+search (deployed config).
- **Control-arm branch (expert NIT, now explicit):** the control runs through
  GATE-B2/B3 equivalents; if the control alone clears them, the honest conclusion is
  "the entropy bonus suffices" — the **control** is promoted, seeds credited only
  for the seeded-minus-control margin.
- **Kill/pivot with disambiguation:** on B3 failure, read `seeded/return` by
  archetype: flat ore-bucket returns → the policy never learned to *convert* those
  openings (pivot: mid-game curriculum / setup fine-tune); rising returns with
  failed B3(2) → value-side gap (pivot: B-alt / search-side). Two failures →
  seeds shelved; BC/piKL stay off the table.

Effort: as v2 (loader+wiring 2–3d; run = multi-day × 2 arms).

## 6. Non-goals — unchanged from v2.

## 7. Sequencing

```
NOW (no corpus):  §2 PRE-GATE-0: archetypes.py committed → v8 entropy + M0
                  + setup_entropy_coef calibration  (~1d + short runs)
corpus lands ──►  C: engine injection API + bridge + null tests (~2-3d)
                  ├─► A: day-1 spike → scoreboard (~3d) → GATE-A (3-way)
                  │     GO ──► B-alt + seeded A/B (B0..B3)
                  │     INCONCLUSIVE ──► extend corpus / decide on §2+M0 grounds
                  │     NO-GO ──► B-alt iff collapsed; standing eval
                  └─► holdout split committed BEFORE any metric
```

## 8. Review traceability

- **v1→v2:** see the v2 changelog (git history of this file) — 6 expert BLOCKERs +
  council round-1 findings, all verified structurally resolved by both tracks in
  round 2.
- **v2→v3 (expert round 2):** archetype featurizer frozen in-doc + committed
  pre-PRE-GATE-0 (E1); per-layout M2 robustness rule replaces the incommensurable
  band (E2); pip-baseline door now hierarchical with paired-bootstrap CI (E3);
  B3(2) comparator = frozen-v8-on-HOLDOUT-B with CI-bound equivalence (E4);
  corrected n arithmetic + MDE + three-way GATE-A with INCONCLUSIVE≠NO-GO (E5);
  population deviation declared + tournament sign-guard + record.py doc-sync (E6);
  engine injection API scoped + per-layout re-bridge ordering + no-op assertions +
  null-test port injection + human idempotence (S1–S2); window-mean+streak
  correction + `anchor_window_stats` test target (S3); GATE-B0 coverage-vs-v8
  (S4); luck-unadjusted primary + forking-paths clause + softened covariate claim
  (S5); archetype-split `seeded/return` disambiguator (S6); control-arm branch,
  single-seat M1 statistic, determinism pins (N1–N3).
- **v2→v3 (council round 2):** population honesty + sign-guard (Architect/Skeptic
  B1); B3(3) vs contemporaneous control (Architect S2, Researcher); port
  selection-bias covariate + INCONCLUSIVE routing (Skeptic B1); paired-CI GO door
  (Researcher B1); `setup_entropy_coef` pre-calibration (Researcher); port-RNG +
  featurizer-forward + league-test MUST-FIXes (Pragmatist); corpus-size floor +
  holdout power check (Architect); B-alt win-prediction fit (Architect).

## 9. Risks — as v2, plus:

| Risk | Mitigation |
|---|---|
| Port SELECTION bias (not just variance) | port-slot-adjacency covariate + non-port-adjacent robustness row + per-layout GO rule |
| Underpowered null read as absence | three-way verdict; MDE stated; corpus floor; "extend, not fail" |
| Winner's curse in B3(2) | frozen-v8-on-HOLDOUT-B comparator, paired |
| Featurizer tuned after seeing data | spec frozen in this doc; `archetypes.py` hash in the refuse-to-run manifest |
| Entropy bonus mis-credited to seeds | all B3 criteria paired seeded-vs-control; control-arm branch explicit |
| Engine no-op failures | bridge post-condition assertions |
