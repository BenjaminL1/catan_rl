# Step 6 — Human-Corpus Use: Opening Scoreboard + Opening-Diversity Training

**Status:** DRAFT v4 (2026-07-05). Review history: v1 expert 6-BLOCKER + council 4×NOT-READY;
v2 resolved all v1 blockers (verified both tracks); v3 resolved all round-2 findings
(verified); round 3 = expert NOT-READY ("READY-shaped v4" with 4 blockers, mostly
formula-precision) + council 2 ACCEPT / 2 REJECT-with-flip-conditions. v4 folds in every
round-3 item; §8 carries traceability.
**Depends on:** the human_data harvest completing. The corpus does **not exist yet**; the
pre-corpus lane (§2, incl. the corpus-free parts of §3) runs first.
**Invariants preserved:** 1v1 ruleset, obs schema, action space, checkpoint lineage,
natural-start-only evaluation/promotion, additive TB.

---

## 0. Inputs, scale, population honesty

Corpus: `GameRecord` schema-v2 JSONL. **Ports are unrecorded**; port **slot positions**
are fixed topology (9 slots) so *slot adjacency is computable*; type assignment is not.

Scale: 204 high videos × ~2–5 games × ~60–80% acceptance ≈ **~245–800 eligible games**;
the **gating cell** (within-ThePhantom-seat, EVAL-A) ≈ **74–240 games clustered in
~50–120 videos**. All power statements are made against clustered effective n (§4).

**Population statement (declared deviation).** `record.py` designates tournament-source
as the only opponent-controlled bucket; it is ~15 videos and cannot power any endpoint
(per-cell n≈7 for a 2-way split). This plan's primary population is therefore the
**within-ThePhantom-seat stratum** — his skill held constant; opponent strength
uncontrolled and acknowledged (the `rank_badge` covariate is near-constant and carries
almost no adversary information). Binding consequences:

1. GATE-A licenses the claim **"v8 misjudges the openings ThePhantom wins with"** —
   not "elite-vs-elite opening truth." The report headlines this framing.
2. **Tournament sign-guard:** binds any GO **only if** the tournament subset has ≥10
   eligible games (below that it is waived and reported — a 5-game point estimate must
   not be a coin-flip veto). Sign disagreement → INCONCLUSIVE. Caveat carried:
   tournament labels are title-regex, frame-unverified.
3. `opponent_strength.source` joins the permutation strata (§4).
4. Workstream A updates **both** `record.py` guidance docstrings (module header AND
   `is_strong_opponent_scoreboard_eligible`) to a coherent superseding contract naming
   the power-justified within-seat primary — never leaving text that contradicts this
   plan (doc-sync rule).

One-human corpus; winner-null games are seed-only and never enter any measurement.
**Terminology:** the measured quantity is **external opening calibration** (OPE caveat:
no claim to isolate value-head error from off-policy/dynamics divergence; bounded by M0
and the basin-mass robustness row, §4).

---

## 1. Pre-registration discipline

- Frozen by commit before any corpus number exists: this doc, **`archetypes.py`**,
  **`scripts/opening_scoreboard.py`** (expert: the slash-choice lives in code, so the
  code hash is part of the freeze), the corpus JSONL hash, checkpoint hash, and the
  split file. The CLI records all hashes and refuses to run on mismatch.
- **Floor (on the gating cell, not the pre-split pool):** if the within-ThePhantom-seat
  EVAL-A cell < **100 games**, GATE-A cannot return NO-GO — only GO or
  INCONCLUSIVE(extend).
- **Split BY VIDEO (round-3 clustering blocker):** videos are the sampling unit —
  games within a video share a session and opponent. EVAL-A (~60% of videos) /
  HOLDOUT-B (~40%), seeded RNG, committed file. Seeds come from EVAL-A videos only;
  GATE-B3 re-measures on HOLDOUT-B only.
- **Cluster statistics everywhere:** bootstrap resamples **videos** (all games of a
  drawn video enter together); permutation tests permute at the **video level** within
  strata. No game-level i.i.d. assumption anywhere.
- **Determinism:** per-record port layouts seeded by
  `int.from_bytes(hashlib.sha256(f"{video_id}:{game_index}".encode()).digest()[:8], "big")`
  (Python `hash()` is PYTHONHASHSEED-salted and is not used anywhere in this plan).

---

## 2. PRE-CORPUS LANE (run now; ~3–4 days total)

### 2.1 The archetype featurizer (frozen; load-bearing in four gates)

`src/catan_rl/human_data/archetypes.py`, committed before PRE-GATE-0 executes.

- **Features** per (seat, opening): 5-vector of pip-share by resource over the seat's
  two settlements (pips of adjacent hexes; desert = 0; normalized; **total pips = 0 ⇒
  bucket BALANCED_LOW directly**), total pips, port-slot adjacency (boolean, either
  settlement on one of the 9 fixed slot vertex-pairs).
- **"Share" always means the named PAIR-SUM** (ORE+WHEAT, WOOD+BRICK) — never a
  single-resource share (round-3 ambiguity). Buckets (fixed count 5, precedence order):
  1. `ORE_ENGINE`: ORE+WHEAT pair-share ≥ 0.45 (deliberately includes wheat-heavy,
     low-ore engines — the ore→city→dev axis; intended).
  2. `WOOD_BRICK`: WOOD+BRICK pair-share ≥ 0.45.
  3. `PORT_LED`: port-adjacent AND neither pair-share ≥ 0.45.
  4. `BALANCED_HIGH`: neither pair-share ≥ 0.45 AND total pips ≥ 26.
  5. `BALANCED_LOW`: otherwise.
- The 26-pip boundary is sanity-checked against PRE-GATE-0's v8 distribution (typical
  two-settlement totals ≈ 19–24, so BALANCED_HIGH may be small); any amendment happens
  **before** the corpus exists and is committed as a doc+code change — after PRE-GATE-0
  reports, the featurizer never changes.
- **Collinearity note (Architect):** the continuous ORE+WHEAT share is M2's regressor
  AND the ORE_ENGINE boundary; buckets are used only for description, diversity
  metrics, and B0 coverage — never as an M2 covariate — so the double use cannot
  contaminate the primary.

### 2.2 PRE-GATE-0 measurements

- **v8 opening distribution:** n≥400 natural v8-vs-v8 + v8-vs-anchor games → archetype
  histogram, entropy, `openings/setup_head_entropy`. Collapse verdict: ≥70% mass in one
  bucket. PRE-GATE-0 pins the **measurement condition** (v8-vs-anchor games — matching
  what the v9 run's periodic anchor evals produce) **and the collapse verdict only**;
  it is NOT the baseline for GATE-B3(3), whose comparator is exclusively the
  contemporaneous control (§5) — (round-3 stale-sentence fix).
- **M0 — in-distribution anchor:** same estimator as §4 (to-move-seat `v_hat`,
  port-marginal once §3's injection API exists) on v8's own eval games vs outcomes.
  Until the injection API lands, M0 is computed with **true ports** and re-run
  port-marginal when C completes (both numbers reported; the true-port M0 is the
  binding anchor — stated now to avoid a silent downgrade later).
- **`setup_entropy_coef` pre-calibration:** the setup-phase-only entropy term **does
  not exist in `arguments.py` yet** — its wiring is part of this pre-corpus lane
  (small additive loss term, default 0.0), then a short control-only sweep (2–3
  values) picks the largest value not degrading anchor WR. Fixed before B.

### 2.3 Corpus-free parts of Workstream C (moved forward — round-3 sequencing fix)

The engine **injection API** (additive, rule-2 flagged, `engine/board.py` in scope:
post-construction board overwrite + deterministic port assignment from an explicit
RNG — `updatePorts` currently draws from global `np.random`), the GameRecord
serialization/re-bridge of v8 self-play states, and **bridge null test (i)** (below)
are all corpus-independent and land now.

---

## 3. Workstream C — bridge + env injection (corpus-dependent parts, ~2d after harvest)

`build_post_setup_game(record, port_rng) -> catanGame | BridgeReject`:

- Resource-literal → engine-order conversion (rule 6); robber on desert.
- Placement legality via the env's action masks/geometry, **asserted with
  post-conditions** (engine `build_*` silently no-op on failure — verified; the bridge
  never relies on exceptions): piece counts, vertex/edge occupancy, distance rule,
  road incidence.
- Starting resources via spec-009 `bank_draw`; `assert_conservation` +
  `rules_invariants` on every bridged state; hand-tracker state == `player.resources`.
- **Per-layout ordering:** `player.portList` is captured inside `build_settlement` and
  the encoder's port one-hot is frozen at construction ⇒ each of the K=8 layouts is a
  **full re-bridge** (set ports → replay placements → grant → build encoder).

`CatanEnv.reset(seed_game=...)`: accepts a bridged game, skips interactive setup,
rebuilds obs-encoder index maps, seats the snapshot opponent, sets the to-move player.

**Bridge null tests:** (i) *(pre-corpus, §2.3)* 50 v8 self-play post-setup states,
true port layouts carried out-of-band and injected via the port parameter → `v_hat`
ε-parity vs the live env + engine state-hash equality; fixture set includes a
port-vertex opening settlement (catches a `portList` weld). K-marginalization is not
exercised here — it has its own determinism test. (ii) *(post-corpus)* bridge N human
records → re-serialize → re-bridge → state-hash idempotence; short random-agent
rollout to terminal, invariants green.

---

## 4. Workstream A — the scoreboard (~3d after C)

Per EVAL-A game: K=8 full re-bridges (per-record sha256-seeded layouts) → **to-move
seat** `v_hat` per layout (other seat = zero-sum complement; single forward pass —
never a fabricated second-seat obs). Port-marginal `v_hat` = mean over layouts;
per-record spread reported as a diagnostic only.

### The pinned formulas (each gate quantity has exactly one)

- **Residual:** `r_i = win_i − σ(3.22·v̂_i − 1.14)` — the **committed search squash**
  (spec 003), never refit on EVAL-A. `win_i`, `v̂_i` oriented to **ThePhantom's seat**.
- **M2 (PRIMARY, α=0.05, predicted sign POSITIVE** — v8 *under*values ore openings ⇒
  residual increases with ore share): **partial Spearman** of `r_i` on **ThePhantom's
  ORE+WHEAT pip-share**, partialling **port-slot adjacency**, exact stratified
  **video-level** permutation within strata `(draft_position,
  opponent_strength.source)` — `seat_is_thephantom` is NOT a stratum here (it is
  constant in the primary population; it appears only in the pooled secondary
  analyses) (Pragmatist fix). The logistic-coefficient variant is a **named robustness
  row**, not the primary.
- **Luck proxy (pinned):** per game, Σ over `dice_log` of (realized pip production of
  ThePhantom's two opening settlements − expected per-roll production), openings only,
  ignoring robber/bank/later builds. **Primary M2 is luck-unadjusted; the
  luck-adjusted M2 is robustness row #2** — and (Skeptic guard) **if the two disagree
  on significance, the verdict is INCONCLUSIVE**, never GO.
- **Basin-mass control (expert; robustness row #3):** re-run M2 with a competing
  covariate = v8's PRE-GATE-0 mass of the opening's archetype bucket. If ore-share
  loses its partial effect to basin-mass, the report attributes the result to
  **generic off-basin miscalibration**, not an ore blind spot, and B-alt's ore-weight
  fit is flagged. (Forking-paths whitelist = exactly these three named rows + the
  {60,65,70}-percentile / ORE-only reporting grid. Nothing else is computed.)
- **M1:** AUC/Spearman of the to-move-seat rank statistic vs outcome (cluster
  bootstrap CI), reported against **M1_pip** (pinned: ThePhantom's two-settlement pip
  total − opponent's, desert = 0, no port terms, same sign orientation) and
  v8+search@50.
- **M3 (supporting only):** stratified quartile WR-gap CI; its cell width (~±9–16pp at
  37–120 games) is quoted in the report so it is never promoted to evidence.
- **M0:** §2's in-distribution reference.

**Power, honest:** with video-clustering (design effect ~1.15–1.4 at ~3.5 games/video,
ICC 0.05–0.15), the effective gating-cell n is ~55–210 ⇒ minimum detectable partial r
(one-sided α=0.05, power 0.8) ≈ **0.19–0.32**. `r = 0.20` is the pre-registered
**minimum effect of interest** (not "the MDE"). Consequence, stated plainly (Skeptic):
**in the expected sample regime the verdict will usually be GO or INCONCLUSIVE; NO-GO
requires an unusually tight null and a gating cell ≳100** — INCONCLUSIVE is a
first-class, honest outcome routing to "extend the corpus or decide on §2+M0 grounds."
Expected-INCONCLUSIVE under a true r=0.2 effect: substantial (per-layout sign guard
~0.9+, tournament guard ~0.7–0.85 when it binds) — accepted by design and disclosed.

### GATE-A — exact decision rule (verdicts are exhaustive; INCONCLUSIVE is the else)

- **GO** ⟺ either:
  - **Primary door:** M2 significant (α=0.05, video-permutation, positive sign) AND
    **per-layout sign-agreement in ≥7/8 layouts** (each layout's M2 point estimate
    positive — sign only; the pooled port-marginal M2 carries significance, replacing
    v3's near-unreachable 8× individually-significant rule) AND the tournament
    sign-guard (§0.2) passes-or-is-waived; **or**
  - **Secondary door** (evaluated only when M2 is a **well-powered null** — CI
    excluding r ≥ 0.20): paired video-cluster bootstrap CI on Δ(M1_v8 − M1_pip)
    entirely below 0 (α=0.05), with the same per-layout sign-agreement analog on
    ΔM1 and the same tournament guard. (The door escapes a powered null only —
    it cannot rescue an underpowered one.)
- **NO-GO** ⟺ gating cell ≥ 100 AND M2's CI excludes r=0.20 (one-sided, predicted
  direction) AND the secondary door fails.
- **INCONCLUSIVE** ⟺ **anything else** (incl. floor breach, luck-consistency
  disagreement, layout sign-split ≤6/8, tournament sign disagreement at n≥10).
  Action: extend the corpus, or decide B on PRE-GATE-0 + M0 grounds — pre-registered
  as "extend, not fail."

Deliverables: `opening_scoreboard.py` (hash-frozen §1), report with per-archetype
acceptance table beside M2. Day-1 spike: bridge 20 tournament games, eyeball raw
values before building the statistics layer.

---

## 5. Workstream B — training response (gated on GATE-A GO + PRE-GATE-0)

Mechanism (unchanged, stated): seeded episodes give the setup heads **zero policy
gradient**; the pathways are value recalibration + deploy-time search (leaf values);
`setup_entropy_coef` (pre-calibrated, §2.2) is co-deployed in **both arms**, so all
credit assignment is **paired seeded-vs-contemporaneous-control**.

**B-alt** (runs on GO regardless): logistic **win-prediction** fit of winner on
opening pip-share features, **fit on EVAL-A only** (round-3: a HOLDOUT-B-fit weight
reaching v8′'s deployed setup prior would contaminate the B3 comparator) — distinct
from Phase A's NNLS-on-terminal-shares path.

### B1. Seed loader
EVAL-A + `unknown`-tier records (winner-null allowed — seeds never enter
measurements); mask-based legality; D6-canonical dedup on `(board, both openings)`;
per-episode seat + port randomization. **GATE-B0 coverage (round-3 deadlock fix):**
qualifying buckets = those with v8 PRE-GATE-0 mass < 5%, **or if none exist, v8's two
lowest-mass buckets**; require ≥25% of pool mass in qualifying buckets. If the pool
cannot meet it, that is recorded as the first-class finding **"the corpus adds no
unexplored opening mass"** and routed to the B decision (a legitimate reason to
prefer B-alt/search levers) — never misread as a loader bug (Architect).

### B2. Wiring
`seed_prob` (default 0.0), `reset_source` runtime tag (never in obs), natural-only
assertions at named call sites: the promotion ratchet is the **sliding-window mean +
streak** (`_anchor_window` / `anchor_window_stats` / `maybe_promote_anchor`; EMA was
rejected in the 2026-07 audit) — the unit test asserts **`anchor_window_stats()`
unchanged** after a seeded anchor-game outcome (EMA secondarily); the eval harness
asserts no seed pool attached. TB (additive): `seeded/frac`, `seeded/return`,
`seeded/return` **per archetype bucket** (the kill disambiguator),
`seeds/legality_reject_rate`, `seeds/pool_entropy`, `openings/setup_head_entropy`,
`openings/entropy_natural` (archetype entropy over the periodic natural-start
anchor-eval games).

### B3. Run design
Warm-start v8; `selfplay_v9_humanseed`; `seed_prob=0.25`; **mandatory contemporaneous
A/B** (control: `seed_prob=0`, same `setup_entropy_coef`, same code, interleaved on
the same machine). Honest wall-clock: per v7/v8 history each arm is **~1–2 weeks**,
not days; GATE-B1's windows are step-based for that reason.

### B4. Gates
- **GATE-B0:** loader/wiring tests; coverage requirement (B1); 1k-episode smoke
  (0 crashes, legality-reject <5%, obs byte-identical, bridged-tracker ==
  `player.resources`, `reset_source` accounting).
- **GATE-B1 (in-run):** seeded arm natural-start anchor WR within 3pp of the
  contemporaneous control, rolling 2M-step windows; first trip → halve `seed_prob`;
  second → kill.
- **GATE-B2 (offline):** champion vs v8, natural-start, **n≥600 symmetrised, Wilson
  lower bound ≥ 0.5** (promotion convention unchanged).
- **GATE-B3 (per-criterion scopes — the header claim "all paired on HOLDOUT-B"
  applies to (2) only):**
  1. Head-to-head ≥0.55 vs v8 — **Wilson lower bound**, n=600, natural-start.
  2. **Calibration (pinned construct, round-3):** paired **video-cluster bootstrap on
     Δ(effect)** over HOLDOUT-B: shared resample indices and identical port draws;
     recompute the pinned M2 coefficient for v8′ and for **frozen v8 re-measured on
     the same resample**; Δ = |effect_v8| − |effect_v8′|. **PASS ⟺ CI(Δ) lower bound
     > 0.5·|effect_v8(HOLDOUT-B)|** (superiority at margin — the word "equivalence"
     is struck; a v9 identical to v8 has Δ≈0 and FAILS). Pre-registered degrade path
     when HOLDOUT-B is small (power check at B3 time): CI(Δ) lower bound > 0 AND
     point estimate improved — same hypothesis family, labeled underpowered.
     Sign-flip overcorrection fails by construction (|·|).
  3. `openings/entropy_natural` exceeds the **contemporaneous control's** by +0.3 nats
     or +1 bucket ≥10% (5 fixed buckets). Skipped as a criterion if PRE-GATE-0 showed
     no collapse.
  4. M1–M3 under v8′+search vs v8+search on HOLDOUT-B: **no regression beyond a
     pre-registered tolerance of 0.02 AUC / 0.03 Spearman**.
- **Control-arm branch:** the control runs the same B2/B3 gates; if the control alone
  clears them, the conclusion is "the entropy bonus suffices" and the **control** is
  promoted — seeds are credited only for the seeded-minus-control margin.
- **Kill/pivot with disambiguation:** flat ore-bucket `seeded/return` → conversion
  failure (pivot: mid-game curriculum / setup fine-tune); rising returns + failed
  B3(2) → value-side gap (pivot: B-alt / search). Two failures → seeds shelved.

## 6. Non-goals (inline, self-contained)

No behaviour cloning and no piKL on this corpus (below the imitation regime;
human-likeness trades strength). No move-agreement metrics anywhere (a superhuman
agent should diverge from humans). No engine-rule, obs-schema, or action-space
changes; the bridge/seeds live outside the policy graph. No 4-player anything. No
port extraction in harvest v1 (if a future harvest records port types, §4's
marginalization becomes exact conditioning — a flagged pipeline-schema change).

## 7. Sequencing

```
NOW (pre-corpus lane, ~3-4d):
  archetypes.py (frozen) → PRE-GATE-0 (entropy + collapse verdict)
  + M0 (true-port; port-marginal re-run when C lands)
  + setup_entropy_coef loss wiring + calibration sweep
  + engine injection API + v8-state re-bridge + bridge null test (i)
corpus lands ──► C (bridge, ~2d) → null test (ii)
  ├─► A: day-1 spike → scoreboard (~3d) → GATE-A (exhaustive 3-way)
  │     GO ──► B-alt(EVAL-A fit) + seeded A/B (B0..B3, ~1-2wk/arm)
  │     INCONCLUSIVE ──► extend corpus / decide on §2+M0 grounds
  │     NO-GO ──► B-alt iff collapsed; scoreboard = standing eval
  └─► BY-VIDEO holdout split committed BEFORE any metric
```

Spec-driven execution: `specs/010-opening-scoreboard`, `specs/011-human-seeds` cite
this document.

## 8. Review traceability

- **v1→v2, v2→v3:** see git history of this file (all resolutions verified by both
  review tracks in the subsequent round).
- **v3→v4 (expert round 3):** M2 pinned verbatim — one statistic, committed squash
  link, stated sign; logistic → robustness row; scoreboard-code hash frozen (E1).
  B3(2) → paired Δ(effect) superiority-at-margin with |·| construct; "equivalence"
  struck; same-family degrade path (E2). By-video split + cluster bootstrap +
  video-level permutation + clustered MDE + floor moved to the gating cell (E3).
  GATE-B0 two-lowest-buckets fallback + first-class "no unexplored mass" finding
  (E4). GO/door logic restated exhaustively with else-INCONCLUSIVE; door escapes only
  powered nulls (S1). §2.2 stale B3(3) sentence removed (S2). Luck proxy + pip
  baseline pinned (S3). Featurizer pair-share disambiguation + zero-pip rule +
  26-pip sanity check (S4). B-alt fit on EVAL-A only (S5). Basin-mass competing
  covariate row (S6). Pre-corpus lane for corpus-free C parts + entropy-loss wiring +
  M0 true-port statement (S7). Non-goals inlined; per-criterion B3 scopes; Wilson-LB
  and tolerance pins; M3 cell width quoted; honest wall-clock; record.py doc-sync
  scope widened (NITs).
- **v3→v4 (council round 3):** per-layout rule → sign-agreement + pooled significance
  (Skeptic/Researcher/Pragmatist); sha256 determinism (Architect); floor on gating
  cell (Architect/Researcher); luck-consistency INCONCLUSIVE guard (Skeptic);
  tournament guard n≥10 waiver (Skeptic); degenerate stratum fix (Pragmatist);
  GATE-B0 conflation branch (Architect); NO-GO-rare honesty statement (Skeptic);
  collinearity note (Architect).

## 9. Risks — as v3, plus:

| Risk | Mitigation |
|---|---|
| Within-video session/opponent clustering | by-video split; cluster bootstrap; video-level permutation; clustered MDE |
| Gate quantities admitting multiple computations | every formula pinned in §4/§5; scoreboard code hash in the §1 manifest |
| Ore effect is generic off-basin miscalibration | basin-mass competing-covariate row; attribution rule pre-registered |
| GATE-B0 deadlock / misread | two-lowest-buckets fallback + first-class no-unexplored-mass finding |
| Verdict gaps between GO/NO-GO rules | INCONCLUSIVE is the exhaustive else-branch |
