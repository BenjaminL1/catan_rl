# Step 6 — Human-Corpus Use: Opening Scoreboard + Seeded Self-Play

**Status:** DRAFT v1 (2026-07-05) — under expert + council review.
**Depends on:** the human_data harvest completing (`docs/plans/human_data_pipeline.md`;
gold gate + glyph firewall green, corpus JSONL on disk).
**Scope:** what we DO with the corpus. Two workstreams: **A — the opening scoreboard**
(measurement) and **B — human-seeded self-play** (training), joined by a shared
**engine bridge** (C). A gates B. Everything here preserves the 1v1 ruleset, obs
schema, and checkpoint lineage (CLAUDE.md invariants).

---

## 0. Inputs & expected scale

The corpus: `GameRecord` schema-v2 JSONL — per game `{board hexes (19× resource+number),
openings (2 settlements + 2 roads per player, engine ids), draft_order, winner|null,
dice_log, opponent_strength.tier, episode_source, passed_crosscheck, rejection_reason,
provenance}`. All records passed the cross-check + orientation firewalls.

Expected scale (from manifest + pipeline yield estimates): 204 high-rank videos ×
~2–5 games/video × ~60–80% acceptance ≈ **~350–800 scoreboard-eligible games**, nearly
all featuring ThePhantom in one seat. The 574 `unknown` videos can add seed-only
records later (phase-2 decision, not needed up front).

Two structural facts shape everything below:

1. **One-human corpus.** ThePhantom occupies a seat in ~every game. Any realized-WR
   statistic is confounded by *his* skill; the plan controls for this (§A3) and never
   treats realized WR as an archetype leaderboard (pipeline brief §5.4).
2. **Winner-null games exist** (resign/cutoff). They are seed-eligible but excluded
   from all outcome statistics.

---

## 1. Workstream C — the engine bridge (shared foundation, build FIRST)

**What:** `src/catan_rl/human_data/engine_bridge.py` — deterministic construction of a
real engine state from a `GameRecord`:

- board from `hexes` (resource literals → engine resource order at this boundary —
  the two-orderings trap, CLAUDE.md rule 6; robber on the desert);
- snake-draft placements replayed via the ENGINE's own placement APIs in
  `draft_order` order (so distance rule, road incidence, and vertex/edge legality are
  re-checked by the engine itself — the mandatory seed-legality gate, brief §5.7);
- starting resources granted from each player's 2nd settlement (engine logic, not the
  record — the record's grant multiset was already used as a parse firewall);
- state finalized at "post-setup, pre-first-roll", correct player to move.

**API:** `build_post_setup_game(record) -> catanGame | BridgeReject(reason)`.
Rejects (illegal under engine re-check) are counted and reported by caller — expected
rare but nonzero (CV snap noise); a reject is a dropped record, never a repaired one.

**Tests:** golden game-1 record round-trips; a deliberately corrupted road id rejects;
resource-ordering conversion unit-tested both directions; determinism (same record →
identical state hash) pinned.

**Why first:** A's value probe and B's seed loader are both thin wrappers around this
one component. Effort ~1d.

---

## 2. Workstream A — the opening scoreboard (measurement; gates B)

### A1. Primary measurement — external value calibration

For each scoreboard-eligible game (winner non-null, tier=high, passed_crosscheck):

1. Bridge the record to the post-setup state.
2. Evaluate **v8's value head** from each seat's perspective (CPU, batch; obs built by
   the standard encoder — no new obs code) → `v_hat(seat)`.
3. Record `(v_hat, outcome, seat_is_thephantom, archetype features, dice-luck index)`.

**Pre-registered primary metrics** (computed once, reported with bootstrap CIs):

- **M1 — outcome discrimination:** AUC / Spearman of `v_hat` vs realized win across
  games. This is the EXTERNAL version of the Tier-0 net-accuracy probe (which measured
  0.43–0.50 opening Spearman *under v8 self-rollouts* — v8 grading its own homework).
  A low external score on human games CONFIRMS the opening-value blind spot with
  evidence v8 cannot game; parity with midgame external calibration would weaken it.
- **M2 — the ore hypothesis (from the human playtest):** define `ore_heavy(opening)`
  (≥ some pip-share of ORE+WHEAT production; exact threshold fixed before looking at
  outcomes). Test: calibration residual (realized outcome − rank-normalized `v_hat`)
  for ore-heavy vs not, permutation test. H1: v8 systematically UNDERVALUES ore-heavy
  human openings.
- **M3 — quartile check:** realized WR of openings v8 ranks in its bottom value
  quartile. If bottom-quartile human openings win ≳50% of their games, v8's opening
  value is inverted on real play, not just noisy.

**Baselines for context (same table):** a trivial pip-count heuristic value, and
v8+search@50 value (does search repair the ranking?). Both cheap.

### A2. Secondary — descriptive archetype table (explicitly NOT a leaderboard)

Coarse archetype buckets (ore-wheat / wood-brick / port-led / balanced; deterministic
featurizer in `src/catan_rl/human_data/archetypes.py`) with realized WR + Wilson CIs,
**presented with per-bucket n and the single-human caveat printed in the table
itself.** Used for eyeballing coverage and for B's diversity metric — never for
"archetype X is best" claims (per-bucket n ≈ 30–100 → ±9–18pp).

### A3. Confound controls (mandatory, cheap)

- **Single-human control:** primary M1–M3 computed (i) on ThePhantom's seats only,
  (ii) on opponents' seats only, (iii) pooled — reported separately. The
  within-ThePhantom comparison removes his skill as a between-archetype confound.
- **Seat/draft control:** first-placement advantage absorbed by including draft
  position as a stratification variable.
- **Dice-luck covariate:** from `dice_log`, per-player realized production minus
  pip-expected production for their opening ("luck index"); M1–M3 re-reported
  luck-adjusted (secondary, not gating).
- **Sensitivity split:** tournament-only subset (n=15 videos) re-run as a
  strong-vs-strong sanity check (wide CIs, directional only).

### A4. Deliverables, effort, decision gate

- `scripts/opening_scoreboard.py` (CLI: corpus in → JSON metrics + a markdown report
  at `data/human/scoreboard_report.md`), archetypes + bridge modules, tests.
- Effort: ~2–3 days after the corpus lands. All CPU, zero training risk.
- **GATE-A (pre-registered, decides Workstream B):**
  - **GO** if M2's residual gap CI excludes 0 in the predicted direction, OR M3 ≥ 0.5,
    OR M1 external Spearman ≤ 0.55 while midgame external calibration is materially
    higher (blind spot localized to openings).
  - **NO-GO** (blind spot not confirmed externally): B is paused; the opening-weakness
    hypothesis is re-examined (possibilities: the playtest impression was search-depth
    not value; the gap is in the POLICY prior only → consider a setup-prior lever from
    `setup_strength_roadmap.md` instead of seeds). This kill-path is the point of
    doing A first — it can save the multi-week B run.

---

## 3. Workstream B — human-seeded self-play (training; gated on GATE-A)

### B1. Seed loader

`src/catan_rl/selfplay/human_seeds.py`: loads the corpus, filters
`is_seed_eligible()` records, bridges each through `engine_bridge` (engine-legality
re-check = the loader's hard gate; rejects logged with reasons), dedupes identical
openings, and exposes `sample_seed(rng) -> catanGame`. Seed pool composition:
`high` + `unknown` tiers (seeds need diversity, not verified strength — brief §"uses");
winner is irrelevant for seeding.

### B2. Env + training wiring (additive, default-off)

- `seed_prob` config in `arguments.py` (default **0.0** — a run must opt in): each
  episode reset draws `u < seed_prob` → start from a sampled human seed (both seats'
  openings from the same record — the position is the unit of diversity), else natural
  empty-board start.
- `episode_source` is metadata ONLY: threaded to the buffer/TB, **never into the obs**
  (no policy-shape change, no leakage; assert in tests).
- **Eval/promotion discipline (the load-bearing rule):** the eval harness, anchor
  gates, and league promotion decisions consume ONLY natural-start episodes/games.
  The existing harness already starts fresh games; add an explicit assertion + test so
  a future refactor cannot silently pool seeded games into a promotion decision
  (else we re-import the human ceiling as a target).
- TB scalars (additive, never renamed): `seeded/return`, `seeded/ep_len`,
  `seeded/frac`, `seeds/legality_reject_rate`, `seeds/pool_size`,
  `openings/entropy_natural` (see B4).

### B3. Run design

- Lineage: warm-start from **v8** (`runs/anchors/v8_promobar_u243.pt`), the proven
  lowered promotion bar (0.63) + frozen-anchor machinery from the v8 recipe.
  Run name `selfplay_v9_humanseed`.
- `seed_prob = 0.25` initially (fixed, no curriculum — simplicity first; revisit only
  on evidence). Rationale: enough gradient exposure to human opening structures
  without starving natural-start play, which remains the eval target.
- **Control arm:** the natural-start anchor-WR trajectory of the v8-recipe run IS the
  baseline (same config minus seeds); if budget allows, prefer an explicit A/B
  (seeded vs unseeded from the same warm start) — decide at launch by M1 wall-clock
  budget.

### B4. Gates & kill criteria (pre-registered)

- **GATE-B0 (before launch):** loader + wiring unit tests green; 1k-episode smoke:
  0 crashes, legality-reject rate < 5%, obs schema byte-identical, natural-vs-seeded
  episode accounting correct.
- **GATE-B1 (in-run):** natural-start anchor WR must not degrade >3pp below the
  control trajectory for >2M consecutive steps (seeds interfering with base play) →
  halve `seed_prob` once; second trip → kill.
- **GATE-B2 (promotion, unchanged):** champion promotion = natural-start WR vs anchor
  ≥ bar, n≥600 symmetrised, Wilson.
- **GATE-B3 (the point of it all, post-run):** for the new champion vs v8:
  1. head-to-head natural-start WR ≥ 0.55 (n=600) — it must be STRONGER, and
  2. **re-run Workstream A's M1–M3 on the new champion** — the pre-registered success
     claim is that the external opening-calibration gap SHRINKS (M2 residual gap CI
     no longer excludes 0, or M3 < 0.4), and
  3. opening diversity: entropy of the champion's natural-start opening distribution
     (archetype-bucketed, from eval games) increases vs v8's.
- **Kill/pivot:** GATE-B3 fails twice (two tuned attempts) → seeds are insufficient;
  escalate to the setup-strength lever (`setup_strength_roadmap.md`) or a
  setup-phase-only fine-tune. **BC on the corpus stays OFF the table** (hundreds of
  openings is below the imitation regime) unless explicitly re-decided.

### B5. Effort

Loader + wiring + smoke ≈ 1–2 days; the run itself is the usual multi-day M1 training
with the review-and-resolve loop running alongside (CLAUDE.md standing convention).

---

## 4. Explicit non-goals (unchanged decisions, restated so they stay decided)

- **No behaviour cloning / no piKL** on this corpus (below-regime; human-likeness
  trades strength).
- **No move-agreement metrics** anywhere (Maia trap — a superhuman agent should
  diverge from humans).
- **No engine-rule, obs-schema, or action-space changes.** The bridge and seeds live
  outside the policy graph entirely.
- **No 4-player anything.**

## 5. Sequencing summary

```
corpus lands ──► C: engine bridge (~1d)
                 ├─► A: scoreboard (~2–3d, CPU only) ──► GATE-A ──► GO ──► B (wire ~1–2d, run ~days)
                 │                                        └─ NO-GO ─► pivot (setup-prior lever)
                 └─► (A's probe machinery is reused verbatim inside GATE-B3)
```

Spec-driven execution: A and B each become a Spec Kit feature
(`specs/010-opening-scoreboard`, `specs/011-human-seeds`) at build time; this document
is the design source both specs cite.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Single-human confound leaks into claims | A3 controls; scoreboard framed as calibration, never leaderboard |
| Seeded returns distort value on natural play | GATE-B1 interference tripwire + control trajectory |
| Eval contamination by seeded episodes | natural-only assertion + test (B2) |
| Bridge legality rejects too many records | reject-rate reported at C; >10% triggers a snap-quality investigation before A |
| GATE-A false NO-GO from low power | M1–M3 are paired/game-level (n≈350–800 → adequate for Spearman/AUC); the NO-GO branch re-examines rather than deletes the hypothesis |
| v8 value head sensitive to obs drift at post-setup states | bridge states validated against `rules_invariants.py`; spot-check 20 bridged states vs GUI-replayed equivalents |
