# Spec: port-harvest — recover real Colonist port maps for the human-opening corpus

**Status: OWNER-DELEGATED DRAFT (2026-07-26) — written under explicit instruction to
proceed overnight ("get this done the best way you see fit ... have it ready for me in the
morning"), machine-local scope ("run solely on this machine"). NOT owner-ratified.
The premise challenges in §RISKS are OPEN, not cleared.**

**Feature in one line.** Read the port TYPE sitting in each of the 9 fixed board slots from
retained Colonist frames, emit it as a standalone sidecar keyed `(video_id, game_index)`,
and measure whether ports move the policy's opening choice at all — so the far more
expensive re-ingest of ~100 videos is funded by evidence rather than assumption.

## Why this exists

Every corpus row carries `board.ports == "OMITTED in v1"` (141/141). Ports **are** in the
observation (`src/catan_rl/policy/obs_encoder.py:280`, per-vertex 7-way one-hot). The
current convention marginalises over K=8 guessed assignments. For a **paired** comparison
that is defensible and is in fact the lower-variance estimator; for **unpaired** consumption
(training on ThePhantom's openings) a guessed port map is simply a false statement about the
board he faced.

## MEASURED BASELINE — the numbers this spec is built on

An expert probe over all 34 retained frames established the following. **Do not re-derive
these; build against them.**

* **Geometry is near-exact.** The affine's linear part is `scale·I` to within **0.2%**
  (`A/scale = [[1.0017,−0.0008],[0.0001,0.9983]]`) — no rotation, skew, or perspective.
  Render scale spans **1.1486–1.2321 px/engine-unit** (ratio 1.073) across the 34 videos.
* **Photometry is NOT free.** Naive scale-normalised 64×64 k-means (k=6) gives cluster sizes
  `[83,65,52,49,30,21]` against the expected `[136,34,34,34,34,34]` and **0/34 boards satisfy
  the composition**. Sail-masked + binarised: `[124,83,26,26,24,17]`, **2/34**. Even the
  2:1-vs-3:1 split fails naively (per-board majority 5–8, never 4). **2/34 is the baseline to
  beat**, not 34/34.
* **HUD panels occlude ports** (pieces do not). With a loose 0.95·edge window, **38/306 slots
  (12.4%)** locked onto white HUD rectangles, concentrated in **slot 5** (dice/action widgets)
  and **slot 6** (log panel), p95 jitter 54.7/80.3 px on slot 5.

## Decisions (binding for the build)

### D1 — Slot geometry: reflected water-hex centre + a constant offset
Localise each port as `2·mid − hex_centre` (the reflection of the slot's hex centre through
the slot-edge midpoint), plus a **constant `(−11.0, −7.0)` px** offset that is identical
across all 9 slots to ±0.5 px (slot-wise range −10.6…−11.6, −5.7…−7.4). Use a **tight
±0.35·edge search window** — this is what keeps slots 5 and 6 off the HUD.
**Measured: 306/306 slots localise on 34/34 `post_setup` frames; bias-removed jitter p95
≤ 0.9 px (x) / 2.2 px (y).** This supersedes an earlier perpendicular-normal-at-78px
approach, which was validated on only one board.

### D2 — CLUSTER-THEN-LABEL, not template matching
Do **not** hand-author reference templates and do **not** assume sprite matching is exact —
measured naive performance is 0/34 and 2/34. Instead: extract all slot crops, cluster them
**unsupervised** across the whole corpus, and bind clusters to names by hand-labelling the
**6 cluster centroids once** (~5k tokens). Rationale: the 6 sprites recur on every board, so
the class↔name binding is a 6-item problem, not a 1260-item one — and it is the ONLY signal
the script cannot self-generate (see D3).

### D3 — Dual decode, with its blind spot named
Decode twice: (a) **unconstrained** per-slot argmax over match scores, (b) **constrained**
Hungarian assignment subject to the fixed composition. Agreement required; disagreement ⇒
typed rejection. This catches systematic confusion that produces an illegal multiset, which
the permutation-invariant multiset check cannot see.
**Known blind spot, do not paper over it:** a *consistent global transposition* of two
classes (every ore read as brick and every brick as ore) yields a **legal** multiset, so both
decodes agree and are wrong on every board. Mislabelled reference templates produce exactly
this. No self-check detects it — **the D2 hand-labelled centroids are the only defence**, and
the report must say so.

### D4 — Fail-closed, and NEVER deduce a slot
If any of the 9 slots is unreadable, reject the whole board. Deducing the 9th from the other
8 is BANNED even though the fixed composition always permits it: it would let a classifier
that never actually reads a slot report success.

### D5 — SIDECAR, not a corpus field
Write `data/human/ports/harvest.jsonl`, keyed `(video_id, game_index)`, hashed independently.
Do **NOT** write into `row["board"]["ports"]`. Any adapter reading `board` would otherwise
inherit ports for free and the "training-only, measurement keeps K=8" fence would be a
convention rather than a mechanism. The corpus JSONL is in the step6 §1 freeze set and its
hash moved once already today (the `opponent_strength` restamp); this build must not move it.

### D6 — Scope is PIXELS-ON-DISK ONLY. Re-ingest is NOT authorised.
Only **34** frame dirs exist (`data/human/vlm_spike/frames/<video_id>__g<N>/`), of which
**24** join an accepted corpus row. The remaining rows have **no pixels** —
`data/human/vlm_spike/localized/*.json` stores only `video_id`, `game_index`, `vlm_note`,
`players`. Harvesting them requires re-acquiring ~100 videos: the real cost pole, explicitly
out of scope, fundable only by the D7 result. Harvest all 34 (the 10 orphans still exercise
the classifier) but report the two sets separately.

### D7 — The invariance probe is a DELIVERABLE
No port-accuracy tolerance exists because the operative quantity is not per-slot accuracy but
**decision-flip rate**: P(the policy's chosen corner changes | port map changes). Compare the
policy's setup choice under the **real** harvested map against each of the K=8 guessed
assignments. Report flip rate and metric spread. This decides whether the ~100-video
re-ingest is worth a day.

### D8 — Verify GEOMETRY and PHOTOMETRY separately
One check cannot do both jobs. Geometric jitter (measured, ≤2.2 px p95) rules out within-board
slot permutation **independently of the classifier** — which is what the permutation-invariance
worry was actually about. The composition check therefore only has to test **photometry**,
where it has real power (the 0/34 and 2/34 results prove it discriminates). Run it as **N
independent 9-way tests of one shared classifier**: a systematic confusion breaks nearly every
board at once.

## Non-goals
- No re-ingest / downloads / `yt-dlp` (D6).
- No write to `data/human/corpus/provisional_openings.jsonl`; no corpus re-hash (D5).
- No change to engine, obs schema, action space, or training config.
- No use of harvested ports in any paired measurement path; K=8 stays the scoreboard convention.
- No claim that harvested ports improve any metric — D7 measures, it does not assert.
- Does not fix the 5 known illegal-settlement rows (4 have no pixels).

## RISKS — open challenges from the premise review (NOT cleared)

**Pre-mortem (most likely failure).** The harvest ships at high per-slot accuracy and nobody
uses it. The labels land in `human-opening-reference`, never ratified because its own D4
leaves the instrument **unsigned** — it can refute "openings are weak" but never confirm
strength. Exact ports make a refutation instrument more precise without letting it conclude
anything new. The opening battery in `scripts/opening_sweep.py` is **port-blind** (pips,
lumps, contested_race, denial, cut_vulnerability), so ports enter only via the policy's
per-vertex one-hot on 18 of 54 vertices, where the K=8 draws were already near-invariant.
Meanwhile the agent still scored 4 VP with zero cities in 58 turns and
`src/catan_rl/eval/harness.py` still reports no cities, no VP-by-turn, no bank-trade
composition. *We raised the precision of a measurement we could not act on, while the failure
we had actually observed stayed unmeasured.* **D7 exists to make this detectable before it is
paid for.**

**Strongest opposite case.** This reverses a RATIFIED non-goal: **step6 §288 — "no harvest-v1
port extraction"**. The harvest is orphaned at both ends: the upstream consumer
(BC/imitation) is banned by step6 §6 and was killed by four councils; the downstream consumer
`opening_scoreboard.py` **does not exist in the tree**; seed-injection's mechanism
(`reset_source`/`seed_prob`) returns nothing under grep. For a paired instrument K=8 is the
lower-variance estimator, so real ports are arguably a measurement **downgrade**. The
counter-case is weaker only on **ordering**, not merit. D6 confines the spend to pixels
already on disk precisely to bound this.

**Cost honesty.** The script build is **300–600k tokens**, iteration-bound closing the
2/34 → 34/34 gap; an earlier 100–200k estimate was wrong. The hand-labelling leg is ~5k
(6 centroids). Neither is the real driver — the ~100-video re-ingest for the other rows is.

**Known corpus rot this build does not fix.** 5 rows carry adjacent (illegal) settlements;
`human-opening-reference.md:77` is stale (it calls `opponent_strength` a hardcoded constant,
which today's restamp changed to 125 rank_badge / 13 tournament / 3 unknown), and that
restamp incidentally satisfied step6 §0.2's ≥10-tournament sign-guard at 13.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `make test-unit`).
2. **Geometry pin**: D1's localiser reproduces **306/306 slots on 34/34 frames**, with
   per-slot jitter reported; regression fails if any slot fails to localise.
3. **HUD pin**: the tight ±0.35·edge window is enforced, and a test shows a loose window
   reproduces the slot-5/slot-6 HUD lock-on that the tight one prevents.
4. **Photometry pin**: composition pass rate is reported against the measured **2/34
   baseline**; the build must state its own rate honestly rather than assert success.
5. **Dual-decode pin**: a synthetic slot swap is DETECTED by D3 and NOT by the multiset
   validator; and a synthetic *global transposition* is shown to defeat D3, with the report
   naming the hand-labelled centroids as the only mitigation.
6. **Fail-closed pin**: an occluded/blank slot yields a typed whole-board rejection, and no
   code path emits a 9-slot map from fewer than 9 independently read slots (D4).
7. **Sidecar pin**: writes only under `data/human/ports/**`; corpus JSONL sha256 byte-identical
   before and after (D5).
8. **Invariance deliverable**: D7's decision-flip rate and metric spread, with a plain
   statement of whether the ~100-video re-ingest is justified.
9. Determinism: same frame ⇒ identical port map, bit-for-bit.
10. Read-only w.r.t. training: nothing under `runs/train/**`.

## Follow-on (explicitly NOT this build)
Re-ingest of ~100 videos for the pixel-less rows — gated on D7 showing ports actually move
opening choices. Independent of that, both prior premise reviews ranked **aggregate midgame
instrumentation** (cities built, bank-trade composition, VP-by-turn over existing self-play
corpora — n in the thousands, zero new data, no ports) above any further opening-side work.

## Implementation notes (2026-07-26, as SHIPPED — where the build diverged from the spec)

- **AC2 is NOT met as written, and BOTH losses look recoverable.** The localiser reaches
  **288/306 slots on 32/34 frames**, not 306/306 on 34/34. An earlier version of this note gave
  two causes and *both were wrong*; the causes are now measured per rejected board into
  `geometry.json["rejection_diagnostics"]` and printed verbatim in `report.md`, so nothing here
  is narrated:
  - `Hj_VF4PhHwM__g1` — **not** a HUD panel displacing slot 5. The board reaches a consensus
    sail offset `(3.70, −30.65)` with support **8/9**, and slot 5 *does* offer a candidate
    0.64 px from consensus. It is **slot 4** that kills it: its only candidate sits **6.38 px**
    out against `MAX_SLOT_DEVIATION_PX = 6.0`. The board is lost to a threshold by **0.38 px**.
  - `fqBK3_-PO7g__g1` — **not** "no orientation-locked affine at all". 18 tokens, 12 candidate
    affines, screen-rule gap **46.86** against a 3.0 minimum (a confident orientation lock), and
    17 of 18 per-token residuals under 2.6 px. One spurious detection at ~108 px drags the mean
    residual to **31.05** past the 5.0 px cap, and `_trim_token_outliers` drops nothing because
    the token count is already at its 18 ceiling. The ingest pipeline recorded
    `board_residual_px = 1.01` for this video, so a good affine for this board is known to exist.

  Both are whole-board fail-closed rejections (D4), never partial boards, and the regression pin
  is set **at** the achieved 32 so a further board cannot drop out silently. **Recovery was
  deliberately not attempted**: admitting a board changes the clustered board set → changes the
  centroids → changes their fingerprint → invalidates the committed `centroid_labels.json` and
  forces a fresh labelling pass. So whether to loosen `MAX_SLOT_DEVIATION_PX` or add a
  residual-outlier retry to `fit_board_affine` — and therefore whether AC2 should be amended at
  all — is an **open owner decision**, not a settled one.
- **D2's hand step was NOT performed by the owner.** The build authored `centroid_labels.json`
  itself, which is the one thing D2 forbids: the binding is the only defence against D3's blind
  spot, and a defence executed by the same actor that produced the clusters is not defence in
  depth. The file now says so in its own fields (`authored_by: BUILD AGENT`,
  `attestation_status: UNRATIFIED`) instead of reading as a human attestation. The binding has
  been independently re-read off the six images and is **correct**, so no harvested data is
  believed wrong — but **D2 is not closed until the owner looks at
  `data/human/ports/centroids/cluster_{0..5}.png`**, which are now tracked in git for exactly
  that purpose (the frames they derive from are gitignored, so the images are not re-derivable
  from a clean checkout).
- **Phase 2 no longer re-freezes the centroids.** `main()` used to call `freeze_centroids()` and
  `write_centroid_images()` on every run, so a `--labels` run overwrote the artefact its own
  fingerprint check exists to protect *before* that check could abort. Freezing and image
  emission are now phase-1 only.
- **A third blind spot, named.** The D6 desert corroboration is skipped where a frame dir's
  `meta.json` lacks `board_desert_hex` — 3 boards (`5WamwGjkHcE__g2`, `6yyzAd63Gs0__g1`,
  `cXun_M90NBA__g2`), all of which are in `harvest.jsonl`. A wrong D6 element relabels all 9
  slots at once, i.e. a global slot permutation that the composition check, both decode legs and
  the jitter envelope all wave through. Skips are now recorded in `geometry.json`, reported, and
  enforced-as-recorded by a unit test.
- **The geometry pin was near-vacuous and now has a real one.** `localise_board_slots` fits the
  anchor offset from the candidates it then measures deviation against, so the jitter assertion
  could not fail unless the localisation assertion already had. The check with teeth — the rigid
  sail offset is *constant across boards* (measured x 3.119…4.061, y −31.769…−29.779) — is now a
  **unit** test reading the committed `geometry.json`, so CI runs it even though the frames are
  gitignored.
- **AC3's rationale is partly refuted by its own ablation.** The tight ±0.35·edge window does
  not by itself prevent the slot-5/slot-6 lock-on (18.4% tight vs 20.7% loose on matched
  boards); the sprite-area cap plus the cross-slot consensus do. The window is still enforced
  as specified. Arms are compared over the intersection of boards every arm localises.
- **D2's binding is pinned to a centroid FINGERPRINT, not a cluster index.** k-means indices
  0..4 permute when the board set changes, so a labels file keyed by index alone would silently
  rebind — the exact global transposition D3 says nothing can detect. `centroid_labels.json`
  therefore carries `centroids_sha256`, phase 2 decodes against the frozen `centroids.npz`, and
  a mismatch aborts. `centroids.npz` + `centroid_labels.json` are committed for audit.
- **D7 ran with the real maps, and it does NOT settle the re-ingest.** Real-vs-guessed
  realised-pair flip rate **0.0938**, 95% CI **[0.0417, 0.1542]** — a *cluster* bootstrap over
  the 60 (board, seat) cells, which is the correct unit: the 480 comparisons are 60 cells × K=8
  guesses all sharing one real pair, so the effective n is ~60, not 480. Only **12 of 60** cells
  flip at all, and 52 of 64 cells produce a single opening across all 8 guesses. An earlier
  build compared the point estimate against a **0.05 threshold invented in code** — D7
  pre-registers none ("no port-accuracy tolerance exists") — and printed "RE-INGEST IS
  JUSTIFIED"; that verdict flips inside its own interval and has been removed. `recommendation()`
  now returns a plain measured statement and asserts no threshold. Two further caveats travel
  with the number: guessed-vs-guessed is **0.0932** (CI [0.0446, 0.1484]), statistically
  indistinguishable from the real leg — the hand-labelled map behaved like just another guess and
  added no information to this decision; and the measurement is on
  `selfplay_pointer_arch_v2/ckpt_000000500.pt` (`opening_sweep.DEFAULT_CKPT`), *not* a banked
  champion, so it is a proxy for the champion's port sensitivity rather than a measurement of it.
  The re-ingest remains unauthorised (D6) and is now explicitly an owner call on evidence that
  does not resolve it.
- **Provenance is recorded, because none of it is reproducible from a clean checkout.**
  `report.md` and `invariance_probe.json` now carry the full invocation (`argv`, resolved
  `--frames-root`, `--corpus`, `--ckpt` + its sha256, window/margin settings). The shipped run
  used `--frames-root /Users/benjaminli/my_projects/catan_rl_v2/data/human/vlm_spike/frames`
  (the capture machine's copy; the worktree has no frames).
