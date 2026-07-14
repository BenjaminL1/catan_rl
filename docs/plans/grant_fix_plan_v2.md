# Grant-fix plan v2 — ONLY what survived the expert panel

**Date:** 2026-07-14 · **Author:** Fable (proposal + panel orchestration) · **Executor:** Opus, unattended
**Provenance:** 4-part proposal reviewed by a 2-lens senior panel (CV/firewall + statistics) with
adversarial synthesis (workflow `wf_729d89b3-170`; verdict archived in the session transcript).
**Panel verdict on the original proposal: NOT-READY.** F2 (settlement-corroborated candidate
selection) is a **BLOCKER — dropped entirely**: it let the board pick the multiset the anchor is
supposed to check *against the board* (a tautology that voids the joint-flip firewall), adding
~k·0.07 confident-wrong-accept probability exactly where today's is zero. `0EtcbG16kHA g1`
(6-vs-3 WHEAT↔ORE) and `yQ4GZloiG08 g1` (disjoint reads) **stay lost — that is the correct outcome.**

What ships instead: **FIX A** (subset-collapse, with the panel's three fail-closed guards),
**FIX B** (the grant-line regex divergence the panel *found* — harvest matches the grant line by
exact substring while the anchor's own detector uses an OCR-tolerant pattern), and **FIX C**
(read-only diagnosis of the zero-boxes video). Plus a mandatory reframing of the gating math
(Wilson lower bound, not point estimate).

House rules (unchanged, binding): fail-closed sacred; additive-accepting only (byte-identical on
everything accepted today); `glyph_anchor.py` MUST NOT be edited (SHA-256 fingerprint-gated; the
validation crops are not in the repo — *importing/reading its symbols is fine, the hash covers the
on-disk source only*). Test-first; ruff + mypy --strict + pytest green per commit; stage by
explicit path; conventional commits; push origin/main; no AI trailers.

---

## FIX A — subset-collapse with the panel's guards (rescues `9Sm86ml04aI g5`; ~1.5 h)

**Basis (measured):** g5's split is `13×{ORE,SHEEP,WOOD}` vs `2×{ORE,SHEEP}` — the minority is a
strict SUBSET of the modal read (an icon missed in that frame), not a competing hypothesis.
**Panel guards (all three mandatory):**

Implement in `src/catan_rl/human_data/harvest.py` (NOT the fingerprinted module), as a
pre-processing step inside `_consensus_grant` — precedence pinned as follows:

```
reads -> [guard 3: drop >3-card reads] -> consensus_granted_glyphs (UNCHANGED, first)
      -> if None: subset-collapse the tally (guards 1+2) -> re-test:
             unanimous-after-collapse (>= MIN_GRANT_CONSENSUS_FRAMES)  OR
             _dominant_grant_read on the collapsed tally (existing >=5 / >=90% rule)
      -> else None (glyph_unreadable, exactly as today)
```

1. **Guard 1 — absolute superset floor AND ratio.** Collapse a subset read into a superset read
   only if `superset_support >= SUBSET_COLLAPSE_MIN_SUPERSET (= 5)` **and**
   `superset_support >= 2 * subset_support`. A bare 2-vs-1 (n=3) must NOT collapse — this
   deliberately drops `9Sm86ml04aI g3` (only 3 reads); the panel accepts that loss.
2. **Guard 2 — deterministic tie-break.** A subset may collapse only into the **unique**
   maximal-support strict superset among the reads. If two candidate supersets tie for max
   support, **skip the collapse** for that subset (fail closed). Single pass over the tally
   sorted by descending support; a read acts as a collapse TARGET only with its ORIGINAL
   (pre-collapse) support, so no chains/cascades — document termination in the docstring.
3. **Guard 3 — >3-card hard reject.** Before tallying, DROP any read whose total card count
   exceeds 3 (`sum(read.values()) > 3`): a setup grant is the 2nd settlement's adjacent resources
   (≤3 hexes), so a 4+-box read is definitionally a detector error and must never become a
   collapse target or a consensus vote.

Constants (module-level in `harvest`, with a docstring citing the measured cases):
`SUBSET_COLLAPSE_MIN_SUPERSET = 5`, `SUBSET_COLLAPSE_MIN_RATIO = 2`, `MAX_GRANT_CARDS = 3`.
Also record `diag["accepted_by"] = "subset_collapse"` when this path accepts (mirrors
`"dominant_read"`), and `diag["collapsed"] = [{subset, into, n}]`.

**Tests (write FIRST, in `tests/unit/human_data/test_glyph_anchor.py` next to the dominance
tests, reusing `_reads`):**
- g5 measured case: `{"ORE:1,SHEEP:1,WOOD:1": 13, "ORE:1,SHEEP:1": 2}` → ACCEPT `{O,S,WO}`.
- g3 measured case: `{"BRICK:1,WHEAT:1,WOOD:1": 2, "WOOD:1": 1}` → None (guard 1 floor).
- 6-vs-3 swap (`0Etc`): unchanged → None (no subset relation).
- 47-vs-1 (dominant path): unchanged → accepts modal (and assert `accepted_by` is
  `dominant_read`, i.e. precedence respected).
- Tie-break: `{"ORE:1,WHEAT:1": 6, "BRICK:1,ORE:1,WHEAT:1": 6, "ORE:1": 2}` → the `{O}` subset has
  two maximal supersets? construct a case where a subset has two equal-support strict supersets →
  collapse SKIPPED → None.
- Guard 3: a 4-card read `{"BRICK:1,ORE:1,SHEEP:1,WOOD:1": 9, "BRICK:1,ORE:1,SHEEP:1": 2}` → the
  4-card read is dropped BEFORE tallying → remaining `2×{B,O,S}` unanimous → accepts `{B,O,S}`
  (and never the 4-card read).
- Byte-identical regression: every case the CURRENT rules accept (unanimous 2-of-2; 47-vs-1
  dominant) returns the identical multiset with the fix in place.

**Write-up obligations (commit message / docstring, per panel):** acknowledge (a) the residual
true-2-card-grant→wrong-3-card-superset path (backstopped by the unchanged anchor at the
pre-existing p≈2/28≈0.07 collision rate — `orientation.py:143-148` checks BOTH settlements), and
(b) that a colliding wrong superset could mis-pin the granting vertex (order-establishment
corruption) at that same rate — unchanged from today's exposure, not new risk.

Commit: `fix(human_data): subset-collapse grant reads (guarded) — dropped-icon partials`.

## FIX B — harmonize the grant-line matcher (the panel's discovery; ~1 h)

**Basis (panel-verified in code):** `harvest.py` matches the grant line by EXACT substring
`"received starting resources"` at **three sites** (`_GRANT_PHRASE` used in `_ingest_two_pass`'s
dense-window trigger; the `_route_frames_to_games` grant-frame regex; `_grant_line_boxes`), while
`glyph_anchor.py:321` deliberately uses the OCR-tolerant `GRANT_RE = r"rece\w{0,4} starting
resources"` — which exists precisely because OCR mangles "received". A mangled line therefore
parses a `starting_resources` EVENT (logparse) yet produces **zero grant frames / windows** —
the exact `grant_frames=0` signature of `KvH76fJI4f0 g2`.

1. **Diagnose first** (scratch, not committed): for `KvH76fJI4f0 g2`, check the segment's events
   for `starting_resources` vs the recorded `grant_frames=0`. If events exist → the matcher hole
   is confirmed, proceed. If no events exist → honest loss; record in the summary and SKIP the
   fix for this game (but still do step 2, the divergence is real regardless).
2. **Fix:** in `harvest.py`, replace all three exact-substring grant matches with the tolerant
   pattern. `from catan_rl.human_data.glyph_anchor import GRANT_RE` (importing a constant does
   not modify the fingerprinted file — the hash covers on-disk source only; do NOT copy the
   pattern text, alias the constant so the two can never diverge again). Keep `_GRANT_PHRASE`
   as the doc name but make the matching go through `GRANT_RE.search(line.lower())`.
3. **Tests:** the documented manglings must now match at every harvest site — parametrize
   `"ThePhantom received starting resources"`, `"ThePhantom recei ed starting resources"` →
   actually use manglings GRANT_RE accepts: `"receivedl"`, `"recelved"`-style (`rece` + ≤4 word
   chars); assert `_grant_dense_windows` trigger, frame routing, and `_grant_line_boxes` all fire
   on a mangled line; assert a non-grant line (`"ThePhantom rolled"`) still does not.

Commit: `fix(human_data): grant-line matching uses the anchor's OCR-tolerant pattern`.

## FIX C — zero-boxes diagnosis, read-only (`eHIdnu4NjEA g1`; ~45 min, DIAGNOSTIC ONLY)

Call the **real** `detect_glyph_boxes` (panel: do NOT hand-replicate its geometry — a replica can
drift and misdiagnose) on the 20 line-found frames: re-prepare or reuse saved frames, run
`_grant_line_boxes` to get the line box, then step through `detect_glyph_boxes` with a scratch
copy of its INPUTS while printing its module constants (`MIN_ICON_CELL_W`, `MAX_ICON_CELL_W`,
`GLYPH_PITCH_LINE_FRAC`, `MERGED_BOX_PITCH_FACTOR`, mask thresholds — all importable, read-only)
to attribute which fail-closed exit fires (no mask components / icon-size band / merged-suspect /
pitch-split). Deliverable: a written attribution in the morning summary + updated memory. **Any
fix would require editing the fingerprinted module → flag to the user and STOP; local
revalidation is impossible (valset crops not committed).** No yield credit.

## RE-MEASURE + corpus + honest gating (~2 h compute + localization)

1. Re-run `prepare-frames` for `9Sm86ml04aI` and `KvH76fJI4f0` (both fixes affect them). Verify:
   `g5` grants readable (expect `accepted_by: subset_collapse`); `KvH g2` per FIX B's diagnosis.
   REGRESSION: all 9 currently-localizable games keep IDENTICAL `granted_resources`.
2. Localize every newly-eligible game (the documented VLM protocol: `localize_overlay.py` →
   grant self-check must pass → `localized/*.json` → `vlm_spike.py localize` → ACCEPTED) and
   re-run `collect_corpus.py`.
3. **Gating write-up (panel-mandated reframing)** in `docs/plans/vlm_spike_report.md`:
   - present the cell as POINT ESTIMATE **and** Wilson-95%-LB propagated (LB on the accept
     fraction; e.g. 11/17 point 0.647 → LB ≈ 0.41 → cell ≈ 106 < 147);
   - flag the single-video concentration (both F1 rescues are `9Sm86ml04aI`) and the double-use
     of the 8-video pilot for BOTH games/video and yield;
   - give the 0.6 order-established factor its own caveat (it is assumed, not measured);
   - conclusion wording: "point estimate clears CONTINUE; the CI does not — the decisive lever is
     MORE VIDEOS, not more fixes." Recommend a wider pilot (e.g. +10 videos) as the next compute
     spend. **The CONTINUE/ARCHIVE decision stays the user's.**

**STOP conditions:** any regression pin fails → revert that commit and write up. FIX B's
diagnosis shows no events for KvH g2 → skip its rescue claim. Anything requiring a
`glyph_anchor.py` edit → STOP, flag. Expected end state: yield 10–11/17 (59–65%), corpus ~11–12
rows, gating presented honestly with the Wilson framing.
