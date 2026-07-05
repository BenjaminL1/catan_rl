# MORNING REPORT — human_data opening-extraction pipeline (overnight autonomous build)

**Run:** overnight, autonomous, unattended. You did not watch this run — you asked me to
build this phase while you slept. This is the 8am read.

**Bottom line, up front:** The pipeline is **partially built and green where it is green,
but the FULL CORPUS WAS NOT HARVESTED.** Four of six modules are **READY** (scaffold,
Stage-1 gate, `validate`, `batch`), but the two modules that reconstruct openings and defend
board orientation — **`openings` and `glyph_anchor` — HALTED RED** after 4 review/resolve
iterations each with a **BLOCKER still open**. Because the full harvest is **hard-gated on
BOTH the gold gate AND the glyph firewall being green**, and the glyph firewall is not
validated, **the harvest did NOT run: zero games were extracted, zero rejected, and the
rejection-bias audit was NOT reached.** Nothing below claims a gate that did not pass.

---

## 1. What shipped (READY modules + commits)

| Module | Status | Commit | Notes |
|---|---|---|---|
| `scaffold` | **READY** (verify-only) | `588e3b9` | No new commit — foundation + orientation-bug fixes were already on `main` (lineage through `8e4bfce`/`734316e`); nothing was missing to commit. |
| Stage-1 gate | **PASSED** | (gate, not a module) | See §2 for the numbers + 2 non-blocking caveats. |
| `validate` | **READY** (iter 1) | `2020cade` | Pushed `36f9a08..2020cad`. Scope-locked to `validate.py` + tests; CPU-only, no gui/engine/training imports. 4 advisory SHOULD-FIX left open (see §5). |
| `batch` | **READY** (iter 3) | `d0af930` | Current `HEAD`. Resume/quarantine plumbing. 4 advisory SHOULD-FIX left open (see §5). |

Current `HEAD` = `d0af930` on `main`, consistent with the batch READY commit above.

**NOT shipped (halted RED):**

| Module | Status | Halt reason |
|---|---|---|
| `openings` | **RED — HALTED "stuck"** | 1 BLOCKER open after 4 iters (see §3). |
| `glyph_anchor` | **RED — HALTED "stuck"** | 2 BLOCKERs open after 4 iters (see §3). |

---

## 2. GATE RESULTS (every gate, with numbers)

### Stage-1 gold gate — **PASSED**
- `tools_green`: **true**
- `gold_reproduced`: **true**
- `invariants_pass`: **true**
- `fixture_untampered`: **true**

Two caveats were recorded on the passing gate. **Neither invalidates the pass**, but both are
carry-forward items:

1. **Download flake (environmental, not pipeline):** `nmk59XWFRBU` (1 of 5 gate videos)
   failed to **download** (not parse) — `yt-dlp` `"No supported JavaScript runtime /
   Requested format is not available"` under the 1080p-only format fallback. The other 4
   downloaded fine with `node` as the nsig runtime. Fix owed in `batch.py`: per-video
   download-failure quarantine + retry, likely `--js-runtimes` or a format re-fallback. The
   segment/logparse pipeline itself is unaffected.
2. **Gate-driver artifact (not a pipeline bug):** the Stage-1 gate harness passed
   `pov_handle=None`, so on POV videos ThePhantom's "You …" lines resolve `actor=None` and
   the ruleset filter reads `actors=1` on games it should pass. The **production** path threads
   the HUD-seat `pov_handle` (covered by `test_pov_handle_maps_leading_you_to_pov_seat`). When
   Stage-2 wires HUD-seat detection, **re-verify `ruleset_ok=True` on POV-video victory
   games.** Winner extraction is unaffected.

### Stage-2 gates — **SKIPPED (not run)**
- Reason recorded: `s1Ready=true`, `s1Gate=true`, but `s2blocked=[openings, glyph_anchor]`.
- Because two Stage-2 modules never reached READY, the Stage-2 gate harness was **skipped**
  (`g2.ok=false`, `g2.skipped=true`). No Stage-2 gate numbers exist because it did not run.

### Full-harvest gate — **GATED, harvest did NOT run**
- `gold=false`, `glyph=false` → harvest refused. (`gold=false` here is the *harvest-scale*
  gold precondition being unmet because Stage-2 is incomplete, distinct from the Stage-1
  fixture gold gate which **did** pass.)
- Recorded decision: *"Pipeline built + validated; corpus harvest needs BOTH the gold gate
  AND the glyph firewall green. Not scaling unprotected."* — this is the correct call, see §4.

---

## 3. WHERE it halted and WHY (the two open BLOCKERs)

### `openings` — BLOCKER: road path emits a confidently-wrong opening under realistic occlusion
The settlement path was hardened with an **area-dominance margin guard**
(`_MIN_SETTLEMENT_AREA_MARGIN`: accepted blob must tower over the best rejected candidate,
else reject "ambiguous"). **The road path has NO equivalent** — only an absolute pixel floor
`_MIN_ROAD_PIXELS=20`. Red-team, reproduced end-to-end on the committed real game-1 frame:

- Settlement 19 (BLACK/ThePhantom) incident-edge road-mask pixel counts:
  edge 35 = **76** (TRUE opening road), edge 10 = **37** (WRONG-but-incident), edge 34 = 0.
- The floor's docstring claims leaks collect ~11–34px and true roads ≥29px. But the real leak
  (37px on edge 10) **exceeds** the claimed ceiling (34) — so **leak (37) > ceiling (34) >
  floor (20)**: the floor does **not** separate leaks from true roads even unmodified.
- Occlude the true road (paint a bright `(255,240,120)` glow along edge 35 — one of the
  module's own named failure modes) and re-run: `_road_for_settlement(19)` now snaps to the
  runner-up **edge 10** (37px > floor 20).

**Why it is dangerous, not merely noisy:** edge 10 = verts (4,19) **is incident** to
settlement 19, so the §5.7 road-incidence legality re-check **and** `record.validate` both
**PASS** the fabricated road. The result is a fully-legal, confidently-wrong `GameRecord` that
would silently corrupt the scoreboard/seed corpus. **Fix owed:** mirror the settlement guard —
require the winning incident edge to dominate the 2nd-best incident edge by a margin
(winner ≥ K × runner-up), else fall to `road_unresolved`; and recalibrate the floor.

### `glyph_anchor` — 2 BLOCKERs (this is the module that hard-gates the safe harvest)
**BLOCKER 1 — joint-flip firewall is resource-multiset-only and matches EITHER settlement.**
The glyph anchor is the **sole** defense against a jointly-flipped board+openings (both stages
flip together, so the desert-binding trivially agrees — the §5.2 orientation trap). But the
check it feeds compares the granted-card multiset against the **resource** multiset only, and
matches if it equals **either** opening settlement's adjacency. The committed board has only
**28 distinct 3-hex resource multisets across 54 vertices**, and **38/54 vertices share a
multiset with another vertex** (pinned in `test_glyph_anchor_multiset_collision_rate`). So a
joint D6 flip landing the 2nd settlement on a collision-partner vertex **passes**
`assert_glyph_anchor`. The two documented mitigations (match only the 2nd/resource-granting
settlement via `draft_order`; corroborate with number-token adjacency) are written as
"the batch MUST apply" in the docstring but **enforced nowhere in code**.

**BLOCKER 2 — `classify_glyph` confidently mislabels a desaturated grey ORE stone as a
coloured resource.** `glyph_anchor.py:333` gates ORE with `if sat < ore_max_saturation`; any
swatch at/above the ceiling falls into the hue branch with no floor. Reproduced end-to-end:
`HSV(5,62,175)` reddish-grey stone → `classify_glyph(...) == 'BRICK'` (should be ORE or an
honest None). **Catastrophic, not lossy:** ORE→BRICK corrupts the granted multiset; a classic
ore-city opening `{ORE,WHEAT,SHEEP}` misreads to `{BRICK,WHEAT,SHEEP}`, a very common 3-hex
adjacency, which under a joint flip can then **match** the flipped settlement and let a
jointly-flipped board weld into the corpus — the exact failure the anchor is the sole defense
against. Root cause: `ore_max_saturation` is derived from **board hex tiles** but applied to
**log card icons** (a different rendered asset). None of the 37 passing tests attack this
saturation-just-above-ceiling direction. **Fix owed:** give the hue branch a saturation
floor / ORE-vs-nearest-hue margin so low-saturation swatches fail closed to None.

---

## 4. HARVEST STATUS — **GATED (did not run)**

**No corpus was harvested.** `harvest.ran = false`.

- **#games extracted: 0**
- **#rejected: 0**
- **Rejection-bias audit (§5.6): NOT REACHED** — it runs only during a harvest; none ran.

**Which gate blocked it:** the **glyph firewall** (`glyph_anchor`, RED with 2 open BLOCKERs)
and the harvest-scale **gold** precondition (Stage-2 incomplete because `openings` is also
RED). The harvest requires **BOTH** green; **both are red**.

**What is needed to unblock the full harvest:**
1. Land the `openings` road-dominance margin guard (§3) → `openings` READY.
2. Land the `glyph_anchor` fixes (§3): 2nd-settlement-only + number-token corroboration
   (BLOCKER 1) and the ORE saturation floor (BLOCKER 2) → `glyph_anchor` READY.
3. **Validate the glyph classifier** so `assert_scale_up_orientation_gates` stops raising
   `GlyphClassifierNotValidated` (this is the human-decision item in §6 — it is the hard gate
   on the *safe* full harvest).
4. Re-run the Stage-2 gates (currently skipped), then the harvest gate re-checks `gold` +
   `glyph`.

The decision **not** to scale an unprotected harvest was the right one: without the glyph
firewall, a jointly-flipped board relabels all 54 vertex / 72 edge IDs and would be welded
into the corpus **confidently wrong, not merely noisy.**

---

## 5. Advisory SHOULD-FIX carried on the READY modules (non-blocking)

These did **not** block READY but are open. Most bear directly on **audit honesty** and the
**glyph firewall being silently optional** — worth a look before any harvest.

**`validate` (4):**
- **Board-CV multiset misclassification substitutes a fixed placeholder board** — biases the
  §5.6 rejection-bias audit for exactly the feature-correlated subset it must measure
  (`validate.py:357-370`; `record.py` has no raw-features field). A board-CV-failed game gets
  relabeled to one synthetic desert-11 archetype, so an analyst would wrongly conclude
  rejection is not feature-correlated. **Fix:** preserve raw CV board on the rejected record
  (`raw_hexes`/`provenance.cv_hexes`) or tag as an explicit unbucketable stratum.
- **Joint-flip glyph-anchor firewall is silently optional in `cross_check`** (×2 findings) —
  `granted_by_player` defaults to `None` and the anchor runs only `if granted_by_player is not
  None` (`validate.py:191,289`); `assert_scale_up_orientation_gates` is never called by
  `cross_check`. A batch caller that forgets it gets `accepted=True` on a jointly-flipped
  board. **Fix:** make the joint-flip defense non-silent — require an explicit opt-out or
  refuse `accepted=True` unless the anchor ran; `batch.py` MUST call
  `assert_scale_up_orientation_gates` per game.
  *(RESOLVED 2026-07-05: `cross_check` now rejects `glyph_unreadable` when a grant read is
  absent/`None` for either player — anchor-ran is a precondition of acceptance;
  `run_batch` calls `assert_scale_up_orientation_gates` once per run and raises
  `GlyphClassifierNotValidated` on an absent/failed validation; `BatchResult` /
  `CrossCheckResult` carry {anchor_ran, anchor_unreadable, anchor_mismatch} + grant-read
  coverage telemetry.)*
- **`cross_check` cannot verify BoardRead is cross-frame-stable** — the §5.2-mandatory
  stability lives in an unenforced calling convention. **Fix:** add `frames_corroborated:int`
  / `cross_frame_stable:bool` set by `read_board_stable`, reject when <2.

**`batch` (4):**
- **`HarvestPlan` collapses 204 "high" videos into one scoreboard number** — hides that the
  defensible strong-vs-strong n is only the **15 tournament** videos (the 189 `ranked_rank`
  highs are opponent-uncontrolled). `record.py` says this split MUST be reported and never
  collapsed. **Fix:** split scoreboard by `source` and print both.
- **`batch-plan` ETA omits the Stage-2 board-OCR term** (~0.85h) — the go/no-go compute number
  structurally under-reports. **Fix:** pass `accepted_board_frames_per_video`.
- **`batch-plan` ETA models only the sparse Pass A** — dense setup-window frames unbudgeted.
- **`net_concurrency` unenforced for a legacy 1-arg parse_fn** — downloads can fan out to
  `max_workers`, violating the §5.11 1–2-wide cap. **Fix:** hard-fail/warn when the gate is
  not accepted and `net_concurrency < max_workers`.

---

## 6. HUMAN-DECISION ITEMS

1. **Glyph classifier is NOT validated (hard gate on the safe full harvest).**
   `assert_scale_up_orientation_gates` correctly **raises `GlyphClassifierNotValidated`**
   today — this is the intended block, and it is why I did not scale the harvest. **Decision
   needed:** how you want the glyph classifier validated (e.g. label a held-out set of log-card
   glyph crops and pin an accuracy floor) before enabling the full corpus run. Until that
   exists, **no accepted record is trustworthy against a joint D6 flip**, so the harvest must
   stay gated.
2. **Two BLOCKERs are genuinely stuck after 4 iters each** (`openings` road-dominance guard,
   `glyph_anchor` 2nd-settlement + ORE-floor). These are well-diagnosed with concrete
   reproductions and named fixes (§3) but were not resolvable inside the autonomous loop's
   iter budget. **Decision needed:** approve the specific fixes in §3 so I can land them next
   session, or adjust the approach.
3. **Scoreboard-power reality:** the defensible strong-vs-strong calibration n is **15**
   (tournament), not 204. Confirm you want the two-number reporting (tournament vs rank_badge)
   before any scoreboard is built.

---

## 7. SINGLE-COMMAND RESUME STATE

- **Branch:** `main`, **HEAD:** `d0af930` (batch READY commit; workspace consistent with the
  journal). Pushed through `2020cad` for `validate`.
- **Green modules to build on:** `scaffold`, Stage-1 gate, `validate`, `batch`.
- **Resume target:** land the two open BLOCKERs (§3), starting with `glyph_anchor` (it hard-
  gates the harvest), then `openings`, then re-run the Stage-2 gates and the harvest gate.
- **Resume command:**

  ```
  make human-data-resume          # re-enters the Stage-2 review/resolve loop at the
                                  # halted modules (openings, glyph_anchor); on green it
                                  # re-runs the Stage-2 gates then re-checks the harvest
                                  # gate (gold + glyph). If that make target is not yet
                                  # wired, resume via the Stage-2 loop entrypoint under
                                  # src/catan_rl/human_data/ targeting [openings, glyph_anchor].
  ```

  The harvest will remain refused until **both** `gold` and `glyph` read green and the glyph
  classifier is validated (§6 item 1).

---

**One plain throughline:** the plumbing is built and the parts that could be proven correct
were proven and committed — but the two safety-critical readers (opening-road detection and
the joint-flip glyph firewall) each have a reproduced, confidently-wrong failure that the
existing checks pass, so I held the harvest at the gate rather than weld a corrupt corpus.
Nothing here was harvested, and nothing here claims a gate it did not pass.
