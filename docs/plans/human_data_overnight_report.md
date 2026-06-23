# Morning Report — human_data pipeline overnight build

**Run window:** 2026-06-22 → 23 overnight (autonomous, unattended)
**Read at:** ~08:00, 2026-06-23
**Bottom line up front:** **The build HALTED at Stage 0 (scaffold). It never reached Stage 1 (segment/strength) or Stage 2 (openings CV). No harvest ran. No corpus exists. No scoreboard number was produced.** The scaffold went through 4 review-and-resolve iterations and was still **NOT-READY** when the loop gave up ("stuck"). Per the gate-first rule, the build correctly refused to start any pipeline work on an un-greened foundation. Do **not** read any calibration / win-rate / opening-prior number from this run — there is none.

---

## 1. What shipped (modules + commits)

The scaffold *code* was written and committed; the scaffold *gate* did not pass. These two facts are separate — code on `origin/main` is **not** an indication the scaffold is READY.

Committed this run (newest first, `git log`):

- `43ce247` docs: flag known_window placeholder + seed-eligible necessary-not-sufficient `[skip ci]`
- `11138aa` fix: close rejection truth-table converse + real-d6-flip glyph-anchor test
- `f7753c0` fix: dice_log gate + board-desert anchor + scoreboard-eligibility predicate
- `60e7167` fix: type `_GAME1_HEXES` as `dict[str, Any]` for mypy --strict on tests
- `105dede` docs: sync human-data brief + report to schema v2 orientation-binding `[skip ci]`
- `ff3e9fb` fix: glyph-anchor firewall + scale-up orientation gates (FIX 4/5)
- `9dcd28e` fix: add road-incidence as snap-sanity gate, labeled NOT orientation
- `e74d58d` fix: provenance orientation-binding firewall (schema v2)

**Modules present in `src/catan_rl/human_data/`** (written, lint/type/test-green at the unit level, but the module as a whole is NOT-READY per the review gate):

| File | State |
|---|---|
| `record.py` (37 KB) | Schema v2 dataclasses, `validate()`, `is_scoreboard_eligible()`, `is_seed_eligible()`, provenance orientation-binding firewall. **Carries 1 unresolved BLOCKER + 4 unresolved SHOULD-FIX (see §4).** |
| `orientation.py` (8.7 KB) | D6 orientation module + `assert_scale_up_orientation_gates` (batch hard-block until glyph classifier validated). |
| `topology.py` + `topology.json` | Engine board topology fixture. |
| `ffmpeg.py` | Frame-extraction helper (thin). |
| `__init__.py` | Package surface. |

**READY modules: none.** The only module reviewed (`scaffold` = `record.py` + `orientation.py` + fixtures) returned `ready: false`, `halted: "stuck"`.

---

## 2. GATE results (every gate, with numbers)

Only one gate ran: the **scaffold review-and-resolve gate** (the §2 "Review" step of the CLAUDE.md loop). It never reached READY.

| Iter | Open findings at entry | Severities | Action |
|---|---|---|---|
| 0 | 4 | SHOULD-FIX, SHOULD-FIX, SHOULD-FIX, **BLOCKER** | resolve |
| 1 | 2 | SHOULD-FIX, SHOULD-FIX | resolve |
| 2 | 2 | SHOULD-FIX, SHOULD-FIX | resolve |
| 3 | 5 | **BLOCKER**, SHOULD-FIX, SHOULD-FIX, SHOULD-FIX, SHOULD-FIX | resolve |
| — | **HALT** | — | NOT-READY after 4 iters; loop declared "stuck" |

**Verdict: NOT-READY.** The loop did not converge: it regressed from 2 open at iter 2 to 5 open at iter 3 (a BLOCKER re-opened and SHOULD-FIXes multiplied), which is why the loop stopped rather than spinning a 5th time. **1 BLOCKER + 4 SHOULD-FIX remain open** (full text in §4).

**Stage 1 gate (segment/strength): NOT REACHED** — `stage1: null`.
**Stage 2 gate (openings CV / orientation firewall): NOT REACHED** — `stage2: null`.

I independently re-verified the three load-bearing claims against the current tree:
- No window table exists anywhere in `data/` or the repo (`find` for `*window*` returns nothing; `data/` contains only `exit/`).
- `record.py:is_scoreboard_eligible()` gates on `winner / passed_crosscheck / tier=="high" / rejection_reason` — and **still does not gate on `episode_source`**.
- `tests/fixtures/human_data/` has no `game1_resnap_overlay.png`; the only copy is the ephemeral spike `scripts/dev/human_data_spikes/opening_cv/game1_resnap_overlay.jpg`.

All three confirm the open findings are real, not stale.

---

## 3. Rejection-bias audit (§5.6)

**NOT REACHED.** The rejection-bias audit is a property of a *harvested corpus* (it compares the accepted vs rejected game populations). No harvest ran, so there is no corpus and no audit. There is **no rejection-bias number** to report. The `is_scoreboard_eligible()` predicate that the audit would call is itself still an open SHOULD-FIX (episode_source asymmetry), so the audit could not have been trusted even if a corpus existed.

---

## 4. WHERE it halted and WHY

**Halt point:** Stage 0 (scaffold), before any Stage 1 / Stage 2 / harvest work.
**Why:** The scaffold review gate stayed NOT-READY through 4 resolve iterations and the loop flagged itself "stuck." Per the gate-first convention (never start the expensive next stage before the current stage's gate is green), the build refused to scaffold-onward and stopped. This was the correct call — proceeding would have built the segment/strength and openings stages on top of an unfalsifiable strength label and a defeatable orientation firewall.

**The 5 open findings blocking READY:**

### BLOCKER — `opponent_strength` `known_window` is unfalsifiable (no committed backing data)
- **Where:** `src/catan_rl/human_data/record.py:136-149` (`StrengthSource`), `:152-165` (`OpponentStrength`); no `data/` window table exists.
- **Issue:** The scaffold constrains only the *shape* of the strength label (tier ∈ {high, unknown}, source ∈ {rank_badge, known_window}); it does NOT anchor `source="known_window"` to any committed `video_id`→window table. A `tier="high", source="known_window"` label is hand-stampable by any caller. Strength is the single most bias-inducing field in a calibration deliverable, and the headline §5.5 guarantee ("never pool a single number across mixed strength") becomes fiction if Stage 1's `segment.py` reads a hand-maintained/implicit window. Verified: no window/rank table exists in the repo or `data/`. (The scaffold's own docstrings at `record.py:140-148` and `:156-164` honestly flag this as a PLACEHOLDER — correct, but the gap must be a hard pre-condition on the segment/strength filter, not a deferred TODO.)
- **Fix (pre-condition on the segment/strength filter):** commit the known-high-rank window as a versioned package-fixture data file (`video_id` → `{window_id, date_range, rationale}`); have `segment.py` DERIVE strength from it; carry `window_id` onto `OpponentStrength` (new field) for audit traceability; exclude games not covered by a committed window (`tier="unknown"`, scoreboard-ineligible). Do NOT invent rank-OCR. Until then, the batch must emit no `high` label.

### SHOULD-FIX — `is_scoreboard_eligible()` does not exclude `episode_source != "natural"`
- **Where:** `record.py:552-569`.
- **Issue:** The scoreboard is a measurement over REAL natural games, but the predicate gates only on `winner / passed_crosscheck / tier / rejection_reason` — not `episode_source`. A `human_seed` record with a winner + `tier="high"` returns `True`. Low blast radius today (no seed records emitted yet) but the predicate is being frozen now as the documented SoT.
- **Fix:** add `and self.episode_source == "natural"` (cleanest), or document the asymmetry explicitly (mirroring the seed-eligible note) so it is a decision, not an oversight.

### SHOULD-FIX — `openings_desert_hex` must be the orientation the openings stage ACTUALLY snapped under
- **Where:** `record.py:495-550`; `orientation.py:1-36`.
- **Issue:** The orientation firewall's power rests on `openings_desert_hex` being an INDEPENDENT report. If the (unwritten) `openings.py` computes its stamp as `board_desert` or reuses the board affine, the firewall becomes a tautology and silently passes every D6 weld — the exact desert17/desert11 bug it exists to catch. This is a Stage-2 implementation trap.
- **Fix:** in Stage 2, DERIVE `openings_desert_hex` from the openings stage's own locked affine; add a Stage-2 test where the openings stage honestly reports desert=17 against a board=11 record and `validate()` rejects it. Until the glyph classifier exists, keep relying on `assert_scale_up_orientation_gates` to block the batch.

### SHOULD-FIX — cited verification artifact `game1_resnap_overlay.png` is not in the tree
- **Where:** `tests/fixtures/human_data/game1_openings.json:10` + `tests/unit/human_data/test_scaffold.py:78`.
- **Issue:** Both the golden fixture and the test comment cite `game1_resnap_overlay.png` as proof the desert=11 re-snap is correct (snap err <0.35px, owner colours confirmed). That PNG does not exist; the only copy is the ephemeral spike `.jpg`. The golden openings IDs (`[1,19]`/`[11,3]`) are the single load-bearing ground truth for the whole orientation firewall, and their only stated proof points at a 404.
- **Fix:** commit the overlay into `tests/fixtures/human_data/` (tiny; §6 sanctions a small board crop) and fix the `.png`/`.jpg` mismatch in both the note and the test, OR rewrite the note to cite a committed artifact.

### SHOULD-FIX — orientation firewall defeatable by a lazy openings stage that copies `board_desert_hex`
- **Where:** `record.py:495-550` + `orientation.py:140-178`.
- **Issue:** Same root as the previous finding, the single-flip case: if the openings stage copies `board_desert_hex` into `openings_desert_hex` (path of least resistance), the equality check is trivially satisfied and the firewall degrades to a no-op against a single-stage flip. Documented in the docstring but unguarded in code.
- **Fix:** in Stage 2, derive `openings_desert_hex` from the openings affine's own desert prediction and assert it is computed from the openings affine, not read from the board record; add the Stage-2 test above. Until then, add one sentence to the `record.py` docstring so the Stage-2 author cannot miss it.

---

## 5. HARVEST status

**GATED — nothing harvested.** No frames were extracted, no games segmented, no records emitted.

- **#games harvested:** 0
- **#rejected:** 0 (nothing to reject)
- **Rejection-bias audit:** not run (no corpus — see §3)

**What is needed to unblock the harvest** (in order):

1. **Resolve the BLOCKER** — commit the known-high-rank window as a versioned data fixture and wire `segment.py` to DERIVE strength from it with a `window_id` provenance stamp. Without this, every `high` label is unfalsifiable and the scoreboard is biased by construction. This is a hard pre-condition on the strength filter.
2. **Resolve the 4 SHOULD-FIX** — `episode_source` gate on the scoreboard predicate; commit the resnap overlay artifact; and (deferred to Stage 2) the two independent-`openings_desert_hex` derivation guards + tests.
3. **Re-run the scaffold gate to READY** — the loop must reach a verdict with no open BLOCKER/SHOULD-FIX before Stage 1 may begin.
4. **Then** Stage 1 (segment/strength) and Stage 2 (openings CV) gates must each pass before a harvest produces a trustworthy corpus.

---

## 6. Single-command resume state

Nothing is running and nothing is mid-write; the tree is clean apart from the committed scaffold and untracked scratch scripts. Resume = re-enter the review-and-resolve loop on the scaffold, starting from the BLOCKER:

```
# resume point: Stage 0 scaffold, NOT-READY, 1 BLOCKER + 4 SHOULD-FIX open
# 1. commit the known-high-rank window fixture + wire segment.py strength derivation (BLOCKER)
# 2. patch record.py is_scoreboard_eligible() episode_source gate (SHOULD-FIX)
# 3. commit tests/fixtures/human_data/game1_resnap_overlay.png + fix .png/.jpg cite (SHOULD-FIX)
# 4. re-green and re-run the scaffold review gate until READY, then start Stage 1
ruff check src/catan_rl/human_data && mypy --strict src/catan_rl/human_data && pytest tests/unit/human_data -q
```

No detached process to reattach to; no checkpoint to resume; no partial corpus to clean up.

---

## 7. Human-decision items

These two need *your* call — the build could not resolve them autonomously and they are the crux of the BLOCKER and a downstream gate.

### (a) Glyph-classifier status
**Status: does NOT exist; the true joint-D6-flip defense is therefore not yet armed.** The scaffold's orientation firewall catches a *single* D6 flip via the desert-hex binding, but a *joint* flip (board + openings both flipped the same way) preserves the resource/number multisets and is only caught by an independent glyph anchor. That glyph classifier is unbuilt. Until it exists and is validated, `assert_scale_up_orientation_gates` **hard-blocks the batch** — which is the correct conservative posture, but it means no scale-up harvest can run regardless of the BLOCKER above.
**Decision needed:** approve building + validating the glyph classifier as a Stage-2 prerequisite, or explicitly accept a single-flip-only firewall for an initial small/manually-verified batch (NOT recommended for any number that feeds the scoreboard).

### (b) Opponent-strength default
**Status: this is the open BLOCKER.** The brief's design-default is a "known-high-rank-window only" approach (a committed `video_id`→window table; games outside any window become `tier="unknown"` and scoreboard-ineligible). The scaffold did NOT commit that table, so today the default is effectively "unfalsifiable hand-stamped high," which is unacceptable for a calibration deliverable.
**Decision needed:** confirm the **"known-high-rank-window only, derive from committed data, exclude uncovered games"** default (recommended — matches §5.5), so Stage 1 can implement it as the BLOCKER fix. The alternative (rank-OCR badge reading) was explicitly rejected by the review and should not be built without your instruction.

---

## Throughline

The overnight build did the honest thing: it wrote the scaffold, reviewed it hard, found a strength-label that can't be falsified and an orientation firewall that a lazy Stage-2 author could neuter, failed to close them in 4 passes, and **stopped before harvesting anything** rather than ship a biased calibration number. There is no corpus, no scoreboard, and no success to claim — there is a clean, well-documented halt with one BLOCKER (commit the strength window as data) standing between you and a green scaffold, plus two decisions only you can make (build the glyph classifier; confirm the known-window strength default).
