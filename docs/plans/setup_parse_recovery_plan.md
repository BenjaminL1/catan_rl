# Setup-Parse Recovery Plan — unblock the ThePhantom opening corpus

**Date:** 2026-07-12 · **Author:** diagnosis session (Fable) · **Executor:** Opus
**Goal:** real-video setup parsing works end-to-end → re-run the VLM spike → fill the yield/gating
numbers in `docs/plans/vlm_spike_report.md` → take the pre-registered corpus decision.
**Status quo:** yield = 0/2 on Tier-5 real videos; every game typed-rejects with
`endgame_stats_overlay_no_opening_frame`.

---

## 1. Root cause (evidence-anchored — do not re-diagnose, verify then fix)

**Colonist.io renders the OBJECT of log lines as an inline icon, not text.** A real log line is
`"ThePhantom built a 🏛 (+1 VP)"` / `"ThePhantom placed a 🏠"` — OCR yields `"built a (+1 vp)"`,
`"placed a"`, with **no noun**.

The grammar (`src/catan_rl/human_data/logparse.py:296-301`) requires the noun:

```python
(re.compile(r"placed a settlement"), "setup_settlement"),   # never fires on real footage
(re.compile(r"placed a road"), "setup_road"),               # never fires
(re.compile(r"built a city"), "built_city"),                # never fires
(re.compile(r"built a settlement"), "built_settlement"),    # never fires
(re.compile(r"built a road"), "built_road"),                # never fires
```

Noun-less verbs in the same table (`\brolled\b`, `\bbought\b`, `gave bank`, `\bstole\b`,
`\bgot\b`, `received starting resources`, victory verb) **do fire** — which is exactly the
signature observed in `data/human/vlm_spike/frames/33KR75rhTgo__g1/meta.json`: grants present,
winner present, `dice_log` empty, `log_setup_sequence` = only grant lines, zero placed/built events.

**Failure cascade** (each step is a consequence, not an independent bug):

1. No `setup_settlement`/`setup_road` events → `log_setup_sequence` degenerates → placement order
   unestablishable → `draft_order` silently falls back to sorted handles (`harvest.py:282`).
2. No `built_*` events → `_post_setup_frame` (`harvest.py:913-947`) finds `boundary is None` and
   **falls back to `bucket[-1]` — the game's LAST frame** (the "Well Played!" endgame stats
   overlay). This fallback encodes "no builds logged ⇒ game never left setup", which is FALSE when
   builds are merely invisible to OCR. This is the **fail-open** hole in an otherwise fail-closed
   pipeline, and it is what handed the VLM an endgame frame.
3. The dense sampling pass (`ingest.py` two-pass design) runs "only inside the bounded setup
   windows the sparse pass discovers" — those windows are derived from setup events → dense pass
   never triggers on real videos → fewer setup-era frames (sparse 4 s cadence still covers, but
   verify in Step 4).
4. `empty_baseline = bucket[0].frame` (`harvest.py:899`) lands on a channel-intro splash because
   nothing validates the baseline actually shows a board.

**Why the anchor game didn't catch this:** the game-1 fixture
(`tests/fixtures/human_data/game1_openings.json:24-27`) hand-writes
`"rayman147 placed a Settlement"` — a *textual noun* that real footage never produces. The fixture
encoded a false assumption about the log rendering; the whole noun-matching design was built and
tested against it.

**Already working — do NOT touch:** winner extraction (victory verb, normalised), segmentation
welds (commits `8e4bfce`, `e8b9899`, `02c1e5a`), board CV (`board_hexes` + residual 1.12 px read
fine even on the failing game), grant glyph consensus, the VLM localization + snap math (8/8 exact
on the anchor), the fail-closed validators.

---

## 2. Fix design (minimal — no VLM change, no OCR change, no new abstraction)

The insight that makes this cheap: **we never needed the nouns.**

- **Setup piece type is determined by position parity.** Colonist 1v1 setup is strictly
  settlement-then-road per placement turn, snake order: actors `A,A,B,B,B,B,A,A`; pieces
  `S,R,S,R,S,R,S,R`. Eight noun-less `"placed a"` lines + actor handles fully reconstruct
  `log_setup_sequence`, `draft_order`, and the placement-order contract (`record.py:380-414`).
  Validate the actor pattern; on any deviation → the existing ORDER-UNESTABLISHED typed path
  (seed-eligible, eval-excluded) — never a guess.
- **The post-setup frame boundary only needs "a build happened", not what was built.** A noun-less
  `built_any` event is a valid member of `_MAIN_GAME_BUILD_KINDS` for bounding the stable
  8-pieces-down window. (Rolls/trades/robber moves do NOT end the window — the board still shows
  exactly the 8 setup pieces until the first build — so the boundary stays first-build.)
- **Make the frame router fail-closed:** if no build boundary exists but the window demonstrably
  progressed past setup (any `roll`/`victory` event present), reject with a typed reason instead
  of returning the endgame frame.

---

## 3. Implementation steps (execute in order; review-and-resolve loop per CLAUDE.md at the end)

### Step 0 — Base commit + verify the icon hypothesis on real setup-era frames (~30 min)

0a. The prior session's spike WIP is uncommitted and idle since Jul 10 (`scripts/vlm_spike.py`,
    `tests/unit/human_data/test_vlm_spike.py`, `data/human/vlm_spike/{localized,truth}/`,
    `provisional_openings.jsonl`, `docs/plans/vlm_spike_report.md`). Commit it AS-IS first (its
    tests must pass) so every subsequent diff is clean:
    `git add` those paths **explicitly** (never `-A`; `$SCRATCH/`, `src.mp4`, `data/exit/`,
    `scripts/dev/*_wf.js`, `scripts/export_dice_vectors.py`, `scripts/record_conformance.py`,
    `.claude/scheduled_tasks.lock`, `scripts/dev/review_step6_plan_wf.js` stay uncommitted) →
    `feat(human_data): land vlm opening-localization spike WIP (harness + report skeleton)`.

0b. Extract 3–5 setup-era frames from video `33KR75rhTgo` (first ~3 min of game 1;
    `scripts/vlm_spike.py prepare-frames` machinery or `ingest_video` directly), crop the log
    panel (`crop_log`), run `ocr_log_crop`, and print the raw lines. **Confirm**: lines matching
    `placed a` appear with NO piece noun in the OCR text (and visually, the piece is an icon).
    Save the cropped PNGs + OCR dump to the scratchpad for the record.
    **HARD GATE: if the noun IS present in the OCR text, STOP — the diagnosis is wrong; report
    back instead of proceeding.**

### Step 1 — Noun-less grammar fallbacks in `logparse.py` (test-first, ~1 h)

New event kinds in the `LogEventKind` literal (`logparse.py:60-76`): `setup_placed_any`,
`built_any`.

Append to `_GRAMMAR` **after** the existing specific patterns (ordering is load-bearing — the
table is first-hit-wins, so fixtures with textual nouns keep their specific kinds; only noun-less
lines fall through to the new patterns):

```python
(re.compile(r"placed a"), "setup_placed_any"),   # after setup_settlement / setup_road
(re.compile(r"built a"), "built_any"),           # after built_city / built_settlement / built_road
```

Placement in the table must keep `"placed a settlement"` → `setup_settlement` (add a test locking
this). Beware: the two new patterns must sit AFTER `starting_resources` etc. only if regex overlap
demands it — actually they overlap nothing else; the critical orderings are within the
placed/built families.

**Tests first** (`tests/unit/human_data/test_logparse.py`, follow existing style):
- `"thephantom placed a"` → kind `setup_placed_any`, actor resolved.
- `"thephantom built a (+1 vp)"` → `built_any`.
- `"thephantom placed a settlement"` still → `setup_settlement` (regression lock).
- `"you placed a"` with `pov_handle` → actor = pov handle.
- Full icon-stripped 1v1 setup transcript (8 placed-any lines + 2 grants, snake order) parses to
  the expected 10-event stream.

Commit: `feat(human_data): noun-less placed/built log events (colonist renders pieces as icons)`.

### Step 2 — Setup-sequence reconstruction by parity (test-first, ~2 h)

Wherever the setup sequence is consumed (grep: `setup_settlement`, `setup_road`,
`log_setup_sequence`, `establish_placement_order` — `harvest.py`, `record.py`,
`orientation.py`), accept a **parity-typed** sequence built from `setup_placed_any`:

- Collect the window's `setup_placed_any` (+ any specific `setup_*`) events in stream order.
- **Validation gate:** exactly 8 placed events AND actor sequence == snake pattern
  (`A,A,B,B,B,B,A,A` for the window's two handles; `A` = first placer). Both orderings of
  handles are checked; grants interleave and are ignored by the validator.
- On pass: assign piece types by parity (odd position = settlement, even = road) and synthesize
  the same structures the textual path produced (`log_setup_sequence` strings, `draft_order`,
  the per-player settlement/road order contract of `record.py:380-414`, and the
  grant-line-position → 2nd-settlement rule feeding `establish_placement_order`).
- On fail (missed line, weird actor pattern): the existing ORDER-UNESTABLISHED path
  (`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED = False`) — the game stays seed-eligible,
  scoreboard-ineligible. Never guess.

`draft_order` must now come from the placed events on real footage — delete/bypass the
sorted-handles fallback *for the validated case* (`harvest.py:282` area) but keep it for the
unvalidated fallback (it is labelled best-effort there).

**Tests:** clean 8-line parity reconstruction equals the textual-noun reconstruction on the same
game (use the game-1 fixture with nouns stripped); 7-line stream → ORDER-UNESTABLISHED; wrong
actor pattern → ORDER-UNESTABLISHED; interleaved grant lines don't break parity.

Commit: `feat(human_data): parity-typed setup sequence from noun-less placed events`.

### Step 3 — Fail-closed frame routing (test-first, ~2 h)

`harvest.py`:

- `_MAIN_GAME_BUILD_KINDS` (line 910): add `"built_any"`.
- `_post_setup_frame` (line 913): when `boundary is None`, check the window for any `roll` or
  `victory` event. If present, the game demonstrably progressed past setup while builds were
  invisible → **do not return `bucket[-1]`**; the game must read out as a typed reject. Implement
  by returning a sentinel/None and having the caller emit `GameFrames`-slot `None` with a new
  typed reason constant `POST_SETUP_FRAME_UNRESOLVED` (mirror the existing `frames_unrouted`
  pattern at `harvest.py:104`). Keep the true "cutoff during setup" case (no roll, no victory,
  no build) returning `bucket[-1]` — that assumption is still sound.
- `empty_baseline` (line 899): choose the earliest bucket frame whose OCR window precedes the
  first `setup_placed_any`/`setup_settlement` event AND that passes the existing board-grid fit
  (reuse the board CV's grid-fit call; reject splash/lobby frames). If none qualifies →
  `empty_baseline = None`; type it `Optional`. The VLM lane (`vlm_spike.py` localizer prompt +
  `FileLocalizer`) must tolerate a missing baseline (it localizes from `post_setup` alone — it
  scored 8/8 with a baseline, and the baseline is advisory); the classical CV lane keeps
  requiring it (typed reject, unchanged behavior).
- Dense-pass windows: find where Stage-1 supplies dense windows to `ingest_video` (grep
  `dense` in `harvest.py`/`ingest.py` callers) and make the window derivation use
  `setup_placed_any` + `starting_resources` events too, so the dense pass actually fires around
  real-footage setup. If the derivation turns out to already key off grants, note it and move on.

**Tests:** synthetic event streams — (a) placed×8 + roll + built_any → post_setup = latest frame
before the built event; (b) placed×8 + roll + victory, no builds → typed reject
`POST_SETUP_FRAME_UNRESOLVED`, NOT `bucket[-1]`; (c) placed×8 only (true cutoff) → `bucket[-1]`
kept; (d) baseline = None when no pre-placement board frame exists, and the routed game still
reaches the VLM lane.

Commit: `fix(human_data): fail-closed post-setup frame routing + validated empty baseline`.

### Step 4 — Re-run the spike on the 3 cached videos (~1–2 h, mostly compute)

- `ruff` + `mypy --strict` + full `pytest` green first.
- Re-run `scripts/vlm_spike.py prepare-frames` for `33KR75rhTgo` (g1, g2) and `AoOXWyxaTkA` (g1).
  **Visually verify** each emitted `post_setup.png` now shows an 8-pieces opening board
  (Read the PNGs — this is the point of the whole fix; do not skip).
- Run `localize` + `score`/`batch-score`. Expected: `unlocalizable` markers gone; localizations
  snap + validate or typed-reject for honest reasons (occlusion, camera pan, etc.).
- Update the meta contract if Step 3 changed `GameFrames` shape (baseline Optional).

Acceptance: ≥1 of the 3 real-video games produces an accepted, validated opening record in
`provisional_openings.jsonl`. If all 3 still fail, STOP and report the new failure reasons —
do not tune thresholds ad hoc.

### Step 5 — Yield measurement + the pre-registered gate (~overnight compute, CPU-capped)

- Expand to **10–15 Tier-5 videos** via the existing batch machinery. **Cap OCR workers ≤ 3**
  while the v10 training run is live on this machine (check `pgrep -f scripts/train.py`; if it
  has finished — ETA was ~Jul 13 midday — use full width).
- Compute: games/video, yield (accepted/seen), recovered games/video.
- Fill the report placeholders (`<!-- YIELD_TABLE -->`, `<!-- RECOVERED -->`, `<!-- GATING -->`,
  `<!-- DECISION -->`) in `docs/plans/vlm_spike_report.md` with measured numbers and the
  **pre-registered decision rule applied verbatim** (cell = games/video × 204 × yield × 0.6;
  ≥147 CONTINUE; 100–147 USER-CALL; <100 ARCHIVE). Do not soften the rule; do not average away
  bad videos.
- Commit: `feat(human_data): tier-5 yield rerun post setup-parse fix + gating verdict`.

### Step 6 — STOP: user decision point

Present the gating cell + per-video table. **Building the production VLM front-end / gold gate /
full 204-video harvest is the user's call per the pre-registered rule — do not start it.**

---

## 4. Execution constraints (all steps)

- **v10 training is LIVE on MPS** (`selfplay_v10`, PID via `pgrep -f scripts/train.py`, ETA ~Jul 13
  midday). Never touch `src/catan_rl/ppo/`, `src/catan_rl/search/`, or anything training-path;
  cap CPU-parallel OCR at ≤3 workers while it runs.
- All changes are in `src/catan_rl/human_data/` + its tests + `scripts/vlm_spike.py`. Nothing else.
- Fail-closed is the house style: every new inference path must reject typed, never guess. New
  reject reasons are additive constants.
- Test-first per step; `ruff` + `mypy --strict` + `pytest` green before every commit; conventional
  commits, lowercase, <72 chars; push `origin/main` directly (no PRs); **no AI co-author trailers**.
- Stage files by explicit path only — the tree may carry other sessions' scratch.
- Keep `docs/plans/vlm_spike_report.md` truthful as behavior changes (docs-sync rule); no new docs
  beyond filling the report placeholders.
- The review-and-resolve loop (CLAUDE.md) applies: after Steps 1–3 land, run the senior-RL/SWE
  workflow review over the combined diff, resolve BLOCKER/SHOULD-FIX, re-green, then proceed to
  Step 4.

## 5. What NOT to do

- No glyph/template classification of the piece icons (parity makes it unnecessary for setup;
  `built_any` needs no subtype). If parity-validation rejects >30% of otherwise-clean games,
  flag it — icon classification via the existing `detect_glyph_boxes` infra is the pre-approved
  fallback, but it is a separate slice, not scope creep inside Step 2.
- No dice-log work (`record.py` dice contract) — known-unproducible (audit item), not on the
  openings-corpus critical path.
- No changes to the VLM localizer prompt/protocol — it already scored 8/8; the bug was never
  perception.
- No re-tuning of segmentation/welds — landed and working.

## 6. Timeline expectation

Steps 0–3 ≈ one focused day. Step 4 same day. Step 5 overnight (CPU-capped) or full-width after
v10 finishes. **Corpus gating decision in hand ~48 h from start** — the "two days" the corpus was
originally budgeted, now with the actual blocker identified.
