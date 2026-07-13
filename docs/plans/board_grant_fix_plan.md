# Board + Grant Instrument Fixes — the two bugs gating the corpus

**Date:** 2026-07-13 · **Author:** Fable (diagnosis + plan) · **Executor:** Opus, no further direction
**Context:** overnight run (`docs/plans/corpus_overnight_plan.md`) built the first corpus
(6 rows) and measured the 8-video yield: **14 games → 3 localizable (21%)**. Every loss is one of
two instrument bugs. Raw gating cell = 46 (ARCHIVE band); with these fixed, ~270 (clear CONTINUE).
**These two fixes decide the corpus program.**

Ledger (from `runs/human/corpus_prep.json` + per-game `meta.json`):
`board_unreadable` ×6 · grant-consensus ×5. The VLM/localization step is 14/14 — never the problem.

**House rules that bind every step below:** fail-closed is sacred — no gate is weakened, ever;
every fix must be ADDITIVE-ACCEPTING (anything accepted today stays accepted with a byte-identical
result; new accepts only where evidence proves the reject was an instrument artifact). Test-first;
`ruff` + `mypy --strict` + `pytest` green before each commit; conventional commits; push origin/main;
stage by explicit path; no AI trailers. If v10 training is still running (`pgrep -f scripts/train.py`),
keep sweeps sequential. THE DISCIPLINE THAT WORKED: instrument → measure → branch on evidence. Three
plausible-looking hypotheses were wrong this session; zero measured ones were.

---

# BUG 1 — `board_unreadable` (6 games lost; the bigger prize)

## Evidence in hand (verbatim, from tonight)

Probing the 6 failing games' SAVED frames (`post_setup.png` / `empty_baseline.png`) with
`_detect_tokens` + `read_board`:

```
0EtcbG16kHA__g1   post_setup tokens=17 read_board=OK    baseline tokens=18 read_board=OK    <- BOTH OK!
9Sm86ml04aI__g3   post_setup tokens=17 read_board=OK    baseline tokens=18 read_board=OK    <- BOTH OK!
9Sm86ml04aI__g5   post_setup tokens=19 read_board=None  baseline tokens=18 read_board=OK    <- baseline OK
9Sm86ml04aI__g4   post_setup tokens=17 read_board=None  baseline tokens=18 read_board=None
9Sm86ml04aI__g2   post_setup tokens=19 read_board=None  baseline tokens=19 read_board=None
9Sm86ml04aI__g6   post_setup tokens=19 read_board=None  baseline tokens=19 read_board=None
```

Two proven sub-causes:

**1a — WRONG FRAMES FED TO THE STABLE READ (≥2, likely 3, of 6 games).** The pipeline reads the
board from `setup_frames = bucket[:max(2, len(bucket)//2)]` (`harvest._route_frames_to_games`).
For any game after the video's first, the earliest bucket frames are routinely the PREVIOUS game's
end screen / lobby / "Well Played!" overlay — so `read_board_stable` never sees the board even
though the game's own saved frames read perfectly. `0EtcbG16kHA__g1` and `9Sm86ml04aI__g3` are
smoking guns (both saved frames OK, game still failed); `g5`'s baseline reads OK too.

**1b — 19 DETECTED TOKENS (3 of 6 games' frames).** `read_board`'s first gate requires 16–18
number-token disks; several frames detect **19**. One false-positive disk somewhere on screen
(scoreboard bubble / avatar — identify, don't assume) both trips the count gate and, if kept,
would poison the RANSAC lattice fit. `g2`/`g6` fail this way on BOTH saved frames.

Unknown residue: `g4` (17/18 tokens, still None on both) — some later gate; the diag in Step 1
will name it. Do NOT guess it.

## Step 1 — instrument `read_board` (the observability hole)

`read_board` returns bare `None` from ~10 gates. Mirror the grant-diag pattern exactly
(`harvest._consensus_grant(..., diag=...)`, commit `7151308`):

- `read_board(frame, *, ..., diag: dict[str, Any] | None = None)`. On every early return, set
  `diag["fail"]` to a stable slug — `token_count` (also record `diag["n_tokens"]`), `no_affine`,
  `screen_rule_gap` (record the gap), `residual` (record it), `desert_not_bimodal`,
  `hue_margin`, `nonstandard_multiset`, `ocr_pip_mismatch` (record which hex),
  `too_few_anchors`, `anchor_disagrees`. On success set `diag["fail"] = None`.
- `read_board_stable(frames, *, diag_list: list[dict] | None = None)` appends one diag per frame
  and records `diag_list`-level `fail="cross_frame_disagreement"` when ≥2 reads disagree.
- Tests: synthetic — a frame producing each of 2–3 easily-forced gates (token_count via a blank
  image; cross-frame disagreement via two different valid boards — reuse existing test fixtures in
  `tests/unit/human_data/test_board_cv.py` which already construct readable boards).

Commit: `feat(human_data): read_board rejection diagnostics`.

## Step 2 — measure: reject-gate histogram over the 6 failing games (no downloads needed)

Inline script (scratchpad, not committed): for each of the 6 games, run the diag'd
`read_board` over `post_setup.png` + `empty_baseline.png` and print the histogram of
`diag["fail"]`. Also, for every frame with `n_tokens == 19`: dump the 19 token (x, y, r) values
(`_detect_tokens` returns them) and save a crop around the token FARTHEST from the token-cloud
centroid — Read the image and IDENTIFY the false disk (score bubble? avatar? port icon?). Record
findings in the PR/commit message. **Gate: do not proceed to Step 4 until the extra disk is
visually identified.**

## Step 3 — fix 1a: widen the stable-read frame pool (no CV changes at all)

In `harvest._read_game_inputs`: today `board = read_board_stable([f.frame for f in gf.setup_frames])`.
Replace with a candidate-pool fallback, PRE-REGISTERED order, same reader, same agreement rule:

1. `setup_frames` (unchanged first try — preserves today's accepts byte-identically);
2. on `None`: `[gf.empty_baseline, post_setup_frame.frame] + [f.frame for f in gf.grant_frames]`
   (all frames of THIS game, by construction) — `read_board_stable` still demands ≥2 byte-identical
   agreeing reads from the pool, so the fail-closed §5.2 stability semantics are untouched;
3. still `None` → `BOARD_UNREADABLE_REASON` exactly as today.

The vlm_spike `prepare_frames_from_video` path computes the board the same way — route both call
sites through one helper `_stable_board_for_game(gf) -> BoardRead | None` so they cannot diverge.

Tests: monkeypatched-stage test where setup_frames are junk but baseline+post_setup carry a valid
board → game proceeds; junk everywhere → still rejected. (Real read_board is exercised by the
existing board_cv suite; this tests the ORCHESTRATION fallback.)

Commit: `fix(human_data): stable board read falls back to the game's own clean frames`.

## Step 4 — fix 1b: 19-token false positive (branch on Step 2's identification)

- **If the extra disk is OFF-BOARD** (score bubble / HUD / avatar — expected): add a
  pre-registered outlier trim in `read_board`, ACTIVE ONLY when `len(tokens) > 18`: compute the
  token-cloud centroid and median token-to-centroid distance; drop tokens with distance
  > `3.0 x` median (record how many in `diag["outliers_dropped"]`); proceed only if the remainder
  lands in 16–18, else reject exactly as today. ≤ 2 drops max — more means the frame is genuinely
  ambiguous, reject. All downstream gates (RANSAC, residual, multiset, OCR-pip, anchors) unchanged
  and still able to reject. This is additive-accepting: frames with ≤18 tokens take today's path
  byte-identically.
- **If the extra disk is ON the board** (e.g. a piece rendered disk-like): STOP, write up, do not
  trim — an on-board false disk means the detector's radius/colour bands need work that must not
  be tuned blind. (Not expected given g1 of the same video read 17–18.)

Tests: synthetic token sets — 19 tokens with 1 far outlier → trimmed, accepted downstream; 19
tight tokens → still rejected; 18 tokens → path unchanged (assert the trim never ran via diag).

Commit: `fix(human_data): trim off-board token outliers (>18 disks) before lattice fit`.

## Step 5 — validate BUG 1 end-to-end

Re-run `PYTHONPATH=src python3 scripts/vlm_spike.py prepare-frames --video 9Sm86ml04aI` (cached at
scratchpad `probe_work/9Sm86ml04aI.mp4`; copy into the workdir the same way earlier probes did or
just let it re-download) and `--video 0EtcbG16kHA`. **Acceptance: ≥3 of the 6 previously
board-failed games now carry `board_hexes`** (expected: `0EtcbG16kHA__g1` + `9Sm86ml04aI__g3` via
Step 3; `g2`/`g6` via Step 4; `g5` via either). REGRESSION: `9Sm86ml04aI__g1`'s `board_hexes` must
be BYTE-IDENTICAL to before (compare against the committed corpus row's board).

---

# BUG 2 — grant consensus (5 games lost)

## Evidence in hand (verbatim)

Sub-mode A — `reads_disagree` (the dense pass exposed it; counts from `0EtcbG16kHA__g1`):

```
LevyChevy : 48 readable reads -> {BRICK:2,SHEEP:1} x47  vs  {BRICK:2,SHEEP:1,ORE:1} x1   47-vs-1
ThePhantom:  9 readable reads -> {BRICK,WHEAT,WOOD} x6  vs  {BRICK,ORE,WOOD}       x3    6-vs-3
```

`consensus_granted_glyphs` demands STRICT UNANIMITY over ≥2 reads. That rule is incoherent at
scale: it accepts a 2-of-2 read but rejects 47-of-48 — far stronger evidence. 47-vs-1 is OCR noise
around one true multiset; **6-vs-3 is genuinely ambiguous and MUST keep failing closed** (one-card
WHEAT↔ORE confusion at 67% is not evidence).

Sub-mode B — `line_found_but_no_glyph_boxes` (`eHIdnu4NjEA__g1`, ThePhantom):

```
grant_frames=99, line_found=20, with_glyph_boxes=0
```

The line is FOUND in 20 frames and `detect_glyph_boxes` returns `[]` in ALL 20 — systematic,
per-video. Suspects (docstring-derived, NOT verified): the merged-box fail-closed rule (a merged
suspect no-reads the whole frame) tripping on this video's icon scale, the fixed pixel bands
`MIN_ICON_CELL_W`/`MAX_ICON_CELL_W` rejecting pitch-split cells, or the wrap-row band missing the
icons. **Measure before fixing** (Step G2).

## Step G1 — the dominance rule (evidence complete; implement directly)

In `glyph_anchor.consensus_granted_glyphs`, replace the accept condition with a PRE-REGISTERED
two-clause rule — clause 1 is EXACTLY today's behaviour, so this is additive-accepting:

- **ACCEPT** iff `len(reads) >= MIN_GRANT_CONSENSUS_FRAMES` and ALL reads agree  *(today's rule)*
- **OR** `len(reads) >= 5` and the modal read covers ≥ 90% of reads
  (`DOMINANT_READ_MIN_READS = 5`, `DOMINANT_READ_MIN_FRAC = 0.9`, module constants with a
  docstring citing the 47-vs-1 / 6-vs-3 measurements and why 6-vs-3 must stay rejected).

Checks on the measured cases: 47/48 = 0.979 → accept (returns the modal multiset); 6/9 = 0.667 →
reject; 2-of-2 unanimous → accept (unchanged); 3-vs-2 → reject (n<5 for the dominance clause and
not unanimous). The accepted multiset flows through the UNCHANGED downstream anchor
(`identify_granting_settlement` — must match exactly ONE of the player's two settlements), which
remains the joint-flip defence; this rule only decides *which multiset to test*, never bypasses it.

Tests (in `tests/unit/human_data/test_glyph_anchor.py`, note module-level
`pytest.importorskip("cv2")` convention — mock `classify_granted_glyphs` inputs the way existing
consensus tests do): 47-vs-1 accepts modal; 6-vs-3 rejects; 4-vs-0 accepts (unanimity); 4-vs-1
rejects (n=5, 0.8 < 0.9); 2-of-2 accepts; 1 read rejects. Also assert the returned Counter is the
MODAL one, not the first.

Commit: `fix(human_data): grant consensus accepts a >=90% dominant read (n>=5)`.

## Step G2 — measure sub-mode B, then branch

1. Add a `diag` out-param to `detect_glyph_boxes` (same pattern): record candidate counts at each
   stage — `n_mask_components`, `n_icon_sized`, `n_on_line_or_wrap`, and which fail-closed exit
   fired (`merged_suspect`, `pitch_split_fail`, `no_candidates`), plus the computed pitch and cell
   widths when a split was attempted. Commit with a trivial unit test (empty crop → `no_candidates`).
2. Re-run `prepare-frames --video eHIdnu4NjEA`; read `grant_diag` from the meta. Branch:
   - **`merged_suspect` / `pitch_split_fail` with cell widths just outside
     [`MIN_ICON_CELL_W`, `MAX_ICON_CELL_W`]** → make the band scale-relative: derive it from the
     grant LINE HEIGHT (`line_box`), e.g. width ∈ [0.55, 1.4] × line-height — compute the exact
     factors from the MEASURED cell widths of the 5 accepted games' frames (print them in the same
     diag run; the band must bracket all known-good widths with margin and still exclude 2-icon
     merges). Keep the absolute band as a sanity clamp OUTSIDE which we still reject.
   - **`no_candidates` (mask finds nothing)** → the panel-background colour-distance mask fails on
     this video's skin → dump the crop + median background colour, compare against a working
     video's, and extend the mask threshold ONLY if the two skins measurably differ; otherwise STOP
     and write up.
   - Anything else → STOP and write up with the diag numbers. Do not improvise.
3. Acceptance: `eHIdnu4NjEA__g1` grants become readable (both players) OR a written diagnosis of
   why it is genuinely unreadable (e.g. the icons are truly occluded — then it is honest data loss,
   not a bug).

## Step G3 — validate BUG 2 end-to-end

Re-run `prepare-frames` for the 5 grant-failed videos (`lCsB4X60YhQ`, `nmk59XWFRBU`,
`KvH76fJI4f0` (g2), `eHIdnu4NjEA`, `yQ4GZloiG08` — plus `0EtcbG16kHA` if not already redone in
Step 5). **Acceptance: ≥3 of the 5 become grant-readable.** REGRESSION: the 3 sweep-accepted games'
`granted_resources` must be UNCHANGED (byte-compare against current meta.json values;
`9Sm86ml04aI__g1`: ThePhantom `{ORE:2,SHEEP:1}`, rayman147 `{BRICK:1,ORE:1,SHEEP:1}`).

---

# FINAL — re-measure, extend the corpus, refresh the numbers

1. With both fixes landed: re-run the tally (`scripts/dev/prepare_corpus_videos.py` skips prepared
   videos — pass `--skip` nothing and let it recompute `tally()` per video, or re-tally inline) and
   produce the updated per-video table (games / localizable / failure modes).
2. **Localize every newly-localizable game** using the documented VLM protocol
   (`scripts/dev/localize_overlay.py` → read the overlay → MANDATORY grant self-check: each
   player's grant multiset equals the adjacent-resource multiset of EXACTLY ONE of their two
   settlements → write `localized/<game>.json` → `vlm_spike.py localize <game>` → expect ACCEPTED)
   and re-run `scripts/dev/collect_corpus.py`. Every accept is a corpus row.
3. Update `docs/plans/vlm_spike_report.md`'s `YIELD_TABLE` / `RECOVERED` / `GATING` / `DECISION`
   placeholders with the POST-FIX measured numbers (state pre-fix 21.4% and post-fix side by side;
   apply the pre-registered rule verbatim to the post-fix cell; the CONTINUE/ARCHIVE decision
   remains the user's).
4. Morning summary: corpus row count, the post-fix yield table, both fixes' validation numbers,
   regression checks, and the two standing user decisions (scoreboard eligibility via
   glyph-anchor-only ordering; the gate call).

**STOP conditions:** any regression check fails → revert that commit and write up. A fix branch's
evidence doesn't match any pre-registered case → STOP and write up; do not invent a new threshold.
Disk < 5 GB → stop re-ingesting. Expected effort: Steps 1–4 ≈ 3–4 h, G1 ≈ 1 h, G2 ≈ 1–2 h,
re-measure + localization ≈ 2–3 h compute + reads.
