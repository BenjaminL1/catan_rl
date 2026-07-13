# Overnight Plan — finish the instrument, measure yield, START BUILDING THE CORPUS

**Date:** 2026-07-13 · **Author:** Fable (planning) · **Executor:** Opus, unattended overnight
**Prior context:** `docs/plans/setup_parse_recovery_plan.md` (Steps 0–4 DONE — commits `9d71c1f`,
`5183991`, `029304a`, `55caf01`, `899b6c9`; 2/3 hand-verified real games ACCEPTED).
**Goal by morning:** the dense-pass fix landed + validated, a trustworthy Tier-5 yield + gating
cell in the report, and a **first provisional opening corpus** (every localizable game from the
swept videos, VLM-localized, glyph-corroborated, accepted through the fail-closed gate).

---

## 0. Why this is the LAST fix (proven, not hypothesized)

The remaining wall is measured to one mechanism. On `9Sm86ml04aI` (6 games, `clean_opening 6/6`,
`localizable 0/6`):

- Grant consensus (`consensus_granted_glyphs`) needs **≥2 readable frames that agree exactly**
  (`MIN_GRANT_CONSENSUS_FRAMES = 2`, fail-closed — correct, do NOT weaken it).
- At the deployed **4 s sparse cadence**, rayman147's grant line is sampled in **exactly 2**
  frames (t=246, 250); at t=246 the line box is found but **`detect_glyph_boxes` returns 0** →
  1 readable < 2 → `None` → `glyph_unreadable` → every game of the video dies.
- At **1 s cadence** over the same window: 6 frames, **5 readable, unanimous**
  `{BRICK, SHEEP, ORE}` → consensus succeeds. (Probe transcript:
  scratchpad `grant_final.log`; scripts `grant_probe.py` / `grant_window.py` / `grant_final.py`.)

The two-pass design already exists (`ingest.build_sampling_schedule(dense_windows=…)`,
`DEFAULT_DENSE_INTERVAL_S = 1.0`) — **`harvest._ingest` simply never passes `dense_windows`.**
Wire it. Everything downstream is mechanical.

Two prior false diagnoses are already committed as legitimate hardening but are NOT the cause
(fuzzy handle match `55caf01`; the scroll-timing story was wrong). Do not revisit them.

---

## STEP A — wire the dense pass (the fix) — ~2 h

**Design (locked — do not redesign):** dense-decode ONLY around grant lines, discovered from the
sparse pass's own OCR. All changes in `src/catan_rl/human_data/harvest.py`; `ingest.py` is NOT
modified (it already exposes `download_video`, `decode_frames_at`, `build_sampling_schedule`,
`TimeWindow`). Avoid double-OCR by computing each frame's log lines ONCE and passing them through.

1. **Pure window derivation** (unit-testable, no I/O):

   ```python
   def _grant_dense_windows(
       grant_ts: Sequence[float], duration_s: float,
       pad_s: float = 10.0,
   ) -> list[TimeWindow]:
   ```
   Input: timestamps of sparse frames whose OCR carries `"received starting resources"` (ANY
   handle). Expand each to `[ts−pad, ts+pad]`, clamp to `[0, duration_s]`, merge overlaps,
   return sorted. ~10–40 s of dense sampling per game ⇒ ≈ +10–15 % OCR cost per video.

2. **Two-pass ingest** — new `_ingest_two_pass(video_id, *, download_gate, work_dir)
   -> tuple[list[DecodedFrame], list[list[str]]]`:
   - `download_video(video_id, dest_dir)` into a tempdir (mirror `ingest_video`'s
     tempdir/`finally`-cleanup semantics — the video must be deleted on EVERY path).
   - sparse schedule (`build_sampling_schedule(duration_s)`), `decode_frames_at`, then OCR each
     sparse frame once: `lines = ocr_log_crop(crop_log(f.frame))`.
   - `grant_ts` = ts of sparse frames whose lines match the grant phrase; build windows via
     `_grant_dense_windows`; if no grant seen, skip the dense pass (nothing to rescue).
   - dense schedule = `build_sampling_schedule(duration_s, dense_windows=…)` **minus** already-
     decoded sparse timestamps (the builder de-dups shared ts itself — verify, don't assume);
     decode + OCR the dense frames.
   - merge (frames, lines) sorted by `ts`; delete the video; return both aligned lists.
   - Hold `download_gate` around the network phase ONLY (same discipline as `_ingest`).

3. **Thread the precomputed lines**: `_extract_context(video_id, frames, per_frame_lines=None)` —
   when supplied, skip its own OCR loop (`harvest.py:464`). `parse_video` switches to
   `_ingest_two_pass` and passes both. Keep `_ingest` working (other callers/tests) but route
   `parse_video`, `scripts/vlm_spike.py prepare-frames`, and `scripts/dev/tier5_yield_rerun.py`
   through the two-pass path.

4. **Tests (write first):** `_grant_dense_windows` (merge/clamp/empty/pad); `_extract_context`
   honours precomputed lines (mock OCR must NOT be called); a routing test that dense-tagged
   frames interleave by ts and grant_frames pick them up. Re-green `ruff` + `mypy` + full
   `pytest`; commit `feat(human_data): wire the dense sampling pass around grant lines`;
   push origin/main. **Never `git add -A`** (other sessions' scratch may be in the tree).

## STEP B — validate on the known-bad video (the A/B) — ~1 h compute

`PYTHONPATH=src python3 scripts/dev/tier5_yield_rerun.py --videos 1` (fresh out-file) so video 1 =
`9Sm86ml04aI`. **Acceptance: `grants_readable ≥ 4` of its 6 games** (was 0/6; some games may
honestly fail for other reasons). If still 0/6, STOP the plan and dump the per-step consensus walk
(reuse `grant_final.py`) — do not proceed to the sweep on a broken instrument. A cached copy of
this video exists at scratchpad `probe_work/9Sm86ml04aI.mp4` for cheap iteration.

## STEP C — the yield sweep + gating numbers — ~6–8 h compute (run detached, sequential)

- `--videos 10` (skip `33KR75rhTgo`, `AoOXWyxaTkA` — already hand-measured), detached via
  `nohup … & disown`, log to scratchpad. ~40–60 min/video (OCR-bound). yt-dlp occasionally
  throws a JS-runtime/format error — the probe already records per-video errors and continues;
  a failed video is fine, note it.
- While it runs, do STEP D on the two already-prepared videos.
- When done, fill the FOUR placeholders in `docs/plans/vlm_spike_report.md`
  (`YIELD_TABLE`, `RECOVERED`, `GATING`, `DECISION`) with measured numbers:
  - yield = `localizable / games` over swept + hand-measured videos (state both separately);
  - `cell = games/video × 204 × yield × 0.6` — apply the pre-registered rule VERBATIM
    (≥147 CONTINUE / 100–147 USER-CALL / <100 ARCHIVE). **Do not soften, do not average away
    bad videos, label the localizable→accepted assumption** (accepted-when-localizable held 3/3
    by hand; say so explicitly).
- Commit the report update. This is the gate INPUT for the user — the CONTINUE/ARCHIVE **decision
  itself remains the user's**; building tonight's corpus below is explicitly authorized
  regardless ("begin building the corpus"), as provisional/seed-eligible data.

## STEP D — BEGIN BUILDING THE CORPUS (the deliverable) — rest of the night

For every game with `clean_opening ∧ grants_readable` (from the sweep's JSON + the 3 existing):

1. `PYTHONPATH=src python3 scripts/vlm_spike.py prepare-frames --video <id>` (persists
   `post_setup.png`, `empty_baseline.png`, `meta.json`). Reuses the same ingest — after STEP A
   this includes dense grant frames automatically.
2. **Localize (YOU are the VLM — this is the spike's documented Opus-vision protocol):**
   - Generate a vertex-id overlay to read against: port the scratchpad tooling into
     `scripts/dev/localize_overlay.py` (committed, reusable): fit the board via
     `read_board_stable([baseline, baseline.copy()])`; if that fails, 3-point affine from three
     unambiguous number-token centres (scratchpad `overlay2.py` shows both). Read the overlay PNG.
   - Read piece positions/colours from `post_setup.png` (zoom crops help; scratchpad
     `board_zoom.py` pattern). Emit `data/human/vlm_spike/localized/<video>__g<n>.json`
     (FileLocalizer schema — hex-adjacency only, never engine ids).
   - **MANDATORY self-check before writing:** each player's `granted_resources` multiset from
     meta.json must equal the adjacent-resource multiset of EXACTLY ONE of their two settlements
     (compute via `topology.vertex_adjacent_hexes` + meta `board_hexes`). Mismatch ⇒ re-examine
     the frame; still mismatched ⇒ write `{"unlocalizable": "<specific reason>"}` — never guess.
     This corroboration held 3/3 on the hand-verified games and is what makes an unattended
     corpus trustworthy.
3. `PYTHONPATH=src python3 scripts/vlm_spike.py localize <video>__g<n>` → ACCEPTED or typed
   reject. Investigate the FIRST unexpected reject before continuing (a systematic one means an
   instrument problem; isolated ones are honest).
4. **Collect:** new `scripts/dev/collect_corpus.py` — read every `data/human/vlm_spike/records/
   *.json` with `accepted: true`, append `record.to_json_line()` to
   `data/human/corpus/provisional_openings.jsonl` (dedup on `(video_id, game_index)`; stable
   order; idempotent). Everything is PROVISIONAL + seed-eligible-only
   (`placement_order_established` stays False — the log-side ordinal is a PENDING USER DECISION;
   do NOT relax it). Commit records + localized JSONs + corpus file + the two new scripts.
5. Track a running tally in the final summary: games seen / localizable / localized / accepted,
   per video.

Localization is careful vision work (~15–30 min/game). Prioritize breadth (≥1 game/video) then
depth. **Every accepted game tonight is corpus data the project has never had.**

## STEP E — side task: v10 completion gate (cheap, do not skip)

v10 (`runs/train/selfplay_v10_20260711_210552`, 900 updates) finishes overnight. Arm a watcher
(`until ! pgrep -f scripts/train.py; do sleep 300; done` in background). When it exits:
1. Pick the candidate exactly as v9 was picked: promoted-anchor-era checkpoint beats the final
   one if `selfplay/anchor_wr_window` (TB) ended < ~0.55 vs the last anchor; else the final ckpt.
   (v9 precedent: ckpt_424 > ckpt_609 because the window ended ~0.507 and falling.)
2. `PYTHONPATH=src nohup python3 scripts/ladder_gate.py --candidate-ckpt <ckpt>
   --candidate-name v10_cand --baseline v9_chain_u424 --nps 300 --workers 6
   --out runs/search/v10_ladder_gate.json` (CPU is free once training ends).
3. Report the clause-1 verdict in the morning summary. **Do not bank/crown anything** — that is
   the user's call (v9 precedent: user crowned on a withheld gate).
Also: once training exits, the OCR/sweep worker cap is lifted — sweep legs may run full width.

## Constraints (all night)

- **While v10 trains:** OCR/eval work stays CPU-frugal (sequential sweep, no extra worker fan-out).
  Never touch `src/catan_rl/ppo/`, `src/catan_rl/search/`, or any training path.
- Fail-closed is the house style: never weaken `MIN_GRANT_CONSENSUS_FRAMES`, the unanimity rule,
  the glyph anchor, or the snap's 0-or->1 reject. Yield comes from MORE FRAMES, not looser gates.
- Test-first per step; `ruff` + `mypy --strict` + `pytest` green before every commit;
  conventional commits (<72 chars, lowercase); push `origin/main`; **no AI co-author trailers**;
  stage by explicit path only.
- Docs-sync: update `vlm_spike_report.md` (placeholders) and append a short "Implementation
  notes" to `setup_parse_recovery_plan.md` (what diverged: parity dropped, dice contract, fuzzy
  handles, dense pass). No new docs beyond the two scripts' docstrings + this plan's outputs.
- **STOP conditions:** STEP B fails its acceptance → stop, write up the consensus walk, leave the
  sweep unlaunched. A systematic (≥3 consecutive) unexpected reject class in STEP D → stop that
  video, document, continue with others. Disk < 5 GB → stop writing corpus artifacts, report.
  (Currently ~17 GB free; sweeps hold ~1–2 GB transient video each, deleted after use.)

## Morning deliverables (the summary the user wakes up to)

1. Dense-pass commit(s) + STEP B A/B numbers (`9Sm86ml04aI`: 0/6 → N/6 grants_readable).
2. The filled yield table + the gating cell + the rule's verbatim verdict band.
3. `data/human/corpus/provisional_openings.jsonl` — the first N accepted human openings, with the
   per-video tally and every reject reason.
4. v10 gate verdict (or training-still-running status).
5. The two standing user decisions, restated: (a) scoreboard eligibility via glyph-anchor-only
   ordering vs log de-dup; (b) CONTINUE/USER-CALL/ARCHIVE on the gating cell.
