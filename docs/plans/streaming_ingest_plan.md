# Streaming ingest — cut per-video RAM from ~7.5 GB to <1.5 GB so the sweep can run beside v11

**Date:** 2026-07-15 · **Author:** Fable (plan) · **Executor:** Opus, unattended
**Why:** the parallel sweep (commits `01bd8f8`, `7d22501` — cache + pinned workers, both proven)
was killed to protect v11: each worker holds a whole video's ~1200 decoded 1080p frames in RAM
(~6.2 MB × 1200 ≈ 7.5 GB) on a 16 GB box where v11 already occupies ~8.8 GB of swap. Five workers
drove free disk 13 → 6.2 GB (v11's hard guard is 5) and v11's cadence 120 → 257 s/update. The fix
is architectural: **pixels are only ever needed for a few dozen frames per game — decode pixels
twice (cheap), hold them never.**

## The design (locked — do not redesign)

Today (`harvest._ingest_two_pass` → `_extract_context` → `_route_frames_to_games` →
`_read_game_inputs`): ALL frames' pixels live in RAM from decode to the end of Stage-2.
Audit of actual pixel consumers:
- OCR (`crop_log` + `ocr_log_crop`) — consumed ONCE per frame at ingest (already cached).
- Routing / `_post_setup_frame` / `_grant_dense_windows` — **metadata only** (ts + per-frame
  lines/events); verified: no pixel access anywhere in the routing path.
- `GameFrames` consumers — the ONLY pixel users after ingest: `_stable_board_for_game`
  (setup frames + fallback pool), `detect_openings_result` (post_setup + baseline),
  `_consensus_grant` (grant frames' log crops), vlm_spike's PNG export (post_setup + baseline).

**Two-phase ingest with targeted re-decode:**

1. **Phase A — streaming OCR (no pixel retention).** Iterate `decode_frames_at` as a GENERATOR
   (do not `list()` it); for each frame: OCR via the existing crop-hash cache, keep a lightweight
   `FrameMeta(ts, pass_name, native_resolution, lines)`, and DROP the pixels. Dense pass
   (`_grant_dense_windows` from the sparse grant hits) streams the same way. Peak: ~1 frame.
2. **Phase B — route on metadata.** `_route_frames_to_games` and `_post_setup_frame` operate on
   `FrameMeta` (they already only use ts/lines/identity — change type hints, not logic). The
   routing outputs per game: baseline ts, setup ts list, post-setup ts, grant ts list.
3. **Phase C — targeted re-decode, then delete the video.** Build ONE `decode_frames_at`
   schedule = the union of every game's SELECTED timestamps, with caps: setup frames ≤ 8/game
   (board stability needs ≥2), grant frames ≤ 60/game evenly sampled across each grant window
   (consensus measured 5-readable-of-6 at 1 s density; 60 is generous), baseline 1, post-setup 1.
   Materialize `GameFrames` from these (~15–75 frames/game ⇒ ~0.1–0.5 GB/video), THEN delete the
   tmp video (`finally:` — deletion must survive exceptions in any phase; the download-then-delete
   contract is sacred).

Structural home: a new `_ingest_route_and_materialize(video_id, ...) -> tuple[VideoContext-inputs]`
(or refactor `_ingest_two_pass` + `_extract_context` so the tmp video's lifetime spans routing —
pick the cleaner cut, but `parse_video` AND `scripts/vlm_spike.py prepare-frames` must both go
through the SAME new path). Keep the old functions working for existing unit tests, or migrate the
tests — do not leave two divergent live paths.

## Pins (non-negotiable, test-first)

1. **End-to-end identical-output pin on a REAL video**: `scratchpad/probe_work/9Sm86ml04aI.mp4`
   (cached; if gone, re-download once). Run the OLD path (sequentially, standalone it fits RAM)
   and the NEW path; assert per-game meta equivalence: `board_hexes`, `granted_resources`,
   `grant_diag[*].fail/accepted_by`, `winner`, `draft_order`, `log_setup_sequence`,
   `dice_values_readable`, and that the post-setup frame's ts is identical. (Byte-identical PNGs
   for post_setup/baseline follow from identical ts — assert ts, not bytes.)
2. **Peak-RSS measurement** on that video via `resource.getrusage(RUSAGE_SELF).ru_maxrss` logged
   at the end: NEW path < 2 GB (expect ~1–1.5 GB incl. easyocr model).
3. **Unit tests**: metadata routing equivalence (synthetic FrameMeta vs today's DecodedFrame
   objects produce identical buckets/selections); grant-frame cap sampling is even across the
   window and deterministic; video deleted on the happy path AND on an injected Phase-C exception.
4. All existing human_data tests stay green (526+); ruff + mypy --strict.

## Then: relaunch the sweep + the overnight chain

- Sweep: `--workers 3 --videos 40` (3 workers × ~1.5 GB fits beside v11's footprint; do NOT use 5
  until v11 finishes), same skip-list as before (the 10 measured videos), detached, disk-guard 8 GB.
- After the FIRST completed video: verify `[ocr-cache]` line, in-worker `torch.get_num_threads()=1`,
  worker RSS < 2 GB (`ps -o rss= -p <pid>`), and v11 cadence within ~140 s/update over 3 updates
  (baseline ~120 s). Any breach → drop to `--workers 2`; breach again → kill sweep, report,
  leave v11 alone. v11 is the priority; the sweep is recoverable.
- Small add-on: locate the FIX-C tracer's output in the scratchpad (it completed unread —
  `eHIdnu4NjEA` zero-glyph-box attribution) and include its finding in the final report. Read-only.

## House rules

Fail-closed sacred; no accept/reject decision may change (pin #1 is the proof). Never touch
`glyph_anchor.py`, `src/catan_rl/ppo/`, `src/catan_rl/selfplay/`. Test-first; green before every
commit; explicit-path staging; conventional commits; push origin/main; no AI trailers. STOP and
report instead of improvising if: the routing path turns out to touch pixels somewhere this plan
missed, the identical-output pin cannot be satisfied, or Phase C's re-decode diverges on ts
alignment (decode_frames_at must return the same frame for the same ts — if it is not exactly
deterministic, quantify the drift before proceeding).
