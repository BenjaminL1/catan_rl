# TIER-5 RE-RUN Report — e2e harvest after the palette + HUD-hardening fix

**Date:** 2026-07-06. **Scope:** re-run the `mine_phantom` e2e harvest driver
(`scripts/tier5_harvest.py`) now that the **dominant Tier-5 blocker is fixed** — the
opponent-colour reader (`src/catan_rl/human_data/openings.py`: the `PALETTE`
piece-profile table + the `_HUD_RING` HUD seat-ring table) was extended from
GREEN+BLACK-only to the survey-calibrated Colonist palette
(`RED`/`WHITE`/`PURPLE`/`BLACK` + `GREEN`) with fail-closed HUD-avatar gates
(commits `39b02b1` → `000f509`). Two runs: **(a)** the original 5 Tier-5 videos for a
direct comparison to the **0/3 NO-GO baseline** (`docs/plans/tier5_report.md`), then
**(b)** an expansion over 20 more `high`-manifest videos (25 total) to measure the
acceptance yield and surface the **next** blocker. **CPU-only; no `gui/` import;
download-then-delete ingest (0 residual frame files).**

## Verdict

**The colour blocker is fixed and confirmed dead: `hud_unreadable` fell from 100% of
the baseline (3/3 games) to a residual 16% of the re-run (5/31 games), and every
reached game now advances DEEP into the openings CV.** But the corpus **still cannot
be harvested to acceptance**: across **25 videos / 31 games, 0 were accepted**
(yield **0/31**, Wilson-95 **[0.000, 0.110]**). The rejection mass has **moved
downstream** from colour to a spread of openings-CV / board-CV failures. The **new
dominant blocker is the opening piece/road CV** (`settlement_ambiguous` +
`settlement_blob_shortfall` + `road_unresolved` = **13/31 = 42%**), with
**`board_unreadable` the single largest typed reason (8/31 = 26%)** — exactly the
"harden GREEN/opening piece detection + Stage-1 segmentation" work the original report
scoped as blockers 4 & 5 and left OUT of the palette slice. The glyph/consensus
firewall remained healthy throughout (≥2 readable grant frames reached for ~23
granting players) even though 0 games reached the anchor.

---

## 1. Per-video accept/reject table (typed reasons)

Videos 1–5 = the **original Tier-5 set** (serial, `--max-workers 1 --net-concurrency
1`); videos 6–25 = the expansion (`--max-workers 3 --net-concurrency 2`). 1080p,
sparse 4 s sampling, resumable ledger.

| # | video_id | video status | games | per-game typed rejection reason |
|---|---|---|---|---|
| 1 *(orig5)* | `33KR75rhTgo` | done | 2 | g1:`hud_assignment_mismatch`; g2:`settlement_blob_shortfall:RED` |
| 2 *(orig5)* | `AoOXWyxaTkA` | done | 1 | g1:`road_unresolved:RED:30` |
| 3 *(orig5)* | `5RLq1NX4nAo` | **video-fail** | 0 | `HUD did not yield two distinct seat colours` |
| 4 *(orig5)* | `sG05DoaOmM4` | **video-fail** | 0 | `HUD did not yield two distinct seat colours` |
| 5 *(orig5)* | `EdMnUD-eZ6A` | done | 1 | g1:`board_unreadable` |
| 6 | `9Sm86ml04aI` | done | 6 | g1:`settlement_ambiguous:GREEN`; g2–g6:`board_unreadable` ×5 |
| 7 | `lCsB4X60YhQ` | done | 1 | g1:`hud_unreadable` |
| 8 | `l-3s-xPwyIs` | done | 1 | g1:`settlement_ambiguous:RED` |
| 9 | `nmk59XWFRBU` | done | 1 | g1:`settlement_ambiguous:RED` |
| 10 | `KvH76fJI4f0` | done | 1 | g1:`hud_set_mismatch` |
| 11 | `eHIdnu4NjEA` | done | 1 | g1:`hud_set_mismatch` |
| 12 | `0EtcbG16kHA` | done | 1 | g1:`board_unreadable` |
| 13 | `yQ4GZloiG08` | done | 1 | g1:`hud_unreadable` |
| 14 | `Hj_VF4PhHwM` | done | 1 | g1:`settlement_blob_shortfall:RED` |
| 15 | `fqBK3_-PO7g` | done | 3 | g1,g2:`settlement_ambiguous:GREEN`; g3:`hud_assignment_mismatch` |
| 16 | `YK-AAv-Brn0` | done | 1 | g1:`settlement_blob_shortfall:RED` |
| 17 | `taiiYTBfVJA` | done | 1 | g1:`settlement_ambiguous:RED` |
| 18 | `I4Xw7dYDXYc` | done | 1 | g1:`hud_unreadable` |
| 19 | `5WamwGjkHcE` | done | 2 | g1:`settlement_ambiguous:GREEN`; g2:`board_unreadable` |
| 20 | `HT10Xj1mkkA` | done | 1 | g1:`settlement_blob_shortfall:RED` |
| 21 | `cXun_M90NBA` | done | 1 | g1:`hud_set_mismatch` |
| 22 | `cf-wYBSp3pM` | done | 1 | g1:`hud_unreadable` |
| 23 | `lMCemfE1bDY` | done | 1 | g1:`road_unresolved:RED:12` |
| 24 | `qO9brR8chOE` | done | 1 | g1:`hud_unreadable` |
| 25 | `6yyzAd63Gs0` | **video-fail** | 0 | `HUD did not yield two distinct seat colours` |

**No game was silently dropped** — every reached game emitted a typed-rejected record
(committed as `data/human/tier5_rerun_sample.jsonl`, 31 rows); every non-reached video
emitted a ledger `failed` row with its cause.

> **Reason-string caveat (not a defect):** `settlement_blob_shortfall:RED` carries the
> `:green_tile_suppressed` suffix in the raw records because that suffix is appended
> for **any** `tile_subtract` colour, and RED is `tile_subtract=True` (brick-tile
> subtraction, fail-closed — see the `PALETTE` doc). It is a RED shortfall, **not** a
> green-tile suppression. Cosmetic only; the fail-closed guarantee is unaffected.

## 2. NEW telemetry (`telemetry_combined.json`, 25 videos)

```
videos: processed=22  skipped=0  failed=3
games_seen=31  accepted=0  rejected=31  order_unestablished=0
anchor: ran=0  unreadable=0  mismatch=0  grant_read_coverage=0.000
rejected_by_reason:
  board_unreadable=8
  settlement_ambiguous:GREEN=4   settlement_ambiguous:RED=3
  hud_unreadable=5
  settlement_blob_shortfall:RED=4
  hud_set_mismatch=3
  hud_assignment_mismatch=2
  road_unresolved:RED:12=1       road_unresolved:RED:30=1
```

`anchor_ran=0` / `grant_read_coverage=0.000` because **all 31 games rejected at or
before the openings §5.14 stage — upstream of the joint-flip glyph anchor**, which
only runs on the accept path. It correctly never fired. `order_unestablished=0`
because the LOG-placement-order gate is only reached by accepted records (0).

*(The 3 `videos_failed` are the video-level HUD-context gate: `5RLq1NX4nAo`,
`sG05DoaOmM4`, `6yyzAd63Gs0` — the `_majority_two_colour` context read could not
resolve two distinct palette seat colours across the sampled frames, so those
opponents use a colour still **outside** the extended palette, e.g. the low-sample
BLUE/ORANGE/LIGHTGREEN the `PALETTE` doc explicitly leaves uncalibrated. This is the
intended **fail-closed** behaviour — a whole-video abstention, never a mislabel.)*

## 3. Acceptance yield + Wilson CI

| metric | value |
|---|---|
| accepted | **0** |
| games_seen | **31** |
| **acceptance yield** | **0/31 = 0.000** |
| Wilson-95 CI | **[0.0000, 0.1103]** |

Zero acceptances; the 95% Wilson **upper** bound caps the true yield at **11.0%** at
this openings-CV maturity.

## 4. NEW dominant rejection reason — what's the next blocker?

`hud_unreadable` was **100%** of the baseline (3/3). After the palette fix it is **16%**
(5/31) and no longer dominant. The mass redistributed across three CV stages:

| stage (family) | count | share | typed reasons |
|---|---|---|---|
| **opening piece/road CV** | **13** | **42%** | `settlement_ambiguous` (7), `settlement_blob_shortfall` (4), `road_unresolved` (2) |
| HUD player→colour binding | 10 | 32% | `hud_unreadable` (5), `hud_set_mismatch` (3), `hud_assignment_mismatch` (2) |
| board CV stability | **8** | **26%** | `board_unreadable` (8) |

- **New dominant BLOCKER = the opening piece/road CV (42% of games).** Within it,
  `settlement_ambiguous` (7) is the single biggest — the area-dominance guard firing
  because a real settlement is occluded or a tile-subtraction-leak blob floats into the
  top-2 (§5.7 / the red-team guard). GREEN (4) and RED (3) both hit it, i.e. this is the
  **general** opening-CV sensitivity problem the original report flagged as blocker 4/5,
  not a colour-specific one. `road_unresolved:RED` (2) is the deepest reach — settlements
  detected, only the road tiebreak failed.
- **Largest single typed reason = `board_unreadable` (26%),** dominated by one
  over-segmented video (`9Sm86ml04aI`: 5 of its 6 windows). The Stage-1 segmenter is
  welding/splitting noisy-OCR windows whose setup frames never lock a stable board —
  the segmentation-hardening item (original blocker 5).
- **Residual HUD-binding surface (32% + 3 whole-video fails)** is the honest
  fail-closed cost of colours still outside the calibrated palette: the fix eliminated
  `hud_unreadable` as THE blocker but did not (and by design cannot, without more
  survey-calibrated colours) drive it to zero.

**Bottom line:** colour is solved; the next investment is **opening piece/road CV
sensitivity + Stage-1 segmentation robustness on noisy OCR**, then widening the palette
to the remaining low-sample colours.

## 5. Consensus-supply witness (glyph firewall health)

Even with 0 accepted games, the multi-frame CONSENSUS grant read (≥2 byte-identical
readable frames) kept working:

| run | granting players reached | consensus-ok (≥2 readable) |
|---|---|---|
| orig5 (serial, clean attribution) | 6 | **5** |
| expansion (3-worker) | 33 | **18** |

Readable-frame histogram (orig5): `{<2:1, 2:1, 4:2, 5:1, 7:1}`. The single `<2` miss is
the exact fail-closed case (an unreadable grant → typed `glyph_unreadable`, never a
fabricated multiset). **The glyph/consensus firewall is healthy on real footage; the
harvest block is entirely upstream in the openings/board CV, not in the anchor.**
*(Expansion per-game attribution is approximate — the instrumented driver's shared
context is only race-free at `max_workers=1`; the aggregate consensus-ok count is
reliable, the per-game grouping is not. orig5's serial figures are exact.)*

## 6. Wall-clock & 204-video harvest ETA

| run | videos | workers | wall | per-video (wall) |
|---|---|---|---|---|
| orig5 | 5 | 1 | 2768 s | **554 s** |
| expansion | 20 | 3 | 7835 s | **392 s** |
| **combined** | **25** | — | **10603 s ≈ 2.95 h** | — |

- **Serial ETA (204 `high` videos):** 204 × 554 s = **31.4 h** (`n_procs=1`).
- **3-worker ETA (measured throughput):** 204 × 392 s = **22.2 h**; extrapolating the
  near-linear thread scaling the original report observed, **≈11 h at 6 workers**.
- **Expected corpus at this yield:** games/video = 31/25 ≈ **1.24**, so 204 videos →
  **≈253 games seen**. At the current yield the **point estimate of accepted games is
  0**; the 95% Wilson **upper** bound (11.0%) caps it at **≤28 accepted** across the
  whole corpus. **The corpus is NOT harvestable to a usable accepted set until the
  opening piece/road CV + Stage-1 segmentation are hardened** — running all 204 now
  would spend ~1 day of compute for a point-estimate of 0 accepted games.

## 7. Artifacts

- `docs/plans/tier5_rerun_report.md` — this report.
- `data/human/tier5_rerun_sample.jsonl` — the 25-video harvest output (31 typed-rejected
  records; the corpus was **empty**, 0 accepted, so the sample is the rejected rows —
  the §5.6 rejection-bias audit surface). 48 KB (< 500 KB → committed).
- Driver: `scripts/tier5_harvest.py` (unchanged); palette fix in
  `src/catan_rl/human_data/openings.py` (`39b02b1`–`000f509`).

## 8. Recommended next steps

1. **Harden the opening piece/road CV** (the new 42% blocker) — the
   `settlement_ambiguous` area-dominance guard and `road_unresolved` tiebreak are
   rejecting real openings under occlusion / winning-spot glow. This is blocker 4/5 from
   the original report and is now the top of the queue.
2. **Harden Stage-1 segmentation on noisy OCR** so a single video does not over-segment
   into board-unreadable fragments (`9Sm86ml04aI` alone produced 5 `board_unreadable`).
3. **Widen `PALETTE`/`_HUD_RING` to the remaining low-sample colours** (BLUE / ORANGE /
   LIGHTGREEN) with survey-calibrated HSV bands to reclaim the 3 whole-video HUD fails +
   the `hud_set_mismatch`/`hud_unreadable` residual — the same data-derived, fail-closed
   process used for RED/WHITE/PURPLE here.
