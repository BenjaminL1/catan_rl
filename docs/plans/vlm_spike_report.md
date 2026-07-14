# VLM Opening-Localization Decision Spike — Report

**Date:** 2026-07-10 · **Harness:** `scripts/vlm_spike.py` · **Data:** `data/human/vlm_spike/`
**Status:** timeboxed decision spike — the decision below is the user's to take.

---

## What was tested (the hybrid, unchanged)

Reuse the **existing** frame extraction (`harvest._extract_context` → `GameFrames`:
the `post_setup_frame` the openings CV reads + the `empty_baseline` frame) → a **VLM**
localizes each settlement/road by **tile adjacency** (which hexes each corner touches;
which two vertices each road joins) → **deterministic geometry**
(`topology.load_topology`) snaps the adjacency to the exact engine vertex (0–53) / edge
(0–71) id, **fail-closed** on any 0-or->1 match → **placement order from the LOG**
(`establish_placement_order`), never the VLM → the **existing fail-closed validators**
(`validate.cross_check`: OpeningResult invariants + joint-flip glyph firewall) accept or
typed-reject. The VLM does **only perception**; ids, order, flip-safety stay deterministic.

**VLM proxy (honest label):** the "VLM" here is an **Opus vision subagent** that `Read`s
the frame PNGs — a faithful frontier-VLM proxy for *"can a VLM localize this?"*. **This is
NOT Gemini / any pinned API.** Production would pin a specific vision API and have it emit
the `localized/<game>.json` the `FileLocalizer` consumes; the snap/order/validate math is
identical regardless of which model produced the adjacency.

---

## (1) HARD ACCURACY — game-1 anchor (hand-verified, its own video)

Ground truth: `tests/fixtures/human_data/game1_openings.json` (8 placements = 4
settlements + 4 roads across the two players). The VLM localization
(`data/human/vlm_spike/localized/game1__g1.json`) was snapped and compared exact
vertex/edge id vs the fixture.

| Player | settlements | roads |
|---|---|---|
| ThePhantom | 2/2 (v1, v19) | 2/2 (e0, e35) |
| rayman147 | 2/2 (v3, v11) | 2/2 (e8, e19) |
| **Total** | **4/4** | **4/4** |

**HARD ACCURACY = 8/8 (100%), `all_exact=True`, 0 missed.** Independently re-derived by a
fresh vision read of `post_setup.png` (green settlements touch hex-sets {2,9,10}→v11 and
{0,1,6}→v3; black settlements {0,2,3}→v1 and {5,6,16}→v19; roads snap to e19/e8/e0/e35),
matching the committed localization exactly. The snap+order+validate math is proven: given
a correct localization on a **clean post-setup frame**, the pipeline recovers the exact
engine ids and accepts.

> Caveat: 8/8 is a single hand-verified game (the only one with committed ground truth). It
> proves the *math and the VLM's localization on a clean frame*, not a population accuracy
> rate. Its frames are the committed fixture PNGs, **not** the real-video extraction path.

---

## (2) YIELD — Tier-5 real videos (the number classical CV got 0/31 on)

Population: complete game windows detected by `harvest` on the Tier-5 videos, localized by
the VLM proxy, run through the fail-closed validators. Yield = accepted / seen. The game-1
anchor is **excluded** (it is the accuracy anchor, not a real-video yield sample).

<!-- YIELD_TABLE -->

### The decisive finding: the wall is FRAME ROUTING, not blob-vs-VLM perception

`vlm_spike` feeds the VLM the **exact same** `gf.post_setup_frame` that
`detect_openings_result` reads (`harvest.py:458`). On real Tier-5 footage that frame is
frequently **not a clean 8-piece opening board at all** — it is the end-of-game
"Well Played!" **stats overlay** (dimmed final board, full city/road buildout), and the
`empty_baseline` is a **channel-intro splash** (masked-avatar art), not an empty board.

Root cause: these videos' sampled logs did not surface `placed a Settlement/Road` setup
events (the routed `log_setup_sequence` degenerates to only `received starting resources`),
so each game window is mis-bounded and `_post_setup_frame` falls back to a late/end-game
frame. **A VLM front-end swapped in at the perception step cannot recover an opening that
is not present in the frame it is handed.** This is upstream of the settlement/road
detection the spike set out to replace — so the classical pipeline's 0/31 was, for these
videos, not (only) a blob-detector weakness but a **frame-selection failure the VLM
inherits unchanged**.

---

## (3) RECOVERED GAMES / VIDEO (expansion)

<!-- RECOVERED -->

---

## Gating-cell arithmetic (re-run with measured inputs)

Formula (pre-registered): **cell = (games/video × 204) × yield × 0.6**, where 204 = the
high-tier corpus size (`strength_manifest.json`: 204 `strength:"high"` videos) and 0.6 is
the scoreboard-eligibility discount.

<!-- GATING -->

**Point estimate AND Wilson-LB, both propagated** (panel-mandated framing — the cell must
never be quoted from the point estimate alone). games/video = 17/8 = 2.125 from the pilot.

| yield (accepted / seen) | accept fraction | cell |
|---|---|---|
| 11/17 — point estimate | 0.647 | **168** (≥ 147) |
| 11/17 — Wilson-95%-LB | 0.413 | **107** (< 147) |
| 10/17 — point estimate | 0.588 | 153 (≥ 147) |
| 10/17 — Wilson-95%-LB | 0.360 | 94 (< 147) |

The 10–11/17 yield is the post-grant-fix projection (guarded subset-collapse +
OCR-tolerant grant-line matcher); the newly grant-eligible games still need VLM
localization + validator acceptance before they are corpus rows, so treat 11/17 as an
upper bound until then.

**Honesty caveats (all three bind the reading of the table):**

1. **Single-video concentration.** The grant-fix rescues land on ONE video
   (`9Sm86ml04aI`); the yield gain is not yet evidence of cross-video generalization.
2. **Double-use of the 8-video pilot.** The SAME 8 videos supply BOTH the games/video
   factor (2.125) and the yield fraction — the two multiplied terms are not independent
   estimates, so the cell's true uncertainty is wider than the yield CI alone.
3. **The 0.6 order-established factor is ASSUMED, not measured.** No pilot measurement
   backs it; if the true scoreboard-eligibility fraction is lower, every cell above
   shrinks proportionally.

### Pre-registered decision rule (verbatim)

```
projected cell >= 147   -> CONTINUE (build the VLM front-end for real + integration + gold gate)
100 <= cell < 147       -> USER-CALL ("large-effects-only study", honest label)
cell < 147 but > 0 and yield clearly high enough that more videos could reach it
                        -> note the extra-videos path
cell < 100              -> ARCHIVE the corpus program
                           (bank PRE-GATE-0 / M0 / injection-API / manifest; not a failure)
```

<!-- DECISION -->

**The point estimate clears CONTINUE; the CI does not — the decisive lever is MORE
VIDEOS, not more fixes.** At n=17 the Wilson lower bound cannot clear 147 even at
perfect yield propagation of the current fixes; the honest next compute spend is a wider
pilot (e.g. **+10 videos**), which tightens the yield CI and independently re-measures
games/video, rather than further per-game rescues. **The CONTINUE/ARCHIVE decision stays
the user's** — this report presents both readings and does not presume either.

---

## Artifacts

- `scripts/vlm_spike.py` — harness (`prepare-frames` / `localize` / `score` / `batch-score`).
- `data/human/vlm_spike/localized/*.json` — per-game VLM localizations (committed; small).
  Games with no clean opening carry an explicit `{"unlocalizable": "<reason>"}` marker —
  a fail-closed typed reject, never a guess.
- `data/human/vlm_spike/truth/game1__g1.json` — the hand-verified anchor.
- `data/human/vlm_spike/provisional_openings.jsonl` — accepted real-video openings,
  **FLAGGED PROVISIONAL** (not hand-verified — for user verification only).
- `data/human/vlm_spike/frames/`, `.../records/` — **gitignored** (frame PNGs / derived
  records; regenerate via `prepare-frames` / `localize`).
