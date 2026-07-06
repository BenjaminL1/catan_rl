# TIER-5 Integration Report — e2e harvest on real ThePhantom videos

**Date:** 2026-07-06. **Scope:** run the `mine_phantom` harvest driver end-to-end on
5 real `high`-tier manifest videos, produce corpus/rejected/telemetry, run the
**joint-D6 flip sweep** on every accepted game, audit **consensus grant supply**, and
report the harvest ETA. **Driver:** `scripts/tier5_harvest.py` (a thin instrumentation
wrapper over `catan_rl.human_data.harvest.run_harvest` that captures per-game
grant-frame counts). **CPU only; no `gui/` import.**

## Verdict

The e2e chain **runs end-to-end on real footage** and the **glyph/consensus firewall
works** (5/6 granting players reached a ≥2-readable-frame consensus grant read). But
the real `high` corpus **cannot yet be harvested to acceptance**: **0/3 reached games
were accepted**, blocked by three integration/CV issues the run surfaced (two fixed
here, one large-CV blocker documented). The **board CV is verified correct** on real
frames (desert + 19/19 resources) and **winner extraction is verified** against a real
victory line. **The plan's §joint-flip residual is NOT closed on real accepted games**
— **0 games were accepted**, so the *game-level* sweep runs only on a single committed
synthetic fixture (the game-1 golden record); the residual across the real corpus is
**unmeasured** pending accepted games. What *is* measured on real data is the
**board-level** leakage surface (openings-independent) over the boards the harvest
actually read — the honest partial signal. See §3.

---

## 1. Per-video accept/reject table (typed reasons)

5 videos, serial (`--max-workers 1 --net-concurrency 1`), 1080p, sparse 4 s sampling.

| video_id | outcome | games | reason (typed) |
|---|---|---|---|
| `33KR75rhTgo` | **processed** | 2 | both games `hud_unreadable` |
| `AoOXWyxaTkA` | **processed** | 1 | `hud_unreadable` |
| `5RLq1NX4nAo` | **video failed** | 0 | `HUD did not yield two distinct seat colours` |
| `sG05DoaOmM4` | **video failed** | 0 | `HUD did not yield two distinct seat colours` |
| `EdMnUD-eZ6A` | **video failed** | 0 | `HUD did not yield two distinct seat colours` |

Per-game rejected records (the real harvest output, committed as
`data/human/tier5_sample.jsonl`):

| video | game | reason | opponent | board_desert | post-setup ts | winner |
|---|---|---|---|---|---|---|
| `33KR75rhTgo` | 1 | `hud_unreadable` | pokeball | hex 13 | 680 | null |
| `33KR75rhTgo` | 2 | `hud_unreadable` | pokeball | hex 9 | 1044 | null |
| `AoOXWyxaTkA` | 1 | `hud_unreadable` | bumbleee | hex 1 | 172 | null |

**No game was silently dropped** — every reached game emitted a typed-rejected record
for the §5.6 rejection-bias audit; every non-reached video emitted a ledger `failed`
row with its cause.

## 2. Telemetry (`tier5_run/telemetry.json`)

```
videos: processed=2  skipped=0  failed=3
games_seen=3  accepted=0  rejected=3  order_unestablished=0
anchor: ran=0  unreadable=0  mismatch=0  grant_read_coverage=0.000
rejected_by_reason: hud_unreadable=3
```

`anchor_ran=0` because all 3 games rejected **before** the joint-flip glyph anchor (at
the openings §5.14 HUD-assignment stage); the anchor only runs on the accept path, so
it correctly never fired.

## 3. Joint-D6 flip sweep — leakage residual

**Question (plan §joint-flip):** the glyph anchor is the sole defence against a
jointly-flipped board+openings (provenance desert-binding cannot see a joint flip). Its
only discriminator is the 3-hex **resource multiset**; the committed board has just
**28 distinct multisets across 54 vertices**, so a D6 relabel that lands the granting
settlement on a collision-partner vertex **false-accepts** (leaks). Measure: relabel
the openings by each of the **11 non-identity D6 elements** (holding the true board read
+ the log-grant ground truth fixed — the presentation a jointly-flipped candidate makes
to the anchor) and count how many `assert_glyph_anchor` calls **reject**, under
(i) the current **either-settlement** matching and (ii) **2nd-settlement-only** matching
(the record now carries log placement order). Tool: `scripts/tier5_flip_sweep.py`.

**Accepted games in the harvest: 0** (all 3 reached games rejected at the §5.14 HUD
stage, before the anchor). The plan's §joint-flip residual asks for the leakage
**over real accepted games**, and that number is therefore **UNMEASURED here** — the
harvest produced no accepted game to sweep. The game-level sweep below is run only on
a **single committed synthetic fixture** (the pipeline's own game-1 golden record —
real board desert=11, hand-verified openings in **log-placement order** so
`settlements[1]` is the true granting vertex, matching `game1_openings.json`'s
`placement_order` block). It demonstrates the sweep mechanism and shows that *this one*
board+openings rejects every flip; it is **NOT** a measurement of the corpus residual:

| matching mode | rejects / (11·n) | leakage |
|---|---|---|
| (i) either-settlement (production `assert_glyph_anchor`) | **11/11** | **0/11** |
| (ii) 2nd-settlement-only | **11/11** | **0/11** |

Both modes reject **every** non-identity joint flip: a game-level leak needs **both**
players' relabeled settlements to simultaneously land on grant-matching vertices, which
does not occur on this board. Modes (i) and (ii) are **equal here** (no first-settlement
collision), but (ii) ≥ (i) always by construction (matching only the granting settlement
halves the collision surface).

**Board-level leakage surface (openings-independent — usable on the real boards the
harvest read).** Per board, over all 54 vertices × 11 flips, the single-settlement
false-accept rate = fraction of *moved* settlements whose resource multiset is preserved
by the relabel:

| board | leak_pairs / moved_pairs | single-settlement leak rate | distinct multisets | colliding vertices |
|---|---|---|---|---|
| reference (desert 11) | 104/576 | **18.1%** | 28 | 38/54 |
| harvested (desert 13) | 94/576 | **16.3%** | 25 | 43/54 |
| harvested (desert 9) | 30/576 | **5.2%** | 34 | 31/54 |

**Interpretation (and what is NOT concluded).** A single settlement's joint-flip
leakage residual is **5–18%** (the board's multiset-collision structure) — this is a
**real measurement** on the harvested boards. At the **game level** (both players'
relabeled settlements must *independently* collide), the single committed fixture leaks
**0** (rejects 11/11), and **2nd-settlement-only matching is the safe default** — it
never leaks more than either-settlement. **But one synthetic game is not the corpus.**
The plan's §joint-flip residual — the game-level false-accept rate **across real
accepted games** — is **still UNMEASURED** (0 accepted); the 11/11 is a single-fixture
demonstration, not evidence the residual is board-wide ≈0. The honest bound we have is
the settlement-level **5–18%**; whether real accepted games' *pairs* of settlements
collide simultaneously is an open number that only real accepted footage can close.
(The purely-symmetric joint flip — board *and* openings both relabeled by the same g —
is a genuine relabeling the grant multiset is invariant under, and is outside the
anchor's reach by construction; the operative defence measured above is against an
openings snap mis-oriented *relative to a correctly-read board*.)

Pinned in `tests/unit/human_data/test_tier5_flip_sweep.py` (reference sweep 11/11;
game-1 board 28 distinct / 38 colliding — matching the committed
`test_glyph_anchor_multiset_collision_rate`).

## 4. Consensus-supply audit (≥2 readable grant frames per granting player)

The multi-frame CONSENSUS grant read (`consensus_granted_glyphs`) requires **≥2
byte-identical readable frames**. Captured per game/player (`tier5_run/grant_supply.json`):

| game | player | candidate grant frames | readable frames | consensus ok |
|---|---|---|---|---|
| `33KR75rhTgo:0` | ThePhantom | 4 | **4** | ✅ |
| `33KR75rhTgo:0` | pokeball | 4 | <2 | ❌ |
| `33KR75rhTgo:1` | ThePhantom | 13 | **4** | ✅ |
| `33KR75rhTgo:1` | pokeball | 13 | **7** | ✅ |
| `AoOXWyxaTkA:0` | ThePhantom | 6 | **2** | ✅ |
| `AoOXWyxaTkA:0` | bumbleee | 6 | **5** | ✅ |

**Consensus supply: 5/6 granting players reached ≥2 readable grant frames** (readable
counts {2, 4, 4, 5, 7}); the single miss (pokeball 33KR:0, <2 readable) is exactly the
fail-closed behaviour — an unreadable grant would be a typed `glyph_unreadable` reject,
never a fabricated multiset. **The glyph/consensus firewall works on real footage.**
Grant-read coverage of *accepted* games would be 1.0 by construction (an accepted game
requires the anchor to have run for both players); no game reached acceptance here.

## 5. Wall-clock & harvest ETA

- **Wall clock: 2370 s total, 474 s/video** (5 videos serial, incl. download +
  full sparse-pass OCR + board/opening/grant CV). Down from 744 s/video before the
  easyocr-reader-cache fix (§6).
- **Corpus ETA:** 474 s/video × 204 `high` videos = **26.9 h serial** (`n_procs=1`);
  **≈6.7 h at 4 workers**, **≈3.4 h at 8 workers**. (This is the *attempt* ETA;
  accepted-game yield is currently ~0 pending the §6 blockers.)

## 6. Integration findings

### Fixed in this slice (`src/catan_rl/human_data/harvest.py`, tested)

1. **Per-video single-frame HUD colour bind → majority vote.** `_extract_context`
   read the HUD seat colours from **one** global-middle frame. ThePhantom compilations
   stitch back-to-back games whose **opponent colour varies**, so a middle frame landing
   on a palette-unsupported game failed the **whole** video even when a supported game
   existed elsewhere. Replaced with `_majority_two_colour` over ~24 spread frames (a
   one-colour read casts no vote). This unblocked `33KR75rhTgo`/`AoOXWyxaTkA` from
   whole-video failure to per-game processing. Tests:
   `test_majority_two_colour_*` in `tests/unit/human_data/test_harvest.py`.
2. **easyocr reader cache in `_grant_line_boxes`.** It re-instantiated
   `easyocr.Reader` (model load ~seconds) **per grant frame per player**, dominating
   wall clock. Cached like `logparse`/`board_cv` (`_boxes_reader`). 744 → 474 s/video.

### Blockers found (documented — NOT fixed here; out of scope / need CV validation)

3. **Player-colour PALETTE is GREEN+BLACK only** (`openings.PALETTE` / `_HUD_RING`).
   ThePhantom self-seats a fixed colour (**BLACK**) but opponents are overwhelmingly
   **non-green**. Survey of 25 valset `high` videos: only **2** (`33KR75rhTgo`,
   `5RLq1NX4nAo`) contain any GREEN-opponent game; **3/5** run videos failed at the HUD
   vote because the opponent colour is unsupported. **This is the dominant blocker to
   harvesting the real corpus** — the palette needs the actual opponent colours
   (red/orange/blue/…) with validated HSV piece/road/tile-subtract/HUD-ring ranges.
4. **GREEN openings suppressed** (`settlement_blob_shortfall:GREEN:green_tile_suppressed`).
   Even on the one supported (green-opponent) game, the openings CV fails: GREEN is the
   tile-subtract colour (§5.13) and a clean empty baseline is unavailable at setup, so
   the green-tile subtraction removes the real green settlements. Spot-verify confirms
   the green settlements **are present and locatable** (bottom-middle ~v18, top-right
   ~v14) — this is a CV-sensitivity failure, not absent pieces.
5. **Segmentation captures setup-only fragments on noisy OCR.** The 3 reached games have
   `winner=null` and empty `dice_log` (no rolls) and post-setup timestamps that predate
   their game's reset (e.g. `33KR75rhTgo` g2 ts=1044 vs the green reset ~1102) — the
   frame router/segmenter welds a game's tail onto its neighbour, so the §5.14 openings
   HUD check reads a wrong-game frame → `hud_unreadable`. Needs Stage-1 segmentation
   hardening on real OCR (the spikes were single-frame throwaways; plan §5.13 warned
   "Stage 2 is the bulk").

## 7. Spot-verifications

- **Board CV (visual, Read the image).** Re-decoded the `33KR75rhTgo` green post-setup
  frame (~t=1170), ran `read_board_stable` (desert=**hex 0**, residual **0.89 px**), and
  overlaid the read resource+number on each hex. **19/19 resources match the tile art**
  (forests→WOOD, fields→WHEAT, pastures→SHEEP, red hills→BRICK, grey mountains→ORE,
  cactus→DESERT) and the desert is correctly hex 0. The board CV — the component the
  openings depend on — is correct on real footage.
- **Winner vs log.** OCR'd `33KR75rhTgo`'s log near the green game's end (t≈1960–1985):
  the real line **`ThePhantom won the gamel`** (OCR of "won the game!") parses via
  `logparse.parse_log` → **winner = "ThePhantom"**. The winner-extraction grammar is
  correct on real footage (the harvested game *fragments* legitimately carry
  `winner=null` — their windows never reached this line, per blocker 5).

## 8. Artifacts

- `scripts/tier5_harvest.py` — instrumented e2e driver (grant-supply capture).
- `scripts/tier5_flip_sweep.py` — joint-D6 flip sweep + board leakage surface.
- `tests/unit/human_data/test_tier5_flip_sweep.py` — pins the reported sweep/board numbers.
- `data/human/tier5_sample.jsonl` — the 5-video harvest output (3 typed-rejected records;
  corpus.jsonl was empty, so the sample is the rejected rows). 4.5 KB.

## 9. Recommended next steps (to reach accepted games on the real corpus)

1. **Extend `openings.PALETTE` to the real opponent colours** with a validated glyph/piece
   colour set (the dominant unblocker — mirrors the glyph-classifier validation gate).
2. **Harden GREEN piece detection / empty-baseline selection** (§5.13 green-tile
   subtraction) so supported green games accept.
3. **Harden Stage-1 segmentation on noisy OCR** so per-game windows are not welded (fixes
   the setup-fragment / wrong-post-setup-frame failure).
