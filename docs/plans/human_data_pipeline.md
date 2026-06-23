# Human-Data Pipeline — Build Handoff Brief

**For:** another Claude session that will build this. **Status:** plan is expert-reviewed and
READY; no code written yet. **Read this top-to-bottom before touching anything.**

---

## 1. What we're building (and why)

A new package **`src/catan_rl/human_data/`** + CLI **`scripts/mine_phantom.py`** that parses
**ThePhantom's YouTube Colonist.io 1v1 game videos** into a dataset of **human OPENINGS +
outcomes**. Two uses, in priority order:

- **(A) An opening SCOREBOARD** — an *external* reference to measure champion **v8**'s opening
  play. (Internal metrics can't: v8 grades its own homework.)
- **(B) DIVERSE opening SEEDS** — real human openings to break v8's self-play opening-diversity
  collapse.

**Why this exists:** a strong human (ThePhantom, Colonist rank ~2–5) played `v8 + determinized
PUCT search` and judged it clearly **subhuman specifically at the OPENING** (it neglects ORE /
the ore→city→dev engine). A Tier-0 diagnostic (`runs/probe_value_winprob.json`) showed v8's
value head judges openings far worse (net_accuracy ~0.43–0.50) than mid/late (~0.88); a 6-lens
research panel traced the root cause to **opening-diversity collapse in self-play** (the whole
v-lineage shares one un-punished opening basin). The fix needs (i) an external human reference
to measure the gap and (ii) diverse opening seeds to force exploration. This pipeline supplies
both. See memory notes `project_surpass_thephantom.md` and `project_thephantom_data_parsing.md`.

**Constraints (hard):** 1v1 Colonist ruleset (15 VP, 2 players, no player-trade, 9-card discard,
Friendly Robber, StackedDice). M1 Pro, single dev. **NEVER import `gui/` or the training path.**
CPU only. This is measurement/seeding ONLY — it changes no engine rules and is never a deploy
policy.

---

## 2. Proven foundation — 4 de-risk spikes, all GREEN

Feasibility is **proven on real footage**, not assumed. Spike code + artifacts are in
**`/tmp/phantom/{m0,board_cv,opening_cv,blockers}/`** — **`/tmp` is ephemeral; treat the spike
code as reference and re-bank anything needed.** Sample video:
`https://www.youtube.com/watch?v=9Sm86ml04aI` (1080p).

- **M0 log-OCR:** Colonist's top-right game-log (crop ≈ `(0.645,0,1.0,0.3)·frame`) OCRs at
  **~99% char accuracy at 1080p** via easyocr (events, names, setup placements, the "Happy
  settling" new-game marker, victory line). **360–480p is garbage → must pull 1080p.**
- **board-CV:** detect the 19 hexes (HoughCircles on number-token disks) → fit an **affine to
  the engine's exact board geometry** → resource (median hex color, **calibrated per-frame**,
  not hardcoded) **19/19**, number (OCR + independent pip-count) **18/18**. Maps to engine
  hex/vertex/edge integer IDs at **sub-2px**.
- **openings:** on the near-empty post-setup board, detect each player's 2 settlements
  (→ vertex IDs) + 2 roads (→ edge IDs), assign by color, **8/8 correct** on game 1, verified
  against the log's snake-draft sequence + the road-incident-to-settlement legality rule.
  (Post-setup single frame works; incremental-diff FAILED — swamped by GUI glow.)
- **winner + orientation** (the 2 correctness-at-root blockers, now proven):
  - **Winner** reads cleanly from the log line **`🏆 <player> won the game! 🏆`**.
  - **Orientation lock** is **frame-stable**: 19/19 hexes byte-identical across 5 frames of one
    game (artifact `blockers/board/AGREEMENT_TABLE.txt`).

---

## 3. Data contract — `GameRecord` → JSONL (freeze this FIRST, before module code)

One JSON record per game, one per line. **All coordinates are engine integer IDs** (19 hex /
54 vertex / 72 edge) from `scripts/export_topology.py`, so records drop straight into the RL
stack. Resources are **string literals** (`"WOOD"`, `"BRICK"`, `"WHEAT"`, `"ORE"`, `"SHEEP"`,
`"DESERT"`).

```jsonc
{
  "schema_version": 1,
  "video_id": "9Sm86ml04aI", "game_index": 2,
  "players": {"agent": "ThePhantom", "opponent": "rayman147"},
  "opponent_strength": {"tier": "high", "source": "rank_badge|known_window", "confidence": 0.x},
  "ruleset": {"num_players": 2, "win_vp": 15},
  "board": {
    "hexes": [{"hex_id": 0, "resource": "ORE", "number": 8}, /* …19; desert: number=null */],
    "ports": "OMITTED in v1"   // never extracted in spikes — see §5
  },
  "draft_order": ["ThePhantom", "rayman147", "rayman147", "ThePhantom"],
  "openings": {
    "ThePhantom": {"settlements": [12, 41], "roads": [18, 55]},   // vertex ids, edge ids
    "rayman147":  {"settlements": [7, 33],  "roads": [9, 47]}
  },
  "dice_log": [/* per-roll values, for the dice covariate — see §5 */],
  "winner": "ThePhantom",      // from the victory LOG line only; null if resign/cutoff
  "episode_source": "natural", // "natural" | "human_seed"  (load-bearing — see §5)
  "rejection_reason": null,    // set on rejected records (kept for the bias audit)
  "passed_crosscheck": true,
  "provenance": {"resolution": 1080, "ts": 247}
}
```

---

## 4. Modules + staged build (each stage has a go/no-go gate)

Modules (each hardened from its spike — see §5 effort note): `ingest.py` (yt-dlp 1080p +
frame sampling), `logparse.py` (crop+easyocr+Colonist log grammar → events+winner),
`segment.py` (game-boundary + ruleset + opponent-strength filter), `board_cv.py`
(orientation-locked lattice fit + tile/number), `openings.py` (post-setup frame + piece detect
+ snap), `validate.py` (cross-check gate), `batch.py` (parallel, resumable).

- **Stage 1** (`ingest`+`logparse`+`segment`): corpus-wide `{names, events, winner, ruleset,
  opponent_strength}` per game. Gives the **game list with winners** fast. Gate: log accuracy.
- **Stage 2** (`board_cv`+`openings`+`validate`): the full opening+outcome record. **This is the
  BULK of the project** (see §5). Gate: gold-set field accuracy bars.
- **Stage 3** (`batch` + gold audit): validate ~30-game gold set vs pre-registered bars
  (layout ≥98%, openings ≥95%, winner ~100%), then batch → the JSONL. Expected yield ~60–80%.

---

## 5. CRITICAL correctness constraints (from the expert review — do NOT relearn these the hard way)

These are *why the plan is the way it is*. Violating them produces a dataset that is
**confidently wrong**, not merely noisy.

1. **Winner = the victory LOG line ONLY.** The top-left `"X - Y"` counter is a **match counter
   that accumulates across games** (it does NOT reset, was 0-0 mid-game) — it is NOT the score.
   The center **"Victory!!!" banner is unreliable** (showed "Victory!!!" in a game the POV
   *lost*). resign / video-cut-off-before-15VP → `winner = null` (exclude from scoreboard, keep
   board+openings for seeds).
2. **Orientation lock is mandatory.** The 19-hex board is **D6-symmetric** → all 12 candidate
   affines fit with *identical* residual (<2px), so "lowest residual" flips orientation on
   noise and silently relabels ALL IDs. **Lock with a screen-space rule** (engine H8 → top-center,
   H11 → rightmost; ~146× margin) **+ ≥2 OCR anchors** (a number token AND its resource must land
   on the predicted engine ID); reject if not exactly 1 of 12 passes. **Add a cross-frame
   stability check** (same game's board map must agree across ≥2 frames) + a residual gate (skip
   frames with mean residual >5px). **Do NOT anchor on the desert** (it moves per game).
3. **Videos contain MANY back-to-back games.** Segment each game by the `"Happy settling!… List
   of commands: /help"` reset (start) and the victory line / end-screen modal (end).
4. **The scoreboard is a CALIBRATION check, NOT an archetype leaderboard.** After rejection +
   rank-filter + archetype bucketing, per-bucket n is ~20–40 (Wilson ±~17pp) and it's **one
   human** (style/opponent confounded). Deliverable: *does v8's VALUE of the human opening
   position predict the human's result?* + at most a coarse 2-way split. Anchor on v8 value
   (dice-marginalized); OCR the **per-roll dice** (`dice_log`) to carry dice luck as a covariate;
   use realized human win-rate only as a weak external calibration check. State achievable n + CI.
5. **Opponent strength is a REQUIRED field.** Use an objective signal (visible rank/elo badge, or
   restrict to a known high-rank window of the channel) — NOT a handle guess. Exclude games where
   strength can't be established from the *reference* (they may still be seeds). Never pool a
   single "vs humans" number across mixed strength.
6. **Rejection bias must be audited.** Every rejected game still emits its parsed features +
   `rejection_reason`. Test + **publish** per-archetype acceptance rate (green-tile subtraction is
   harder near wood/sheep tiles → rejection is feature-correlated). Gate Stage 3 on it.
7. **Seeds must not re-import the human cap.** Enforce `episode_source`: eval/anchor see ONLY
   `"natural"` episodes. **Re-run the engine's own opening-legality check** (settlement spacing,
   road incidence, in-range) on every loaded seed and hard-reject illegal ones. Only
   `passed_crosscheck=true` records are seeds. (Spike midgame snap-err was 15–24px — never trust a
   snapped piece without the legality re-check.)
8. **Resources = string literals.** There is **NO `RESOURCES` enum** in `engine/` (3+ inconsistent
   ad-hoc orderings exist). The only stable constant is **`RESOURCES_CW` (WOOD,BRICK,WHEAT,ORE,
   SHEEP) at `src/catan_rl/env/catan_env.py:82`** — convert to it only at the RL boundary.
9. **Ports omitted from v1** (never extracted in any spike; board comparisons are port-agnostic
   until ports get their own gold bar). Don't ship a half-populated `ports` field.
10. **Compute is real, not "overnight."** easyocr ≈ 0.58s / 1080p crop; 1fps × ~300 videos ×
    ~79 min ≈ **~10 days single-process**. Use **two-pass sampling**: Pass A sparse (1 frame /
    3–5s) for the log + game boundaries + the 8 setup-event timestamps; Pass B dense ONLY in the
    bounded setup window + at the winner line. Parallelize across perf cores. State ETA as
    `fps × 0.58s × n_videos / n_procs`.
11. **yt-dlp at scale:** the pre-resolved stream URL is a short-lived `ANDROID_VR` googlevideo URL
    → **don't stream it**. Let yt-dlp **download-then-delete** each 1080p video-only DASH per
    video (~0.22 GB, resumable) + `--retries`/`--sleep-interval`/format-fallback; cap network
    concurrency to 1–2. `node` IS installed (satisfies yt-dlp's nsig JS runtime); `deno` is not.
12. **Frames in-memory** (ffmpeg → numpy via stdout) — never accumulate per-frame PNGs to disk.
13. **Effort:** "harden from the spike" understates it — spikes are single-frame, hard-coded
    throwaways (the resource classifier is even stubbed; per-game color/palette/baseline/orientation
    all need generalizing). **Stage 2 is the bulk of the project.**
14. **Player→color from the per-game HUD seat row**, never a global constant (a spike comment had
    it inverted; HUD is authoritative). Validate the 2 detected piece colors match the 2 HUD seat
    colors for that game.

---

## 6. Conventions / dependencies

- New package `src/catan_rl/human_data/`; CLI `scripts/mine_phantom.py`. **No `gui/` or training
  import.** CPU. Launch the batch detached (`nohup`).
- **Commit the untracked deps first**: `scripts/export_topology.py` and `src/catan_rl/conformance/`
  are `??` in git but used as stable APIs. Generate the topology once and **commit it as a package
  fixture** (`importlib.resources`, NOT `/tmp`).
- `brew install ffmpeg` (not installed); `ingest.py` should fail fast with a clear message if absent.
  (`yt-dlp` is installed; portable ffmpeg also available via `pip install imageio-ffmpeg`.)
- **Commit the game-1 frames** (post-setup `t=247`, empty-baseline `t=105`, a log crop) as a small
  **CI golden fixture**; add a deterministic test reproducing the game-1 `openings.json`; unit-test
  the log grammar against the real noisy `ocr_*.txt` (incl. the "Happy settlingl" typo).
- Add `schema_version` (mirror `conformance/recorder.py` `CONFORMANCE_SCHEMA_VERSION`).
- ToS: retain only derived `GameRecord`s (no redistributed video beyond the tiny test fixture),
  rate-limit downloads, keep `video_id` for attribution. Research use.
- ruff + mypy + pytest green; conventional commits; push to `origin/main` (solo, no PRs).

---

## 7. Status & immediate next steps

- **DONE:** 4 de-risk spikes (all green), expert review of the plan (**all 10 blockers + 8
  should-fixes resolved**), deliverable re-scoped to calibration-grade.
- **NEXT:** (1) freeze the `GameRecord` dataclass (§3) + `schema_version`; (2) resolve the one
  open detail — the **opponent-strength signal** (rank badge vs known high-rank window; ~20-min
  check, fold into Stage 1); (3) build **Stage 1** test-first → corpus-wide game list with
  winners; then Stage 2 (the bulk), then Stage 3.
- **Pointers:** memory `project_thephantom_data_parsing.md` + `project_surpass_thephantom.md`;
  spike artifacts `/tmp/phantom/`; engine `src/catan_rl/engine/`, `scripts/export_topology.py`,
  `src/catan_rl/conformance/` (untracked); reuse `eval/wilson.py` for CIs,
  `eval/rules_invariants.py` spirit for the multiset gate.
