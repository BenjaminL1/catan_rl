# Morning Report — human_data opening-extraction pipeline (overnight build)

**Run window:** 2026-07-03 → 04 overnight (autonomous, unattended — you did not watch)
**Read at:** ~08:00, 2026-07-04
**Author:** overnight build agent (this is the phase you asked me to build while you slept)

## Bottom line up front

**The build HALTED at Stage 0 (scaffold) and never started Stage 1 (segment/strength) or Stage 2 (openings CV). No game corpus was harvested. No opening was extracted. No scoreboard number exists.** The scaffold review-and-resolve loop ran **4 iterations** and gave up **NOT-READY ("stuck")** with **3 open SHOULD-FIX** and **0 BLOCKER**. Per the gate-first rule the build correctly refused to run any harvest on an un-greened foundation.

**Do not read any calibration / win-rate / opening-prior number from this run — there is none.** The one committed data artifact (`data/human/strength_manifest.json`, 814 videos labeled) is a **video-strength-labeling** artifact, **not** a harvested game corpus. It labels *candidate* videos; it does not contain a single extracted game, board, or opening.

**HARVEST STATUS: GATED — not run.** It is blocked at the Stage-0 scaffold gate and, even if the scaffold greened, it is hard-gated a second time by the missing **glyph classifier** (see human-decision items). Zero games harvested, zero rejected (the rejection-bias audit was never reached because no game ever entered the pipeline).

---

## 1. What shipped (modules + commits)

Scaffold *code* was written, committed, and is unit-level green (`ruff` + `mypy --strict` + `pytest`). The scaffold *review gate* did **not** pass. These are separate facts: **code on `origin/main` is NOT evidence the scaffold is READY.**

Committed this run (newest first, `git log`):

- `72e6ba3` fix: game-1 winner=null (concession, no victory line) + multiset gate is necessary-not-sufficient
- `6031a39` fix: human_data resolve pass — inverted game-1 winner + 3 firewall gaps
- `1b0efdf` fix(human_data): exclude known_window placeholder from scoreboard eligibility
- `b9b876a` feat(human_data): reconcile OpponentStrength source with strength manifest
- `7af8f4a` chore: human-data overnight build workflow (manifest-integrated)
- `2e99c47` data: thephantom strength manifest — 814 videos labeled (204 high)
- `d1ecc6e` feat: shard the strength-manifest batch (--shard/--out + merge)
- `d6b1a8f` feat: thephantom strength-manifest classifier (rank-OCR + tournament)

**Modules present in `src/catan_rl/human_data/`** (written, lint/type/test-green at the unit level, but the scaffold as a whole is **NOT-READY** per the review gate):

| File | State |
|---|---|
| `record.py` (50 KB) | Schema dataclasses, `validate()`, `OpponentStrength` / `derive_opponent_strength()`, `is_scoreboard_eligible()` + `is_strong_opponent_scoreboard_eligible()` (tournament-only primary scoreboard), `STANDARD_RESOURCE_COUNTS` multiset gate (documented necessary-not-sufficient), provenance orientation-binding firewall. **§4.1/§4.2 SHOULD-FIX now RESOLVED; §4.3 (board_cv-dependent) still open (see §4).** |
| `orientation.py` (8.9 KB) | D6 orientation module + `assert_glyph_anchor` (orientation-dependent prediction side) + `assert_scale_up_orientation_gates` (**raises** — batch hard-block until a validated glyph classifier is wired). |
| `topology.py` + `topology.json` | Engine board-topology fixture. |
| `ffmpeg.py` | Frame-extraction helper (thin). |
| `__init__.py` | Package surface. |

**READY modules: none.** The only module reviewed (`scaffold` = `record.py` + `orientation.py` + fixtures) returned `ready: false`, `halted: "stuck"`.

**Not present (never built — pipeline never reached them):** `board_cv.py` (Stage-2 board classifier), the glyph-colour classifier, any Stage-1 segment/strength driver, any harvest runner.

---

## 2. GATE results (every gate, with numbers)

Only one gate ran: the **scaffold review-and-resolve gate** (the "Review" step of the CLAUDE.md loop). It never reached READY.

| Iter | Open findings (severity) | Verdict |
|---|---|---|
| 0 | 5 open — SHOULD-FIX ×4, **BLOCKER ×1** | resolve |
| 1 | 5 open — SHOULD-FIX ×4, **BLOCKER ×1** | resolve |
| 2 | 3 open — SHOULD-FIX ×2, **BLOCKER ×1** | resolve |
| 3 | 3 open — SHOULD-FIX ×3 | resolve |
| **HALT** | **3 open — SHOULD-FIX ×3, BLOCKER ×0** | **NOT-READY ("stuck") after 4 iters** |

Progress was real but incomplete: the loop cleared the **BLOCKER** (present through iters 0–2, gone by iter 3) and closed 2 of the SHOULD-FIX items, but plateaued at 3 residual SHOULD-FIX it could not clear within the iteration budget. It then halted rather than loosen a test or hand-wave a fix.

**No Stage-1 gate, no Stage-2 gate, no harvest gate, no rejection-bias audit ran** — those stages were never reached.

---

## 3. Where it halted, and why

**Halt point:** Stage 0 (scaffold), after 4 review-and-resolve iterations. `stage1: null`, `stage2: null` in the results — literally never started.

**Why:** the scaffold gate returned `ready: false, halted: "stuck"`. Under the gate-first rule ("never commit the expensive next stage before the current stage's go/no-go gate result is in"), an un-greened scaffold blocks all downstream pipeline work. The 3 residual SHOULD-FIX are all **measurement-semantics / data-honesty** findings on the *strength* and *resource-correctness* labels — exactly the class of issue that, left unresolved, produces a confidently-wrong scoreboard rather than a noisy one. Shipping the harvest on top would bake those biases into the corpus. Halting was correct.

---

## 4. The 3 residual SHOULD-FIX (why they matter for the eventual scoreboard)

All three are **labelling / measurement-semantics**, not code-behaviour bugs. None is a crash; each is a way the eventual scoreboard could be *systematically* (not randomly) biased.

1. **`opponent_strength` is a matchmaking PROXY, not a measurement.** `scripts/build_strength_manifest.py` reads *ThePhantom's own* Global-leaderboard world rank (it matches only the channel owner's handle), and `derive_opponent_strength()` stamps that onto `OpponentStrength(tier=...)`, which `is_scoreboard_eligible()` gates on (`tier == "high"`). So "high opponent" actually encodes "the POV player was ≤rank-200 that session". In 1v1 ranked matchmaking that is a reasonable *proxy* for a strong opponent, but it is a proxy, and it is undocumented in the package — a JSONL reader will believe the opponent was rank-verified. Systematic over-statement in the direction of the POV's own rank (ladder variance / smurf / off-peak can draw a much-lower opponent that still enters as "high").
   **RESOLVED (labelling only, no behaviour change):** the `record.py` module docstring, `derive_opponent_strength()`, and `is_scoreboard_eligible()` now all plainly document that the label is ThePhantom's-MATCH-strength, not the adversary's rank, and that a scoreboard builder must split its `n` by `opponent_strength.source`. (The POV-`rank`-int provenance passthrough is deferred to `segment.py`/`build_strength_manifest.py`, not the pure record contract.)

2. **`tournament` highs are title-keyword-only (no frame verification) but enter the same "high" pool.** `derive_opponent_strength()` maps manifest `tournament`→high at confidence 0.8, labeled purely by `TOURNAMENT_RE`. On an n≈20–40 high scoreboard a single title false-positive is a several-point systematic bias. The split-and-robustness caveat lives only in `is_scoreboard_eligible`'s prose; nothing forces the Stage-3 builder to split n by `source` or rerun tournament-excluded.
   **RESOLVED:** the tournament-only (strong-vs-strong) subset is now exported as the primary `GameRecord.is_strong_opponent_scoreboard_eligible()` predicate — a strict subset of `is_scoreboard_eligible()` that admits only `source == "tournament"` — so the Stage-3 builder has a first-class code path for the defensible headline number and the source-split robustness check. The weaker-provenance caveat is also mirrored onto `derive_opponent_strength`'s docstring.

3. **The multiset resource gate is necessary-not-sufficient with no test proving future code understands the tautology risk.** `STANDARD_RESOURCE_COUNTS` correctly warns that (a) a multiset-preserving WOOD↔SHEEP swap passes the gate confidently-wrong, and (b) if `board_cv.py` ever forces the standard multiset (Hungarian cluster→resource), the gate degrades to a tautological no-op. The real cross-check is the orientation glyph-anchor firewall — which is **deferred (classifier not built) and hard-gates the batch** — so at Stage-2 the only live resource-correctness signal could be a tautological gate.
   *Fix (when `board_cv.py` is built):* add a per-game calibration-residual gate + independent per-hex corroboration (pip-count vs OCR-number, already prototyped in the spike's `count_pips` / `ocr_digit`) + a test that a multiset-preserving swap on a real per-hex classifier is caught by corroboration, not merely the count gate.

**Rejection-bias audit: NOT REACHED.** No game entered the pipeline, so there is nothing to audit for rejection bias. This audit is owed at harvest time (Stage 1+), not at scaffold time.

---

## 5. HARVEST status

**GATED — full corpus was NOT harvested. 0 games extracted, 0 games rejected.**

Two gates block the harvest, in order:

1. **Stage-0 scaffold gate — NOT-READY** (3 open SHOULD-FIX, above). Nothing downstream may run until this greens.
2. **Glyph-classifier hard gate (would block even if #1 greened).** `orientation.py::assert_scale_up_orientation_gates` **raises** until a validated log-glyph colour classifier is wired. The batch/harvest path calls it, so it **can never run with the glyph anchor silently absent**. The glyph anchor is the real CV-correctness firewall (the resource multiset gate alone is tautology-vulnerable, per SHOULD-FIX #3). This classifier **was not built this run** — deferred, explicitly hard-gating the safe full harvest.

The one committed data artifact is a **strength-labeling manifest**, not a corpus:

`data/human/strength_manifest.json` — **814 videos labeled** (`build_strength_manifest.py`, rank-OCR + tournament title regex):
- **204 high** (`tier == "high"`) — not scoreboard-eligible as games yet; these are candidate videos only.
- **574 unknown** (`source: none`) — not scoreboard-eligible.
- **36 excluded** (non-own-game / copycat / reaction uploads filtered out).
- By source token: **225 `ranked_rank`**, **15 `tournament`**, **574 `none`**. (Source ≠ final tier: `ranked_rank` rows with rank > `rank_high_max=200` do not become high — this is why source-`ranked_rank`=225 exceeds tier-high=204.)

**This manifest is candidate-video labels only. It contains zero extracted games, boards, openings, or winners.** "204 high videos" is NOT "204 harvested games" — those videos have never been segmented or CV-parsed. Do not treat 204 as a scoreboard n.

---

## 6. Single-command resume state

Nothing is running; nothing needs killing. Workspace is on `origin/main` (`72e6ba3`), git clean except pre-existing untracked paths (`data/exit/`, `scripts/export_dice_vectors.py`, `scripts/record_conformance.py`) and the scheduled-tasks lock. Resume the loop where it halted:

```
# Re-enter the scaffold review-and-resolve loop at Stage 0 (halt point):
# resolve the 3 residual SHOULD-FIX in src/catan_rl/human_data/record.py
#   (§4.1 opponent_strength proxy docstring + rank passthrough,
#    §4.2 tournament source split/robustness,
#    §4.3 multiset-gate corroboration test-owed-at-board_cv),
# re-green ruff + mypy --strict + pytest, re-review until READY.
pytest src/catan_rl/human_data -q && ruff check src/catan_rl/human_data && mypy --strict src/catan_rl/human_data
```

Only after the scaffold is READY do Stage 1 (segment/strength driver) and Stage 2 (`board_cv.py` + glyph classifier) unlock. The harvest cannot be launched before **both** the scaffold greens **and** the glyph classifier is built + validated.

---

## 7. Human-decision items (need you)

1. **Glyph classifier (HARD-GATES the safe full harvest) — build it, or accept a weaker net?** The batch is intentionally hard-gated (`assert_scale_up_orientation_gates` raises) until a validated log-glyph colour classifier exists. It was not built this run. **Decision:** authorize building the glyph classifier (log-icon colour reader, validated against real D6-flip collisions) as the next work item, OR explicitly accept running Stage-2 with only the tautology-vulnerable multiset gate (NOT recommended — see SHOULD-FIX #3). Until this is resolved, no safe full harvest is possible.

2. **`opponent_strength` proxy — accept the proxy, or require frame-verified opponent rank?** The "high" label is ThePhantom's *own* top-200 session rank, a matchmaking proxy for opponent strength, not a direct read. **Decision:** accept the documented proxy (cheap, ships now with the §4.1 docstring/rank-passthrough fix) OR require the harder opponent-rank frame read (more accurate, more CV work, likely lower yield).

3. **Tournament highs — keep the 15 title-only highs in the pool, or scoreboard-split them?** They have no frame verification. **Decision:** require the Stage-3 scoreboard to split n by `source` and report a `tournament`-excluded robustness number (recommended), OR drop tournament-sourced highs from the eligible pool entirely.

4. **Scaffold iteration budget — the loop halted "stuck" at 3 SHOULD-FIX with a clear resolution path.** These are labelling fixes, not architectural rethinks. **Decision:** confirm I should resolve them next session and continue to Stage 1, rather than re-scope the scaffold.

---

## One plain throughline

I built and committed the scaffold and an 814-video strength-labeling manifest, but the scaffold review gate went NOT-READY (3 open SHOULD-FIX, 0 BLOCKER) and I stopped there — so **no games were harvested, no scoreboard exists, and the full harvest is double-gated** (scaffold-not-ready + glyph-classifier-not-built). The residual issues are honest-labeling fixes with a clear path; the biggest human-decision blocker is whether to build the glyph classifier that hard-gates the safe harvest.
