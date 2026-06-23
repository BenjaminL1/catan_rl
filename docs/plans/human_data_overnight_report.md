# Human-Data Pipeline — Morning Report (overnight autonomous build)

**Run date:** 2026-06-22 → 23 overnight (unattended). **Read time:** ~4 min.
**Bottom line:** The build **HALTED at Stage 0 (scaffold)** and **did not ship any pipeline
stage**. Stage 0 is the contract-firewall scaffold; its review never reached READY after 4
review-and-resolve iterations, ending on **2 BLOCKERs + a red-team counterexample** that all
point at one real defect. Per the gate-first rule, no Stage-1 or Stage-2 work was attempted.
**Nothing here claims success the gates did not show — they did not.**

---

## TL;DR for the 8am decision

1. **What shipped to main:** the scaffold *code* (commits `f390a6d`, `d5647a9`, `b1ae7e7`,
   `be7f6f5`) — `GameRecord` contract firewall, topology fixture, ffmpeg shim. It is NOT
   marked READY: the review verdict is **NOT-READY (stuck)**.
2. **Why it halted:** the committed **golden game-1 fixture is confidently mislabeled** — the
   board hexes are in the orientation-locked frame (**desert = hex 11**) but the opening
   vertex/edge IDs were snapped under the **rejected** un-locked frame (**desert = hex 17**).
   Those two frames are a D6 reflection of the same physical board, so the openings index the
   wrong vertices. This is precisely the §5.2 "confidently wrong, not noisy" failure mode the
   firewall exists to prevent, and the firewall does **not** catch it.
3. **Decisions you must make (HALT-and-escalate):**
   - **(a) Opponent-strength default** — the `opponent_strength` schema is open-design and was
     left **unresolved** (a SHOULD-FIX). It needs your call before Stage 2 pools any games.
   - **(b) Deliberate halt-before-300** — the build was instructed to stop before scaling to
     300 games; it never got near that (it died at Stage 0), so this halt is **moot for now**
     but remains armed for whenever the pipeline is unblocked.
   - **(c) The orientation BLOCKER fix** — re-derive openings under the locked affine (below).

---

## What shipped (modules + commits)

| Module | State | Notes |
|---|---|---|
| `src/catan_rl/human_data/record.py` (`GameRecord` firewall) | **scaffold present, NOT READY** | resource/id/ruleset/snake-draft/distinct-seat/int-only gates landed (`d5647a9`, `b1ae7e7`). Review found the firewall **cannot catch the dominant failure mode** (cross-orientation relabel). |
| `src/catan_rl/human_data/topology.py` + `topology.json` | present | geometric topology fixture, loadable via `load_topology()`. |
| `src/catan_rl/human_data/ffmpeg.py` | present | ffmpeg resolver / `_no_imageio` shim, typed for mypy strict (`be7f6f5`). |
| `tests/unit/human_data/test_scaffold.py` | present, **enshrines a bad golden** | board uses correct desert=11 (`_GAME1_HEXES`), but pairs it with desert=17-frame opening IDs. |
| `tests/fixtures/human_data/game1_openings.json` | **wrong orientation** | `fit.desert_hex: 17`; openings `[4,10]/[20,0]` snapped under the rejected affine. |

Commits on `origin/main` (this run's lineage):
`f390a6d` scaffold package · `44264ea` build brief (docs) · `d5647a9` firewall · `be7f6f5`
mypy shim · `b1ae7e7` hardened firewall.

**No new module reached READY.** Stage 1 and Stage 2 were never started (see stage table).

---

## Stage gate results (every gate, with numbers)

### Stage 0 — Scaffold / contract firewall — **GATE: FAIL (NOT-READY, stuck)**

Review-and-resolve loop, 4 iterations, did not converge:

| Iter | Opened | Severity mix | Action |
|---|---|---|---|
| 0 | 4 | SHOULD-FIX ×4 | resolved |
| 1 | (verify) | RED on re-verify | resolved |
| 2 | 7 | SHOULD-FIX ×6, BLOCKER ×1 | resolved |
| 3 | 9 | BLOCKER ×3, SHOULD-FIX ×6 | resolved |
| 4 | — | **still NOT-READY** | **HALT (stuck)** |

**Final open at halt: 2 BLOCKERs + 1 red-team BLOCKER counterexample + 6 SHOULD-FIX.** A
review that still surfaces BLOCKERs after 4 resolve passes is, per the loop's own rule, **not
READY** — the loop refused to declare victory, which is the correct behavior, not a success.

- **BLOCKER 1 — golden openings in the wrong (un-locked) orientation.** The committed
  `game1_openings.json` IDs come from the **opening_cv** spike, whose own `VERDICT.txt`
  caveat #2 admits its affine used a **non-deterministic, un-locked** orientation
  (`desert == hex17`). The BLOCKER that *resolved* §5.2 (`blockers/orient_lock2.py`,
  screen-space H8→top-center / H11→rightmost) produced the **orientation-LOCKED** board with
  **desert = hex 11, byte-identical across 5 frames** (`blockers/board/AGREEMENT_TABLE.txt`,
  `locked_board_240.json`). `desert=17` vs `desert=11` are a D6 reflection of the *same*
  physical board. The scaffold's `_GAME1_HEXES` uses the correct **desert=11** frame, but the
  golden openings were snapped under the **rejected desert=17** affine → the record pairs a
  desert=11 board with desert=17-frame opening IDs.
- **BLOCKER 2 — no cross-artifact orientation consistency check.** `GameRecord.validate()`
  stores both board hexes and opening IDs but never asserts they share one affine. The
  multiset gate (resource counts, number bag) is **orientation-invariant** — a D6-flipped
  board has identical multisets — so it passes a mislabeled record silently.
- **BLOCKER (red-team counterexample) — verified, reproducible.** Re-resolving the same screen
  pixels across the two frames (snap < 0.2px): ThePhantom's recorded settlement vertex 4
  (px 853,497) is engine vertex **3** in the board frame; vertex 10 → **15**; rayman vertex 20
  → **8**; vertex 0 → **1**. Reconstructing tiles from the record: it says ThePhantom touched
  `{ORE,SHEEP,SHEEP}` + `{BRICK,WHEAT,WHEAT}`, but the human physically built on
  `{BRICK,ORE,SHEEP}` (v3) + `{BRICK,SHEEP,WHEAT}` (v15). **Wrong tiles, wrong numbers** —
  exactly the kind of confidently-wrong opening that would mismeasure v8's ORE/opening blind
  spot, which is the entire point of the pipeline. `GameRecord(...)` with this mix is
  **ACCEPTED** by `validate()` (zero orientation/incidence reconciliation in
  `src/catan_rl/human_data/`).

**Open SHOULD-FIX at halt (carried, not blocking the halt verdict):**
1. `validate()` does not gate `provenance.resolution >= 1080` (§5/§5.10: "360–480p is garbage").
2. `dice_log` stored but never validated — it is the §5.4 dice-luck covariate; entries not
   range-checked to 2..12.
3. `dice_log` is `int()`-coerced on load (the same float-coerce hole the BLOCKER float tests
   closed for IDs) and bypasses the strict `_as_int` gate.
4. `opponent_strength` has no source/tier/confidence consistency rule, and the open-design
   "known-high-rank-window only" default is **not flagged in code** (see escalation (a)).
5. Scoreboard/seed eligibility predicate exists **only as a docstring truth-table**, not as
   `is_scoreboard_eligible()` / `is_seed_eligible()` methods — the scoreboard filter and the
   §5.6 bias audit will each re-implement it and drift.
6. The free-floating `desert_hex: 17` in the golden contradicts ground truth three independent
   ways (AGREEMENT_TABLE, every `locked_board_*.json` for t≥240, `_GAME1_HEXES`).

### Stage 1 — **NOT STARTED.** Gate: not reached (Stage 0 failed first).
### Stage 2 (openings parse + scoreboard) — **NOT STARTED.** Gate: not reached.
### Rejection-bias audit (§5.6) — **NOT REACHED.** Requires Stage 2 output; none exists.

---

## Rejection-bias audit

**Not reached.** The §5.6 audit runs on parsed/rejected records produced by Stage 2. Stage 2
never ran, so there is **no audit, no numbers, and no claim** about rejection bias. Flagging
ahead: the audit and the scoreboard filter must call one canonical eligibility predicate
(SHOULD-FIX #5) or they will drift — fix that when Stage 2 is built.

---

## Exactly where it halted and why

- **Where:** end of Stage 0 (scaffold), review iteration 4.
- **Why:** the contract-firewall scaffold is **structurally complete but unsound** — it admits
  a confidently-mislabeled golden record (cross-orientation D6 relabel) that the firewall
  cannot detect. Two independent BLOCKERs plus a reproduced red-team counterexample all reduce
  to: **board and openings were reconstructed under two different D6 orientations and welded
  into one record.** The loop correctly refused to mark this READY and, per gate-first rules,
  did **not** proceed to any pipeline stage.

This is **not** a "loosen the test to go green" situation: the data is wrong vs verified ground
truth. The fix corrects a Stage-0 fixture to match the authoritative spike artifact.

---

## Single-command resume state

The fix is well-specified; resume by reconciling both artifacts to the **one** locked affine,
then re-greening the loop:

```bash
# 1. Re-derive game-1 openings under the LOCKED affine (desert=11 frame), overwriting the golden:
python scripts/dev/human_data_spikes/opening_cv/final_detect.py \
    --affine scripts/dev/human_data_spikes/blockers/orient_lock2.py \
    --out tests/fixtures/human_data/game1_openings.json
# (re-derive ThePhantom/rayman147 settlement+road engine IDs in the desert=11 frame;
#  set fit.desert_hex = 11; update test_scaffold.py expected opening IDs to match.)

# 2. Add the cross-orientation firewall + eligibility methods, then re-run the loop:
ruff check src/catan_rl/human_data tests/unit/human_data && \
mypy --strict src/catan_rl/human_data && \
pytest tests/unit/human_data -q
```

Required code change before re-review can pass: add an orientation cross-check to
`GameRecord.validate()` that binds board hexes and opening IDs to one frame (assert each
player's roads are incident to that player's settlements via `load_topology()`, and that the
board's desert hex_id matches the affine that snapped the openings), and record the chosen
orientation in `provenance`. Also add `is_scoreboard_eligible()` / `is_seed_eligible()`.

**Working tree at report time:** branch `main`, clean except untracked `data/`,
`scripts/export_dice_vectors.py`, `scripts/record_conformance.py`, and the modified
`.claude/scheduled_tasks.lock` — none touched by this build.

---

## HALT-and-escalate items the human must decide

1. **Orientation BLOCKER (must fix before any Stage-2 code).** Confirm the resume plan above:
   re-derive openings under the locked desert=11 affine and add the cross-artifact check. This
   is the load-bearing correctness gate for the whole scoreboard.
2. **Opponent-strength default (open-design, UNRESOLVED).** The brief left
   `opponent_strength` open: schema supports `source='known_window'` and `tier in
   {high,unknown}` (right shape), but the spike never read a rank badge. **Decision needed:**
   confirm v1 establishes strength via **known high-rank channel windows only**
   (`known_window`), treat `rank_badge` as a forward-compat slot (not implemented), and make
   `tier='unknown'` scoreboard-ineligible. §5.5 forbids pooling mixed strength, so this gate
   must be nailed down before Stage 2 buckets any games.
3. **Deliberate halt-before-300 (armed, currently moot).** The build was told to stop before
   scaling to 300 games. It died at Stage 0, far short of that, so the halt did not fire —
   but it remains in force: do **not** scale to the full 300-game corpus without an explicit
   go after Stage 2's gate and the rejection-bias audit are green.

---

## Honest status line

Stage 0 scaffold code is on main but **failed its review gate**. No opening was parsed, no
scoreboard built, no seeds produced, and the rejection-bias audit never ran. The single defect
blocking everything is a verified cross-orientation mislabel in the committed golden; the fix
is specified and the resume command is above.
