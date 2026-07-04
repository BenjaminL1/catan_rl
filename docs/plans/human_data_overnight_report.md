# MORNING REPORT — human_data opening-extraction pipeline (overnight autonomous build)

**Run:** overnight, autonomous, unattended — **you did not watch this run** (you asked me
to build this phase while you slept).
**Bottom line, up front:** The pipeline was **NOT** completed and the **full corpus was NOT
harvested.** The build **halted with every Stage-1 and Stage-2 module RED**, so **both
harvest gates (gold + glyph) are unmet** and **zero games were extracted.** Only the
**scaffold** is READY. Nothing here claims a passing gate that did not pass.

---

## 1. What shipped (READY modules + commits)

| Module | Status | Commit(s) |
|---|---|---|
| **scaffold** | **READY** | already on `main`; no new commit this run (see note) |
| ingest | RED (halted, stuck) | partial: `3b83f8b`, `d6bccc0` |
| logparse | RED (halted, stuck) | in-tree, not gate-clean |
| segment | RED (halted, stuck) | in-tree, not gate-clean |
| board_cv | RED (impl agent died) | partial in-tree (through `734316e`) |
| openings | **not built** (impl agent died iter 0) | none |
| validate | **not built** (impl agent died iter 0) | none |
| glyph_anchor | **not built** (impl agent died iter 0) | none |
| batch | **not built** (impl agent died iter 0) | none |

**Only `scaffold` is READY.** Every downstream module is either RED or never reached
implementation.

**Scaffold detail (the one green thing):**
- Committed artifacts: `src/catan_rl/human_data/{record.py, orientation.py, ffmpeg.py,
  topology.py, topology.json}`, `tests/unit/human_data/{test_scaffold.py,
  test_orientation.py}`, `tests/fixtures/human_data/game1_openings.json` (desert=11),
  `data/human/strength_manifest.json`.
- Tests: prompt expected 59; there are now **92, all green** (coverage grew).
- `scripts/mine_phantom.py` exists as an ingest entrypoint (later slice); the harvest
  orchestrator (`batch`) was never built.

**Commit-SHA discrepancy — honest flag.** The scaffold-stage journal recorded the
foundation as banked at `c7938f6` and deliberately made **no new commit** ("empty/no-op
commit deliberately not created"). The **actual current HEAD is `734316e`**
("fix(human_data): harden board_cv orientation + resource firewalls"); `c7938f6`,
`3b83f8b`, `d6bccc0` are all present earlier in history (verified). So work *was* committed
past scaffold (through the board_cv hardening), but **no module past scaffold reached a
green gate.** The tree is coherent — the "commit: none" note applies only to the scaffold
stage, not the whole run.

---

## 2. Every GATE result (with numbers)

| Gate | Result | Numbers |
|---|---|---|
| **Stage-1 gate (g1)** | **SKIPPED** | blocked = `ingest, logparse, segment` (all RED); gate never ran |
| **Stage-2 gates (g2)** | **SKIPPED** | `s1Ready=false, s1Gate=false`; s2 blocked = `board_cv, openings, validate, glyph_anchor, batch`; never ran |
| **Gold gate** (harvest precondition) | **NOT MET** — `gold=false` | never evaluated; no clean end-to-end record produced |
| **Glyph firewall** (harvest precondition) | **NOT MET** — `glyph=false` | `glyph_anchor` never built |
| **Full harvest** | **GATED / did not run** | `harvest.ran=false` |

**No gate produced a pass number.** There are no WR / precision / recall / accept-rate
figures because no gate executed. There is no harvest-quality number to report — inventing
one would be dishonest.

---

## 3. Rejection-bias audit

**Not reached.** The rejection-bias audit runs only over a harvested corpus (accepted vs.
rejected games, checking rejections don't systematically drop a player / opening / board
class). Harvest never ran, so there is **no corpus, no #games, no #rejected, and no
rejection-bias audit.**

---

## 4. Exactly WHERE it halted and WHY

Five independent halt points, cascading Stage-1 → gate-skip → Stage-2 → harvest-gate. The
three Stage-1 halts are **real correctness BLOCKERs** (red-team counterexamples that would
corrupt the scoreboard); the Stage-2 halts are **infrastructure failures** (the implement
sub-agent died on spawn, three retries each).

### Stage-1 — three modules stuck after 4 resolve iterations each (real bugs)

1. **`ingest` — verify RED (test/code mock gap).** `_cmd_ingest` → `ingest_video(...)` now
   calls `resolve_ffprobe()` (added `d6bccc0`), but the committed test
   `tests/unit/scripts/test_mine_phantom.py` (`3b83f8b`) only monkeypatches
   `resolve_ffmpeg`, not `resolve_ffprobe`. On any box without `ffprobe` on PATH (confirmed
   here), both ETA tests fail with `FFmpegNotFoundError`. The slice is **RED (2 failing
   committed tests).** Fix for real (stub `resolve_ffprobe` or thread a resolver kwarg
   through `_cmd_ingest`) — **not** by loosening the test.

2. **`logparse` — red-team counterexample (fabricates a winner).**
   `parse_log(["rayman147:gg you won the game"], (...))` returns
   `winner="rayman147", kind="victory"` — **WRONG.** This is a *chat* line; must yield
   `winner=None` (§5.1). Cause: chat-firewall regex `_CHAT_LINE = r"^[\w.\-|]+\s*:(?:\s|$)"`
   (`logparse.py:186`) requires whitespace/EOL after the colon; when OCR drops the
   post-colon space (within this module's *own* demonstrated OCR-noise envelope), the line
   escapes the firewall, falls to the `"won the game"` victory branch (`logparse.py:349`),
   and latches a winner. Also fires on `"ThePhantom:you won the game noob"` and
   `"rayman147:lol i won the game"`. Existing chat tests all use `"handle: message"`
   (colon+space), so the suite stays green while emitting a **confidently-wrong winner into
   scoreboard-eligible records.** Fix: relax `_CHAT_LINE` for a missing post-colon space,
   keep the single-leading-token guard; failing test first.

3. **`segment` — red-team counterexample (false-splits one real game into two).**
   `segment_games` false-splits a legitimate game whenever the lingering `"Happy settling"`
   reset marker is re-OCR'd **after both seats have rolled once.** Cause: `segment.py:265`
   predicate `... or len(rollers_since_start) >= 2`; `rollers_since_start` clears only at a
   real start (`line 271`), stays saturated the rest of the game, so every lingering-reset
   re-OCR after round 1 opens a spurious new game. The docstring premise ("a lingering
   re-OCR never has both seats rolling yet") is false on the **flattened multi-frame
   stream.** Wrong two ways at once:
   - `seg0`: `winner=None, ended_by="cutoff"`, holds **both** opening-settlement lines,
     `is_scoreboard_terminal()=False` — the real win is demoted and dropped.
   - `seg1`: `winner="ThePhantom", ended_by="victory"`, `setup_lines=[]`, actors =
     `{ThePhantom}` only → `ruleset_ok()=False` — a hollow terminal, no board/openings, no
     opponent.

   This is the **§5.1 outcome-to-position mispairing the module exists to prevent, in
   reverse**, and it hits the **common case** (every completed game reaches a full round).
   Fix: gate the `>=2 rollers` separator on `event.text not in reset_texts_in_window` (a
   lingering re-OCR is text-identical by construction); care needed vs. test line 381.

→ **Stage-1 gate SKIPPED** — `ingest, logparse, segment` all blocked.

### Stage-2 — five modules, implement sub-agent DIED (infrastructure, not a code verdict)

- `board_cv` — impl agent died **iter 1** (3 attempts, all `null`). Partial code landed
  through `734316e` before the death.
- `openings` — impl agent died **iter 0** (3 attempts). **Never built.**
- `validate` — impl agent died **iter 0** (3 attempts). **Never built.**
- `glyph_anchor` — impl agent died **iter 0** (3 attempts). **Never built.** ← hard-gates harvest.
- `batch` — impl agent died **iter 0** (3 attempts). **Never built.**

→ **Stage-2 gates SKIPPED** (`s1Ready=false, s1Gate=false`, all five s2 modules blocked).

**Distinction that matters for the fix:** Stage-1 halts are *stuck-on-a-real-bug* (resolver
couldn't clear a BLOCKER in 4 iterations). Stage-2 halts are *the implement agent failing to
run at all* (process death, not a review verdict). Different remedies — see §7.

---

## 5. HARVEST status

**GATED. Nothing harvested.** `harvest.ran=false`.

- **#games extracted:** 0
- **#games rejected:** 0
- **Rejection-bias audit:** not produced (no corpus).

Harvest requires **BOTH** preconditions green and **neither is:**
- **Gold gate: `gold=false`** — no clean end-to-end record exists (Stage-1 never produced a
  gate-passing `ingest→logparse→segment` record; `openings`/`validate` never built).
- **Glyph firewall: `glyph=false`** — `glyph_anchor` was never built (impl agent died).

The run **correctly refused to scale an unprotected pipeline** (journal final line: "Not
scaling unprotected."). That is the right call: harvesting now — with the logparse
winner-fabrication bug and the segment false-split bug live, and **no glyph firewall** —
would produce a corpus of confidently-wrong outcome→opening pairings, exactly what this
pipeline exists to prevent.

---

## 6. Single-command resume state

Resumable from the top of Stage-1 — no partial-harvest state to clean up (nothing was
harvested):

```
node scripts/dev/human_data_build_wf.js
```

The workflow re-enters at the first blocked module. **Caveat:** the three Stage-1 BLOCKERs
are **deterministic** and will re-halt at the same place unless fixed first (they are real
counterexamples, not flaky). The Stage-2 deaths *may* clear on a clean re-run if transient;
if the impl agent keeps dying at iter 0 across modules, that is an environment/harness
problem, not per-module.

**Working tree at report time:** HEAD `734316e`, clean except out-of-scope noise
(`.claude/scheduled_tasks.lock` modified; untracked `data/exit/`, `src.mp4`,
`scripts/export_dice_vectors.py`, `scripts/record_conformance.py`). None are pipeline
artifacts; none were touched by this build.

---

## 7. Human-decision items (your call at 8am)

1. **Glyph classifier — HARD GATE, needs your decision.** `glyph_anchor` was **never
   built** (impl agent died iter 0 ×3). It is the **glyph firewall that hard-gates the safe
   full harvest** — without it, harvest cannot legitimately run. Decide: (a) retry the
   autonomous build of `glyph_anchor` in isolation, (b) hand-build/spec it yourself, or
   (c) define an interim manual-verification firewall. **No harvest until this is green.**

2. **Stage-2 implement-agent deaths — infra triage.** Five modules died at spawn (`null
   (agent died)`), four at iter 0 (never wrote a line), three retries each with no help.
   This reads as a harness/environment failure (sub-agent process death), **not** five
   independent code bugs. Decide whether to investigate the runner (timeouts, memory, spawn
   limits) before the next launch, so the next overnight run doesn't lose all of Stage-2 the
   same way.

3. **Fix the three Stage-1 BLOCKERs first (cheapest unblocks, all prerequisites for the
   gold gate).** Concrete fixes in §4: (i) `ingest` — mock `resolve_ffprobe`;
   (ii) `logparse` — relax `_CHAT_LINE` for missing post-colon space; (iii) `segment` — gate
   the `>=2 rollers` split on `reset_texts_in_window` membership. Each needs a **failing
   test first**; none should be "fixed" by loosening an existing test.

4. **Do NOT harvest yet.** Both gates are red. Any corpus produced now would carry the
   winner-fabrication and false-split bugs and have no glyph firewall.

---

**One-line throughline:** The scaffold is solid and green (92 tests), but the pipeline never
cleared Stage-1 (three real, scoreboard-corrupting bugs the resolver couldn't fix in 4
tries) and Stage-2 never ran (the implement agent died on every module), so **both harvest
gates are red and zero games were harvested — the run correctly refused to scale an
unprotected, unverified pipeline.**
