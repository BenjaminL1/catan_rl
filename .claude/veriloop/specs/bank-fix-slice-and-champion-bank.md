# Spec: bank-fix-slice-and-champion-bank — land the leak fix without retiring the tree behind the champion

**Status: DRAFT (2026-07-29) — awaiting owner ratification.**

**Feature in one line.** Re-slice the gate-green `feat/human-devcard-picker-and-bank` work into
three ordered commits so the finite-bank correctness fix lands *without* dragging the engine
re-pin that retires the tree the accept-gate clause-(a) result was measured on — then bank the
pointer-arch champion durably, record clause (a) in a tracked file, and void the leak-era
evidence with its magnitude written down rather than merely stamped.

## Why re-slice at all

`ba8f8ac` is gate-green (`mypy` 0, `ruff` 0, 2980 pytest passed) and pushed, 14 files
`+939/-101`. It bundles three unrelated things:

1. the **finite-bank correctness fix** (`gui/view.py`) — a live violation of the ratified
   spec-009 invariant `bank[R] + Σ hands[R] == 19` (`engine/board.py:246`) on a path that
   feeds a policy observation (`policy/obs_encoder.py:375-380`);
2. the **human dev-card picker routing fix + conservation guard** (`scripts/play_vs_model.py`);
3. a **HUD window resize** (`engine/board.py`, 1000×800 → 1200×900) which changes the engine
   tree and therefore forces `eval/engine_parity.py` to re-pin `261098d190c8` → `3388b69026cb`.

`runs/logs/xarch_u500_vs_v11_n600.log` records `engine-parity: engine=261098d190c8` for the
clause-(a) result (WR 0.7500, Wilson LB 0.7138, n=600). Merging (3) means no forward checkout
reproduces the tree that number was measured on. (1) and (2) touch nothing under
`src/catan_rl/engine/` and need no re-pin — so the correctness fix can land while the
provenance stays intact.

---

## Decisions (binding once ratified)

### D1 — THREE commits, and the ORDER is load-bearing

**The ordering constraint is a correctness requirement, not a style preference.** On `main`,
`engine/player.py:452` branches Year-of-Plenty on `isAI`: the AI branch calls
`bank_draw` gated on supply (`:458-462`) — **correct**; the `else` branch opens the GUI picker,
which on `main` does a bare `self.resources[res] += 1` with no draw — **a mint**.
`discardResources` (`:612`) has **no `isAI` branch at all**, which is why the discard leak is
the only bank defect live today. Therefore **flipping `isAI = False` before the `view.py` fix
lands would ADD a minting bug that does not currently exist.**

- **C1 — `fix(gui): recirculate/draw on the human resource picker`** (engine tree UNCHANGED)
  - `src/catan_rl/gui/view.py` — the bank hunks ONLY: `bank_recirculate` ×2 (discard;
    YoP cancel/revert), `bank_draw` ×1, and `bank_can_supply` at **three** sites
    (`:786` title hint, `:812` grey-out, `:881` the load-bearing click gate).
  - `tests/unit/gui/test_human_picker_bank.py`
  - Hunk separation is clean: the bank hunks sit inside `get_resource_selection`, ~178 lines
    from the layout constants, sharing no diff hunk. Every click in the test derives from
    `board.width`/`board.height`, never a literal, so the file is window-size-independent and
    **4 of its 7 tests fail on `main` at 1000×800** — it is a real bug reproduction.

- **C2 — `fix(playtest): route the human seat to the picker + conservation guard`**
  (engine tree UNCHANGED). **MUST land at or after C1, never before.**
  - `scripts/play_vs_model.py` — the permanent `isAI = False` flip in `HumanVsBotEnv.reset`,
    `_make_bank_conservation_reporter`, the `check_bank(env)` calls (after reset and after
    every step), and the `"bank_ok"` record key.
  - `tests/unit/scripts/test_play_vs_model.py`
  - Its only resize touchpoint is `intended_hex_size=(int(env.game.board.width),
    int(env.game.board.height))` — read **live**, so it is correct at either window size.
    `MOVE_LOG_LINES = 6` already exists unchanged on `main` (`gui/view.py:28`).

- **C3 — `feat(gui): 1200x900 window + HUD layout`** (engine tree → `3388b69026cb`)
  - `src/catan_rl/engine/board.py` (resize), `src/catan_rl/gui/view.py` (layout constants:
    `HAND_PANEL_LINE_HEIGHT` 18→20, `MOVE_LOG_RECT` y 695→745),
    `src/catan_rl/eval/engine_parity.py` (re-pin **plus D3 below**),
    `src/catan_rl/eval/cross_arch.py` (docstring only),
    `src/catan_rl/replay/recorder_loop.py` (default `(1000,800)`→`(1200,900)`),
    `tests/unit/gui/test_hand_panel.py` (asserts `y >= 745` — genuinely coupled; it FAILS
    against `main`'s layout), `tests/unit/engine/test_topology_stability.py`, and the docs.

All **14** changed paths are assigned. The earlier 12-of-14 accounting left
`test_hand_panel.py` and `test_play_vs_model.py` unplaced, which is precisely how a re-slice
drops a pin.

### D2 — New branch, not a force-push

Build the three commits on `feat/human-devcard-picker-and-bank-v2`, leaving
`origin/feat/human-devcard-picker-and-bank` @ `ba8f8ac` intact. Rationale: there is **no
attestation record** for this branch in `.claude/veriloop/history/`, so `origin` is currently
the only durable pointer to a gate-green tree, and the branch is checked out in the worktree
`.catan-rl-v2-veriloop/human-devcard-picker-and-bank` (git refuses to rewrite a branch checked
out elsewhere). Write the attestation against the new head. Delete the old remote ref only
after `main` has the work.

### D3 — Restore the cleanliness probe in `engine_parity.py`; KEEP working-tree hashing

The branch deleted `main`'s dirty check (`engine_parity.py:91-97`) on the grounds that
working-tree hashing subsumes it. **It does not.** The two answer different questions:
working-tree-hash == pin says *the content is the validated content*; a clean tree under the
guarded paths says *the content is in a commit*, so `engine-parity: engine=<sha>` names a tree
reachable from history. Without the probe, the guard can pass on a tree that exists only as a
loose object the guard's own `git add` just wrote, and stamp a 600-game result on a tree no
commit contains. Keep both. Committing the engine change and the pin together (which C3 does)
leaves the tree clean, so both checks pass.

### D4 — Bank the champion, and repoint ASYMMETRICALLY

- Produce `runs/anchors/ptr_v1_u500.pt` via `bank_anchor` with an explicit `dest`
  (`checkpoint/manager.py:518-521` — non-destructive, policy-only). This is a genuine
  durability gain, not a copy: the source carries an external league sidecar
  (`ckpt_000000500.pt.refs.json`), and `save_policy_only` writes none, so the banked artifact
  outlives the run directory.
- **Repoint** the two `DEFAULT_CKPT` literals — `scripts/play_vs_model.py:107` and
  `scripts/opening_sweep.py:206` — at the banked path. This IS a tracked diff.
- **Do NOT repoint `configs/selfplay_pointer_arch_v3.yaml:113,153`.** That config describes a
  run that already happened (v3 soft-stopped at u200); repointing it would make a config stop
  describing its own run.
- `bank_anchor` currently **replaces** rather than merges `payload.metadata`
  (`manager.py:547`). Merge it, and stamp an explicit arch key, so a mixed-arch
  `runs/anchors/` fails with a diagnosis instead of a state-dict shape error. (Pre-existing
  drift: the legacy `v7`–`v11` anchors are **not** loadable by the current default policy
  shape; the `ptr_` prefix is a convention, not a check.)

### D5 — Record clause (a) in a TRACKED file

The clause-(a) numbers exist in **no tracked file** — only in gitignored `runs/logs/`. Add to
`docs/plans/v2/pointer_arch_lineage.md` under the accept gate: clause (a) **SATISFIED** —
u500 vs `v11_cand` WR **0.7500**, Wilson 95% CI [0.7138, 0.7830] (LB 0.7138), n=600 (300/seat),
0 truncations, 0 rules violations, on engine tree `261098d190c8` / board_geometry
`70813dcf76fd`; re-pinned to `3388b69026cb` on 2026-07-28 (pure lattice translation).
Also record that `ptr_v1_u500.pt` was banked. The `engine_parity.py` docstring log records the
*mechanism*; the lineage doc must record the *result*.

### D6 — Void the leak-era evidence, with the magnitude and a carve-out

- Stamp the **playtest-derived** sections of `docs/plans/opening_deficit_verdict.md` as
  `VOID — SUBSTRATE`, naming **both** defects: the bank leak, *and* that the human's YoP and
  Monopoly were resolved by `np.random.choice`. Write the measured magnitude in rather than
  asserting voidness: **17 cards** leaked in `20260727T155739_seed0.json` (WOOD 11 of a
  19-card supply) and **24** in `20260727T235203_seed329853734.json` (WOOD 9, SHEEP 8, ORE 5,
  BRICK 2). This matters because the doc indicts the policy for "0 cities in 58 turns" and
  "6 of 12 bank trades buying ore" — cities cost 3 ore, and `resolve_bank_production`
  (`engine/game.py:40-46`) denies production to **both** seats when supply is short.
- **Carve the ≥40% ore-substitution bar at `:63-66` OUT of the void scope.** It is sweep
  machinery, not a playtest finding, and the surviving kill at `:89-99` cites it — voiding it
  would leave that kill resting on a void premise. **Keep `:89-99`** (all three kills are
  sweep-derived and touch no playtest game).
- Fix the two line-number citations a banner would shift —
  `.claude/veriloop/specs/human-opening-reference.md:15` and the doc's own `:91` — or convert
  them to heading anchors.
- The three games in `runs/human_playtest/games.jsonl` also span **two information regimes**
  (games 0 and 1 carry no `hud` key); name that alongside the bank defect, or a later reader
  will fix the bank and believe the evidence is restored.
- Annotate `games.jsonl` via an **append-only sidecar keyed on `replay_path`**, not by editing
  rows in place, with game 0 explicitly marked "no replay".
- **Do not write the string `play_vs_v8` into any tracked file outside
  `.claude/veriloop/specs/`** — `test_play_vs_model.py::test_no_old_name_reference_survives`
  greps for it and would go red.

### D7 — Amend the existing spec; do not supersede it

`.claude/veriloop/specs/human-devcard-picker-and-bank.md` describes what shipped in file and
decision terms and cites no commit SHA, so the re-slice does not falsify it. Amend its
implementation notes to record the three-commit shape, correct note 2 (working-tree hashing
does **not** subsume the dirty probe — see D3), and flip `Status: DRAFT` → RATIFIED. A
permanently-DRAFT spec whose content shipped is the exact drift these specs exist to prevent.

---

## Non-goals

- **No `bank_ok` consumer.** Nothing reads the key yet; D6's hand-annotation is the manual
  stand-in for the missing scoreboard filter. Say so out loud rather than letting D6 look like
  closure.
- **No re-ratification of accept-gate clause (b).** Separate owner decision; its tool is
  finished and unmerged on `feat/opening-scoreboard`.
- **No engine game-rule change.** The finite-bank rule is being *enforced*, not altered.
- **No touching the stopped training runs or the four `launchd` agents.**
- **No fix for the ungated bank-trade receive side** (`engine/player.py:556-584`) — see the
  first RISK below; this is a scope question for the owner, not a silent omission.

## Acceptance criteria (the `/dev-loop` gate is the authority on checks)

1. Full gate green per the dev-loop's real exit codes (typecheck · lint · rust fmt · unit tests).
2. **Tree-equality check on the re-slice:** `git diff ba8f8ac <new-head>` is empty **except**
   the deliberate additions in D3, D4, D5, D6, D7. If it differs anywhere else, the gate is
   **re-run, not inherited** — "`ba8f8ac` was green" transfers only under tree equality.
3. **C1 in isolation:** at C1, `src/catan_rl/engine` hashes to `261098d190c8` and
   `board_geometry.py` to `70813dcf76fd`, and the existing
   `tests/unit/eval/test_cross_arch.py::test_engine_parity_holds_at_head` passes. Gate C1 in a
   **clean checkout** — a dirty tree carrying C3's `board.py` edit turns that test red for the
   right reason.
4. **C1 reproduces the bug:** `test_human_picker_bank.py` fails against `main`'s source and
   passes at C1, covering all four bank paths — YoP draw, bank-empty decline, YoP cancel/revert,
   discard recirculate.
5. **No intermediate state mints or leaks:** conservation holds at C1, at C2, and at C3 across a
   scripted human YoP, Monopoly, and discard.
6. **Banked-anchor equivalence** (not a byte hash — `bank_anchor` round-trips through
   `apply_migrations`): load both, assert identical `policy_state_dict` key sets, `torch.equal`
   on every tensor, and `update_idx`/`global_step` preserved.
7. Clause (a) numbers and the bank appear in a **tracked** file (D5).
8. Docs sync per CLAUDE.md §6, including the two shifted citations (D6) and the amended spec (D7).

---

## RISKS — open challenges, NOT cleared

**Same bug class, one call away, and it can kill a live game.** `engine/player.py:556-584`
`trade_with_bank` calls `bank_draw` on the receive side with **no `bank_can_supply` gate**, and
`board.py:226-240` asserts on underflow. The picker greys only `mode == "YOP"`
(`view.py:812`); the BANK receive swatch gets no signal. A human bank-trading for a depleted
resource raises `AssertionError` **inside `env.step`**, which propagates past
`_make_bank_conservation_reporter` (it wraps only the guard call) — the interactive game dies
and the replay, written only after the loop ends, is lost. C1's commit message will claim the
picker is bank-correct; with BANK mode ungated that claim is half true.

**Pre-mortem (premise review, carried verbatim in substance).** A year on, the human-vs-bot
cohort is empty and nobody can say whether the bot was ever as bad as it looked. The clean
cohort is size zero because `bank_ok` shipped in a commit titled after a window resize, that
commit sat unmerged beside `feat/opening-scoreboard` and `feat/human-opening-reference` (both
already unmerged for months), and every game recorded in the interim carries no `bank_ok` key —
which this very change's own `CLAUDE.md` rule reads as "predates the check, not that it passed."
Meanwhile the one thing genuinely fixed went unmeasured, because D6 was executed as a blanket
stamp rather than a measurement, and the strongest available rehabilitation of the champion was
replaced with the word VOID. The pins meant to catch this were loosened in the same change: the
re-pin's justification is prose in a docstring, the `--old-arch new` equivalence self-check was
never run on the resized board, and the first-ever exercise of the guard's "re-validated"
clause taught the repo that re-validated means argued.

**Argue the other side — and on two of three decisions this is STRONGER than the plan.**
*Merge `ba8f8ac` whole, today.* The split's payoff is avoiding one constant bump plus a log
line that is already written, reviewed, and pushed; the resize — treated as the dangerous half —
is the *better-covered* half (pixels enter the legacy encoder only as live dict keys, never as
feature values; renumbering is empirically pinned by `test_topology_stability.py`, which is
green at both window sizes). Against that small benefit the split imposes real costs: it
separates the fix from its own proof, its own guard, and its own provenance marker, and it
requires three units of git ceremony. And the opportunity cost is the strongest part: clause (a)
has already passed, clause (b)'s tool is **finished and unmerged** on `feat/opening-scoreboard`
(+1695), `feat/human-opening-reference` is likewise finished and unmerged (+6345), both training
runs are stopped, and this repo's own most recent ruling ranks *aggregate midgame instrumentation
over existing corpora — zero new games, no baseline problem* above any further opening metric.
The fork is one merge plus one metric run from formal acceptance, which is the decision that
legitimizes crowning `ckpt_500` at all — the thing D4 does by filename without the record.
