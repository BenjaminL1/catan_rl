# Spec: playable-and-recordable — make a human-vs-champion game valid, fair, and attributable

**Status: RATIFIED (2026-07-31) — BINDING. Owner delegated the decisions for this slice
("fix all issues how you see fit until it is ready to play and record the games… do not ask
for my permission or input"). Ratified under that delegation.**

**Feature in one line.** Land the minimum set of fixes that makes a recorded human-vs-champion
game **valid** (no rule corruption), **fair** (no seat has a capability the other lacks),
**survivable** (a misclick cannot destroy an hour of play), and **attributable** (the record
names the exact policy and code it was played against).

## Why this slice exists

Everything below is already diagnosed and verified. Nothing here is new investigation; this is
the closing of known gaps so a game can be played and analysed. Three prior games are void.
The goal is a fourth that is not.

---

## Decisions (binding)

### D1 — Land the finite-bank fix in the human picker. ORDER IS LOAD-BEARING.

`src/catan_rl/gui/view.py` on `main` has **zero** calls to `bank_recirculate` / `bank_draw`.
The human discard picker does a bare `player.resources[res] -= 1`, destroying cards. Measured
leakage in the two replayed games: **17 and 24 cards**. Once a resource drains,
`engine/game.py` `resolve_bank_production` returns `(0, 0, avail)` when both seats are owed and
supply is short — production stops for **both** players. And `bank[r]/19` is in the bot's
observation, so the corruption is fed to the policy.

Apply, inside `get_resource_selection` only:
- DISCARD branch → `bank_recirculate({res: 1})`
- YoP grant → `bank_draw({res: 1})`, gated on `bank_can_supply`
- YoP cancel/revert → `bank_recirculate` whatever it already drew

All three accessors already exist on `main` (`engine/board.py`). **No engine change. No
`PINNED_ENGINE_TREE` re-pin.** The reference implementation is
`feat/human-devcard-picker-and-bank` @ `ba8f8ac`; take only its `gui/view.py` bank hunks and
`tests/unit/gui/test_human_picker_bank.py`. **Do NOT take** the window resize
(`engine/board.py`), the layout constants, or the `eval/engine_parity.py` re-pin — those retire
the engine tree the clause-(a) n=600 result was measured on, and buy nothing for playability.

### D2 — Route the human to the picker. MUST land at or after D1, never before.

`env/catan_env.py` stamps `opp.isAI = True` on every reset, so the human's Monopoly and Year of
Plenty are resolved by `np.random.choice` (`engine/player.py`) instead of the picker. Fix by the
permanent `isAI = False` flip in `HumanVsBotEnv.reset` (`scripts/play_vs_model.py`).

**Ordering is a correctness requirement, not style.** On `main` the AI branch of `play_devCard`
already bank-draws correctly; the GUI branch does not. Flipping `isAI` *before* D1 opens the
picker onto an ungated grant path and **adds a minting bug that does not exist today**.

Land with it, from the same reference branch: `_make_bank_conservation_reporter`, the
`check_bank(env)` calls after reset and after every step, and the `"bank_ok"` key in the game
record. A missing `bank_ok` means the game predates the check, not that it passed — so the key
is what makes a clean game *provable*.

### D3 — Remove the human's pre-roll dev-card window (fairness, until R1 ships)

`scripts/play_vs_model.py` calls `self._human_pre_roll()` before `game.rollDice()`. The bot has
no counterpart: `env/masks.py` sets **only** `ROLL_DICE` when `roll_pending`, and
`env/catan_env.py` no-ops anything else. So the human may play a Knight before rolling — move
the robber off their own number, or block one — and the policy structurally cannot, in this
harness or in its entire training history.

This is the single largest bias in a playtest: it makes every human win unattributable between
"the policy is weak here" and "the bot was not allowed to play its Knight."

**Delete the call.** Both seats become post-roll-only, matching the ruleset the champion was
actually trained on. This is explicitly a stopgap: `.claude/veriloop/specs/preroll-dev-cards-r1.md`
(RATIFIED) gives *both* seats the window and is the permanent fix; it requires a retrain and is
not in this slice. Record the stopgap in the game record as `"preroll": false` so a future reader
can separate these games from post-R1 ones.

### D4 — A misclick must not destroy the game

`env.step(action)` sits in a bare `while` loop and **both** artifact writes happen after the
loop. Five GUI paths call `sys.exit(0)` on `pygame.QUIT`. So closing the window mid-game
discards everything with no partial write. `Metadata` already carries a `partial` field that is
hardcoded `False` and never set.

Wrap the loop body in `try/except BaseException` — **not** `except Exception`, because
`SystemExit` is a `BaseException` and the `sys.exit(0)` paths would slip through — and fall
through to the existing write block with `partial=True`.

### D5 — The record must name what it was played against

`Metadata` carries `ckpt_path`, `seed`, `mode`, `sims`, `clairvoyant`, `reveal_bot`,
`recorded_at_utc` — but **no git SHA and no checkpoint content hash**. `ckpt_path` points into a
gitignored run directory under `keep_last_n: 6` rotation, so tomorrow it may hold different
bytes or nothing. Add `git rev-parse HEAD` and a checkpoint `sha256` as additive keys.

Additive only: existing readers must keep working, and a missing key means the game predates the
field.

### D6 — Bank the champion so the path cannot rot

Copy `runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt` to
`runs/anchors/ptr_v1_u500.pt` via `bank_anchor` (`checkpoint/manager.py`, policy-only slim,
non-destructive with an explicit `dest`), and repoint `DEFAULT_CKPT` in
`scripts/play_vs_model.py` at the banked path. `runs/` is gitignored, so the copy is an ops
step; the `DEFAULT_CKPT` repoint is the tracked part.

Do **not** repoint `configs/selfplay_pointer_arch_v3.yaml` — that config describes a run that
already happened, and editing it would make it stop describing its own run.

---

## Non-goals

- **The window resize and the engine-parity re-pin.** Cosmetic; they cost the clause-(a)
  provenance and buy no playability.
- **Pre-roll for the bot** — that is `preroll-dev-cards-r1.md`, and it needs a retrain.
- **The BC seed non-determinism** (`bc/dataset.py` seeds only `np.random` while `engine/dice.py`
  draws its dice seed from the unseeded stdlib `random`). Real, inside the pinned engine, and
  irrelevant to playing a game by hand.
- **No training runs, no BC regeneration.**

## Acceptance criteria (the `/dev-loop` gate is the authority)

1. Full gate green on real exit codes.
2. `PINNED_ENGINE_TREE` unchanged; `src/catan_rl/engine/` untouched.
3. Conservation holds across a scripted human YoP, Monopoly, and discard — and the pin FAILS
   against the pre-change tree.
4. A bank-unsupplyable YoP pick is not grantable; cancelling a partial YoP restores the bank.
5. `isAI = False` is live for the human seat, and no ordering exists where the flip precedes D1.
6. No pre-roll window is reachable for either seat; the record carries `"preroll": false`.
7. A simulated mid-game `SystemExit` still writes both artifacts with `partial=True`.
8. The record carries a git SHA and a checkpoint sha256.
9. `--self-test` passes end-to-end headless.

## Known limitation, recorded not fixed

The champion predates D3 of `bc-coverage-and-bank-legality`: `CURR_EXTRA_DIM` went 5→6, so an
observation slot that was strictly `0.0` throughout its training now carries the discard-owed
count. The tensor shape is unchanged, so the checkpoint loads and plays — but at discard nodes
it multiplies a live value by a weight that never received gradient. Games played now are sound
for **bug-hunting and qualitative feel**; they are **not** a clean strength measurement. That
resolves only when the policy is retrained under the current observation.
