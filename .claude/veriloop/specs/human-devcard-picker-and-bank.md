# Spec: human-devcard-picker-and-bank — let the human pick, without minting resources

**Status: DRAFT (2026-07-28) — awaiting owner ratification.**

**Feature in one line.** Let the human choose their own Monopoly / Year-of-Plenty
resources through the picker that already exists — and fix the finite-bank accounting on
those picker paths, which is broken today and would be made worse by the obvious fix.

## The reported bug is a flag, not a missing feature

`catanGameView.get_resource_selection(player, mode, num_to_select=1)`
(`src/catan_rl/gui/view.py:726`) already exists, already handles Monopoly (1 pick) and
Year of Plenty (2 picks), and `game.boardView` is already the live view at both
`play_devCard` call sites (`scripts/play_vs_model.py:431`, `:503`) because `_ViewWindow`
swaps it in (`:257`). **Nothing needs to be built.**

The base `player.play_devCard` branches on `self.isAI` — `np.random.choice` when true
(`engine/player.py:452`, `:480`), the GUI picker when false. The human occupies the ENV's
**opponent** seat, and `env/catan_env.py:360` stamps `opp.isAI = True`. `heuristicAIPlayer`
does **not** override `play_devCard` (verified), so the base method runs and takes the AI
branch. That is the whole bug.

## The trap in the obvious fix — and a live bug next to it

**Routing the human into the picker as-is would mint resources from nothing.** The picker's
YOP branch is `player.resources[clicked_res] += 1` (`view.py:838-844`) with **no
`bank_draw` and no availability check**, while the AI branch it replaces does both
(`player.py:460-462`). That breaks the spec-009 invariant `bank[R] + Σ hands[R] == 19`
(`engine/board.py:246`) — and the bank is **in the bot's observation**
(`policy/obs_encoder.py:375-380`), so a corrupted bank feeds the policy a corrupted global
feature for the rest of the game.

**The same bug is already live on discard.** `_opponent_discard` (`play_vs_model.py:351`)
calls base `PlainPlayer.discardResources`, whose picker branch does
`player.resources[clicked_res] -= 1` (`view.py:830-836`) with **no `bank_recirculate`** —
while `heuristic.py:278`, `random_ai.py:76` and `env/catan_env.py:452/656/900` all
recirculate. Every human discard played so far has leaked cards permanently out of the
bank, including in the games already recorded.

**Why nobody noticed:** `assert_conservation` is wired only into
`human_data/engine_bridge.py:282` and `conformance/recorder.py:401`. It is never called
from `scripts/play_vs_model.py` — the one driver a human actually watches has no bank guard.

## Decisions (binding for the build)

### D1 — Set the human seat's `isAI = False` PERMANENTLY, not temporarily
A temporary save/restore around `play_devCard` was the first instinct and is **worse**: it
adds an exception-leak path, a `finally`, and an interaction with `_ViewWindow` nesting. If
the restore is ever skipped, the seat stays `isAI=False` inside the bot's deepcopied search
rollouts, where `_HeadlessView.__getattr__` (`engine/game.py:57`) returns `None` — so the
dev card is refunded and never played, and **the bot's MCTS silently models an opponent who
cannot use YoP or Monopoly**, with no log line.

Every other `isAI` reader was checked for reachability from this harness and **all are
unreachable**: `game.py:187/203` (only under `render_mode == "human"`; the env builds
`render_mode=None`, `catan_env.py:344`), `game.py:402` (only reached when `dice != 7`),
`game.py:522` (`playCatan`), `view.py:478` (guarded by `human_player is None`, always set at
`play_vs_model.py:172`), and MCTS clones (neither `heuristic.move` nor the snapshot driver
reaches `player.py:452/:480`). The permanent flip therefore has the **smaller** blast radius.

### D2 — Fix the picker's bank accounting (required, not optional)
- **YOP** (`view.py:838-844`): draw from the bank via `bank_draw({res: 1})` and gate on
  `resourceBank.get(res, 0) > 0`, matching `player.py:460-462`. A resource the bank cannot
  supply must not be grantable.
- **DISCARD** (`view.py:830-836`): `bank_recirculate({res: 1})`, matching every other
  discard path.
- **The YOP cancel/revert path** (`view.py:816-818`) must `bank_recirculate` whatever it
  reverts, or cancelling a partially-picked YOP leaks in the other direction.
This lands **whether or not** a human ever sees a menu: adding UI on top of an uncorrected
mutation path only produces wrong games faster.

### D3 — Add the conservation pin to the human driver
Call `board.assert_conservation(players)` in `scripts/play_vs_model.py` at end-of-turn (or
per `env.step`). This is the single check that catches both the original bug and the minting
bug the fix would otherwise introduce. Cheap: a 5-key sum.

### D4 — Expand the window; the HUD is genuinely over budget
The owner's read was right and the first analysis answered the wrong question (it checked
the *menu*, which does not overlap: menu y 325-475 vs `MOVE_LOG_RECT` y 695-795). The **HUD**
is the constrained thing: `_draw_hand_panel`'s own comment (`view.py:512-513`) records that
at the previous 20px leading a revealed bot panel at y=460 "would extend to y=824 in an
800px window" — the 18px leading shipped as a workaround for an already-overflowing layout.
Enlarge the window (`engine/board.py:139`, currently `1000, 800`) enough to restore
comfortable leading and fit the panels + move log without compression.
**Constraint:** board geometry is derived from `self.size`, so every hex/vertex/edge pixel
coordinate moves. The topology export, any pixel-space test, and `board_geometry.py` must be
re-checked — treat this as the risky half of the change, not a cosmetic tweak.

### D5 — Testable headlessly; pin it
`tests/conftest.py` forces `SDL_VIDEODRIVER=dummy` suite-wide, and `get_resource_selection`
loops on `pygame.event.get()` — a test can `pygame.event.post` synthetic `MOUSEBUTTONDOWN`
at the known rects. Pin a scripted human YOP and Monopoly with `assert_conservation` before
and after.

## Non-goals
- No new picker UI — it exists and already supports both cases.
- No change to how the BOT chooses dev-card resources (the AI branch is correct).
- No fix for `player.py:631` passing `discarded_resources` unconditionally to `log_discard`
  (`None` under `_HeadlessView` would `TypeError`) — pre-existing, flagged not touched.
- No engine game-rule change: the finite-bank rule is being *enforced*, not altered.

## RISKS — open challenges from the premise review (NOT cleared)

**Pre-mortem.** The menus ship and look right; the wreck is in the accounting. Every Year of
Plenty mints two cards from nothing, `bank[R] + Σ hands[R] == 19` is false for the rest of
each game, and because the bank is in the observation the bot plays on a corrupted feature.
It goes unnoticed because no test covers a pygame path and the human driver never asserts
conservation. The damage compounds through `_HumanGameRecorder`: human-vs-bot games are the
input to the human scoreboard, and a 15-VP race turns on less than two free cards. D2 and D3
exist precisely to make this failure impossible rather than merely unlikely.

**Strongest opposite case.** *Do not touch `isAI` at all.* It is a global mode flag meaning
"this seat is driven programmatically"; lying about it makes one flag mean two things inside
a call frame. The harness owns both call sites, so it could prompt and apply the grant itself
through `bank_draw`, or `play_devCard` could take an explicit chooser argument — either keeps
`isAI` honest. This is close on merit; the reviewers split, with the correctness lens finding
the permanent flip's blast radius empirically *smaller* than the alternatives. It is
defensible **only if D2 ships with it**.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `make test-unit`).
2. **Picker pin:** a human Monopoly opens the selector and applies the chosen resource; a
   human Year of Plenty opens it and applies exactly two chosen resources — neither uses
   `np.random.choice`.
3. **Conservation pin (the load-bearing one):** `assert_conservation` holds before and after
   a scripted human YOP, Monopoly, and DISCARD. This must FAIL on today's code.
4. **Bank-empty pin:** a YOP pick the bank cannot supply is not grantable.
5. **Cancel pin:** cancelling a partially-picked YOP restores the card, clears
   `devCardPlayedThisTurn`, and leaves the bank exactly as it was.
6. **Search-integrity pin:** the permanent `isAI=False` does not change what a deepcopied
   MCTS clone does — no clone reaches `player.py:452/:480`.
7. **Layout pin:** at the new window size, neither hand panel nor the move log overflows,
   and board geometry (topology export / pixel-space tests) still passes.
8. Read-only w.r.t. training: nothing under `runs/train/**`; no game-rule change.
