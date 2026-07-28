# Spec: playtest-hud-and-log — show the public facts the real game shows

**Status: RATIFIED (2026-07-27) — implemented on `feat/playtest-hud-and-log`.**

**Feature in one line.** Show knights-played and longest-road length for both players, and
add an on-screen move log, so the playtest stops being *harder than real Catan* by
withholding facts a player at a table can see.

## Framing (binding)

These are **PUBLIC** facts: knights are played face-up, roads are visible on the board, and
an opponent's turn is watched as it happens. This does **not** conflict with the blind-bot
leak fix, which hides the bot's **private** hand contents (resource types, hidden dev-card
types). Withholding public facts biases the playtest toward the bot — the same error as the
DISCARD/YOP over-blinding reverted in `6f900e9`.

## Corrections to the original plan (recon overturned three assumptions)

* **The log already exists — in stdout.** `print(f"\n[BOT] {_describe_bot_move(action)}")`
  (`scripts/play_vs_model.py:1210`) plus the VP line at `:1234` is already a complete
  per-decision bot transcript. The gap is that it is in the terminal, not the pygame window.
* **The "events are lost inside one env.step" premise was inverted.** One `env.step` is
  exactly ONE bot action (`env/catan_env.py:421-547`) and the harness repaints after each
  (`play_vs_model.py:1226`); it is `END_TURN` that folds the *human's* turn via
  `_run_opponent_turn` (`catan_env.py:511`). The real cause is that `broadcast_message`
  renders only **3 of 12** event types (`gui/view.py:88-94`) — BUILD, MOVE_ROBBER, STEAL and
  MONOPOLY are emitted and never drawn — in a single overwritten line with no history.
* **The `maxRoadLength` staleness bug does not exist.** `check_longest_road` recomputes ALL
  players (`engine/game.py:434-435`), and every mutating path calls it (bot settlement
  `catan_env.py:582`, bot road `:588`, road-builder `:494`, EndTurn `:505`; human
  settlement/road `play_vs_model.py:450/453`, dev plays `:391/:461`, turn end `:414`), while
  `build_road` self-recomputes (`engine/player.py:101-102`) covering setup. City correctly
  skips it.

## Decisions (binding for the build)

### D1 — Knights and longest road, both players, both reveal modes
Add to `hand_panel_lines` (`gui/view.py:25`) in **both** the `reveal=True` and
`reveal=False` branches: knights played and current longest-road length.
- `player.knightsPlayed` (`engine/player.py:35`, incremented `:437`) is live and correct.
- **Wording constraint:** do NOT label the line `Knight: N` — `test_no_dev_card_type_is_reachable`
  (`tests/unit/gui/test_hand_panel.py:59`) asserts no dev-card type string is reachable in the
  blind panel and would trip. Use e.g. `Knights played: N`, and update that test's allow-list
  deliberately rather than by accident.
- **Test fixture:** `_FakePlayer` (`test_hand_panel.py:34-50`) has neither attribute; every
  test in that file will `AttributeError` until it gains them.
- **Layout:** the revealed human panel grows 14→16 lines, moving its bottom to y≈379 against
  the BANK TRADE button at y=400 (`view.py:359`) — 21px of clearance. Verify, do not assume.

### D2 — Compute road length LIVE
Use `p.get_road_length(board)` for display rather than the cached `maxRoadLength`. The cache
is in fact fresh on all traced paths (above), so this is belt-and-braces, not a bug fix — but
it is affordable and strictly truthful. **Cost is verified, not assumed:** `displayGameScreen`
is event-driven, not per-frame — `_animated_pick` calls it once and caches
`base = self.screen.copy()` (`view.py:319-320`), and the `clock.tick(30)` loop only blits that
cache. The panel therefore renders once per pick. The same DFS is already paid twice per build
inside MCTS at 400 sims/move.

### D3 — Feed the log from the ACTION, not the broadcast stream
Render the log from `_describe_bot_move(action)` (`play_vs_model.py`, already computed at
`:1210`) plus the human's own resolved actions. **Do NOT subscribe to the broadcast for v1.**
Three reasons, each disqualifying on its own:
1. **`PLAY_KNIGHT` and `PLAY_ROAD_BUILDER` emit NO broadcast event at all** — `knightsPlayed`
   and `roadBuilderPlayed` are incremented directly in `env/catan_env.py:593-599, 642-647`.
   A broadcast-fed log would silently omit knights, which is literally feature (1).
2. **`BUILD` events carry `location=-1`** as a documented sentinel (`engine/broadcast.py:174-199`,
   emitted `engine/player.py:94,159,211`) because the engine keys vertices by pixel tuples.
   Only the recorder resolves them by diffing `buildGraph`. The action tuple already has the
   index — use it.
3. It needs no engine change and no deepcopy guard (see D5).

### D4 — No blinding in the log, with ONE gate
Every broadcast source is public: DICE, DISCARD, YOP, BUILD_*, STEAL, MONOPOLY, TRADE_BANK,
SETUP, BUY_DEV_CARD. Notably `draw_devCard` (`engine/player.py:340-375`) emits only
`RESOURCE_CHANGE source="BUY_DEV_CARD"` carrying the ORE/WHEAT/SHEEP **cost** — never the card
type drawn.
**The one gate:** `GAME_END` (`engine/broadcast.py:233-252`) carries a `vp_breakdown` that is
VP-card-inclusive. Terminal-only, so harmless in practice, but render it through the existing
`_visible_vp` helper (`play_vs_model.py:686`) or suppress it, and add a leak pin beside
`TestBroadcastBanner` (`test_hand_panel.py:106`).

### D5 — Do NOT reuse `EventCollector`; use a bounded deque
`EventCollector.drain()` is **destructive and single-consumer** (`replay/recorder.py:154`:
"the internal buffer is replaced with an empty one"), and `_HumanGameRecorder` already owns
one, subscribed at `play_vs_model.py:836-837` and drained at every `env.step` to build the
`Replay`. A second consumer would race it: the GUI drains first on some frames, the recorder
gets an empty list, and **the replay silently loses events** — corrupting the one artifact
built to be trustworthy. Reusing it would also import the replay package into the GUI and
inherit no deepcopy guard (that guard lives in `_RecorderSubscriber.__deepcopy__`,
`play_vs_model.py:740-763`, and exists because `clone_env` deepcopies the whole env at
`search/mcts.py:163`). Use a plain `collections.deque(maxlen=N)` owned by the harness.

### D6 — Placement and size: the honest constraint
The window is 1000×800 (`engine/board.py:139`) and the right rail is fully allocated (human
panel y≈15-299, trade button y=400-440, bot panel y≈460-744). The owner asked for ~12 lines;
**that does not fit.** Two real options:
- **Bottom strip** `x∈[115,820]`, `y∈[695,795]` — wide (705px, good for `Bot built settlement
  at v23`), **~5-6 lines**.
- Under the bot panel `x∈[828,996]`, `y∈[615,790]` — ~9 lines but only 168px wide.
**Chosen: the bottom strip**, because log lines are long and a typical bot turn is 3-6 entries
(roll, 1-3 actions, end turn) — so ~6 wide lines covers "what did the bot just do", which is
the actual request. No refactor needed: `displayGameScreen` is a full repaint
(`view.py:451`) and `_draw_hand_panel`'s backdrop pattern (`:433-437`) generalizes verbatim.

### D7 — Log scope: both players (owner-chosen)
Dice rolls, robber steals and card plays read in sequence, and the bot's move often responds
to the human's. This also matches what the replay records, so the on-screen stream and the
post-game review stream agree.

## Non-goals
- No broadcast subscriber in v1 (D3). Revisit only if dice/steal detail is still missing
  after playing a game.
- No resolution of the `location=-1` sentinel — the action tuple already carries the index.
- No scrollback or overlay; fixed recent window only.
- No engine change, no obs/action-space/reward change, no change to the replay schema.
- Does not widen `broadcast_message` beyond its current 3 event types; the log supersedes it.

## RISKS — open challenges from the premise review (NOT cleared)

**Pre-mortem.** The log ships and the replay corpus is quietly worthless, because a second
consumer shares the recorder's destructive `EventCollector` and steals its events on some
frames — losing a scatter of `StepEvent`s that nobody notices, since the replay is what you
consult *after* you stop trusting your memory. D5 exists to make this impossible; it is the
single most important decision in this spec.

**Strongest opposite case.** *You are changing the measurement instrument mid-measurement.*
Every game in `runs/human_playtest/games.jsonl` — including the one that prompted this — was
played with strictly less information, and the harness already carries this scar (a missing
`reveal_bot` key marks a game as not a clean result). This mints a **third incomparable
regime** while the binding constraint is games played, not fidelity per game. The reviewer's
own verdict: **for D1 the case is clearly weaker — ship it**, since withholding public facts
biases every result toward the bot; **for the log, the case against is real**, because the
replay already exists, already resolves build locations, and `scripts/replay_viewer.py`
already ships. The counter-argument this spec rests on: reviewing after the fact does not help
you *play better in the moment*, which is the owner's stated need.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `make test-unit`).
2. **Panel pin:** both players' panels show knights played and longest-road length in BOTH
   reveal modes; `_FakePlayer` gains both attributes; no line matches a dev-card type string.
3. **Freshness pin:** a road broken by an opponent settlement updates the displayed length.
4. **Layout pin:** the revealed human panel does not overlap the BANK TRADE button at y=400.
5. **Log pin:** a bot turn containing a knight play appears in the log — the case a
   broadcast-fed log would have missed entirely.
6. **Location pin:** a logged build names its vertex/edge index (not `-1`).
7. **Recorder-integrity pin:** with the log active, a recorded game's `Replay` contains the
   same events it would without the log — proving no drain contention (D5).
8. **Leak pin:** no bot resource type or hidden dev-card type is reachable in the log, and
   `GAME_END` VP is rendered visible-VP-only.
9. Read-only w.r.t. training: nothing under `runs/train/**`; no engine or schema change.

## Implementation notes (what actually shipped)

Built as specified (D1-D7). Two additive deviations, both flagged rather than
substituted for a decision:

1. **Hand-panel leading 20px -> 18px** (`HAND_PANEL_LINE_HEIGHT`, `gui/view.py`).
   D1's layout analysis checked only the human panel against BANK TRADE and
   missed the BOT panel, which is drawn at y=460: a REVEALED 16-line bot panel
   (`--reveal-bot`) would have run to y=824 in an 800px window and been clipped.
   At 18px the human panel bottoms out at y=343 (57px of clearance under BANK
   TRADE at y=400) and the revealed bot panel at y=788. Pinned as arithmetic in
   `TestPanelLayout`, not eyeballed.
2. **`test_no_dev_card_type_is_reachable`'s allow-list is left INTACT.** D1 said
   to update it deliberately; the chosen wording (`Knights played: N`) clears it
   unchanged, so the leak pin is preserved rather than weakened, and a positive
   assertion (`test_the_knights_line_is_not_a_dev_card_type_leak`) was added
   beside it instead.

Also added, beyond the spec's letter, against the "third incomparable regime"
risk the spec carries UNCLEARED: an additive `"hud": 2` key on the `games.jsonl`
record, alongside the existing `reveal_bot` precedent, so a game played under
this information regime stays legible as such. No reader breaks (`pregate0.py`
reads by key) and the Replay schema is untouched.

Concrete surface:
- `gui/view.py`: `_public_progress_lines`, `hand_panel_lines(..., board=None)`,
  `catanGameView.move_log` + `displayMoveLog()` (drawn BEFORE `displayPorts()`
  and the buildings loop: the strip overlaps the lower hexes, three real vertices
  at y=720 AND two of the nine ports, so drawing it last hid placed pieces and
  port access — a new withholding of a public fact, the exact failure this
  feature exists to fix; pinned by
  `test_the_strip_is_drawn_before_the_buildings_loop`),
  `displayPorts()`, `HAND_PANEL_LINE_HEIGHT`, `MOVE_LOG_*`.
- `scripts/play_vs_model.py`: `_describe_bot_move(..., with_location=False)`
  (the DEFAULT string is byte-frozen — it is `bot_action_label` on disk),
  `_human_snapshot` / `_describe_human_delta`, `_game_over_log_line` (D4's one
  gate, extracted from `main` so `TestGameOverLine` can pin it),
  `HumanVsBotEnv._log_move` /
  `_log_human_action`, and a `deque(maxlen=MOVE_LOG_LINES)` owned by `main` and
  attached through the view factory (so the env holds no reference and an MCTS
  clone, which already drops `_human_view`, structurally cannot log), plus
  `_hold_final_screen()` / `FINAL_SCREEN_HOLD_S`.

### Review fixes (pass 2)

3. **`PLAY_KNIGHT` names no hex.** The HUD variant originally appended
   `to hex{action[3]}` for `PLAY_KNIGHT` as well as `MOVE_ROBBER`. `PLAY_KNIGHT`
   does not consume head 3 — `_apply_main_action` only decrements the card,
   bumps `knightsPlayed` and sets `robber_placement_pending`; the destination
   arrives in the SEPARATE `MOVE_ROBBER` action one step later, and `masks.py`
   leaves `tile_mask` all-False during a main turn, so `action[3]` is a
   uniformly RANDOM index. The log therefore printed two contradictory hexes for
   one knight, one of them invented — asserting a false public fact, which is
   worse than withholding one. Pinned by
   `test_a_knight_play_never_names_a_hex`.
4. **The strip no longer erases two ports.** Ports are drawn inside
   `displayInitialBoard`, which runs BEFORE `displayMoveLog()`; only the
   buildings loop redrew afterwards. The 2:1 WHEAT ship at (541, 759) and a 3:1
   generic at (296, 751) sit inside `MOVE_LOG_RECT`, so the backdrop hid them for
   the whole game — the same regression the buildings redraw already guarded
   against. The port loop is now `displayPorts()`, repainted after the strip in
   `displayGameScreen` exactly like the buildings. Pinned by a pixel test
   (`test_the_ports_are_repainted_after_the_strip`) plus the premise measurement
   `test_the_strip_really_does_overlap_two_ports`.
5. **Harness wiring is pinned** (`TestHarnessWiring`): `_log_move` behaviourally
   (view-only path, no-op without a view or a log, no blank lines) and the two
   harness-only statements at the source, the technique `TestGameOverLine`
   already used. Deleting either previously left an empty strip with the full
   suite green.
6. **The final screen is held** until a click / key / window-close (120s
   liveness cap, `_hold_final_screen`). This feature is the first to put content
   on screen after the last human interaction, so the pre-existing instant
   window teardown made the terminal result line unreadable. Called last, after
   every artifact is on disk.

Out of feature scope, fixed because it blocked the gate: `_FileSemaphore`
(`scripts/vlm_spike.py`) created its slot file EMPTY and wrote the pid
afterwards, so a rival could read pid 0 from the gap, judge a genuinely-held
slot stale and unlink it — letting concurrency exceed `n`
(`test_file_semaphore_caps_concurrency` failed ~1 run in 3). The pid is now
written to a temp file and hard-LINKED into place, which is atomic and carries
the pid from the moment the slot exists. 12/12 clean runs after.
