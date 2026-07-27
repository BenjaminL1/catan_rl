# Spec: play-vs-model-recorder — an honest, full-fidelity human-vs-policy game record

**Status: OWNER-DELEGATED DRAFT (2026-07-27) — written under explicit instruction
("I am going to sleep after this prompt … don't pause for my input since i will be
asleep"). NOT owner-ratified. The challenges in §RISKS are OPEN, not cleared.**

**Feature in one line.** Rename `scripts/play_vs_v8.py` to `scripts/play_vs_model.py`,
**close the information leak that makes every game played so far uninterpretable**, and
record each game as a real `Replay` carrying both players' full move stream plus the
policy's per-decision internals — so the owner can play, then review the game and see what
the policy was actually weighing at the moments that mattered.

## PHASE ORDER IS BINDING

**Phase 1 (D1 + D5) must land and be verified by playing a game BEFORE Phase 2 begins.**
Every game recorded while the leak is open is worthless no matter how faithful the record —
so fidelity work before the leak fix is wasted. This ordering is the single most important
decision in this spec.

## Decisions (binding for the build)

### D5 — BLOCKING: close the bot-hand leak (rescoped to the GUI, not the console)
`src/catan_rl/gui/view.py:317-325` renders a panel titled **`"{bot} — FULL HAND"`** via
`_draw_hand_panel` (`view.py:327-360`), showing every resource count, `Victory Points:
{player.victoryPoints}`, and `player.devCards + player.newDevCards` — **every frame**. The
comment at `scripts/play_vs_v8.py:132-133` claiming it shows "hand SIZE (counts only)" is
**false**. The console prints at `:608-612` are the *small* half of the same leak.

- **Blind by default.** The bot panel shows hand SIZE and **visible** VP only.
- Visible VP is `victoryPoints - devCards["VP"]`. No `newDevCards` term — VP cards bypass
  that bucket (`engine/player.py:379-383`); this matches `policy/obs_encoder.py:569`. Do
  **not** use `visibleVictoryPoints`, which is stale (refreshed only at init + VP-card buy).
- **`--reveal-bot` opt-in** restores the analysis view, and the flag is **recorded in the
  replay metadata**, so a revealed game can never later be misread as a strength result.
- Fix the false comment at `:132-133` and the false `"(default: champion)"` help string at
  `:787`.
- The one game in `runs/human_playtest/games.jsonl` was played with the leak open. Mark it
  so in the record; do not silently keep citing it as a clean result.

### D1 — Rename to `scripts/play_vs_model.py`
"v8" is stale: `DEFAULT_CKPT` is already the pointer-arch `ckpt_000000500`. Zero code
references; only the script's own docstring and prose in
`specs/009-finite-resource-bank/review-resolution.md:20`. **No deprecation shim.**
- `docs/plans/v2/design.md:721` already RESERVES `scripts/play_vs_model.py — Human play
  (GUI)`, so the rename *satisfies* the locked design. **Do not edit `design.md`** — an
  implementer editing a ratified doc to match what it built is a pattern already flagged
  this week.
- `CLAUDE.md:76` asserts "(No v1 `evaluate.py`/`play_vs_model.py`.)" — that line refers to
  removed **v1** artifacts and becomes misleading. Update it to name the v2 script.

### D2 — Emit a real `Replay`; do not extend the bespoke JSONL
Use `src/catan_rl/replay/`: `BoardStatic`/`HexStatic`/`VertexStatic`/`EdgeStatic`/
`PortStatic`, `StepStateSnapshot`, `PlayerStateSnapshot`, `recorder.snapshot_step_state`,
`io.save_replay`. This inherits the existing viewer at `replay/viewer/`.
- **Reuse is smaller than it looks — do not pretend otherwise.** `recorder_loop.record_game`
  (`:790`) builds its **own** `CatanEnv` and drives a fixed 4-step setup loop against a
  different actor surface; it **cannot** host an interactive human game. Only the main-loop
  primitives are reusable. Honest estimate: **~120–150 new lines** in the script plus
  promoting `_consume_main_event_block`, `_split_at_setup_complete` and
  `_setup_steps_seat_0/1` to the public surface (they are module-private and absent from
  `replay/__init__.py`'s export map).
- **Widen `PlayerKind`** (`replay/schema.py:93`, mirrored `replay/player_factory.py:43`) to
  include `"human"`. `player_factory.build_actor` and `recorder_loop._resolve_seat_and_opp`
  must **reject** `"human"` explicitly rather than attempt to build an actor. Labelling the
  human as `"heuristic"` would poison every downstream consumer and is forbidden.
- **Torch must not reach the schema path.** `replay/__init__.py:10-11` states the viewer
  imports only pygame + this package (no torch, no env, no engine). Convert every tensor to
  plain Python floats at the boundary.

### D3 — Capture the human stream via `EventCollector`, and FIX the attribution bug
The human's turn runs inside `env.step` via the `_opponent_*` hooks, so human actions
already emit broadcast events (`engine/broadcast.py:47-68`).
**Known bug that must be fixed, not inherited:** `_partition_main_events_by_actor` splits
turns on `DICE_ROLL`, but the human's **pre-roll knight window**
(`scripts/play_vs_v8.py:317`, `_human_pre_roll`) fires `MOVE_ROBBER` / `STEAL` /
`LARGEST_ARMY_CHANGE` *before* the human's roll — so those events land in the previous
group and are **credited to the bot**. A review of "what the bot did" would show moves the
human made. Attribution must be by acting seat, not by dice-roll boundary, and a test must
pin the pre-roll-knight case.

### D4 — Policy internals as a TYPED, OPTIONAL field (bump + no-op migration)
Per agent decision record: the `type` mask (13 bools), the masked-softmax distribution over
the 13 action types, the chosen 6-tuple, the value-head scalar, and `belief_logits`.
- **Mechanism.** Add `policy_internals: ... = ()` (a default) to `ReplayStep`
  (`schema.py:449`, which is `frozen=True, slots=True` — so runtime attachment and
  monkey-patched sidecars are both impossible), and thread it through `_step_to_dict`
  (`io.py:82-92`) and `_step` (`io.py:290`, via `d.get`). A loose JSON key is **not**
  acceptable: `load_replay` documents that unknown nested keys are **silently dropped**
  (`io.py:316-322`), so a load→save cycle would delete exactly the data being added.
- **Version.** `schema.py:48-53` mandates a bump whenever a key changes shape, and adding a
  field is a shape change — so bump `REPLAY_SCHEMA_VERSION` 1→2 **and ship a no-op v1→v2
  migration in the same commit**. Bumping without one makes `apply_migrations`
  (`migrations.py:83-87`) raise on all existing replays in `runs/replays/`. The reviewers
  split on whether to bump at all; doing both satisfies both positions and breaks nothing.
- **Cost.** `MultiActionHeads.sample` already computes all six masked log-softmaxes in one
  pass (`heads.py:370-398`) and then discards them, and `network.sample` already returns
  `trunk`, `_node_v/_node_e/_node_h`, `_is_setup`, `value`, `belief_logits` in one dict
  (`network.py:211-223, 237-245`) — so recompute the heads **script-side** from that dict.
  **Zero change to the PPO hot path.**
- **Top-8 per dense head, not dense.** Dense `corner`(54)/`edge`(72)/`tile`(19) is ~168
  floats/decision ≈ **+250 KB/game, roughly doubling the file** (base is 291 KB / 261 steps
  ≈ 1.1 KB/step). Top-8 gives ~15 KB/game and loses nothing for human review. The 13-way
  `type` distribution stays dense.

### D6 — Seat alternation: CUT
`human_seat` is already a CLI argument, and seat confounds matter at n≥100 while this
harness has n=1. This is not the strength instrument — `eval/harness.py` is. If the owner
does alternate seats later, **the seed must vary too**; alternating seats on a fixed
`seed=0` keeps every game on the same board and yields degenerate evidence.

### D7 — Raw policy stays default; label search loudly
Search is clairvoyant against a human by construction (`search/mcts.py:247-259` reseeds the
opponent model, but the dice bag is deep-copied intact). Already the default; the actionable
part is that `mode` / `sims` / `clairvoyant` must appear in the replay metadata.

### D8 — Opportunity-vs-selection helper: CUT
Because D4 records the per-step `type` mask, "times legal vs times chosen" is an offline
query over any record — a notebook, not a capture requirement. The 92.7% city figure
measured this session came from **bot-vs-heuristic** games and has no bearing on "play, then
review", so it is not pinned as a fixture here.

## Non-goals
- No migration or rewrite of `runs/human_playtest/games.jsonl`. It is not convertible (no
  board statics, no per-step state, no human actions). **Dual-write** the old four-field
  format for the next several games; rewriting the single most-cited artifact in the project
  is worse than carrying two formats. If display switches to visible VP, **add new keys** —
  never re-point `bot_vp`/`human_vp`, which are TOTAL VP.
- No change to the engine, obs schema, action space, reward, or any training config.
- No change to the PPO rollout path.
- No claim that this harness measures agent strength.

## RISKS — open challenges from the premise review (NOT cleared)

**Pre-mortem.** July 2027: fifteen recorded games, a year of training steered by them, every
conclusion contaminated — because the leak was misidentified as a console print and the GUI
panel kept rendering the bot's full hand at 60fps. Compounding it: `record_game` could not
host the human game, the implementer forked the pipeline, and the fork inherited two
properties nobody re-read — **setup steps are SYNTHESIZED, not observed**
(`_setup_steps_seat_0/1` reconstruct placements from action tuples and hardcode
`longest_road_holder=None`), and `state_after` is shared across every sub-step within one
`env.step`. **The opening — the exact phase the owner suspects — is the least faithful part
of the record.** This must be stated in the artifact, not discovered later.

**Strongest opposite case, which the reviewer judged STRONGER on ordering.** The binding
constraint is **games played, not bytes per game**: `games.jsonl` holds exactly one line,
written 2026-07-24. Eight deliverables of capture infrastructure are being built on a habit
with n=1, to preserve information that only becomes evidence at n≈20. The opposite build is
one line and an evening — fix `view.py`, play twenty games, append to the four-field JSONL
that already exists; the board is reproducible from `seed` + `human_seat`. This spec answers
that only partially, by making D5 blocking and Phase 2 conditional on it.

**Expectation to set now.** For a raw-policy agent, "what was it thinking" is a 13-way
softmax over action types, and at most steps it reads roughly `0.97 END_TURN`. It will
rarely explain an opening. The internals are most informative at *contested* decisions
(build-vs-trade, robber placement), and the review workflow should be aimed there.

**Introspection is the agent grading its own homework** — the failure mode already recorded
in `project_surpass_thephantom`, where a diagnostic could not confirm a blind spot for this
reason. Per-decision softmax is a hypothesis generator, never a verdict.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `make test-unit`).
2. **Leak pin (blocking, Phase 1):** with `--reveal-bot` absent, no bot resource type, hidden
   dev card, or hidden-VP-inclusive total is reachable in the rendered panel or console; a
   test asserts the bot panel exposes size + visible VP only. With `--reveal-bot`, the flag
   appears in the replay metadata.
3. **Visible-VP pin:** the displayed value equals `victoryPoints - devCards["VP"]`, and a
   test covers a bot holding at least one VP card.
4. **Rename pin:** no reference to `play_vs_v8` remains in code, docs, or `CLAUDE.md`;
   `docs/plans/v2/design.md` is UNMODIFIED.
5. **Attribution pin:** a pre-roll knight played by the human is attributed to the human, not
   the bot (the `_partition_main_events_by_actor` bug).
6. **Round-trip pin:** a recorded game saves and loads with `policy_internals` intact through
   `save_replay` → `load_replay`; a v1 replay still loads under the bumped version via the
   no-op migration; the existing viewer opens a new replay.
7. **`PlayerKind` pin:** `"human"` is a legal replay value, and `build_actor` rejects it with
   a typed error rather than constructing an actor.
8. **No-torch pin:** `replay/schema.py` and `replay/io.py` import no torch; internals are
   plain floats.
9. **Size pin:** a full recorded game stays within ~2× the base replay size (top-8 heads).
10. `runs/human_playtest/games.jsonl` is byte-identical for existing lines; new games append
    in both formats.
