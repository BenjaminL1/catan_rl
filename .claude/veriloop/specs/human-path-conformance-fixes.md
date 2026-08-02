# Spec: human-path-conformance-fixes — close the GUI-only rule gaps before the corpus grows

**Status: RATIFIED (2026-08-02) — BINDING under owner delegation ("go with recommended fixes").
Ratified in full view of the open RISKS below, INCLUDING a premise review that argued against the
engine-edit route the owner had separately chosen. See D1's note for how that was resolved.**

**Feature in one line.** Fix the HIGH and MEDIUM defects an audit found in the human play path — a
cancelled dev card that permanently lies to the policy, a self-targetable robber, a Monopoly that
leaves no record, and a deck the human cannot see — by restructuring the one function all three
dev-card defects share, rather than patching each symptom.

## Provenance

A five-lens audit of the HUMAN path (`scripts/play_vs_model.py` + `src/catan_rl/gui/view.py`)
against the Colonist 1v1 ruleset produced 7 defects that survived adversarial verification. An
expert council then split on the central question and the split is recorded in D1.

---

## Decisions (binding)

### D1 — Restructure `play_devCard`: resolve every choice, THEN commit once

`engine/player.py:405-518` interleaves *choice resolution* with *state commitment*. The bot's
equivalent (`env/catan_env.py:665-719`) receives the choice pre-resolved and commits in one block.
That single structural difference produces **all three** dev-card defects:

1. **Counter stranding (the HIGH one).** `yopPlayed += 1` (`:450`) and `monopolyPlayed += 1`
   (`:478`) run *before* a cancellable picker. The cancel arms (`:470-474`, `:488-493`) refund the
   card and reset `devCardPlayedThisTurn` but never the counter, and nothing anywhere decrements
   them. `policy/obs_encoder.py:811-814` feeds those counters to the policy as the opponent's
   *publicly played* dev cards, and because the cancel also restores the turn allowance the cycle is
   **unbounded within one turn** — a human can drive the count past the number of that card in the
   deck. `eval/rules_invariants.py:335-341` then audits the same counters against rolled turns, so a
   legitimate game FAILS its own audit.
2. **Knight ordering divergence.** Bot: `knightsPlayed += 1` → `check_largest_army` → *then* robber
   placement, so the legality set is computed with Largest Army's +2 VP already awarded. Human:
   `game.robber(self)` runs the whole pick first (`:436`), the counter second (`:437`), and
   `check_largest_army` only later from the caller. Since `board.get_robber_spots():711-718` excludes
   a hex adjacent to any player under 3 visible VP, **the two seats can be offered different legal
   hexes from identical positions.**
3. **Road Builder forfeit.** The card is spent at `:432`, then two `game.build(..., is_free=True)`
   calls route to `buildRoad_display` with `allow_cancel=not gameSetup` — cancellable in main play,
   and `game.build` silently no-ops on `None`. A stray click forfeits a free road with the card
   already gone and no feedback. The bot cannot lose this (`catan_env.py:562-570` keeps
   `road_building_roads_left` and zeroes it only when no legal road exists).

**Patching two `+= 1` statements fixes one of three.** Restructuring so no engine state mutates until
every choice is resolved makes all three structurally impossible, and is what makes the claim
"mirrors the bot's shape" actually true. Road Builder placement must additionally be
non-cancellable while a legal road exists.

**COUNCIL SPLIT, recorded rather than smoothed.** The premise review argued for a harness-layer fix
(`scripts/play_vs_model.py` snapshots the counters and restores them iff `devCardPlayedThisTurn` is
False — an exact cancel detector) precisely to avoid the D5 re-pin, and called the engine route
"clearly weaker." Its core objection was that D5 re-validates the *guard* rather than the *vendored
encoder*. That objection is answered by the fact in D5, not waved away — but note the premise review
would still not sign this, and its reasoning is preserved in RISKS.

### D2 — Emit `broadcast.monopoly` from BOTH missing emitters

The human Monopoly (`engine/player.py:477-518`) transfers cards correctly but emits only
RESOURCE_CHANGE, never `game.broadcast.monopoly(...)`, which the bot emits unconditionally at
`env/catan_env.py:713`. **`agents/heuristic.py:394-408` has the identical omission** — so the
heuristic seat is invisible to `rules_invariants.py:325-327`'s per-turn attribution in *every*
self-play and eval game, not just human ones. Fix both. Verified safe: `env/hand_tracker.py:111-114`
returns early for any non-`RESOURCE_CHANGE` event, so the transfer cannot be double-counted.

### D3 — Victim picker: filter self, auto-select the forced case, fail open

`gui/view.py:691-701` receives `currentPlayer` and never uses it; `board.get_players_to_rob` is
keyed on the hex alone, so the human's own building is offered as a steal target.

- **Filter self**, matching `env/catan_env.py:876-882` and `conformance/recorder.py:588-589`.
- **Auto-select when exactly one victim survives.** In 1v1 there is never a real choice after
  filtering, and the bot never prompts. This matches the bot exactly *and* removes the need for a
  cancel path in that picker.
- **Fail open on an empty spot set**, mirroring `env/masks.py:220-221` (all 19 tiles). Today an
  empty set reaches `_animated_pick(..., allow_cancel=False)` with no matching rect and no
  `pygame.QUIT` arm — a hard window-lock. Believed unreachable in 1v1, but the bot has the fallback
  and this picker is being opened anyway.

**Direction matters and goes in the record:** `steal_resource(self)` is a net-zero self-transfer, so
a human who mis-clicked their own settlement **forfeited the steal**. The bug was biased AGAINST the
human; fixing it moves the win rate in the human's favour, and prior games were played under a
*harder* rule, not merely a different one.

### D4 — Deck visibility, and grey the button on the FULL predicate

The bot observes public-reveal-derived dev-deck remaining (`policy/obs_encoder.py:390-399`); the
human sees nothing. Display it.

`gui/view.py:445` draws BUY DEV CARD gold unconditionally, while the bot's mask is gated on **both**
`deck_total > 0` **and** affordability (`env/masks.py:289-297`). `draw_devCard` silently no-ops on
both conditions (`engine/player.py:343-347`, `:353-356`) — it returns before spending, so this is UX
not correctness. Grey on `not (deck_total > 0 and can_afford)`: one predicate, both halves.

### D5 — Re-pin `PINNED_ENGINE_TREE`, with the REAL argument and the correct sequencing

**The re-validation argument** (this is what must go in the log entry — "I ran the guard and it
passed" is not one): the guard's contract (`eval/engine_parity.py:5-8`) is that the engine is
byte-identical to the tree the vendored v11 encoder was written against, and
`eval/legacy_arch/obs_encoder.py:703-706` reads exactly the counters D1 moves. But **`play_devCard`
has only two call sites — `engine/game.py:624` (the legacy pygame loop the RL stack never enters)
and `scripts/play_vs_model.py:722`** — and `agents/heuristic.py:225-232` deliberately routes around
it. `cross_arch_h2h` is bot-vs-bot through `_apply_main_action`, which D1 does not touch. So D1/D2
change the byte pin **without changing any behaviour a cross-arch measurement can observe.** That is
a re-validation of the vendored arch, not of the guard.

**Sequencing is load-bearing — the naive order guarantees a red gate.** `assert_engine_parity` does
two checks: the constant vs `git rev-parse HEAD:src/catan_rl/engine`, **and**
`git diff --quiet HEAD -- src/catan_rl/engine` for *uncommitted* changes (`:91-96`). The sha is read
off HEAD, so the new pin cannot be computed from a dirty worktree. Order: (1) commit the
`engine/player.py` edits alone; (2) read `git rev-parse HEAD:src/catan_rl/engine`; (3) write that
value into `engine_parity.py` in a second commit — that file is not under `src/catan_rl/engine/`, so
it cannot perturb its own input. The gate is authoritative only at (3).

**Do NOT append to a history that does not exist.** `.claude/veriloop/specs/bank-fix-slice-and-champion-bank.md`
(status DRAFT) records a re-pin `261098d190c8 → 3388b69026cb` and claims it landed 2026-07-28.
Verified: `3388b69026cb` appears nowhere in the tree and the live constants are still
`261098d190c8` / `70813dcf76fd`. The log entry must start from the real current value.

### D6 — The interactive audit RECORDS; it must never assert

`run_all_invariants` currently runs only in `--self-test`, consumed as `assert not violations` —
which is exactly why a GUI-only bug class survived. Run it interactively, but **persist rather than
raise**: the driver's whole crash-safe contract is that artifacts are written even on abort, and an
assert would destroy the record it exists to validate. Mirror the `bank_ok` precedent
(`scripts/play_vs_model.py:1864-1871`): write `"rules_ok": not violations` plus the violation list,
same missing-key convention. Cheaper than it looks — `audit_events=True` is already set on this env
(`:342-344`), so the event stream is already retained.

**Pass the abort state through.** `check_terminal_state` on a window-closed game with
`truncated=False` reports "terminated but no player reached 15 VP" — and the most recent recorded
game is exactly that shape (`runs/human_playtest/games.jsonl`, seed 512940688, `winner: null`,
partial). Wired naively, D6 flags every interrupted game and the operator learns within two sessions
that the alarm means "you closed the window."

### D7 — Provenance for a changed rule set

Bump the `hud` information-regime stamp 2 → 3 (D4 is the only decision that changes what the human
can SEE). Nothing reads `hud` today — `scripts/play_vs_model.py:1859` is the sole writer — so it is a
pure provenance marker and this is its purpose.

`hud` alone is insufficient: **D1 and D3 change the human's legal option set**, which is a rules
change, not an information regime. Add a per-game rules-epoch key so a later reader can screen games
played under the old (self-robbable, counter-leaking) rules. The `bank_ok` / `rules_ok` per-game
integrity keys are the established pattern here, not a regime counter.

---

## Non-goals

- **No retraining, no BC regeneration, no long jobs.**
- **No change to `env/masks.py` or the bot's action path** — the bot side is already correct on every
  one of these; that is what made the asymmetries visible.
- **The pre-roll window itself** is a separate in-flight slice (`preroll-dev-cards-r1` applied to the
  harness); this spec must not conflict with it.
- **The synthesized-setup replay caveat** (`scripts/play_vs_model.py:66-70`) is NOT addressed. The
  opening is the least faithful part of every recorded game and it is the phase the owner suspects.
  Flagged, deliberately out of scope, and see RISKS.

## Acceptance criteria (the `/dev-loop` gate is the authority)

1. Full gate green on real exit codes, evaluated after D5 step (3).
2. Cancelling a YoP or Monopoly N times leaves `yopPlayed` / `monopolyPlayed` unchanged, and the
   policy's observed opponent dev-counts unchanged. Must FAIL against the pre-change tree.
3. A Knight that wins Largest Army offers the human the same legal robber-hex set the bot's mask
   would offer from the identical position.
4. A Road Builder play cannot forfeit a free road while a legal road exists.
5. The human cannot select themselves as a steal victim; a single surviving victim is auto-selected;
   an empty spot set falls open rather than locking the window.
6. Both the human AND heuristic Monopoly paths emit a `MONOPOLY` broadcast event.
7. BUY DEV CARD is greyed when the deck is empty OR the player cannot afford it; deck-remaining is
   visible to the human.
8. An interrupted game still writes both artifacts, now carrying `rules_ok` — and a window-close does
   NOT produce a spurious violation.
9. `PINNED_ENGINE_TREE` re-pinned with a log entry carrying the call-site-unreachability argument,
   starting from the real current value.

---

## RISKS — open, NOT cleared

**The premise review would not sign this, and its case is preserved verbatim in substance.** It
argued the engine edit is unjustified: D1/D2/D3 are unreachable from training, eval, BC, ExIt, search
and conformance — they need a human at a pygame window — and the artifact they protect is **four
recorded games**. An exact harness-layer fix exists at comparable cost. Its conclusion: "On D1 + D2 +
D5 as a bundle, the counter-case is stronger, and I do not think it is close." D5's cost is paid in a
different subsystem than the bug. The answer offered here is D5's unreachability argument; the
premise review's rejoinder is that establishing *the routine response to this tripwire firing is to
move the pin* is the damage, and the diff being harmless is what makes it a trap.

**Pre-mortem.** A year on, the pointer-arch fork is stranded — neither accepted nor refused. Clause
(a) passed long ago stamped `engine=261098d190c8`; clause (b) returns a number just under v11's and
nobody can defend it, because the tree it was measured against no longer exists and the re-pin taught
the repo that a tripwire firing is a bookkeeping chore. **Clause (b)'s tool does not exist on `main`
today** (`src/catan_rl/human_data/opening_scoreboard.py` is absent) — so the pin is being re-pointed
*before its one outstanding consumer has ever run.*

**Corpus fragmentation.** `runs/human_playtest/games.jsonl` holds four games across two information
regimes (two with no `hud` key, two at `hud: 2`). D7 makes a third, with zero games in it. A
four-game corpus partitioned three ways compounds nothing.

**Opportunity cost.** Two ratified decisions are blocked right now: what replaces the failed R1
lineage (`runs/analysis/d9_r1_vs_r0.json`: WR 0.5067, CI [0.4667, 0.5465], FAIL) and whether the
pointer-arch fork is accepted. Sequencing GUI conformance ahead of both is a choice, not an
inheritance from the order the audit happened to run in.
