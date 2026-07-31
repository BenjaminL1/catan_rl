# Spec: preroll-dev-cards-r1 — give both seats the Colonist pre-roll dev-card window

**Status: RATIFIED (2026-07-30) — BINDING. Owner ratified in full view of the open RISKS below
(pre-mortem, cost re-pricing to 4-8 M1-days, and the "do D6 first / merge the finished branches
instead" counter-case), and elected to proceed with the full slice as written.**

**Feature in one line.** Close a real ruleset gap — under Colonist 1v1 a player may play one dev
card *before* rolling, and the RL env has never allowed it — by opening the pre-roll window for
both seats inside `env/` only, fixing the two turn-boundary bugs that opening it would activate,
and retraining under the corrected ruleset with a measurable acceptance criterion.

## The gap

`src/catan_rl/env/masks.py:142-145` sets only `ActionType.ROLL_DICE` when `roll_pending` and
returns; `src/catan_rl/env/catan_env.py:482-487` silently no-ops anything else. Both seats lack
the window **symmetrically**, which is why self-play never surfaced it. The pre-roll Knight
(move the robber off your own hex *before* production resolves) is a genuine, frequently-relevant
line. The only asymmetry today is `scripts/play_vs_model.py:376`, which gives the **human** a
pre-roll window the bot cannot have — which is why human playtest results are currently
unattributable between "the policy is weak" and "the bot was not allowed to play its Knight."

**Ruleset epochs.** R0 = the shipped no-pre-roll rules. R1 = this spec.

---

## Decisions (binding once ratified)

### D1 — Scope is `env/` ONLY. Do NOT touch `src/catan_rl/engine/`
The `# TO-DO: Add option of AI Player playing a dev card prior to dice roll` at
`engine/game.py:522` lives inside `playCatan` (`:497`), the legacy pygame loop whose only caller
is `game.py`'s `__main__` block. The RL stack never enters it (`catan_env.py:344` builds
`render_mode=None`). Touching it changes the git tree of `src/catan_rl/engine`, which
`eval/engine_parity.py:34` pins and `tests/unit/eval/test_cross_arch.py:297` asserts
unconditionally — so a comment-only edit to dead code turns `make test-unit` RED and forces a
re-pin that retires `261098d190c8`, the tree the accept-gate clause-(a) result was measured on.
**Leave it. The TODO remains an accurate description of `playCatan`.**

### D2 — Hoist the turn-boundary bookkeeping out of the roll handler (load-bearing)
`catan_env.py:756-757` runs `agent.updateDevCards()` and `agent.devCardPlayedThisTurn = False`
**inside** `_do_roll_for_agent`. Two bugs follow directly from opening a window before it:
1. **Two dev cards per turn.** Pre-roll play sets the flag → the roll clears it → `masks.py:204`
   re-offers the card. This is reward-positive (two knights/turn reaches Largest Army in two
   turns), so PPO will find and exploit it within single-digit updates and every R1 number would
   measure a cheat.
2. **Newly-bought cards unplayable pre-roll.** `updateDevCards()` is what moves
   `newDevCards → devCards` (`engine/player.py:397-402`), so a card bought turn N is invisible at
   turn N+1's pre-roll node — a third ruleset, neither R0 nor Colonist.

Move both statements to the true turn boundary — every site that sets `roll_pending = True` for
the agent (`catan_env.py:464`, `:514`, `:725`, `:728`). The opponent seat is **already correct**
(`:794-795` precede `rollDice()` at `:797`); insert its window between `:795` and `:797` and
leave that ordering alone. `scripts/play_vs_model.py:373-377` is a working reference
implementation, including the comment naming this exact bug. `bc/dataset.py:452-453` has the
identical ordering and must move with it.

### D3 — Share the mask helper; never hand-roll the pre-roll branch
`masked_log_softmax` returns **uniform** for an all-zero mask, silently (`policy/heads.py:55-58`:
*"Rows with no valid entries return a uniform log-prob"*). `res1_def`/`res2_def` are
zero-initialised (`masks.py:88-91`) and set **only** inside the main-turn block (`:209-210`,
`:214`). So a hand-rolled pre-roll branch that sets the four type bits but forgets the resource
defaults yields a pre-roll YoP granting two uniformly-random resources and a pre-roll Monopoly
monopolising a random one — valid indices, no crash, green suite, silently random play.

Extract `masks.py:204-217` into a module-private helper that sets the type bits **and** the
resource defaults, and call it from both the pre-roll branch and the main-turn block.
`ROLL_DICE` must be set first so the `if not type_mask.any()` `END_TURN` fallback (`:231-232`)
can never fire at a pre-roll node — pin that as a test. `catan_env.py`'s roll branch must
dispatch on an explicit **whitelist** of pre-roll dev types, keeping the existing no-op fallback
for everything else.

### D4 — EXCLUDE `PLAY_ROAD_BUILDER` from the pre-roll set
Road Builder writes `road_building_roads_left = 2` while `roll_pending` is still true, and the
road-builder block sits **below** the roll block in both files (`catan_env.py:482` vs `:490`;
`masks.py:143` vs `:148`), so the two free roads defer across the roll, a possible 7, discard and
robber. The obvious fix is a trap: hoisting that block above the roll gate makes `masks.py:155-156`
reachable pre-roll, which sets `type_mask[END_TURN]` — **the agent would end its turn without
ever rolling.** Pre-roll Road Builder has ~zero tactical value (roads cannot interact with a dice
roll), so excluding it deletes the whole failure class at no cost. Knight, Monopoly and YoP are
included; Monopoly pre-roll legitimately hedges a 7, YoP pre-roll matters in a bank-depletion race.

### D5 — MEASURE the widened distribution before building any exploration machinery
The premise that a warm-started policy needs forcing to discover pre-roll is **false**. With one
legal type, `masked_fill` + `log_softmax` gives `log_prob = 0` and gradient `1 − softmax = 0`
*exactly*, so the type-head logits at `roll_pending` have received **zero gradient for the entire
lineage**. Widening the mask exposes whatever the trunk generalises from main-phase states —
where `PLAY_KNIGHT` is often the argmax. The realistic failure is the policy **over**-playing
pre-roll and burning knights at update 0.

**Therefore: warm-start, widen the mask, freeze the policy, and log the empirical pre-roll type
histogram over ~1k rollout steps — BEFORE deciding on any exploration mechanism.** If pre-roll
play is already a few percent, this collapses to "warm-start + ordinary PPO" and ships zero new
machinery. Only if the measured rate is ~0 does an exploration lever get built.

### D6 — Opponent-side pre-roll, per the repo's own prior verdict
Give the heuristic (`agents/heuristic.py:211-227`, currently commented out) and the snapshot
opponent driver a pre-roll Knight rule, so the learner **meets** the tactic in its opponent
distribution. `docs/plans/opening_deficit_verdict.md:54-55` bans the learner-side alternative by
name — *"raising `setup_entropy_coef` — precedent: 0.02 -> entropy 0.21->2.3, 0 promotions, 17h
lost. Diversify the OPPONENT's setup instead (same distributional effect, learner untouched)."*
Pre-roll is a worse case than setup for that mechanism: ~99 roll nodes/game vs ~4 setup
decisions, and every exploratory action burns a real dev card. Owner ratified "Both"; the
learner-side half is **conditional on D5's measurement** showing a ~0 pre-roll rate.

If an exploration knob is built: no config knob exists (`ppo/arguments.py:252-261`
`setup_entropy_coef` is hard-wired through a 4-file path), and it must apply to the **type head's
entropy only** — `joint_entropy` folds in a constant `log 19 = 2.944` nats of ungradientable tile
entropy on knight rows (`heads.py:244,496-498`), i.e. a ~4× bonus for *choosing* the card rather
than for uncertainty.

### D7 — BC regeneration is mandatory, and must not precede D6
The existing shards contain **zero** pre-roll decisions by construction. `bc/dataset.py:234` marks
a row forced when the type mask has ≤1 bit and forced rows are dropped
(`manifest.json: include_forced: false`); measured, `shard_0000.npz`'s `phase` column is
**928,039 / 928,039 = 100% `"main"`**. Under R1 those rows stop being forced and enter at full CE
weight — labelled `ROLL_DICE`, because the heuristic teacher never plays a dev card. **Regenerating
before D6 would pay to teach "always roll."** Order: D6 → BC regen → BC train → bootstrap →
self-play. Add a ruleset stamp to the manifest (`dataset.py:764-782` carries only `git_sha`).

### D8 — Make the ruleset a per-env constructor arg, and REFUSE cross-ruleset h2h
`eval/cross_arch.py:179` calls the **live** `compute_action_masks`, so an R0 checkpoint would be
silently evaluated under R1 rules, sampling from a region of its own type head that received zero
gradient. And `eval/harness.py:305-306` `_run_matchup_games` takes a **single shared env**, so
"R0 policy under R0 rules vs R1 policy under R1 rules" is **not expressible today**. Thread a
`ruleset` constructor arg through `CatanEnv` and `EvalHarness`, and make a cross-ruleset h2h
**refuse** rather than annotate.

### D9 — Replace accept-gate clause (a); keep clause (b)
Under R1, clause (a) (h2h vs `v11_cand`, Wilson-LB > 0.50, n=600) becomes near-trivially
winnable: `v11_cand`'s type head is untrained at `roll_pending`, so the R1 challenger farms
opponent out-of-distribution-ness rather than strength. **Replace it** with the per-seat flag
experiment D8 enables: R1 policy (pre-roll ON) vs `ptr_v1_u500` (pre-roll OFF for its seat),
n=600 symmetric-seat, Wilson LB > 0.50 — each policy playing the ruleset it was trained for. If
R1 cannot beat an R0 policy *denied* the option, the retrain bought nothing and that is learned
for one eval instead of one lineage. **Clause (b) survives unchanged** — the human-scoreboard
opening metric covers the setup snake, before any dev card exists, so it is R0/R1-invariant.

### D10 — Land Torevan TS and the conformance recorder in the SAME slice
`Torevan/packages/engine/src/actions.ts:1112` throws `'Must roll the dice before building'` when
`rollPending`, and `legal-moves.ts:11-12` states it *"mirrors the reference `compute_action_masks`
(`catan_rl_v2` `env/masks.py`)"* — TS faithfully copied a Python limitation. Torevan's `CLAUDE.md`
designates this repo the *"differential-test reference oracle — never let the two engines drift."*
**And the conformance harness cannot detect this change class**: `conformance/recorder.py:445`
rolls unconditionally at every turn top, so all four fixtures are 100% `RollDice` openers and
re-recording under R1 yields byte-identical logs — green while the engines fork. The recorder
needs a pre-roll `_play_actions` call, `CONFORMANCE_SCHEMA_VERSION` (`:84`) needs a bump, and the
TS change lands with it. Spec 009 is the template: it changed both engines in one slice.

### D11 — Add the invariant that would have caught D2
`eval/rules_invariants.py:181-215` has no per-turn dev-card check. Add "at most one non-VP dev
card per turn." `run_all_invariants` is already wired into `eval/harness.py:429-430` and
`search/eval_search.py:86-87`, so it earns coverage across every eval and every search game for
free. **Highest value-per-line item in the slice.**

---

## Non-goals
- **No `src/catan_rl/engine/` change** (D1) — and therefore no `engine_parity.py` re-pin.
- **No pre-roll Road Builder** (D4).
- **No exploration machinery** unless D5's measurement justifies it.
- **No obs-schema, action-space, or checkpoint-shape change.** `roll_pending` is already an obs
  phase flag (`obs_encoder.py:639`); types 6-9 already exist; mask keys and widths are unchanged.
- No fix for mid-turn win-detection latency (`catan_env.py:479`, `:487` return hardcoded
  `False, False`) — pre-existing in kind, flagged not fixed.
- No replay-layer change — `replay/recorder_loop.py:685-720` already partitions by acting seat and
  explicitly names the pre-roll Knight case.

## Acceptance criteria (the `/dev-loop` gate is the authority on checks)
1. Full gate green on real exit codes; `test_engine_parity_holds_at_head` still passes with the
   pin **unchanged** at `261098d190c8` (proves D1's scope discipline).
2. **One-dev-card pin:** a pre-roll play followed by a roll leaves no second dev card legal.
   Must FAIL against the un-hoisted `_do_roll_for_agent`.
3. **Fresh-card pin:** a card bought turn N is playable at turn N+1's pre-roll node.
4. **Resource-mask pin:** pre-roll YoP/Monopoly expose the same `resource1_default` /
   `resource2_default` masks as their main-phase counterparts (guards the silent-uniform bug).
5. **No-`END_TURN` pin:** `END_TURN` is never legal at a pre-roll node; `BUILD_*` types are absent.
6. **Mask pins, split not weakened:** `roll_pending` + empty dev hand ⇒ exactly `[ROLL_DICE]`
   (the existing `tests/unit/env/test_masks.py:85-97` assertion, still true); `roll_pending` +
   a knight ⇒ exactly `{ROLL_DICE, PLAY_KNIGHT}`.
7. **Road Builder excluded** at the pre-roll node.
8. **Seat symmetry:** agent and opponent seats expose identical pre-roll legality.
9. **D5 measurement recorded** — the frozen-policy pre-roll histogram, in a tracked file, before
   any exploration knob is added.
10. **Cross-ruleset h2h refuses** (D8), and the D9 flag experiment is runnable.
11. Conformance schema bumped and TS parity landed (D10); docs sync per CLAUDE.md §6, including
    the R0/R1 epoch tag across the citation surface.

---

## RISKS — open challenges, NOT cleared

**Pre-mortem.** A year on, the agent plays the correct ruleset and **nobody can say whether it is
stronger or weaker than `ckpt_500` was.** The wreck: D9's replacement gate was deferred or the
harness work (D8) was descoped as "plumbing," so R1 shipped with no readable baseline — the
heuristic eval was already saturated at 0.97-0.99 and now faces an opponent that cannot play a
dev card at all; anchor/league win-rate is a mirror whose R0-trained snapshots hand the learner a
systematically exploitable opponent it farms for inflated win-rate; and clause (b)'s tool is still
unmerged on `feat/opening-scoreboard`. The plateau diagnosis could not be re-run because the
plateau had been measured in a currency that no longer existed. Meanwhile the three human games
that motivated all of this were already void. **Epitaph: the plan spent the repo's only readable
baseline to buy a rule, and then declined to measure the rule.**

**Cost is mispriced.** "~1 day" is the bootstrap stage alone. Measured wall-clocks: bootstrap
**14h45m**; champion self-play **~41h**; one failed self-play arm **~15h with 0 promotions**.
With mandatory BC regen (D7, ~3.9 GB / 12 shards) a realistic budget is **4-8 M1-days including
one failed arm**.

**Search gets silently weaker.** `search/mcts.py:337` short-circuits forced nodes, so today every
roll node costs **zero** simulations. Under R1 each becomes a full search — roughly one extra
`sims_per_move` per turn. `pointer-arch-fork.md:110-112` gates on **sims/s**, not
sims-per-decision, so a real deepening regression would pass that gate. The banked +54.6 Elo
search uplift also becomes an R0 number.

**Argue the other side — and it is not clearly weaker.** *Do not do this now.* The largest
fidelity hole is not pre-roll: the heuristic plays **no dev cards at all**, and it is 100% of the
bootstrap opponent, ~10% of the league mix, **and the eval baseline whose 0.97-0.99 saturation is
what made this lineage's progress unreadable**. That is ~20 lines, costs no retrain, and
*un-saturates* a baseline instead of destroying one. Meanwhile three finished branches sit
unmerged — `feat/opening-scoreboard` (+1695, the clause-(b) tool), `feat/human-opening-reference`
(+6345), `feat/human-devcard-picker-and-bank` (+939/−101, the live bank-correctness fix) — and the
repo's own most recent ranked ruling puts *aggregate midgame instrumentation over existing
corpora, zero new games, no baseline problem* above any further metric work. The fork is one merge
plus one metric run from formal acceptance. This spec instead spends 4-8 M1-days and returns a
champion whose gate had to be rewritten to accommodate it.
