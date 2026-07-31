# Spec: bc-coverage-and-bank-legality — fix what the bootstrap never taught, and stop offering illegal bank moves

**Status: RATIFIED (2026-07-30) — BINDING. Owner ratified in full view of the open RISKS below.**

**Feature in one line.** Close the four defects that make the behavioural-cloning bootstrap teach
the wrong thing — a `forced` filter that silently deletes real decisions, a practice opponent that
has never played a development card, an unobservable discard counter, and a bank-blind action mask
that offers moves the reference engine refuses.

## Provenance

A six-domain audit of the engine against the Colonist.io 1v1 ruleset (27 agents, each candidate gap
adversarially re-verified) surfaced six findings; a three-expert council then **overturned three of
the five originally proposed fixes**. Two would have made the codebase worse. What follows is the
surviving plan, with the measurements that decided it.

---

## Decisions (binding)

### D1 — `forced` becomes relevance-aware, scoped to SETUP + ROBBER only. DISCARD stays excluded.

`bc/dataset.py:234` computes `forced = bool(int(mask["type"].sum()) <= 1)` from the **type head
alone**, so any decision with a singleton type mask and a multi-way downstream head is deleted at
write time. Measured on `shard_0000.npz`: **928,039 / 928,039 rows are `phase == "main"`**.

A 40-game probe measured exactly what relevance-aware forcing would restore:

| restored | rows | verdict |
|---|---|---|
| `MOVE_ROBBER` | 3,312 (19 distinct tiles) | **real** — `choose_player_to_rob` is a genuine hex-scoring heuristic |
| setup | 320 (24 corners, 41 edges) | **real** — actual recorded placements |
| `DISCARD` | 2,342 | **FABRICATED — must stay excluded** |

**Why DISCARD is excluded, and this reason goes in the docstring:** `bc/dataset.py:478` records
`_make_action(ActionType.DISCARD)` with no resource argument, so `resource1_idx` defaults to `0` =
WOOD. **All 2,342 rows say WOOD**, while their `mask/resource1_discard` averages **4.46 legal
picks** and 100% have more than one. And the teacher discards `np.random.choice(...)`
(`agents/heuristic.py:273`) — uniformly at random. Restoring them would put ~11% of the corpus into
teaching a constant lie from a skill-free teacher. The existing comment *"Acceptable: they're
forced"* is the only guard; D1 must replace it with an explicit exclusion and this rationale.

**Discard is learned by self-play instead, and always has been** — `catan_env.py:448-456` takes one
`env.step` per discarded card with the policy choosing the resource, and the `resource1` head is
already trained by 87,125 bank-trade rows (`policy/heads.py:246-249` shares it across YoP /
Monopoly / BankTrade / Discard). D3 makes that learning tractable.

**Implementation constraints (all verified):**
- `head_relevance` is a **torch buffer** on `MultiActionHeads` (`heads.py:232-252`); `bc/dataset.py`
  is deliberately torch-free. Move the relevance table **and** a `(type → head → mask-key)` map into
  `policy/obs_schema.py` (the torch-free SoT), have `heads.py` build its buffer from it, and have
  `dataset.py` read it.
- There are **9 mask keys for 6 heads** (`obs_schema.py:157`): head 1 splits
  `corner_settlement`/`corner_city`, head 4 splits `resource1_trade`/`resource1_discard`/
  `resource1_default`. `replay/recorder_loop.py:184-197` already has `_corner_mask_for_type` /
  `_resource1_mask_for_type` — reuse them.
- Head 5 is **autoregressive** (`heads._resource2_mask` needs a chosen res1) — treat it as
  never-singleton.
- `search/mcts.py:337` carries a **third** copy of this rule; cross-test the new predicate against
  its enumeration rather than adding a fourth definition.
- `expert_iteration/labeler.py:61` is a byte-identical copy where `forced` also drives the
  **loop-termination counter** `total_nonforced` (`:139,150`). Fix both call sites in this change.
- Setup rows expand the corpus ~**+26%** (8 rows/game × 30k). `bc/loss.py:71-83` has no phase
  reweighting, so ~26% of policy gradient becomes opening placement, hitting the corner head the
  fork is rebuilding. This is accepted deliberately, not incidentally.

### D2 — Give the practice opponent a real dev-card policy (the highest-value item, and it was not on the original list)

`agents/heuristic.py:210-227` — `heuristic_play_dev_card` is **entirely commented out**. The corpus
holds **50,426 BUY_DEV_CARD rows and zero plays**: types 6/7/8/9 (Knight, YoP, Monopoly, Road
Builder) have never appeared as a positive example. **No filter change recovers an action the
teacher never took.**

Write a real policy — Knight when the robber sits on your own hex or to block the opponent's best
number; YoP/Monopoly when it completes a build; Road Builder when it extends the longest road — and
patch the dev-card play methods into `_instrumented_player` (`dataset.py:249-260` currently patches
only `build_*` / `draw_devCard` / `trade_with_bank` / `move_robber`).

Second benefit: this un-saturates the heuristic eval baseline, whose 0.97–0.99 is what made this
lineage's progress unreadable (`configs/selfplay_pointer_arch.yaml` S4/S5 notes).

**This SUBSUMES `preroll-dev-cards-r1.md` D6** (opponent-side pre-roll). Implement the pre-roll
Knight case here; R1's D6 then consumes it rather than re-specifying it.

### D3 — Make the remaining-owed discard count observable

The discard is decomposed into `floor(H₀/2)` independent `env.step` calls and the count is fixed at
roll time then decremented invisibly — so at a 9-card hand the policy cannot distinguish "owes 3,
started at 12" from "owes 4, started at 9". Self-play is solving a sequential problem blind to its
position in the sequence.

- **Do NOT shrink `RESERVED_PLAYER_SLOTS`** (`obs_schema.py:68`). It feeds **both**
  `CURR_PLAYER_DIM` (67) and `NEXT_PLAYER_DIM` (69); decrementing forks every checkpoint — the exact
  opposite of the reserved-slot design's purpose. Bump `CURR_EXTRA_DIM` 5→6 and introduce a separate
  `CURR_RESERVED_SLOTS = 7`.
- Add `cards_to_discard: int = 0` to `EnvObsState` (default keeps every other caller valid) and
  thread it at `catan_env.py:1124` **and** through `_opponent_env_state` (`:846-872`) — the
  opponent's count currently lives only as a local in `_opponent_discard` (`:889-902`). Missing the
  second seat trains a self-play asymmetry into the exact slot being added.
- Normalise to the block's convention, `min(1.0, n / 8.0)`, matching `obs_encoder.py:687-690`.
- **Must land BEFORE any BC regeneration.** Shards store *featurized* obs; if rows are written while
  the slot is still 0.0, BC teaches "always zero" while the env emits nonzero — silent train/serve
  skew the loader cannot catch.
- `tests/unit/policy/test_pointer_arch.py:97` passes **vacuously** (a fresh env is never
  discarding). Narrow it to the still-reserved tail and add a discard-state assertion.

### D4 — Split `resource2_default`; bank-gate BOTH Year of Plenty and BankTrade at the legality layer

`env/masks.py:227-228` sets `res2_def[:] = True` for `BANK_TRADE` with **no bank check**, and
`engine/player.py:533-535` early-returns leaving state byte-identical. The mask keeps offering the
move, so a stable-argmax policy re-picks it forever: `eval/harness.py:401` loops on
`while not terminated and not truncated`, and truncation only advances at a turn boundary. **This is
an unbounded loop that already exists today** — a probe reproduced it as a fixed point across five
steps with identical hand, obs, and flags.

- Split `resource2_default` into `resource2_yop` and `resource2_trade` (a 10th mask key). The shared
  key is genuinely why bank-awareness is currently inexpressible.
- Gate at legality: `BANK_TRADE` requires `bank[r2] >= 1`; YoP requires `bank[first] >= 2` when
  `first == second`, else `bank[first] >= 1 and bank[second] >= 1`. This mirrors
  `Torevan/packages/engine/src/legal-moves.ts:366-376`, which enumerates only fully-supplyable picks.
- **Do NOT add an early-return no-op.** That is what creates the fixed point. YoP is currently
  livelock-free precisely because `catan_env.py:604` sets `devCardPlayedThisTurn = True` before
  granting; removing that guarantee would add a hang, not fix one. With the mask gated, the
  unreachable-branch question is moot.
- Scope stays in `env/` — the engine's AI branch already bank-checks (`player.py:459-463`), so
  `PINNED_ENGINE_TREE` is **not** re-pinned and clause (a)'s provenance survives.
- Add a **step cap** to `eval/harness.py:401` as defence-in-depth so any future no-op-mask bug is
  survivable rather than a hang.
- Amend `specs/009-finite-resource-bank/review-resolution.md:37`, which claims S1 landed this guard
  on "every YOP/bank-trade draw path". Only bank-trade got it. That is drift, and worse than the bug.

### D5 — Make the ordering enforceable, not prose

Add `(forced_rule_version, ruleset_version)` to the BC manifest and make `bc/loader.py` **refuse** an
unstamped or stale directory. Today `bc/loader.py:294,392` reads `manifest["shards"]` and validates
no version, so post-D1 old and new shards are indistinguishable and the 3.9 GB stale corpus would
silently train the new lineage. This consumes `preroll-dev-cards-r1.md` D7's ruleset-stamp
requirement rather than re-specifying it.

Also repair the dead telemetry that let a 100%-forced-drop ship unnoticed: the shipped manifest reads
`total_decisions_pre_filter = 0`, `forced_move_drop_pct = 0.0` on a 928k-row corpus, because the
counters initialise at `dataset.py:759-760` and increment only inside `for i in range(games_done, …)`
(`:837`). `tests/unit/bc/test_dataset.py:328` asserts `0.0 <= pct < 1.0`, which passes on 0.0 —
vacuous. Restore counters from `progress.json` on resume and tighten the assertion.

### D6 — Binding build order

**D2 → D3 → D4 → D1 → (BC regen, in `preroll-dev-cards-r1.md` D7).**
D2 first because the teacher must be worth cloning before any filter change matters; D3 before any
regen; D1 last because every iteration of it costs a full 30k-game regeneration (nothing dropped is
recoverable — `bc/loader.py:111`: `forced` is never consumed by the loader, so the write-time filter
is the only gate).

### D7 — TDD is binding for this slice

Every decision above lands **test-first**: the failing test is written and observed failing against
the current tree before the fix. Explicitly including the test whose absence cost a 30,000-game
generation — a **shard-level** assertion that a small generated dataset contains
`phase ⊇ {setup, main, robber}` and a type histogram covering `{0,1,2,3,4,5,10}`.
`tests/unit/bc/test_dataset.py:57` asserts phase tagging at `play_game` level, *before*
`_flatten_records` applies the filter, so it stayed green while 100% of setup rows were dropped.

---

## Non-goals

- **Mid-turn win detection — DROPPED.** `masks.py:162` makes `END_TURN` unconditionally available and
  `catan_env.py:814` short-circuits the opponent at `maxPoints`, so no win is ever lost or reversed.
  The only delta is the `vp_margin_bonus` reward channel (`:1050`) — a reward-shaping change, not a
  correctness fix, and shipping it inside a slice that re-baselines everything would make it
  permanently unattributable.
- **Search dice clairvoyance — its own spec.** `search/mcts.py:163` deep-copies the Rust
  `StackedDice` including its ChaCha8 counter, so every determinization shares the true future rolls
  (`n_determinizations` varies only the opponent model). Real, and the prerequisite for any honest
  chance-node work — but it needs a Rust reshuffle API that does not exist, a posterior-consistent
  bag-resampling decision, and a pre-registered re-measurement of the banked +54.6 Elo. Verified: the
  `[23.9, 85.4]` window is **printed, never asserted** (`elo_ladder.py:461` feeds one f-string), and
  **no search rung was ever a promotion anchor**, so v9/v10/v11 are uncontaminated.
- **Cloning discard behaviour** (D1) — self-play owns it.
- **No engine game-rule change** and no `PINNED_ENGINE_TREE` re-pin.

## Acceptance criteria (the `/dev-loop` gate is the authority)

1. Full gate green on real exit codes.
2. Every fix has a test that **failed first** against the pre-change tree (D7).
3. Shard-level coverage test: a generated dataset contains setup and robber rows and **no** DISCARD
   rows, with the type histogram covering `{0,1,2,3,4,5,10}` and dev-card play types present.
4. `forced` agrees with `search/mcts.py:337`'s independent enumeration on random states.
5. Both `bc/dataset.py` and `expert_iteration/labeler.py` use the single shared predicate.
6. Discard-owed scalar is correct for **both** seats at every sub-step; the reserved-slot test is
   narrowed, not deleted; `CURR_PLAYER_DIM`/`NEXT_PLAYER_DIM` arithmetic pinned.
7. A bank-unsupplyable YoP or BankTrade receive is **not offered**; a no-op-repeat cannot loop
   (harness step cap pinned).
8. `bc/loader.py` refuses a stale/unstamped manifest; `forced_move_drop_pct` reports a real number.
9. `PINNED_ENGINE_TREE` unchanged at `261098d190c8`.
10. Docs sync per CLAUDE.md §6, including the `review-resolution.md:37` correction.

---

## RISKS — open, NOT cleared

**Pre-mortem.** The slice spends the one free BC-regen window, and a year later the champion is
gated against itself: accept-gate clause (a) is in-lineage h2h where both seats trained on the same
corpus, and clause (b) is setup-only — so the dual gate is structurally blind to a corpus defect in
the middle game. The originally-specced version of this failure (2,342 fabricated WOOD labels at
full CE weight) is closed by D1's exclusion; the *shape* — a training-data change validated by a gate
that cannot see training data — is not.

**Cost.** BC regen is 12 shards / ~3.9 GB / ~16h, then BC train, then bootstrap (14h45m), then
self-play (~41h). D1 is deliberately last because each iteration re-pays that.

**Argue the other side.** These four do not obviously belong in one spec — one gate over four
independent hypotheses, in a repo whose largest win came from isolating a single line (the v8
promotion bar, +121 Elo) and whose most expensive nulls came from bundles. And an unforced error sits
upstream of all of it: `runs/train/bootstrap_pointer_arch_20260728_195817/` reached update 140/1526
with no live process and `dirty: 'true'` in its metadata — the ratified re-bootstrap path is
unaccounted for. Committing to a slice that mandates re-running BC gen + BC train + bootstrap,
layered under an already-ratified spec mandating the same chain, without first establishing whether
that run died or was killed, is the highest-probability route to spending 8 M1-days and not being
able to say what any of it bought.
