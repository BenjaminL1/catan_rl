# Opening-deficit debate — verdict (2026-07-24)

Trigger: first human playtest of the CURRENT champion (`selfplay_pointer_arch_v2/ckpt_000000500`,
RAW policy). Human won 15-4. Owner reported: weak vertex selection, weak road direction, midgame
"human level".

## Finding: the instinct was right, both localizations were wrong

**Vertex selection was pip-OPTIMAL** — the bot's pair (v13,v11) = 22 joint pips, top ~1% of 1359
legal pairs. All 23-pip pairs contained v8, which the human took first (bot drafted second). 22 was
the *maximum available conditional on the board it faced*.

**Road direction was the 2-hop pip argmax** on both roads; the plan executed (settled v16 at step 26).

**The real defect is resource COMPOSITION — specifically ore-blindness.** Board total ore = 6 pips.
Bot opening take: WHEAT 12 / BRICK 5 / SHEEP 3 / WOOD 2 / **ORE 0**. It declined a 1-pip-for-3-ore
swap at the settlement and a 2-pip-for-4-ore swap at the road — the same trade twice.

Downstream consequences (INDEPENDENTLY VERIFIED from the raw move log, not from the debate):
- **0 cities in 58 turns** (`BuildCity` count = 0)
- **6 of 12 bank trades were BUYING ore** (3x wheat->ore, 2x sheep->ore, 1x wood->ore)
- 7 dev cards, 11 roads, 4 settlements; final 4 VP = 4 settlements, no cities

## The opening did NOT lose the game

VP trajectory (verified): the bot **LED 5-4 at step 46**, flatlined at 5 VP, then **lost 2 VP at
step 109** (longest-road flip) while the human ran 6 -> 15. The position off that opening was
competitive. The game was lost in the MIDGAME conversion, not the opening.

Consequence: the owner's stated confound ("weak openings make the midgame unjudgeable") is itself
contradicted — and the "midgame is human level" read is unblinded grading of *visibility*. Dev cards
and roads are visible at a glance; a 58-turn failure to ever build a city while buying ore at 4:1 is
not. The valuation defect spans setup AND midgame.

Magnitude bound: perfect ore reallocation is worth ~2 VP against an 11-VP gap. **No proposal on the
table is within ~4x of the observed gap.** n=1.

## Refuted (do NOT fund)

- **pip/pip-sum injection into vertex dims 14-15** — would sharpen the exact metric that produced the
  failure (the opening was already pip-optimal). Also `vertex_features` is `Box(low=0,high=1)`
  (`catan_env.py:257-259`) so any injection MUST be scaled to [0,1].
- **edge-head setup FiLM as an information fix** — `in_setup` is already a raw phase flag at index 34
  of BOTH player blocks (`obs_encoder.py:629-642`) feeding the trunk the edge head reads. FiLM adds a
  modulation channel, not information.
- **the "one root cause spans both heads" story** — REFUTED. corner reads `v` (own hexes aggregated
  twice), edge reads `e2` (production only via 2 endpoints). Different states, different failures.
- **`setup_phase/analytic_value.py` as an aux target** — linear Sum(dots*w_r), scores the exact failure
  21.1 vs ~21 (indifferent to the mistake it was meant to fix); `CHARLESWORTH_V0` is a 4-PLAYER prior;
  and `edge_yield_after_settlement` (:121-129) IGNORES its `edge_idx` arg — a live bug if wired.
- **one-ply value lookahead as an ore fix** — signed BACKWARDS: the value head's argmax (v14) is
  10 pips / 0 ore, i.e. lookahead moves the opening further from ore.
- **refitting the value squash to change rankings** — `squash_value` is monotone, Spearman-invariant.
- **raising `setup_entropy_coef`** — precedent: 0.02 -> entropy 0.21->2.3, 0 promotions, 17h lost.
  Diversify the OPPONENT's setup instead (same distributional effect, learner untouched).
- Citation error to avoid: **verdict (a) value-ceiling has NO jurisdiction over setup** — that probe
  reservoir-sampled non-forced-type states and setup's type head is forced.

## Next (all CPU-only; v3 on the M1 continues, soft-stops naturally)

1. **Conditional opening sweep** — repoint `scripts/pregate0.py` at ckpt_500, 200 boards x both seats
   x {greedy opponent, forced-diverse opponent}. Setup roots are forced-type so MCTS short-circuits:
   ~1600 policy forwards, MINUTES. Outputs: conditional pip percentile, **ore-substitution rate**,
   road argmax rate + the shared/non-shared hex partition.
   **GATE: ore-substitution rate >= 40%** -> composition deficit is systematic. **< 40% -> the whole
   opening thread closes** and attention moves to midgame allocation.
2. **Instrument the playtest logger** (owner, ~1h, 0 compute): log HUMAN moves + per-step hands
   (currently only `bot_action` is recorded — the human's v8 pick had to be *inferred*). Play 4 more
   games -> n=5.
3. **Then measure VALUE, not rate** — paired counterfactual setup rollouts: force the ore-substituted
   opening vs the greedy one, identical seeds/policies, play to terminal, measure paired dWR.
   n~2000 pairs ~= 1.4 CPU-core-hours. This is the ExIt STEP-2 design that correctly killed
   distillation (overrides fired at 7.3% and were individually WIN-NEUTRAL). A confirmed rate with
   zero value is the same trap.
