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

---

## 2026-07-25 — TRACK B FALSIFIED + the opening-composition thesis KILLED

**Self-play v3 (`pfsp_k` 1.0→2.0 arm) auto-stopped SOFT at update 200**: 0 promotions,
`updates_since_promotion=200`, anchor-window recent_median **0.5233** vs bar 0.58, `setup_head_entropy`
0.249 (healthy), heuristic eval 0.99 (saturated). Clean exit, final ckpt saved. The fail-fast soft
threshold (lowered 300→150 for this arm) stopped it at u200 instead of u500.

**Two independent runs now agree the lineage has plateaued at `ckpt_500`**: v2 at u500 (median 0.50),
v3 at u200 (0.52). A league-mix recipe knob is NOT the lever. No funded training hypothesis remains;
the M1 is deliberately idle.

**The opening resource-composition thesis is DEAD — three independent kills, each verified:**
1. The pre-registered gate had ALREADY fired CLOSE: ore-substitution 0.2662 [0.2452, 0.2884] vs the
   ≥40% bar set at `:63-66`. Continuing the thread was gate-shopping.
2. The flagship metric measured BOARD SUPPLY, not policy choice: "75.6% below random on
   `pair_max_ore_lump`" is arithmetically identical to "a strictly-higher-lump alternative existed"
   (605/800); the metric is 3-valued {0,1,2} = 67/694/39.
3. "Below random" was a MEDIAN-vs-MEAN reading error. Mid-rank percentile has E[pct]=50 (a MEAN
   property); on tied discrete metrics a uniform chooser's MEDIAN is far below 50. Simulated on
   identical candidate sets: `pair_max_ore_lump` random median 46.5 vs policy 46.7; `ore_pips` random
   38.4 vs policy 41.9; policy MEANS 54.5 / 61.6 vs random 50.1 / 49.9. **Everything called "below
   random" is at or ABOVE random.**

Also verified: **the policy DOES condition on what it already holds** — settlement #1 without ore ⇒
2nd-settlement ore percentile 83.7, P(touch ore) 0.672; with ore ⇒ 40.2 and 0.391 (596/800 already
hold ore, so the pooled 41.9 is a Simpson artifact).

**What survives: the REFERENCE-CLASS problem.** Every judgement has been made against a baseline that
is unreadable ("vs random"), mirror-contaminated ("vs v11"), or saturated ("vs heuristic" ≈1.00). The
axes the owner actually named — contested middle, denial, road-cut vulnerability — have ZERO metrics
(the sweep's own limitations 7-8 disclaim jurisdiction). 140 corpus games with BOTH players' openings
sit unused. Direction: `.claude/veriloop/specs/human-opening-reference.md`.

**Ranked next after that (cross-exam ruling): aggregate midgame instrumentation** — cities-built,
bank-trade composition, VP-rate over existing self-play/eval corpora, n in the thousands, zero new
games, no baseline problem. Above any further opening metric.
