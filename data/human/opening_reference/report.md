# Human opening reference — champion vs ThePhantom on HIS boards

**READ THIS BEFORE READING ANY NUMBER.**

* **This is a REFUTATION instrument (spec D4).** ThePhantom is a strong human,
  not an oracle, so every divergence below is **UNSIGNED**: the policy differing
  from him is not evidence of a defect, and matching him is not evidence of
  strength. The instrument can *refute* "the openings are weak"; it can **never
  confirm** superiority. It measures **no win-probability** and runs **no
  rollouts**.
* **Ports are marginalised, not known (spec D2).** All corpus rows carry
  `board.ports = "OMITTED in v1"`. Ports ARE in the policy obs, so the policy
  here chooses under 8 deterministic sampled port assignments per board and the
  results are means over them. This is a real limitation. The **port-marginal
  churn** column quantifies it: it is the fraction of assignments that did not
  take the modal choice.
* **`opponent_strength` is a hardcoded constant across all 140 corpus rows
  (spec D6) — ZERO BITS.** It is not stratified on, not weighted by, and not
  cited as validation anywhere in this report.
* **Agreement rate is descriptive colour, NOT a gate (spec D5).** Move-agreement
  is an explicit non-goal of step6 §6.
* **Multiplicity — the CIs below are UNADJUSTED, and omitting p-values does NOT
  remove the multiplicity.** The pre-registered PRIMARY family is the D3 metric
  gap on settlement decisions (`contested_race_count`, `contested_race_pips`, `denial`); every other row is exploratory. A 95%
  CI that excludes zero *is* a rejected two-sided test at alpha=0.05 — dropping
  the p-value drops the label, not the test. See "Multiplicity accounting" below
  for the row count, the number of exploratory CIs excluding zero, and how many
  would be expected to do so by chance alone. Treat each exploratory row as a
  hypothesis to re-test on fresh data, not as a finding.
* **Some metrics are UNDEFINED for some picks, and dropping those decisions
  selects on the outcome.** Every table carries the definedness columns
  (`human undef` / `policy undef` / `partial k`) and any metric with a nonzero
  drop count gets an explicit informative-missingness warning. Do not read a
  wide CI on a shrunken `n` as "no difference".
* **The D3 metrics are hypotheses about the axes the owner named, not
  established quantities.** Their definitions are restated in full under "Metric
  definitions (D3)" below, and live in the module docstring of
  `scripts/human_opening_reference.py`.


## Provenance

* checkpoint under test: `/Users/benjaminli/my_projects/catan_rl_v2/runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt`
* corpus rows seen: 140; replayed: 134
* paired decisions: 536 (268 settlement, 268 road)
* distinct videos among the REPLAYED rows (bootstrap clusters): 113 — this is a POST-exclusion count and is smaller than the corpus video census below whenever rows are excluded.
* port assignments per board (K): 8
* per-decision raw records (every number here is re-aggregatable from them): `data/human/opening_reference/raw.json` — one JSON object per paired decision, including the human's pick, the K policy picks, and every metric value for both.

## Corpus census (spec D6 — recomputed, not quoted)

| quantity | measured | spec D6 expects |
|---|---:|---:|
| rows | 140 | 140 |
| videos | 118 | 118 |
| unique_layouts | 135 | 135 |
| winners_known | 117 | 117 |

The census reproduces spec D6 exactly, so the population behind every number below is the population the spec was written against.

Two provenance facts that bear on how much to trust the placement ORDER (the load-bearing input to the whole pairing) — disclosed, not filtered:

* `provenance.order_source` across the corpus: `None` x2, `glyph_only` x137, `log+glyph` x1. Glyph-anchor-only ordering is an OPEN owner decision, so a result resting on it inherits that open question.
* rows whose `video_id` is a SYNTHETIC placeholder (the hardcoded default in `scripts/vlm_spike.py`): `vlm_spike g1`. Such a row is not a real video: it still contributes decisions and counts as one bootstrap cluster.

## PRIMARY — D3 metric gap on settlement decisions

Pre-registered. A gap of 0 means the policy's pick scores exactly as ThePhantom's pick did on the same board state; the sign is NOT a quality ordering (D4).

| metric | decisions used | of kind | human undef | policy undef | partial k | videos | human mean | policy mean | gap (policy - human) | 95% CI (cluster) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `contested_race_count` | 180 | 268 | 88 | 0 | 0 | 113 | 4.572 | 4.642 | +0.070 | [-0.167, +0.313] |
| `contested_race_pips` | 180 | 268 | 88 | 0 | 0 | 113 | 32.489 | 31.833 | -0.656 | [-2.184, +0.908] |
| `denial` | 180 | 268 | 88 | 0 | 0 | 113 | 3.594 | 3.544 | -0.050 | [-0.147, +0.048] |

### Why the PRIMARY `n` is smaller than the settlement-decision count

The D3 contested/denial metrics are UNDEFINED when the opponent has not placed yet, which is exactly the seat-0 first settlement — those decisions are excluded by construction, not lost to failure. The remaining regimes differ in how many opponent settlements the metrics are measured against, so the pooled mean is a mean over a MIX:

| agent seat | decision index | opponent settlements | decisions | in PRIMARY? |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 88 | NO (undefined by construction) |
| 0 | 2 | 2 | 88 | yes |
| 1 | 0 | 1 | 46 | yes |
| 1 | 2 | 1 | 46 | yes |

The pooled gap is only readable if it is not an average of offsetting signs, so here is the same PRIMARY gap computed WITHIN each pooled regime. Point estimates only — the per-regime `n` does not support a cluster bootstrap, so these are a sign-cancellation check, not findings:

| regime (seat / decision / opp settlements) | n | `contested_race_count` gap | `contested_race_pips` gap | `denial` gap |
|---|---:|---:|---:|---:|
| seat 0 / decision 2 / opp 2 | 88 | +0.178 | -0.182 | -0.077 |
| seat 1 / decision 0 / opp 1 | 46 | -0.008 | -1.486 | -0.041 |
| seat 1 / decision 2 / opp 1 | 46 | -0.057 | -0.731 | -0.008 |

### Bottom line on the pre-registered question

**All 3 primary CIs straddle zero: there is NO DETECTABLE DIFFERENCE between the policy's picks and ThePhantom's on his own contested-middle / denial axes, on his own boards, at his own decision points.** Under D4 that is the refutation direction: this instrument found no support for "the openings are weak" on the axes the owner named. It is NOT evidence of superiority, it is not a win-probability, and 'no detectable difference' at this `n` is not 'identical'.

**How tight is the null? (D4's refutation logic rests entirely on this.)** A null only licenses closing a thread if it EXCLUDES gaps big enough to matter, so here is the precision, as the **two-sided exclusion bound** `max(|CI_lo|, |CI_hi|)` expressed against the human's own level on the same metric. It is deliberately NOT the CI half-width: these CIs are not centred on zero, and a half-width would understate the magnitude the interval actually still admits — in the direction that flatters the refutation.

| metric | human mean | exclusion bound `max(\|CI_lo\|, \|CI_hi\|)` | excludes gaps larger than |
|---|---:|---:|---:|
| `contested_race_count` | 4.572 | 0.313 | ~6.8% of the human level |
| `contested_race_pips` | 32.489 | 2.184 | ~6.7% of the human level |
| `denial` | 3.594 | 0.147 | ~4.1% of the human level |

So this run rules out gaps whose MAGNITUDE is at or above the bound in the last column (here 4.1% to 6.8% of the human's own level on the axis); anything SMALLER than that is inside the noise and would need a bigger corpus (more videos, i.e. more bootstrap clusters — not more port samples, which do not add independent information).

Primary rows verbatim: `contested_race_count` +0.070 [-0.167, +0.313] (n=180); `contested_race_pips` -0.656 [-2.184, +0.908] (n=180); `denial` -0.050 [-0.147, +0.048] (n=180).

### The exploratory row that bears directly on the owner's original question

The feature was commissioned because of a suspected **opening ore/city blind spot**. The only paired, same-position, non-reference-class row here that speaks to it is `pair_city_self_sufficient`: gap -0.165 [-0.253, -0.083] (human 0.896 vs policy 0.730, n=134). It is EXPLORATORY, not pre-registered, and it scores a TEACHER-FORCED pair (ThePhantom's settlement #1 + the chooser's #2), so it is a hypothesis for a fresh sample — but it is the decision-relevant number this run produced, and it is stated here rather than buried in a footnote about another metric.

**The mechanism INVERTS the 'ore-blind' framing.** On the same decisions the policy takes MORE ore, not less (`pair_ore_pips` in the exploratory table), yet fails self-sufficiency more often, and the failures are the wrong KIND: `pair_city_self_sufficient` needs ore > 0 AND wheat > 0.

| failure mode of the pair | human (expected count) | policy (expected count) |
|---|---:|---:|
| has ore but ZERO wheat | 1.0 | 22.2 |
| no ore at all | 13.0 | 13.9 |

(of 134 paired second-settlement decisions; policy counts are expectations over the K port assignments.) The ore-absent rates are comparable — the deficit is concentrated in **ore-bearing pairs with no wheat**, i.e. a wheat/complementarity omission, NOT an ore deficit. That is the opposite of the hypothesis that motivated the build, and the follow-up it suggests is about pairing ore with wheat, not about taking more ore.

## Exploratory — existing battery on settlement decisions

**`denial_reach` is the flagged sharpening of the PRIMARY `denial` row, not a replacement for it** — see "Metric definitions (D3)": the primary row implements spec D3 literally, and the restricted variant is reported here for the owner to rule on.

**Every `pair_*` row scores a TEACHER-FORCED HYBRID, not the policy's own opening pair.** The pair metrics are only defined at the second-settlement decision, and by D1's pairing the first settlement there is *ThePhantom's* (the policy never chose it). So a `pair_*` "policy mean" is the score of (ThePhantom's settlement #1, policy's settlement #2): it measures how the policy COMPLETES a human first settlement, and it cannot support any claim about the pairs the policy builds when it opens for itself.

| metric | decisions used | of kind | human undef | policy undef | partial k | videos | human mean | policy mean | gap (policy - human) | 95% CI (cluster) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `denial_reach` | 180 | 268 | 88 | 0 | 0 | 113 | 2.572 | 2.456 | -0.116 | [-0.299, +0.057] |
| `pip_sum` | 268 | 268 | 0 | 0 | 0 | 113 | 10.612 | 11.310 | +0.698 | [+0.530, +0.856] |
| `ore_pips` | 268 | 268 | 0 | 0 | 0 | 113 | 2.343 | 2.923 | +0.579 | [+0.202, +0.964] |
| `has_ore` | 268 | 268 | 0 | 0 | 0 | 113 | 0.549 | 0.604 | +0.055 | [-0.014, +0.125] |
| `wheat_pips` | 268 | 268 | 0 | 0 | 0 | 113 | 2.384 | 2.472 | +0.087 | [-0.261, +0.426] |
| `robber_robustness` | 268 | 268 | 0 | 0 | 0 | 113 | 5.854 | 6.552 | +0.698 | [+0.546, +0.846] |
| `exp_d2` | 268 | 268 | 0 | 0 | 0 | 113 | 5.336 | 5.432 | +0.097 | [+0.001, +0.193] |
| `exp_d3` | 268 | 268 | 0 | 0 | 0 | 113 | 6.604 | 6.395 | -0.210 | [-0.350, -0.062] |
| `exp_d4` | 268 | 268 | 0 | 0 | 0 | 113 | 6.884 | 6.303 | -0.582 | [-0.799, -0.367] |
| `exp_d2_pip` | 268 | 268 | 0 | 0 | 0 | 113 | 37.970 | 36.472 | -1.499 | [-2.425, -0.540] |
| `exp_d3_pip` | 268 | 268 | 0 | 0 | 0 | 113 | 43.399 | 40.598 | -2.801 | [-3.719, -1.846] |
| `exp_d4_pip` | 268 | 268 | 0 | 0 | 0 | 113 | 44.142 | 41.676 | -2.465 | [-3.735, -1.352] |
| `centrality_dist` | 268 | 268 | 0 | 0 | 0 | 113 | 2.269 | 2.493 | +0.224 | [+0.146, +0.302] |
| `pair_pip_sum` | 134 | 268 | 134 | 0 | 0 | 113 | 21.224 | 22.042 | +0.818 | [+0.572, +1.060] |
| `pair_ore_pips` | 134 | 268 | 134 | 0 | 0 | 113 | 4.687 | 5.331 | +0.645 | [+0.290, +1.024] |
| `pair_wheat_pips` | 134 | 268 | 134 | 0 | 0 | 113 | 4.769 | 4.768 | -0.001 | [-0.452, +0.429] |
| `pair_city_self_sufficient` | 134 | 268 | 134 | 0 | 0 | 113 | 0.896 | 0.730 | -0.165 | [-0.253, -0.083] |
| `pair_exp_rolls_to_city` | 92 | 268 | 148 | 28 | 2 | 80 | 26.038 | 25.997 | -0.041 | [-1.241, +1.418] |
| `pair_max_ore_lump` | 134 | 268 | 134 | 0 | 0 | 113 | 0.910 | 0.938 | +0.027 | [-0.029, +0.090] |
| `pair_robber_robustness` | 134 | 268 | 134 | 0 | 0 | 113 | 16.276 | 17.052 | +0.776 | [+0.543, +1.010] |
| `pair_spread` | 134 | 268 | 134 | 0 | 0 | 113 | 3.997 | 4.141 | +0.144 | [-0.022, +0.317] |

> **INFORMATIVE MISSINGNESS — READ BEFORE THE ROWS ABOVE.** For the metrics listed here the comparison subset is SELECTED ON THE OUTCOME: a decision is dropped precisely because the policy's pick left the metric undefined, so the surviving rows are the ones where the policy did NOT fail. Their gaps are survivor-biased and must not be read as "no difference".
>
> * `pair_exp_rolls_to_city`: 268 decisions of this kind, 148 undefined for the human, **28 dropped because EVERY port sample of the policy's pick was undefined**, 2 averaged over a partial k -> n = 92.
>
> `pair_exp_rolls_to_city` is the load-bearing case. It is undefined unless the pair produces both ore and wheat, and `docs/plans/opening_sweep_results.md:456` already declares it NOT comparable across slices with different self-sufficiency rates. The `pair_city_self_sufficient` row in the same table is exactly that difference (the policy is materially LESS often self-sufficient), so its gap is computed on the subset where the policy did not fail. The honest reading of the pair is the `pair_city_self_sufficient` row, not the `pair_exp_rolls_to_city` row.

## Exploratory — road decisions

**`cut_vulnerability` came out IDENTICALLY ZERO on every decision in this run.** All 268 human evaluations and all their policy counterparts are exactly 0.0, so the column carries no variance to compare here — but that denominator is **not all empirical**: 134 of the 268 are the FIRST-road decision, where the chooser owns no prior road, the induced subgraph is the single candidate edge, and 0.0 is true **BY CONSTRUCTION** (a 1-edge graph has no articulation point; the module's own unit test pins it). The honest empirical denominator is the **134 SECOND-road decisions**, where a self-cuttable pair was reachable and neither chooser built one. On those the null is genuinely EMPIRICAL, not a theorem: the distance rule forbids ADJACENT settlements, but two of my settlements may sit at graph distance 2 sharing a middle vertex, and if both setup roads point at that shared vertex the road subgraph is a 3-vertex PATH whose middle IS an articulation point. That double-back is legal and available — verified against the engine masks on corpus row `b_NXNI-l0kI` g6 (agent settlements 15 and 14, shared neighbour 13, recorded first road 13-15, and edge 13-14 a legal second-road candidate; scoring that pair returns `cut_vulnerability` = 1.0). So the reading is: **on the 134 decisions where a self-cut was even expressible, neither ThePhantom nor the policy ever built a self-cuttable road pair**, which is a finding about both choosers, and the articulation-point family is NOT shown to be incapable of signal outside the setup regime. Note separately that this definition (articulation points of MY OWN roads) cannot express the 'plow' the owner described — an OPPONENT road severing my longest path — which would need the opponent's road network and a longest-trail recomputation.

| metric | decisions used | of kind | human undef | policy undef | partial k | videos | human mean | policy mean | gap (policy - human) | 95% CI (cluster) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `road_2hop_pip_max` | 268 | 268 | 0 | 0 | 0 | 113 | 9.060 | 9.631 | +0.571 | [+0.328, +0.833] |
| `road_2hop_pip_sum` | 268 | 268 | 0 | 0 | 0 | 113 | 14.586 | 15.525 | +0.939 | [+0.478, +1.403] |
| `road_breadth` | 268 | 268 | 0 | 0 | 0 | 113 | 5.138 | 5.229 | +0.091 | [-0.043, +0.224] |
| `road_breadth_pip` | 268 | 268 | 0 | 0 | 0 | 113 | 38.377 | 41.418 | +3.042 | [+1.837, +4.219] |
| `nonshared_pip_sum` | 268 | 268 | 0 | 0 | 0 | 113 | 2.638 | 3.019 | +0.381 | [+0.251, +0.522] |
| `nonshared_ore_pips` | 268 | 268 | 0 | 0 | 0 | 113 | 0.597 | 0.683 | +0.086 | [-0.067, +0.242] |
| `shared_pip_sum` | 268 | 268 | 0 | 0 | 0 | 113 | 6.903 | 7.057 | +0.154 | [-0.022, +0.328] |
| `shared_ore_pips` | 268 | 268 | 0 | 0 | 0 | 113 | 1.537 | 1.662 | +0.125 | [-0.057, +0.314] |
| `cut_vulnerability` | 268 | 268 | 0 | 0 | 0 | 113 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |

## Multiplicity accounting

* rows reported: 33 = 3 pre-registered primary + 30 exploratory.
* exploratory CIs excluding zero: **18** of 30 (`pip_sum`, `ore_pips`, `robber_robustness`, `exp_d2`, `exp_d3`, `exp_d4`, `exp_d2_pip`, `exp_d3_pip`, `exp_d4_pip`, `centrality_dist`, `pair_pip_sum`, `pair_ore_pips`, `pair_city_self_sufficient`, `pair_robber_robustness`, `road_2hop_pip_max`, `road_2hop_pip_sum`, `road_breadth_pip`, `nonshared_pip_sum`).
* expected to exclude zero by chance alone at alpha=0.05 if the policy and ThePhantom were identical on every axis: ~1.5.
* the CIs are **UNADJUSTED**. A 95% CI excluding zero is a rejected two-sided test; not printing a p-value removes the label, not the multiplicity. With this many rows, individual exploratory 'findings' are hypotheses for a fresh sample — the pre-registered family is the only one this run is entitled to interpret.
* **but the expected-false-positive count is the wrong frame for a COHERENT block of rows.** 18 of 30 exploratory CIs exclude zero, far above the ~1.5 chance expectation, and they are not scattered — 12 point UP (`pip_sum`, `ore_pips`, `robber_robustness`, `exp_d2`, `centrality_dist`, `pair_pip_sum`, `pair_ore_pips`, `pair_robber_robustness`, `road_2hop_pip_max`, `road_2hop_pip_sum`, `road_breadth_pip`, `nonshared_pip_sum`) and 6 point DOWN (`exp_d3`, `exp_d4`, `exp_d2_pip`, `exp_d3_pip`, `exp_d4_pip`, `pair_city_self_sufficient`), which on this battery reads as immediate production up against expansion room and city complementarity down. A per-row chance count says nothing about a directional block like that. The honest statement is therefore TWO-PART: the pre-registered contested-middle / denial family shows **no detectable difference**, while the exploratory battery shows a **coherent, unvalidated direction** — the policy buys immediate pips at the cost of expansion room and city complementarity. The second half is a hypothesis this run cannot confirm (exploratory, unadjusted, `pair_*` teacher-forced, human is not an oracle), but it is the strongest signal here on the owner's original question and must not be read as refuted by the primary null — the primary family measures different axes.

## Port-marginal churn (quantifies the D2 limitation)

| decision kind | decisions | mean churn | fraction with churn > 0 |
|---|---:|---:|---:|
| settlement | 268 | 0.018 | 0.086 |
| road | 268 | 0.010 | 0.045 |

## Descriptive only — exact-match agreement (NOT A GATE, D5)

| decision kind | decisions | mean agreement | mean candidates |
|---|---:|---:|---:|
| settlement | 268 | 0.301 | 48.2 |
| road | 268 | 0.513 | 3.0 |

## Exclusion ledger

| video_id | game | passed_crosscheck | order_source | reason |
|---|---:|---|---|---|
| `7Ft5ZxIm64Q` | 1 | True | `None` | placement_order_not_established |
| `IfgsXjBbtfQ` | 1 | True | `glyph_only` | replay_error: scripted opponent settlement 11 is ILLEGAL at placement #1 |
| `U4MIRMDbFeI` | 1 | True | `glyph_only` | replay_error: scripted opponent settlement 20 is ILLEGAL at placement #3 |
| `lCsB4X60YhQ` | 1 | True | `glyph_only` | replay_error: recorded settlement 23 is ILLEGAL at decision 2 |
| `AruQxi1X_PQ` | 1 | True | `glyph_only` | replay_error: recorded settlement 23 is ILLEGAL at decision 2 |
| `EE7OCzgz0Ws` | 1 | True | `glyph_only` | replay_error: scripted opponent settlement 22 is ILLEGAL at placement #1 |

**Replay exclusions by CAUSE** (5 order-established rows failed replay). `CorpusReplayError` is raised from several structurally different places, so the causes are reported as observed rather than assumed:

* `illegal_recorded_placement` — 5 row(s), 5 with `passed_crosscheck: true`: a RECORDED placement (the agent's own or the scripted opponent's) is geometrically illegal at the point it is replayed.

**Finding, not just bookkeeping:** 5 of those rows fail because a RECORDED placement is geometrically illegal (mutually adjacent settlements), and 5 of them carry `passed_crosscheck: true`. The corpus cross-check therefore does NOT validate mutual legality of the recorded placements — `passed_crosscheck` is a weaker guarantee than its name suggests. That matters for anything downstream (e.g. an outcome-anchored scoreboard) that reads the flag as 'this row is consistent'.


## Metric definitions (D3)

* **`contested_race_count` / `contested_race_pips`** — over the vertices that would still be legal settlement sites after my candidate is placed and that are within road-distance 4 of BOTH my settlements (candidate included) and the opponent's, the count of those I reach strictly first, and their pip weight. Undefined (`null`, excluded) when the opponent has no settlement yet.
* **`denial`** (PRE-REGISTERED, literal spec D3) — `|legal(occ)| - |legal(occ + v)|`: the reduction in the opponent's legal future-settlement set caused by my placement, **unrestricted**, exactly as spec D3 words it.
* **`denial_reach`** (EXPLORATORY sharpening — reported ALONGSIDE, not instead of, `denial`) — the same shrinkage restricted to the opponent's *road-reachable* set: `|legal(occ) ∩ reach| - |legal(occ + v) ∩ reach|`, `reach` = within road-distance 4 of his settlements. Why it is offered: the unrestricted `denial` is close to 'how many legal neighbours does the candidate have', i.e. largely board supply rather than a policy choice — the failure mode that killed the previous opening direction. **OWNER DECISION PENDING:** the spec is owner-delegated and `denial` is pre-registered, so this variant is flagged rather than substituted; if the owner prefers it, re-designating the primary row is a one-line change (`PRIMARY_METRICS`).
* **Both denial rows measure only the DISTANCE-RULE channel.** `denial` counts sites the candidate removes from the legal set via the distance rule, and `denial_reach` computes its `reach` from the PRE-candidate blocking set on both sides — so the other channel, my settlement WALLING the opponent's road expansion *through* that vertex (plausibly what 'denial of the opponent's expansion' means in 1v1), is **not counted by either row**. That channel is unmeasured by this instrument.
* **`cut_vulnerability`** — articulation points of the vertex-induced graph of MY OWN roads including the candidate. See the road section: it came out identically zero on every decision in this run (an EMPIRICAL null, with a legal counterexample available), and it does not model the opponent-road 'plow'.
