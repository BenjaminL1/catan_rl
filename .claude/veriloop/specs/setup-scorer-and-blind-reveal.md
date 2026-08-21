# Spec: setup-scorer-and-blind-reveal — the owner's opening theory as a fitted scorer, verified blind-first

**Status: RATIFIED (2026-08-21) — BINDING. Owner ratified after risk discussion. Amended after premise review (2026-08-20); the
rider's pre-mortem and opposite-case are carried as open RISKS below.**

**Feature in one line.** Productionize the interpretable setup scorer (upgraded with the owner's
port and denial theory, settlements AND roads), refit it on all labels, and add a blind-then-reveal
mode to the labeling tool — with the measurement design (self-consistency first, paired
per-position gate, kill bar, anchoring control, and a forced-opening win-rate probe) that makes
the numbers mean something.

## Provenance

2026-08-15 pilot (212 labels): every neural fine-tune arm memorized (train 96–100%, held-out
15.9–22.7%, negative transfer) while a 10-feature linear scorer scored 34.1% held-out top-1 vs a
30.5% bar. Direction adopted: scorer-as-teacher. Owner theory captured verbatim 2026-08-20
(memory `reference_owner_opening_theory`): the port penalty is a confound (1–2-hex corners →
less production, less starting material on the second settlement; real port value is conditional
on trade fit and "building to" a missing resource, usually sheep); denial is relational (take the
opponent's best spot when it starves a scarce needed resource, or kill it from 1-road-adjacent;
expansion direction matters). Premise review verified: u500 already matches the owner 55%/91%
(top-1/top-3) on pick 1 — the owner's edge is picks 2–4 (0–18%), so the gate design must see
positions, not aggregates.

## Decisions (binding)

### D0 — Self-consistency replay runs FIRST (needs no scorer)
Before any scorer work gates anything: `--replay-session <id>` re-presents a past session's exact
boards blind (prior picks hidden), rows written as normal labels linked by `replay_of`. The owner
relabels ~15–20 boards (~20 min). The owner-vs-owner agreement number is the LABELER-NOISE
CEILING every later number is read against — if it is low, the bar structure is recalibrated
before anything else is trusted. Replay rows are EXCLUDED from scorer fitting (dedup by
`replay_of`), included only in consistency reporting.

### D1 — Scorer module: theory-shaped features, settlements AND roads
`src/catan_rl/setup_phase/scorer.py`. Settlement features per candidate vertex: per-resource pips
(5), total pips, distinct resources, new-resources-for-me, n_adjacent_hexes, n_hexes ×
is_second_settlement (starting-material term), port flag, 2:1-port × own matching production,
3:1-port flag, expansion value (best legally-settleable vertex within road distance 1–2,
own-yield-scored), opponent value of the candidate (their pips + their missing resources),
opponent-best-spot margin, adjacency-block (distance-1 from opponent's best remaining vertex),
scarcity-starve (covers a board-scarce resource the opponent lacks). **Road model is FIT, not
asserted**: a second masked-softmax over legal edges (features: value of the best vertex the edge
opens at distance 1–2, blocks-opponent-target flag, toward-port flag) trained on the owner's 212
road labels — the "point at the expansion target" rule becomes the null hypothesis it must beat.
Exact arithmetic pinned by hand-computed fixtures; `FEATURE_VERSION` stamped.

### D2 — Fit script + weights artifact
`scripts/fit_setup_scorer.py`: masked-softmax likelihood over legal vertices + legal edges, light
L2, fit on all non-replay labels in the store. Output `data/setup_phase/scorer_weights_v1.json`:
weights + provenance (n_labels, store path, git_sha, FEATURE_VERSION, fit date, per-position fit
metrics, settlement AND road agreement). Deterministic given store + seed. The pilot scorer's
scratchpad script is committed into the repo as part of this slice (provenance, not authority).

### D3 — Blind-then-reveal mode, with an anchoring control
After the owner SUBMITS (row durably written), an overlay shows the scorer's top-1 settlement
(+top-3 dimmed) and its road pick beside the owner's. Reveal NEVER precedes submit; skip shows no
reveal; undo disabled after reveal. Additive row fields (store schema v3, read-time defaults):
`scorer_version`, `scorer_top1`, `scorer_rank_of_pick`, `agree`, `reveal_mode`. **Anchoring
control:** sessions carry a `--no-reveal` flag and ≥20% of all fresh exam picks must come from
no-reveal sessions; the gate report compares reveal vs no-reveal agreement — divergence = the
reveals are training the owner, and only no-reveal picks count for the gate until understood.
The scorer never writes or suggests a pick.

### D4 — Pre-registered forward exam: PAIRED, per-position, with a kill bar
The rolling exam = fresh blind-first picks created after the scorer ships (`scorer_version`
stamps which scorer was live; refits legitimate per D6). Gate to unlock synthetic generation, on
≥150 fresh picks: **paired comparison** — scorer vs u500 evaluated on the IDENTICAL picks, gate =
paired agreement difference with CI (the u500-baseline evaluator is extended to run on arbitrary
label subsets, not shard manifests), reported **per draft position** with the relational features'
fitted weights published at each refit. PASS requires the paired difference CI lower bound > 0
overall AND scorer > u500 on the picks-2–4 subset (point estimate, CI reported). **Kill bar:** if
after 300 cumulative fresh picks the scorer's picks-2–4 agreement has not exceeded u500's, the
theory-feature approach is declared DEAD and the program re-plans — no unbounded iteration.

### D5 — Forced-opening win-rate probe (machine-time, parallel arm, required)
Before synthetic generation unlocks, run the ExIt-STEP-2-shaped probe: paired seeds, u500 plays
both sides, one arm's openings FORCED to the scorer's picks, the other to u500's own; report
ΔWR with CI. Pre-registered reading: ΔWR>0 validates the opening's win-value through this
midgame; ΔWR≈0 is AMBIGUOUS (u500's midgame may be unable to exploit better openings — recorded
as such, not spun) but a large negative kills synthetic generation. Machine-time only.

### D6 — Refit cadence
Refit on blind-first labels is legitimate (blind-first + the D3 control keep the grader
uncoupled); each refit bumps the artifact version; agreement is always reported against the
scorer version live at label time.

### D7 — Vehicle neutrality
The scorer artifact is a pure function (board + prior picks → vertex/edge scores) consumable by
EITHER downstream vehicle: synthetic-corpus fine-tune (current plan) or setup-node search priors
(the banked +55 Elo search lever; multi-rep prior infra exists). Nothing in this slice may bake
in the fine-tune assumption. Vehicle choice is a post-gate owner decision, informed by D5.

## Non-goals
- NO synthetic-corpus generation, NO network fine-tune, NO self-play changes (next slice, gated on D4+D5).
- No engine changes; `PINNED_ENGINE_TREE` untouched. No mutation of existing label rows.
- The scorer never replaces the network at play time — teacher/grader/prior-source only.

## Acceptance criteria (the /dev-loop gate is the authority)
1. Full gate green on real exit codes; TDD per repo convention.
2. Scorer features pinned by hand-computed fixtures; FEATURE_VERSION stamped; road model fit and
   reported, not asserted.
3. Regression continuity: the D2 fit, restricted to the pilot's 168-label split and 10 pilot
   features, reproduces the pilot fit; the full-feature fit on the same split scores WITHIN THE
   WILSON CI of 34.1% on the spent 44-pick set (soft floor — one pick = 2.3pp there; the spent
   set never gates anything again).
4. Blind-first invariants pinned: no reveal before submit; no reveal on skip; no undo after
   reveal; no-reveal sessions carry no scorer fields; v1/v2 rows load unchanged.
5. D0 replay produces linked rows + a self-agreement report on a fixture session; replay rows
   provably excluded from fit.
6. The paired per-position gate evaluator and the D5 probe driver exist with tests (fixture-level;
   the real runs happen at ≥150 picks).
7. Docs sync per CLAUDE.md §6.

## RISKS — open, NOT cleared (premise-rider, carried in substance)

**Pre-mortem headline: agreement all the way down.** The chain labels → scorer(agreement) →
synthetic corpus(agreement) → fine-tune(agreement gate) → only then WR can succeed on every
stated criterion and deliver nothing, because its currency is agreement until the last link — and
the fine-tune vehicle is 0-for-2 in this repo (ptr_r1_u300 D9 tie; pilot negative transfer),
while the self-play tail carries a recorded do-not-fund blocker (three ~0.51 stalls, zero
promotions). D5 (the WR probe) and D7 (vehicle neutrality, with search-priors as the alternate
vehicle) are this spec's answer; they are mitigations, not proof. Sub-threads the rider grounded:
the adopting evidence was a sub-noise pass (scorer 34.1% vs a bar whose own CI reached 34.5);
static per-candidate features may again miss draft-sequential picks-2–4 logic even with the new
relational features (D4's per-position kill bar exists to catch this, not prevent it); reveal
anchoring can re-couple grader and gradee (D3's control detects, does not prevent); D0's noise
ceiling may come back low enough (~40%) to force a full bar recalibration.

**Opposite case (rider judged the amended plan defensible, the unamended plan not):** the direct
instrument — forced-opening ΔWR — could run TODAY without any scorer, and a null there moots the
whole program; it is now D5, but note its null is ambiguous through a midgame that may not
exploit openings (v8-grades-own-homework, again). And the deployment vehicle with the best track
record is search priors, not fine-tune — D7 keeps that door open; the owner should expect to
choose it if D5 is positive and the fine-tune path stalls again.
