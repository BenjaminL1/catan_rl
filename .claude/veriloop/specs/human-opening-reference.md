# Spec: human-opening-reference — score the policy against ThePhantom on HIS boards

**Status: OWNER-DELEGATED DRAFT (2026-07-25) — written under explicit instruction to proceed
overnight; NOT owner-ratified. Review before it drives anything further.**

**Feature in one line.** Replay ThePhantom's 140 parsed games on their real hex layouts, ask the
champion policy what it would place at each of *his* decision points, and score both choices with the
existing opening battery **plus** new territory/denial metrics — producing a **paired policy-vs-human
metric gap on identical positions**.

## Why this, and why the previous direction died

A 3200-decision opening sweep plus a 4-member expert council **killed** the "opening resource-composition
blind spot" thesis three independent ways, all verified:
1. **The pre-registered gate had already fired CLOSE.** `docs/plans/opening_deficit_verdict.md:63-66`
   set ore-substitution ≥40% ⇒ systematic, <40% ⇒ "the whole opening thread closes." Measured
   **0.2662 [0.2452, 0.2884]**.
2. **The flagship metric measured board supply, not policy choice.** "75.6% below random on
   `pair_max_ore_lump`" is arithmetically identical to "a strictly-higher-lump alternative existed"
   (605/800); the metric is 3-valued {0,1,2} = 67/694/39.
3. **"Below random" was a median-vs-mean reading error.** Mid-rank percentile has E[pct]=50 (a MEAN
   property); on tied discrete metrics a uniform chooser's MEDIAN sits far below 50. Simulated on
   identical candidate sets: `pair_max_ore_lump` random median 46.5 vs policy 46.7; `ore_pips` random
   38.4 vs policy 41.9; policy MEANS 54.5 / 61.6 vs random 50.1 / 49.9. Everything called "below
   random" is at or **above** random.
Also verified: the policy **does** condition on what it holds (settlement #1 without ore ⇒ 2nd-settlement
ore percentile 83.7 / P(touch)=0.672; with ore ⇒ 40.2 / 0.391).

What survives is a **reference-class problem**: every judgement this project has made is against a
baseline that is unreadable ("vs random"), mirror-contaminated ("vs v11", which shares the biases) or
saturated ("vs heuristic" ≈1.00). Meanwhile the axes the owner actually named — control of the
contested middle, denial of the opponent's expansion, road-cut vulnerability — have **zero metrics**;
the sweep's own limitations 7 and 8 (`docs/plans/opening_sweep_results.md:459-460`) explicitly disclaim
jurisdiction ("Neither measures control of the middle relative to the opponent's settlements").
This feature supplies the missing axis **and** the only non-chance baseline the project owns.

## Decisions (binding for the build)

### D1 — Paired, same-position comparison (not a distribution comparison)
For each corpus game: inject the real hex layout, replay the real draft order, script the *opponent's*
placements from the corpus, and at each of **ThePhantom's** decision points record what the policy
would choose. Both choices are then scored by the same battery on the same board state. This yields a
**paired metric gap**, which is what makes it readable without any "random" baseline — the exact error
that killed the previous direction.

### D2 — Ports: K=8 port-marginal, do NOT harvest
All 140 rows carry `board.ports = "OMITTED in v1"`, but ports **are** in the obs (per-vertex 7-way
one-hot, `obs_encoder.py:155`), so a naive injection makes the policy choose under fictional port
information. Harvesting 9 slot→type labels per board is a video-parsing job and is **out of scope**.
Instead marginalise: port *slots* are fixed geometry, so sample K=8 port assignments per board
(deterministic per-board-per-k sha256 seed), have the policy choose under each, and aggregate. This is
the **already-ratified step6 §4 convention** for this exact gap. State the marginalisation in every
output; it is a real limitation, not a fix.

**Implementation note (2026-07-26, `port-harvest`).** The "do NOT harvest" clause has been
superseded *only for the 34 boards whose pixels are still on disk*: `port-harvest` (D6) reads
their 9 slots into the sidecar `data/human/ports/harvest.jsonl` (32 of 34 boards decoded). D2 is
otherwise **unchanged** — the harvest never writes `board.ports` (port-harvest D5), and **K=8
remains the scoreboard / measurement convention** (port-harvest non-goals). The ~100 pixel-less
rows are still marginalised and are only re-ingested if the D7 invariance probe
(`scripts/port_invariance_probe.py`) shows ports actually move the opening choice.

First measurement, on `runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt` —
update 500 of the **superseded** pointer-arch v2 self-play run; it is `opening_sweep.DEFAULT_CKPT`,
NOT a banked anchor, not the champion (`runs/anchors/v8_promobar_u243.pt`) and not the in-progress
v3 lineage. The **mandated real-vs-guessed** realised-pair flip rate is **0.0938**, 95% CI
**[0.0417, 0.1542]** (settlement 1 0.0479, settlement 2 0.0938). The interval is a **cluster
bootstrap over the 60 (board, seat) cells**, not over the 480 comparisons: each cell contributes
8 comparisons that all share the same real pair, so the effective n is ~60, not 480. Only
**12 of 60** cells flip at all. The label-free guessed-vs-guessed supplement is **0.0932** (CI
[0.0446, 0.1484]) — statistically indistinguishable from the real leg, i.e. the hand-labelled
real map behaved like just another guess and added no information to this decision.

**This does not settle the re-ingest, and the probe no longer pretends it does.** An earlier
version of `recommendation()` compared the point estimate against a `0.05` cut invented in code
(port-harvest D7 pre-registers no tolerance at all — "no port-accuracy tolerance exists") and
emitted "RE-INGEST IS JUSTIFIED"; that verdict flips to its opposite inside its own confidence
interval, so it has been removed in favour of a plain measured statement. Two further caveats,
neither cleared: stepping the env drives the **heuristic opponent**, which also reads ports, so
this bounds the *policy's own* port sensitivity from above; and `port-harvest` §RISKS' pre-mortem
(a more precise instrument nobody acts on) is untouched by a flip rate. The standing decision is
unchanged — the re-ingest is **not** authorised by this note; the call is the owner's, on
evidence that does not resolve it.

### D3 — New metrics (the owner's stated axes — currently unmeasured)
Add to the existing battery, defined precisely in the module docstring:
- **`contested_race`** — for each contested vertex (legal, unowned, reachable by both), road-distance
  from mine minus from the opponent's nearest settlement; summarise as count/pip-weight of vertices I
  reach strictly first.
- **`denial`** — reduction in the *opponent's* legal future-settlement set caused by my placement.
- **`cut_vulnerability`** — articulation points in my own road network (the "plow" the owner described:
  a single opponent road that disconnects my longest path).
These are hypotheses about what the owner reads, not established quantities. Report them as such.

### D4 — Scope: a REFUTATION instrument
Human ≠ optimal, so a divergence is **unsigned**. This tool can **refute** the "openings are weak"
thesis (policy matches or exceeds ThePhantom on his own axes ⇒ the thread closes) but can **never
confirm** superiority, and it measures **no win-probability**. Say this in the report header. It is the
same trap the ExIt closure already named (overrides fired at 7.3% and were individually WIN-NEUTRAL).

### D5 — Metric-gap primary; agreement-rate descriptive only
step6 §6 lists move-agreement metrics as a **non-goal**. The deliverable is the metric GAP; an
exact-match agreement rate may be reported as descriptive colour but is **not** a gate and must be
labelled as such.

### D6 — Honesty constraints carried from the corpus
`opponent_strength` is a hardcoded constant across all 140 rows (`scripts/vlm_spike.py`) — **zero bits**:
never stratify, weight, or cite it as validation. 139/140 rows have `placement_order_established`; skip
the one that does not rather than guessing. Corpus census to reproduce: 140 rows, 118 videos, 135
unique layouts, 117 winners known.

### D7 — Read-only w.r.t. training
`selfplay_pointer_arch_v3` is training on the M1. CPU only (`--device cpu`), `nice -n 19`, ≤4 workers,
never write under `runs/train/**`, read only numbered/immutable checkpoints. Champion under test:
`runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt`.

## Implementation seams (verified during the council)
- **Engine injection EXISTS**: `catanBoard.inject_hex_layout(resources, numbers, robber_hex)`
  (`src/catan_rl/engine/board.py:469-518`) — post-construction overwrite, geometry untouched.
- **Env seam is ABSENT** and must be added additively: `CatanEnv.reset` hard-builds `catanGame` at
  `src/catan_rl/env/catan_env.py:344` with `ObsEncoder`/`_build_index_maps` right after (`:374-383`).
  Injection lands between them via a new `options["board_layout"]` (~10 additive lines). Default path
  must stay byte-identical.
- **Battery is board-pure**: `BoardScorer` reads only `board.boardGraph`/`hexTileDict`
  (`scripts/opening_sweep.py:313-487`) — scores an injected board unchanged.
- **Index compatibility is free**: corpus ids are engine vertex 0..53 / edge 0..71.
- **Scripted opponent fits**: `set_snapshot_opponent` (`catan_env.py:833`) duck-types on
  `.sample/.device/.reset_rng` (`:914-917`).
- Port assignment welds via `updatePorts(port_assignment=...)` (`board.py:368-390`).

## Non-goals
- No port harvesting from video. No win-probability / counterfactual rollout. No architecture change.
- No training-config change, no engine rule change, no obs-schema change; `glyph_anchor.py` untouched.
- No claim of superiority over ThePhantom from this instrument.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `cargo fmt --all -- --check` · `make test-unit`).
2. **Injection fidelity pin**: a board injected from a corpus row reproduces that row's hex
   resources/numbers exactly, and the default (non-injected) reset path is byte-identical.
3. **Index-mapping pin**: corpus vertex/edge ids map to the same engine entities the topology export uses.
4. **Determinism pin**: same seed ⇒ identical policy choices and identical metric output, bit-for-bit;
   plus a different-seed negative control.
5. **Port-marginal pin**: K=8 assignments are deterministic per (row, k) and the report states the
   marginalisation.
6. **Pairing pin**: the policy's scored decision is taken at the *same* board state ThePhantom faced
   (same prior placements), verified on at least one hand-checked row.
7. Skips the 1 row lacking `placement_order_established`; never guesses order.
8. Read-only pins: writes only under `data/human/**` or `runs/analysis/**`, never `runs/train/**`.
9. Report carries the D4 refutation-scope header, the D2 port limitation, and the D6 constant-
   `opponent_strength` warning.

## Follow-on (explicitly NOT this build)
The cross-examination ranked **aggregate midgame instrumentation** above any further opening metric:
cities-built, bank-trade composition, and VP-rate over the existing self-play/eval corpora — n in the
thousands, zero new games, no baseline problem. The owner's single game showed 0 cities in 58 turns and
6 of 12 bank trades buying ore, but that is n=1; the aggregate version is the honest instrument.
