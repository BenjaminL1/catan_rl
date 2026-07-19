# Spec: pointer-arch fork — the one-shot policy architecture change (RATIFIED 2026-07-19, BINDING)

**Feature in one line.** One batched lineage fork that gives all three location heads
(corner/edge/tile) pointer readouts over the GNN's currently-discarded per-node states, adds
every honest missing obs signal, and reserves schema headroom — designed so this is the LAST
forced training restart for the foreseeable future.

**Provenance.** 2026-07-18 /dev-plan: 4 exploration lenses (obs, topology, aux/search,
capacity/prior-art) + expert council (Baseline Reviewer, Drift Sentinel; independent positions
+ one cross-examination round). Param audit: current net ≈1.376M, mis-allocated — 48.6% heads
reading one global 512-d bottleneck, 31.2% one fusion Linear, and the 4.8% GNN discards its
per-node outputs at the mean-pool (`src/catan_rl/policy/encoders.py:289`). The fork buys
STRUCTURE, not width — capacity-alone is the most-falsified hypothesis in this repo.

---

## Decisions (binding once ratified)

### D1 — Pointer readouts for ALL THREE location heads
`GraphEncoder.forward` additionally returns the per-node states `v (B,54,64)`, `e (B,72,64)`,
`h (B,19,64)` it already computes. Corner/edge/tile heads become per-node pointer readouts:
`logit_i = MLP([trunk_proj, node_i])`, with **separate readout MLPs per node type** (vertex ≠
edge ≠ hex semantics; council: lean separate). Trunk width (512) unchanged; head output sizes
(54/72/19) unchanged; the 6-head autoregressive structure, `MultiDiscrete([13,54,72,19,5,5])`,
masking semantics (`masked_log_softmax`, `_corner_mask` et al.) all unchanged. The corner
head's FiLM context modulates its `trunk_proj` before the per-node concat.

### D2 — Setup-context bit into the corner head
`CORNER_CONTEXT_DIM` 2→3: append `is_setup` to the `[settlement, city]` context so snake-draft
placement gets dedicated modulation. The flag must be threaded at inference (env state), not
only in training buffers. This is the one measured-blind-spot fix that hard-forks a tensor.

### D3 — Obs additions (all verified honest; leak-history audited)
1. **Own total-hand + discard-pressure** (2 scalars, current-player block): today per-resource
   clips saturate at `/8` below the 9-card discard cliff (`obs_encoder.py:479-480`) and no own
   total exists (`:607-608`). Own-private info — clean.
2. **Own played YoP/Mono/RB counts** (3 scalars, current-player block). Publicly revealed — clean.
3. **NEW single POV-neutral GLOBAL block** (not duplicated in the per-player pair; must survive
   seat swap unchanged — seat-swap invariance test required):
   - **Finite-bank remaining** (5 scalars, `bank[r]/19`). Public. This deliberately REVERSES the
     spec-009 "bank is engine state only — NOT in the obs" line, whose justification (no
     policy-shape change) evaporates in a fork whose purpose is a shape change. CLAUDE.md and
     spec-009 docs must be updated in the same change (ratified here, not smuggled).
   - **Dev-deck per-type remaining (5 scalars) — PUBLIC-REVEAL-DERIVED ONLY**: computed as
     initial composition − own cards (private-to-self, legitimate) − all publicly played/revealed.
     This equals "remaining in deck ∪ opponent hidden hand" — what an honest player knows. It
     must NEVER read engine deck ground truth (that partition is the reverted-leak pattern,
     cf. honest-obs fix 72dbb0d). A test must pin the derivation against a constructed state.
4. **Reserved strict-0.0 slots**: 8 per player block + 4 in the global block. Precedent:
   `vertex_features`/`edge_features` already carry spare dims. Because BC shards store
   FEATURIZED obs (`bc/dataset.py`), reserved slots are what keeps future shards/scalars
   shape-stable — a future scalar fills a slot mid-self-play with no fork and no BC-regen.
   A unit test enforces exact-0.0 fill of every reserved slot (a non-zero constant would train
   a bias and shift distribution when repurposed).
5. Engine access is **read-only accessors only** — no new mutation path near
   `bank_recirculate`/`bank_draw`; the bank conservation invariant is untouched (rule 2).

### D4 — Auxiliary value-target head (representation-shaping)
Small head off the trunk predicting the **already-stored discounted outcome `z_disc`**
(present in BC shards; the PPO rollout analogue is the existing GAE/return machinery), low
coef (~0.05), mirroring the belief-head plumbing (buffer field + coef guard). Framing is
binding: this shapes the TRUNK representation; it does not relitigate spec008's value-ceiling
verdict (which concerned root selection under a fixed value, not representation quality). It is
soft-additive (does not gate the fork) but is co-trained from update 0 for the trunk-shaping
benefit. `coef=0` must be byte-identical to no-head (guard test, pattern at `trainer.py:334`).

### D5 — What ships around the net
- **BC dataset regen** on the new schema (carries the queued AUG-1 edge-augmentation inversion
  + belief-target unification fixes — shards were stale anyway).
- **One-shot migration utility** (rule 3): loads any existing v2 checkpoint, transplants
  tile-encoder + GNN verbatim (schemas unchanged), zero-pads the new input columns of the
  player encoders/fusion, fresh-initializes the three pointer readouts + aux head. Must be
  test-executed against a real historical checkpoint (e.g. the bootstrap or v8 anchor) —
  load + forward must run. This keeps BOTH seeding paths (re-bootstrap / transplant) open.
- **Lineage note** documenting the new checkpoint lineage per rule 3.
- TB scalars: new names only, never renames (rule 4).

### D6 — GNN stays rounds=2, hidden=64 in this fork
Verified in cross-exam: the message-passing update is residual+LayerNorm, so a 3rd round can
be appended LATER with zero-init weights as a near-identity soft addition, mid-lineage, no
restart — "now-or-never" is false. Hidden-width 64→128 and mean→[mean,max] pooling are
excluded (capacity-shaped bets; the pointer heads reduce the pooled vector's load anyway).
A later rounds-3 experiment requires a pre-registered gate (WR + sims/s).

## Non-goals (each with its no-restart path, so exclusion costs nothing later)
- **Dice-bag obs** — EXCLUDED as a leak: noise swap + unsignalled reshuffle boundary make exact
  bag state non-derivable by a real Colonist player (both council experts concur).
- Setup-value / opening-prior head — output-side, additive-later; opening coverage is owned by
  the step6 corpus program (seeds + setup entropy), not this fork.
- Belief/deck/bag posterior heads for chance-node search — build the search first; heads are
  additive (belief-head precedent).
- Opp-action aux head — additive-later, low EV under perfect 1v1 hand tracking.
- Trunk widening, entity transformer, attention tile-flatten, value structured readout,
  recurrence/GRU (breaks search determinization), D6-equivariant encoder (augmentation
  suffices; equivariance is only exact on the GNN branch — see acceptance test 4).
- piKL/human-prior CE head — contradicts step6 §6 non-goals; corpus far too small.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (typecheck, lint, rust fmt, unit tests) per the dev-loop's exit-code gate.
2. Honest-derivation pin: dev-deck feature equals the public-reveal formula on a constructed
   state; no engine-truth deck read reachable from the encoder.
3. Global-block seat-swap invariance pin; reserved-slot strict-0.0 pin.
4. D6 pin: GNN-branch equivariance + pointer-head action-remap consistency under all 12
   elements (exactness claim is limited to the GNN branch — the trunk is position-aware).
5. Migration pin: historical checkpoint → migrate → forward runs; transplanted blocks
   byte-equal; new columns zero.
6. Schema single-source pin: every encoder in-dim derives from `obs_schema` constants
   (no hardcoded dims — CLAUDE.md obs rule).
7. **Inference throughput gate: CPU search sims/s within 10% of the v11 baseline** measured on
   the same machine/settings; a larger regression is a BLOCKER regardless of training metrics.
8. BC regen produces new-schema shards + BC training smoke passes.
9. Docs sync: CLAUDE.md spec-009 bank line, obs docs, lineage note — same change.

## Ratified decisions (owner, 2026-07-19)
- **Seeding: FULL RE-BOOTSTRAP** — BC regen → BC train → heuristic bootstrap → lowered-bar
  self-play. Clean priors for the fresh pointer heads; the regenerated BC data gets used.
  The D5 migration utility still ships (keeps the transplant path open as contingency).
- **Accept gate: DUAL GATE, pre-registered** — the new lineage is accepted only if BOTH:
  (a) h2h vs v11_cand Wilson-LB > 0.50 at n=600 (in-lineage non-regression), AND
  (b) the human-scoreboard opening metric ≥ v11's on the same eligible games (15 live at
  ratification, growing with the Intel sweep). One live playtest (opening/ore focus) is
  recorded as qualitative evidence, not a gate.
