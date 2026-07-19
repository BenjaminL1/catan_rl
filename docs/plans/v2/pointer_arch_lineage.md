# Pointer-arch fork — checkpoint lineage note

Spec: `.claude/veriloop/specs/pointer-arch-fork.md` (RATIFIED 2026-07-19).

This fork changes the `CatanPolicy` state-dict shape (D1 pointer readouts for
the corner/edge/tile heads; D2 `CORNER_CONTEXT_DIM` 2→3; D3 obs additions →
wider player encoders + fusion; D4 aux value head). Per rule 3 (checkpoint
compatibility is within v2, a shape change needs a one-shot migration +
documented v2 lineage), this note records the new lineage and both seeding
paths.

## New architecture summary

- **Trunk width unchanged** (512); **head output sizes unchanged** (54/72/19);
  `MultiDiscrete([13,54,72,19,5,5])` and masking semantics unchanged.
- **GraphEncoder** now returns its per-node states `v (B,54,64)`, `e (B,72,64)`,
  `h (B,19,64)` alongside the pooled trunk contribution (topology byte-unchanged,
  rounds=2/hidden=64/mean pooling — D6).
- **Location heads** are per-node pointer readouts `logit_i = MLP([trunk_proj,
  node_i])` with separate MLPs per node type; the corner head FiLM-modulates its
  `trunk_proj` with `[settlement, city, is_setup]` before the per-node concat.
- **Obs additions** (all honest): current-player own hand-total + discard-pressure
  + own played YoP/Mono/RB; a POV-neutral `global_features` block (bank remaining
  + public-reveal-derived dev-deck remaining + reserved); `is_setup`; reserved
  strict-0.0 headroom slots per player block.
- **Aux value head** off the trunk predicting the discounted return (`z_disc`
  analogue); `aux_value_coef=0.05` default, guarded byte-neutral at coef=0.

Param count ≈ 1.38M (was ≈ 1.376M) — the fork buys STRUCTURE, not width.

## Seeding paths

- **PRIMARY — full re-bootstrap (ratified).** BC regen (new schema) → BC train →
  heuristic bootstrap → lowered-bar self-play. Clean priors for the fresh pointer
  heads; the regenerated BC data is used. This is the seeding path that produces
  the accepted lineage.
- **CONTINGENCY — transplant (kept open).** `scripts/migrate_pointer_arch.py`
  (`catan_rl.checkpoint.pointer_arch_migration`) loads any legacy v2 checkpoint
  and transplants the tile-encoder + GNN verbatim, zero-pads the new player-
  encoder / fusion input columns, and fresh-initialises the pointer readouts +
  aux head. Verified against `runs/anchors/v11_cand_u724.pt` (123 blocks
  transplanted byte-equal, 3 zero-padded, 23 fresh-init; forward runs). The
  optimizer state is dropped on transplant (a shape change restarts the
  optimizer).

## Accept gate (dual, pre-registered)

The new lineage is accepted only if BOTH hold:
- (a) h2h vs `v11_cand` Wilson-LB > 0.50 at n=600 (in-lineage non-regression); AND
- (b) the human-scoreboard opening metric ≥ v11's on the same eligible games.

The AC-7 inference-throughput gate (CPU search sims/s within 10% of the v11
baseline, `scripts/bench_search_sims.py`) is a BLOCKER regardless of training
metrics; run the harness on the baseline branch and on this branch on the same
machine/settings and compare `sims_per_s`.
