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
  - BC train writes a BARE `{policy_state_dict, step, val_nll}` `best.pt` (no
    `schema_version`), which `load_checkpoint` / the PPO learner warm-start
    (`init_policy_checkpoint`) / `build_actor` refuse. Bridge it once with
    `scripts/bc_to_checkpoint.py --in runs/bc/<run>/best.pt` → a schema'd,
    policy-only `best_ckpt.pt` (reuses `save_policy_only`; mirrors the
    `expert_iteration/distill.py` re-save precedent; weights byte-equal).
  - The heuristic-bootstrap PPO stage runs from
    `configs/bootstrap_pointer_arch.yaml` (fixed-heuristic opponent, self-play
    OFF, `init_policy_checkpoint` = the bridged BC `best_ckpt.pt`, MPS). The
    pointer-arch is the default `CatanPolicy`, so no arch flags are needed.
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

### Clause (a) — SATISFIED (recorded 2026-08-02)

`selfplay_pointer_arch_v2/ckpt_000000500` (banked as `runs/anchors/ptr_v1_u500.pt`) vs
`runs/anchors/v11_cand_u724.pt`:

| | |
|---|---|
| Win rate | **0.7500** (450 / 600) |
| Wilson 95% CI | **[0.7138, 0.7830]** — LB 0.7138 > 0.50, PASS |
| Per-seat | seat0 0.7733 (n=300) · seat1 0.7267 (n=300); gap p≈0.19, noise |
| Truncated / rules violations | 0 / 0 |
| Engine tree | **`261098d190c8`** · board_geometry `70813dcf76fd` |
| Ruleset epoch | **R0** (pre-roll dev cards NOT available to either seat) |

Roughly +159 to +223 Elo over `v11_cand` (the CI, not the midpoint; a bilateral
estimate against one frozen opponent, **not** ladder-transitive).

**Why this is written here.** It was measured into
`runs/logs/xarch_u500_vs_v11_n600.log`, which is **gitignored** — so until now the only
record of the strongest result this project has produced lived in an untracked file, one
`rm -rf runs/` from gone. `.claude/veriloop/specs/bank-fix-slice-and-champion-bank.md`
D5 required this entry and it was never written.

**Read the engine-tree line as historical, not reproducible-by-checkout.** The
`human-path-conformance-fixes` slice edits `src/catan_rl/engine/player.py` and re-pins
`PINNED_ENGINE_TREE` off `261098d190c8`. That does not invalidate the number: the change
is confined to `player.play_devCard`, which has exactly two call sites — the legacy
`playCatan` loop the RL stack never enters, and the human GUI harness — while
`cross_arch_h2h` is bot-vs-bot through `_apply_main_action`, which is untouched. The
measurement is therefore unaffected in substance; what it loses is the one-command
equality check between this line and `git rev-parse HEAD:src/catan_rl/engine`.

**Clause (b) is still unbuilt.** `src/catan_rl/human_data/opening_scoreboard.py` does not
exist on `main` (it lives on the unmerged `feat/opening-scoreboard` branch), so the fork
is not accepted — clause (a) alone does not accept it, and banking on (a) alone would be
gate-shopping.

The AC-7 inference-throughput gate (CPU search sims/s within 10% of the v11
baseline, `scripts/bench_search_sims.py`) is a BLOCKER regardless of training
metrics; run the harness on the baseline branch and on this branch on the same
machine/settings and compare `sims_per_s`.
