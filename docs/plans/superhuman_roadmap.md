# Superhuman 1v1 Catan — North-Star Vision

**Status:** Vision only. The detailed, current implementation plan lives in
`docs/plans/v2/` (`design.md` + `step3_bc.md`, `step4_ppo.md`,
`step5_mcts.md`, `setup_strength_roadmap.md`). This file used to be the
1125-line master roadmap; it was superseded by the v2 plans and trimmed to
a one-page north star that does not contradict them.

## Goal

Train a **superhuman 1v1** Settlers of Catan agent under the Colonist.io
ruleset using custom PPO + league self-play, on a single Apple M1 Pro (MPS
for training, CPU for eval). **1v1 only** — never generalize to 4-player
(see `docs/1v1_rules.md` and ADR 0001 for the hard invariants).

## Success criteria (north star)

| # | Criterion | Threshold | How |
|---|-----------|-----------|-----|
| C1 | Beats the engine heuristic | ≥99% over ≥500 symmetrised games | `src/catan_rl/eval/harness.py` |
| C2 | Beats the prior best **v2** champion | ≥70%, both seat orderings | policy-vs-policy eval |
| C3 | Low exploitability | <5% WR for a fresh adversary | exploitability eval |

Pin win rates with Wilson intervals over ≥500 games, not n=100. The
heuristic-bootstrap champion plateaus ~0.66 symmetrised WR; do not expect
0.9 from the bootstrap alone — that is the cue to graduate to self-play.

## Phase arc (high level)

The path, ratified in `docs/plans/v2/` and the ADRs, is:

1. **Heuristic bootstrap** (BC → PPO vs fixed heuristic) — seeds
   `bootstrap_v1`. (`step3_bc.md`, `step4_ppo.md`)
2. **Setup strength** — analytic/learned settlement-placement priors.
   (`setup_strength_roadmap.md`, `setup_labeling.md`)
3. **League self-play** — snapshot-opponent driver is wired; PFSP /
   TrueSkill / Nash pruning / exploiters remain future work. (ADR 0008)
4. **Search & recurrence** — MCTS / piKL, belief & opp-action aux heads.
   (`step5_mcts.md`, ADR 0009)

## Hard constraints (never violate)

- The 1v1 Colonist.io ruleset (`docs/1v1_rules.md`, ADR 0001): 15 VP, 2
  players, no P2P trade, 9-card discard, Friendly Robber, `StackedDice`.
- **v2-only lineage.** No v1 (`catan/`) policies, checkpoints, or
  baselines — ever. v2 checkpoints must stay loadable across changes (one
  migration per state-dict change).
- TensorBoard scalar names are additive (never renamed).
- `arguments.py` is the config source of truth.

For everything concrete — file paths, code changes, ablations, decision
gates — defer to `docs/plans/v2/` and the ADRs under `docs/decisions/`.
