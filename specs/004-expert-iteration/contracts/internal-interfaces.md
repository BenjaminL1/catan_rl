# Phase 1 Contracts: Expert Iteration (internal interfaces)

Internal library + runner feature; contracts are the new module surfaces. All new code in
`src/catan_rl/expert_iteration/`; the only existing-code edit is the additive `train_bc`
warm-start param.

## C1 — Search-label generation

```text
expert_iteration/labeler.py
  generate_search_labels(cfg: SearchLabelConfig) -> dict   # manifest (shards + counts)
```
**Contract**: plays `SearchAgent(base_ckpt, sims_per_move)` vs `cfg.opponent` games,
seat-symmetrized + seeded; records each NON-forced agent decision as a BcDataset-compatible
row `(obs, search_action, mask, belief_target)`; fills `z_disc` from the discounted game
outcome; writes NPZ shards + `manifest.json` to `cfg.out_dir` until `n_positions` reached.
The recorded `action` is the search's chosen 6-tuple. Reuses `bc.dataset` shard format;
does NOT change the search or the engine. Reproducible at `cfg.seed`.

## C2 — Warm-started distillation (BC trainer hook)

```text
bc/train.py
  train_bc(*, ..., init_ckpt: Path | None = None) -> dict   # NEW param, default None
expert_iteration/distill.py
  distill(cfg: DistillConfig) -> Path   # distilled checkpoint path
```
**Contract**: `train_bc(init_ckpt=...)` loads that checkpoint's policy weights into the fresh
`CatanPolicy` BEFORE the loop (strict load; v2-lineage); `init_ckpt=None` is byte-identical to
today. `distill` runs `train_bc` warm-started from the round base on the labeled shards at the
fine-tune LR/epochs (D8) and returns a v2-lineage distilled checkpoint loadable by the existing
manager (FR-004).

## C3 — Pilot gate

```text
expert_iteration/gate.py
  run_gate(distilled_ckpt, v6_ckpt, *, n_quick=200, n_confirm=500, seed=0) -> GateResult
```
**Contract** (SC-001): SEARCH-FREE `evaluate_policy_vs_policy(distilled, v6)`, seat-symmetrized,
PASS iff Wilson lower bound > 0.50 at n≥200 then a disjoint n≥500; also reports WR vs the
heuristic + a prior-lineage rung (forgetting guard). Returns the verdict + CIs + failure mode.

## C4 — ExIt round (flywheel unit, US2)

```text
expert_iteration/round.py
  run_round(round_idx, base_ckpt, ...) -> RoundResult   # generate -> distill -> gate
```
**Contract**: one {label with `base_ckpt` → distill → gate} cycle; returns the distilled ckpt
+ gate + Elo delta; the orchestrator keeps the best distilled ckpt as the next round's base
and stops a round from seeding the next if it regressed (spec edge case).

## C5 — Pilot runner (CLI/script)

```text
scripts/run_exit_pilot.py
```
**Contract**: runs C1 (label) → C2 (distill) → C3 (gate) for the pilot, writing
`data/exit/round_0/` + `runs/exit/round_0/{distill,gate.json}`; detached-launch friendly
(mirrors `scripts/run_search_bakeoff.py`); CPU search/eval, MPS distill; no GUI import.

## Invariants (all surfaces)

- **Additive/inert**: importing/using `expert_iteration/` + the `train_bc(init_ckpt=None)`
  default leave existing BC/search/eval behavior byte-identical (FR-011, SC-005).
- **v2-lineage**: distilled checkpoints load via the existing manager; no state-dict shape
  change (FR-004).
- **Ruleset**: distilled policies + label games only ever play legal actions; zero
  rules-invariant violations (FR-010, SC-005).
- **Reproducibility**: C1 + C2 deterministic at fixed seed (FR-009, SC-006).
- **Search-free deployment**: the distilled agent plays via a single forward pass (no MCTS).
