# Phase 0 Research: Inference-Time Search

All decisions below are **already de-risked** by empirical probes (2026-06-14) + a senior-RL-game-dev audit + a 7-lens decision analysis. This consolidates them; no `NEEDS CLARIFICATION` remain. Probe scripts: `/tmp/{search_viability_probe,value_calib_peer,value_calibration,clone_fidelity}.py`. Scoreboard: `runs/elo_ladder_full.json`.

## D1 — Search algorithm

- **Decision**: Determinized **PUCT-MCTS** with the trained policy's priors and a value-head leaf; the **bake-off (US1) validates it at a minimal budget before any scaling**.
- **Rationale**: PUCT focuses the limited budget (~120 sims/sec) via priors — better than bare expectimax which wastes budget on equally-weighted children. No rollouts (bootstrap on the value head, AlphaZero-style) — the engine is too slow for deep rollouts and the value head is a validated leaf (D4).
- **Alternatives considered**: (a) **depth-limited expectimax** — simpler, the audit's fallback if PUCT underperforms at the gate; kept as the documented pivot. (b) **Full AlphaZero rollouts** — rejected (engine too slow; value head suffices). (c) **piKL-Hedge search** — deferred (a later refinement that anchors search to priors; presupposes the MCTS here).

## D2 — Stochasticity (dice / dev draws / robber)

- **Decision**: **Determinized sampling** — clone the env per simulation; the clone's `StackedDice` bag fixes that line's dice future; aggregate value over determinizations. **No opponent-hand belief sampling.**
- **Rationale**: Perfect 1v1 hand-tracking (`BroadcastHandTracker`, no P2P trade) means the opponent's hand is known → imperfect-info search collapses to cheap determinized search. Verified: a cloned env carries an independent dice bag (rolling a clone doesn't consume the original's).
- **Alternatives**: explicit chance nodes (more faithful, more complex) — deferred; start with 1 determinization per descent, resampled, and **measure determinization variance in the prototype** before adding more.

## D3 — Autoregressive action-space expansion

- **Decision**: **Progressive widening on the 13-way TYPE head** (the real branching factor, ~2-6 legal mid-game); for a chosen type, fill corner/edge/tile/res1/res2 by sampling the **conditional** FiLM-head priors.
- **Rationale**: The joint `MultiDiscrete([13,54,72,19,5,5])` is huge, but the type head gates it; sub-choices are conditional and cheap. Avoids combinatorial blow-up at a node.
- **Alternatives**: full joint enumeration (intractable, rejected); flat top-K joint sampling (loses the structure, weaker priors).

## D4 — Leaf evaluation (THE gating component)

- **Decision**: Leaf value = **squashed value head**, `V_squashed = sigmoid(3.22·V − 1.14)` ∈ (0,1); back this up in PUCT, **not raw V**.
- **Rationale**: Probed — raw V ranks states well (Spearman 0.69 peer-game) AND the Platt-fitted logistic calibrates it (Brier 0.149 vs 0.243 base, ECE 0.039). **But raw V spans [−1.6, +1.8] (27% outside [−1,1])**, which breaks PUCT's exploration-constant balance → must squash to a bounded win-prob.
- **Alternatives**: raw V (rejected — unbounded); a learned/retrained value (out of scope — no retraining); **bounded rollout to a more-terminal state** (late-game V is better calibrated, corr 0.86) before evaluating — kept as the **pivot if the bake-off shows off-distribution V is weak**.
- **Open (retired by the bake-off, not pre-fixable)**: off-distribution calibration — V is validated only on on-policy states; search visits off-manifold states. US1 is the explicit gate.

## D5 — Engine interface + throughput

- **Decision**: `copy.deepcopy` the **whole `CatanEnv`** per node/determinization (validated faithful + independent). **Rust engine NOT used.** Throughput lever (if more sims needed) = **batched leaf evaluation**.
- **Rationale**: Probed — deepcopy(env) 1.8ms, step 0.4ms, but the **NN forward 6.7ms dominates** (~120 sims/sec) → search is NN-bound, not engine-bound. A free engine saves ~30%; batching the value forward over many leaves saves multiples.
- **Alternatives**: Rust engine first (rejected — wrong bottleneck, big build); `__deepcopy__` of `catanGame` only (insufficient — need the full env incl. hand-tracker + masks).

## D6 — Scope + the env-access problem

- **Decision**: **Offline-only.** A **new search-aware play/eval loop** hands the search agent the **live env** to clone+simulate (the existing harness passes only `(obs, masks)`, which is lossy — search cannot reconstruct the game from obs).
- **Rationale**: The single non-obvious integration fact: `evaluate_policy_vs_policy`/`EvalHarness` call `policy.sample(obs, masks)`; search needs the env. So a new loop is required (the `catan_rl/search/` module + a search eval entry).
- **Alternatives**: in-the-loop expert iteration / search-as-league-teacher (deferred — the highest-value *next* phase, but presupposes a working offline search).

## D7 — Measurement + the go/no-go gate

- **Decision**: Uplift via `evaluate_policy_vs_policy(search_agent, raw_ckpt)` (seat-symmetrized, Wilson CI) + a rung on the existing Elo ladder. **Bake-off gate (task #1): Wilson LB > 0.50 at n≥200 then ≥500.** Strength-scaling: a 0.25/1/5s time-budget ladder must show monotone WR.
- **Rationale**: Reuses the wired, RNG-safe, seat-symmetrized harness; the ladder (`runs/elo_ladder_full.json`) is the locked baseline (raw v6 ≈ 1100 Elo). The time-budget ladder is the cheap guard that the search is real lookahead, not a noise-adding bug.
- **Alternatives**: vs-heuristic eval (rejected — saturated at 0.93+, no resolution); n=100 (rejected — too noisy per the project's ≥500-for-promotion rule).

## Baseline context (from the Elo ladder)

Raw v6 frontier ≈ **1100 Elo** (heuristic pinned 500; v5 ~860; bootstrap ~490; random −307). The 5 v6 snapshots are within ~27 Elo (plateau). Search uplift will be reported as Elo over this raw-v6 rung; a 0.60 WR vs raw v6 ≈ **+70 Elo**.
