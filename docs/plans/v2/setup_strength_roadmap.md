# Setup-Phase Strength Roadmap — Analytic Yield, Port-Aware D6, MCTS Decision

**Status**: design draft (single author, senior-review distilled 2026-06-03); implementation gated on the §0 preflight gates. Plan layered on top of v2 Step 3 BC + v2 setup-labeling deliverables.

**Provenance**: senior RL research-engineer review of the (removed) v1 Monte-Carlo setup-pretrain plan (v1 MC ancestor) and `docs/plans/v2_setup_labeling.md` (human-label fine-tune). The review's central claim: the v1 MC rollout estimator for setup-phase vertex value is *strictly dominated* by a closed-form expected-pip-yield score now that the chip layout is the verified 18-chip ABC spiral (`src/catan_rl/engine/board.py:34-44`), and the dihedral-aug pipeline silently leaks port information in BC training because `symmetry_tables.py` does not export a port permutation. Phase C is a meta-redirect surfacing the senior's "you may be solving the wrong problem" critique.

---

## 0. Preflight gates (block phase kickoff)

Three checks, run before Phase A starts. Failure routes to the diagnosis ladder rather than continuing.

### 0.1 Chip-layout determinism — **PASSED 2026-06-03**

**Refined claim** (the original "per-hex chip assignment is invariant under board reseeding" wording was overly strong and caught during test authoring). The actual invariant the Phase A analytic scorer needs is:

> Given `(spiral_orientation, desert_hex)`, the per-hex chip assignment is deterministic.

The desert dependence matters because the walker **skips** the desert hex and the chip sequence shifts past it. Two boards that share orientation but place desert at different hexes will have *different* chips at positions past the earliest desert; this is correct walker behavior, not a violation.

What Phase A actually needs: the scorer reads `board.hexTileDict[h].number_token` directly, so per-board chip determinism (a strictly weaker property than per-orientation) is sufficient. The four invariants pinned:
1. `_build_spiral_path(corner, cw)` is RNG-independent.
2. `_walk_chips(path, desert_hex)` is a pure function.
3. Two boards sharing `(orientation, desert_hex)` produce identical per-hex chips.
4. Skipping desert preserves `SPIRAL_CHIP_SEQUENCE` exactly (no permutation).

Pinned in `tests/unit/setup_phase/test_chip_layout_invariance.py` (210 tests, all green).

### 0.2 Champion checkpoint loads under current code — **DEFERRED 2026-06-03**

**Outcome**: no usable champion checkpoint exists.
- `catan_rl_v2/checkpoints/` does not exist.
- `catan_rl/checkpoints/train/` contains only `checkpoint_16162816.pt` (16.16M steps); the `checkpoint_07390040.pt` cited in v1 CLAUDE.md is gone.
- v2 BC pipeline (`scripts/train_bc.py`) is untracked — no v2 BC checkpoint either.

**Decision** (recorded in `MEMORY.md → project_setup_strength_calibration.md`): defer champion-derived calibration. **Phase A.2 ships dual tables**:
- **Table 1**: Charlesworth's published prior (1.0, 1.0, 1.0, 1.1, 0.7) — principled hand prior.
- **Table 2**: heuristic-derived — fit NNLS against v2 heuristic-vs-heuristic end-game resource shares (no model needed).

Both are pickable via the §A.4 ablation. Champion-derived option re-evaluated when v2 PPO produces a competitive checkpoint.

### 0.3 Port-permutation gap audit — **PASSED 2026-06-03**

Confirmed: zero `port_perm` references in `src/catan_rl/augmentation/symmetry_tables.py` or `src/catan_rl/augmentation/dihedral.py` (only match is the word "ported" in a comment). Reviewer's diagnosis holds. Phase B implementation proceeds as scoped.

---

## Phase A — Analytic value rollouts in the setup trainer

**Problem.** The setup-phase trainer (v1 location: the (removed) v1 Monte-Carlo setup-pretrain plan §3, `mc_rollout_collector.py`; v2 location: NOT YET PORTED — this plan ports it) estimates vertex value via Monte Carlo dice rollouts at `n_rollouts=50` per position (the v1 default; the senior cited 20 in review — both are too noisy). With the chip layout deterministic, the first-moment expected pip yield has a closed form and dice variance is unnecessary supervision noise. Reviewer quote: *"the resource weight table is the whole ballgame — get it wrong and you bake a systematic placement bias into every game."*

**Target.** Replace MC with `value(v) = Σ_h ∈ adjacent(v) dots(chip[h]) × resource_weight(res[h])` where `dots(·)` maps the 2..12 token to its pip count (2→1, 3→2, …, 6→5, 7→0, 8→5, …, 12→1) and `resource_weight: dict[str, float]` is a **calibrated** scalar per resource type. Calibration is a real design subtask, not a constant.

**Expected impact.** +0.5 to +1.5% WR vs current self-play champion; setup-training step wall-time ≥ 10× faster than the v1 MC path (Phase 4 budget at `n_positions=10k, n_rollouts=50` was 3-6h; analytic is one closed-form pass per position → seconds).

**Tradeoffs the analytic scorer cannot model.** Yield variance (variance-of-yield matters for risk-of-bust under StackedDice + Karma); robber-threat (Friendly Robber filtered; still relevant once a player crosses 3 visible VP); longest-road skeleton via the 2nd road; port reachability when a settle is 1 road away from a port vertex. The senior conceded the MC estimator was *at best* stumbling onto these second-order effects without modeling them; analytic is strictly better on the first moment.

### A.1 — Analytic scorer module

**File (new)**: `src/catan_rl/setup_phase/analytic_value.py`.

**Surface**:
```python
def vertex_yield(
    board: catanBoard,
    vertex_idx: int,
    resource_weight: Mapping[str, float],
) -> float: ...

def all_vertex_yields(
    board: catanBoard,
    resource_weight: Mapping[str, float],
) -> np.ndarray:  # (54,) float32
    """Vectorized: returns the closed-form score for every vertex on the board."""

def edge_yield_after_settlement(
    board: catanBoard,
    settle_vertex: int,
    edge_idx: int,
    resource_weight: Mapping[str, float],
) -> float:
    """Score the road's downstream settlement option: the *best* vertex
    reachable by extending one more road from the road's far endpoint."""
```

Pip-count table inlined as `_DOTS_BY_TOKEN = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1, 7:0, None:0}` (matches `obs_encoder.py:_DOTS_BY_TOKEN`). Single source of truth: import from `obs_encoder.py` rather than re-define.

**Tests** (`tests/unit/setup_phase/test_analytic_value.py`):
- `test_desert_contributes_zero` — vertex adjacent only to desert returns 0.
- `test_pip_sum_matches_hand_computed_fixture` — five hand-computed reference vertices on a fixed-seed board; assert exact equality.
- `test_invariant_under_d6` — for every D6 element `g` (1..11), permuting the board's chips + resources through `g` and re-querying via the analytic scorer at `corner_perm(g)[v]` yields the same score (modulo float ε). **This is the unit-level pin** for symmetry consistency — a precondition for Phase B's port augmentation to be safe.
- `test_known_top_vertex` — on a fixed seed, the analytic top-3 vertices match the hand-computed top-3 list.

### A.2 — Resource-weight calibration (dual-table approach, per §0.2 decision)

**Calibration source decision** (recorded 2026-06-03): no champion ckpt available; ship two named tables and let §A.4 pick.

**Table 1 — Charlesworth's published prior** (`charlesworth_v0`): hardcoded `{WOOD: 1.0, BRICK: 1.0, WHEAT: 1.0, ORE: 1.1, SHEEP: 0.7}`. Ships as a static constant in `src/catan_rl/setup_phase/resource_weights.py` (new file). No calibration step required.

**Table 2 — Heuristic-derived** (`heuristic_v0`): fit NNLS against v2 heuristic-vs-heuristic end-game resource shares. File (new): `scripts/calibrate_setup_resource_weights.py`. Runs `n_games=2000` heuristic-vs-heuristic games (no model needed; only the v2 heuristic in `src/catan_rl/agents/heuristic.py`), records per-game **end-game resource accounting** per player (engine broadcast event bus exposes this), fits weights by minimizing `||analytic_score(start_vertices) − terminal_resource_share||²` via NNLS over the 5-resource simplex. Output: `data/setup_phase/resource_weights_heuristic_v0.json` with `{wood:..., brick:..., wheat:..., ore:..., sheep:...}` + provenance block (`source: heuristic_vs_heuristic`, `n_games`, `seed`, `fit_residual`, `fit_date`).

**Why heuristic-derived is informative even without a champion**: the heuristic plays to its own dot-count value function; the weights it converges to are NOT a principled "win-correlated" signal, but they ARE a reproducible alternative-to-prior that can be ablated against Charlesworth. If the heuristic-derived weights diverge wildly from Charlesworth's, that's diagnostic.

**Table 3 — Champion-derived** (`champion_v0`): **DEFERRED**. Same calibration code path as Table 2 but sourced from a v2 PPO champion's end-game accounting once one exists. Tracked as a follow-up; not gating Phase A.

**Tests** (`tests/unit/setup_phase/test_weight_calibration.py`):
- `test_charlesworth_table_loadable` — Table 1 constant is importable, has all 5 keys, all values in `(0, 2.0)`.
- `test_heuristic_calibration_reproducible_given_seed` — running the calibrator twice with the same seed produces bit-identical Table 2 weights.
- `test_heuristic_weights_sum_close_to_target` — Table 2 weights sum to within ±20% of 5.0 (relaxed vs the original ±5% — heuristic is biased and this is acceptable noise).
- `test_weight_tables_named_registry` — both tables are accessible via a `get_resource_weight_table(name: str)` entry point that returns a dict.

**Tests** (`tests/unit/setup_phase/test_weight_calibration.py`):
- `test_calibration_reproducible_given_seed` — running the calibrator twice with the same seed + checkpoint produces bit-identical weights.
- `test_weights_sum_close_to_target` — the fitted weights sum to within ±5% of the 5-component baseline (5.0), guarding against degenerate all-zero or all-large fits.
- `test_weights_charlesworth_alternative_available` — both named tables loadable from one entry point.

### A.3 — Trainer integration

**File (modified)**: `src/catan_rl/setup_phase/setup_trainer.py` (creates this v2 path; the v1 trainer was `src/catan_rl/rl/setup/`).

The setup trainer's MC scoring call site (the inner loop that produces `mc_value` per `(board_seed, decision_index)`) is replaced by `all_vertex_yields(board, resource_weight)`. The trainer's public API — what the main policy head consumes via `pretrain_setup_heads.py` / its v2 equivalent — is **preserved**: it still emits `(obs, action_corner, action_edge, decision_index, value)` rows. `value` is now the analytic score, not an MC win-rate proxy; the loss code path is identical.

**Lineage fork**: the setup-head checkpoint is renamed `checkpoints/setup_pretrain_analytic/<run_id>/pretrain_final.pt` so the MC and analytic lineages coexist on disk; the merge script (`scripts/merge_setup_heads.py` per v1 §5) takes `--scorer {mc|analytic}` and refuses to merge if the source scorer doesn't match the expected lineage tag.

**Tests** (`tests/integration/setup_phase/test_analytic_trainer.py`):
- `test_low_noise_reference_equivalence` — synthesize a low-noise reference board (forced chip layout, no dice randomness, n_rollouts=∞ MC limit), verify MC and analytic agree to within float ε. The senior asked for this explicitly.
- `test_step_wall_time_10x_faster_than_mc` — time both paths on 100 positions; assert analytic ≥ 10× faster.

### A.4 — Acceptance gate

**Gate A** (compound; all pass):
1. **Analytic vs MC ablation** at matched setup-pretrain budget: 200 games × 5 seeds × 2 seats vs heuristic opponent (no champion-vs-champion available — gating opponent is the v2 heuristic). Symmetrized WR(analytic) − WR(MC) ≥ +0.005 with Wilson 95% CI clearing zero. Run the ablation TWICE — once per resource-weight table (Charlesworth `charlesworth_v0`, heuristic-derived `heuristic_v0`) — pick the winner.
2. **Wall-time** per setup-training step ≥ 10× faster than the MC equivalent at matched `n_positions`.
3. **§A.3 low-noise equivalence test** green (analytic ≈ MC at n_rollouts=∞ on a deterministic reference board).
4. **Sanity canary**: heuristic-WR over 200 games not regressed by > 2pp relative to the pre-analytic-scorer setup-pretrain run (catches catastrophic encoder contamination via head retraining).

**STOP/RESUME after Gate A**: record (a) which resource-weight table won the ablation, (b) the realized WR delta, (c) the wall-time multiplier in `MEMORY.md`. **Decision points**:
- If Gate A.1 fails for BOTH tables but A.2-A.4 pass → analytic scorer is *available* (correct, fast) but does not lift WR. Setup heads are saturated; merge as a correctness fix; Phase B becomes the next intervention.
- If Gate A.1 passes for one table but not the other → the winner becomes default; record the loser as deprecated.
- If Gate A.1 passes for both tables → pick the simpler one (Charlesworth) as default; document the heuristic-derived as an alternative for future re-ablation against a real champion.

---

## Phase B — Port-aware D6 symmetry augmentation

**Problem.** `symmetry_tables.py` exports `tile_perm`, `corner_perm`, `edge_perm`, `within_tile_corner_perm`, `within_tile_edge_perm` — **but no `port_perm`**. The senior's catch: ports are assigned to specific vertex positions on the board's outer ring; under D6 rotation the *vertex* the port lives at moves, but `obs_encoder.py`'s `_vertex_port_static` table (a static (54, 7) one-hot built once per encoder construction) was assembled from the engine's port→vertex assignment at encoder-init time and is permuted via the corner axis (`vertex_features` ride the corner perm in `dihedral.py:15-17`).

The augmentation is correct *for the per-vertex port one-hot it sees*, but the BC pipeline canonicalizes labels through D6 expecting the engine's port layout to also rotate; at eval time the engine presents ports at the engine-fixed vertices, not the canonicalized vertices. The round-trip property tests (`reduce ∘ unreduce == identity`) pass because they only touch the symmetry tables, not the engine port assignment. **The PPO main-loop `symmetry_aug_prob=0.5` survived this for >18M v1 steps only because both self-play sides see the same transform and the bias cancels** (CLAUDE.md Phase 1.5 note). The BC path has no such cancellation — labels and obs canonicalize independently of the eval-time engine state.

**Target.**
1. Extend `symmetry_tables.py` with a 9-port permutation table per D6 group element (5×2:1 + 4×3:1 = 9 port objects on the standard board).
2. Apply the full D6 transform (tiles + corners + edges + within-tile slots + **ports**) in the BC loader at training time, with `bc_symmetry_aug_prob=0.5` (BC's existing default).
3. Property tests: `reduce ∘ unreduce == identity` AND every game-rule invariant survives (port-edges still point at port-vertices that connect to outer-ring tiles; resource adjacency preserved).

**Expected impact.** +1 to +3% WR depending on BC dataset size. Stronger effect on smaller BC datasets (the 30k-game baseline per `v2_step3_bc.md` §1) than on full-PPO models.

**Do not touch the main-loop `symmetry_aug_prob` path until port permutation is independently verified.** The v1 main-loop aug works because of bilateral cancellation; flipping it on with a buggy port_perm would regress training silently. Phase B is BC-loader-only until §B.4 property tests are green.

### B.1 — Port permutation table

**File (modified)**: `src/catan_rl/augmentation/symmetry_tables.py`.

Add to `_BoardData` TypedDict: `port_vertex_pairs: list[tuple[int, int]]` (the 9 port objects' vertex anchor pairs — each port sits on an edge of the outer ring, anchoring two adjacent vertices). Extract from `board.boardGraph[px].port` per-vertex; for each port, collect its two anchor vertices.

Add to `_all_perms()`: `port: list[np.ndarray]` of length 12, each `(9,)` int64. Derivation: for D6 element `g`, the port at original index `i` with anchor vertices `(va, vb)` rotates to anchor vertices `(corner_perm(g)[va], corner_perm(g)[vb])`; find the port at this rotated anchor pair. Pinned by a brute-force lookup test in `test_port_perm_correctness.py`.

Public accessor: `def port_perm(g: int) -> np.ndarray:` mirroring `corner_perm`'s signature.

### B.2 — BC loader port augmentation

**File (modified)**: `src/catan_rl/bc/loader.py`.

The current `apply_symmetry(obs_b, action_b, mask_b, g)` call permutes `vertex_features` along the corner axis (which already correctly drags the per-vertex port one-hot). The new requirement is that BC-augmented obs **also** carries the engine-rotated port assignment — i.e., if a port at engine-vertex `v` shows up in the obs at canonicalized-vertex `corner_perm(g)[v]`, and the trained policy is queried at eval time on a state where the port is at engine-vertex `v` (unrotated), the model has seen this port both at `v` and at `corner_perm(g)[v]` during training. This is what unlocks the +1 to +3% WR.

Implementation: `apply_symmetry` is extended to accept a `permute_ports: bool = True` kwarg. When True, before the existing corner-axis permutation, the per-vertex port one-hot slice within `vertex_features` is **separately permuted via `port_perm(g)`** at the port-index level (the 9 port objects), then re-mapped onto vertices via the rotated anchor lookup. The existing tile/corner/edge axis perm runs after.

The BC dataset's `__getitem__` already calls `apply_symmetry` — no caller change needed; the default kwarg behavior is the new (correct) port-aware path.

### B.3 — Kill-switch test

**File (new)**: `tests/unit/augmentation/test_port_symmetry_killswitch.py`.

Intentionally swap two entries of the `port_perm(g)` table for some `g ≠ 0`; assert a downstream evaluation test (the engine roundtrip in §B.4) catches the corruption. This is the senior's explicit requirement: a broken port table must produce a *visible* failure, not a silent regression.

### B.4 — Property tests

**File (new)**: `tests/unit/augmentation/test_port_symmetry.py`.

- `test_port_perm_inverse_identity` — for every `g`, `port_perm(D6_INVERSE(g))[port_perm(g)] == np.arange(9)`.
- `test_port_anchor_consistency` — for every `g, i`, the port at index `port_perm(g)[i]` has anchor vertices `(corner_perm(g)[v_a], corner_perm(g)[v_b])` where `(v_a, v_b)` are the original port's anchors.
- `test_port_type_preserved` — D6 acts on the *geometry*, not on the port type; assert `board.ports[i].port_type == board.ports[port_perm(g)[i]].port_type` after rotation.
- `test_outer_ring_invariant` — port-anchored edges are outer-ring edges before AND after every rotation.

**File (new)**: `tests/integration/test_bc_loader_aug.py`.

End-to-end: with `bc_symmetry_aug_prob=1.0` (force aug every sample), draw 100 batches from a tiny mock BC dataset, assert (1) every batch's `vertex_features` port one-hot slice is consistent with the engine port layout under the sampled `g`, (2) the action's `corner` index moved through `corner_perm(g)`. Pinned to catch the "augmentation is state-only" regression that v1 codified as a correctness bug.

### B.5 — Acceptance gate

**Gate B** (compound; all pass):
1. **BC WR ablation**: BC-trained model with port-aware aug ≥ +1pp WR vs BC no-aug on the same v2 BC labeled-scenarios dataset, 200 games × 3 seeds × 2 seats, symmetrized, Wilson 95% CI clears zero.
2. **Kill-switch test passes**: deliberately broken port table is detected by the §B.4 roundtrip integration test.
3. **Phase A's `test_invariant_under_d6` still green** — port augmentation must not break the analytic scorer's D6 invariance (Phase A is the cross-validation surface for Phase B; this is the explicit dependency).

**STOP/RESUME after Gate B**: record the BC WR delta in `MEMORY.md`. If Gate B.1 fails, leave the BC loader on the legacy state-only path (which is correctness-buggy for port-sensitive policies but is the v1-validated default). Do **NOT** flip on the PPO main-loop port aug — that requires a separate ablation outside this plan's scope.

---

## Phase C (OPTIONAL meta-redirect) — ISMCTS activation OR belief-head reinforcement

**The senior's pushback**: setup-phase tweaks give +1% WR per +1% setup quality, but the training plateau is **most likely a value-learning ceiling, not a setup-quality ceiling**. If true, Phases A and B each lift WR by a few points but the plateau resurfaces — the marginal compute is better spent on the value-learning path.

**Phase C does not pre-commit to either option.** The plan surfaces the decision and proposes a one-day diagnostic to resolve which interpretation is right.

### C.0 — Diagnostic experiment (1 day)

**Question**: is the WR plateau dominated by (a) suboptimal setup picks or (b) miscalibrated value estimates mid-game?

**Method**: take the current champion. Run 1000 self-play games. For each game, record:
- (a) Setup-phase quality proxy: `analytic_score(p1_settle_1) + analytic_score(p1_settle_2)` minus same for p2. Histogram across all games where p1 wins vs loses.
- (b) Mid-game value error: at every step after setup, record `|V(s) − terminal_z|`. Aggregate by game outcome.

**Decision rule**:
- If (a) discriminates win/loss with AUC > 0.65 → setup-quality bottleneck → Phase A and B carry the load; Phase C is descoped.
- If (a)'s AUC < 0.55 but (b)'s mean-abs-error stays > 0.20 throughout mid-game → value-learning ceiling → Phase C should ship.
- If neither discriminates → policy bottleneck (action selection given correct value); Phase C's belief variant (Option C2) is more promising than ISMCTS.

The diagnostic ships as `scripts/diagnose_plateau_source.py` and writes `runs/plateau_diagnosis/<run_id>/summary.json`. Cost: ~3h wall time on M1 Pro CPU.

### C.1 (Option) ISMCTS activation

**File (audit)**: `src/catan_rl/algorithms/search/ismcts.py` — **does NOT exist in the v2 tree** (verified 2026-06-03). CLAUDE.md describes it as "shipped but not wired into rollouts" in v1 (`/Users/benjaminli/my_projects/catan_rl/src/catan_rl/algorithms/search/ismcts.py`). Activation in v2 requires either porting the v1 module OR re-implementing per `docs/plans/v2_step5_mcts.md`. **The senior's "already shipped per CLAUDE.md Phase 4" claim does not hold for v2** — this is a from-scratch implementation in the v2 codebase.

Also requires `CatanGame.copy()` — confirmed missing in v2 (`src/catan_rl/engine/game.py`). Adding it is a separate prerequisite PR scoped at ~200 LOC.

Decision: **defer to `docs/plans/v2_step5_mcts.md`'s implementation track**; this plan does not re-spec MCTS.

### C.2 (Option) Belief-head reinforcement

Push `belief_loss_weight` from 0.05 → 0.10 (or 0.15) and add the historical-league-filtered opp-action variant (existing `use_opponent_action_head=True` plus the league-filter the trainer already applies). Cheaper than ISMCTS; smaller upside.

### C.3 Acceptance gate

**Out of scope for this plan unless the team commits.** Plan surfaces the decision; commitment routes to either the v2 MCTS plan or a new belief-reinforcement plan.

---

## Dependency graph

```
Phase A
 ├─ A.1 analytic scorer + tests
 ├─ A.2 weight calibration (depends on A.1, §0.2 source ckpt)
 ├─ A.3 trainer integration (depends on A.1, A.2)
 └─ A.4 acceptance gate (depends on A.1-A.3)
        │
        └─► Phase A's analytic scorer is the cross-validation surface for Phase B
                                                                          ▼
Phase B
 ├─ B.1 port_perm table (independent)
 ├─ B.2 BC loader aug (depends on B.1)
 ├─ B.3 kill-switch test (depends on B.1, B.2)
 ├─ B.4 property tests (depends on B.1, A.1's D6 invariance test as cross-check)
 └─ B.5 acceptance gate (depends on B.1-B.4)

Phase C (independent of A, B)
 ├─ C.0 diagnostic experiment (independent)
 └─ C.1/C.2 implementation (decision-gated on C.0)
```

Phase A and Phase B are sequentially ordered (B reuses A.1's invariant test). Phase C is fully independent; the diagnostic in C.0 can run *in parallel* with A and B.

---

## Eval-harness integration

`scripts/eval_harness.py` does **not exist** in v2 (verified 2026-06-03). CLAUDE.md references it as v1 surface. This plan does NOT spec the harness; it specs the flags the harness must expose once it lands.

**Flags the harness must grow** (deferred to whichever PR ports/builds the v2 eval harness):
- `--mode setup-strength-ablation` — runs the §A.4 Gate 1 ablation (analytic vs MC setup pretrain at matched checkpoint lineage).
- `--mode port-aug-ablation` — runs the §B.5 Gate 1 ablation (BC port-aug vs no-aug).
- `--mode plateau-diagnostic` — runs the §C.0 setup-quality vs value-error split.

In the interim, each phase ships its acceptance-gate runner as a standalone script under `scripts/setup_strength/` so the gates can be evaluated without the harness in place.

---

## Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Resource-weight calibration overfits one champion's bias | Medium | High | Ship both champion-derived AND Charlesworth's tables; pick via §A.4 ablation; record decision in `MEMORY.md`. |
| `checkpoint_07390040.pt` does not load under v2 code | High | Medium | §0.2 preflight surfaces this; fallback to strongest v2 BC checkpoint for calibration. |
| Port permutation table is subtly wrong | Medium | High | §B.3 kill-switch test + §B.4 property tests; do NOT touch PPO main-loop aug until verified. |
| Lineage fork (analytic vs MC setup-head checkpoints) gets crossed | Low | High | Merge script (`scripts/merge_setup_heads.py`) takes `--scorer {mc|analytic}` and refuses cross-lineage merge. Lineage tag in checkpoint metadata. |
| Phase A WR gate fails (analytic scorer correct but no WR lift) | Medium | Low | Gate A treats this as "analytic is the new correct baseline"; the closed-form scorer + 10× speedup still merges. Phase B becomes the next intervention. |
| Phase B BC WR gate fails | Medium | Medium | Leave BC loader on the legacy state-only path; diagnose via port-perm fixture inspection; do not touch PPO main-loop aug. |
| Phase C diagnostic gives ambiguous result | Medium | Low | Diagnostic ships pass/fail thresholds; ambiguous → no Phase C commitment; phases A+B's deltas determine total roadmap value. |
| v2 ISMCTS path not yet implemented (vs CLAUDE.md claim) | High | Low | Defer to `v2_step5_mcts.md`. Phase C.1 is a *decision*, not an implementation, in this plan. |
| `CatanGame.copy()` still missing for any rollout-based phase | High | Medium | Surface in §C.0; if C.0 routes to ISMCTS, the `copy()` PR is a prereq tracked separately. |
| 1v1 ruleset drift through any code path here | Low | Critical | Plan touches NO engine code. Analytic scorer is read-only on `catanBoard`. Augmentation tables are permutations, not state mutations. Port table extends, does not modify, the existing D6 group. |
| Champion checkpoint backward-compat broken by analytic scorer trainer | Low | High | `CatanPPO.load()` is not touched. Setup-head fork is a separate file; main checkpoint untouched. |
| TensorBoard scalar renaming | Low | Low | New scalars only (`setup_analytic/*`, `bc/port_aug_*`); existing scalars under `runs/train/` preserved per CLAUDE.md rule 3. |

---

## STOP/RESUME checkpoints

| Where | What to verify | Human decision |
|---|---|---|
| **After §0 preflight gates** | Chip-layout determinism green; champion ckpt load result recorded; port-permutation gap confirmed. | OK → Phase A kickoff. FAIL chip-determinism → audit `_build_spiral_path`. FAIL ckpt load → switch calibration source to v2 BC ckpt. |
| **After §A.4 Gate A** | Symmetrized WR delta computed; wall-time 10× verified; analytic vs MC reproducibility green. | PASS all → Phase B kickoff. FAIL A.1 (no WR lift) → analytic scorer still merges as correctness fix; record in `MEMORY.md`; jump to Phase B. FAIL A.4 sanity → revert head-pretrain; debug encoder contamination before retry. |
| **After §B.5 Gate B kill-switch** | Broken port table detected by §B.4 property test in CI. | PASS → proceed to BC WR ablation. FAIL → port table derivation is wrong; do not run WR ablation. |
| **After §B.5 Gate B WR ablation** | BC WR delta computed across 3 seeds × 200 games × 2 seats. | PASS → port-aware aug becomes BC default; document delta. FAIL → leave BC loader on legacy state-only; record null result. **Do NOT flip PPO main-loop port aug.** |
| **After §C.0 diagnostic** | Setup-quality AUC, mid-game value MAE, decision-rule applied. | Routes to Phase A/B sufficiency, ISMCTS commitment, or belief-reinforcement commitment. **Open team decision.** |

---

## Open questions / decision points

1. **Resource-weight source** — **resolved 2026-06-03 via §0.2**: champion-derived option DEFERRED (no v2 ckpt available). Phase A ships dual tables: Charlesworth's prior (`charlesworth_v0`) and heuristic-derived (`heuristic_v0`). §A.4 ablation picks the winner. Champion-derived option re-evaluates when v2 PPO produces a competitive checkpoint.

2. **MC rollout count cited in this plan's prompt: 20 vs the v1 default of 50.** The v1 setup_pretrain_plan uses 50; reviewer cited 20. Either way analytic dominates first-moment estimation. Decision: ignore the discrepancy — analytic is the replacement; the historical MC count does not gate anything.

3. **Phase C diagnosis routing.** Surfaced as the explicit team decision; the diagnostic experiment in §C.0 is the resolution mechanism. Do not pre-commit.

4. **v2 eval harness landing schedule.** Phase A and B gates ship interim standalone scripts under `scripts/setup_strength/` until the harness lands. When the harness exists, retire those scripts and migrate the gates to harness flags per the "Eval-harness integration" section.

5. **Champion-checkpoint lineage**: `checkpoints/train/checkpoint_07390040.pt` is v1; v2 has its own BC checkpoint lineage per `v2_step3_bc.md`. The senior's roadmap implicitly assumed the v1 ckpt is the calibration target; if the v2 BC checkpoint clears its Gates 1+2+3 first, that becomes the analytic-scorer calibration source instead.

---

## Compute budget

| Phase | Engineering days | Compute |
|---|---|---|
| Phase A (A.1 scorer + tests + A.2 calibrator + A.3 trainer + A.4 gate) | 3 | A.2 calibration: ~3h (2000 games × ~3s/game / 8 parallel). A.4 ablation: ~2h (200 × 5 × 2 = 2000 games). |
| Phase B (B.1 port_perm + B.2 BC loader + B.3 kill-switch + B.4 property + B.5 gate) | 2 | B.5 ablation: ~1.5h (200 × 3 × 2 = 1200 games × ~3s, plus the fine-tune training of ~30 min). |
| Phase C diagnostic (C.0 only) | 0.5 | ~3h on M1 Pro CPU (1000 games + per-step value-error pass). |
| **Total (A + B + C.0)** | **5.5 days** | **~10h compute** |

---

## File scope (absolute paths, all NEW unless marked)

**New (Phase A)**:
- `src/catan_rl/setup_phase/__init__.py`
- `src/catan_rl/setup_phase/analytic_value.py`
- `src/catan_rl/setup_phase/resource_weights.py` — registry: Charlesworth hardcoded + heuristic-derived loaded from JSON
- `src/catan_rl/setup_phase/setup_trainer.py`
- `scripts/calibrate_setup_resource_weights.py` (heuristic-vs-heuristic; champion-derived deferred)
- `scripts/setup_strength/run_phase_a_ablation.py`
- `data/setup_phase/resource_weights_heuristic_v0.json` (generated by §A.2 calibrator)
- `tests/unit/setup_phase/test_chip_layout_invariance.py` ✅ shipped 2026-06-03 (§0.1 preflight)
- `tests/unit/setup_phase/test_analytic_value.py`
- `tests/unit/setup_phase/test_weight_calibration.py`
- `tests/integration/setup_phase/test_analytic_trainer.py`

**New (Phase B)**:
- `scripts/setup_strength/run_phase_b_ablation.py`
- `tests/unit/augmentation/test_port_symmetry.py`
- `tests/unit/augmentation/test_port_symmetry_killswitch.py`
- `tests/integration/test_bc_loader_aug.py`

**New (Phase C diagnostic)**:
- `scripts/diagnose_plateau_source.py`

**Modified**:
- `src/catan_rl/augmentation/symmetry_tables.py` — adds `port_perm` accessor; extends `_BoardData` / `_AllPerms`.
- `src/catan_rl/augmentation/dihedral.py` — `apply_symmetry` gains `permute_ports=True` kwarg; default-on path is the new port-aware behavior.
- `src/catan_rl/bc/loader.py` — no API change; loader picks up the new `apply_symmetry` default. Configurable `bc_symmetry_aug_prob` already present.
- `CLAUDE.md` — single line under "Active roadmap" noting analytic setup scorer + port-aware aug landed.
- `MEMORY.md` — one-line entries after each phase lands with the realized WR deltas.

---

**WAITING FOR CONFIRMATION**
