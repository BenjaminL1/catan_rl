# ADR 0007: Phase 2 Architecture Upgrades

**Status:** Accepted
**Date:** 2026-05-02

## Context

Phase 1 squeezed sample efficiency out of the fixed 1.54M-parameter network
(value clipping, advantage normalization, compact obs, count-encoded dev
cards, D6 symmetry augmentation). The next bottleneck is the encoder /
decoder architecture itself: the tile transformer was permutation-equivariant
across hex tiles (wrong: the board has fixed spatial structure), the action
heads conditioned on context via concatenation (low-rank), and the value
head shared a trunk with the policy (gradient interference).

This ADR records the four Phase 2 upgrades and the design choices that fell
out of fitting them onto the Phase 1 stack while preserving back-compat,
the resume path, and the YAML config surface.

## Decisions

### 2.1 Axial positional embedding for tiles

A learned 2D embedding indexed by axial hex coords `(q, r)` is concatenated
to each tile's input feature vector before the projection into `d_model`.
With 19 tiles and a fixed board topology, the indices are static — we
precompute them once via `engine.board.catanBoard()` and register them as a
non-parameter buffer.

- **Why axial coords (not pixel xy):** axial is the natural lattice for hex
  grids; q ∈ [-2, 2] and r ∈ [-2, 2] map to a 5×5 lookup after a +2 shift,
  costing one tiny `nn.Embedding` per axis instead of a 2D continuous MLP.
- **Why split q and r into half-dim halves and sum:** factorized 2D
  positional encodings; cheap and sufficient at 19 positions.
- **Init std=0.02:** small enough that the encoder behaves like the no-pos
  baseline for the first few updates, large enough that the gradient is not
  vanishing.
- **Off by default** (`use_axial_pos_emb=False`) so phase1 lineage stays
  compatible.

### 2.2 Modern transformer recipe (pre-norm, GELU, dropout)

The encoder layer was already pre-norm. Phase 2 adds two opt-in tweaks:

- `transformer_dropout` overrides the global `dropout` for the
  `TransformerEncoderLayer` only — separates "tile encoder dropout" from
  "any-other-dropout" so we can tune them independently.
- `transformer_activation: 'gelu'` swaps the FFN nonlinearity. GELU is the
  modern default and helps small-batch stability without changing param
  count.

Both default to the legacy values (None / "relu") so the recipe is opt-in.

### 2.4 AdaLN/FiLM-conditioned action heads

The 6 action heads previously concatenated context (e.g. settle/city
one-hot) onto the observation before a 2-layer MLP. Concat is low-rank: the
context can shift the input but cannot rescale or modulate features
selectively. FiLM lets the head learn per-feature `(γ, β)` from context:

```
x = fc1(main); x = (1+γ_1) ⊙ LN(x) + β_1; x = act(x);
x = fc2(x);    x = (1+γ_2) ⊙ LN(x) + β_2; x = act(x);
```

`(γ_1, β_1, γ_2, β_2)` come from a single linear `film_gen: context → 4·hidden`
initialized to zero so `1+γ = 1` and `β = 0` at construction (identity
modulation, LLaMA-style). Heads with no context (`type`, `edge`, `tile`)
keep the legacy concat path because there is nothing to modulate from.

- **Why param-neutral:** the FiLM gen replaces the concat-input width on the
  context-using heads; net param count differs by <10%.
- **`elementwise_affine=False` on the LayerNorms:** the modulation is the
  affine layer; redundant trainable γ/β on the LN itself would double-count.

### 2.5 Decoupled value tower (Option A)

Phase 0/1 used a single `ObservationModule` whose output fed both the
policy heads and the value MLP. This couples the two losses' gradients:
the value loss reshapes the trunk that the policy depends on. The
"decoupled" variant (`value_head_mode='decoupled'`) builds a second
`ObservationModule` exclusively for the value head:

- **Option A (chosen):** symmetric architectures (same hyperparameters),
  separate weights. Simplest to implement, clean gradient isolation, costs
  ~+0.7M params at our scale.
- **Option B (rejected for now):** smaller "value-only" encoder (e.g. fewer
  transformer layers). Saves params but adds a hyperparameter to tune.
  Revisit if Option A's compute overhead becomes a problem.
- **Option C (rejected):** PPG-style separate phases. Higher engineering
  complexity, marginal expected gain at this scale.

The two encoders are wired separately end-to-end: `act()` and
`evaluate_actions()` route the value path through `_value_features`, and
`get_value()` skips the policy encoder entirely. The unit test
`test_decoupled_value_gradient_isolation` verifies gradients from a
value-only loss never reach policy-encoder parameters.

## Consequences

- Fresh-training only. Phase 2 lineage is not state-dict-compatible with
  `checkpoint_07390040.pt` or any Phase 1 checkpoint that didn't have a
  `value_observation_module` or FiLM `film_gen` parameters.
- Param count: ~1.54M (legacy) → ~2.22M (phase2_full). Most of the increase
  is the second observation encoder for the value head.
- FPS: ~30% slower per-update on M1 Pro CPU due to the second encoder
  forward pass; this is the dominant cost.
- All four upgrades are independently flagged with leave-one-out configs
  (`configs/phase2_no_*.yaml`) so each can be ablated individually.

## Alternatives considered

- **GNN encoder over the (tile, vertex, edge) tripartite graph (originally
  scoped 2.3):** deferred to a follow-up. The axial pos embedding closes
  most of the topology gap at a fraction of the engineering cost; revisit
  if Phase 2 ablations show a residual gap that pos-emb alone can't close.
- **Opponent-action auxiliary loss (originally scoped 2.5c):** still
  deferred. Distinct from 2.5b because it predicts the opponent's *next
  action distribution* against historical league policies (not their
  hidden hand), and requires the rollout buffer to carry per-step opponent
  policy IDs *and* the actually-observed opponent action — a larger
  surface change.

## Follow-up: 2.5b belief head (landed)

The opponent hidden-dev-card belief head was originally bundled with 2.5
"decoupled value tower" but deferred when 2.5 shipped. It landed in a
follow-up branch (`feat/phase-2-5b-belief-head`):

- New `src/catan_rl/models/belief_head.py`. Two-layer MLP from the policy
  encoder's 512-dim output to a 5-way logit over dev-card types
  `{KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY}`. Final layer init gain=0.01
  so initial outputs are near zero → softmax near uniform → loss starts
  at the well-defined `log(5) ≈ 1.609` rather than at an arbitrary value.
- Loss is `BeliefHead.soft_cross_entropy(logits, target)` =
  `-Σ target_i · log_softmax(logits)_i`. Equivalent to KL up to the
  constant entropy of the target, so the gradient is identical.
- Env exposes `obs['belief_target']` only when `use_belief_head=True` —
  it's a training-only field, never read by the policy at inference time
  so the policy genuinely has to predict (not look up) the distribution.
- When the opponent has zero hidden cards the env returns the uniform
  distribution; soft CE on a degenerate all-zeros target is undefined.
- Buffer opt-in `store_belief_target=True` mirrors Phase 3.6's
  `store_opponent_id` pattern. Both default off so legacy lineages don't
  pay the buffer footprint.
- **Why 1v1-only:** with no P2P trade and a single opponent, the
  broadcast tracker reveals everything except dev-card *type* — exactly
  what we model. With P2P trade the env's "true" target is stale by the
  time the loss fires; with 4 players the joint distribution output dim
  explodes (5⁴=625) or factorizes away most of the structure.

Param count: phase2_full ~2.22M → phase2_belief_head ~2.29M (~70k for the
new MLP). Config: `configs/phase2_belief_head.yaml` extends `phase2_full`.
