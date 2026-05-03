# ADR 0009: Phase 4 — ISMCTS and Recurrent Value Head

**Status:** Accepted (4.1 module landed; rollout-loop activation deferred)
**Date:** 2026-05-02

## Context

Phases 0–3 closed the bulk of the roadmap: trainer correctness, sample
efficiency, architecture, self-play diversity, plus 2.5b belief head and
the 3.3 duo-exploiter follow-up. The remaining items in §7 (Phase 4)
were *optional / compute-permitting* — only relevant if Phase 3 misses
its success criteria. We don't have empirical Phase 3 eval results yet,
but the user asked to ship the rest of the phases as one PR. This ADR
records what landed and how the deferred-by-design parts (ISMCTS rollout
integration, multi-step env clone) are scoped.

## Decisions

### 4.2 GRU recurrent value head

**File:** `src/catan_rl/models/recurrent_value.py`.

`RecurrentValueHead` wraps an `nn.GRUCell` with the existing value MLP.
The value path becomes `obs_out → GRUCell(obs_out, h_t) → h_{t+1} →
concat(obs_out, h_{t+1}) → value_mlp → V`. When `use_recurrent_value=True`,
`CatanPolicy` builds `recurrent_value_head` and sets `value_net=None`;
all value forward passes route through the recurrent head.

Per-env hidden state is maintained by the **trainer**, not the policy:

  - `CatanPPO._value_hidden_state: Tensor(n_envs, gru_hidden_dim)`,
    initialized to zeros.
  - Each rollout step snapshots `h_t` BEFORE `policy.act` and stores it
    in the rollout buffer's `value_hidden_in[step]`.
  - After `policy.act`, the returned `hidden_out` updates the live state.
  - On `terminated=True` (a real game-over), `_value_hidden_state[env_i]`
    resets to zeros. On `truncated=True` it's preserved — the trajectory
    is being cut for bookkeeping, the underlying state hasn't changed.

PPO update replays the GRU once per sample with the buffered `h_t`, so
gradient flows through *one* GRU step per sample but not through the
full sequence (BPTT-length-1). This is the standard simplification when
minibatches are shuffled non-sequentially, and it's enough to teach the
GRU to be a useful summary because the value MLP keeps direct access to
`obs_out` via the concat residual.

**Why concat instead of replacing the obs feature with the hidden state:**
preserves the existing value MLP's signal path; the network can ignore
the GRU output entirely if it doesn't help, instead of being forced to
route everything through a 64-dim bottleneck.

**Why reset on `terminated` but not `truncated`:** standard recurrent-PPO
recipe. Game-over is a hard discontinuity; truncation is a window we
open for bookkeeping while the underlying state continues smoothly.

### 2.5c Opponent next-action auxiliary head (closed; was deferred from Phase 2)

**File:** `src/catan_rl/models/opponent_action_head.py`.

Predicts opponent's next action type (13-way) from the policy encoder.
Targets are captured trainer-side from `_run_batched_opponent_turns`'s
*first* opponent action per env, then injected into `obs_batch[env_i]`
before `buffer.add`. Validity masking excludes:

  - random / heuristic opponents (no policy ID, no comparable behavior),
  - `current_self` opponent (the AlphaStar-warned degenerate fixed point
    where head and policy chase each other),
  - any rollout where no deferred-opp turn fired between the agent's
    step and the next observation.

`OpponentActionHead.masked_cross_entropy` returns `None` when no batch
row qualifies; the trainer skips the loss term in that case. Loss
weight 0.03 — smaller than 2.5b's 0.05 because the supervision is
noisier (action depends on hidden hand which we don't see).

### 2.3 GNN encoder (closed; was deferred from Phase 2)

**File:** `src/catan_rl/models/graph_encoder.py`.

Tripartite GNN over 19 hex + 54 vertex + 72 edge nodes. Adjacency tables
are precomputed once from a single `catanBoard()` build and registered
as buffers. Initial features: `hex_node[h] = proj(tile_features[h])`;
`vertex_node[v] = proj(mean over adjacent hex tiles)`; `edge_node[e] =
proj(mean over endpoint vertex aggregates)`. 2 rounds of message passing,
each round simultaneously updates all three node types from a snapshot
of the current state (no in-round contamination). Pooled by mean-over-
node-type, concatenated to a 3×hidden vector, projected to `out_dim`.

The output concatenates to the fusion input alongside the tile-encoder
output — additive, not replacement. This preserves the tile transformer
(which captures pairwise tile-tile attention the GNN's mean-pooled
message passing doesn't) while adding explicit topology priors for the
vertex / edge / hex incidence structure.

### 4.1 ISMCTS (module landed; rollout activation deferred)

**File:** `src/catan_rl/algorithms/search/ismcts.py`.

Single-step PUCT search over the 13 action types, using:

  - `policy.action_heads.type_head` for action priors,
  - `policy.get_value` for leaf evaluation,
  - `policy.belief_head` (when present) for opponent-card determinization.

`n_determinizations=4` separate search trees, each with `n_sims_per_det=50`,
sums to effective n_sims=200 via Cazenave averaging. `visits_to_distribution`
converts visit counts to a probability vector at a configurable temperature.

**What's deliberately not built:**

  - **Multi-step env clone.** True ISMCTS rolls out the env from a
    determinized state and uses the rollout's terminal value. The current
    `catanGame` engine is mutable, shares mutable state with the broadcast
    tracker, and lacks a copy API. Adding `CatanGame.copy()` is a sizable
    sub-project — it would require auditing every mutable field across
    `engine/{game,board,player,dice}.py` and the broadcast bus. Scoping
    it as a follow-up keeps this PR landing on green.
  - **Rollout-loop activation.** The roadmap gates ISMCTS on belief loss
    < 1.0 nats per token; we don't have empirical belief loss yet. The
    library is available; integrating a `mcts_prob` config that calls
    `ISMCTS.search` during `collect_rollouts` and uses the visit-count
    distribution as a CE target during PPO update lands once we have
    eval data showing the belief head is good enough.

The single-step search degenerates into pure PUCT-policy-improvement
(visit counts proportional to priors when leaf values don't differentiate
across actions). This is the same regime AlphaZero uses at *inference*
with shallow searches, so it's a meaningful policy-improvement signal —
just not as strong as multi-step lookahead would be.

## Consequences

- Param count: phase3_full ~2.24M → phase4_full ~2.74M. The +0.5M splits
  across the GNN encoder (~30k), recurrent value head (~150k from the
  GRU + wider value MLP), belief head (~70k), opp-action head (~70k),
  and the encoder fan-out from concatenating graph features into the
  fusion input.
- FPS impact: GNN encoder is the heaviest add — each forward pass runs
  2 rounds of mean-pooled message passing over 19+54+72 nodes. On M1
  Pro CPU, expect ~10-15% slower per rollout vs phase3_full.
- Phase 4 lineage stays incompatible with `checkpoint_07390040.pt` (Phase
  1 already broke it; subsequent phases compounded).
- All 4 features are config-flagged with leave-one-out configs.

## Alternatives considered

- **PBT (Phase 4.3):** hardware-gated. Requires 4× compute, can't run on
  M1 Pro alone. Documented in roadmap as future work.
- **Adding `CatanGame.copy()` to enable multi-step ISMCTS:** scoped as a
  follow-up. Estimated 2–3 days of careful engine work + tests covering
  the broadcast tracker + dice state + player buildGraph mutations.
- **Concatenating GRU hidden state INTO the obs encoder feature instead
  of being purely value-side:** would make the *policy* recurrent too.
  Roadmap explicitly says policy stays Markovian — recurrence on the
  policy makes the action-selection process non-stationary across episode
  steps in ways that complicate PPO's IS ratio. Stick with value-only.

## Follow-up work

- **CatanGame.copy() + multi-step ISMCTS rollouts.** The single biggest
  unlock for Phase 4.1's actual training value.
- **Rollout-loop integration of ISMCTS.** Once belief loss is verified
  < 1.0 nats: add `mcts_prob` config; call `ISMCTS.search` for that
  fraction of agent steps; use `visits_to_distribution` as a CE target
  on the type head during PPO update.
- **Phase 4.3 PBT** if multi-machine compute becomes available.
