# Phase 0 Research: self-play snapshot opponent

No open `NEEDS CLARIFICATION` remained from the spec; the decisions below resolve
the design choices the implementation must commit to.

## D1 — Opponent inference: batched, main-process

**Decision**: Run the snapshot opponent's policy forward **batched across the N
rollout envs in the main process**, inference-only (`torch.no_grad`, `eval()`),
on the learner device.

**Rationale**: The measured v2 training bottleneck is the small-batch policy
forward pass; a per-env batch-of-one snapshot forward would multiply that cost by
N. Batching the opponent forward across envs in the main process amortizes it.
(This is a version-independent performance pattern — the implementation is new
v2 code, not a v1 dependency.)

**Alternatives rejected**: per-env in-env inference (batch=1 → slow); subprocess
opponents (IPC pickling of obs dicts erases the win, per the vec-env memory note).

## D2 — Mid-rollout opponent swap

**Decision**: Add `vec_env.set_opponents(assignments)` that threads each env's
opponent kind + snapshot id into its **next `reset`**; the training loop calls it
once per rollout from `league.build_env_opponent_mix(...)`.

**Rationale**: The opponent mix is locked at construction (`training_loop.py:201`,
comment ":16-17"); newly-added snapshots can never enter play without a refresh.
The env already accepts opponent params at reset.

**Alternatives rejected**: rebuild the whole vec env each rollout (expensive);
mutate opponents mid-episode (breaks in-flight games / determinism).

## D3 — Policy-vs-policy eval reuse

**Decision**: Build the opponent for eval with `replay/player_factory.build_actor`
(→ `_PolicyActor`, which loads a checkpoint strictly and delegates to
`policy.sample`). Add one eval entry point that pits champion vs a loaded actor
over N seat-symmetrized games and returns WR + Wilson CI (reuse the existing
`eval/wilson.py`).

**Rationale**: `build_actor` already does strict checkpoint loading + sampling;
writing a second loader duplicates logic and risks shape-check drift.

**Alternatives rejected**: a new eval-only policy loader (duplication).

## D4 — Opponent identity, no shape change

**Decision**: Feed the existing `_OppIdEmbedding` (`network.py:36-52`) the real
opponent kind/id for the current opponent; do **not** resize the embeddings
(they already size to `N_OPP_KINDS` / `N_OPP_POLICY_SLOTS`, defaulting to
`UNKNOWN`).

**Rationale**: Preserves policy state-dict shape → `bootstrap_v1` and all archived
checkpoints stay loadable (Constitution III).

**Alternatives rejected**: adding/resizing an embedding (breaks back-compat).

## D5 — Empty-league / evicted-snapshot fallback

**Decision**: When `snapshot_weight>0` but the pool is empty (true at self-play
start) or a requested id has been evicted, fall back to a non-snapshot opponent
(heuristic/random per the remaining mix weights). Never raise.

**Rationale**: The pool is empty until the first `add_snapshot`; erroring would
make self-play impossible to bootstrap.

## D6 — Frozen-policy lifecycle & caching

**Decision**: Instantiate one frozen `CatanPolicy` per distinct in-play snapshot
id, load its state-dict once, set `eval()`, and cache it for the rollout;
refresh the cache when `set_opponents` changes the assignment. Never put a
snapshot policy in the optimizer or call `.backward()` on it.

**Rationale**: Avoids reloading state-dicts every step; keeps opponent strictly
inference-only.

## D7 — Determinism scope

**Decision**: Eval determinism is **bit-for-bit on CPU at a fixed seed** (FR/SC
on the eval path). Rollout snapshot-opponent determinism is **same action
sequence given the same seed + snapshot id on a fixed device** (low-order FP
bits may vary across MPS batch groupings).

**Rationale**: Batched MPS reductions are not bit-reproducible across batch
groupings; over-claiming would make acceptance tests flaky for reasons unrelated
to the feature (per spec review).

## 2026-06-08 revisions (senior-RL review + clarify pass)

## D8 — Shared `_apply_action(player, action)`
**Decision**: Extract the agent's inlined apply logic from `step()` into one
player-parameterized helper used by both seats; no duplicate opponent path.
**Rationale**: one code path = no rules drift (Constitution II). Gated by a
behavior-identical regression test.

## D9 — Opponent-POV obs/masks (no info leak)
**Decision**: Build the opponent's obs from the OPPONENT's perspective (its own
hidden dev cards visible, only the agent's *played* cards, hand-tracker from the
opponent's view) with an opponent-local `EnvObsState` — do NOT reuse the env's
agent-centric `roll_pending`/`discard_pending` flags. **Rationale**: the obs
encoder hardcodes agent-POV dev-card visibility (`obs_encoder.py:317`); naive
reuse leaks the learner's hidden hand → self-play-voiding bug. Gated by a no-leak
assertion test (FR-012).

## D10 — Opponent sampling: stochastic, isolated RNG (refines D7)
**Decision**: Sample stochastically with a dedicated `torch.Generator` seeded
from the env/league seed. **Rationale**: faithful to the policy's distribution
AND does not advance the learner's rollout RNG (greedy = less faithful/more
exploitable; shared RNG = breaks learner reproducibility).

## D11 — Turn completion + US2 seating
**Decision**: Opponent main turn runs to completion within one `env.step` + a
hard action cap (force EndTurn + log on exceed); the 7-roll/agent-discard
interleave keeps the existing `_opp_pending` suspension. Policy-vs-policy eval
seats the opponent via THIS driver — `build_actor` only *loads* a checkpoint into
a frozen `CatanPolicy`, it does NOT drive the opponent seat. So **US2 depends on
US1**, and `EvalMatchupResult` extends `EvalResult` (review SHOULD-FIX 5).
