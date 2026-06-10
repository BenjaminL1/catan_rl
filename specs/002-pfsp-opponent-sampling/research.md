# Research: PFSP Opponent Sampling

Design decisions resolving the spec's deferred choices. Format: Decision / Rationale / Alternatives.

## R1 — Win-rate estimator

**Decision**: Store integer `(wins, games)` per `snapshot_id`. Point estimate `p̂ = wins / games` (no smoothing in the estimate itself); the cold-start gate (`games < pfsp_min_games`) decides whether `p̂` is used at all.

**Rationale**: Integer counts are exactly resumable (no float drift across checkpoint round-trips → SC-004 is trivial), O(1) update, and decouple "is the estimate reliable yet?" (the games count) from "how hard is it?" (`p̂`). This matches AlphaStar-style PFSP bookkeeping.

**Alternatives**: EMA of outcomes — loses an explicit games count (still need one for cold-start) and introduces float drift that complicates exact resume. Beta-Bernoulli posterior mean `(wins+1)/(games+2)` — fine, but the +1/+2 smoothing muddies the cold-start gate (a 0-game snapshot reads 0.5, not "unknown"); we keep the raw ratio + an explicit gate instead.

## R2 — PFSP curve + no-starvation floor

**Decision**: `pfsp_curve ∈ {uniform, hard}`. `hard` weight `w_j = max(pfsp_floor, (1 − p̂_j) ** pfsp_k)` with a small fixed `pfsp_floor` (≈0.05) and `pfsp_k` default 1.0. `uniform` → constant weight (reproduces today's behaviour). Weights are normalised over the eligible pool.

**Rationale**: `(1−p̂)^k` up-weights opponents the agent loses to (the PFSP intent); `k` tunes sharpness. The `floor` keeps a fully-beaten opponent (`p̂=1`) at a small non-zero share (FR-008 no-starvation) so coverage doesn't collapse to only-hard opponents. At `p̂=0` the weight is `1` (finite, FR-008). Equal `p̂` across the pool → equal weights → uniform (no degeneracy, FR-008).

**Alternatives**: `p(1−p)` "even" curve (focus on balanced matchups) — a useful future curve but not needed for the drift fix; left as a documented extension point. Hard floor of 0 (true starvation) — rejected per FR-008.

## R3 — Outcome → opponent attribution under auto-reset (the subtle one)

**Decision**: The game that just ended belongs to the opponent **in force when it started**, which is the env's opponent *before* this step's auto-reset. `vec_env` maintains a per-env `current_opponent_id` (set at each `_reset_env` when a pending swap applies). The collector reads `pre = vec_env.current_opponent_ids()` **before** `step_all`, then for every env flagged terminated|truncated by that `step_all`, records the outcome against `pre[i]`.

**Rationale**: `set_opponents` defers the swap to the next reset, so a mid-rollout auto-reset is exactly where an env's opponent can change. Reading the opponent ids *before* the step captures the opponent the finishing game was actually played against (FR-002). Within a single rollout `set_opponents` is called once (at the start), so most envs keep one opponent all rollout; the pre-step read is still the correct general rule and is robust if the swap cadence ever changes. Only pool/anchor (snapshot-kind) ids are recorded; heuristic/random envs have no snapshot id and are skipped.

**Alternatives**: Attribute to the *post*-reset assignment — wrong (credits the next game's opponent). Have `step_all` return the pre-reset opponent ids — equivalent but a wider signature change; the pre-step accessor read is smaller and local to the collector.

## R4 — Win / loss determination at episode end

**Decision**: `agent_won = terminated AND agent reached 15 VP` (a true win). `truncated` (max-turns) and opponent-reached-15 both count as **not a win** (a loss for WR purposes). Source the win flag from the terminal signal already available at episode end (the env's terminal reward sign / the agent-vs-opponent VP at termination) — confirm the exact terminal-reward convention during Increment 1 and assert it in a test.

**Rationale**: PFSP needs a binary win/loss per game; the agent's objective is reaching 15 first. Truncations are rare with a competent policy and counting them as non-wins is conservative (doesn't inflate WR). One outcome per finished game keeps it O(n_envs).

**Alternatives**: Count truncations as 0.5 (draw) — adds fractional-win bookkeeping for a rare case; rejected for simplicity. Use raw reward magnitude — unnecessary; the binary win is what the curve needs.

## R5 — Persistence (resume)

**Decision**: Extend the checkpoint league-capture (`_capture_league_state`) with `opponent_stats: {snapshot_id: [wins, games]}` and restore it in `apply_to_league`. Old checkpoints lacking the field restore to an empty store (additive, Principle III). The anchor's own stats are captured too (its id is stable).

**Rationale**: Integer counts serialise trivially and exactly (SC-004). Additive field = no migration, old checkpoints still load.

**Alternatives**: Recompute WR from scratch on resume — loses history, defeats the point on a resumed long run.

## R6 — Cold-start

**Decision**: A snapshot with `games < pfsp_min_games` (default ≈5–10) gets weight `pfsp_cold_start_weight` (default = the max eligible pool weight, so new snapshots are eagerly tried) instead of a curve weight, then transitions to the curve once it has enough games.

**Rationale**: New snapshots (the agent's latest selves) must be played to accrue data (FR-004 / SC-002); seeding them at the top weight guarantees early trials without a separate scheduler.

**Alternatives**: Treat 0-game snapshots as `p̂=0.5` — under the hard curve that's a middling weight, risking starvation behind genuinely-hard opponents; the explicit cold-start weight is stronger and clearer.

## R7 — Off-by-default byte-identity

**Decision**: `pfsp_enabled=False` (default) and `pfsp_curve="uniform"` both route the pool draw through the existing `rng.choice(pool_ids)` path unchanged; the category-level weights (random/heuristic/pool/anchor) are untouched by PFSP in all cases.

**Rationale**: FR-005 / SC-003 — existing runs (including the live v4 anchored run) must be unaffected. Guarding the new path behind the flag + curve keeps the default code path identical.

## Plan-review resolutions (folded in before implementation)

Senior-RL plan review found 3 BLOCKERs + 2 SHOULD-FIX; the design is updated as:

- **[B1] Terminal win signal must be plumbed out of the vec env.** `step_all` currently drops `info` and auto-resets before returning, so the collector cannot read the win post-hoc. Resolution: `step_all` (or `final_obs`) additionally returns per-terminated-env **terminal info** captured BEFORE auto-reset — `{agent_vp, opp_vp}` (and/or an `is_success`/`won` flag). The collector reads this, never post-reset env state.
- **[B2] Per-inner-step attribution.** `current_opponent_ids()` is captured immediately BEFORE EACH `step_all` inside the n_steps loop (not once per rollout), because envs auto-reset multiple times per `collect()`.
- **[B3] Latch the TRUE snapshot id.** vec_env stores the real `snapshot_id` per env at each `_reset_env` (the existing `opponent_policy_id` is a lossy `% N` and is NOT invertible). `current_opponent_ids()` returns those true ids.
- **[SF4 — supersedes R1] Recency-weighted win rate.** Replace lifetime `wins/games` with an **EMA** of the per-game outcome (`p̂ ← (1−α)·p̂ + α·won`, α default ≈ 0.1) plus an integer `games` count used only for the cold-start gate. EMA tracks the *moving* learner (a self rated easy long ago becomes hard again as the policy drifts) — lifetime counts would stay stale and re-admit the drift PFSP exists to stop. Both `p̂` (float) and `games` (int) serialise exactly → resume still bit-stable.
- **[SF5 — supersedes R4] Win by VP margin at episode end.** `won = agent_vp > opp_vp` at termination OR truncation (not "terminated-and-reached-15, truncation=loss"). This avoids spuriously rating strong opponents "hard" via stalemate truncations and reuses B1's terminal `{agent_vp, opp_vp}`.
- **[CONSIDER] Evicted-stats pruning** (drop a snapshot's EMA/games when it leaves the FIFO deque, so the store stays ≤ maxlen+1); **scope note**: PFSP reweights only within the retained pool — forgetting beyond `maxlen` snapshots is covered by the anchor, not PFSP; optional **TB diagnostic** `pfsp/sampling_entropy` so a collapse-to-uniform is visible; keep the PFSP-off path's exact per-env `rng.choice` call order for byte-identity.
