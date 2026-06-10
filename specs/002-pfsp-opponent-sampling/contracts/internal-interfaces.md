# Internal Interface Contracts: PFSP

All internal (no external/CLI surface). Signatures are the contract; behaviour notes are testable.

## League (src/catan_rl/selfplay/league.py)

```python
def record_outcome(self, snapshot_id: int, *, agent_won: bool) -> None
```
- Increment `games` (and `wins` if `agent_won`) for `snapshot_id`. Creates the entry lazily. No-op for ids not in the pool or anchor (defensively ignore stale ids). O(1).

```python
def opponent_win_rate(self, snapshot_id: int) -> tuple[int, int]   # (wins, games)
```
- Read accessor for tests/diagnostics. Returns `(0, 0)` for an unseen id.

```python
def build_env_opponent_assignments(self, *, n_envs, rng) -> list[OpponentAssignment]   # MODIFIED
```
- Unchanged category draw (random/heuristic/pool/anchor). Within the **pool** category: if `cfg.pfsp_enabled and cfg.pfsp_curve != "uniform"`, draw pool snapshot ids weighted by `_pfsp_weights(pool_ids)`; else uniform `rng.choice(pool_ids)` (byte-identical to current). Anchor id selection unchanged.

```python
def _pfsp_weights(self, ids: list[int]) -> np.ndarray   # NEW (private)
```
- Per-id weight: `cold_start_weight` if `games < pfsp_min_games`, else `max(pfsp_floor, (1 - wins/games) ** pfsp_k)` for `"hard"`. Normalised. Equal inputs → equal outputs.

State capture (consumed by checkpoint manager):
```python
def opponent_stats_state(self) -> dict[int, tuple[int, int]]      # NEW: {snapshot_id: (wins, games)}
def load_opponent_stats(self, state: Mapping[int, tuple[int, int]]) -> None   # NEW
```

## SerialVecEnv (src/catan_rl/ppo/vec_env.py)

```python
def current_opponent_ids(self) -> list[int | None]   # NEW
```
- Per-env snapshot_id of the opponent the env is CURRENTLY playing (the in-progress game), or `None` for heuristic/random envs. Updated when `_reset_env` applies a pending snapshot swap. Read by the collector BEFORE `step_all` to attribute the finishing game.

## RolloutCollector (src/catan_rl/ppo/game_manager.py)

- In `collect`, when the buffer/league is PFSP-active: capture `pre_ids = self.vec_env.current_opponent_ids()` before `step_all`; after `step_all`, for each env `i` with `terminated[i] or truncated[i]` and `pre_ids[i] is not None`, call `self.league.record_outcome(pre_ids[i], agent_won=<win flag from the terminal signal>)`.
- Gated so a non-PFSP / no-league collector is unchanged (the collector gains an optional `league` handle; `None` ⇒ no attribution).
- Win flag: derived from the env's terminal outcome (terminated + agent reached 15). Confirm the exact terminal-reward/VP convention in Increment 1; assert it.

## CheckpointManager (src/catan_rl/checkpoint/manager.py)

- `_capture_league_state(league)` → additionally store `opponent_stats = league.opponent_stats_state()`.
- `apply_to_league(league, state)` → if `state` has `opponent_stats`, `league.load_opponent_stats(...)`; absent ⇒ leave empty (old-checkpoint fallback).

## LeagueConfig (src/catan_rl/ppo/arguments.py)

- New fields per data-model.md with `__post_init__` validation (`pfsp_k>0`, `pfsp_min_games>=1`, `pfsp_cold_start_weight>0`, `pfsp_curve in {"uniform","hard"}`). Defaults keep PFSP off + behaviour unchanged.
