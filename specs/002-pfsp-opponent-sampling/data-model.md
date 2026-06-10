# Data Model: PFSP Opponent Sampling

## OpponentStats (per snapshot)
- **Key**: `snapshot_id: int` (the league's stable monotonic id; the frozen anchor's id is included).
- **Fields**: `wins: int`, `games: int` (agent wins / total finished games vs this snapshot).
- **Derived**: `win_rate = wins / games` (only consulted when `games >= pfsp_min_games`).
- **Lifecycle**: created lazily on first `record_outcome` for an id; **evicted with its snapshot** (when a pool snapshot leaves the FIFO deque, its stats entry is dropped). The anchor's entry persists with the anchor.
- **Validation**: `0 <= wins <= games`; counts are non-negative ints.
- **Persistence**: serialised in the checkpoint as `opponent_stats: {snapshot_id: [wins, games]}`; absent ⇒ empty store (old checkpoints).

## PFSP configuration (LeagueConfig additions)
- `pfsp_enabled: bool = False` — master switch; off ⇒ uniform pool draw (byte-identical).
- `pfsp_curve: str = "uniform"` — one of `{"uniform", "hard"}`. `uniform` reproduces today's behaviour even when enabled.
- `pfsp_k: float = 1.0` — sharpness exponent for the `hard` curve `(1 − p̂)^k`. Validation: `> 0`.
- `pfsp_min_games: int = 5` — cold-start threshold; below it a snapshot uses `pfsp_cold_start_weight`. Validation: `>= 1`.
- `pfsp_cold_start_weight: float = 1.0` — weight for under-sampled snapshots (relative to the curve's max ≈1.0, so they are eagerly tried). Validation: `> 0`.
- (internal constant) `pfsp_floor ≈ 0.05` — minimum per-snapshot weight under `hard`, preventing starvation at `p̂=1`.

Validation interactions: PFSP fields are validated whenever set; `pfsp_enabled=True` with `pfsp_curve="uniform"` is legal (a no-op-ish enable). PFSP does not change the existing weight-sum/heuristic-floor invariants.

## Relationships
- `OpponentStats` is owned by `League`, parallel to the snapshot deque + the anchor, keyed by the same `snapshot_id` space.
- `build_env_opponent_assignments` reads `OpponentStats` + `LeagueConfig` PFSP fields to weight the **pool** category only; the anchor/heuristic/random category weights are unchanged.
- `record_outcome` is the only writer, called by the rollout collector with the attributed `(snapshot_id, agent_won)`.
