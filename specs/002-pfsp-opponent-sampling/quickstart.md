# Quickstart / Validation: PFSP Opponent Sampling

Runnable checks that prove the feature. Unit/integration tests are the gate; SC-005 needs a training run.

## Unit (fast, CPU)
1. **WR tracking** (`tests/unit/selfplay/test_league.py`): `record_outcome` updates `(wins, games)`; `opponent_win_rate` reflects it; an evicted snapshot's stats are dropped.
2. **Hard-curve ordering (SC-001)**: a snapshot at p̂=0.3 is drawn ≥2× as often as one at p̂=0.7 over a large `build_env_opponent_assignments` draw (`pfsp_enabled=True, curve="hard"`).
3. **Cold-start (SC-002)**: a snapshot with 0 games gets a non-zero share alongside established snapshots.
4. **Degenerate (FR-008)**: equal WRs → ~uniform; p̂=1 still drawn (floor), p̂=0 finite; no NaN.
5. **Off byte-identity (SC-003)**: with `pfsp_enabled=False` (and with `curve="uniform"`), assignments for a fixed seed equal the pre-PFSP output.
6. **Attribution under auto-reset** (`tests/unit/ppo/test_game_manager.py`): with a stub vec env whose `current_opponent_ids` changes at a mid-rollout reset, the finishing game is recorded against the PRE-step opponent id.
7. **Config validation** (`test_arguments.py`): bad `pfsp_curve`, `pfsp_k<=0`, `pfsp_min_games<1` rejected; defaults keep PFSP off.
8. **Persistence round-trip (SC-004)** (`tests/unit/checkpoint/...`): capture → load reproduces `opponent_stats` exactly; an old checkpoint (no field) loads to empty.

Run: `python -m pytest tests/unit/selfplay/test_league.py tests/unit/ppo/test_game_manager.py tests/unit/ppo/test_arguments.py -q`

## Integration (CPU smoke)
9. **PFSP-on rollout updates WR** (`tests/integration/test_selfplay_smoke.py`): a tiny run (n_envs=2, snapshot_weight>0, pfsp_enabled=True, a seeded league) completes a rollout and the league's `opponent_stats` games count increased.
10. **PFSP-off byte-identity smoke**: the same tiny run with `pfsp_enabled=False` produces the same opponent assignment sequence as before the change.

Run: `python -m pytest tests/integration/test_selfplay_smoke.py -q`

## Training-run validation (SC-005, manual)
- Launch a self-play run (seed from u299, anchor on, `pfsp_enabled=True, pfsp_curve="hard"`) and track strength-vs-frozen-u799 across checkpoints. PASS if it does NOT drop ≥0.20 from its peak the way v2/v3 did (holds a tighter band or climbs), with the anchor + heuristic floor intact and per-update wall-clock within ~2% of PFSP-off (SC-006).
