# Phase 1 Contracts: internal interfaces

This is an internal library — the "contracts" are the new/changed Python
interfaces at the named seams. Signatures are indicative; tests assert behavior,
not exact shapes.

## 1. Frozen in-env opponent (env)

`catan_env.py` — `_run_opponent_main_turn` gains a snapshot branch. The env is
given an opponent move source for `opponent_type == "snapshot"`:

```python
# Injected callback (set when the env is assigned a snapshot opponent):
OpponentActFn = Callable[[ObsDict, MaskDict], np.ndarray]  # (obs, masks) -> action

# Behavior contract:
# - When opponent_type == "snapshot", the opponent turn is driven by encoding the
#   opponent's POV obs + masks and calling OpponentActFn (NOT opp.move()).
# - The NotImplementedError at catan_env.py:181 is removed.
# - 1v1 ruleset, reward, action space unchanged.
```

## 2. League consumer (selfplay)

```python
# league.build_env_opponent_mix(n_envs, rng) -> list[OpponentAssignment]
#   - May return kind="snapshot" with a concrete snapshot_id sampled
#     uniformly from the pool (PFSP deferred).
#   - If pool empty / snapshot_weight==0: returns only heuristic/random.
# - The NotImplementedError at league.py:245 is removed; sampling a snapshot
#   against a non-empty pool now succeeds.
```

## 3. Mid-rollout opponent swap (vec env)

```python
# vec_env.set_opponents(assignments: list[OpponentAssignment]) -> None
#   - Threads each env's (kind, snapshot_id) into its NEXT reset.
#   - Called once per rollout by the training loop.
#   - Snapshot opponent inference is batched across envs in the main process
#     (no per-env batch=1 forward).
```

## 4. Policy-vs-policy eval (eval)

```python
# eval/harness.py
# evaluate_policy_vs_policy(
#     champion: CatanPolicy,
#     opponent_ref: str | int,     # checkpoint path or snapshot_id
#     n_games: int,
#     seed: int,
#     device="cpu",
# ) -> EvalMatchupResult
#   - Builds the opponent via replay/player_factory.build_actor (strict load).
#   - Plays N seat-symmetrized games; returns WR + Wilson CI.
#   - Bit-for-bit reproducible at fixed seed on CPU.
#   - Emits TB scalar eval/wr_vs_<opp> (additive; never renames existing scalars).
```

## 5. Config (configs/ppo_default.yaml)

```yaml
league:
  snapshot_weight: 0.0        # >0 now WORKS (no NotImplementedError)
  # new: static opponent mix, no schedule
  heuristic_weight: ...       # the heuristic:snapshot split is configurable
```

## Invariants asserted by tests

- No policy state-dict shape change (load `bootstrap_v1` after the change → OK).
- Snapshot opponent never enters the optimizer / no grad flows through it.
- Empty-pool fallback never raises.
