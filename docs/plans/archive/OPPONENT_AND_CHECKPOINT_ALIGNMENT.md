# Opponent & Checkpoint Nomenclature

## Summary

**Your model has been training against policy opponents (League), not random.** The `OpponentManager` was dead code—never imported or used. Training uses the in-memory `League` for opponent sampling.

## Checkpoint Layout

| Path | Purpose |
|------|---------|
| `checkpoints/train/` | Training checkpoint directory (config: `checkpoint_dir`) |
| `checkpoints/train/checkpoint_*.pt` | Full trainer checkpoints (policy, optimizer, step, etc.) |
| `checkpoints/train/interrupt_*.pt` | Interrupt checkpoints (same format, for resume) |
| `checkpoints/train/checkpoint_opponent_*.pt` | Optional policy-only snapshots (from OpponentManager.save_snapshot) |

## Opponent Sources

### League (used during training)

- **Location**: `catan/rl/ppo/league.py`
- **Storage**: In-memory `deque` of `state_dict` copies
- **When**: Always used when `GameManager` has a league with policies
- **Sampling**: `league.sample()` → `("policy", state_dict)` or `("random", None)` when `league_random_weight > 0`
- **Config**: `league_random_weight=0.0` → pure self-play (no random)
- **Additions**: Every `add_policy_every` (4) updates via `league.maybe_add()`

### OpponentManager (optional, disk-based)

- **Location**: `catan/rl/ppo/opponent_manager.py`
- **Storage**: Reads from `checkpoints/train/checkpoint_*.pt`
- **When**: Not used by training; for scripts that need disk-based opponent loading
- **Methods**: `get_pool_paths()`, `sample_opponent_path()`, `save_snapshot()`

## Training Flow

1. `CatanPPO` creates `League` and adds the initial policy.
2. `GameManager` uses the league and creates envs with `opponent_type="policy"`.
3. On each reset: `_sample_and_prepare_opponent()` → `league.sample()` → load `state_dict` into `_opponent_policies[i]` → `env.reset(options={"opponent_type": "policy", "opponent_policy": ...})`.
4. With `league_random_weight=0`, sampling is always from the league (policy opponents).

## Evaluation

- **EvaluationManager** uses `opponent_type="random"` or `"heuristic"` (from config or `evaluate.py`).
- **Env** supports `"policy"`, `"random"`, and `"heuristic"` opponent types.

## Naming Alignment

| Concept | Old (confusing) | New (aligned) |
|---------|-----------------|----------------|
| Opponent pool (training) | N/A | League (in-memory) |
| Opponent pool (disk) | OpponentManager, `checkpoints/opponents`, `opponent_*.pt` | OpponentManager, `checkpoints/train`, `checkpoint_*.pt` |
| Training checkpoints | `checkpoints/train/checkpoint_*.pt` | Same |
