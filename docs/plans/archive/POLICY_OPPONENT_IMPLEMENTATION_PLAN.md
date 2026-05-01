# Policy Opponent Implementation Plan

**Goal:** Pure Charlesworth-style self-play—rollout training against past policies only, no random or heuristic opponents. Adapted for 1v1 Catan.

---

## Current State vs Target

| Aspect | Current | Target (Charlesworth) |
|--------|---------|------------------------|
| Rollout opponent | Random | Past policies (league) |
| Eval opponent | Random | Random (unchanged) |
| League usage | Populated, never sampled | Sample at every game reset |
| Setup (opponent) | Random/heuristic | Policy |
| Main game (opponent) | Random/heuristic | Policy (action loop) |

---

## Phase 1: Env — Observation & Masks for Any Player

**File:** `catan/rl/env.py`

### 1.1 `_get_obs_for_player(acting_player) -> Dict`

Observation from a given player's perspective. Currently `_get_obs()` assumes agent = current, opponent = next.

- **Tile features:** Swap "agent" vs "opponent" road ownership based on `acting_player`. In `_get_tile_features(player)`, treat `acting_player`'s roads as "us", the other as "them".
- **Player inputs:** `_get_player_inputs` uses "current"/"next" mapped to agent/opponent. Add `_get_player_inputs_for(target_player)` that returns (main, played, hidden) for that player. For obs: current = acting_player, next = other_player.
- **Current resources:** Use `acting_player.resources`.

**Dependencies:** `_get_tile_features` must accept `(current_player, other_player)` or `acting_player` and swap internally.

### 1.2 `get_action_masks_for_player(player) -> Dict`

Refactor `get_action_masks()` to take optional `player`; default `agent_player`. Replace all `p = self.agent_player` with `p = player or self.agent_player`. Masks are computed for whichever player is acting.

---

## Phase 2: Env — Action Execution for Any Player

**File:** `catan/rl/env.py`

### 2.1 `_execute_action_for_player(acting_player, action) -> Tuple[bool, bool, bool, dict]`

Extract the action execution logic from `step()` into a reusable method. Parameters:
- `acting_player`: The player taking the action (agent or opponent).
- `action`: Composite action array `(type, corner, edge, tile, res1, res2)`.

Returns: `(turn_ended, terminated, truncated, info)`.

**Logic to extract:** All branches in `step()` that mutate state (build settlement, build road, play dev card, etc.). Key swaps:
- `p` → `acting_player`
- `self.opponent_player` → `other_player` (the one we steal from in Monopoly, rob in robber, etc.)
- `other_player = self.agent_player if acting_player is self.opponent_player else self.opponent_player`

**Phases to handle:**
- Setup (settlement, road) — acting_player does placement
- Roll pending (roll, dev card before roll)
- Discard phase
- Road building phase (Road Builder card)
- Robber phase
- Normal actions (build, trade, end turn, etc.)

**Refactor:** `step()` becomes: validate mask → `_execute_action_for_player(agent_player, action)` → `_finish_step()`.

### 2.2 Handle `_finish_step` Side Effects

`_finish_step` and the logic after action execution trigger `_run_opponent_turn`, `_start_agent_turn`, etc. For `_execute_action_for_player`, we only want the state mutation. The "what happens next" (run opponent turn, etc.) stays in the caller. So `_execute_action_for_player` should:
- Mutate game state
- Return whether the acting player's turn ended, game terminated, or truncated
- Not call `_run_opponent_turn` or `_start_agent_turn` — that stays in `step()` and in the policy opponent loop

---

## Phase 3: Env — Policy Opponent Setup

**File:** `catan/rl/env.py`

### 3.1 Opponent Type "policy"

In `reset()`, when `opp_type == "policy"`:
- Require `options["opponent_policy"]` — a `CatanPolicy` instance (eval mode, loaded with league state dict).
- Create a minimal player object for the opponent (e.g. `player("Opponent", "darkslateblue")`) — we need something in `opponent_player` for game logic, but the *decisions* come from the policy.
- Store `self._opponent_policy = options["opponent_policy"]`.

### 3.2 `_do_opponent_setup_placement` for Policy

When `p is self.opponent_player` and we have `self._opponent_policy`:
- Build obs from opponent's perspective: `_get_obs_for_player(p)`
- Build masks: `get_action_masks_for_player(p)`
- For setup, masks only allow settlement (type 0) or road (type 2) depending on `placement_type`
- Call `self._opponent_policy.act(obs_t, masks_t, deterministic=True)`
- Execute via `_execute_action_for_player(p, action)` — but setup is a special case; we may need `_execute_setup_placement_for_player(p, action, placement_type)` since setup doesn't go through the full step flow.

**Simpler approach:** Add `_execute_setup_action_for_player(acting_player, action, placement_type)` that only handles setup settlement/road. The policy outputs the same action format; we validate against setup masks.

---

## Phase 4: Env — Policy Opponent Main Game

**File:** `catan/rl/env.py`

### 4.1 `_run_opponent_policy_turn() -> bool`

Replace the heuristic/random `_run_opponent_actions` with a policy-driven loop when `self._opponent_policy` is set.

**Flow:**
1. `_run_opponent_turn()` still does: roll dice, if 7 do opponent discard (policy chooses discard), set `robber_placement_pending` if 7.
2. Call `_run_opponent_policy_actions()` instead of `_run_opponent_actions()`.

**`_run_opponent_policy_actions(opponent) -> bool`:**
```
while True:
    if robber_placement_pending:
        obs = _get_obs_for_player(opponent)
        masks = get_action_masks_for_player(opponent)
        action = policy.act(obs, masks, deterministic=True)
        # Execute robber only (action type 4)
        _execute_action_for_player(opponent, action)  # or a robber-specific helper
        robber_placement_pending = False
        continue
    if discard_pending and opponent must discard:
        # Policy chooses discard (action type 11)
        ...
    obs = _get_obs_for_player(opponent)
    masks = get_action_masks_for_player(opponent)
    action = policy.act(obs, masks, deterministic=True)
    turn_ended, terminated, truncated, info = _execute_action_for_player(opponent, action)
    if turn_ended or terminated or truncated:
        break
return truncated
```

**Opponent discard (when opponent rolls 7):** When opponent rolls 7 and has >7 cards, we must discard. Currently `opponent.discardResources(self.game)` uses heuristic/random (RandomAIPlayer/heuristicAIPlayer override it). For policy opponent:
- Do NOT call `opponent.discardResources()`.
- Set `discard_pending=True`, `discard_remaining=total//2` for the opponent.
- Run a loop: get obs (discard phase from opponent's view), masks (resource1_discard), policy.act(), execute one discard (action type 11, res_idx). Repeat until `discard_remaining==0`.
- Then proceed to robber placement. Use `game.log_discard(opponent, [res_name])` to broadcast; the env's discard logic removes one resource per action.

---

## Phase 5: League — Pure Self-Play Mode

**File:** `catan/rl/ppo/league.py`

### 5.1 Add Initial Policy at Start

Charlesworth's `earlier_policies` starts with one copy of the initial policy. Before any training:
- `league.add(initial_policy.state_dict())` — so we have at least one policy to sample.

### 5.2 Pure Self-Play: No Random

- Set `league_random_weight = 0` in config when using policy opponents.
- `sample()` must always return `("policy", state_dict)`.
- If `len(policies) == 0`, raise or add initial policy first.

### 5.3 `set_build_policy_fn`

PPO must call `league.set_build_policy_fn(build_agent_model)` so the league can create policy instances for inference.

---

## Phase 6: GameManager — League Integration

**File:** `catan/rl/ppo/game_manager.py`

### 6.1 Constructor Changes

- Accept `league: League` and `build_policy_fn` (or get from league).
- Create envs with `opponent_type="policy"` (when league is used).
- Maintain `n_envs` opponent policy instances — one per env, for loading different state dicts.

### 6.2 `reset_all` with League Sampling

- For each env `i`: sample `(opp_type, state_dict) = league.sample()`.
- If `opp_type == "policy"`: load `state_dict` into `opponent_policy[i]`, call `env.reset(options={"opponent_type": "policy", "opponent_policy": opponent_policy[i]})`.
- If `opp_type == "random"` (fallback during transition): `env.reset(options={"opponent_type": "random"})`.

### 6.3 `step_one` Reset with Options

When a game ends, `step_one` calls `env.reset()`. It must pass the same options structure. So `step_one` needs to accept a way to get reset options — e.g. a callback `get_reset_options(env_idx) -> dict`. The PPO would pass a closure that samples from the league and returns options.

**Alternative:** GameManager holds a `sample_and_prepare_opponent(env_idx)` method. Before resetting env `env_idx`, it samples from league, loads into `opponent_policies[env_idx]`, and returns the options dict. `step_one` would call this when `done` before `env.reset(options=...)`.

---

## Phase 7: PPO — Wire League to Rollout

**File:** `catan/rl/ppo/ppo.py`

### 7.1 Add Initial Policy to League

Before the training loop (or in `__init__` after creating the policy):
- `self.league.add(copy.deepcopy(self.policy.state_dict()))`

### 7.2 GameManager with League

- Pass `league` and `build_policy_fn` to GameManager.
- GameManager creates envs with `opponent_type="policy"` and uses league for sampling at reset.

### 7.3 Config: Pure Self-Play

- `league_random_weight: 0` — no random opponents.
- Ensure `add_policy_every` and `league_maxlen` are set.

---

## Phase 8: Obs/Mask Format Compatibility

The policy expects a specific obs dict format (e.g. from `play_vs_model`). `_get_obs_for_player(opponent)` must produce the same structure, with "current" = opponent and "next" = agent. The policy's forward pass does not care which is agent vs opponent; it just needs consistent shapes. Verify that swapping current/next produces valid input for the policy.

---

## Implementation Order

1. **Phase 1** — `_get_obs_for_player`, `get_action_masks_for_player` (no behavior change yet).
2. **Phase 2** — `_execute_action_for_player` (refactor `step()` to use it; agent behavior unchanged).
3. **Phase 3** — Policy opponent setup in `_do_opponent_setup_placement`.
4. **Phase 4** — `_run_opponent_policy_actions` and policy opponent main game loop.
5. **Phase 5** — League: add initial policy, pure self-play mode.
6. **Phase 6** — GameManager: league sampling, reset options.
7. **Phase 7** — PPO: wire league, add initial policy, use policy opponents.

---

## Testing Strategy

1. **Unit test:** `_get_obs_for_player(opponent)` vs `_get_obs()` — swap agent/opponent, compare structure.
2. **Unit test:** `_execute_action_for_player(agent_player, action)` matches `step()` behavior for agent actions.
3. **Integration:** One game with `opponent_type="policy"`, policy = copy of current. Agent and opponent should behave identically; game should complete.
4. **Integration:** Short training run with `league_random_weight=0`, verify no random opponents, league sampling works.

---

## Edge Cases

- **Opponent discard (roll 7):** Policy must output discard actions. Discard phase may require multiple steps (discard N cards). Implement loop.
- **Road Builder:** Opponent plays Road Builder → 2 free roads. Same action loop; policy outputs road placements.
- **First game:** League has 1 policy (initial). Sample returns it. Opponent = self.
- **Deterministic:** Use `deterministic=True` for opponent policy to avoid stochasticity in evaluation.

---

## Files to Modify

| File | Changes |
|------|---------|
| `catan/rl/env.py` | `_get_obs_for_player`, `get_action_masks_for_player`, `_execute_action_for_player`, `_do_opponent_setup_placement` (policy), `_run_opponent_policy_actions`, `reset` (policy options) |
| `catan/rl/ppo/league.py` | `add_initial`, `random_weight=0` for pure self-play |
| `catan/rl/ppo/game_manager.py` | League, opponent policy pool, `reset_all`/`step_one` with options |
| `catan/rl/ppo/ppo.py` | Add initial to league, pass league to GameManager, config |
| `catan/rl/ppo/arguments.py` | `league_random_weight: 0` for pure self-play |
