# Broadcast-Based Perfect Opponent Hand Tracking Plan

This document outlines a plan to use the `GameBroadcast` system to maintain a **perfect** (deterministic) estimate of the opponent's resource hand and feed it into the RL model through the opponent observation module.

## Current State

- **GameBroadcast** exists in `catan/engine/broadcast.py` and emits events:
  - `DICE_ROLL` — who rolled, what value
  - `RESOURCE_CHANGE` — player, delta dict, source (DICE, BUILD_*, TRADE_BANK, ROB, DISCARD, YOP, etc.)
  - `DISCARD` — player, list of resources discarded
  - `YOP` — player, list of resources taken (Year of Plenty)

- **Emitters**: `catanGame.rollDice()`, `game.update_playerResources()`, `player.build_road()`, `player.build_settlement()`, etc., all emit `resource_change` events.

- **RL env**: Does NOT subscribe to broadcast. Observations use direct game state (`player.resources`, board) and min/max bucket encodings for the opponent.

## Why Perfect Tracking Is Possible (1v1, No P2P Trading)

In 1v1 Catan without player-to-player trading, every resource change is **observable**:

| Event | What We Know |
|-------|--------------|
| **DICE** | Roll value + opponent's settlement/city positions → exact resources gained |
| **BUILD_*** | Opponent built something → we see the build, know exact cost |
| **TRADE_BANK** | Opponent traded with bank → we see the trade (4:1 or port) |
| **ROB** | We stole from opponent → we know what we took |
| **DISCARD** | 7 was rolled → we see the discard list |
| **YOP** | Opponent played Year of Plenty → we see resources taken |
| **Initial** | Second settlement placement → we see adjacent hexes |

## Implementation Plan

### Phase 1: Broadcast Subscriber (Hand Tracker)

**File**: `catan/rl/hand_tracker.py` (new)

```python
class BroadcastHandTracker:
    """Subscribes to GameBroadcast and maintains exact resource counts per player."""
    
    def __init__(self, player_names: List[str]):
        self.hands = {name: {r: 0 for r in RESOURCES_CW} for name in player_names}
        self._callback = self._on_event
        
    def subscribe(self, broadcast: GameBroadcast):
        broadcast.subscribe(self._callback)
        
    def unsubscribe(self, broadcast: GameBroadcast):
        broadcast.unsubscribe(self._callback)
        
    def _on_event(self, event: dict):
        if event["type"] == "RESOURCE_CHANGE":
            name = event["player"]
            delta = event["delta"]
            for r, d in delta.items():
                self.hands[name][r] = max(0, self.hands[name][r] + d)
        elif event["type"] == "DISCARD":
            # DISCARD gives list of resources; we already get RESOURCE_CHANGE from game
            pass  # Optional: cross-check
        elif event["type"] == "YOP":
            pass  # Same: RESOURCE_CHANGE covers it
            
    def get_hand(self, player_name: str) -> Dict[str, int]:
        return self.hands[player_name].copy()
    
    def reset(self):
        for name in self.hands:
            for r in self.hands[name]:
                self.hands[name][r] = 0
```

**Initialization**: After second settlement placement, the game emits resource gains. The tracker must be subscribed **before** the first `update_playerResources` call. Wire it in `CatanEnv.reset()` after creating the game and before running setup.

### Phase 2: Seed Tracker from Initial Resources

The broadcast does not emit initial resources from second settlement. Options:

1. **Emit from game**: Add `broadcast.resource_change(player, delta, source="INITIAL")` in `catanGame` when granting initial resources after setup.
2. **Seed in env**: After setup, read `opponent_player.resources` once and set `tracker.hands["Opponent"]` accordingly. Then subscribe for all subsequent events.

Recommend **Option 2** for minimal engine changes: seed once, then rely on broadcast for the rest.

### Phase 3: Wire Tracker into CatanEnv

**In `CatanEnv.reset()`**:

```python
# After game + players created, before setup
self.hand_tracker = BroadcastHandTracker([self.agent_player.name, self.opponent_player.name])
self.hand_tracker.subscribe(self.game.broadcast)

# After setup (both passes, initial resources granted)
self.hand_tracker.hands[self.agent_player.name] = dict(self.agent_player.resources)
self.hand_tracker.hands[self.opponent_player.name] = dict(self.opponent_player.resources)
```

**In `CatanEnv._get_player_inputs("next")`**:

Replace the min/max bucket encoding for opponent resources with the **tracked hand**:

```python
# Current: min_res = bucket8(0), max_res = bucket8(19) for opponent
# New: use hand_tracker.get_hand(opponent.name)
tracked = self.hand_tracker.get_hand(target.name)
resources = np.concatenate([bucket8(tracked[r]) for r in RESOURCES_CW])
```

### Phase 4: Observation Dimension Change

**Current** `OTHER_PLAYER_MAIN_DIM = 173` uses:
- min_res (5×8) + max_res (5×8) + vp + res_access + lr + la + harbour + other_id (3) + num_hidden_oh (6) + dice_feats (12) + karma (2)

**With perfect tracking**, we can replace min/max with a single exact encoding:
- `exact_res` (5×8) — one bucket8 per resource from tracked hand
- Remove `max_res` (5×8) — saves 40 dims, or keep structure and zero out max

**Option A (minimal change)**: Use tracked hand for both min and max (they’re equal). `OTHER_PLAYER_MAIN_DIM` stays 173.

**Option B (cleaner)**: Introduce `opponent_tracked_resources` (5×8 = 40) and drop the redundant max_res. Adjust `OTHER_PLAYER_MAIN_DIM` and `OtherPlayersModule.main_input_dim` accordingly.

### Phase 5: OtherPlayersModule

`OtherPlayersModule` expects `main_input_dim` (currently 173). If we keep the same structure (min = max = tracked), no model change. If we change the layout, update `main_input_dim` and the concatenation order in `_get_player_inputs`.

### Phase 6: Edge Cases

1. **Robber steal**: When we steal, we learn the resource. `move_robber` → `resource_change(victim, delta, source="ROB")`. The tracker receives this. ✓

2. **Opponent’s discard**: `discardResources` triggers `log_discard` → `broadcast.discard` and `resource_change`. Tracker gets `RESOURCE_CHANGE`. ✓

3. **Opponent’s builds**: `player.build_road` etc. call `game.broadcast.resource_change`. Tracker gets it. ✓

4. **Opponent’s bank trade**: Player trade methods emit `resource_change` with source `TRADE_BANK`. ✓

5. **Dev cards**: Hidden. We still use `num_hidden_oh` (6-dim) for count. No change.

### Phase 7: Verification

Add a debug check (e.g. behind `verbose`):

```python
tracked = self.hand_tracker.get_hand(self.opponent_player.name)
actual = dict(self.opponent_player.resources)
assert tracked == actual, f"Tracker drift: {tracked} vs {actual}"
```

If this ever fails, there is an event path not covered by the broadcast.

## Summary Checklist

- [x] Create `BroadcastHandTracker` in `catan/rl/hand_tracker.py`
- [x] Subscribe tracker in `CatanEnv.reset()` after game creation
- [x] Seed tracker with initial resources after setup
- [x] Replace opponent min/max in `_get_player_inputs("next")` with tracked hand
- [x] Keep `OTHER_PLAYER_MAIN_DIM` = 173 (use tracked for both min and max)
- [x] Add optional verification via `verify_hand_tracker=True` env arg
- [x] Unsubscribe tracker in `close()` and on reset (before creating new tracker)

## Usage

Hand tracking is enabled by default. For debug/CI, enable verification:

```python
env = CatanEnv(verify_hand_tracker=True)  # Asserts tracked == actual on each get_hand
```

## Benefits

- **Perfect information** about opponent resources (in 1v1, no P2P trading)
- **No belief state** needed — deterministic
- **Better credit assignment** — model sees exact opponent hand, can reason about their options
- **Reuses existing broadcast** — no new event types required
