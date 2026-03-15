"""
Broadcast-based hand tracker for perfect opponent resource tracking.

Subscribes to GameBroadcast RESOURCE_CHANGE events and maintains exact
resource counts per player. In 1v1 without player-to-player trading,
every opponent resource change is observable via broadcast, enabling
deterministic (perfect) hand tracking.

Usage:
    tracker = BroadcastHandTracker(player_names=["Agent", "Opponent"])
    tracker.subscribe(game.broadcast)
    tracker.seed_from_player(agent_player)
    tracker.seed_from_player(opponent_player)
    # ... game runs, tracker receives events ...
    hand = tracker.get_hand("Opponent")
"""
from typing import Any, Dict, List, Optional

# Charlesworth order for player inputs (Wood, Brick, Wheat, Ore, Sheep)
RESOURCES_CW = ['WOOD', 'BRICK', 'WHEAT', 'ORE', 'SHEEP']


class BroadcastHandTracker:
    """Maintains exact resource counts per player by subscribing to GameBroadcast.

    In 1v1 Catan without P2P trading, all resource changes are emitted:
    - DICE: game.update_playerResources emits per-player deltas
    - BUILD_*: player.build_road/settlement/city emit
    - TRADE_BANK: player.trade_with_bank emits
    - ROB/STEAL: player.move_robber emits for victim and thief
    - DISCARD: game.log_discard emits
    - YOP: game logs year_of_plenty emits
    """

    def __init__(self, player_names: List[str], verify: bool = False):
        """Create a hand tracker for the given player names.

        Args:
            player_names: List of player names (e.g. ["Agent", "Opponent"]).
            verify: If True, assert tracker matches actual on get_hand (debug only).
        """
        self._hands: Dict[str, Dict[str, int]] = {
            name: {r: 0 for r in RESOURCES_CW} for name in player_names
        }
        self._verify = verify
        self._broadcast = None
        self._callback = self._on_event

    def subscribe(self, broadcast: Any) -> None:
        """Register to receive RESOURCE_CHANGE events.

        Args:
            broadcast: GameBroadcast instance from catanGame.
        """
        if self._broadcast is not None:
            self.unsubscribe()
        self._broadcast = broadcast
        broadcast.subscribe(self._callback)

    def unsubscribe(self) -> None:
        """Stop receiving events. Safe to call if not subscribed."""
        if self._broadcast is not None:
            self._broadcast.unsubscribe(self._callback)
            self._broadcast = None

    def _on_event(self, event: Dict[str, Any]) -> None:
        """Process broadcast events. Only RESOURCE_CHANGE updates hands."""
        if event.get("type") != "RESOURCE_CHANGE":
            return
        name = event.get("player")
        delta = event.get("delta", {})
        if name is None or name not in self._hands:
            return
        for r, d in delta.items():
            if r in self._hands[name]:
                self._hands[name][r] = max(0, self._hands[name][r] + d)

    def seed_from_player(self, player: Any) -> None:
        """Initialize hand from a player's current resources.

        Call after setup (initial resources granted) and before any
        broadcast events that affect this player. Typically used once
        per reset.

        Args:
            player: Player object with .name and .resources dict.
        """
        name = getattr(player, "name", None)
        if name is None or name not in self._hands:
            return
        res = getattr(player, "resources", {})
        for r in RESOURCES_CW:
            self._hands[name][r] = int(res.get(r, 0))

    def get_hand(self, player_name: str, actual: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """Return a copy of the tracked hand for the given player.

        Args:
            player_name: Name of the player.
            actual: If provided and verify=True, assert tracked == actual.

        Returns:
            Dict mapping resource name (RESOURCES_CW order) to count.
        """
        if player_name not in self._hands:
            return {r: 0 for r in RESOURCES_CW}
        hand = self._hands[player_name].copy()
        if self._verify and actual is not None:
            for r in RESOURCES_CW:
                if hand.get(r, 0) != actual.get(r, 0):
                    raise AssertionError(
                        f"HandTracker drift for {player_name}: "
                        f"tracked={hand} vs actual={actual}"
                    )
        return hand

    def reset(self) -> None:
        """Zero all hands. Call when starting a new episode."""
        for name in self._hands:
            for r in RESOURCES_CW:
                self._hands[name][r] = 0
