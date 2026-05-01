"""
Centralised game event broadcaster.

This class encapsulates all \"broadcast\" logic so that code which
produces events (engine, players) doesn't need to know who is
listening. Consumers (RL env, loggers, trackers) can subscribe
callbacks to receive structured event dicts.

Events are simple dicts with at least a 'type' key, plus any
additional payload fields.
"""

from collections.abc import Callable
from typing import Any

GameEvent = dict[str, Any]
Subscriber = Callable[[GameEvent], None]


class GameBroadcast:
    def __init__(self) -> None:
        self.last_event: GameEvent | None = None
        self._subscribers: list[Subscriber] = []

    # Subscription API -------------------------------------------------

    def subscribe(self, callback: Subscriber) -> None:
        """Register a callback to receive every emitted event."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        """Remove a previously-registered callback (no-op if not present)."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # Core emit / reset API --------------------------------------------

    def clear_last(self) -> None:
        """Forget the most recent event (does not affect subscribers)."""
        self.last_event = None

    def emit(self, event_type: str, **payload: Any) -> GameEvent:
        """Emit a generic event and notify all subscribers."""
        event: GameEvent = {"type": event_type, **payload}
        self.last_event = event
        for cb in list(self._subscribers):
            cb(event)
        return event

    # Convenience helpers for common game events -----------------------

    def dice_roll(self, roller_name: str, value: int) -> GameEvent:
        return self.emit("DICE_ROLL", player=roller_name, value=value)

    def discard(self, player_name: str, resources: list[str]) -> GameEvent:
        # resources: list like ['BRICK', 'WOOD', ...] that were discarded
        return self.emit("DISCARD", player=player_name, resources=resources)

    def year_of_plenty(self, player_name: str, resources: list[str]) -> GameEvent:
        # resources: list like ['BRICK', 'ORE'] that were taken from bank
        return self.emit("YOP", player=player_name, resources=resources)

    def resource_change(
        self,
        player_name: str,
        delta: dict[str, int],
        source: str,
    ) -> GameEvent:
        """Generic resource delta event for future hand-tracking.

        Args:
            player_name: whose hand changed
            delta: dict mapping resource name → signed change
            source: short tag like 'BUILD_ROAD', 'TRADE_BANK', 'ROB', etc.
        """
        return self.emit("RESOURCE_CHANGE", player=player_name, delta=delta, source=source)
