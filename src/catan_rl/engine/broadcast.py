"""
Centralised game event broadcaster.

This class encapsulates all \"broadcast\" logic so that code which
produces events (engine, players) doesn't need to know who is
listening. Consumers (RL env, loggers, trackers) can subscribe
callbacks to receive structured event dicts.

Events are simple dicts with at least a ``"type"`` key, plus any
additional payload fields. The :class:`BroadcastEventType` enum
pins every emit type string used by the engine — string-typo drift
between emitters and consumers is then caught at import time rather
than at runtime.

Phase 0.5 of the replay-system build added 7 emitters
(``MONOPOLY``, ``MOVE_ROBBER``, ``STEAL``, ``BUILD``,
``LONGEST_ROAD_CHANGE``, ``LARGEST_ARMY_CHANGE``, ``GAME_END``)
alongside the legacy ``DICE_ROLL`` / ``DISCARD`` / ``YOP`` /
``RESOURCE_CHANGE``. The replay recorder subscribes to all of them;
the existing hand-tracker / belief-tracker code paths only need the
legacy four and ignore the new ones.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

GameEvent = dict[str, Any]
Subscriber = Callable[[GameEvent], None]


class BroadcastEventType(str, Enum):
    """Every event type the engine emits.

    The enum members are ``str`` subclasses so an event dict can be
    constructed as ``{"type": BroadcastEventType.MONOPOLY, ...}`` and
    consumers can still compare against the plain string
    ``"MONOPOLY"`` for back-compat. Keep this set in sync with the
    replay system's :data:`catan_rl.replay.EVENT_REGISTRY` discriminator
    strings — :mod:`tests.unit.engine.test_broadcast_emitters` asserts
    the two are aligned.
    """

    # Legacy emitters (predate Phase 0.5; the four original event types).
    DICE_ROLL = "DICE_ROLL"
    DISCARD = "DISCARD"
    YOP = "YOP"
    RESOURCE_CHANGE = "RESOURCE_CHANGE"

    # Phase 0.5 additions — the recorder's StepEvent variants get their
    # raw material from these. ``BUILD`` is a generic emit; the
    # ``kind`` payload field distinguishes settlement / city / road.
    MONOPOLY = "MONOPOLY"
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL = "STEAL"
    BUILD = "BUILD"
    LONGEST_ROAD_CHANGE = "LONGEST_ROAD_CHANGE"
    LARGEST_ARMY_CHANGE = "LARGEST_ARMY_CHANGE"
    GAME_END = "GAME_END"
    # Phase 2d addition — fires exactly once per game at the moment
    # the setup phase ends, after the last grant_setup_resources but
    # BEFORE any main-phase action (dice roll, opp first turn, etc.).
    # Used by the replay recorder to capture a clean "end of setup,
    # main phase not started" snapshot — necessary for seat=1 where
    # env.step #3 atomically runs opp's first main turn.
    SETUP_COMPLETE = "SETUP_COMPLETE"


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

    def emit(self, event_type: str | BroadcastEventType, **payload: Any) -> GameEvent:
        """Emit a generic event and notify all subscribers.

        ``event_type`` accepts both a raw string and a
        :class:`BroadcastEventType` enum member. Enum members are
        coerced to their underlying string so the wire format stays
        backwards-compatible with consumers that compare against
        ``str`` literals."""
        type_str = event_type.value if isinstance(event_type, BroadcastEventType) else event_type
        event: GameEvent = {"type": type_str, **payload}
        self.last_event = event
        for cb in list(self._subscribers):
            cb(event)
        return event

    # Convenience helpers for legacy game events -----------------------

    def dice_roll(self, roller_name: str, value: int) -> GameEvent:
        return self.emit(BroadcastEventType.DICE_ROLL, player=roller_name, value=value)

    def discard(self, player_name: str, resources: list[str]) -> GameEvent:
        # resources: list like ['BRICK', 'WOOD', ...] that were discarded
        return self.emit(BroadcastEventType.DISCARD, player=player_name, resources=resources)

    def year_of_plenty(self, player_name: str, resources: list[str]) -> GameEvent:
        # resources: list like ['BRICK', 'ORE'] that were taken from bank
        return self.emit(BroadcastEventType.YOP, player=player_name, resources=resources)

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
        return self.emit(
            BroadcastEventType.RESOURCE_CHANGE, player=player_name, delta=delta, source=source
        )

    # Phase 0.5 helpers ------------------------------------------------

    def monopoly(self, player_name: str, resource: str, count: int) -> GameEvent:
        """A player played a Monopoly dev card. ``count`` is the
        total resources of ``resource`` transferred to ``player_name``
        from every other player (sum across all victims)."""
        return self.emit(
            BroadcastEventType.MONOPOLY,
            player=player_name,
            resource=resource,
            count=count,
        )

    def move_robber(self, player_name: str, hex_idx: int) -> GameEvent:
        """The robber moved to a new hex. Fired regardless of whether
        the move triggered a subsequent ``steal`` event."""
        return self.emit(
            BroadcastEventType.MOVE_ROBBER,
            player=player_name,
            hex_idx=hex_idx,
        )

    def steal(self, robber_name: str, victim_name: str, resource: str) -> GameEvent:
        """A player stole one card from a victim. ``resource`` is the
        actual resource taken when known (omniscient context, since
        the replay system records perfect-information state); pass
        ``"UNKNOWN"`` if the recorder cannot determine the resource."""
        return self.emit(
            BroadcastEventType.STEAL,
            robber=robber_name,
            victim=victim_name,
            resource=resource,
        )

    def build(self, player_name: str, kind: str, location: int) -> GameEvent:
        """A player completed a build action. ``kind`` is one of
        ``"SETTLEMENT" | "CITY" | "ROAD"``.

        **``location`` sentinel**: the engine itself only knows
        pixel-coord vertex/edge keys (see ``catanBoard.boardGraph``).
        The env (:class:`catan_rl.env.catan_env.CatanEnv`) owns the
        integer ``_vertex_to_idx`` / ``_edge_to_idx`` maps. The engine
        therefore emits ``location=-1`` as a documented sentinel; the
        replay recorder's subscriber MUST resolve the real index by
        diffing the player's ``buildGraph`` against the previous
        snapshot, OR (better, for a future Rust/C++ engine port) the
        env can plumb a resolver callback to the engine via
        ``self.game.vertex_pixel_to_idx`` and the engine emits the
        resolved integer directly.

        Distinct from the existing ``RESOURCE_CHANGE source=BUILD_*``
        emit: that one fires on the cost side; this one fires on the
        structural side and includes ``location`` so the recorder can
        attribute the build to a specific vertex/edge."""
        return self.emit(
            BroadcastEventType.BUILD,
            player=player_name,
            kind=kind,
            location=location,
        )

    def longest_road_change(
        self,
        prev_owner: str | None,
        new_owner: str | None,
        length: int,
    ) -> GameEvent:
        """Holder of the Longest Road bonus changed. Only fires when
        ``prev_owner != new_owner`` — the engine should never call
        this with both fields equal (a re-confirmation of the same
        holder would spam consumers)."""
        return self.emit(
            BroadcastEventType.LONGEST_ROAD_CHANGE,
            prev_owner=prev_owner,
            new_owner=new_owner,
            length=length,
        )

    def largest_army_change(
        self,
        prev_owner: str | None,
        new_owner: str | None,
        knights: int,
    ) -> GameEvent:
        """Holder of the Largest Army bonus changed. Same
        no-self-fire contract as :meth:`longest_road_change`."""
        return self.emit(
            BroadcastEventType.LARGEST_ARMY_CHANGE,
            prev_owner=prev_owner,
            new_owner=new_owner,
            knights=knights,
        )

    def game_end(self, winner_name: str, vp_breakdown: dict[str, int]) -> GameEvent:
        """The game terminated with a winner. ``vp_breakdown`` maps
        each player's name to their final VP count (winner has
        ``>= 15`` per the 1v1 ruleset; the loser is whatever they
        ended on).

        **Truncation contract**: this event is **NOT** emitted when a
        game truncates without a VP-15 winner (e.g.,
        ``_turn_count >= max_turns``). The replay recorder handles
        truncation by synthesising its own terminal ``ReplayStep`` from
        the env's terminal-flag side-channel — no event needed. A
        future schema bump could widen ``GameEnd.winner`` to allow
        ``None`` for truncations, but until then the recorder
        attributes truncated games via the per-step ``kind="terminal"``
        marker without a corresponding ``GameEnd`` step event."""
        return self.emit(
            BroadcastEventType.GAME_END,
            winner=winner_name,
            vp_breakdown=dict(vp_breakdown),
        )

    def setup_complete(self) -> GameEvent:
        """The 4-placement snake-draft setup phase finished — all
        settlements + roads placed, all starting resources granted, no
        main-phase action has run yet. The recorder snapshots on this
        marker to capture the "end of setup, pre-main" board state,
        which env.step alone cannot deliver in the seat=1 case (the
        opp's first main turn runs inside the same env.step that
        finalises setup)."""
        return self.emit(BroadcastEventType.SETUP_COMPLETE)
