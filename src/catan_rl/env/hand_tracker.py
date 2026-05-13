"""Perfect opponent-resource tracker via the engine broadcast bus.

In 1v1 Colonist.io ruleset, *every* resource-mutating event is observable
because there is no player-to-player trading. The engine's
:class:`catan_rl.engine.broadcast.GameBroadcast` emits a ``RESOURCE_CHANGE``
event at each of the ~15 mutation sites (build/buy/trade/discard/rob/
year-of-plenty/monopoly/dice-yield/setup-grant). Subscribing to those
events from game-start gives the agent's obs encoder *exact* opponent
resource counts — a luxury that disappears the instant P2P trading
returns or a third player joins, but is rule-correct here.

Usage::

    tracker = BroadcastHandTracker(["Agent", "Opponent"])
    tracker.subscribe(game.broadcast)           # before setup runs
    # ... env.reset() runs setup, which emits SETUP resource_change events ...
    hand = tracker.get_hand("Opponent")          # {WOOD: ..., BRICK: ..., ...}
    tracker.verify_against_players({p.name: p for p in players})  # sanity

The tracker maintains hands in the obs encoder's **Charlesworth resource
order** (WOOD, BRICK, WHEAT, ORE, SHEEP) so consumers can directly cast
to a 5-vector without re-permuting.

Design choices:

  * Subscription is idempotent — calling :meth:`subscribe` twice on the
    same broadcast registers only one callback. This avoids double-counting
    if a caller re-subscribes by mistake.
  * :meth:`reset` zeros the hands but leaves the broadcast subscription
    intact, because env.reset() typically re-uses the same broadcast bus
    (it's part of the engine, not the env).
  * :meth:`verify_against_players` is a defensive utility — call it at
    game-end in tests or with an env-level invariant flag to catch any
    emit-site that forgot to fire a ``RESOURCE_CHANGE`` event.
  * Negative final counts are clamped to 0 (matches v1 behaviour). If the
    tracker ever underflows, ``verify_against_players`` will catch the
    drift on the next call.
"""

from __future__ import annotations

from typing import Any

from catan_rl.engine.broadcast import GameBroadcast, GameEvent

#: Charlesworth-order resource list. Identical to
#: :data:`catan_rl.policy.obs_schema.RESOURCES_CW` (re-declared here so the
#: env module doesn't import from the policy package — keeps the dep graph
#: pointing one way).
RESOURCES_CW: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")


class BroadcastHandTracker:
    """Maintains exact per-player resource counts via broadcast subscription.

    Args:
        player_names: List of player names the tracker should follow.
            Names must match the ``player.name`` field that the engine
            puts on every ``RESOURCE_CHANGE`` event.
    """

    def __init__(self, player_names: list[str]) -> None:
        if not player_names:
            raise ValueError("BroadcastHandTracker needs at least one player name")
        if len(set(player_names)) != len(player_names):
            raise ValueError(f"Duplicate player names: {player_names}")
        self._hands: dict[str, dict[str, int]] = {
            name: dict.fromkeys(RESOURCES_CW, 0) for name in player_names
        }
        self._broadcast: GameBroadcast | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def subscribe(self, broadcast: GameBroadcast) -> None:
        """Register to receive every event on ``broadcast``.

        Idempotent: re-subscribing to the same broadcast is a no-op.
        Switching to a different broadcast unsubscribes from the prior one
        first.
        """
        if self._broadcast is broadcast:
            return
        if self._broadcast is not None:
            self.unsubscribe()
        broadcast.subscribe(self._on_event)
        self._broadcast = broadcast

    def unsubscribe(self) -> None:
        """Stop receiving events. Safe to call when not subscribed."""
        if self._broadcast is not None:
            self._broadcast.unsubscribe(self._on_event)
            self._broadcast = None

    def reset(self) -> None:
        """Zero all hands. Subscription is preserved.

        Use this when an env episode ends and the next episode begins on
        the *same* broadcast bus; the alternative is to construct a fresh
        tracker each episode.
        """
        for hand in self._hands.values():
            for r in RESOURCES_CW:
                hand[r] = 0

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event(self, event: GameEvent) -> None:
        """Process one broadcast event. Only ``RESOURCE_CHANGE`` updates hands."""
        if event.get("type") != "RESOURCE_CHANGE":
            return
        name = event.get("player")
        delta = event.get("delta", {})
        if not isinstance(name, str) or name not in self._hands:
            return
        if not isinstance(delta, dict):
            return
        hand = self._hands[name]
        for resource, d in delta.items():
            if resource in hand:
                # Clamp to 0 — if the engine sends a negative delta that
                # would drop below zero, that's an event-emission bug
                # and verify_against_players() will catch it next call.
                hand[resource] = max(0, hand[resource] + int(d))

    # ------------------------------------------------------------------
    # Seeding / inspection
    # ------------------------------------------------------------------

    def seed_from_player(self, player: Any) -> None:
        """Initialise one player's tracked hand from their current resources.

        Use this when the tracker is constructed *after* some resource-
        granting events have already fired (e.g. midway through setup).
        For new-episode use, prefer ``reset()`` plus subscribing before
        ``CatanEnv.reset()``.
        """
        name = getattr(player, "name", None)
        if not isinstance(name, str) or name not in self._hands:
            return
        res = getattr(player, "resources", {}) or {}
        for r in RESOURCES_CW:
            self._hands[name][r] = int(res.get(r, 0))

    def get_hand(self, player_name: str) -> dict[str, int]:
        """Return a copy of one player's tracked hand.

        Returns an all-zero dict (in Charlesworth order) if the player is
        unknown — keeps consumers from having to special-case "missing".
        """
        if player_name not in self._hands:
            return dict.fromkeys(RESOURCES_CW, 0)
        return self._hands[player_name].copy()

    def total(self, player_name: str) -> int:
        """Sum of all tracked resources for one player."""
        return sum(self.get_hand(player_name).values())

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_against_players(self, players_by_name: dict[str, Any]) -> None:
        """Assert that every tracked hand matches the player's actual ``resources``.

        Catches silent bugs in event emission. Use at game-end as an
        invariant or in tests. Raises ``AssertionError`` on the first
        mismatch with a diff between tracked and actual.
        """
        for name, hand in self._hands.items():
            player = players_by_name.get(name)
            if player is None:
                continue
            actual_raw = getattr(player, "resources", {}) or {}
            actual = {r: int(actual_raw.get(r, 0)) for r in RESOURCES_CW}
            if hand != actual:
                diff = {r: (hand[r], actual[r]) for r in RESOURCES_CW if hand[r] != actual[r]}
                raise AssertionError(
                    f"BroadcastHandTracker drift for {name!r}: "
                    f"tracker={hand}, actual={actual}, diff={diff}"
                )
