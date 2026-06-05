"""Recorder: build replay states from a live ``catanGame``.

Phases 2a-2b of the replay-system build-out. This module owns the
headless conversion from engine state + broadcast events into the
replay schema's dataclasses. The recorder driver loop (Phase 2d)
calls into these functions per step.

Pieces in this file:

* :func:`snapshot_step_state` (Phase 2a) — wraps Phase 0.5's
  ``game.snapshot_state`` accessor, returns a
  :class:`StepStateSnapshot` dataclass.
* :class:`EventCollector` (Phase 2b) — subscribes to
  ``game.broadcast`` and collects raw broadcast events into a buffer
  the recorder drains at each step boundary.
* :func:`classify_step_events` (Phase 2b) — maps a list of raw
  broadcast events to typed :class:`StepEvent` instances + parallel
  ``log_lines`` for the viewer's event panel. Player names are
  translated through ``seat_to_actor``.
* :func:`extract_sub_actions` (Phase 2b) — derives a list of
  :class:`SubAction` from the event stream. ``BUILD`` events carry
  ``location=-1`` from the engine (the engine doesn't own the int
  index map); this function back-fills the location by diffing the
  previous snapshot against the current snapshot.

Future phase 2c will add a setup-burst splitter (seat-conditional
logic to produce exactly 4 setup steps); 2d wires everything into a
turn-driven loop in the recorder driver.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

from catan_rl.engine.broadcast import BroadcastEventType, GameBroadcast, GameEvent
from catan_rl.replay.schema import (
    GameEnd,
    LargestArmyChange,
    LongestRoadChange,
    Monopoly,
    PlayerStateSnapshot,
    Robber,
    Steal,
    StepEvent,
    StepStateSnapshot,
    SubAction,
)

_LOG = logging.getLogger("catan_rl.replay")


def snapshot_step_state(
    game: Any,
    *,
    seat_to_actor: dict[str, str],
    vertex_pixel_to_idx: dict[Any, int],
    edge_key_to_idx: dict[tuple[str, str], int],
) -> StepStateSnapshot:
    """Capture ``game``'s state and return a :class:`StepStateSnapshot`.

    Args:
        game: a live :class:`catanGame` instance. Its
            ``snapshot_state(...)`` accessor (added Phase 0.5) is the
            only engine surface this function touches.
        seat_to_actor: maps engine player ``name`` →
            ``"player_a"``/``"player_b"``. Built once at recorder
            reset based on ``agent_seat`` and passed unchanged
            thereafter.
        vertex_pixel_to_idx: the env's ``_vertex_to_idx`` map (engine
            keys vertices by pixel-coord tuples; the recorder needs
            integer indices for the JSON).
        edge_key_to_idx: the env's edge-key → integer index map; the
            keys are ``(s1, s2)`` lex-sorted string tuples per the
            convention in :func:`catan_env._edge_key`.

    Returns a frozen :class:`StepStateSnapshot`. The function is
    deep-copy-safe: subsequent engine mutations on ``game`` do not
    alter the returned snapshot.
    """
    raw = game.snapshot_state(seat_to_actor, vertex_pixel_to_idx, edge_key_to_idx)
    return _state_after_from_dict(raw)


def _state_after_from_dict(raw: dict[str, Any]) -> StepStateSnapshot:
    """Convert the engine's snapshot dict to a frozen
    :class:`StepStateSnapshot`. Defensive: deep-copies every mutable
    sub-container so the schema instance is independent of the source
    dict (the engine already deep-copies internally but the wrapper
    repeats it to guarantee the contract end-to-end)."""
    settlements = {k: tuple(int(i) for i in v) for k, v in raw["settlements"].items()}
    cities = {k: tuple(int(i) for i in v) for k, v in raw["cities"].items()}
    roads = {k: tuple(int(i) for i in v) for k, v in raw["roads"].items()}

    players: dict[str, PlayerStateSnapshot] = {}
    for actor, snap in raw["players"].items():
        players[actor] = PlayerStateSnapshot(
            name=str(snap["name"]),
            vp=int(snap["vp"]),
            resources=copy.deepcopy(dict(snap["resources"])),
            dev_cards_hand=copy.deepcopy(dict(snap["dev_cards_hand"])),
            dev_cards_played=copy.deepcopy(dict(snap["dev_cards_played"])),
        )

    return StepStateSnapshot(
        settlements=settlements,
        cities=cities,
        roads=roads,
        robber_hex=int(raw["robber_hex"]),
        players=players,
        longest_road_holder=raw.get("longest_road_holder"),
        largest_army_holder=raw.get("largest_army_holder"),
        last_seven_roller=raw.get("last_seven_roller"),
    )


# ---------------------------------------------------------------------------
# Phase 2b: event collection + classification
# ---------------------------------------------------------------------------


class EventCollector:
    """Subscribes to a :class:`GameBroadcast` and buffers raw events.

    Usage::

        collector = EventCollector()
        collector.subscribe(game.broadcast)
        # ... engine actions fire events into the buffer ...
        events_this_step = collector.drain()

    Thread-safety: not safe across threads. The recorder loop is
    single-threaded so this is fine.
    """

    def __init__(self) -> None:
        self._events: list[GameEvent] = []

    def callback(self, event: GameEvent) -> None:
        """Subscriber callback. Engine calls this on every emit."""
        self._events.append(event)

    def subscribe(self, broadcast: GameBroadcast) -> None:
        """Register :meth:`callback` with ``broadcast``."""
        broadcast.subscribe(self.callback)

    def unsubscribe(self, broadcast: GameBroadcast) -> None:
        """Remove :meth:`callback` from ``broadcast``."""
        broadcast.unsubscribe(self.callback)

    def drain(self) -> list[GameEvent]:
        """Return the accumulated events and reset the buffer.

        Returns a fresh list; the internal buffer is replaced with an
        empty one so subsequent emits start a new step."""
        out = self._events
        self._events = []
        return out

    def peek(self) -> tuple[GameEvent, ...]:
        """Read-only view of the current buffer without draining.
        Tests use this to assert intermediate state."""
        return tuple(self._events)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _actor_for(name: str | None, seat_to_actor: dict[str, str]) -> str | None:
    """Translate engine player name → actor name. Returns ``None`` if
    ``name`` is ``None`` (some broadcast events carry ``None`` for
    ``prev_owner`` on first acquisition). Raises :class:`ValueError`
    if ``name`` is a non-None string that ``seat_to_actor`` doesn't
    know about — silent lenient fallthrough would write engine
    names into the JSON instead of ``player_a`` / ``player_b``."""
    if name is None:
        return None
    actor = seat_to_actor.get(name)
    if actor is None:
        raise ValueError(
            f"recorder: event references player {name!r} not in "
            f"seat_to_actor (have {sorted(seat_to_actor)}). Populate "
            "the mapping at recorder reset before draining events."
        )
    return actor


# ---------------------------------------------------------------------------
# Classification: raw broadcast → typed StepEvents + log_lines
# ---------------------------------------------------------------------------


def classify_step_events(
    raw_events: list[GameEvent],
    *,
    seat_to_actor: dict[str, str],
) -> tuple[list[StepEvent], list[str]]:
    """Convert a step's worth of raw broadcast events into typed
    :class:`StepEvent` instances + a temporally-ordered list of
    human-readable log lines.

    Args:
        raw_events: the list returned by :meth:`EventCollector.drain`.
        seat_to_actor: maps engine player ``name`` → JSON actor
            (``"player_a"`` / ``"player_b"``). Must cover every
            player name the events reference; an unmapped name raises
            :class:`ValueError`.

    Returns ``(step_events, log_lines)``.

    ``log_lines`` is a per-event narration covering every emit
    (including legacy ``DICE_ROLL`` / ``DISCARD`` / ``YOP`` and the
    new ``BUILD`` event). It is in temporal order matching
    ``raw_events``.

    ``step_events`` is the filtered subset corresponding to the 6
    schema variants that drive step-bar markers (``Monopoly``,
    ``Robber``, ``Steal``, ``LongestRoadChange``,
    ``LargestArmyChange``, ``GameEnd``). It is also in temporal order
    but is NOT zip-parallel to ``log_lines`` — they are filtered
    differently.

    Raises:
        ValueError: if any event carries a player name not present in
            ``seat_to_actor`` (defensive — the recorder should
            populate the map with both players at reset time).
    """
    step_events: list[StepEvent] = []
    log_lines: list[str] = []
    for event in raw_events:
        type_str = event.get("type", "")
        if type_str == BroadcastEventType.DICE_ROLL.value:
            player = _actor_for(event.get("player"), seat_to_actor) or "?"
            value = event.get("value")
            log_lines.append(f"{player} rolled {value}")
        elif type_str == BroadcastEventType.DISCARD.value:
            player = _actor_for(event.get("player"), seat_to_actor) or "?"
            res = event.get("resources", [])
            log_lines.append(f"{player} discarded {', '.join(res)}")
        elif type_str == BroadcastEventType.YOP.value:
            player = _actor_for(event.get("player"), seat_to_actor) or "?"
            res = event.get("resources", [])
            log_lines.append(f"{player} Year of Plenty: took {', '.join(res)}")
        elif type_str == BroadcastEventType.RESOURCE_CHANGE.value:
            # Skip the narration; per-build cost RESOURCE_CHANGEs would
            # spam the log panel. The structural BUILD event below
            # covers the user-facing narration. (A future debugging
            # subscriber can still walk the full RESOURCE_CHANGE
            # stream via the broadcast bus.)
            pass
        elif type_str == BroadcastEventType.BUILD.value:
            # No StepEvent variant for BUILD; surfaced via SubActions.
            player = _actor_for(event.get("player"), seat_to_actor) or "?"
            kind = event.get("kind", "?")
            log_lines.append(f"{player} built {kind}")
        elif type_str == BroadcastEventType.MOVE_ROBBER.value:
            player_name = event.get("player")
            hex_idx = int(event.get("hex_idx", -1))
            actor = _actor_for(player_name, seat_to_actor) or "?"
            step_events.append(Robber(player=actor, hex_idx=hex_idx))
            log_lines.append(f"{actor} moved robber to hex {hex_idx}")
        elif type_str == BroadcastEventType.STEAL.value:
            robber = _actor_for(event.get("robber"), seat_to_actor) or "?"
            victim = _actor_for(event.get("victim"), seat_to_actor) or "?"
            resource = str(event.get("resource", "UNKNOWN"))
            step_events.append(Steal(robber=robber, victim=victim, resource=resource))
            log_lines.append(f"{robber} stole {resource} from {victim}")
        elif type_str == BroadcastEventType.MONOPOLY.value:
            actor = _actor_for(event.get("player"), seat_to_actor) or "?"
            resource = str(event.get("resource", "?"))
            count = int(event.get("count", 0))
            # Monopoly.resource is a Literal; the broadcast emit
            # already restricts it to the 5 resource names, but mypy
            # can't see through the dict.get(). The cast pins the
            # contract and a downstream load_replay(strict=True) test
            # would catch any drift.
            step_events.append(
                Monopoly(
                    player=actor,
                    resource=cast(Any, resource),
                    count=count,
                )
            )
            log_lines.append(f"{actor} played Monopoly on {resource} (+{count})")
        elif type_str == BroadcastEventType.LONGEST_ROAD_CHANGE.value:
            prev = _actor_for(event.get("prev_owner"), seat_to_actor)
            new = _actor_for(event.get("new_owner"), seat_to_actor)
            length = int(event.get("length", 0))
            step_events.append(LongestRoadChange(prev_owner=prev, new_owner=new, length=length))
            log_lines.append(
                f"Longest Road → {new} (length {length})"
                + (f" from {prev}" if prev is not None else "")
            )
        elif type_str == BroadcastEventType.LARGEST_ARMY_CHANGE.value:
            prev = _actor_for(event.get("prev_owner"), seat_to_actor)
            new = _actor_for(event.get("new_owner"), seat_to_actor)
            knights = int(event.get("knights", 0))
            step_events.append(LargestArmyChange(prev_owner=prev, new_owner=new, knights=knights))
            log_lines.append(
                f"Largest Army → {new} ({knights} knights)"
                + (f" from {prev}" if prev is not None else "")
            )
        elif type_str == BroadcastEventType.GAME_END.value:
            winner = _actor_for(event.get("winner"), seat_to_actor) or "?"
            raw_vp = event.get("vp_breakdown", {}) or {}
            vp_breakdown: dict[str, int] = {}
            for k, v in raw_vp.items():
                # ``_actor_for`` raises if the key isn't mapped, so
                # this dict comprehension surfaces the boundary
                # error loudly rather than writing engine names into
                # the JSON.
                translated = _actor_for(k, seat_to_actor)
                if translated is None:
                    # Engine name was ``None`` itself — unlikely but
                    # be defensive: skip rather than store None as a
                    # dict key.
                    continue
                vp_breakdown[translated] = int(v)
            step_events.append(GameEnd(winner=winner, vp_breakdown=vp_breakdown))
            log_lines.append(f"GAME END — {winner} wins {vp_breakdown}")
        else:
            # Unknown event type — surface as a log line but skip the
            # StepEvent variant (UnknownEvent is reserved for v2-replay
            # forward compat at JSON read time, not the engine event
            # stream).
            _LOG.warning("recorder: unknown broadcast type %r", type_str)
            log_lines.append(f"<unknown event {type_str}>")
    return step_events, log_lines


# ---------------------------------------------------------------------------
# Sub-action extraction
# ---------------------------------------------------------------------------


def _diff_new_builds(
    prev: StepStateSnapshot, curr: StepStateSnapshot, actor: str, kind: str
) -> list[int]:
    """Return the list of new ``vertex_idx`` (for settle/city) or
    ``edge_idx`` (for road) belonging to ``actor`` between two
    snapshots, in **temporal order** as the engine recorded them.

    The engine appends to ``player.buildGraph["ROADS"]/["SETTLEMENTS"]
    /["CITIES"]`` in play order; ``snapshot_state`` preserves that
    order (see ``game.snapshot_state`` and the tuple ordering on the
    snapshot dataclass). We therefore walk ``curr`` left-to-right and
    yield each item that wasn't in ``prev`` — preserving the
    temporal sequence so a road-builder dev card that places road
    A (edge 22) then road B (edge 10) emits ``[22, 10]`` and pairs
    correctly with the two ``BUILD ROAD`` events in the same order.

    ``kind`` is one of ``"SETTLEMENT" | "CITY" | "ROAD"`` and matches
    the engine's BUILD emit payload."""
    if kind == "SETTLEMENT":
        before_seq = prev.settlements.get(actor, ())
        after_seq = curr.settlements.get(actor, ())
    elif kind == "CITY":
        before_seq = prev.cities.get(actor, ())
        after_seq = curr.cities.get(actor, ())
    elif kind == "ROAD":
        before_seq = prev.roads.get(actor, ())
        after_seq = curr.roads.get(actor, ())
    else:
        return []
    before_set = set(before_seq)
    return [int(idx) for idx in after_seq if idx not in before_set]


def extract_sub_actions(
    raw_events: list[GameEvent],
    *,
    prev_snapshot: StepStateSnapshot,
    curr_snapshot: StepStateSnapshot,
    dice_roll: tuple[int, int] | None,
    seat_to_actor: dict[str, str],
) -> list[SubAction]:
    """Derive a list of :class:`SubAction` instances from the step's
    raw broadcast events.

    The mapping is intentionally lossy:

    * Cost-side ``RESOURCE_CHANGE`` events are skipped — the recorded
      ``state_after`` snapshot is the source of truth for resource
      counts.
    * ``BUILD`` events emit at ``location=-1`` (the engine doesn't
      own the int index); this function back-fills by diffing
      ``curr_snapshot`` against ``prev_snapshot``. If multiple builds
      of the same kind fired in one step (e.g. road-builder placing
      two roads), the new indices are assigned to the BUILD emits in
      sorted order.
    * ``DICE_ROLL`` becomes a ``RollDice`` sub-action carrying both
      die faces — ``dice_roll`` is the (d1, d2) tuple passed in by
      the caller (the engine emit only carries the sum).
    * ``MOVE_ROBBER`` becomes ``MoveRobber`` carrying the hex_idx
      plus the victim (resolved from a paired ``STEAL`` event in the
      same step, if any).

    Args:
        raw_events: full event stream for the step.
        prev_snapshot: state at end of prior step (used for BUILD
            location resolution).
        curr_snapshot: state at end of this step.
        dice_roll: ``(d1, d2)`` if a roll happened, else ``None``.
        seat_to_actor: engine player name → actor name.

    Returns the list of SubActions in chronological order.
    """
    # Freshness invariant: ``raw_events`` must be a fresh list of
    # distinct dict objects per emit. ``_events_after`` uses identity
    # comparison (``is``) to anchor the look-ahead, so a port that
    # interns identical events would break the MOVE_ROBBER → STEAL
    # pairing. EventCollector produces fresh dicts per emit, so this
    # invariant holds for the in-tree caller.

    out: list[SubAction] = []
    # Track per-actor per-kind iterators into the diffed-new build
    # indices, so multiple BUILD events of the same kind get
    # successive indices.
    build_iters: dict[tuple[str, str], list[int]] = {}

    def _next_build_idx(actor: str, kind: str) -> int:
        key = (actor, kind)
        if key not in build_iters:
            build_iters[key] = _diff_new_builds(prev_snapshot, curr_snapshot, actor, kind)
        if build_iters[key]:
            return build_iters[key].pop(0)
        return -1

    # Buffer DISCARD events; the player-name keys collide if both
    # players discard on a 7-roll. We emit one Discard SubAction per
    # event.
    for event in raw_events:
        type_str = event.get("type", "")
        if type_str == BroadcastEventType.DICE_ROLL.value:
            if dice_roll is not None:
                d1, d2 = int(dice_roll[0]), int(dice_roll[1])
            else:
                # Fallback: split the engine's summed value heuristically.
                total = int(event.get("value", 0))
                d1, d2 = total // 2, total - total // 2
            out.append(SubAction(kind="RollDice", args={"d1": d1, "d2": d2}))
        elif type_str == BroadcastEventType.DISCARD.value:
            res = list(event.get("resources", []))
            tally: dict[str, int] = {}
            for r in res:
                tally[str(r)] = tally.get(str(r), 0) + 1
            out.append(SubAction(kind="Discard", args={"discarded": tally}))
        elif type_str == BroadcastEventType.YOP.value:
            res = list(event.get("resources", []))
            # Engine contract: YoP always takes exactly 2 cards. A
            # mismatched count signals an engine bug; surface loudly
            # rather than silently truncate.
            assert len(res) == 2, f"YoP emit had {len(res)} resources, expected 2 (engine drift?)"
            args: dict[str, Any] = {"res_a": str(res[0]), "res_b": str(res[1])}
            out.append(SubAction(kind="PlayYearOfPlenty", args=args))
        elif type_str == BroadcastEventType.MONOPOLY.value:
            out.append(
                SubAction(
                    kind="PlayMonopoly",
                    args={"resource": str(event.get("resource", "?"))},
                )
            )
        elif type_str == BroadcastEventType.STEAL.value:
            # STEAL is captured into the corresponding MoveRobber via
            # the look-ahead pairing below. The StepEvent classifier
            # already records the standalone Steal marker. No SubAction
            # emit here.
            pass
        elif type_str == BroadcastEventType.MOVE_ROBBER.value:
            hex_idx = int(event.get("hex_idx", -1))
            # Look ahead for a STEAL with the same robber that fires
            # immediately after. The engine emits STEAL after
            # RESOURCE_CHANGE, so a paired event sequence is
            # MOVE_ROBBER → RESOURCE_CHANGE → RESOURCE_CHANGE → STEAL.
            # If the next-after-RESOURCE_CHANGE is STEAL, attribute
            # the victim; otherwise leave victim=None.
            victim: str | None = None
            for follow in _events_after(raw_events, event):
                if follow.get("type") == BroadcastEventType.STEAL.value:
                    victim = _actor_for(follow.get("victim"), seat_to_actor)
                    break
                if follow.get("type") == BroadcastEventType.MOVE_ROBBER.value:
                    break  # next robber move; this one had no steal
            out.append(
                SubAction(
                    kind="MoveRobber",
                    args={"hex_idx": hex_idx, "victim": victim},
                )
            )
        elif type_str == BroadcastEventType.BUILD.value:
            player_name = event.get("player")
            actor = _actor_for(player_name, seat_to_actor) or "?"
            kind = str(event.get("kind", "?"))
            loc = _next_build_idx(actor, kind)
            sub_kind_map = {
                "SETTLEMENT": "BuildSettlement",
                "CITY": "BuildCity",
                "ROAD": "BuildRoad",
            }
            sub_kind = sub_kind_map.get(kind, f"Build{kind.title()}")
            arg_key = "edge_idx" if kind == "ROAD" else "vertex_idx"
            out.append(SubAction(kind=sub_kind, args={arg_key: loc}))
        # All other event types (RESOURCE_CHANGE, LR/LA changes,
        # GAME_END) do not have a corresponding SubAction —
        # RESOURCE_CHANGE is bookkeeping and the rest are step-bar
        # markers (handled by classify_step_events).
        #
        # TODO(phase-2d): the env-level action types ``PlayKnight``
        # and ``PlayRoadBuilder`` are NOT reconstructable from the
        # broadcast stream alone (they share trigger events with
        # roll-7 robber moves / regular road builds). Phase 2d will
        # plumb the env's pre-dispatch ``action_type`` into a
        # side-channel so the recorder can prepend these SubActions
        # at the right point in the per-step list.
    return out


def _events_after(raw_events: list[GameEvent], anchor: GameEvent) -> list[GameEvent]:
    """Return the slice of ``raw_events`` strictly after ``anchor``
    (by identity). Used by :func:`extract_sub_actions` to look ahead
    for STEAL events paired with MOVE_ROBBER."""
    try:
        idx = next(i for i, e in enumerate(raw_events) if e is anchor)
    except StopIteration:
        return []
    return raw_events[idx + 1 :]
