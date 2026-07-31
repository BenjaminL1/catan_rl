"""Acting-seat attribution in ``_partition_main_events_by_actor``.

A player who plays a Knight BEFORE rolling emits ``MOVE_ROBBER`` / ``STEAL`` /
``LARGEST_ARMY_CHANGE`` ahead of their own ``DICE_ROLL``. Splitting on dice
rolls alone credited them to the PREVIOUS player — "what the bot did" would
list moves the human made.

The motivating case was the human harness's pre-roll dev-card window
(``scripts/play_vs_model.py``), which is now GONE: the bot has no counterpart
(``env/masks.py`` emits only ``ROLL_DICE`` while ``roll_pending``), so the
window was a one-sided advantage and was removed. These pins stay load-bearing
regardless — the bot's own Knight-then-robber ordering inside a single
``env.step`` produces the same event sequence, and ``preroll-dev-cards-r1.md``
will restore the window for BOTH seats. The event dicts below are synthetic, so
nothing here depends on the harness's shape.
"""

from __future__ import annotations

from catan_rl.replay.recorder_loop import _partition_main_events_by_actor

_SEAT_TO_ACTOR = {"Agent": "player_a", "Opponent": "player_b"}


def _partition(events: list[dict], initial_actor: str = "player_a"):  # type: ignore[no-untyped-def]
    return _partition_main_events_by_actor(
        events, initial_actor=initial_actor, seat_to_actor=_SEAT_TO_ACTOR
    )


def _actor_of(groups, event: dict) -> str:  # type: ignore[no-untyped-def]
    for actor, evs in groups:
        if event in evs:
            return actor
    raise AssertionError(f"event {event} was dropped by the partition")


class TestPreRollKnight:
    def test_pre_roll_knight_is_credited_to_the_player_who_played_it(self) -> None:
        # One env.step: the BOT (Agent) builds, then the HUMAN's (Opponent)
        # whole turn is folded in — starting with a pre-roll Knight.
        bot_build = {"type": "BUILD", "player": "Agent", "kind": "CITY", "location": -1}
        robber = {"type": "MOVE_ROBBER", "player": "Opponent", "hex_idx": 4}
        steal = {
            "type": "STEAL",
            "robber": "Opponent",
            "victim": "Agent",
            "resource": "ORE",
        }
        army = {"type": "LARGEST_ARMY_CHANGE", "prev_owner": None, "new_owner": "Opponent"}
        roll = {"type": "DICE_ROLL", "player": "Opponent", "value": 9}
        human_build = {"type": "BUILD", "player": "Opponent", "kind": "ROAD", "location": -1}

        groups = _partition([bot_build, robber, steal, army, roll, human_build])

        assert _actor_of(groups, bot_build) == "player_a"
        assert _actor_of(groups, robber) == "player_b"
        assert _actor_of(groups, steal) == "player_b"
        # Bookkeeping events are not actor evidence; they ride with the
        # current actor, which by then is correctly the human.
        assert _actor_of(groups, army) == "player_b"
        assert _actor_of(groups, roll) == "player_b"
        assert _actor_of(groups, human_build) == "player_b"

    def test_steal_is_attributed_to_the_robber_not_the_victim(self) -> None:
        roll = {"type": "DICE_ROLL", "player": "Agent", "value": 7}
        robber = {"type": "MOVE_ROBBER", "player": "Agent", "hex_idx": 1}
        steal = {"type": "STEAL", "robber": "Agent", "victim": "Opponent", "resource": "WOOD"}
        victim_delta = {
            "type": "RESOURCE_CHANGE",
            "player": "Opponent",
            "delta": {"WOOD": -1},
            "source": "ROB",
        }
        groups = _partition([roll, robber, steal, victim_delta])
        assert len(groups) == 1
        assert groups[0][0] == "player_a"


class TestNonActorEvidence:
    def test_production_resource_change_does_not_split_a_group(self) -> None:
        # Dice production credits BOTH seats; a RESOURCE_CHANGE naming the
        # non-roller must not open a group for them.
        roll = {"type": "DICE_ROLL", "player": "Agent", "value": 8}
        mine = {
            "type": "RESOURCE_CHANGE",
            "player": "Agent",
            "delta": {"ORE": 1},
            "source": "DICE",
        }
        theirs = {
            "type": "RESOURCE_CHANGE",
            "player": "Opponent",
            "delta": {"WOOD": 2},
            "source": "DICE",
        }
        groups = _partition([roll, mine, theirs])
        assert len(groups) == 1
        assert groups[0][0] == "player_a"

    def test_forced_discard_by_the_non_actor_keeps_its_own_group(self) -> None:
        # Pre-existing behaviour, re-pinned: the DISCARD special case must
        # survive the acting-seat generalisation.
        roll = {"type": "DICE_ROLL", "player": "Agent", "value": 7}
        discard = {"type": "DISCARD", "player": "Opponent", "resources": ["WOOD", "ORE"]}
        robber = {"type": "MOVE_ROBBER", "player": "Agent", "hex_idx": 2}
        groups = _partition([roll, discard, robber])
        assert _actor_of(groups, discard) == "player_b"
        assert _actor_of(groups, roll) == "player_a"
        assert _actor_of(groups, robber) == "player_a"
