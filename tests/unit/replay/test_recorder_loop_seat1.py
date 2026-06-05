"""Unit tests for the Phase 2d seat=1 setup assembly path.

The seat=1 flow is asymmetric — opp's first placement happens during
``env.reset`` (before any agent action) and opp's second placement
happens inside the same env.step that finalises the agent's road 2.
The integration smoke (`tests/integration/test_record_smoke.py`) only
exercises seat=0 matchups (the supported non-policy pairs all map to
seat=0), so seat=1 needs unit coverage here against hand-built
snapshots.
"""

from __future__ import annotations

from catan_rl.replay.recorder_loop import (
    _empty_state_like,
    _setup_steps_seat_0,
    _setup_steps_seat_1,
    _zero_actor_in_snapshot,
)
from catan_rl.replay.schema import PlayerStateSnapshot, StepStateSnapshot


def _zero_player(name: str, vp: int = 0) -> PlayerStateSnapshot:
    return PlayerStateSnapshot(
        name=name,
        vp=vp,
        resources={"WOOD": 0, "BRICK": 0, "WHEAT": 0, "ORE": 0, "SHEEP": 0},
        dev_cards_hand={
            "KNIGHT": 0,
            "VP": 0,
            "ROAD_BUILDER": 0,
            "YEAR_OF_PLENTY": 0,
            "MONOPOLY": 0,
        },
        dev_cards_played={
            "KNIGHT": 0,
            "VP": 0,
            "ROAD_BUILDER": 0,
            "YEAR_OF_PLENTY": 0,
            "MONOPOLY": 0,
        },
    )


def _snap(
    *,
    settles_a: tuple[int, ...] = (),
    settles_b: tuple[int, ...] = (),
    roads_a: tuple[int, ...] = (),
    roads_b: tuple[int, ...] = (),
    vp_a: int = 0,
    vp_b: int = 0,
    res_a: dict | None = None,
    res_b: dict | None = None,
) -> StepStateSnapshot:
    pa = _zero_player("Opponent", vp=vp_a)  # seat=1: opp is player_a
    pb = _zero_player("Agent", vp=vp_b)  # seat=1: agent is player_b
    if res_a is not None:
        pa = PlayerStateSnapshot(
            name=pa.name,
            vp=pa.vp,
            resources=dict(res_a),
            dev_cards_hand=pa.dev_cards_hand,
            dev_cards_played=pa.dev_cards_played,
        )
    if res_b is not None:
        pb = PlayerStateSnapshot(
            name=pb.name,
            vp=pb.vp,
            resources=dict(res_b),
            dev_cards_hand=pb.dev_cards_hand,
            dev_cards_played=pb.dev_cards_played,
        )
    return StepStateSnapshot(
        settlements={"player_a": settles_a, "player_b": settles_b},
        cities={"player_a": (), "player_b": ()},
        roads={"player_a": roads_a, "player_b": roads_b},
        robber_hex=9,
        players={"player_a": pa, "player_b": pb},
        longest_road_holder=None,
        largest_army_holder=None,
    )


_SEAT_TO_ACTOR_1 = {"Agent": "player_b", "Opponent": "player_a"}


class TestSetupStepsSeat1:
    def test_four_steps_in_snake_draft_order(self) -> None:
        # Reset state: opp (player_a) has first placement.
        snap_reset = _snap(settles_a=(10,), roads_a=(20,), vp_a=1)
        # After agent's step 1 (settle+road): both have first placements.
        snap_after_step1 = _snap(
            settles_a=(10,),
            roads_a=(20,),
            vp_a=1,
            settles_b=(30,),
            roads_b=(40,),
            vp_b=1,
        )
        # After all setup: opp has both, agent has both, both with grants.
        snap_setup_complete = _snap(
            settles_a=(10, 11),
            roads_a=(20, 21),
            vp_a=2,
            settles_b=(30, 31),
            roads_b=(40, 41),
            vp_b=2,
            res_a={"WOOD": 1, "BRICK": 0, "WHEAT": 0, "ORE": 0, "SHEEP": 1},
            res_b={"WOOD": 0, "BRICK": 1, "WHEAT": 1, "ORE": 0, "SHEEP": 0},
        )
        # Agent's two placements (recorded actions).
        agent_actions = [(30, 40), (31, 41)]

        steps, _log = _setup_steps_seat_1(
            agent_actions=agent_actions,
            snap_after_reset=snap_reset,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=snap_setup_complete,
            seat_to_actor=_SEAT_TO_ACTOR_1,
        )
        assert len(steps) == 4
        # Snake draft for seat 1: actor sequence is [a, b, b, a].
        assert [s.actor for s in steps] == [
            "player_a",
            "player_b",
            "player_b",
            "player_a",
        ]
        # Each step has exactly 2 sub-actions.
        for s in steps:
            assert len(s.actions) == 2
            assert s.actions[0].kind == "BuildSettlement"
            assert s.actions[1].kind == "BuildRoad"

    def test_sub_action_indices_match_engine_diffs(self) -> None:
        snap_reset = _snap(settles_a=(10,), roads_a=(20,), vp_a=1)
        snap_after_step1 = _snap(
            settles_a=(10,),
            roads_a=(20,),
            vp_a=1,
            settles_b=(30,),
            roads_b=(40,),
            vp_b=1,
        )
        snap_setup_complete = _snap(
            settles_a=(10, 11),
            roads_a=(20, 21),
            vp_a=2,
            settles_b=(30, 31),
            roads_b=(40, 41),
            vp_b=2,
        )
        agent_actions = [(30, 40), (31, 41)]
        steps, _ = _setup_steps_seat_1(
            agent_actions=agent_actions,
            snap_after_reset=snap_reset,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=snap_setup_complete,
            seat_to_actor=_SEAT_TO_ACTOR_1,
        )
        # Step 0 = opp's first placement: derived from reset diff.
        assert steps[0].actions[0].args == {"vertex_idx": 10}
        assert steps[0].actions[1].args == {"edge_idx": 20}
        # Step 1 = agent's first placement (from agent_actions).
        assert steps[1].actions[0].args == {"vertex_idx": 30}
        assert steps[1].actions[1].args == {"edge_idx": 40}
        # Step 2 = agent's second placement.
        assert steps[2].actions[0].args == {"vertex_idx": 31}
        assert steps[2].actions[1].args == {"edge_idx": 41}
        # Step 3 = opp's second placement: derived from diff between
        # snap_after_step1 and setup_complete.
        assert steps[3].actions[0].args == {"vertex_idx": 11}
        assert steps[3].actions[1].args == {"edge_idx": 21}

    def test_state_after_step_3_is_setup_complete(self) -> None:
        snap_reset = _snap(settles_a=(10,), roads_a=(20,), vp_a=1)
        snap_after_step1 = _snap(
            settles_a=(10,),
            roads_a=(20,),
            vp_a=1,
            settles_b=(30,),
            roads_b=(40,),
            vp_b=1,
        )
        snap_setup_complete = _snap(
            settles_a=(10, 11),
            roads_a=(20, 21),
            vp_a=2,
            settles_b=(30, 31),
            roads_b=(40, 41),
            vp_b=2,
        )
        steps, _ = _setup_steps_seat_1(
            agent_actions=[(30, 40), (31, 41)],
            snap_after_reset=snap_reset,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=snap_setup_complete,
            seat_to_actor=_SEAT_TO_ACTOR_1,
        )
        assert steps[3].state_after is snap_setup_complete

    def test_state_after_step_2_has_agent_full_opp_first_only(self) -> None:
        # The intermediate state for step 2 is "agent done, opp at
        # first placement only" — synthesised via Phase 2c's
        # synthesize_intermediate against setup_complete_snap.
        snap_reset = _snap(settles_a=(10,), roads_a=(20,), vp_a=1)
        snap_after_step1 = _snap(
            settles_a=(10,),
            roads_a=(20,),
            vp_a=1,
            settles_b=(30,),
            roads_b=(40,),
            vp_b=1,
        )
        snap_setup_complete = _snap(
            settles_a=(10, 11),
            roads_a=(20, 21),
            vp_a=2,
            settles_b=(30, 31),
            roads_b=(40, 41),
            vp_b=2,
        )
        steps, _ = _setup_steps_seat_1(
            agent_actions=[(30, 40), (31, 41)],
            snap_after_reset=snap_reset,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=snap_setup_complete,
            seat_to_actor=_SEAT_TO_ACTOR_1,
        )
        st2 = steps[2].state_after
        # opp at first placement only.
        assert st2.settlements["player_a"] == (10,)
        assert st2.roads["player_a"] == (20,)
        assert st2.players["player_a"].vp == 1
        # agent full (taken from setup_complete via synthesize-other-unchanged).
        assert st2.settlements["player_b"] == (30, 31)
        assert st2.roads["player_b"] == (40, 41)
        assert st2.players["player_b"].vp == 2


class TestSetupStepsSeat0:
    def test_four_steps_in_snake_draft_order(self) -> None:
        # After agent's step 1 (settle+road) PLUS opp's atomic burst:
        # agent has first placement, opp has both placements.
        snap_after_step1 = _snap(
            settles_a=(50,),
            roads_a=(60,),
            vp_a=1,
            settles_b=(70, 71),
            roads_b=(80, 81),
            vp_b=2,
        )
        snap_setup_complete = _snap(
            settles_a=(50, 51),
            roads_a=(60, 61),
            vp_a=2,
            settles_b=(70, 71),
            roads_b=(80, 81),
            vp_b=2,
        )
        agent_actions = [(50, 60), (51, 61)]
        # For seat=0, seat_to_actor maps "Agent"→player_a, "Opponent"→player_b.
        seat_to_actor_0 = {"Agent": "player_a", "Opponent": "player_b"}
        steps, _ = _setup_steps_seat_0(
            agent_actions=agent_actions,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=snap_setup_complete,
            seat_to_actor=seat_to_actor_0,
        )
        assert len(steps) == 4
        # Snake draft for seat 0: [a, b, b, a].
        assert [s.actor for s in steps] == [
            "player_a",
            "player_b",
            "player_b",
            "player_a",
        ]


class TestEmptyStateLike:
    def test_zeroes_everything_but_preserves_keys(self) -> None:
        snap = _snap(
            settles_a=(7, 8),
            roads_a=(11, 12),
            vp_a=2,
            settles_b=(99,),
            res_a={"WOOD": 5, "BRICK": 0, "WHEAT": 0, "ORE": 0, "SHEEP": 0},
        )
        empty = _empty_state_like(snap)
        assert empty.settlements["player_a"] == ()
        assert empty.settlements["player_b"] == ()
        assert empty.roads["player_a"] == ()
        assert empty.players["player_a"].vp == 0
        assert empty.players["player_a"].resources["WOOD"] == 0
        # Robber hex passes through; LR/LA holders nullified.
        assert empty.robber_hex == snap.robber_hex
        assert empty.longest_road_holder is None
        assert empty.largest_army_holder is None


class TestZeroActorInSnapshot:
    def test_target_actor_zeroed_other_preserved(self) -> None:
        snap = _snap(
            settles_a=(7, 8),
            roads_a=(11, 12),
            vp_a=2,
            settles_b=(99,),
            roads_b=(80,),
            vp_b=1,
        )
        out = _zero_actor_in_snapshot(snap, "player_b")
        # player_b zeroed.
        assert out.settlements["player_b"] == ()
        assert out.roads["player_b"] == ()
        assert out.players["player_b"].vp == 0
        # player_a preserved.
        assert out.settlements["player_a"] == (7, 8)
        assert out.roads["player_a"] == (11, 12)
        assert out.players["player_a"].vp == 2

    def test_lr_la_holders_always_none(self) -> None:
        # Reviewer-flagged invariant: setup-phase intermediate state
        # must have None LR/LA holders even if the post-burst
        # snapshot carries stale values from future engine drift.
        snap = StepStateSnapshot(
            settlements={"player_a": (7,), "player_b": (99,)},
            cities={"player_a": (), "player_b": ()},
            roads={"player_a": (11,), "player_b": (80,)},
            robber_hex=9,
            players={
                "player_a": _zero_player("Alice", vp=1),
                "player_b": _zero_player("Bob", vp=1),
            },
            # Stale values that should NOT leak through:
            longest_road_holder="player_a",
            largest_army_holder="player_a",
            last_seven_roller="player_a",
        )
        out = _zero_actor_in_snapshot(snap, "player_b")
        assert out.longest_road_holder is None
        assert out.largest_army_holder is None
        assert out.last_seven_roller is None


class TestPartitionMainEventsByActor:
    def test_splits_on_dice_roll_boundary(self) -> None:
        from catan_rl.replay.recorder_loop import _partition_main_events_by_actor

        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        events: list[dict] = [
            {"type": "BUILD", "player": "Agent", "kind": "SETTLEMENT", "location": -1},
            {"type": "DICE_ROLL", "player": "Opponent", "value": 8},
            {"type": "BUILD", "player": "Opponent", "kind": "ROAD", "location": -1},
        ]
        groups = _partition_main_events_by_actor(
            events, initial_actor="player_a", seat_to_actor=seat_to_actor
        )
        assert len(groups) == 2
        assert groups[0][0] == "player_a"
        assert groups[0][1] == events[:1]
        assert groups[1][0] == "player_b"
        assert groups[1][1] == events[1:]

    def test_empty_event_list_returns_empty(self) -> None:
        from catan_rl.replay.recorder_loop import _partition_main_events_by_actor

        groups = _partition_main_events_by_actor(
            [],
            initial_actor="player_a",
            seat_to_actor={"Agent": "player_a", "Opponent": "player_b"},
        )
        assert groups == []

    def test_no_dice_roll_single_group(self) -> None:
        from catan_rl.replay.recorder_loop import _partition_main_events_by_actor

        events: list[dict] = [
            {"type": "BUILD", "player": "Agent", "kind": "SETTLEMENT", "location": -1},
            {"type": "BUILD", "player": "Agent", "kind": "ROAD", "location": -1},
        ]
        groups = _partition_main_events_by_actor(
            events,
            initial_actor="player_a",
            seat_to_actor={"Agent": "player_a", "Opponent": "player_b"},
        )
        assert len(groups) == 1
        assert groups[0][0] == "player_a"


class TestSplitAtSetupComplete:
    def test_returns_only_post_marker_events(self) -> None:
        from catan_rl.replay.recorder_loop import _split_at_setup_complete

        events: list[dict] = [
            {"type": "BUILD", "player": "Opponent", "kind": "SETTLEMENT", "location": -1},
            {"type": "RESOURCE_CHANGE", "player": "Opponent", "deltas": {}, "reason": "SETUP"},
            {"type": "SETUP_COMPLETE"},
            {"type": "DICE_ROLL", "player": "Opponent", "value": 6},
            {"type": "BUILD", "player": "Opponent", "kind": "ROAD", "location": -1},
        ]
        out = _split_at_setup_complete(events)
        assert len(out) == 2
        assert out[0]["type"] == "DICE_ROLL"
        assert out[1]["type"] == "BUILD"

    def test_no_marker_returns_empty(self) -> None:
        from catan_rl.replay.recorder_loop import _split_at_setup_complete

        events: list[dict] = [{"type": "BUILD"}, {"type": "RESOURCE_CHANGE"}]
        assert _split_at_setup_complete(events) == []
