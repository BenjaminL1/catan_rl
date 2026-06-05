"""Tests for Phase 2c: setup-burst splitter + intermediate snapshot."""

from __future__ import annotations

import pytest

from catan_rl.replay.recorder import (
    split_burst_one_placement,
    split_burst_two_placements,
    synthesize_intermediate_setup_snapshot,
)
from catan_rl.replay.schema import PlayerStateSnapshot, StepStateSnapshot


def _empty_player(name: str, vp: int = 0) -> PlayerStateSnapshot:
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


def _state(
    *,
    settlements_a: tuple[int, ...] = (),
    cities_a: tuple[int, ...] = (),
    roads_a: tuple[int, ...] = (),
    settlements_b: tuple[int, ...] = (),
    cities_b: tuple[int, ...] = (),
    roads_b: tuple[int, ...] = (),
    robber_hex: int = 9,
    player_a_vp: int = 0,
    player_a_resources: dict | None = None,
) -> StepStateSnapshot:
    pa = _empty_player("Agent", vp=player_a_vp)
    if player_a_resources is not None:
        pa = PlayerStateSnapshot(
            name=pa.name,
            vp=pa.vp,
            resources=dict(player_a_resources),
            dev_cards_hand=pa.dev_cards_hand,
            dev_cards_played=pa.dev_cards_played,
        )
    return StepStateSnapshot(
        settlements={"player_a": settlements_a, "player_b": settlements_b},
        cities={"player_a": cities_a, "player_b": cities_b},
        roads={"player_a": roads_a, "player_b": roads_b},
        robber_hex=robber_hex,
        players={
            "player_a": pa,
            "player_b": _empty_player("Opponent"),
        },
        longest_road_holder=None,
        largest_army_holder=None,
    )


# ---------------------------------------------------------------------------
# split_burst_two_placements
# ---------------------------------------------------------------------------


class TestSplitBurstTwoPlacements:
    def test_returns_two_pairs_in_temporal_order(self) -> None:
        prev = _state()  # all empty
        # Engine places (settle=17, road=23) then (settle=4, road=11).
        # snapshot_state preserves the temporal append order.
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        first, second = split_burst_two_placements(
            actor="player_b",
            prev_snapshot=prev,
            post_snapshot=post,
        )
        assert first == (17, 23)
        assert second == (4, 11)

    def test_temporal_order_preserved_under_descending_int(self) -> None:
        # Regression: ensure the splitter respects buildGraph's
        # temporal append order even when indices are not monotonic.
        prev = _state()
        post = _state(settlements_b=(42, 7), roads_b=(60, 1))
        first, second = split_burst_two_placements(
            actor="player_b",
            prev_snapshot=prev,
            post_snapshot=post,
        )
        assert first == (42, 60)
        assert second == (7, 1)

    def test_wrong_settle_count_raises(self) -> None:
        prev = _state()
        # Only 1 new settle but expected 2.
        post = _state(settlements_b=(17,), roads_b=(23, 11))
        with pytest.raises(ValueError, match="expected 2 new settlements"):
            split_burst_two_placements(
                actor="player_b",
                prev_snapshot=prev,
                post_snapshot=post,
            )

    def test_wrong_road_count_raises(self) -> None:
        prev = _state()
        post = _state(settlements_b=(17, 4), roads_b=(23,))
        with pytest.raises(ValueError, match="expected 2 new roads"):
            split_burst_two_placements(
                actor="player_b",
                prev_snapshot=prev,
                post_snapshot=post,
            )


# ---------------------------------------------------------------------------
# split_burst_one_placement
# ---------------------------------------------------------------------------


class TestSplitBurstOnePlacement:
    def test_returns_settle_road_pair(self) -> None:
        prev = _state()
        post = _state(settlements_b=(17,), roads_b=(23,))
        settle, road = split_burst_one_placement(
            actor="player_b",
            prev_snapshot=prev,
            post_snapshot=post,
        )
        assert settle == 17
        assert road == 23

    def test_wrong_count_raises(self) -> None:
        prev = _state()
        post = _state(settlements_b=(17, 4), roads_b=(23,))  # 2 settles
        with pytest.raises(ValueError, match="expected 1 new settlement"):
            split_burst_one_placement(
                actor="player_b",
                prev_snapshot=prev,
                post_snapshot=post,
            )


# ---------------------------------------------------------------------------
# synthesize_intermediate_setup_snapshot
# ---------------------------------------------------------------------------


class TestSynthesizeIntermediate:
    def test_actor_has_first_placement_only(self) -> None:
        post = _state(
            settlements_b=(17, 4),
            roads_b=(23, 11),
        )
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        assert mid.settlements["player_b"] == (17,)
        assert mid.roads["player_b"] == (23,)

    def test_actor_resources_are_zero(self) -> None:
        # First setup placement does NOT grant resources (only the
        # second settle grants them per the 1v1 Colonist rule).
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        assert all(v == 0 for v in mid.players["player_b"].resources.values())

    def test_actor_vp_is_one(self) -> None:
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        assert mid.players["player_b"].vp == 1

    def test_other_actor_state_unchanged(self) -> None:
        # The other player's state must be taken AS-IS from the post
        # snapshot — they didn't act during this burst.
        post = _state(
            settlements_a=(99,),
            roads_a=(50,),
            player_a_vp=1,
            player_a_resources={
                "WOOD": 1,
                "BRICK": 0,
                "WHEAT": 1,
                "ORE": 0,
                "SHEEP": 0,
            },
            settlements_b=(17, 4),
            roads_b=(23, 11),
        )
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        # player_a's state is the post state.
        assert mid.settlements["player_a"] == (99,)
        assert mid.roads["player_a"] == (50,)
        assert mid.players["player_a"].vp == 1
        assert mid.players["player_a"].resources["WOOD"] == 1
        assert mid.players["player_a"].resources["WHEAT"] == 1

    def test_no_lr_la_holders_during_setup(self) -> None:
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        assert mid.longest_road_holder is None
        assert mid.largest_army_holder is None
        assert mid.last_seven_roller is None

    def test_dev_cards_zero(self) -> None:
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        snap = mid.players["player_b"]
        assert all(v == 0 for v in snap.dev_cards_hand.values())
        assert all(v == 0 for v in snap.dev_cards_played.values())

    def test_works_for_player_a_too(self) -> None:
        # The actor parameter must work for either seat label.
        post = _state(settlements_a=(7, 12), roads_a=(33, 41))
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_a",
            post_snapshot=post,
            first_settle_idx=7,
            first_road_idx=33,
        )
        assert mid.settlements["player_a"] == (7,)
        assert mid.roads["player_a"] == (33,)
        assert mid.players["player_a"].vp == 1


# ---------------------------------------------------------------------------
# End-to-end Phase 2c flow: split + synthesize → 2 reconstructed snapshots
# ---------------------------------------------------------------------------


class TestSynthesizeValidation:
    def test_raises_if_settle_not_in_post(self) -> None:
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        with pytest.raises(ValueError, match="first_settle_idx=99"):
            synthesize_intermediate_setup_snapshot(
                actor="player_b",
                post_snapshot=post,
                first_settle_idx=99,
                first_road_idx=23,
            )

    def test_raises_if_road_not_in_post(self) -> None:
        post = _state(settlements_b=(17, 4), roads_b=(23, 11))
        with pytest.raises(ValueError, match="first_road_idx=99"):
            synthesize_intermediate_setup_snapshot(
                actor="player_b",
                post_snapshot=post,
                first_settle_idx=17,
                first_road_idx=99,
            )

    def test_raises_if_actor_has_cities_in_post(self) -> None:
        # Engine drift safeguard: setup phase must never have cities.
        post = _state(
            settlements_b=(17, 4),
            roads_b=(23, 11),
            cities_b=(17,),
        )
        with pytest.raises(ValueError, match="never builds cities"):
            synthesize_intermediate_setup_snapshot(
                actor="player_b",
                post_snapshot=post,
                first_settle_idx=17,
                first_road_idx=23,
            )

    def test_does_not_mutate_post_snapshot(self) -> None:
        # Synth must produce an independent snapshot — mutating fields
        # on the synth result must not bleed into post_snapshot.
        post = _state(
            settlements_a=(50,),
            roads_a=(60,),
            player_a_vp=1,
            player_a_resources={
                "WOOD": 1,
                "BRICK": 0,
                "WHEAT": 1,
                "ORE": 0,
                "SHEEP": 0,
            },
            settlements_b=(17, 4),
            roads_b=(23, 11),
        )
        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=17,
            first_road_idx=23,
        )
        # Mutate the other actor's resources on the synth result.
        mid.players["player_a"].resources["WOOD"] = 999
        # Post must be unaffected.
        assert post.players["player_a"].resources["WOOD"] == 1


class TestSetupBurstFlow:
    def test_full_burst_reconstruction_seat_0_opponent_burst(self) -> None:
        # Seat 0 scenario: agent placed settle1+road1, then opponent's
        # atomic 4-action burst happens. The recorder reconstructs two
        # ReplayStep state snapshots by splitting the burst.
        prev = _state(
            settlements_a=(50,),
            roads_a=(60,),
            player_a_vp=1,
        )
        post = _state(
            settlements_a=(50,),
            roads_a=(60,),
            player_a_vp=1,
            settlements_b=(17, 4),
            roads_b=(23, 11),
        )
        (s1, r1), (s2, r2) = split_burst_two_placements(
            actor="player_b",
            prev_snapshot=prev,
            post_snapshot=post,
        )
        assert (s1, r1) == (17, 23)
        assert (s2, r2) == (4, 11)

        mid = synthesize_intermediate_setup_snapshot(
            actor="player_b",
            post_snapshot=post,
            first_settle_idx=s1,
            first_road_idx=r1,
        )
        # Mid snapshot: player_b has only the first placement;
        # player_a is unchanged from post.
        assert mid.settlements["player_b"] == (17,)
        assert mid.roads["player_b"] == (23,)
        assert mid.settlements["player_a"] == (50,)
        assert mid.roads["player_a"] == (60,)
        # The recorder uses ``post`` as the state_after for the second
        # step; it's already the engine-captured "after burst" state.
