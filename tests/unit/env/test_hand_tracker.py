"""Unit tests for BroadcastHandTracker.

Strategy: drive synthetic broadcast events directly against the tracker
for the per-event-type unit tests, then drive real engine games for the
end-to-end "tracker stays in sync after every event" invariant. The
synthetic tests pin the wire format; the integration tests pin the
engine's emission completeness.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.engine.broadcast import GameBroadcast
from catan_rl.engine.game import catanGame
from catan_rl.env.hand_tracker import RESOURCES_CW, BroadcastHandTracker

# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


def test_construct_with_two_player_names() -> None:
    tracker = BroadcastHandTracker(["Agent", "Opponent"])
    assert tracker.get_hand("Agent") == dict.fromkeys(RESOURCES_CW, 0)
    assert tracker.get_hand("Opponent") == dict.fromkeys(RESOURCES_CW, 0)


def test_construct_rejects_empty_names() -> None:
    with pytest.raises(ValueError):
        BroadcastHandTracker([])


def test_construct_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        BroadcastHandTracker(["A", "A"])


def test_get_hand_unknown_player_returns_zeros() -> None:
    tracker = BroadcastHandTracker(["Agent"])
    assert tracker.get_hand("Ghost") == dict.fromkeys(RESOURCES_CW, 0)


def test_total_sums_hand() -> None:
    tracker = BroadcastHandTracker(["A"])
    tracker._hands["A"]["WOOD"] = 3
    tracker._hands["A"]["WHEAT"] = 2
    assert tracker.total("A") == 5


# ---------------------------------------------------------------------------
# Subscription idempotency
# ---------------------------------------------------------------------------


def test_subscribe_is_idempotent() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    tracker.subscribe(bus)  # second call should not double-register
    bus.resource_change("A", {"WOOD": 1}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 1, "double-subscribe should not double-count"


def test_subscribe_switching_bus_unsubscribes_old() -> None:
    bus_a = GameBroadcast()
    bus_b = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus_a)
    tracker.subscribe(bus_b)
    bus_a.resource_change("A", {"WOOD": 1}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 0, "tracker should have left bus_a"
    bus_b.resource_change("A", {"WOOD": 1}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 1, "tracker should be listening on bus_b"


def test_unsubscribe_stops_receiving() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": 2}, "TEST")
    tracker.unsubscribe()
    bus.resource_change("A", {"WOOD": 5}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 2


def test_unsubscribe_safe_when_not_subscribed() -> None:
    tracker = BroadcastHandTracker(["A"])
    tracker.unsubscribe()  # no exception


def test_reset_zeros_hands_preserves_subscription() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": 3}, "TEST")
    tracker.reset()
    assert tracker.get_hand("A")["WOOD"] == 0
    # Subscription preserved: new events should still land.
    bus.resource_change("A", {"WOOD": 1}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 1


# ---------------------------------------------------------------------------
# Event handling — positive / negative deltas, clamping, unknown players
# ---------------------------------------------------------------------------


def test_positive_delta_increases_hand() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": 2, "WHEAT": 1}, "DICE")
    assert tracker.get_hand("A") == {"WOOD": 2, "BRICK": 0, "WHEAT": 1, "ORE": 0, "SHEEP": 0}


def test_negative_delta_decreases_hand() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": 3}, "DICE")
    bus.resource_change("A", {"WOOD": -1}, "BUILD_ROAD")
    assert tracker.get_hand("A")["WOOD"] == 2


def test_negative_delta_clamps_at_zero() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": -5}, "BUG_OVERDRAW")
    assert tracker.get_hand("A")["WOOD"] == 0


def test_unknown_player_events_ignored() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("Ghost", {"WOOD": 1}, "TEST")
    assert tracker.get_hand("A") == dict.fromkeys(RESOURCES_CW, 0)


def test_non_resource_events_ignored() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.dice_roll("A", 6)
    bus.discard("A", ["WOOD"])
    bus.year_of_plenty("A", ["WOOD", "BRICK"])
    bus.emit("CUSTOM_TYPE", player="A", payload={"WOOD": 999})
    assert tracker.get_hand("A") == dict.fromkeys(RESOURCES_CW, 0)


def test_unknown_resources_in_delta_ignored() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    bus.resource_change("A", {"WOOD": 1, "GOLD": 99}, "TEST")
    assert tracker.get_hand("A")["WOOD"] == 1
    assert "GOLD" not in tracker.get_hand("A")


def test_malformed_event_payload_ignored() -> None:
    bus = GameBroadcast()
    tracker = BroadcastHandTracker(["A"])
    tracker.subscribe(bus)
    # Missing keys / wrong types must not crash the tracker.
    bus.emit("RESOURCE_CHANGE")  # no player, no delta
    bus.emit("RESOURCE_CHANGE", player="A", delta=None)
    bus.emit("RESOURCE_CHANGE", player=123, delta={"WOOD": 1})
    assert tracker.get_hand("A") == dict.fromkeys(RESOURCES_CW, 0)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class _MockPlayer:
    def __init__(self, name: str, resources: dict[str, int]) -> None:
        self.name = name
        self.resources = resources


def test_seed_from_player_writes_hand_in_cw_order() -> None:
    tracker = BroadcastHandTracker(["Agent", "Opponent"])
    player = _MockPlayer("Agent", {"WOOD": 1, "BRICK": 2, "WHEAT": 3, "ORE": 4, "SHEEP": 5})
    tracker.seed_from_player(player)
    assert tracker.get_hand("Agent") == {
        "WOOD": 1,
        "BRICK": 2,
        "WHEAT": 3,
        "ORE": 4,
        "SHEEP": 5,
    }


def test_seed_from_player_unknown_name_is_noop() -> None:
    tracker = BroadcastHandTracker(["Agent"])
    tracker.seed_from_player(_MockPlayer("Ghost", {"WOOD": 5}))
    assert tracker.get_hand("Agent") == dict.fromkeys(RESOURCES_CW, 0)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_passes_when_matched() -> None:
    tracker = BroadcastHandTracker(["A"])
    tracker._hands["A"] = {"WOOD": 1, "BRICK": 0, "WHEAT": 2, "ORE": 0, "SHEEP": 3}
    player = _MockPlayer("A", {"WOOD": 1, "WHEAT": 2, "SHEEP": 3})
    tracker.verify_against_players({"A": player})  # no exception


def test_verify_raises_on_drift_with_diff() -> None:
    tracker = BroadcastHandTracker(["A"])
    tracker._hands["A"]["WOOD"] = 2
    player = _MockPlayer("A", {"WOOD": 5})
    with pytest.raises(AssertionError) as exc:
        tracker.verify_against_players({"A": player})
    assert "WOOD" in str(exc.value)
    assert "diff" in str(exc.value).lower()


def test_verify_skips_unknown_players() -> None:
    tracker = BroadcastHandTracker(["A"])
    # players_by_name has only "B"; tracker tracks "A"; should silently skip.
    tracker.verify_against_players({"B": _MockPlayer("B", {"WOOD": 1})})


# ---------------------------------------------------------------------------
# Integration — track an actual headless game end-to-end
# ---------------------------------------------------------------------------


def test_trade_with_same_give_and_receive_resource_emits_correct_delta() -> None:
    """Regression: trade_with_bank with r1 == r2 must emit a net delta.

    Discovered while developing :class:`BroadcastHandTracker`. The engine
    previously used a dict-literal ``{r1: -give, r2: +1}`` which collapsed
    to ``{r1: +1}`` when r1 == r2, silently corrupting downstream trackers.
    The fix sums into a dict to preserve the net.
    """

    game = catanGame(render_mode=None)
    agent = next(iter(game.playerQueue.queue))
    agent.resources = dict.fromkeys(("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"), 0)
    agent.resources["WHEAT"] = 4  # 4:1 trade available

    tracker = BroadcastHandTracker([agent.name])
    tracker.subscribe(game.broadcast)
    tracker.seed_from_player(agent)

    agent.trade_with_bank("WHEAT", "WHEAT", game.board)  # degenerate but legal

    # Actual: -4 + 1 = -3 → 4 - 3 = 1 WHEAT
    assert agent.resources["WHEAT"] == 1
    assert tracker.get_hand(agent.name)["WHEAT"] == 1, (
        "tracker drifted on r1==r2 trade — engine emit must produce a "
        "summed delta, not a dict literal that collapses keys"
    )


def test_tracks_full_game_under_random_play() -> None:
    """End-to-end: the tracker matches both players' resources after every event.

    Drives the v2 CatanEnv (which already wires the engine + headless setup
    + RandomAIPlayer for the opponent) and verifies the tracker stays in
    sync after every step. If the engine ever emits a resource change
    without a RESOURCE_CHANGE broadcast event, this test catches the drift.
    """
    from catan_rl.env.catan_env import CatanEnv

    rng = np.random.default_rng(7)
    env = CatanEnv(opponent_type="random", max_turns=200)
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    assert env.game is not None and env.agent_player is not None and env.opponent_player is not None

    tracker = BroadcastHandTracker([env.agent_player.name, env.opponent_player.name])
    tracker.subscribe(env.game.broadcast)
    # Seed from the current player resources *after* setup (which already
    # ran during env.reset() — those events fired before we subscribed).
    tracker.seed_from_player(env.agent_player)
    tracker.seed_from_player(env.opponent_player)

    players_by_name = {
        env.agent_player.name: env.agent_player,
        env.opponent_player.name: env.opponent_player,
    }

    # Play up to 500 steps, verifying after each. The smoke-gate test
    # already established that random-vs-random terminates within ~1600
    # steps on average; 500 is enough to exercise every event type.
    for _ in range(500):
        masks = env.get_action_masks()
        legal_types = np.flatnonzero(masks["type"])
        t = int(rng.choice(legal_types))
        action = np.zeros(6, dtype=np.int64)
        action[0] = t
        for head_idx, key in enumerate(
            ("corner_settlement", "edge", "tile", "resource1_default", "resource2_default"),
            start=1,
        ):
            # Pick a legal value if the head has any; else 0 (irrelevant
            # heads' values are ignored by the env per the relevance map).
            if key == "corner_settlement" and t == 1:
                key = "corner_city"
            if key == "resource1_default" and t == 11:
                key = "resource1_discard"
            elif key == "resource1_default" and t == 10:
                key = "resource1_trade"
            legal = np.flatnonzero(masks[key])
            if legal.size:
                action[head_idx] = int(rng.choice(legal))

        _, _, terminated, truncated, _ = env.step(action)
        tracker.verify_against_players(players_by_name)
        if terminated or truncated:
            break
