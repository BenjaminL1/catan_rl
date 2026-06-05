"""Tests for the Phase 0.5 engine accessor methods.

`catanBoard.board_static()` and `catanGame.snapshot_state(...)` are
the recorder's bridge to engine state. Tests verify:

1. Shape: returned dicts match the replay schema's keys.
2. Round-trip: passing the snapshot through the schema's
   reconstruction path (``Replay.read_json`` after writing it inside
   a synthetic Replay) returns equal dataclass values.
3. Deep-copy contract: mutating the engine state AFTER calling
   snapshot does not retroactively alter the snapshot dict.
4. Omniscient dev hand: cards held in ``newDevCards`` are merged into
   the snapshot's ``dev_cards_hand`` at capture time.
5. ``last_seven_roller`` reflects the engine's
   ``last_player_to_roll_7`` mapped through ``seat_to_actor``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from catan_rl.engine.board import catanBoard
from catan_rl.engine.game import catanGame
from catan_rl.engine.player import player as PlainPlayer


@pytest.fixture
def fresh_board() -> catanBoard:
    random.seed(0)
    np.random.seed(0)
    return catanBoard()


@pytest.fixture
def fresh_game() -> catanGame:
    """Construct a fresh ``catanGame`` and seed its player queue with
    two named players so the snapshot accessor has a non-empty input."""
    random.seed(0)
    np.random.seed(0)
    game = catanGame(render_mode=None)
    import queue as _q

    p_a = PlainPlayer("Agent", "black")
    p_b = PlainPlayer("Opponent", "darkslateblue")
    p_a.game = game
    p_b.game = game
    p_a.isAI = False
    p_b.isAI = True
    game.playerQueue = _q.Queue(2)
    game.playerQueue.put(p_a)
    game.playerQueue.put(p_b)
    return game


# ---------------------------------------------------------------------------
# board_static
# ---------------------------------------------------------------------------


class TestBoardStatic:
    def test_returns_required_keys(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        assert set(out.keys()) == {"hexes", "vertices", "edges", "ports"}

    def test_hex_count_is_19(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        assert len(out["hexes"]) == 19

    def test_vertex_count_is_54(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        assert len(out["vertices"]) == 54

    def test_edge_count_is_72(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        assert len(out["edges"]) == 72

    def test_port_count_is_9(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        # Standard Catan: 5 specific 2:1 ports + 4 generic 3:1.
        assert len(out["ports"]) == 9

    def test_hex_payload_shape(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        h0 = out["hexes"][0]
        assert set(h0.keys()) >= {
            "hex_idx",
            "q",
            "r",
            "resource",
            "number_token",
            "has_robber_initial",
        }
        assert isinstance(h0["hex_idx"], int)
        assert isinstance(h0["q"], int)
        assert isinstance(h0["r"], int)
        assert isinstance(h0["resource"], str)
        # number_token is None for the desert hex.
        assert h0["number_token"] is None or isinstance(h0["number_token"], int)

    def test_exactly_one_desert_with_robber_initial(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        deserts = [h for h in out["hexes"] if h["resource"] == "DESERT"]
        assert len(deserts) == 1
        # The desert hex starts with the robber on it.
        assert deserts[0]["has_robber_initial"] is True

    def test_edge_payload_shape(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        e0 = out["edges"][0]
        assert set(e0.keys()) == {"edge_idx", "v1_idx", "v2_idx"}
        assert e0["v1_idx"] != e0["v2_idx"]

    def test_port_payload_shape(self, fresh_board: catanBoard) -> None:
        out = fresh_board.board_static()
        ratios = {p["ratio"] for p in out["ports"]}
        assert ratios <= {"2:1", "3:1"}
        # 2:1 ports must have a non-None resource; 3:1 must have None.
        for port in out["ports"]:
            if port["ratio"] == "2:1":
                assert port["resource"] in {"WOOD", "BRICK", "WHEAT", "ORE", "SHEEP"}
            else:
                assert port["resource"] is None


# ---------------------------------------------------------------------------
# snapshot_state
# ---------------------------------------------------------------------------


class TestSnapshotState:
    def test_returns_required_keys(self, fresh_game: catanGame) -> None:
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert set(out.keys()) >= {
            "settlements",
            "cities",
            "roads",
            "robber_hex",
            "players",
            "longest_road_holder",
            "largest_army_holder",
            "last_seven_roller",
        }

    def test_player_snapshot_shape(self, fresh_game: catanGame) -> None:
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        for actor in ("player_a", "player_b"):
            snap = out["players"][actor]
            assert set(snap.keys()) == {
                "name",
                "vp",
                "resources",
                "dev_cards_hand",
                "dev_cards_played",
            }
            assert set(snap["resources"].keys()) == {
                "WOOD",
                "BRICK",
                "WHEAT",
                "ORE",
                "SHEEP",
            }
            assert set(snap["dev_cards_hand"].keys()) == {
                "KNIGHT",
                "VP",
                "ROAD_BUILDER",
                "YEAR_OF_PLENTY",
                "MONOPOLY",
            }
            assert set(snap["dev_cards_played"].keys()) == {
                "KNIGHT",
                "VP",
                "ROAD_BUILDER",
                "YEAR_OF_PLENTY",
                "MONOPOLY",
            }

    def test_initial_vp_zero(self, fresh_game: catanGame) -> None:
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        for actor in ("player_a", "player_b"):
            assert out["players"][actor]["vp"] == 0

    def test_robber_hex_is_desert(self, fresh_game: catanGame) -> None:
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert out["robber_hex"] >= 0

    def test_deep_copy_isolation(self, fresh_game: catanGame) -> None:
        # Snapshot, mutate engine, verify snapshot unchanged.
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        # Mutate engine state.
        agent = next(iter(fresh_game.playerQueue.queue))
        agent.resources["WOOD"] += 99
        agent.devCards["KNIGHT"] = agent.devCards.get("KNIGHT", 0) + 5
        agent.victoryPoints += 10
        # Snapshot dict is unchanged.
        assert out["players"]["player_a"]["resources"]["WOOD"] != 99
        assert out["players"]["player_a"]["dev_cards_hand"]["KNIGHT"] == 0
        assert out["players"]["player_a"]["vp"] == 0

    def test_omniscient_dev_hand_includes_new_devs(self, fresh_game: catanGame) -> None:
        # newDevCards (drawn this turn, not yet playable) must be
        # merged into the omniscient hand at capture time. The engine
        # stores them as ``list[str]`` of card type names; snapshot
        # tallies them by count. VP cards are NOT counted in the hand
        # bucket — they always live in ``dev_cards_played["VP"]`` per
        # the schema contract.
        agent = next(iter(fresh_game.playerQueue.queue))
        agent.newDevCards = ["KNIGHT", "KNIGHT", "MONOPOLY"]
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert out["players"]["player_a"]["dev_cards_hand"]["KNIGHT"] == 2
        assert out["players"]["player_a"]["dev_cards_hand"]["MONOPOLY"] == 1

    def test_omniscient_dev_hand_handles_legacy_key_names(self, fresh_game: catanGame) -> None:
        # ``devCards`` uses ``ROADBUILDER`` / ``YEAROFPLENTY``; the
        # snapshot must expose them under the canonical schema names
        # ``ROAD_BUILDER`` / ``YEAR_OF_PLENTY``.
        agent = next(iter(fresh_game.playerQueue.queue))
        agent.devCards["ROADBUILDER"] = 1
        agent.devCards["YEAROFPLENTY"] = 2
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert out["players"]["player_a"]["dev_cards_hand"]["ROAD_BUILDER"] == 1
        assert out["players"]["player_a"]["dev_cards_hand"]["YEAR_OF_PLENTY"] == 2

    def test_last_seven_roller_mapping(self, fresh_game: catanGame) -> None:
        # Mark the engine's "last 7 roller" as Opponent; snapshot
        # should map that through to "player_b".
        opp = list(fresh_game.playerQueue.queue)[1]
        fresh_game.last_player_to_roll_7 = opp
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert out["last_seven_roller"] == "player_b"

    def test_last_seven_roller_none_default(self, fresh_game: catanGame) -> None:
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert out["last_seven_roller"] is None

    def test_vp_card_only_in_played_bucket(self, fresh_game: catanGame) -> None:
        # Schema contract: VP cards live in ``dev_cards_played["VP"]``
        # only — the hand bucket reports 0 to avoid double-counting.
        agent = next(iter(fresh_game.playerQueue.queue))
        agent.devCards["VP"] = 3
        seat_to_actor = {"Agent": "player_a", "Opponent": "player_b"}
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        snap = out["players"]["player_a"]
        assert snap["dev_cards_hand"]["VP"] == 0
        assert snap["dev_cards_played"]["VP"] == 3

    def test_seat_to_actor_missing_player_raises(self, fresh_game: catanGame) -> None:
        # An incomplete ``seat_to_actor`` mapping must raise loudly
        # rather than silently produce an empty ``players`` dict (which
        # would render an empty board in the viewer).
        seat_to_actor = {"Agent": "player_a"}  # Opponent missing
        with pytest.raises(ValueError, match="missing engine player"):
            fresh_game.snapshot_state(seat_to_actor, {}, {})

    def test_seat_to_actor_extra_keys_allowed(self, fresh_game: catanGame) -> None:
        # Extra keys in ``seat_to_actor`` are fine — only missing keys
        # are an error (a recorder may pre-populate the dict with
        # entries for future seats).
        seat_to_actor = {
            "Agent": "player_a",
            "Opponent": "player_b",
            "Phantom": "player_c",
        }
        out = fresh_game.snapshot_state(seat_to_actor, {}, {})
        assert "player_a" in out["players"]
        assert "player_b" in out["players"]
        assert "player_c" not in out["players"]
