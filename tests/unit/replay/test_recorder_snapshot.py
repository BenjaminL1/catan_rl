"""Tests for `catan_rl.replay.recorder.snapshot_step_state` (Phase 2a).

Covers:
1. Shape: returned StepStateSnapshot has every required field.
2. Deep-copy isolation: mutating engine state after capture does not
   alter the snapshot dataclass.
3. Round-trip: the StepStateSnapshot can be serialised via the
   io.save_replay path and reloaded via load_replay (cross-checks the
   schema + IO contract end-to-end).
4. Seat-to-actor mapping: snapshots respect the agent_seat
   convention.
"""

from __future__ import annotations

import queue as _q
import random

import numpy as np
import pytest

from catan_rl.engine.game import catanGame
from catan_rl.engine.player import player as PlainPlayer
from catan_rl.replay.recorder import snapshot_step_state
from catan_rl.replay.schema import PlayerStateSnapshot, StepStateSnapshot


@pytest.fixture
def fresh_game() -> catanGame:
    random.seed(0)
    np.random.seed(0)
    game = catanGame(render_mode=None)
    p_a = PlainPlayer("Agent", "black")
    p_b = PlainPlayer("Opponent", "darkslateblue")
    p_a.game = game
    p_b.game = game
    game.playerQueue = _q.Queue(2)
    game.playerQueue.put(p_a)
    game.playerQueue.put(p_b)
    return game


_SEAT_TO_ACTOR_0 = {"Agent": "player_a", "Opponent": "player_b"}
_SEAT_TO_ACTOR_1 = {"Agent": "player_b", "Opponent": "player_a"}


class TestSnapshotShape:
    def test_returns_step_state_snapshot(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        assert isinstance(snap, StepStateSnapshot)

    def test_both_actors_present(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        assert set(snap.players.keys()) == {"player_a", "player_b"}
        assert isinstance(snap.players["player_a"], PlayerStateSnapshot)

    def test_empty_buildings_at_start(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        for actor in ("player_a", "player_b"):
            assert snap.settlements[actor] == ()
            assert snap.cities[actor] == ()
            assert snap.roads[actor] == ()

    def test_robber_hex_is_valid(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        assert 0 <= snap.robber_hex < 19

    def test_last_seven_roller_initially_none(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        assert snap.last_seven_roller is None


class TestDeepCopyIsolation:
    def test_mutating_engine_does_not_alter_snapshot(self, fresh_game: catanGame) -> None:
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        agent = next(iter(fresh_game.playerQueue.queue))
        agent.resources["WOOD"] += 99
        agent.devCards["KNIGHT"] = agent.devCards.get("KNIGHT", 0) + 5
        agent.victoryPoints += 10
        snap_p_a = snap.players["player_a"]
        assert snap_p_a.resources["WOOD"] != 99
        assert snap_p_a.dev_cards_hand["KNIGHT"] == 0
        assert snap_p_a.vp == 0


class TestSeatSwap:
    def test_seat_1_swaps_actor_labels(self, fresh_game: catanGame) -> None:
        # With seat=1, the engine's "Opponent" is player_a in the JSON.
        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_1,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )
        # player_a should be the engine's Opponent under seat=1.
        assert snap.players["player_a"].name == "Opponent"
        assert snap.players["player_b"].name == "Agent"


class TestSnapshotRoundTripsThroughReplaySchema:
    def test_snapshot_fits_in_replay_step(self, fresh_game: catanGame) -> None:
        # Construct a minimal Replay around the snapshot and verify it
        # round-trips through save_replay/load_replay end-to-end.
        from pathlib import Path

        from catan_rl.replay import (
            REPLAY_SCHEMA_VERSION,
            BoardStatic,
            HexStatic,
            Metadata,
            PlayerSpec,
            PortStatic,
            Replay,
            ReplayStep,
            VertexStatic,
            load_replay,
            save_replay,
        )

        snap = snapshot_step_state(
            fresh_game,
            seat_to_actor=_SEAT_TO_ACTOR_0,
            vertex_pixel_to_idx={},
            edge_key_to_idx={},
        )

        replay = Replay(
            schema_version=REPLAY_SCHEMA_VERSION,
            metadata=Metadata(
                player_a=PlayerSpec(kind="heuristic", ckpt_path=None, color="black", seat_index=0),
                player_b=PlayerSpec(
                    kind="random",
                    ckpt_path=None,
                    color="darkslateblue",
                    seat_index=1,
                ),
                seed=0,
                max_turns=400,
                intended_hex_size=(1000, 800),
                recorded_at_utc="2026-06-04T00:00:00Z",
                winner=None,
                winner_seat=None,
                final_vp=(0, 0),
                total_steps=1,
                partial=False,
            ),
            board_static=BoardStatic(
                hexes=(
                    HexStatic(
                        hex_idx=0,
                        q=0,
                        r=0,
                        resource="DESERT",
                        number_token=None,
                        has_robber_initial=True,
                    ),
                ),
                vertices=(VertexStatic(vertex_idx=0, adjacent_hex_indices=(0,)),),
                edges=(),
                ports=(
                    PortStatic(
                        port_idx=0,
                        vertex_idx_pair=(0, 0),
                        ratio="3:1",
                        resource=None,
                    ),
                ),
            ),
            steps=(
                ReplayStep(
                    step_idx=0,
                    kind="setup",
                    actor="player_a",
                    dice_roll=None,
                    actions=(),
                    events=(),
                    log_lines=(),
                    state_after=snap,
                ),
            ),
        )

        path = Path("/tmp/_snapshot_roundtrip_test.json")
        if path.exists():
            path.unlink()
        save_replay(replay, path)
        loaded = load_replay(path, strict=True)
        assert loaded.steps[0].state_after == snap
        path.unlink()
