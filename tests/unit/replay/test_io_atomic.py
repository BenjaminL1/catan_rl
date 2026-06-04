"""Tests for `catan_rl.replay.io`.

Covers:
1. Round-trip a constructed Replay: save_replay → load_replay → equal.
2. Atomic write: existing file unchanged when torch/json/replace
   raises mid-write; tmp file cleaned up.
3. `force=False` refuses to overwrite existing files.
4. Read raises ReplaySchemaError on every error class (missing,
   malformed, missing schema_version, forward-incompatible).
5. UnknownEvent round-trips through save/load when strict=False.
6. Migration chain runs on read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_rl.replay.io import load_replay, save_replay
from catan_rl.replay.migrations import (
    register_migration,
    registered_versions,
    unregister_migration,
)
from catan_rl.replay.schema import (
    REPLAY_SCHEMA_VERSION,
    BoardStatic,
    EdgeStatic,
    GameEnd,
    HexStatic,
    LargestArmyChange,
    LongestRoadChange,
    Metadata,
    Monopoly,
    PlayerSpec,
    PlayerStateSnapshot,
    PortStatic,
    Replay,
    ReplaySchemaError,
    ReplayStep,
    Robber,
    Steal,
    StepStateSnapshot,
    SubAction,
    VertexStatic,
)

# ---------------------------------------------------------------------------
# Fixtures: small fully-typed Replay
# ---------------------------------------------------------------------------


def _empty_player_snapshot(name: str, vp: int) -> PlayerStateSnapshot:
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


def _minimal_state_after() -> StepStateSnapshot:
    return StepStateSnapshot(
        settlements={"player_a": (), "player_b": ()},
        cities={"player_a": (), "player_b": ()},
        roads={"player_a": (), "player_b": ()},
        robber_hex=9,
        players={
            "player_a": _empty_player_snapshot("Agent", 0),
            "player_b": _empty_player_snapshot("Opponent", 0),
        },
        longest_road_holder=None,
        largest_army_holder=None,
    )


def _minimal_board_static() -> BoardStatic:
    return BoardStatic(
        hexes=(
            HexStatic(
                hex_idx=0,
                q=0,
                r=0,
                resource="WOOD",
                number_token=11,
                has_robber_initial=False,
            ),
            HexStatic(
                hex_idx=1,
                q=1,
                r=0,
                resource="DESERT",
                number_token=None,
                has_robber_initial=True,
            ),
        ),
        vertices=(
            VertexStatic(vertex_idx=0, adjacent_hex_indices=(0,)),
            VertexStatic(vertex_idx=1, adjacent_hex_indices=(0, 1)),
        ),
        edges=(EdgeStatic(edge_idx=0, v1_idx=0, v2_idx=1),),
        ports=(PortStatic(port_idx=0, vertex_idx_pair=(0, 1), ratio="3:1", resource=None),),
    )


def _minimal_metadata() -> Metadata:
    return Metadata(
        player_a=PlayerSpec(kind="heuristic", ckpt_path=None, color="red", seat_index=0),
        player_b=PlayerSpec(kind="random", ckpt_path=None, color="blue", seat_index=1),
        seed=42,
        max_turns=400,
        intended_hex_size=(1000, 800),
        recorded_at_utc="2026-06-04T07:00:00Z",
        winner="player_a",
        winner_seat=0,
        final_vp=(15, 11),
        total_steps=2,
        partial=False,
    )


def _replay_with_events() -> Replay:
    """Three-step replay exercising every event variant + sub-actions."""
    step0 = ReplayStep(
        step_idx=0,
        kind="setup",
        actor="player_a",
        dice_roll=None,
        actions=(
            SubAction(kind="BuildSettlement", args={"vertex_idx": 0}),
            SubAction(kind="BuildRoad", args={"edge_idx": 0}),
        ),
        events=(),
        log_lines=("player_a placed first settlement",),
        state_after=_minimal_state_after(),
    )
    step1 = ReplayStep(
        step_idx=1,
        kind="main",
        actor="player_b",
        dice_roll=(3, 4),
        actions=(
            SubAction(kind="RollDice", args={"d1": 3, "d2": 4}),
            SubAction(kind="PlayMonopoly", args={"resource": "WHEAT"}),
            SubAction(kind="EndTurn"),
        ),
        events=(
            Monopoly(player="player_b", resource="WHEAT", count=3),
            LongestRoadChange(prev_owner=None, new_owner="player_b", length=5),
            LargestArmyChange(prev_owner=None, new_owner="player_a", knights=3),
            Robber(player="player_b", hex_idx=4),
            Steal(robber="player_b", victim="player_a", resource="ORE"),
        ),
        log_lines=("player_b rolled (3, 4)", "Monopoly on WHEAT (3 taken)"),
        state_after=_minimal_state_after(),
    )
    step2 = ReplayStep(
        step_idx=2,
        kind="terminal",
        actor="player_a",
        dice_roll=(6, 5),
        actions=(SubAction(kind="EndTurn"),),
        events=(GameEnd(winner="player_a", vp_breakdown={"player_a": 15, "player_b": 11}),),
        log_lines=("player_a wins with 15 VP",),
        state_after=_minimal_state_after(),
    )
    return Replay(
        schema_version=REPLAY_SCHEMA_VERSION,
        metadata=_minimal_metadata(),
        board_static=_minimal_board_static(),
        steps=(step0, step1, step2),
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_known_events_roundtrip(self, tmp_path: Path) -> None:
        original = _replay_with_events()
        path = tmp_path / "r.json"
        save_replay(original, path)
        loaded = load_replay(path, strict=True)
        assert loaded == original

    def test_minimal_replay_roundtrips(self, tmp_path: Path) -> None:
        replay = Replay(
            schema_version=REPLAY_SCHEMA_VERSION,
            metadata=_minimal_metadata(),
            board_static=_minimal_board_static(),
            steps=(
                ReplayStep(
                    step_idx=0,
                    kind="terminal",
                    actor="player_a",
                    dice_roll=None,
                    actions=(),
                    events=(),
                    log_lines=(),
                    state_after=_minimal_state_after(),
                ),
            ),
        )
        path = tmp_path / "min.json"
        save_replay(replay, path)
        loaded = load_replay(path, strict=True)
        assert loaded == replay


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_existing_dest_survives_mid_write_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replay = _replay_with_events()
        dest = tmp_path / "r.json"
        # First, write a valid file we'll try to clobber.
        save_replay(replay, dest)
        good_bytes = dest.read_bytes()

        # Patch json.dumps inside the io module to raise mid-write.
        from catan_rl.replay import io as io_mod

        def _boom(*_args, **_kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr(io_mod, "json", type("J", (), {"dumps": _boom}))
        with pytest.raises(OSError, match="simulated disk full"):
            save_replay(replay, dest, force=True)
        # Original file untouched.
        assert dest.read_bytes() == good_bytes
        # Tmp from the failed save must be cleaned up.
        assert not (tmp_path / "r.json.tmp").exists()

    def test_force_false_refuses_existing_file(self, tmp_path: Path) -> None:
        replay = _replay_with_events()
        dest = tmp_path / "r.json"
        save_replay(replay, dest)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            save_replay(replay, dest, force=False)

    def test_force_true_overwrites(self, tmp_path: Path) -> None:
        first = _replay_with_events()
        # Tweak the second replay so we can tell it apart.
        second = Replay(
            schema_version=REPLAY_SCHEMA_VERSION,
            metadata=Metadata(
                player_a=first.metadata.player_a,
                player_b=first.metadata.player_b,
                seed=99,  # different seed
                max_turns=first.metadata.max_turns,
                intended_hex_size=first.metadata.intended_hex_size,
                recorded_at_utc=first.metadata.recorded_at_utc,
                winner=first.metadata.winner,
                winner_seat=first.metadata.winner_seat,
                final_vp=first.metadata.final_vp,
                total_steps=first.metadata.total_steps,
                partial=first.metadata.partial,
            ),
            board_static=first.board_static,
            steps=first.steps,
        )
        dest = tmp_path / "r.json"
        save_replay(first, dest)
        save_replay(second, dest, force=True)
        loaded = load_replay(dest, strict=True)
        assert loaded.metadata.seed == 99


# ---------------------------------------------------------------------------
# Read error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ReplaySchemaError, match="not found"):
            load_replay(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json")
        with pytest.raises(ReplaySchemaError, match="malformed JSON"):
            load_replay(bad)

    def test_non_dict_top_level_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.json"
        bad.write_text("[1, 2, 3]")
        with pytest.raises(ReplaySchemaError, match="top-level must be a dict"):
            load_replay(bad)

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "flat.json"
        bad.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ReplaySchemaError, match="missing 'schema_version'"):
            load_replay(bad)

    def test_forward_incompatible_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "future.json"
        bad.write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ReplaySchemaError, match="only supports up to v"):
            load_replay(bad)

    def test_null_in_required_field_raises_replay_schema_error(self, tmp_path: Path) -> None:
        # A corrupt or v2 replay with ``"intended_hex_size": null``
        # would otherwise raise raw TypeError from inside tuple(None).
        # Confirm it surfaces as ReplaySchemaError.
        replay = _replay_with_events()
        path = tmp_path / "r.json"
        save_replay(replay, path)
        raw = json.loads(path.read_text())
        raw["metadata"]["intended_hex_size"] = None
        path.write_text(json.dumps(raw))
        with pytest.raises(ReplaySchemaError, match="malformed"):
            load_replay(path)


class TestForwardCompatTopLevel:
    def test_unknown_top_level_field_silently_dropped(self, tmp_path: Path) -> None:
        # v1 reader on a v2 file with an unknown top-level metadata
        # field must load fine and drop the unknown key.
        replay = _replay_with_events()
        path = tmp_path / "r.json"
        save_replay(replay, path)
        raw = json.loads(path.read_text())
        raw["metadata"]["future_v2_field"] = "abc"
        raw["future_v2_top_level"] = [1, 2, 3]
        path.write_text(json.dumps(raw))
        loaded = load_replay(path, strict=True)
        # Loads cleanly, the unknown fields aren't accessible.
        assert loaded.metadata.seed == replay.metadata.seed


# ---------------------------------------------------------------------------
# Forward-compat: UnknownEvent round-trips through file
# ---------------------------------------------------------------------------


class TestUnknownEventOnDisk:
    def test_unknown_event_in_file_loads_in_lenient_mode(self, tmp_path: Path) -> None:
        # Hand-craft a JSON file with an unknown event kind so we
        # don't have to invent a v2-aware writer just for this test.
        replay = _replay_with_events()
        path = tmp_path / "r.json"
        save_replay(replay, path)
        # Mutate the JSON to inject an unknown event into step 1.
        raw = json.loads(path.read_text())
        raw["steps"][1]["events"].append(
            {"kind": "FutureV2Marker", "extra_field": "from_v2", "n": 5}
        )
        path.write_text(json.dumps(raw))

        loaded = load_replay(path, strict=False)
        last_step_events = loaded.steps[1].events
        from catan_rl.replay.schema import UnknownEvent

        unknowns = [e for e in last_step_events if isinstance(e, UnknownEvent)]
        assert len(unknowns) == 1
        assert unknowns[0].original_kind == "FutureV2Marker"
        assert unknowns[0].payload["extra_field"] == "from_v2"
        assert unknowns[0].payload["n"] == 5

    def test_unknown_event_strict_load_raises(self, tmp_path: Path) -> None:
        replay = _replay_with_events()
        path = tmp_path / "r.json"
        save_replay(replay, path)
        raw = json.loads(path.read_text())
        raw["steps"][1]["events"].append({"kind": "FutureV2Marker"})
        path.write_text(json.dumps(raw))
        with pytest.raises(ReplaySchemaError, match="unknown event kind"):
            load_replay(path, strict=True)


# ---------------------------------------------------------------------------
# Migration chain on read
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_registry():
    """Snapshot/restore migration registry."""
    before = registered_versions()
    yield
    after = registered_versions()
    for v in after:
        if v not in before:
            unregister_migration(v)


class TestMigrationChainOnRead:
    def test_migration_runs_before_dataclass_decode(
        self, tmp_path: Path, _clean_registry: None
    ) -> None:
        # Pretend we're reading a v0 file. Register a v0→v1 migration
        # that adds the missing fields. Verify load_replay walks the
        # chain before instantiating dataclasses.
        v1 = _replay_with_events()
        path = tmp_path / "v0.json"
        save_replay(v1, path)
        raw = json.loads(path.read_text())
        raw["schema_version"] = 0  # downgrade marker
        path.write_text(json.dumps(raw))

        def v0_to_v1(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1}

        register_migration(0, v0_to_v1)
        loaded = load_replay(path, strict=True)
        assert loaded.schema_version == 1
