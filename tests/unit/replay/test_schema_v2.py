"""Schema v2 pins: ``policy_internals``, the ``Metadata`` provenance flags,
the ``"human"`` player kind, and the no-op v1 → v2 migration.

``load_replay`` documents that unknown NESTED keys are silently dropped, so a
loose JSON key would be deleted by the first load→save cycle — hence the typed
field. These tests pin that it survives a round-trip, that v1 files still load,
and that the schema/IO path stays torch-free.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from catan_rl.replay.io import load_replay, save_replay
from catan_rl.replay.schema import (
    REPLAY_SCHEMA_VERSION,
    Metadata,
    PlayerSpec,
    PolicyInternals,
    Replay,
)

from .test_io_atomic import _minimal_board_static, _minimal_metadata, _replay_with_events


def _internals(seed: int = 0) -> PolicyInternals:
    probs = [0.0] * 13
    probs[3] = 0.97
    probs[0] = 0.03
    return PolicyInternals(
        type_mask=tuple(i in (0, 3) for i in range(13)),
        type_probs=tuple(probs),
        chosen_action=(3, 0, 0, 0, 0, 0),
        value=0.42,
        corner_top=((17, 0.6), (18, 0.4)),
        edge_top=((2, 1.0),),
        tile_top=((9, 0.5), (10, 0.5)),
        belief_logits=(0.1, -0.2, 0.3, 0.0),
    )


def _replay_with_internals() -> Replay:
    base = _replay_with_events()
    steps = list(base.steps)
    steps[1] = dataclasses.replace(steps[1], policy_internals=(_internals(),))
    meta = dataclasses.replace(
        base.metadata, mode="search", sims=400, clairvoyant=True, reveal_bot=True
    )
    return dataclasses.replace(base, metadata=meta, steps=tuple(steps))


class TestRoundTrip:
    def test_policy_internals_survive_save_load(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        save_replay(_replay_with_internals(), path)
        loaded = load_replay(path, strict=True)
        got = loaded.steps[1].policy_internals
        assert len(got) == 1
        assert got[0] == _internals()
        # Steps without internals keep the empty default.
        assert loaded.steps[0].policy_internals == ()

    def test_load_save_load_is_not_lossy(self, tmp_path: Path) -> None:
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        save_replay(_replay_with_internals(), first)
        save_replay(load_replay(first, strict=True), second)
        assert load_replay(second, strict=True) == load_replay(first, strict=True)

    def test_metadata_provenance_flags_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        save_replay(_replay_with_internals(), path)
        meta = load_replay(path, strict=True).metadata
        assert (meta.mode, meta.sims, meta.clairvoyant, meta.reveal_bot) == (
            "search",
            400,
            True,
            True,
        )

    def test_human_kind_is_a_legal_record_value(self, tmp_path: Path) -> None:
        base = _replay_with_events()
        meta = dataclasses.replace(
            base.metadata,
            player_b=PlayerSpec(kind="human", ckpt_path=None, color="blue", seat_index=1),
        )
        path = tmp_path / "human.json"
        save_replay(dataclasses.replace(base, metadata=meta), path)
        assert load_replay(path, strict=True).metadata.player_b.kind == "human"


class TestV1Compat:
    def test_a_v1_payload_still_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "v1.json"
        save_replay(_replay_with_events(), path)
        raw = json.loads(path.read_text())
        # Strip v2 back down to a genuine v1 payload.
        raw["schema_version"] = 1
        for key in ("mode", "sims", "clairvoyant", "reveal_bot"):
            raw["metadata"].pop(key)
        for step in raw["steps"]:
            step.pop("policy_internals")
        path.write_text(json.dumps(raw))

        loaded = load_replay(path, strict=True)
        assert loaded.schema_version == REPLAY_SCHEMA_VERSION == 2
        assert all(s.policy_internals == () for s in loaded.steps)
        assert loaded.metadata.mode == "raw_policy"
        assert loaded.metadata.sims is None
        assert loaded.metadata.clairvoyant is False
        assert loaded.metadata.reveal_bot is False

    def test_the_bump_shipped_with_its_migration(self) -> None:
        from catan_rl.replay.migrations import apply_migrations, registered_versions

        assert REPLAY_SCHEMA_VERSION - 1 in registered_versions()
        assert apply_migrations({"schema_version": 1, "x": 1}, target_version=2) == {
            "schema_version": 2,
            "x": 1,
        }


class TestNoTorch:
    def test_schema_and_io_import_no_torch(self) -> None:
        # A fresh subprocess: importing the schema/IO path must not drag torch
        # in, or the viewer's import contract breaks.
        code = (
            "import sys; import catan_rl.replay.schema, catan_rl.replay.io; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
        )
        import subprocess

        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


class TestSize:
    def test_top_8_internals_stay_within_2x_the_base(self, tmp_path: Path) -> None:
        # A dense 54/72/19 pointer dump is ~168 floats/decision and roughly
        # doubles the file on its own; top-8 must cost far less than that.
        base = _replay_with_events()
        long_steps = tuple(
            dataclasses.replace(base.steps[i % len(base.steps)], step_idx=i) for i in range(260)
        )
        plain = dataclasses.replace(base, steps=long_steps)
        rich = dataclasses.replace(
            base,
            steps=tuple(
                dataclasses.replace(s, policy_internals=(_internals(),)) for s in long_steps
            ),
        )
        p1, p2 = tmp_path / "plain.json", tmp_path / "rich.json"
        save_replay(plain, p1)
        save_replay(rich, p2)
        assert p2.stat().st_size < 2 * p1.stat().st_size


def test_helpers_are_importable() -> None:
    # Guard the cross-module fixture import above.
    assert _minimal_metadata().seed >= 0
    assert len(_minimal_board_static().hexes) >= 1


@pytest.mark.parametrize("kind", ["random", "heuristic", "policy", "human"])
def test_all_player_kinds_construct(kind: str) -> None:
    spec = PlayerSpec(kind=kind, ckpt_path=None, color="red", seat_index=0)  # type: ignore[arg-type]
    assert spec.kind == kind


def test_metadata_defaults_are_backwards_compatible() -> None:
    # The v2 fields must all be defaulted, or every existing keyword
    # construction of Metadata breaks.
    meta = Metadata(
        player_a=PlayerSpec(kind="policy", ckpt_path="x", color="black", seat_index=0),
        player_b=PlayerSpec(kind="human", ckpt_path=None, color="blue", seat_index=1),
        seed=1,
        max_turns=10,
        intended_hex_size=(1000, 800),
        recorded_at_utc="2026-07-27T00:00:00",
        winner=None,
        winner_seat=None,
        final_vp=(0, 0),
        total_steps=0,
        partial=False,
    )
    assert (meta.mode, meta.sims, meta.clairvoyant, meta.reveal_bot) == (
        "raw_policy",
        None,
        False,
        False,
    )
