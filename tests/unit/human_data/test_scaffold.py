"""Scaffold tests for the human-data pipeline (build brief §3, §6).

Test-first guarantees for the ``scaffold`` slice:

- ``GameRecord`` (de)serializes round-trip and always carries ``schema_version``.
- the committed ``topology.json`` package fixture has the standard-board
  19 / 54 / 72 / 9 counts.
- the committed game-1 golden CI fixtures exist and load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_rl.human_data import (
    SCHEMA_VERSION,
    FFmpegNotFoundError,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
    load_topology,
    resolve_ffmpeg,
)
from catan_rl.human_data.topology import (
    NUM_EDGES,
    NUM_HEXES,
    NUM_PORTS,
    NUM_VERTICES,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"


def _sample_record() -> GameRecord:
    """A representative ``GameRecord`` built from the game-1 spike result."""
    hexes = [{"hex_id": i, "resource": "ORE", "number": 8} for i in range(18)]
    hexes.append({"hex_id": 18, "resource": "DESERT", "number": None})
    return GameRecord(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="known_window", confidence=0.8),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=tuple(hexes),
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        openings={
            "ThePhantom": PlayerOpening(settlements=(4, 10), roads=(7, 20)),
            "rayman147": PlayerOpening(settlements=(20, 0), roads=(34, 2)),
        },
        dice_log=(8, 6, 11, 4),
        winner="ThePhantom",
        episode_source="natural",
        passed_crosscheck=True,
        provenance={"resolution": 1080, "ts": 247},
        rejection_reason=None,
    )


def test_game_record_roundtrip_via_dict() -> None:
    rec = _sample_record()
    restored = GameRecord.from_dict(rec.to_dict())
    assert restored == rec


def test_game_record_roundtrip_via_json_line() -> None:
    rec = _sample_record()
    line = rec.to_json_line()
    assert "\n" not in line
    restored = GameRecord.from_json_line(line)
    assert restored == rec


def test_game_record_schema_version_present() -> None:
    rec = _sample_record()
    assert rec.schema_version == SCHEMA_VERSION == 1
    payload = rec.to_dict()
    assert payload["schema_version"] == 1
    assert json.loads(rec.to_json_line())["schema_version"] == 1


def test_game_record_resources_are_string_literals() -> None:
    """Resources are bare strings (no enum) — desert carries number=None."""
    payload = _sample_record().to_dict()
    desert = payload["board"]["hexes"][-1]
    assert desert["resource"] == "DESERT"
    assert desert["number"] is None
    assert isinstance(payload["board"]["hexes"][0]["resource"], str)


def test_game_record_ports_omitted_v1() -> None:
    assert _sample_record().to_dict()["board"]["ports"] == "OMITTED in v1"


def test_game_record_rejects_newer_schema() -> None:
    payload = _sample_record().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than supported"):
        GameRecord.from_dict(payload)


def test_topology_fixture_counts() -> None:
    topo = load_topology()
    assert len(topo.hex_corner_to_vertex) == NUM_HEXES == 19
    assert len(topo.vertex_adjacent_hexes) == NUM_VERTICES == 54
    assert len(topo.edge_vertices) == NUM_EDGES == 72
    assert len(topo.port_slots) == NUM_PORTS == 9
    # Every edge is an ascending vertex pair within range.
    for a, b in topo.edge_vertices:
        assert 0 <= a < b < NUM_VERTICES


def test_golden_fixtures_exist_and_load() -> None:
    postsetup = FIXTURES / "game1_postsetup_t247.png"
    baseline = FIXTURES / "game1_empty_baseline_t105.png"
    log_crop = FIXTURES / "game1_log_crop_t120.png"
    openings = FIXTURES / "game1_openings.json"
    ocr = FIXTURES / "ocr_f1080_120.txt"
    for png in (postsetup, baseline, log_crop):
        assert png.is_file(), png
        # PNG magic bytes.
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png

    parsed = json.loads(openings.read_text(encoding="utf-8"))
    assert parsed["openings"]["ThePhantom"]["settlements"] == [4, 10]
    assert parsed["openings"]["rayman147"]["settlements"] == [20, 0]

    text = ocr.read_text(encoding="utf-8")
    # The real noisy OCR carries the "Happy settlingl" typo and setup events.
    assert "Happy settling" in text
    assert "placed a Settlement" in text


def test_resolve_ffmpeg_returns_a_usable_binary() -> None:
    """Resolves to a system or imageio-ffmpeg binary (both available in CI)."""
    path = resolve_ffmpeg()
    assert Path(path).exists()


def test_resolve_ffmpeg_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no system ffmpeg and no imageio-ffmpeg, raise with install guidance."""
    import builtins

    monkeypatch.setattr("catan_rl.human_data.ffmpeg.shutil.which", lambda _: None)
    real_import = builtins.__import__

    def _no_imageio(name: str, *args: object, **kwargs: object) -> object:
        if name == "imageio_ffmpeg":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_imageio)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg not found"):
        resolve_ffmpeg()
