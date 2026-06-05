"""Unit tests for the labeling persistence layer (plan §D + §8.3).

Pins:
- Atomic append survives crashes mid-write.
- Schema versioning is always populated.
- `repair_jsonl` truncates a malformed trailing line.
- Round-trip equality of scenarios written then read back.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from catan_rl.labeling.store import (
    SCHEMA_VERSION,
    append_scenario,
    count_scenarios,
    load_scenarios,
    repair_jsonl,
)


def _example_row() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "scenario-uuid-1",
        "session_id": "session-uuid-1",
        "labeled_at": "2026-06-01T15:30:00Z",
        "labeler_id": "test",
        "game_seed": 123,
        "draft_position": 1,
        "acting_player": 1,
        "prior_picks": [],
        "archetype": "balanced",
        "settlement_vertex": 23,
        "road_edge": 45,
        "decision_time_ms": 47200,
        "notes": "",
    }


class TestAppendScenario:
    def test_writes_jsonl_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        append_scenario(_example_row(), path)
        text = path.read_text()
        assert text.endswith("\n"), "every JSONL row must terminate with newline"
        row = json.loads(text)
        assert row["scenario_id"] == "scenario-uuid-1"

    def test_multiple_appends_produce_multiple_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for i in range(5):
            row = _example_row()
            row["scenario_id"] = f"scenario-{i}"
            append_scenario(row, path)
        lines = path.read_text().splitlines()
        assert len(lines) == 5
        recovered = [json.loads(line)["scenario_id"] for line in lines]
        assert recovered == [f"scenario-{i}" for i in range(5)]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dirs" / "scenarios.jsonl"
        append_scenario(_example_row(), path)
        assert path.exists()

    def test_rejects_missing_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        bad = _example_row()
        del bad["schema_version"]
        with pytest.raises(ValueError):
            append_scenario(bad, path)

    def test_rejects_non_serialisable(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        row = _example_row()
        row["bad"] = object()
        with pytest.raises(TypeError):
            append_scenario(row, path)


class TestLoadScenarios:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        original = _example_row()
        append_scenario(original, path)
        loaded = load_scenarios(path)
        assert len(loaded) == 1
        assert loaded[0] == original

    def test_loads_multiple_rows_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for i in range(3):
            row = _example_row()
            row["scenario_id"] = f"s-{i}"
            append_scenario(row, path)
        loaded = load_scenarios(path)
        assert [r["scenario_id"] for r in loaded] == ["s-0", "s-1", "s-2"]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        path.touch()
        assert load_scenarios(path) == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jsonl"
        assert load_scenarios(path) == []

    def test_schema_version_field_present_on_every_row(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for _ in range(3):
            append_scenario(_example_row(), path)
        for row in load_scenarios(path):
            assert "schema_version" in row
            assert row["schema_version"] == SCHEMA_VERSION


class TestCountScenarios:
    def test_count_equals_load_length(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for _ in range(7):
            append_scenario(_example_row(), path)
        assert count_scenarios(path) == 7

    def test_count_zero_on_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.jsonl"
        assert count_scenarios(path) == 0

    def test_count_zero_on_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        path.touch()
        assert count_scenarios(path) == 0


class TestRepairJsonl:
    def test_no_op_on_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for _ in range(3):
            append_scenario(_example_row(), path)
        truncated = repair_jsonl(path)
        assert truncated == 0
        assert count_scenarios(path) == 3

    def test_truncates_malformed_trailing_line(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        for _ in range(3):
            append_scenario(_example_row(), path)
        # Corrupt: append a partial JSON line as if a crash happened mid-write.
        with path.open("ab") as f:
            f.write(b'{"scenario_id": "partial"')  # no closing brace, no newline
        truncated = repair_jsonl(path)
        assert truncated > 0
        assert count_scenarios(path) == 3
        loaded = load_scenarios(path)
        assert all("scenario_id" in r for r in loaded)

    def test_truncates_partial_line_without_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        # Single partial row, no completed rows.
        path.write_bytes(b'{"foo": "bar"')
        truncated = repair_jsonl(path)
        assert truncated > 0
        assert load_scenarios(path) == []

    def test_no_op_on_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        assert repair_jsonl(path) == 0


class TestAtomicAppendUnderSigkill:
    """The load-bearing crash-safety test (plan §8.3).

    Spawns a subprocess that writes many rows. Kills it mid-write.
    Asserts: every line that exists in the file is a *complete*,
    parseable JSON object. The atomicity guarantee is: writes <= PIPE_BUF
    to an O_APPEND fd are atomic on POSIX.
    """

    def test_completed_lines_parse_after_sigkill(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        # Subprocess writes rows in a tight loop until killed.
        script = f"""
import json
import time
import sys

sys.path.insert(0, {str(Path("src").resolve())!r})
from catan_rl.labeling.store import append_scenario, SCHEMA_VERSION
from pathlib import Path

path = Path({str(path)!r})
i = 0
while True:
    row = {{
        "schema_version": SCHEMA_VERSION,
        "scenario_id": f"s-{{i}}",
        "session_id": "x",
        "labeled_at": "2026-06-01T00:00:00Z",
        "labeler_id": "t",
        "game_seed": i,
        "draft_position": 1,
        "acting_player": 1,
        "prior_picks": [],
        "archetype": "balanced",
        "settlement_vertex": 1,
        "road_edge": 1,
        "decision_time_ms": 0,
        "notes": "",
    }}
    append_scenario(row, path)
    i += 1
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.2)  # let it write some rows
        proc.kill()
        proc.wait(timeout=5)

        # After SIGKILL: every *complete* line must parse. The last line
        # may be partial — repair_jsonl removes it.
        from itertools import pairwise

        repair_jsonl(path)
        loaded = load_scenarios(path)
        # At least one row should have been written before the kill on a
        # reasonable machine; if not, the test is uninformative but not
        # wrong.
        for r in loaded:
            assert isinstance(r, dict)
            assert r.get("schema_version") == SCHEMA_VERSION
        # Row IDs should be monotonic if no torn writes.
        ids = [r["scenario_id"] for r in loaded]
        seq = [int(x.split("-")[1]) for x in ids]
        assert seq == sorted(seq), "rows should be in append order"
        for prev, curr in pairwise(seq):
            assert curr == prev + 1, "no rows should be torn or skipped"
