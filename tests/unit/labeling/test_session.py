"""Unit tests for the labeling session manager (plan §B + §8.2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.session import LabelingSession


def _first_legal_pair(scenario) -> tuple[int, int]:
    settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
    edges = scenario.compute_legal_road_edges(settlement)
    road = int(np.where(edges)[0][0])
    return settlement, road


class TestManifest:
    def test_start_writes_manifest_with_required_fields(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        manifest_path = tmp_path / "sessions" / session.session_id / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        for field in (
            "session_id",
            "start_time",
            "labeler_id",
            "scenarios_completed",
        ):
            assert field in manifest, f"manifest missing field {field}"
        assert manifest["labeler_id"] == "ben"
        assert manifest["scenarios_completed"] == 0

    def test_quit_finalizes_manifest_with_end_time(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        session.quit()
        manifest_path = tmp_path / "sessions" / session.session_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "end_time" in manifest
        assert manifest["end_time"] >= manifest["start_time"]


class TestSubmit:
    def test_submit_appends_jsonl_row(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        scenario = session.current_scenario()
        assert scenario is not None
        settlement, road = _first_legal_pair(scenario)

        session.submit(
            settlement_vertex=settlement,
            road_edge=road,
            archetype=Archetype.BALANCED,
            notes="",
            decision_time_ms=1000,
        )

        scenarios_path = tmp_path / "scenarios.jsonl"
        assert scenarios_path.exists()
        rows = scenarios_path.read_text().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert row["settlement_vertex"] == settlement
        assert row["road_edge"] == road
        assert row["draft_position"] == 1
        assert row["archetype"] == "balanced"
        assert row["labeler_id"] == "ben"
        assert row["session_id"] == session.session_id
        assert row["game_seed"] is not None
        assert row["prior_picks"] == []

    def test_submit_advances_scenario(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        s1 = session.current_scenario()
        assert s1 is not None
        settlement, road = _first_legal_pair(s1)
        session.submit(
            settlement_vertex=settlement,
            road_edge=road,
            archetype=Archetype.BALANCED,
        )
        s2 = session.current_scenario()
        assert s2 is not None
        assert s2.draft_position == 2
        assert len(s2.prior_picks) == 1

    def test_submit_after_full_draft_starts_new_board(self, tmp_path: Path) -> None:
        """After the 4th pick on a board, a new board's pick 1 appears."""
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        seeds_seen: list[int] = []
        for _ in range(5):
            scenario = session.current_scenario()
            assert scenario is not None
            seeds_seen.append(scenario.game_seed)
            settlement, road = _first_legal_pair(scenario)
            session.submit(
                settlement_vertex=settlement,
                road_edge=road,
                archetype=Archetype.BALANCED,
            )
        # First four picks share a seed; the fifth pick must come from a
        # new board (different seed) at draft_position 1.
        assert seeds_seen[0] == seeds_seen[1] == seeds_seen[2] == seeds_seen[3]
        assert seeds_seen[4] != seeds_seen[0]

    def test_submit_increments_scenarios_completed(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        for i in range(3):
            scenario = session.current_scenario()
            assert scenario is not None
            settlement, road = _first_legal_pair(scenario)
            session.submit(
                settlement_vertex=settlement,
                road_edge=road,
                archetype=Archetype.BALANCED,
            )
            assert session.scenarios_completed == i + 1


class TestSkip:
    def test_skip_does_not_append_jsonl(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        session.skip()
        scenarios_path = tmp_path / "scenarios.jsonl"
        assert not scenarios_path.exists() or scenarios_path.stat().st_size == 0

    def test_skip_advances_to_new_board(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        s1 = session.current_scenario()
        assert s1 is not None
        seed1 = s1.game_seed
        session.skip()
        s2 = session.current_scenario()
        assert s2 is not None
        # Skip jumps to a fresh board (skip discards the whole draft).
        assert s2.game_seed != seed1
        assert s2.draft_position == 1

    def test_skip_does_not_count_toward_completed(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        assert session.scenarios_completed == 0
        session.skip()
        assert session.scenarios_completed == 0


class TestResume:
    def test_resume_restores_scenario_count_after_quit(self, tmp_path: Path) -> None:
        # First session: label 2 scenarios then quit.
        session1 = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session1.start()
        for _ in range(2):
            scenario = session1.current_scenario()
            assert scenario is not None
            settlement, road = _first_legal_pair(scenario)
            session1.submit(
                settlement_vertex=settlement,
                road_edge=road,
                archetype=Archetype.BALANCED,
            )
        session1.quit()

        # Second session sees 2 scenarios already in JSONL.
        session2 = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=43)
        assert session2.total_scenarios_in_dataset() == 2
        session2.start()
        session2.quit()

    def test_resume_repairs_corrupt_trailing_jsonl_line(self, tmp_path: Path) -> None:
        # Pre-corrupt a partial row in the JSONL.
        scenarios_path = tmp_path / "scenarios.jsonl"
        scenarios_path.parent.mkdir(parents=True, exist_ok=True)
        scenarios_path.write_bytes(b'{"partial": "row"')  # malformed, no newline
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        # repair_jsonl should have wiped the partial row.
        assert scenarios_path.stat().st_size == 0


class TestUnlimitedSessions:
    """Plan §B: sessions run indefinitely until quit (no auto-cap)."""

    def test_can_label_many_scenarios_in_one_session(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        for _ in range(8):  # 2 boards' worth
            scenario = session.current_scenario()
            assert scenario is not None
            settlement, road = _first_legal_pair(scenario)
            session.submit(
                settlement_vertex=settlement,
                road_edge=road,
                archetype=Archetype.BALANCED,
            )
        # Still has more scenarios available.
        assert session.current_scenario() is not None


class TestSessionTimer:
    def test_session_elapsed_seconds_monotonic(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
        session.start()
        t1 = session.elapsed_seconds()
        time.sleep(0.05)
        t2 = session.elapsed_seconds()
        assert t2 > t1
        assert t2 >= 0.05
