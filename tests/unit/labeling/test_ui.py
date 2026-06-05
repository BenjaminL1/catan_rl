"""Unit tests for the UI state machine (plan §C + §8.4).

Tests the pure-Python state-machine class. The pygame rendering layer
is exercised by the integration smoke test in
``tests/integration/test_labeling_smoke.py`` under
SDL_VIDEODRIVER=dummy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.ui import (
    PHASE_ROAD_PICK,
    PHASE_SETTLEMENT_PICK,
    LabelingUIState,
    nearest_vertex,
)


def _make_session(tmp_path: Path) -> LabelingSession:
    s = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=42)
    s.start()
    return s


class TestNearestVertex:
    def test_returns_exact_match_at_centroid(self) -> None:
        vertex_pixels = {
            0: (100.0, 100.0),
            1: (200.0, 200.0),
            2: (50.0, 300.0),
        }
        assert nearest_vertex(100, 100, vertex_pixels, max_radius=20) == 0
        assert nearest_vertex(200, 200, vertex_pixels, max_radius=20) == 1
        assert nearest_vertex(50, 300, vertex_pixels, max_radius=20) == 2

    def test_returns_none_when_no_vertex_within_radius(self) -> None:
        vertex_pixels = {0: (100.0, 100.0)}
        assert nearest_vertex(500, 500, vertex_pixels, max_radius=20) is None

    def test_picks_closest_when_two_in_radius(self) -> None:
        vertex_pixels = {0: (100.0, 100.0), 1: (110.0, 100.0)}
        # Click at (102, 100) is 2px from vertex 0, 8px from vertex 1.
        assert nearest_vertex(102, 100, vertex_pixels, max_radius=20) == 0

    def test_respects_legal_filter(self) -> None:
        vertex_pixels = {0: (100.0, 100.0), 1: (200.0, 100.0)}
        # Click directly on vertex 0 — but it's illegal; nearest legal is vertex 1.
        result = nearest_vertex(100, 100, vertex_pixels, max_radius=200, legal={1})
        assert result == 1


class TestStateMachineTransitions:
    def test_initial_phase_is_settlement_pick(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.selected_settlement is None
        assert state.selected_road is None

    def test_select_settlement_transitions_to_road_pick(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        state.select_settlement(settlement)
        assert state.phase == PHASE_ROAD_PICK
        assert state.selected_settlement == settlement

    def test_select_illegal_settlement_rejected(self, tmp_path: Path) -> None:
        # At pick 2, the prior P1 settlement vertex is illegal.
        session = _make_session(tmp_path)
        # Advance to pick 2 by submitting a legal pick.
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        edges = scenario.compute_legal_road_edges(settlement)
        road = int(np.where(edges)[0][0])
        session.submit(
            settlement_vertex=settlement,
            road_edge=road,
            archetype=Archetype.BALANCED,
        )
        # Now at pick 2. Try clicking on the illegal vertex.
        state = LabelingUIState(session)
        s2 = session.current_scenario()
        assert s2 is not None
        illegal = s2.prior_picks[0].settlement_vertex
        accepted = state.select_settlement(illegal)
        assert accepted is False
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.last_click_rejected is True

    def test_select_road_after_settlement_transitions_to_ready(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        state.select_settlement(settlement)
        edges = scenario.compute_legal_road_edges(settlement)
        road = int(np.where(edges)[0][0])
        state.select_road(road)
        assert state.phase == PHASE_ROAD_PICK
        assert state.selected_road == road
        assert state.is_ready_to_submit()

    def test_select_road_before_settlement_rejected(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        accepted = state.select_road(0)
        assert accepted is False
        assert state.selected_road is None

    def test_undo_from_road_pick_clears_settlement(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        state.select_settlement(settlement)
        assert state.phase == PHASE_ROAD_PICK
        state.undo()
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.selected_settlement is None
        assert state.selected_road is None

    def test_undo_after_road_selected_clears_road(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        state.select_settlement(settlement)
        edges = scenario.compute_legal_road_edges(settlement)
        road = int(np.where(edges)[0][0])
        state.select_road(road)
        state.undo()
        assert state.phase == PHASE_ROAD_PICK
        assert state.selected_settlement == settlement
        assert state.selected_road is None


class TestSubmitFromState:
    def test_submit_when_not_ready_raises(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        with pytest.raises(RuntimeError):
            state.submit(archetype=Archetype.BALANCED)

    def test_submit_when_ready_writes_row_and_advances(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        state = LabelingUIState(session)
        scenario = session.current_scenario()
        assert scenario is not None
        settlement = int(np.where(scenario.legal_settlement_corners)[0][0])
        state.select_settlement(settlement)
        edges = scenario.compute_legal_road_edges(settlement)
        road = int(np.where(edges)[0][0])
        state.select_road(road)
        state.submit(archetype=Archetype.BALANCED)
        # JSONL row written.
        scenarios_path = tmp_path / "scenarios.jsonl"
        assert scenarios_path.exists()
        assert scenarios_path.read_text().count("\n") == 1
        # State reset for next scenario.
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.selected_settlement is None
        assert state.selected_road is None
        # Next scenario is pick 2.
        s2 = session.current_scenario()
        assert s2 is not None
        assert s2.draft_position == 2


class TestPortRendering:
    """Ports must be rendered — Colonist.io 1v1 has 5×2:1 + 4×3:1 ports."""

    def test_collect_port_edges_returns_nine_pairs(self, tmp_path: Path) -> None:
        from catan_rl.labeling.ui import collect_port_edges

        session = _make_session(tmp_path)
        scenario = session.current_scenario()
        assert scenario is not None
        board = scenario.game.board
        port_edges = collect_port_edges(board)
        # 9 ports total: 5 specific (2:1 of each resource) + 4 generic (3:1).
        assert len(port_edges) == 9
        for v1_idx, v2_idx, ratio, resource in port_edges:
            assert 0 <= v1_idx < 54
            assert 0 <= v2_idx < 54
            assert v1_idx != v2_idx
            assert ratio in ("2:1", "3:1")
            # Resource is None iff this is a generic 3:1 port.
            if ratio == "3:1":
                assert resource is None
            else:
                assert resource in ("BRICK", "WOOD", "WHEAT", "SHEEP", "ORE")

    def test_port_edges_cover_all_five_resources_and_generic(self, tmp_path: Path) -> None:
        from catan_rl.labeling.ui import collect_port_edges

        session = _make_session(tmp_path)
        scenario = session.current_scenario()
        assert scenario is not None
        board = scenario.game.board
        edges = collect_port_edges(board)
        ratios = [ratio for _, _, ratio, _ in edges]
        resources = {resource for _, _, _, resource in edges if resource}
        assert ratios.count("3:1") == 4, f"expected 4 generic 3:1 ports, got {ratios}"
        assert ratios.count("2:1") == 5, f"expected 5 resource 2:1 ports, got {ratios}"
        assert resources == {"BRICK", "WOOD", "WHEAT", "SHEEP", "ORE"}, (
            f"missing resources in port set: {resources}"
        )


class TestArchetypeShortcut:
    def test_archetype_first_letter_lookup(self) -> None:
        from catan_rl.labeling.ui import archetype_from_key

        assert archetype_from_key("b") is Archetype.BALANCED
        assert archetype_from_key("o") is Archetype.OWS
        assert archetype_from_key("h") is Archetype.OWS_HYBRID
        assert archetype_from_key("r") is Archetype.ROAD_BUILDER
        assert archetype_from_key("x") is Archetype.OTHER

    def test_unknown_key_returns_none(self) -> None:
        from catan_rl.labeling.ui import archetype_from_key

        assert archetype_from_key("z") is None
