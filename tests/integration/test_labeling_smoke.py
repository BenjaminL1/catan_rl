"""End-to-end smoke test for the setup labeling tool (plan §8.5).

Drives the full LabelingUI through a complete snake-draft worth of
scenarios using synthetic mouse + keyboard events. Asserts:
- All four picks of a board are recorded.
- The next scenario after the 4th pick comes from a fresh board.
- JSONL rows are well-formed and reloadable.

Runs headless via SDL_VIDEODRIVER=dummy (the test fixture sets it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import load_scenarios
from catan_rl.labeling.ui import LabelingUI


def _first_legal_settlement_pixel(scenario: Any) -> tuple[int, tuple[int, int]]:
    """(idx, pixel) of the first legal settlement vertex."""
    idx = int(np.where(scenario.legal_settlement_corners)[0][0])
    px = scenario._idx_to_vertex_pixel[idx]
    return idx, (int(px.x), int(px.y))


def _first_legal_road_midpoint(scenario: Any, settlement_idx: int) -> tuple[int, tuple[int, int]]:
    edges = scenario.compute_legal_road_edges(settlement_idx)
    idx = int(np.where(edges)[0][0])
    v1, v2 = scenario._idx_to_edge_pixel_pair[idx]
    return idx, ((int(v1.x) + int(v2.x)) // 2, (int(v1.y) + int(v2.y)) // 2)


@pytest.mark.integration
def test_full_draft_via_simulated_clicks(tmp_path: Path) -> None:
    """Simulate 4 scenarios via direct UI driver calls (not pygame events).

    We exercise the click → state-machine → submit path that mouse
    events would trigger, without booting the full event loop. This is
    the load-bearing integration coverage; the pygame event loop itself
    is thin glue and is exercised by the manual pre-labeling smoke
    (plan §STOP/RESUME).
    """
    session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=2026)
    session.start()

    ui = LabelingUI(session=session, screen_size=(1100, 900))

    expected_picks: list[tuple[int, int, int]] = []  # (draft_pos, settle, road)
    for _ in range(5):  # 4 picks on board 1 + 1 pick on board 2
        scenario = session.current_scenario()
        assert scenario is not None
        s_idx, s_px = _first_legal_settlement_pixel(scenario)
        ui._handle_click(s_px)  # click settlement
        e_idx, e_px = _first_legal_road_midpoint(scenario, s_idx)
        ui._handle_click(e_px)  # click road
        # Trigger submit (using direct call rather than synthetic
        # KEYDOWN — same path the keyboard handler takes).
        ui._try_submit()
        expected_picks.append((scenario.draft_position, s_idx, e_idx))

    session.quit()
    ui._pygame.quit()

    rows = load_scenarios(tmp_path / "scenarios.jsonl")
    assert len(rows) == 5

    # First 4 rows have draft positions 1..4; the 5th is draft 1 again
    # on a fresh board.
    assert [r["draft_position"] for r in rows] == [1, 2, 3, 4, 1]
    seeds = [r["game_seed"] for r in rows]
    assert seeds[0] == seeds[1] == seeds[2] == seeds[3]
    assert seeds[4] != seeds[0]

    # Picks match what the UI was driven to choose.
    for row, (draft_pos, s_idx, e_idx) in zip(rows, expected_picks, strict=True):
        assert row["draft_position"] == draft_pos
        assert row["settlement_vertex"] == s_idx
        assert row["road_edge"] == e_idx
        assert row["archetype"] == "balanced"
        assert row["labeler_id"] == "ben"


@pytest.mark.integration
def test_skip_jumps_to_fresh_board(tmp_path: Path) -> None:
    session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=2026)
    session.start()
    ui = LabelingUI(session=session, screen_size=(1100, 900))
    seed1 = session.current_scenario().game_seed
    ui._skip()
    seed2 = session.current_scenario().game_seed
    assert seed1 != seed2
    # Skip writes nothing to JSONL.
    scenarios_path = tmp_path / "scenarios.jsonl"
    assert not scenarios_path.exists() or scenarios_path.stat().st_size == 0
    session.quit()
    ui._pygame.quit()


@pytest.mark.integration
def test_archetype_keypress_updates_current_archetype(tmp_path: Path) -> None:
    import pygame

    session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=2026)
    session.start()
    ui = LabelingUI(session=session, screen_size=(1100, 900))

    class FakeEvent:
        def __init__(self, key: int, unicode: str) -> None:
            self.key = key
            self.unicode = unicode

    # Press 'o' → OWS.
    ui._handle_keydown(FakeEvent(pygame.K_o, "o"))
    assert ui.current_archetype is Archetype.OWS
    # Press 'r' → road builder.
    ui._handle_keydown(FakeEvent(pygame.K_r, "r"))
    assert ui.current_archetype is Archetype.ROAD_BUILDER
    # Press 'z' → unknown, no change.
    ui._handle_keydown(FakeEvent(pygame.K_z, "z"))
    assert ui.current_archetype is Archetype.ROAD_BUILDER
    # Press 'q' → quit signal.
    cont = ui._handle_keydown(FakeEvent(pygame.K_q, "q"))
    assert cont is False
    session.quit()
    ui._pygame.quit()


@pytest.mark.integration
def test_render_does_not_crash_on_fresh_board(tmp_path: Path) -> None:
    session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=2026)
    session.start()
    ui = LabelingUI(session=session, screen_size=(1100, 900))
    # Just call render — if any of the engine attributes / pixel coords
    # are wrong, this throws.
    ui._render()
    session.quit()
    ui._pygame.quit()
