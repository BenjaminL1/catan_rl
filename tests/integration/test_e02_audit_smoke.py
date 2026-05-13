"""Smoke test for the E0.2 heuristic-distribution audit script.

Runs 3 games via the script's private API and asserts the output JSON
has the expected keys and value ranges. Smoke-only — the full 1000-game
audit lives in ``scripts/audit_heuristic_distribution.py`` and is run
manually to produce the calibration data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "audit_heuristic_distribution",
        Path(__file__).resolve().parents[2] / "scripts" / "audit_heuristic_distribution.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The dataclass decorator looks up cls.__module__ in sys.modules during
    # class construction. We must register before exec_module so that lookup
    # succeeds; otherwise dataclass.__init_subclass__ raises with a confusing
    # 'NoneType has no __dict__' error.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_audit_three_games_produces_expected_keys(audit_module, tmp_path: Path) -> None:
    records = [audit_module._play_one_game(seed=i, max_turns=200) for i in range(3)]
    summary = audit_module._aggregate(records)

    for key in (
        "metadata",
        "BASE_WR_HEUR_SELF",
        "BASE_TOP1_FREQ",
        "BASE_NLL_FREQ",
        "main_type_marginal",
        "type_head_entropy_main",
        "schema",
    ):
        assert key in summary, f"missing key: {key}"

    wr = summary["BASE_WR_HEUR_SELF"]
    for k in ("p1_seat", "p2_seat", "symmetrised"):
        assert k in wr
        assert 0.0 <= wr[k] <= 1.0, f"WR {k}={wr[k]} out of [0,1]"

    metadata = summary["metadata"]
    assert metadata["n_games"] == 3
    assert metadata["p1_wins"] + metadata["p2_wins"] + metadata["n_truncated"] >= 0
    assert metadata["total_decisions"] > 0
    assert metadata["setup_phase_decisions"] >= 24  # 3 games × 8 setup actions

    schema = summary["schema"]
    assert schema["resource_order"] == ["WOOD", "BRICK", "WHEAT", "ORE", "SHEEP"]
    assert schema["N_ACTION_TYPES"] == 13

    # Spot-check JSON serialisability.
    out_path = tmp_path / "smoke.json"
    out_path.write_text(json.dumps(summary, indent=2))
    parsed = json.loads(out_path.read_text())
    assert parsed["metadata"]["n_games"] == 3


def test_decision_records_have_valid_indices(audit_module) -> None:
    """Each recorded action has indices in their schema-valid range."""
    record = audit_module._play_one_game(seed=42, max_turns=150)
    from catan_rl.policy.obs_schema import (
        N_ACTION_TYPES,
        N_EDGES,
        N_RESOURCES,
        N_TILES,
        N_VERTICES,
    )

    for d in record.decisions:
        assert 0 <= d.action_type < N_ACTION_TYPES, f"bad type: {d.action_type}"
        if d.corner_idx >= 0:
            assert 0 <= d.corner_idx < N_VERTICES
        if d.edge_idx >= 0:
            assert 0 <= d.edge_idx < N_EDGES
        if d.tile_idx >= 0:
            assert 0 <= d.tile_idx < N_TILES
        if d.resource1_idx >= 0:
            assert 0 <= d.resource1_idx < N_RESOURCES
        if d.resource2_idx >= 0:
            assert 0 <= d.resource2_idx < N_RESOURCES
        assert d.phase in ("setup", "main", "roll", "robber", "discard")
        assert d.player_seat in (0, 1)


def test_instrumentation_restores_original_methods(audit_module) -> None:
    """After a game, the heuristic player's class methods are NOT shadowed."""
    from catan_rl.agents.heuristic import heuristicAIPlayer

    audit_module._play_one_game(seed=7, max_turns=100)
    # The class itself must not be mutated.
    p = heuristicAIPlayer("Test", "black")
    p.updateAI()
    # bound-method should resolve to the class implementation, not a patched closure.
    assert p.build_settlement.__func__ is heuristicAIPlayer.build_settlement
    assert p.trade_with_bank.__func__ is heuristicAIPlayer.trade_with_bank
