"""Unit tests for the labeling scenario generator (plan §A + §8.1).

The generator drives the snake-draft state machine:
  Pick 1 (P1) → Pick 2 (P2) → Pick 3 (P2) → Pick 4 (P1) → done

Pins:
- Determinism per seed.
- prior_picks length matches snake-draft position.
- Legal masks match `compute_action_masks` exactly.
- `apply()` advances state correctly; double-apply errors.
- After 4 applies, `current()` returns None.
- Snake-draft acting player assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator


class TestDeterminism:
    """Same seed → same board → same legal masks (the load-bearing
    invariant for the JSONL → NPZ pipeline).
    """

    def test_same_seed_produces_same_legal_settlements_at_pick_1(self) -> None:
        a = ScenarioGenerator(seed=42)
        b = ScenarioGenerator(seed=42)
        s_a = a.current()
        s_b = b.current()
        assert s_a is not None and s_b is not None
        assert np.array_equal(s_a.legal_settlement_corners, s_b.legal_settlement_corners)

    def test_different_seeds_produce_different_boards(self) -> None:
        a = ScenarioGenerator(seed=42)
        b = ScenarioGenerator(seed=43)
        # Boards differ because resource shuffles + token shuffles differ.
        s_a = a.current()
        s_b = b.current()
        assert s_a is not None and s_b is not None
        # Tile resource types should differ between the two boards.
        tiles_a = [s_a.game.board.hexTileDict[i].resource_type for i in range(19)]
        tiles_b = [s_b.game.board.hexTileDict[i].resource_type for i in range(19)]
        assert tiles_a != tiles_b


class TestSnakeDraftPositions:
    def test_pick_1_has_empty_prior_picks(self) -> None:
        g = ScenarioGenerator(seed=42)
        s = g.current()
        assert s is not None
        assert s.draft_position == 1
        assert s.prior_picks == []

    def test_pick_1_acting_player_is_p1(self) -> None:
        g = ScenarioGenerator(seed=42)
        s = g.current()
        assert s is not None
        assert s.acting_player_idx == 0  # P1

    def test_pick_2_has_one_prior_pick(self) -> None:
        g = ScenarioGenerator(seed=42)
        _advance_one(g)
        s = g.current()
        assert s is not None
        assert s.draft_position == 2
        assert len(s.prior_picks) == 1
        assert s.acting_player_idx == 1  # P2

    def test_pick_3_has_two_prior_picks_acting_p2(self) -> None:
        g = ScenarioGenerator(seed=42)
        _advance_one(g)
        _advance_one(g)
        s = g.current()
        assert s is not None
        assert s.draft_position == 3
        assert len(s.prior_picks) == 2
        # Snake draft 1-2-2-1: pick 3 is also P2.
        assert s.acting_player_idx == 1

    def test_pick_4_has_three_prior_picks_acting_p1(self) -> None:
        g = ScenarioGenerator(seed=42)
        for _ in range(3):
            _advance_one(g)
        s = g.current()
        assert s is not None
        assert s.draft_position == 4
        assert len(s.prior_picks) == 3
        assert s.acting_player_idx == 0

    def test_after_four_applies_done(self) -> None:
        g = ScenarioGenerator(seed=42)
        for _ in range(4):
            _advance_one(g)
        assert g.current() is None


class TestLegalMasks:
    def test_legal_mask_has_at_least_one_legal_corner_at_each_pick(self) -> None:
        g = ScenarioGenerator(seed=42)
        for _ in range(4):
            s = g.current()
            assert s is not None
            assert s.legal_settlement_corners.dtype == bool
            assert s.legal_settlement_corners.shape == (54,)
            assert s.legal_settlement_corners.any(), (
                f"pick {s.draft_position}: no legal settlements"
            )
            _advance_one(g)

    def test_legal_road_edges_after_settlement_constrained_to_adjacent(self) -> None:
        g = ScenarioGenerator(seed=42)
        s = g.current()
        assert s is not None
        legal = np.where(s.legal_settlement_corners)[0]
        settlement_idx = int(legal[0])
        edges = s.compute_legal_road_edges(settlement_idx)
        assert edges.dtype == bool
        assert edges.shape == (72,)
        assert edges.any(), "no legal roads after settlement"
        # Sanity: very few edges should be legal (only those incident to
        # the newly placed settlement). For setup, this is 2-3 edges.
        assert edges.sum() <= 5


class TestApplyAdvancesState:
    def test_apply_with_illegal_settlement_raises(self) -> None:
        # At pick 2, the opponent's pick 1 settlement and its two
        # adjacent vertices are now illegal — pick one of those.
        g = ScenarioGenerator(seed=42)
        _advance_one(g)  # complete pick 1
        s = g.current()
        assert s is not None
        # The vertex P1 just settled on is now illegal at pick 2.
        prev_pick = s.prior_picks[0]
        illegal = prev_pick.settlement_vertex
        assert not s.legal_settlement_corners[illegal], (
            "test setup invariant: prior-pick vertex must be illegal at pick 2"
        )
        legal = np.where(s.legal_settlement_corners)[0]
        edges = s.compute_legal_road_edges(int(legal[0]))
        road = int(np.where(edges)[0][0])
        with pytest.raises(ValueError, match="illegal"):
            g.apply(illegal, road)

    def test_apply_with_out_of_range_settlement_raises(self) -> None:
        g = ScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="out of range"):
            g.apply(54, 0)
        with pytest.raises(ValueError, match="out of range"):
            g.apply(-1, 0)

    def test_apply_with_illegal_road_raises(self) -> None:
        g = ScenarioGenerator(seed=42)
        s = g.current()
        assert s is not None
        legal = np.where(s.legal_settlement_corners)[0]
        settlement = int(legal[0])
        edges = s.compute_legal_road_edges(settlement)
        illegal_road = int(np.where(~edges)[0][0])
        with pytest.raises(ValueError, match="illegal"):
            g.apply(settlement, illegal_road)

    def test_double_apply_to_done_state_raises(self) -> None:
        g = ScenarioGenerator(seed=42)
        for _ in range(4):
            _advance_one(g)
        with pytest.raises(RuntimeError):
            g.apply(0, 0)


class TestPick:
    def test_pick_dataclass_roundtrip_dict(self) -> None:
        p = Pick(player=0, settlement_vertex=10, road_edge=20)
        d = p.to_dict()
        recovered = Pick.from_dict(d)
        assert recovered == p

    def test_pick_dict_only_has_three_keys(self) -> None:
        p = Pick(player=0, settlement_vertex=10, road_edge=20)
        assert set(p.to_dict().keys()) == {"player", "settlement_vertex", "road_edge"}


# Helpers -------------------------------------------------------------


def _advance_one(gen: ScenarioGenerator) -> None:
    """Apply a legal (first-available) settlement + road to advance one pick."""
    s = gen.current()
    assert s is not None, "no current scenario to advance"
    legal_corners = np.where(s.legal_settlement_corners)[0]
    settlement = int(legal_corners[0])
    edges = s.compute_legal_road_edges(settlement)
    road = int(np.where(edges)[0][0])
    gen.apply(settlement, road)


class TestRecordedPicksMatchApplied:
    """The prior_picks list at pick N+1 must equal what was applied at picks 1..N."""

    def test_prior_picks_match_applied_actions(self) -> None:
        g = ScenarioGenerator(seed=42)
        applied: list[Pick] = []
        for _ in range(3):
            s = g.current()
            assert s is not None
            legal = int(np.where(s.legal_settlement_corners)[0][0])
            edges = s.compute_legal_road_edges(legal)
            road = int(np.where(edges)[0][0])
            applied.append(Pick(s.acting_player_idx, legal, road))
            g.apply(legal, road)
        # Now at pick 4, prior_picks should equal applied.
        s_final = g.current()
        assert s_final is not None
        assert s_final.prior_picks == applied
