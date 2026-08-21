"""Hand-computed fixtures for the D1 setup-scorer feature block.

Acceptance criterion 2: the arithmetic is PINNED, not smoke-tested. Every
expected number below is derived by hand from the seed-42 board's chip layout,
which is printed inline so a future reader can re-derive it without running
anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.scenario_gen import ScenarioGenerator
from catan_rl.setup_phase.scorer import ScorerVersionError, ScorerWeights, rank_of, top_k
from catan_rl.setup_phase.scorer_features import (
    FEATURE_VERSION,
    N_SETTLEMENT_FEATURES,
    PILOT_FEATURE_NAMES,
    SETTLEMENT_FEATURE_NAMES,
    SetupContext,
    all_settlement_features,
    board_resource_pips,
    edge_vertex_pairs,
    road_far_endpoint,
    road_features,
    settlement_features,
)

SEED = 42


@pytest.fixture()
def ctx() -> SetupContext:
    gen = ScenarioGenerator(seed=SEED)
    scenario = gen.current()
    assert scenario is not None
    return SetupContext.build(
        gen._board,
        scenario.prior_picks,
        int(scenario.acting_player_idx),
        scenario.legal_settlement_corners,
    )


def _feat(ctx: SetupContext, vertex: int, name: str) -> float:
    return float(settlement_features(ctx, vertex)[SETTLEMENT_FEATURE_NAMES.index(name)])


class TestSettlementArithmetic:
    def test_vertex_0_is_desert_ore4_wheat5(self, ctx: SetupContext) -> None:
        # Seed-42 vertex 0 touches DESERT(-), ORE(4 -> 3 dots), WHEAT(5 -> 4 dots).
        row = settlement_features(ctx, 0)
        assert row.shape == (N_SETTLEMENT_FEATURES,)
        # Charlesworth column order: WOOD, BRICK, WHEAT, ORE, SHEEP.
        assert list(row[:5]) == [0.0, 0.0, 4.0, 3.0, 0.0]
        assert _feat(ctx, 0, "pips_total") == 7.0
        assert _feat(ctx, 0, "n_distinct_resources") == 2.0
        assert _feat(ctx, 0, "n_adjacent_hexes") == 3.0  # the desert still counts as a hex
        # No prior picks: every produced resource is new for both seats.
        assert _feat(ctx, 0, "n_new_resources") == 2.0
        assert _feat(ctx, 0, "opponent_new_resources") == 2.0
        # First settlement of the draft -> the starting-material term is off.
        assert _feat(ctx, 0, "n_hexes_x_second") == 0.0

    def test_opponent_best_margin_is_zero_at_the_best_vertex(self, ctx: SetupContext) -> None:
        # Vertex 18 = WHEAT(5)+BRICK(8)+WOOD(10) = 4+5+3 dots; under the PINNED
        # Charlesworth weights that is the highest-value legal vertex on this board.
        assert ctx.opponent_best_vertex == 18
        assert _feat(ctx, 18, "opponent_best_margin") == 0.0
        # Vertex 0's base value is 3*1.1 (ore) + 4*1.0 (wheat) = 7.3; 7.3 - 12.0.
        assert _feat(ctx, 0, "opponent_best_margin") == pytest.approx(-4.7)

    def test_adjacency_block_fires_only_next_to_the_opponent_target(
        self, ctx: SetupContext
    ) -> None:
        assert _feat(ctx, 0, "adjacency_block") == 0.0
        neighbour = ctx.adjacency[18][0]
        assert _feat(ctx, neighbour, "adjacency_block") == 1.0

    def test_scarcity_starve_needs_a_scarce_resource_the_opponent_lacks(
        self, ctx: SetupContext
    ) -> None:
        # Seed-42 board pips per resource: WOOD 14, BRICK 10, WHEAT 14, ORE 6,
        # SHEEP 14 -> ORE is the unique scarcest.
        assert list(board_resource_pips(ctx.board)) == [14.0, 10.0, 14.0, 6.0, 14.0]
        assert list(ctx.scarce_mask) == [False, False, False, True, False]
        assert _feat(ctx, 0, "scarcity_starve") == 1.0  # vertex 0 produces ORE
        assert _feat(ctx, 18, "scarcity_starve") == 0.0  # vertex 18 does not

    def test_port_flags(self, ctx: SetupContext) -> None:
        # Vertex 25 carries the 3:1 port; vertex 50 carries 2:1 ORE but produces
        # WOOD + BRICK, so the matched-port term stays off — that separation is
        # the whole point of splitting the port flags.
        assert _feat(ctx, 25, "port_any") == 1.0
        assert _feat(ctx, 25, "port_3to1") == 1.0
        assert _feat(ctx, 25, "port_2to1_matched") == 0.0
        assert _feat(ctx, 50, "port_any") == 1.0
        assert _feat(ctx, 50, "port_3to1") == 0.0
        assert _feat(ctx, 50, "port_2to1_matched") == 0.0
        assert _feat(ctx, 0, "port_any") == 0.0

    def test_second_settlement_turns_on_the_starting_material_term(self) -> None:
        gen = ScenarioGenerator(seed=SEED)
        gen.apply(0, 0)  # seat 0
        gen.apply(18, int(np.flatnonzero(gen.current().compute_legal_road_edges(18))[0]))
        gen.apply(
            20, int(np.flatnonzero(gen.current().compute_legal_road_edges(20))[0])
        )  # seat 1 again
        scenario = gen.current()
        assert scenario is not None and scenario.draft_position == 4
        ctx = SetupContext.build(
            gen._board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
        )
        assert ctx.is_second_settlement
        legal = int(np.flatnonzero(scenario.legal_settlement_corners)[0])
        row = settlement_features(ctx, legal)
        i_hexes = SETTLEMENT_FEATURE_NAMES.index("n_adjacent_hexes")
        i_second = SETTLEMENT_FEATURE_NAMES.index("n_hexes_x_second")
        assert row[i_second] == row[i_hexes]

    def test_illegal_rows_are_zero_and_never_compete(self, ctx: SetupContext) -> None:
        block = all_settlement_features(ctx)
        illegal = np.flatnonzero(~ctx.legal_settlements)
        if illegal.size:
            assert np.all(block[illegal] == 0.0)


class TestRoadArithmetic:
    def test_far_endpoint_and_features(self, ctx: SetupContext) -> None:
        gen = ScenarioGenerator(seed=SEED)
        scenario = gen.current()
        assert scenario is not None
        legal = np.flatnonzero(scenario.compute_legal_road_edges(0))
        assert list(legal) == [0, 1, 2]
        # Vertex 0's three neighbours are 1, 5, 15 -> edges 0, 1, 2 in order.
        assert [road_far_endpoint(ctx, 0, int(e)) for e in legal] == [1, 5, 15]
        # Edges 1 and 2 both open a 12.0-value target; edge 0 only opens 9.4.
        values = [float(road_features(ctx, 0, int(e))[0]) for e in legal]
        assert values == pytest.approx([9.4, 12.0, 12.0])
        # No opponent piece anywhere near vertex 0, and none of 1/5/15 has a port.
        for e in legal:
            assert list(road_features(ctx, 0, int(e))[1:]) == [0.0, 0.0]

    def test_edge_not_incident_to_the_settlement_raises(self, ctx: SetupContext) -> None:
        with pytest.raises(ValueError, match="not incident"):
            road_features(ctx, 0, 40)

    def test_edge_index_map_is_the_canonical_one(self, ctx: SetupContext) -> None:
        pairs = edge_vertex_pairs(ctx.board)
        assert len(pairs) == 72
        # Endpoint ORDER follows the canonical lexicographic pixel key, not the
        # vertex index; what matters is that every edge names the right PAIR.
        assert set(pairs[0]) == {0, 1}
        assert len({frozenset(v) for v in pairs.values()}) == 72


class TestScorerPlumbing:
    def test_pilot_subset_is_a_real_subset(self) -> None:
        assert set(PILOT_FEATURE_NAMES) <= set(SETTLEMENT_FEATURE_NAMES)
        assert len(PILOT_FEATURE_NAMES) == 10

    def test_feature_version_mismatch_refuses_to_load(self) -> None:
        payload = ScorerWeights(
            feature_names=("a",),
            weights=np.zeros(1),
            mean=np.zeros(1),
            scale=np.ones(1),
        ).to_dict()
        payload["feature_version"] = "v0"
        with pytest.raises(ScorerVersionError, match="feature_version"):
            ScorerWeights.from_dict(payload)
        payload["feature_version"] = FEATURE_VERSION
        assert ScorerWeights.from_dict(payload).feature_names == ("a",)

    def test_rank_and_top_k_ignore_illegal_candidates(self) -> None:
        scores = np.asarray([1.0, -np.inf, 3.0, 2.0])
        assert top_k(scores, 3) == [2, 3, 0]
        assert rank_of(scores, 2) == 1
        assert rank_of(scores, 0) == 3
        with pytest.raises(ValueError, match="not a legal candidate"):
            rank_of(scores, 1)

    def test_scorer_module_pulls_in_no_training_stack(self) -> None:
        """D7 vehicle neutrality, enforced structurally.

        Import ``catan_rl.setup_phase.scorer`` in a clean interpreter and assert
        the TRAINING stack never appears in ``sys.modules``: no ``catan_rl.bc``
        (fine-tune), no ``catan_rl.gui``, no checkpoint loader, no trainer, no
        league, no search. Checking the live interpreter would be meaningless —
        pytest has already imported all of it.

        ``torch`` and ``catan_rl.policy.network`` are deliberately NOT on the
        list, and the reason is a pre-existing fact rather than a concession:
        reading the shared schema constants (``RESOURCES_CW``,
        ``DOTS_BY_TOKEN``) goes through ``catan_rl.policy``, whose ``__init__``
        imports ``CatanPolicy`` — exactly the route
        ``setup_phase.analytic_value`` already took before this slice. The claim
        D7 actually needs is that scoring requires no CHECKPOINT and no
        fine-tune code, which is what is asserted here; duplicating the schema
        constants to dodge a package ``__init__`` would trade a real
        single-source-of-truth for a cosmetic import graph.
        """
        import os
        import subprocess
        import sys

        code = (
            "import sys; import catan_rl.setup_phase.scorer as s; "
            "bad=[m for m in sys.modules if m.split('.')[:2] in "
            "([['catan_rl','bc'],['catan_rl','gui'],['catan_rl','checkpoint'],"
            "['catan_rl','algorithms'],['catan_rl','ppo'],['catan_rl','selfplay'],"
            "['catan_rl','search']])]; "
            "print(bad)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
