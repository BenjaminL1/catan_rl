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
from catan_rl.setup_phase.scorer import (
    ScorerVersionError,
    ScorerWeights,
    grade_scores,
    log_prob_of,
    probabilities,
    rank_of,
    top_k,
)
from catan_rl.setup_phase.scorer_features import (
    FEATURE_VERSION,
    N_SETTLEMENT_FEATURES,
    NEW_RESOURCE_BONUS,
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


def _first_edge(scenario, vertex: int) -> int:
    return int(np.flatnonzero(scenario.compute_legal_road_edges(vertex))[0])


def _context(gen: ScenarioGenerator, *, acting_player: int | None = None) -> SetupContext:
    scenario = gen.current()
    assert scenario is not None
    seat = int(scenario.acting_player_idx) if acting_player is None else acting_player
    return SetupContext.build(
        gen._board, scenario.prior_picks, seat, scenario.legal_settlement_corners
    )


@pytest.fixture()
def ctx() -> SetupContext:
    """Draft position 1 on the seed-42 board: no prior picks, all 54 legal."""
    return _context(ScenarioGenerator(seed=SEED))


@pytest.fixture()
def pos4() -> SetupContext:
    """Draft position 4 on the seed-42 board — the NON-DEGENERATE fixture.

    Snake draft 1-2-2-1, so after ``v0`` (seat 0, pos 1), ``v18`` and ``v20``
    (seat 1, pos 2 and 3) the acting seat is 0 again with:

    * ``own_pips``  = v0  = WHEAT 4, ORE 3
    * ``opp_pips``  = v18 + v20 = WOOD 3, BRICK 5, WHEAT 9, SHEEP 5, ORE 0

    Both are non-zero and they are DIFFERENT, which is what a position-1 fixture
    cannot give: there ``own_pips`` and ``opp_pips`` are both all-zero, so
    ``n_new_resources``, ``opponent_new_resources`` and ``n_distinct_resources``
    all collapse onto the same number and pinning them proves nothing.
    """
    gen = ScenarioGenerator(seed=SEED)
    gen.apply(0, 0)
    for vertex in (18, 20):
        scenario = gen.current()
        assert scenario is not None
        gen.apply(vertex, _first_edge(scenario, vertex))
    scenario = gen.current()
    assert scenario is not None and scenario.draft_position == 4
    assert int(scenario.acting_player_idx) == 0
    return _context(gen)


def _feat(ctx: SetupContext, vertex: int, name: str) -> float:
    return float(settlement_features(ctx, vertex)[SETTLEMENT_FEATURE_NAMES.index(name)])


class TestSettlementArithmetic:
    def test_vertex_0_is_desert_ore4_wheat5(self, ctx: SetupContext) -> None:
        # Seed-42 vertex 0 touches DESERT(-), ORE(4 -> 3 dots), WHEAT(5 -> 4 dots).
        row = settlement_features(ctx, 0)
        assert row.shape == (N_SETTLEMENT_FEATURES,) == (16,)
        # Charlesworth column order: WOOD, BRICK, WHEAT, ORE, SHEEP.
        assert list(row[:5]) == [0.0, 0.0, 4.0, 3.0, 0.0]
        # v3 carries NO ``pips_total`` column. The total is 0+0+4+3+0 = 7.0, and
        # that identity — total == the sum of the five columns already here — is
        # exactly why the column was retired: it was a rank deficiency, not a
        # feature. It stays recoverable by anyone who wants it.
        assert float(row[:5].sum()) == 7.0
        assert "pips_total" not in SETTLEMENT_FEATURE_NAMES
        assert _feat(ctx, 0, "n_distinct_resources") == 2.0
        assert _feat(ctx, 0, "n_adjacent_hexes") == 3.0  # the desert still counts as a hex
        # No prior picks: every produced resource is new for both seats.
        assert _feat(ctx, 0, "n_new_resources") == 2.0
        assert _feat(ctx, 0, "opponent_new_resources") == 2.0
        # First settlement of the draft -> the starting-material term is off.
        assert _feat(ctx, 0, "n_hexes_x_second") == 0.0

    def test_opponent_best_margin_is_the_margin_over_what_is_LEFT(self, ctx: SetupContext) -> None:
        """The v2 arithmetic: the reference is candidate-DEPENDENT (blocker 1a).

        Seed-42 legal vertices by pinned Charlesworth value, best first:
        v18 = 12.0, v17 = 10.5, v21 = 10.5, v16 = 10.4, v8 = 10.0.
        v18's neighbours are (16, 17, 41), so settling v18 removes 18, 16, 17
        and 41 from every seat's option set and the best the opponent is LEFT
        with is v21 at 10.5.
        """
        assert ctx.opponent_best_vertex == 18
        assert [round(float(ctx.base_value[v]), 3) for v in ctx.legal_by_base_value[:4]] == [
            12.0,
            10.5,
            10.5,
            10.4,
        ]
        # 12.0 - 10.5. Under v1 this was base(18) - base(18) = 0.0 and denial
        # was inexpressible: taking the board's best spot scored the same as
        # taking a spot nobody wanted.
        assert _feat(ctx, 18, "opponent_best_margin") == pytest.approx(1.5)
        # Vertex 0 (base 4*1.0 wheat + 3*1.1 ore = 7.3) blocks 1, 5 and 15 —
        # none of them the opponent's target — so v18 survives and the margin is
        # 7.3 - 12.0, unchanged from v1.
        assert _feat(ctx, 0, "opponent_best_margin") == pytest.approx(-4.7)
        # A NEIGHBOUR of the target also moves the reference: v16 (10.4) blocks
        # v18, leaving 10.5.
        assert _feat(ctx, 16, "opponent_best_margin") == pytest.approx(-0.1)

    def test_opponent_best_margin_is_not_a_linear_function_of_the_pips_columns(self) -> None:
        """Blocker 1a, verified NUMERICALLY rather than by reading the code.

        ``base_value`` is ``sum(dots * charlesworth_weight)``, i.e. exactly a
        linear combination of the five per-resource pips columns. So the v1
        margin — ``base_value[v]`` minus a board-CONSTANT — was an exact linear
        combination of columns already in the design matrix plus an intercept:
        an unidentifiable column carrying no information the fit did not already
        have. The v2 margin subtracts a per-candidate reference, which is a max
        over an excluded set and therefore NOT in that span.

        The check regresses each column on ``[pips_wood..pips_sheep, 1]`` and
        reads the residual. The RETIRED ``pips_total`` column (rebuilt here as
        the row-sum of the five pips columns) and the reconstructed v1 margin
        are the CONTROLS: both are genuinely in the span, so a residual near
        machine epsilon there is what proves the probe can detect collinearity
        at all — and, for ``pips_total``, it is the byte-exactness the v3
        amendment claims when it drops the column.
        """
        i_margin = SETTLEMENT_FEATURE_NAMES.index("opponent_best_margin")

        def residual_rms(values: np.ndarray, design: np.ndarray) -> float:
            beta, *_ = np.linalg.lstsq(design, values, rcond=None)
            return float(np.sqrt(np.mean((values - design @ beta) ** 2)))

        seen = 0
        for seed in (42, 7):
            gen = ScenarioGenerator(seed=seed)
            while (scenario := gen.current()) is not None:
                ctx = _context(gen)
                legal = np.flatnonzero(ctx.legal_settlements)
                block = all_settlement_features(ctx)[legal]
                design = np.column_stack([block[:, :5], np.ones(legal.size)])

                # The v3-retired ``pips_total``, rebuilt: it IS the row sum, so
                # regressing it on the five parts is exact by construction.
                retired_total = block[:, :5].sum(axis=1)
                collinear = residual_rms(retired_total, design)
                v1_column = ctx.base_value[legal] - float(
                    ctx.base_value[ctx.opponent_best_vertex or 0]
                )
                v1_residual = residual_rms(v1_column, design)
                v2_residual = residual_rms(block[:, i_margin], design)

                if scenario.draft_position >= 2:
                    assert collinear < 1e-9, f"seed {seed} pos {scenario.draft_position}"
                    assert v1_residual < 1e-9, (
                        f"seed {seed} pos {scenario.draft_position}: the REJECTED v1 column "
                        f"should be exactly collinear, got {v1_residual}"
                    )
                    assert v2_residual > 0.02, (
                        f"seed {seed} pos {scenario.draft_position}: opponent_best_margin "
                        f"residual {v2_residual:.4f} is not materially nonzero — the column "
                        f"has collapsed back into the pips span"
                    )
                    seen += 1

                # Deterministic advance: the middle-ish legal corner, so the
                # draft visits genuinely different positions rather than always
                # the lowest-index one.
                pick = int(legal[legal.size // 3])
                gen.apply(pick, _first_edge(scenario, pick))
        assert seen == 6  # two seeds x draft positions 2, 3, 4

    def test_the_margin_absorbs_the_retired_adjacency_block_flag(self, ctx: SetupContext) -> None:
        """v3's merge, checked as an IDENTITY rather than asserted.

        ``adjacency_block`` was ``distance(v, opponent_best) == 1``. Blocking
        that vertex — by taking it (distance 0) or by neighbouring it (distance
        1) — is exactly what makes ``_opponent_best_remaining`` fall back to a
        worse reference. The flag was therefore a strict SUB-CASE of an event
        the margin already carries, and carries with magnitude.

        Seed 42, position 1: the target is v18 at 12.0, its neighbours are
        (16, 17, 41). Reading each candidate's implied reference back out of the
        margin as ``base_value[v] - margin``:

        * v0  (distance 3): 7.3 - (-4.7)  = 12.0 — the target survives;
        * v21 (distance 2): 10.5 - (-1.5) = 12.0 — likewise;
        * v18 (distance 0): 12.0 - (+1.5) = 10.5 — the target is TAKEN;
        * v16 (distance 1): 10.4 - (-0.1) = 10.5 — the flag's own case;
        * v41 (distance 1): 8.0  - (-2.5) = 10.5 — the same event, and the
          margin separates it from v16 by 2.4 where the flag scored both 1.0.
        """
        i_margin = SETTLEMENT_FEATURE_NAMES.index("opponent_best_margin")
        assert "adjacency_block" not in SETTLEMENT_FEATURE_NAMES
        assert ctx.opponent_best_vertex == 18
        assert ctx.adjacency[18] == (16, 17, 41)
        board_best = float(ctx.base_value[18])
        assert board_best == pytest.approx(12.0)
        assert _feat(ctx, 41, "opponent_best_margin") == pytest.approx(-2.5)

        legal = np.flatnonzero(ctx.legal_settlements)
        block = all_settlement_features(ctx)[legal]
        for row, vertex in zip(block, legal, strict=True):
            reference = float(ctx.base_value[vertex]) - float(row[i_margin])
            retired_flag = int(ctx.distances[vertex, 18]) == 1
            moved = reference < board_best - 1e-9
            # The flag fired only at distance 1; the margin moves at distance
            # <= 1. Every flagged vertex is a moved one, so nothing the flag
            # said is lost.
            assert moved == (int(ctx.distances[vertex, 18]) <= 1)
            assert not retired_flag or moved

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

    def test_illegal_rows_are_zero_and_never_compete(self, pos4: SetupContext) -> None:
        """Uses the POSITION-4 fixture on purpose.

        At draft position 1 all 54 vertices are legal, so ``illegal`` is empty
        and the assertion under it never runs — the test passed by describing an
        empty set. Position 4 has three settlements down, so the distance rule
        has genuinely forbidden vertices to check.
        """
        block = all_settlement_features(pos4)
        illegal = np.flatnonzero(~pos4.legal_settlements)
        assert illegal.size > 0, "fixture invariant: position 4 must have illegal vertices"
        assert np.all(block[illegal] == 0.0)
        assert np.any(block[np.flatnonzero(pos4.legal_settlements)] != 0.0)


class TestExpansionValue:
    """Blocker 1b: D1's "own-yield-scored" expansion target."""

    def test_expansion_target_is_scored_with_the_acting_seat_need_profile(
        self, pos4: SetupContext
    ) -> None:
        """The hand-computed fixture, and the one that shows the fix BITES.

        At position 4 the acting seat is 0, holding v0 = WHEAT 4 + ORE 3. From
        candidate v11 the legal vertices exactly 2 roads away include

        * v6  = WOOD 4, BRICK 2, SHEEP 3 -> base 4*1.0 + 2*1.0 + 3*0.7 = 8.1,
          and all three resources are NEW for seat 0 -> own 8.1 + 3*1.0 = 11.1;
        * v14 = WHEAT 5, ORE 4 -> base 5*1.0 + 4*1.1 = 9.4, and NEITHER resource
          is new -> own 9.4 + 0 = 9.4.

        Seat-neutral scoring picks v14 (9.4 > 8.1) — more raw pips, nothing the
        seat is missing. Own-yield scoring picks v6 at 11.1: that is the owner's
        "building to a missing resource" in one number, and it is a DIFFERENT
        target, not merely a different scale.
        """
        assert NEW_RESOURCE_BONUS == 1.0
        assert _feat(pos4, 11, "expansion_value") == pytest.approx(11.1)
        # The seat-neutral answer the rejected implementation would have given.
        distance_2 = [
            v
            for v in range(54)
            if pos4.legal_settlements[v]
            and int(pos4.distances[11, v]) == 2
            and v not in {11, *pos4.adjacency[11]}
        ]
        assert max(float(pos4.base_value[v]) for v in distance_2) == pytest.approx(9.4)

    def test_expansion_value_differs_between_the_two_seats(self, pos4: SetupContext) -> None:
        """The property the seat-neutral version could not have.

        Same board, same prior picks, same legal mask — only the acting seat
        changes. A feature that is identical here cannot express "this seat
        needs sheep and that one does not".
        """
        gen = ScenarioGenerator(seed=SEED)
        gen.apply(0, 0)
        for vertex in (18, 20):
            scenario = gen.current()
            assert scenario is not None
            gen.apply(vertex, _first_edge(scenario, vertex))
        other = _context(gen, acting_player=1)
        assert other.acting_player != pos4.acting_player

        i_exp = SETTLEMENT_FEATURE_NAMES.index("expansion_value")
        legal = np.flatnonzero(pos4.legal_settlements)
        mine = all_settlement_features(pos4)[legal][:, i_exp]
        theirs = all_settlement_features(other)[legal][:, i_exp]
        assert not np.allclose(mine, theirs)
        # Seat 0 holds WHEAT+ORE and seat 1 holds WOOD/BRICK/WHEAT/SHEEP, so
        # v2's expansion reads 12.1 for seat 0 and 9.4 for seat 1.
        assert _feat(pos4, 2, "expansion_value") == pytest.approx(12.1)
        assert _feat(other, 2, "expansion_value") == pytest.approx(9.4)

    def test_own_value_is_the_pinned_base_plus_the_new_resource_bonus(
        self, pos4: SetupContext
    ) -> None:
        # v6 = WOOD 4, BRICK 2, SHEEP 3; seat 0 produces none of the three.
        assert float(pos4.base_value[6]) == pytest.approx(8.1)
        assert float(pos4.own_value[6]) == pytest.approx(8.1 + 3 * NEW_RESOURCE_BONUS)
        # v14 = WHEAT 5, ORE 4; seat 0 already produces both.
        assert float(pos4.base_value[14]) == pytest.approx(9.4)
        assert float(pos4.own_value[14]) == pytest.approx(9.4)


class TestNonDegenerateSettlementFixture:
    """Pins the columns a position-1 fixture cannot separate (blocker 1c).

    At draft position 1 ``own_pips`` and ``opp_pips`` are both all-zero, so
    ``n_new_resources``, ``opponent_new_resources`` and ``n_distinct_resources``
    are equal BY CONSTRUCTION and ``n_hexes_x_second`` is always 0. Every one of
    them was previously pinned only in that collapsed state.
    """

    def test_the_fixture_really_is_asymmetric(self, pos4: SetupContext) -> None:
        assert pos4.own_vertices == (0,)
        assert pos4.opp_vertices == (18, 20)
        assert list(pos4.own_pips) == [0.0, 0.0, 4.0, 3.0, 0.0]
        assert list(pos4.opp_pips) == [3.0, 5.0, 9.0, 0.0, 5.0]

    def test_vertex_11_pins_five_columns_that_do_not_collapse(self, pos4: SetupContext) -> None:
        # v11 = WOOD 4, ORE 1, SHEEP 4 -> base 4*1.0 + 1*1.1 + 4*0.7 = 7.9.
        assert list(settlement_features(pos4, 11)[:5]) == [4.0, 0.0, 0.0, 1.0, 4.0]
        assert _feat(pos4, 11, "n_distinct_resources") == 3.0
        # NEW for seat 0 (holds WHEAT+ORE): WOOD and SHEEP -> 2, not 3.
        assert _feat(pos4, 11, "n_new_resources") == 2.0
        # NEW for seat 1 (holds WOOD/BRICK/WHEAT/SHEEP): ORE alone -> 1.
        assert _feat(pos4, 11, "opponent_new_resources") == 1.0
        # Second settlement for seat 0 -> the starting-material term switches on.
        assert _feat(pos4, 11, "n_adjacent_hexes") == 3.0
        assert _feat(pos4, 11, "n_hexes_x_second") == 3.0
        # ORE is the board's unique scarcest resource and seat 1 has none of it,
        # so covering ORE starves them. At position 1 this fired for every
        # ore-producing vertex regardless of the opponent, because there was no
        # opponent yet.
        assert list(pos4.scarce_mask) == [False, False, False, True, False]
        assert _feat(pos4, 11, "scarcity_starve") == 1.0
        # v6 = WOOD 4, BRICK 2, SHEEP 3 produces no ORE, so it starves nobody.
        assert _feat(pos4, 6, "scarcity_starve") == 0.0
        # base 7.9 minus the best REMAINING after v11 blocks 10, 12 and 30 —
        # v8 at 10.0 survives.
        assert _feat(pos4, 11, "opponent_best_margin") == pytest.approx(-2.1)

    def test_taking_the_opponents_target_pays_at_position_4_too(self, pos4: SetupContext) -> None:
        # v8 = WOOD 5 + BRICK 5 = 10.0 is the best legal vertex here; its
        # neighbours are (7, 9, 27), so taking it leaves v13 at 9.7.
        assert pos4.opponent_best_vertex == 8
        assert _feat(pos4, 8, "opponent_best_margin") == pytest.approx(0.3)
        # A NEIGHBOUR of the target — what ``adjacency_block`` used to flag —
        # moves the same reference: v7 (base 9.1) blocks v8, leaving v13 at 9.7,
        # so 9.1 - 9.7 = -0.6. A vertex FAR from the target keeps the 10.0
        # reference: v11 (base 7.9) reads 7.9 - 10.0 = -2.1. One column now
        # separates the blocker from the non-blocker AND says by how much.
        assert pos4.adjacency[8][0] == 7
        assert _feat(pos4, 7, "opponent_best_margin") == pytest.approx(-0.6)
        assert _feat(pos4, 11, "opponent_best_margin") == pytest.approx(-2.1)


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
        """Nine columns since v3, not the pilot's ten.

        ``pips_total`` went with the rest of the design matrix. Keeping it here
        would make this tuple stop being a subset of the columns the feature
        function emits — a "reconstruction" of a fit that cannot be run.
        """
        assert set(PILOT_FEATURE_NAMES) <= set(SETTLEMENT_FEATURE_NAMES)
        assert len(PILOT_FEATURE_NAMES) == 9
        assert "pips_total" not in PILOT_FEATURE_NAMES

    def test_feature_version_is_v3(self) -> None:
        """The bump is part of the contract, not an implementation detail.

        ``v1`` weights were fitted against a design matrix whose
        ``opponent_best_margin`` column was exactly collinear with the pips
        columns and whose ``expansion_value`` column was seat-neutral. ``v2``
        weights were fitted against 18 columns, two of which v3 removed
        (``pips_total``, ``adjacency_block``) — a v2 vector scored against the
        16-column v3 matrix would not even align, let alone mean anything. So
        the artifact must refuse to load rather than warn.
        """
        assert FEATURE_VERSION == "v3"
        assert len(SETTLEMENT_FEATURE_NAMES) == N_SETTLEMENT_FEATURES == 16

    def test_feature_version_mismatch_refuses_to_load(self) -> None:
        payload = ScorerWeights(
            feature_names=("a",),
            weights=np.zeros(1),
            mean=np.zeros(1),
            scale=np.ones(1),
        ).to_dict()
        for stale in ("v0", "v1", "v2"):
            payload["feature_version"] = stale
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

    def test_legality_comes_from_the_mask_when_one_is_given(self) -> None:
        """``-1e9`` is FINITE.

        A policy head that masks with a large negative rather than ``-inf``
        would have every slot pass an ``isfinite`` legality test, so an illegal
        vertex could win the argmax and be reported as the grader's top-1. The
        boolean mask is the statement of legality; the score value is not.
        """
        scores = np.asarray([1.0, 9.0, 3.0, 2.0])
        mask = np.asarray([True, False, True, True])  # index 1 is masked out
        assert top_k(scores, 3) == [1, 2, 3]  # isfinite fallback: the fake wins
        assert top_k(scores, 3, legal=mask) == [2, 3, 0]
        assert rank_of(scores, 2, legal=mask) == 1
        assert probabilities(scores, legal=mask)[1] == 0.0
        assert log_prob_of(scores, 2, legal=mask) == pytest.approx(
            float(np.log(np.exp(3.0) / (np.exp(1.0) + np.exp(3.0) + np.exp(2.0))))
        )
        with pytest.raises(ValueError, match="not a legal candidate"):
            log_prob_of(scores, 1, legal=mask)
        grade = grade_scores(scores, 2, legal=mask)
        assert grade.top1 == 2 and grade.agree is True and grade.rank == 1

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
