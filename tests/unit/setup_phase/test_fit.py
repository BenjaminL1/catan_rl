"""D2 fit: replay exclusion, duplicate refusal, determinism, null road baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.fit import (
    ROAD_NULL_FEATURE,
    FitError,
    build_examples,
    fit_scorer,
    training_rows,
)
from catan_rl.setup_phase.scorer import load_weights, save_weights
from catan_rl.setup_phase.scorer_features import PILOT_FEATURE_NAMES


def _legal_pair(scenario, *, nth: int = 0) -> tuple[int, int]:
    corners = np.flatnonzero(scenario.legal_settlement_corners)
    settlement = int(corners[min(nth, len(corners) - 1)])
    edges = np.flatnonzero(scenario.compute_legal_road_edges(settlement))
    return settlement, int(edges[0])


@pytest.fixture()
def corpus(tmp_path: Path) -> list[dict]:
    """Two labeled boards plus a full replay of the first."""
    first = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=5)
    first.start()
    for _ in range(8):  # two boards
        scenario = first.current_scenario()
        assert scenario is not None
        pick = _legal_pair(scenario)
        first.submit(settlement_vertex=pick[0], road_edge=pick[1])
    first.quit()

    replay = LabelingSession(
        data_dir=tmp_path, labeler_id="ben", replay_of_session=first.session_id
    )
    replay.start()
    while (scenario := replay.current_scenario()) is not None:
        pick = _legal_pair(scenario, nth=1)
        replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
    replay.quit()
    return load_scenarios(tmp_path / "scenarios.jsonl")


class TestReplayExclusion:
    def test_replay_rows_excluded_by_identity_not_by_count(self, corpus: list[dict]) -> None:
        kept = training_rows(corpus)
        replay_ids = {r["scenario_id"] for r in corpus if r["replay_of"] is not None}
        assert replay_ids  # the fixture really did replay
        assert {r["scenario_id"] for r in kept}.isdisjoint(replay_ids)
        assert {r["scenario_id"] for r in kept} == {
            r["scenario_id"] for r in corpus if r["replay_of"] is None
        }

    def test_fit_never_sees_a_replay_row(self, corpus: list[dict]) -> None:
        examples = build_examples(training_rows(corpus))
        replay_ids = {r["scenario_id"] for r in corpus if r["replay_of"] is not None}
        assert {ex.scenario_id for ex in examples}.isdisjoint(replay_ids)


class TestDuplicateGuard:
    def _dupe(self, corpus: list[dict]) -> list[dict]:
        legacy = dict(corpus[0])
        legacy["scenario_id"] = "legacy-dupe"
        legacy["replay_of"] = None
        legacy["labeled_at"] = "2099-01-01T00:00:00Z"
        return [*corpus, legacy]

    def test_unannotated_duplicate_is_refused(self, corpus: list[dict]) -> None:
        with pytest.raises(FitError, match="two NON-replay rows"):
            training_rows(self._dupe(corpus))

    def test_first_labeled_policy_keeps_the_earlier_row(self, corpus: list[dict]) -> None:
        kept = training_rows(self._dupe(corpus), duplicate_policy="first-labeled")
        ids = {r["scenario_id"] for r in kept}
        assert "legacy-dupe" not in ids
        assert corpus[0]["scenario_id"] in ids

    def test_unknown_policy_rejected(self, corpus: list[dict]) -> None:
        with pytest.raises(FitError, match="duplicate_policy"):
            training_rows(corpus, duplicate_policy="whatever")


class TestFit:
    def test_deterministic_given_rows_and_seed(self, corpus: list[dict]) -> None:
        a = fit_scorer(corpus, version="t", seed=3, iters=50)
        b = fit_scorer(corpus, version="t", seed=3, iters=50)
        assert np.allclose(a.scorer.settlement.weights, b.scorer.settlement.weights)
        assert np.allclose(a.scorer.road.weights, b.scorer.road.weights)
        c = fit_scorer(corpus, version="t", seed=4, iters=50)
        assert not np.allclose(a.scorer.settlement.weights, c.scorer.settlement.weights)

    def test_reports_the_road_null_baseline_rather_than_asserting_the_rule(
        self, corpus: list[dict]
    ) -> None:
        result = fit_scorer(corpus, version="t", seed=0, iters=200)
        null = result.metrics["road_null_baseline"]
        assert null["feature"] == ROAD_NULL_FEATURE
        assert "agreement" in null and "beaten_by_fit" in null
        # The null is REPORTED, never enforced: a fit that loses to it is a
        # finding, so nothing here may assert beaten_by_fit is True.
        assert isinstance(null["beaten_by_fit"], bool)

    def test_the_null_comparison_is_out_of_sample_and_grouped_by_board(
        self, corpus: list[dict]
    ) -> None:
        """A 3-feature model nests the 1-feature null, so IN-SAMPLE it can
        hardly lose — that comparison is arithmetic, not evidence.

        The fold numbers are recorded in provenance so the reported verdict can
        be re-derived rather than trusted, and folds are grouped by
        ``game_seed`` because the four picks of one board share a layout.
        """
        result = fit_scorer(corpus, version="t", seed=0, iters=100)
        null = result.metrics["road_null_baseline"]
        assert null["evaluation"] == "out_of_sample_kfold_grouped_by_game_seed"
        assert null["k"] == len({r["game_seed"] for r in corpus if r["replay_of"] is None})
        assert len(null["folds"]) == null["k"]
        held_out_seeds = [s for f in null["folds"] for s in f["game_seeds_held_out"]]
        assert len(held_out_seeds) == len(set(held_out_seeds))  # no board in two folds
        assert sum(f["n_held_out"] for f in null["folds"]) == result.metrics["n_labels"]
        for fold in null["folds"]:
            assert fold["n_train"] + fold["n_held_out"] == result.metrics["n_labels"]
        # ``beaten_by_fit`` keys on the NLL both heads optimise; the top-1 read
        # is reported next to it, not instead of it.
        assert null["beaten_by_fit"] is (null["nll"] < null["null_nll"])
        assert null["beaten_by_fit_top1"] is (null["agreement"] > null["null_agreement"])
        # The in-sample numbers survive as a separate, clearly-labelled block.
        assert set(null["in_sample"]) == {
            "road_nll",
            "null_nll",
            "road_agreement",
            "null_agreement",
        }

    def test_a_single_board_corpus_says_the_comparison_is_in_sample(
        self, corpus: list[dict]
    ) -> None:
        one_board = [r for r in corpus if r["game_seed"] == corpus[0]["game_seed"]]
        result = fit_scorer(one_board, version="t", seed=0, iters=50)
        null = result.metrics["road_null_baseline"]
        assert null["evaluation"] == "in_sample_only"
        assert null["k"] == 0 and null["folds"] == []
        assert "smoke test" in null["why"]

    def test_metrics_break_out_by_draft_position(self, corpus: list[dict]) -> None:
        result = fit_scorer(corpus, version="t", seed=0, iters=50)
        by_pos = result.metrics["settlement"]["by_position"]
        assert set(by_pos) == {"1", "2", "3", "4"}
        assert sum(v["n"] for v in by_pos.values()) == result.metrics["n_labels"]

    def test_pilot_subset_fits_ten_columns(self, corpus: list[dict]) -> None:
        result = fit_scorer(
            corpus,
            version="t",
            seed=0,
            iters=50,
            settlement_feature_subset=list(PILOT_FEATURE_NAMES),
        )
        assert result.scorer.settlement.feature_names == PILOT_FEATURE_NAMES
        assert result.scorer.settlement.weights.shape == (10,)

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(FitError, match="no non-replay"):
            fit_scorer([], version="t")

    def test_artifact_round_trips(self, corpus: list[dict], tmp_path: Path) -> None:
        result = fit_scorer(corpus, version="v1", seed=0, iters=50)
        path = tmp_path / "weights.json"
        save_weights(result.scorer, path)
        loaded = load_weights(path)
        assert loaded.version == "v1"
        assert np.allclose(loaded.settlement.weights, result.scorer.settlement.weights)
        assert loaded.provenance["fit_metrics"]["n_labels"] == result.metrics["n_labels"]

    def test_scored_vertices_mask_the_illegal_ones(self, corpus: list[dict]) -> None:
        """Advanced past draft position 1 on purpose.

        At position 1 all 54 vertices are legal, so ``~legal`` selects nothing
        and the ``isneginf`` assertion below describes an empty array — the test
        passed without ever exercising the mask.
        """
        from catan_rl.labeling.scenario_gen import ScenarioGenerator

        result = fit_scorer(corpus, version="v1", seed=0, iters=50)
        gen = ScenarioGenerator(seed=5)
        first = gen.current()
        assert first is not None
        pick = _legal_pair(first)
        gen.apply(pick[0], pick[1])
        scenario = gen.current()
        assert scenario is not None and scenario.draft_position == 2

        illegal = np.flatnonzero(~scenario.legal_settlement_corners)
        assert illegal.size > 0, "fixture invariant: position 2 must forbid some vertices"
        scores = result.scorer.score_vertices(
            gen._board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
        )
        assert np.all(np.isfinite(scores[scenario.legal_settlement_corners]))
        assert np.all(np.isneginf(scores[illegal]))
