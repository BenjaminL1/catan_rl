"""D0 self-consistency replay + the v3 store fields.

Spec ``setup-scorer-and-blind-reveal`` D0 + acceptance criteria 4/5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.consistency import (
    ESTIMATOR_FREE_REPLAY,
    ESTIMATOR_LINKED,
    ESTIMATOR_SAME_POSITION,
    ConsistencyError,
    consistency_report,
    free_replay_pairs,
    legacy_pairs,
    pair_replay_rows,
)
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import (
    SCHEMA_VERSION,
    SCORER_ROW_FIELDS,
    append_scenario,
    load_scenarios,
)


def _legal_pair(scenario, *, nth: int = 0) -> tuple[int, int]:
    corners = np.flatnonzero(scenario.legal_settlement_corners)
    settlement = int(corners[min(nth, len(corners) - 1)])
    edges = np.flatnonzero(scenario.compute_legal_road_edges(settlement))
    return settlement, int(edges[0])


def _label_one_board(session: LabelingSession, *, nth: int = 0) -> list[tuple[int, int]]:
    picks = []
    for _ in range(4):
        scenario = session.current_scenario()
        assert scenario is not None
        pick = _legal_pair(scenario, nth=nth)
        picks.append(pick)
        session.submit(settlement_vertex=pick[0], road_edge=pick[1])
    return picks


class TestStoreV3:
    def test_schema_version_is_three(self) -> None:
        assert SCHEMA_VERSION == 3

    def test_v1_and_v2_rows_load_unchanged_with_none_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.jsonl"
        v1 = {
            "schema_version": 1,
            "scenario_id": "a",
            "session_id": "s",
            "labeled_at": "2026-01-01T00:00:00Z",
            "labeler_id": "ben",
            "game_seed": 1,
            "draft_position": 1,
            "acting_player": 0,
            "prior_picks": [],
            "settlement_vertex": 3,
            "road_edge": 4,
        }
        raw = json.dumps(v1, separators=(",", ":"), ensure_ascii=False)
        append_scenario(dict(v1), path)
        # The FILE is never rewritten: byte-identical to what was written, and
        # in particular it gains NONE of the v3 keys.
        assert path.read_text().strip() == raw

        loaded = load_scenarios(path)[0]
        assert loaded["schema_version"] == 1
        assert loaded["settlement_vertex"] == 3
        assert loaded["replay_of"] is None
        for field in SCORER_ROW_FIELDS:
            assert loaded[field] is None, f"{field} must default to None, not a value"

    def test_agree_defaults_to_none_not_false(self, tmp_path: Path) -> None:
        # A defaulted ``agree=False`` would inject a phantom disagreement for
        # every pre-v3 label into the D4 gate.
        path = tmp_path / "scenarios.jsonl"
        append_scenario(
            {
                "schema_version": 2,
                "scenario_id": "a",
                "session_id": "s",
                "labeled_at": "2026-01-01T00:00:00Z",
                "labeler_id": "ben",
                "game_seed": 1,
                "draft_position": 1,
                "acting_player": 0,
                "prior_picks": [],
                "settlement_vertex": 3,
                "road_edge": 4,
            },
            path,
        )
        assert load_scenarios(path)[0]["agree"] is None


class TestReplaySession:
    def test_replay_represents_identical_boards(self, tmp_path: Path) -> None:
        first = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=7)
        first.start()
        _label_one_board(first)
        _label_one_board(first)
        first.quit()
        original_seeds = [int(r["game_seed"]) for r in load_scenarios(tmp_path / "scenarios.jsonl")]

        replay = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            replay_of_session=first.session_id,
        )
        replay.start()
        seen = []
        while (scenario := replay.current_scenario()) is not None:
            seen.append(int(scenario.game_seed))
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
        assert seen == original_seeds
        replay.quit()
        # Provenance: the replay manifest records the seed the boards it
        # PRESENTED came from, not only the fresh sequence it generated and
        # never used. Without this the claimed link is unreadable from the
        # manifest alone.
        manifest = json.loads(
            (tmp_path / "sessions" / replay.session_id / "manifest.json").read_text()
        )
        assert manifest["replay_of_session"] == first.session_id
        assert manifest["replay_of_master_seed"] == 7

    def test_replay_rows_link_to_the_right_originals(self, tmp_path: Path) -> None:
        first = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=7)
        first.start()
        _label_one_board(first)
        first.quit()
        before = (tmp_path / "scenarios.jsonl").read_text()

        replay = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=first.session_id
        )
        replay.start()
        while (scenario := replay.current_scenario()) is not None:
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
        replay.quit()

        rows = load_scenarios(tmp_path / "scenarios.jsonl")
        originals = [r for r in rows if r["session_id"] == first.session_id]
        replays = [r for r in rows if r["session_id"] == replay.session_id]
        assert len(replays) == len(originals) == 4
        by_id = {r["scenario_id"]: r for r in originals}
        for row in replays:
            linked = by_id[row["replay_of"]]
            assert linked["game_seed"] == row["game_seed"]
            assert linked["draft_position"] == row["draft_position"]
        # The originals are untouched — the file is append-only.
        assert (tmp_path / "scenarios.jsonl").read_text().startswith(before)

    def test_replay_advances_with_the_original_pick_not_the_new_one(self, tmp_path: Path) -> None:
        """FORCED-ORIGINAL: every replayed position must be the SAME position."""
        first = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=11)
        first.start()
        _label_one_board(first)
        first.quit()
        originals = load_scenarios(tmp_path / "scenarios.jsonl")

        replay = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=first.session_id
        )
        replay.start()
        while (scenario := replay.current_scenario()) is not None:
            expected = next(
                r
                for r in originals
                if r["draft_position"] == scenario.draft_position
                and r["game_seed"] == scenario.game_seed
            )
            # The position the owner is shown is the ORIGINAL position.
            assert [p.to_dict() for p in scenario.prior_picks] == expected["prior_picks"]
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])

        new_rows = [
            r
            for r in load_scenarios(tmp_path / "scenarios.jsonl")
            if r["session_id"] == replay.session_id
        ]
        # The owner's DIFFERENT picks were recorded...
        assert any(
            r["settlement_vertex"] != o["settlement_vertex"]
            for r, o in zip(new_rows, originals, strict=True)
        )
        # ...but the prior-pick context stayed identical to the original run.
        for r, o in zip(new_rows, originals, strict=True):
            assert r["prior_picks"] == o["prior_picks"]

    def test_replay_without_manifest_raises(self, tmp_path: Path) -> None:
        session = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session="no-such-session"
        )
        with pytest.raises(FileNotFoundError):
            session.start()

    def test_replay_session_with_no_rows_raises(self, tmp_path: Path) -> None:
        empty = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=3)
        empty.start()
        empty.quit()
        session = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=empty.session_id
        )
        with pytest.raises(ValueError, match="wrote no rows"):
            session.start()

    def test_manifest_records_the_mode(self, tmp_path: Path) -> None:
        session = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=1,
            reveal_mode="no_reveal",
            scorer_version="v1",
        )
        session.start()
        manifest = json.loads(
            (tmp_path / "sessions" / session.session_id / "manifest.json").read_text()
        )
        assert manifest["reveal_mode"] == "no_reveal"
        assert manifest["scorer_version"] == "v1"
        assert manifest["replay_of_master_seed"] is None

    def test_bad_reveal_mode_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reveal_mode"):
            LabelingSession(data_dir=tmp_path, labeler_id="ben", reveal_mode="sometimes")


def _row(
    sid: str,
    *,
    pos: int,
    vertex: int,
    edge: int,
    replay_of: str | None = None,
    seed: int = 1,
    at: str = "2026-01-01T00:00:00Z",
    prior=None,
) -> dict:
    row = {
        "schema_version": 3,
        "scenario_id": sid,
        "session_id": "s" if replay_of is None else "s2",
        "labeled_at": at,
        "labeler_id": "ben",
        "game_seed": seed,
        "draft_position": pos,
        "acting_player": 0,
        "prior_picks": prior or [],
        "settlement_vertex": vertex,
        "road_edge": edge,
        "replay_of": replay_of,
    }
    return row


class TestConsistencyArithmetic:
    def test_per_position_agreement(self) -> None:
        rows = [
            _row("o1", pos=1, vertex=10, edge=1),
            _row("o2", pos=2, vertex=20, edge=2),
            _row("r1", pos=1, vertex=10, edge=1, replay_of="o1"),  # both agree
            _row("r2", pos=2, vertex=99, edge=2, replay_of="o2"),  # settlement differs
        ]
        report = consistency_report(rows)
        # ``auto`` resolves to the UNBIASED linked estimator whenever the corpus
        # carries ``replay_of`` rows, and the report says which one it used.
        assert report["estimator"] == ESTIMATOR_LINKED
        assert report["n_pairs"] == 2
        assert report["settlement"]["by_position"]["1"]["n_agree"] == 1
        assert report["settlement"]["by_position"]["2"]["n_agree"] == 0
        assert report["settlement"]["overall"]["rate"] == 0.5
        # The road number is CONDITIONAL on the settlement agreeing: pair r2
        # picked a different settlement, so its road was never in the same
        # legal set and comparing the two edge indices would manufacture an
        # agreement out of two different choice sets.
        road = report["road_given_same_settlement"]
        assert road["conditioned_on"] == "settlement_agrees"
        assert road["n_conditioning_pairs"] == 1
        assert road["overall"]["n"] == 1
        assert road["overall"]["rate"] == 1.0
        assert "road" not in report
        assert report["settlement"]["picks_2_4"]["n"] == 1

    def test_dangling_replay_of_raises(self) -> None:
        with pytest.raises(ConsistencyError, match="not in the corpus"):
            pair_replay_rows([_row("r1", pos=1, vertex=1, edge=1, replay_of="ghost")])

    def test_legacy_duplicates_pair_only_when_the_position_matches(self) -> None:
        same_prior = [{"player": 0, "settlement_vertex": 5, "road_edge": 6}]
        other_prior = [{"player": 0, "settlement_vertex": 7, "road_edge": 8}]
        rows = [
            _row("a", pos=2, vertex=10, edge=1, prior=same_prior, at="2026-01-01T00:00:00Z"),
            _row("b", pos=2, vertex=10, edge=1, prior=same_prior, at="2026-01-02T00:00:00Z"),
            _row("c", pos=3, vertex=11, edge=1, prior=same_prior, at="2026-01-01T00:00:00Z"),
            _row("d", pos=3, vertex=12, edge=1, prior=other_prior, at="2026-01-02T00:00:00Z"),
        ]
        pairs, n_filtered_out = legacy_pairs(rows)
        assert [(p.original["scenario_id"], p.replay["scenario_id"]) for p in pairs] == [("a", "b")]
        assert n_filtered_out == 1
        # The free-replay estimator keeps BOTH: it joins on (game_seed,
        # draft_position) alone, which is what the banked D0 RESULT was computed
        # with. The two estimators must stay distinguishable — this is the 40pp
        # gap the report exists to keep visible.
        free = free_replay_pairs(rows)
        assert [(p.original["scenario_id"], p.replay["scenario_id"]) for p in free] == [
            ("a", "b"),
            ("c", "d"),
        ]

    def test_the_report_names_its_estimator_and_publishes_all_three(self) -> None:
        """The D0 headline must never travel without the estimator that made it.

        The shipped corpus has ZERO ``replay_of`` rows, so ``auto`` must fall
        back to ``free_replay`` — the estimator that reproduces the banked
        7/20 = 35% — and NOT to the upward-biased ``same_position`` rate.
        """
        same_prior = [{"player": 0, "settlement_vertex": 5, "road_edge": 6}]
        other_prior = [{"player": 0, "settlement_vertex": 7, "road_edge": 8}]
        rows = [
            _row("a", pos=2, vertex=10, edge=1, prior=same_prior, at="2026-01-01T00:00:00Z"),
            _row("b", pos=2, vertex=10, edge=1, prior=same_prior, at="2026-01-02T00:00:00Z"),
            _row("c", pos=3, vertex=11, edge=1, prior=same_prior, at="2026-01-01T00:00:00Z"),
            _row("d", pos=3, vertex=12, edge=1, prior=other_prior, at="2026-01-02T00:00:00Z"),
        ]
        report = consistency_report(rows)
        assert report["estimator"] == ESTIMATOR_FREE_REPLAY
        assert report["n_pairs"] == 2
        # free_replay sees 1/2; same_position sees 1/1 — the upward bias, made
        # visible side by side rather than blended into one headline.
        est = report["estimators"]
        assert est[ESTIMATOR_FREE_REPLAY]["settlement_overall"]["n"] == 2
        assert est[ESTIMATOR_FREE_REPLAY]["settlement_overall"]["n_agree"] == 1
        assert est[ESTIMATOR_SAME_POSITION]["settlement_overall"]["n"] == 1
        assert est[ESTIMATOR_SAME_POSITION]["settlement_overall"]["n_agree"] == 1
        assert est[ESTIMATOR_LINKED]["n_pairs"] == 0
        assert "biased UPWARD" in est[ESTIMATOR_SAME_POSITION]["bias"]
        assert "7/20 = 35%" in report["estimator_bias"]

    def test_an_unknown_estimator_is_refused(self) -> None:
        with pytest.raises(ConsistencyError, match="estimator must be one of"):
            consistency_report([], estimator="whatever")
