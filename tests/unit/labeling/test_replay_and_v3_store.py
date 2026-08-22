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
from catan_rl.labeling.ui import PHASE_REVEAL, PHASE_SETTLEMENT_PICK, LabelingUI, LabelingUIState
from catan_rl.setup_phase.fit import fit_scorer, training_rows
from catan_rl.setup_phase.gate import fresh_exam_picks, session_metadata


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
        assert manifest["replay_boards"] == 0

    def test_bad_reveal_mode_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reveal_mode"):
            LabelingSession(data_dir=tmp_path, labeler_id="ben", reveal_mode="sometimes")


def _set_end_time(data_dir: Path, session_id: str, end_time: str) -> None:
    """Pin a manifest's ``end_time``.

    The auto-choice rule is "the most recent COMPLETED session", and
    ``_utcnow_iso`` has one-second resolution — every session built inside one
    test shares a timestamp, so the recency a test means to assert has to be
    written down rather than raced for.
    """
    path = data_dir / "sessions" / session_id / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["end_time"] = end_time
    path.write_text(json.dumps(manifest, indent=2))


def _completed_session(
    data_dir: Path, *, seed: int, boards: int = 2, end_time: str | None = None
) -> LabelingSession:
    """A finished source session holding ``boards`` fully-labeled boards."""
    session = LabelingSession(data_dir=data_dir, labeler_id="ben", session_seed=seed)
    session.start()
    for _ in range(boards):
        _label_one_board(session)
    session.quit()
    if end_time is not None:
        _set_end_time(data_dir, session.session_id, end_time)
    return session


def _distinct_seeds(rows: list[dict], session_id: str) -> list[int]:
    """The board seeds a session presented, in presentation order."""
    out: list[int] = []
    for row in rows:
        if str(row["session_id"]) != session_id:
            continue
        seed = int(row["game_seed"])
        if seed not in out:
            out.append(seed)
    return out


def _label_boards(session: LabelingSession, *, boards: int, nth: int = 1) -> list[int]:
    """Label ``boards`` whole boards; return the seed presented for each pick."""
    seeds: list[int] = []
    for _ in range(boards * 4):
        scenario = session.current_scenario()
        assert scenario is not None
        seeds.append(int(scenario.game_seed))
        pick = _legal_pair(scenario, nth=nth)
        session.submit(settlement_vertex=pick[0], road_edge=pick[1])
    return seeds


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory):
    """A cheaply-fitted scorer — the reveal only needs a well-formed artifact."""
    seed_dir = tmp_path_factory.mktemp("fold_scorer_fixture_labels")
    session = LabelingSession(data_dir=seed_dir, labeler_id="fixture", session_seed=5)
    session.start()
    _label_one_board(session)
    session.quit()
    labels = load_scenarios(seed_dir / "scenarios.jsonl")
    return fit_scorer(labels, version="vtest", seed=0, iters=30).scorer


class TestFoldedReplayBoards:
    """D0's ceiling estimate is EXTENDED by folding ~5 replay boards into
    otherwise-normal sessions.

    A folded board has to be the SAME measurement the exclusive mode makes —
    forced-original, linked by ``replay_of``, graded by nobody — while the rest
    of the sitting stays an ordinary fresh-board session.
    """

    def test_folded_boards_come_first_then_the_fresh_sequence(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=2)
        source_seeds = _distinct_seeds(
            load_scenarios(tmp_path / "scenarios.jsonl"), source.session_id
        )
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        seen = _label_boards(mixed, boards=2)
        # The fold is FRONT-loaded: the replayed boards are presented before the
        # session's first reveal can exist, so nothing seen this sitting can
        # anchor the one owner-vs-owner measurement.
        assert seen[:4] == [source_seeds[0]] * 4
        # ...and the fresh half is the UNSHIFTED master-seed sequence: folding
        # must not burn board seeds the normal session would have used.
        assert seen[4:] == [500] * 4

    def test_only_the_folded_rows_carry_replay_of(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        _label_boards(mixed, boards=2)
        mixed.quit()

        rows = load_scenarios(tmp_path / "scenarios.jsonl")
        originals = {r["scenario_id"]: r for r in rows if r["session_id"] == source.session_id}
        mine = [r for r in rows if r["session_id"] == mixed.session_id]
        folded, fresh = mine[:4], mine[4:]
        assert len(folded) == len(fresh) == 4
        for row in folded:
            linked = originals[row["replay_of"]]
            assert linked["game_seed"] == row["game_seed"]
            assert linked["draft_position"] == row["draft_position"]
        assert all(row["replay_of"] is None for row in fresh)

    def test_folded_boards_are_forced_original(self, tmp_path: Path) -> None:
        """Same guarantee the exclusive mode gives: every one of the four
        positions is the identical position it was the first time."""
        source = _completed_session(tmp_path, seed=11, boards=1)
        originals = [
            r
            for r in load_scenarios(tmp_path / "scenarios.jsonl")
            if r["session_id"] == source.session_id
        ]
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        _label_boards(mixed, boards=1)
        mixed.quit()

        folded = [
            r
            for r in load_scenarios(tmp_path / "scenarios.jsonl")
            if r["session_id"] == mixed.session_id
        ]
        # The owner's DIFFERENT picks were recorded...
        assert any(
            new["settlement_vertex"] != old["settlement_vertex"]
            for new, old in zip(folded, originals, strict=True)
        )
        # ...but the draft was advanced with the ORIGINAL ones, so the prior-pick
        # context never diverged.
        for new, old in zip(folded, originals, strict=True):
            assert new["prior_picks"] == old["prior_picks"]

    def test_folding_more_boards_than_the_source_has_refuses(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=2)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=3,
        )
        with pytest.raises(ValueError, match="only 2 board"):
            mixed.start()

    def test_folding_with_no_past_session_refuses(self, tmp_path: Path) -> None:
        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=1
        )
        with pytest.raises(ValueError, match="no past session"):
            mixed.start()

    def test_a_negative_fold_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="replay_boards"):
            LabelingSession(data_dir=tmp_path, labeler_id="ben", replay_boards=-1)

    def test_the_source_is_auto_chosen_as_the_most_recent_completed(self, tmp_path: Path) -> None:
        _completed_session(tmp_path, seed=7, boards=1, end_time="2026-01-01T00:00:00Z")
        newer = _completed_session(tmp_path, seed=300, boards=1, end_time="2026-02-01T00:00:00Z")
        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=1
        )
        mixed.start()
        assert mixed.replay_of_session == newer.session_id
        # ``newer``'s only board came from its master seed.
        assert _label_boards(mixed, boards=1) == [300] * 4

    def test_auto_choice_skips_a_session_with_too_few_boards(self, tmp_path: Path) -> None:
        big = _completed_session(tmp_path, seed=7, boards=2, end_time="2026-01-01T00:00:00Z")
        _completed_session(tmp_path, seed=300, boards=1, end_time="2026-02-01T00:00:00Z")
        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=2
        )
        mixed.start()
        assert mixed.replay_of_session == big.session_id

    def test_auto_choice_skips_an_unfinished_session(self, tmp_path: Path) -> None:
        """A session with no ``end_time`` may still be appending rows; replaying
        it would be replaying a moving target."""
        done = _completed_session(tmp_path, seed=7, boards=1, end_time="2026-01-01T00:00:00Z")
        live = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=300)
        live.start()
        _label_one_board(live)  # rows on disk, but the manifest has no end_time

        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=1
        )
        mixed.start()
        assert mixed.replay_of_session == done.session_id

    def test_auto_choice_skips_a_session_of_nothing_but_replayed_rows(self, tmp_path: Path) -> None:
        """Folding a replay session would chain ``replay_of`` onto a replay row,
        pairing the owner against their own re-label instead of the original."""
        source = _completed_session(tmp_path, seed=7, boards=1, end_time="2026-01-01T00:00:00Z")
        replay = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=source.session_id
        )
        replay.start()
        while (scenario := replay.current_scenario()) is not None:
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
        replay.quit()
        _set_end_time(tmp_path, replay.session_id, "2026-03-01T00:00:00Z")

        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=1
        )
        mixed.start()
        assert mixed.replay_of_session == source.session_id

    def test_the_manifest_records_the_fold(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=2)
        mixed = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", session_seed=500, replay_boards=1
        )
        mixed.start()
        mixed.quit()
        manifest = json.loads(
            (tmp_path / "sessions" / mixed.session_id / "manifest.json").read_text()
        )
        assert manifest["replay_boards"] == 1
        assert manifest["replay_of_session"] == source.session_id
        assert manifest["replay_of_master_seed"] == 7
        assert manifest["master_seed"] == 500

    def test_a_half_labeled_source_board_falls_through_to_fresh_boards(
        self, tmp_path: Path
    ) -> None:
        """The source quit mid-draft, so its last board has no positions 3-4.

        The replay skips forward over them (there is no original pick to
        advance the draft with), and in a FOLD "skip forward" has one more
        place to land than it does in the exclusive mode: the fresh sequence.
        """
        source = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=7)
        source.start()
        _label_one_board(source)  # board A, all four positions
        for _ in range(2):  # board B, then the owner quit
            scenario = source.current_scenario()
            assert scenario is not None
            pick = _legal_pair(scenario)
            source.submit(settlement_vertex=pick[0], road_edge=pick[1])
        source.quit()

        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=2,
        )
        mixed.start()
        presented: list[tuple[int, int, bool]] = []
        for _ in range(8):
            scenario = mixed.current_scenario()
            assert scenario is not None
            presented.append((scenario.game_seed, scenario.draft_position, mixed.current_is_replay))
            pick = _legal_pair(scenario, nth=1)
            mixed.submit(settlement_vertex=pick[0], road_edge=pick[1])
        assert presented == [
            (7, 1, True),
            (7, 2, True),
            (7, 3, True),
            (7, 4, True),
            (8, 1, True),
            (8, 2, True),
            (500, 1, False),
            (500, 2, False),
        ]
        assert mixed.exhausted is False

    def test_a_folded_session_never_exhausts(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        _label_boards(mixed, boards=2)
        assert mixed.exhausted is False
        assert mixed.current_scenario() is not None

    def test_current_is_replay_tracks_the_board(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        for _ in range(4):
            assert mixed.current_is_replay is True
            scenario = mixed.current_scenario()
            assert scenario is not None
            pick = _legal_pair(scenario, nth=1)
            mixed.submit(settlement_vertex=pick[0], road_edge=pick[1])
        assert mixed.current_is_replay is False

    def test_an_exclusive_replay_reports_exhausted_not_quit(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=2)
        replay = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=source.session_id
        )
        replay.start()
        assert replay.exhausted is False
        while (scenario := replay.current_scenario()) is not None:
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
        assert replay.exhausted is True
        assert replay.replay_boards_presented == 2

    def test_a_folded_row_refuses_scorer_fields(self, tmp_path: Path) -> None:
        """Fail CLOSED at the writer, not only at the UI that suppresses them."""
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
        )
        mixed.start()
        scenario = mixed.current_scenario()
        assert scenario is not None
        settlement, road = _legal_pair(scenario, nth=1)
        with pytest.raises(ValueError, match="replayed row"):
            mixed.submit(
                settlement_vertex=settlement,
                road_edge=road,
                scorer_fields={"scorer_top1": settlement},
            )

    def test_folded_rows_reach_neither_the_fit_nor_the_exam(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
            scorer_version="vtest",
        )
        mixed.start()
        _label_boards(mixed, boards=2)
        mixed.quit()

        rows = load_scenarios(tmp_path / "scenarios.jsonl")
        folded = {r["scenario_id"] for r in rows if r["replay_of"] is not None}
        assert len(folded) == 4
        # D2: the fit excludes them by ``replay_of``, so a folded board can
        # never be fitted on twice.
        assert folded & {r["scenario_id"] for r in training_rows(rows)} == set()
        # D4: and ``fresh_exam_picks`` drops them too, so the fold costs the
        # exam nothing and contributes nothing.
        exam = {r["scenario_id"] for r in fresh_exam_picks(rows, session_metadata(tmp_path))}
        assert exam & folded == set()
        assert len(exam) == 4  # the session's FRESH picks still count

    def test_a_reveal_session_paints_no_overlay_on_a_folded_board(
        self, tmp_path: Path, scorer
    ) -> None:
        """A scorer overlay mid-replay anchors the owner on the scorer during
        the one owner-vs-owner measurement — the same reason the EXCLUSIVE mode
        refuses ``--scorer-weights`` outright. Folding cannot smuggle it back
        in, so the reveal is suppressed board-by-board instead."""
        source = _completed_session(tmp_path, seed=7, boards=1)
        mixed = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=500,
            replay_of_session=source.session_id,
            replay_boards=1,
            scorer_version=scorer.version,
        )
        mixed.start()
        state = LabelingUIState(mixed, scorer=scorer)

        def _submit_one() -> dict:
            scenario = state.session.current_scenario()
            assert scenario is not None
            settlement, road = _legal_pair(scenario, nth=1)
            assert state.select_settlement(settlement)
            assert state.select_road(road)
            state.submit()
            return json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[-1])

        for _ in range(4):  # the folded board
            raw = _submit_one()
            assert state.reveal is None
            assert state.phase == PHASE_SETTLEMENT_PICK
            assert raw["replay_of"] is not None
            for field in SCORER_ROW_FIELDS:
                assert field not in raw, f"a replayed row must not carry {field}"

        raw = _submit_one()  # the first FRESH board of the same session
        assert state.phase == PHASE_REVEAL
        assert state.reveal is not None
        assert "replay_of" not in raw
        assert raw["scorer_version"] == scorer.version


class TestDoneMessage:
    """``current_scenario() is None`` has two very different meanings."""

    @staticmethod
    def _message(session: LabelingSession) -> str:
        """``_done_message`` with the pygame half of the UI left unbuilt."""
        ui = object.__new__(LabelingUI)
        ui.session = session
        return ui._done_message()

    def test_a_quit_session_still_says_quit(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=1)
        session.start()
        session.quit()
        assert "Session quit" in self._message(session)

    def test_an_exhausted_replay_says_the_replay_completed(self, tmp_path: Path) -> None:
        source = _completed_session(tmp_path, seed=7, boards=2)
        replay = LabelingSession(
            data_dir=tmp_path, labeler_id="ben", replay_of_session=source.session_id
        )
        replay.start()
        while (scenario := replay.current_scenario()) is not None:
            pick = _legal_pair(scenario, nth=1)
            replay.submit(settlement_vertex=pick[0], road_edge=pick[1])
        message = self._message(replay)
        assert "Replay complete" in message
        assert "2 boards" in message


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
