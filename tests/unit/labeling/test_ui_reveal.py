"""D3 blind-then-reveal invariants (acceptance criterion 4).

Every test here is an invariant the spec spells out, not a behaviour check:
no reveal before submit, no reveal on skip, no undo after reveal, ``--no-reveal``
rows carry none of the five scorer fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import SCORER_ROW_FIELDS, load_scenarios
from catan_rl.labeling.ui import (
    PHASE_REVEAL,
    PHASE_ROAD_PICK,
    PHASE_SETTLEMENT_PICK,
    LabelingUIState,
)
from catan_rl.setup_phase.fit import fit_scorer


def _legal_pair(scenario, *, nth: int = 0) -> tuple[int, int]:
    corners = np.flatnonzero(scenario.legal_settlement_corners)
    settlement = int(corners[min(nth, len(corners) - 1)])
    edges = np.flatnonzero(scenario.compute_legal_road_edges(settlement))
    return settlement, int(edges[0])


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory):
    """A cheaply-fitted scorer — the reveal only needs a well-formed artifact."""
    seed_dir = tmp_path_factory.mktemp("scorer_fixture_labels")
    path = seed_dir / "scenarios.jsonl"
    session = LabelingSession(data_dir=seed_dir, labeler_id="fixture", session_seed=5)
    session.start()
    for _ in range(4):
        s = session.current_scenario()
        assert s is not None
        pick = _legal_pair(s)
        session.submit(settlement_vertex=pick[0], road_edge=pick[1])
    session.quit()
    return fit_scorer(load_scenarios(path), version="vtest", seed=0, iters=30).scorer


def _state(tmp_path: Path, scorer, *, reveal_mode: str = "reveal") -> LabelingUIState:
    session = LabelingSession(
        data_dir=tmp_path,
        labeler_id="ben",
        session_seed=9,
        reveal_mode=reveal_mode,
        scorer_version=scorer.version,
    )
    session.start()
    return LabelingUIState(session, scorer=None if reveal_mode == "no_reveal" else scorer)


def _pick_both(state: LabelingUIState) -> tuple[int, int]:
    scenario = state.session.current_scenario()
    assert scenario is not None
    settlement, road = _legal_pair(scenario)
    assert state.select_settlement(settlement)
    assert state.select_road(road)
    return settlement, road


class TestBlindFirst:
    def test_no_reveal_during_either_pick_phase(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.reveal is None
        scenario = state.session.current_scenario()
        assert scenario is not None
        settlement, road = _legal_pair(scenario)
        state.select_settlement(settlement)
        assert state.phase == PHASE_ROAD_PICK
        assert state.reveal is None
        state.select_road(road)
        assert state.reveal is None

    def test_reveal_appears_only_after_the_durable_write(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)
        assert state.reveal is None
        state.submit()
        assert state.phase == PHASE_REVEAL
        assert state.reveal is not None
        # The row really is on disk before the overlay exists.
        assert len(load_scenarios(tmp_path / "scenarios.jsonl")) == 1

    def test_a_failed_write_leaves_no_reveal(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)

        def boom(**kwargs):
            raise OSError("disk full")

        state.session.submit = boom  # type: ignore[method-assign]
        with pytest.raises(OSError):
            state.submit()
        assert state.reveal is None
        assert state.phase == PHASE_ROAD_PICK

    def test_skip_never_reveals(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)
        state.skip()
        assert state.reveal is None
        assert state.phase == PHASE_SETTLEMENT_PICK

    def test_undo_is_inert_after_reveal(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)
        state.submit()
        before = dict(state.reveal or {})
        state.undo()
        assert state.phase == PHASE_REVEAL
        assert state.reveal == before

    def test_the_overlay_is_pinned_to_the_board_it_graded(self, tmp_path: Path, scorer) -> None:
        """``session.submit`` advances the snake draft, so the session's CURRENT
        scenario is already the next decision point once the overlay is up.
        Rendering the reveal against it would paint the scorer's rings on the
        board the owner is about to label — an anchoring leak D3's control is
        designed to DETECT, not to absorb.
        """
        state = _state(tmp_path, scorer)
        _pick_both(state)
        state.submit()
        graded = state.reveal_scenario
        assert graded is not None
        row = load_scenarios(tmp_path / "scenarios.jsonl")[0]
        # It is the very position the row records. (``current()`` mints a fresh
        # ``scenario_id`` per call, so the identity that matters is the decision
        # point — seed, draft position, prior picks — not the uuid.)
        assert graded.game_seed == row["game_seed"]
        assert graded.draft_position == row["draft_position"] == 1
        assert [p.to_dict() for p in graded.prior_picks] == row["prior_picks"]
        # ...and NOT the one the session has already advanced to.
        nxt = state.session.current_scenario()
        assert nxt is not None
        assert nxt.draft_position == 2
        # The scorer's top-3 is legal on the GRADED board, which is the only
        # board the rings can honestly be drawn on.
        for vertex in state.reveal["settlement_top3"]:
            assert bool(graded.legal_settlement_corners[vertex])

    def test_the_last_pick_of_a_board_still_has_its_reveal_board(
        self, tmp_path: Path, scorer
    ) -> None:
        state = _state(tmp_path, scorer)
        for position in (1, 2, 3, 4):
            _pick_both(state)
            state.submit()
            assert state.phase == PHASE_REVEAL
            assert state.reveal_scenario is not None
            assert state.reveal_scenario.draft_position == position
            state.dismiss_reveal()
        rows = load_scenarios(tmp_path / "scenarios.jsonl")
        assert [r["draft_position"] for r in rows] == [1, 2, 3, 4]

    def test_the_reveal_shows_probabilities_not_bare_picks(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        settlement, road = _pick_both(state)
        state.submit()
        reveal = state.reveal
        assert reveal is not None
        probs = reveal["settlement_top3_probs"]
        assert len(probs) == len(reveal["settlement_top3"])
        assert probs == sorted(probs, reverse=True)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert sum(probs) <= 1.0 + 1e-9
        assert 0.0 <= reveal["owner_settlement_prob"] <= 1.0
        assert 0.0 <= reveal["owner_road_prob"] <= 1.0
        assert reveal["owner_settlement"] == settlement
        assert reveal["owner_road"] == road

    def test_dismiss_returns_to_the_next_pick(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)
        state.submit()
        state.dismiss_reveal()
        assert state.phase == PHASE_SETTLEMENT_PICK
        assert state.reveal is None
        assert state.session.current_scenario() is not None


class TestRowFields:
    def test_reveal_rows_carry_all_five_fields(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        settlement, _road = _pick_both(state)
        state.submit()
        raw = json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[0])
        for field in SCORER_ROW_FIELDS:
            assert field in raw, f"reveal row missing {field}"
        assert raw["reveal_mode"] == "reveal"
        assert raw["scorer_version"] == scorer.version
        assert raw["agree"] == (raw["scorer_top1"] == settlement)
        assert raw["scorer_rank_of_pick"] >= 1
        # The scorer graded the pick; it never wrote one.
        assert raw["settlement_vertex"] == settlement

    def test_no_reveal_rows_carry_none_of_the_five(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer, reveal_mode="no_reveal")
        _pick_both(state)
        state.submit()
        assert state.reveal is None
        assert state.phase == PHASE_SETTLEMENT_PICK
        raw = json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[0])
        for field in SCORER_ROW_FIELDS:
            assert field not in raw, f"no-reveal row must not carry {field}"
        # ...but the MANIFEST still names the arm, so the gate can find it.
        manifest = json.loads(
            (tmp_path / "sessions" / state.session.session_id / "manifest.json").read_text()
        )
        assert manifest["reveal_mode"] == "no_reveal"

    def test_no_scorer_means_no_scorer_fields(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=9)
        session.start()
        state = LabelingUIState(session, scorer=None)
        _pick_both(state)
        state.submit()
        assert state.reveal is None
        raw = json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[0])
        for field in SCORER_ROW_FIELDS:
            assert field not in raw

    def test_the_clarity_tag_is_written_in_both_arms(self, tmp_path: Path, scorer) -> None:
        """``pick_clarity`` is the OWNER's tag, not a scorer field, so the
        anchoring-control arm carries it too — D4 reads the ``clear``
        strictness bar on those picks."""
        for mode in ("reveal", "no_reveal"):
            data_dir = tmp_path / mode
            state = _state(data_dir, scorer, reveal_mode=mode)
            _pick_both(state)
            state.submit(pick_clarity="clear")
            raw = json.loads((data_dir / "scenarios.jsonl").read_text().splitlines()[0])
            assert raw["pick_clarity"] == "clear"
            assert "pick_clarity" not in SCORER_ROW_FIELDS

    def test_an_untagged_submit_reads_as_close(self, tmp_path: Path, scorer) -> None:
        state = _state(tmp_path, scorer)
        _pick_both(state)
        state.submit()
        raw = json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[0])
        assert raw["pick_clarity"] == "close"

    def test_an_unknown_clarity_tag_is_refused(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=9)
        session.start()
        scenario = session.current_scenario()
        assert scenario is not None
        settlement, road = _legal_pair(scenario)
        with pytest.raises(ValueError, match="pick_clarity"):
            session.submit(settlement_vertex=settlement, road_edge=road, pick_clarity="maybe")

    def test_session_rejects_unknown_scorer_fields(self, tmp_path: Path) -> None:
        session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=9)
        session.start()
        scenario = session.current_scenario()
        assert scenario is not None
        settlement, road = _legal_pair(scenario)
        with pytest.raises(ValueError, match="unknown scorer_fields"):
            session.submit(
                settlement_vertex=settlement,
                road_edge=road,
                scorer_fields={"scorer_top1": 1, "not_a_field": 2},
            )


class TestSubmitKeyBinding:
    """The reflexive key must carry the CONSERVATIVE tag.

    ``S`` has meant plain "submit" for the whole 292-label corpus, and only
    ``clear`` picks are held to D4's >=70% top-1 bar (``gate._clarity_report``),
    which the spec calls "revisable only upward". Binding muscle memory to the
    strict tag would let habit contaminate that bar, so ``S`` submits ``close``
    — matching ``store._V3_DEFAULTS`` and every default in the stack — and the
    deliberate, unfamiliar ``B`` asserts "clear best".
    """

    @staticmethod
    def _dispatch(char: str) -> list[str]:
        """Run ``_handle_keydown`` with everything but the binding stubbed out."""
        from catan_rl.labeling.ui import PHASE_SETTLEMENT_PICK, LabelingUI

        tagged: list[str] = []
        ui = object.__new__(LabelingUI)
        ui._pygame = type("P", (), {"K_q": -1})()
        ui.state = type("S", (), {"phase": PHASE_SETTLEMENT_PICK})()
        ui._try_submit = tagged.append  # type: ignore[method-assign]
        assert ui._handle_keydown(type("E", (), {"key": 0, "unicode": char})()) is True
        return tagged

    def test_s_submits_close_and_b_submits_clear(self) -> None:
        from catan_rl.labeling.store import PICK_CLARITY_CLEAR, PICK_CLARITY_CLOSE

        assert self._dispatch("s") == [PICK_CLARITY_CLOSE]
        assert self._dispatch("b") == [PICK_CLARITY_CLEAR]
        # ``C`` is no longer a submit key at all: leaving it bound to "close"
        # while ``S`` also meant "close" would make the retagging invisible to
        # anyone reading the key map.
        assert self._dispatch("c") == []


class TestSpaceOnlyDismissal:
    """The reveal overlay is dismissed by SPACE alone (owner request 2026-08-22):
    an arbitrary keystroke — or a screenshot chord — must never advance past a
    reveal the owner wants to capture. Clicks are equally inert in the reveal
    phase (pinned at the handler level; ``dismiss_reveal`` itself stays a plain
    state transition)."""

    @staticmethod
    def _reveal_ui() -> tuple[object, list[str]]:
        from catan_rl.labeling.ui import PHASE_REVEAL, LabelingUI

        dismissed: list[str] = []
        ui = object.__new__(LabelingUI)
        ui._pygame = type("P", (), {"K_q": -1, "K_SPACE": 32})()
        state = type("S", (), {"phase": PHASE_REVEAL})()
        state.dismiss_reveal = lambda: dismissed.append("dismissed")
        ui.state = state
        ui._now_ms = lambda: 0
        return ui, dismissed

    def test_random_keys_do_not_dismiss(self) -> None:
        ui, dismissed = self._reveal_ui()
        for key, char in ((0, "x"), (0, "u"), (0, "s"), (0, "b"), (13, "\r")):
            assert ui._handle_keydown(type("E", (), {"key": key, "unicode": char})()) is True
        assert dismissed == []

    def test_space_dismisses(self) -> None:
        ui, dismissed = self._reveal_ui()
        assert ui._handle_keydown(type("E", (), {"key": 32, "unicode": " "})()) is True
        assert dismissed == ["dismissed"]

    def test_click_does_not_dismiss(self) -> None:
        from catan_rl.labeling.ui import PHASE_REVEAL, LabelingUI

        dismissed: list[str] = []
        ui = object.__new__(LabelingUI)
        state = type("S", (), {"phase": PHASE_REVEAL})()
        state.dismiss_reveal = lambda: dismissed.append("dismissed")
        ui.state = state
        ui._handle_click((10, 10))
        assert dismissed == []
