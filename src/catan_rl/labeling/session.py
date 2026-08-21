"""Session manager for the labeling tool (plan §B).

Wraps a :class:`ScenarioGenerator` with persistence, manifest tracking,
and the snake-draft → fresh-board loop. Per plan §B (user-paced
sessions): sessions run indefinitely; the user controls when to quit.

Each board produces 4 scenarios (one per snake-draft position). After
the 4th submit on a board, the session transparently generates a fresh
random board (new game_seed) and starts again at draft position 1.

Persistence:
- ``data_dir/scenarios.jsonl`` — durable append-only labels.
- ``data_dir/sessions/<uuid>/manifest.json`` — per-session metadata.
- ``data_dir/sessions/<uuid>/inflight_state.json`` — *future*: per-
  scenario checkpoint for mid-scenario crash recovery. (Phase 1 does
  not implement mid-scenario recovery — submits are atomic; quitting
  between submits is the resume granularity.)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catan_rl.env.ruleset import RULESET_R0
from catan_rl.labeling.scenario_gen import Scenario, ScenarioGenerator
from catan_rl.labeling.store import (
    PICK_CLARITIES,
    PICK_CLARITY_CLOSE,
    PICK_CLARITY_FIELD,
    REVEAL_MODE_REVEAL,
    REVEAL_MODES,
    SCHEMA_VERSION,
    SCORER_ROW_FIELDS,
    append_scenario,
    count_scenarios,
    load_scenarios,
    repair_jsonl,
)

_SCENARIOS_FILE = "scenarios.jsonl"
_SESSIONS_DIR = "sessions"
_MANIFEST_FILE = "manifest.json"


class LabelingSession:
    """A single labeling session.

    Attributes:
        session_id: UUID generated on construction.
        data_dir: Root data directory (e.g. ``data/labels/setup/v1/``).
        labeler_id: Identity recorded per row.
        scenarios_completed: Count of rows appended this session.

    Usage:

        session = LabelingSession(data_dir=Path("data/labels/setup/v1"), labeler_id="ben")
        session.start()
        while (scenario := session.current_scenario()) is not None:
            ...  # render scenario; collect user pick
            session.submit(settlement_vertex=..., road_edge=...)
        session.quit()
    """

    def __init__(
        self,
        data_dir: Path,
        labeler_id: str,
        session_seed: int | None = None,
        *,
        replay_of_session: str | None = None,
        reveal_mode: str = REVEAL_MODE_REVEAL,
        scorer_version: str | None = None,
    ) -> None:
        if reveal_mode not in REVEAL_MODES:
            raise ValueError(f"reveal_mode must be one of {REVEAL_MODES}, got {reveal_mode!r}")
        self.session_id = str(uuid.uuid4())
        self.data_dir = Path(data_dir)
        self.labeler_id = labeler_id
        # Master seed for this session — drives the board-seed sequence.
        # If None, derive a non-deterministic seed from the wall clock so
        # consecutive sessions don't replay identical boards. Deterministic
        # seeds are useful for testing.
        if session_seed is None:
            self._master_seed = int.from_bytes(os.urandom(4), "little")
        else:
            self._master_seed = int(session_seed)

        self.scenarios_completed = 0
        self._start_wall_time = 0.0
        self._gen: ScenarioGenerator | None = None
        self._next_board_seed_offset = 0
        self._started = False
        self._quit = False

        # --- D0 self-consistency replay + D3 blind-then-reveal ---------------
        self.replay_of_session = replay_of_session
        self.reveal_mode = reveal_mode
        self.scorer_version = scorer_version
        #: (game_seed, draft_position) -> the ORIGINAL row being re-presented.
        self._replay_rows: dict[tuple[int, int], dict[str, Any]] = {}
        #: Board seeds still to re-present, in the original session's order.
        self._replay_seed_queue: list[int] = []
        self._replay_master_seed: int | None = None
        #: Set once the replay plan is exhausted; ``current_scenario`` then
        #: returns ``None`` exactly as a quit session does.
        self._exhausted = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def scenarios_path(self) -> Path:
        return self.data_dir / _SCENARIOS_FILE

    @property
    def session_dir(self) -> Path:
        return self.data_dir / _SESSIONS_DIR / self.session_id

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / _MANIFEST_FILE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin the session: create dirs, write manifest, run JSONL repair."""
        if self._started:
            raise RuntimeError("session already started")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Crash recovery: truncate any malformed trailing line from a
        # prior session's crash mid-write. No-op if the file is clean
        # (plan §D).
        repair_jsonl(self.scenarios_path)
        self._start_wall_time = time.monotonic()
        if self.replay_of_session is not None:
            self._load_replay_plan()
        self._gen = self._new_generator()
        self._write_manifest(end_time=None)
        self._started = True

    def quit(self) -> None:
        """Finalise manifest with end_time and stop."""
        if not self._started:
            raise RuntimeError("session was never started")
        if self._quit:
            return
        self._write_manifest(end_time=_utcnow_iso())
        self._quit = True

    # ------------------------------------------------------------------
    # Scenario access
    # ------------------------------------------------------------------

    def current_scenario(self) -> Scenario | None:
        """The current scenario, or ``None`` if the session is quit/exhausted.

        In REPLAY mode the loop skips forward over any board position the
        original session never labeled (it skipped that draft), because there
        would be no original pick to compare against — and, more importantly,
        no original pick to advance the draft with.
        """
        if self._quit or self._exhausted:
            return None
        if self._gen is None:
            raise RuntimeError("session not started")
        while True:
            gen = self._gen
            if gen is None:
                return None
            scenario = gen.current()
            if scenario is None:
                self._gen = self._new_generator()
                continue
            if self.replay_of_session is None:
                return scenario
            if (scenario.game_seed, scenario.draft_position) in self._replay_rows:
                return scenario
            self._gen = self._new_generator()

    def submit(
        self,
        settlement_vertex: int,
        road_edge: int,
        notes: str = "",
        decision_time_ms: int = 0,
        scorer_fields: dict[str, Any] | None = None,
        pick_clarity: str = PICK_CLARITY_CLOSE,
    ) -> None:
        """Record the current scenario and advance.

        Writes a JSONL row, applies the pick to the engine to advance
        the snake-draft state. If the 4th pick of a board is submitted,
        the next call to ``current_scenario()`` returns pick 1 of a
        fresh random board.

        ``pick_clarity`` is the owner's own tag for this decision (D3's two
        submit keys). It is written in BOTH arms — it is not a scorer field —
        because D4 reads the ``clear`` strictness bar on no-reveal picks too.
        It defaults to :data:`~catan_rl.labeling.store.PICK_CLARITY_CLOSE`, the
        same conservative reading the store gives an untagged legacy row.

        ``scorer_fields`` carries the D3 reveal payload (the five
        :data:`~catan_rl.labeling.store.SCORER_ROW_FIELDS`). It is written ONLY
        in ``reveal`` sessions: a ``--no-reveal`` row must carry none of the
        five, so the anchoring-control arm is identifiable from the row itself
        and not merely from a manifest join. The caller computes the payload
        before calling — the reveal is only DISPLAYED after this method
        returns, which is what makes "no reveal before a durable submit"
        structural rather than a convention.

        **Replay semantics are FORCED-ORIGINAL** (D0). In a replay session the
        row records the owner's NEW pick, but the draft is advanced with the
        ORIGINAL session's pick. Every one of the four decision points is then
        the identical position it was the first time, which is the only way
        positions 2-4 yield a well-defined self-agreement number — a free
        replay diverges after pick 1 and silently makes picks 2-4
        non-comparable.
        """
        if self._quit:
            raise RuntimeError("cannot submit after quit")
        if self._gen is None:
            raise RuntimeError("session not started")
        scenario = self._gen.current()
        if scenario is None:
            raise RuntimeError("no current scenario to submit")
        if len(notes) > 200:
            raise ValueError("notes field length cap is 200 chars")
        if pick_clarity not in PICK_CLARITIES:
            raise ValueError(f"pick_clarity must be one of {PICK_CLARITIES}, got {pick_clarity!r}")
        replay_of: str | None = None
        if self.replay_of_session is not None:
            key = (scenario.game_seed, scenario.draft_position)
            original = self._replay_rows.get(key)
            if original is None:  # pragma: no cover - current_scenario filters these
                raise RuntimeError(f"replay session has no original row for {key}")
            replay_of = str(original["scenario_id"])
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "session_id": self.session_id,
            "labeled_at": _utcnow_iso(),
            "labeler_id": self.labeler_id,
            "game_seed": scenario.game_seed,
            "draft_position": scenario.draft_position,
            "acting_player": scenario.acting_player_idx,
            "prior_picks": [p.to_dict() for p in scenario.prior_picks],
            "settlement_vertex": int(settlement_vertex),
            "road_edge": int(road_edge),
            "decision_time_ms": int(decision_time_ms),
            "notes": notes,
            "quality_flag": "fast" if decision_time_ms and decision_time_ms < 15000 else "",
            # Schema v2 (D3). Stamped explicitly rather than left to the
            # reader's default, so a row written today names its own
            # provenance: this is the TOOL path, and ``ScenarioGenerator``
            # builds every mask with ``RULESET_R0``.
            "source": "tool",
            "ruleset": RULESET_R0,
            PICK_CLARITY_FIELD: pick_clarity,
        }
        if replay_of is not None:
            row["replay_of"] = replay_of
        if self.reveal_mode == REVEAL_MODE_REVEAL and scorer_fields is not None:
            unknown = sorted(set(scorer_fields) - set(SCORER_ROW_FIELDS))
            if unknown:
                raise ValueError(f"unknown scorer_fields keys: {unknown}")
            row.update(scorer_fields)
        # Apply to engine BEFORE persisting — if the pick is illegal,
        # the row never lands.
        if self.replay_of_session is None:
            self._gen.apply(int(settlement_vertex), int(road_edge))
        else:
            _validate_pick(scenario, int(settlement_vertex), int(road_edge))
            original = self._replay_rows[(scenario.game_seed, scenario.draft_position)]
            self._gen.apply(int(original["settlement_vertex"]), int(original["road_edge"]))
        append_scenario(row, self.scenarios_path)
        self.scenarios_completed += 1
        # Refresh manifest on every submit so a crash doesn't lose the count.
        self._write_manifest(end_time=None)
        # If we just submitted pick 4, advance to a fresh board.
        if self._gen.current() is None:
            self._gen = self._new_generator()

    def skip(self) -> None:
        """Abandon the current draft and jump to a fresh board.

        Skipped scenarios are not written to JSONL. The whole draft is
        discarded — partially-labeled drafts are not preserved because
        the snake-draft picks 2-4 are conditional on prior picks; if
        the user skipped pick 1 the remaining picks would have no
        meaningful prior context.
        """
        if self._quit:
            raise RuntimeError("cannot skip after quit")
        if self._gen is None:
            raise RuntimeError("session not started")
        # Move to next board.
        self._gen = self._new_generator()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def total_scenarios_in_dataset(self) -> int:
        """Total rows across all sessions (not just this one)."""
        return count_scenarios(self.scenarios_path)

    def elapsed_seconds(self) -> float:
        if not self._started:
            return 0.0
        return time.monotonic() - self._start_wall_time

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_generator(self) -> ScenarioGenerator | None:
        """Build a fresh ScenarioGenerator with the next board seed.

        Returns ``None`` (and marks the session exhausted) when a REPLAY
        session has re-presented every board of the original session.
        """
        if self.replay_of_session is not None:
            if not self._replay_seed_queue:
                self._exhausted = True
                return None
            return ScenarioGenerator(seed=self._replay_seed_queue.pop(0))
        seed = self._master_seed + self._next_board_seed_offset
        self._next_board_seed_offset += 1
        return ScenarioGenerator(seed=seed)

    def _load_replay_plan(self) -> None:
        """Resolve the replayed session's manifest + rows (D0).

        The manifest is REQUIRED — a replay against a session id with no
        manifest raises rather than silently labeling fresh boards, which would
        contaminate the corpus with rows the consistency report would then pair
        against nothing.

        The board SEQUENCE is taken from the original session's rows, not
        recomputed as ``master_seed + offset``: a skip in the original session
        burned a seed offset without writing a row, so re-deriving the sequence
        arithmetically would present boards that were never labeled. The
        manifest's ``master_seed`` is read and written back out as this
        session's ``replay_of_master_seed``, so a replay manifest records the
        seed the boards it PRESENTED came from — its own ``master_seed`` names
        the fresh sequence it never used.
        """
        assert self.replay_of_session is not None
        manifest_path = self.data_dir / _SESSIONS_DIR / self.replay_of_session / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"replay target session {self.replay_of_session!r} has no manifest at "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        self._replay_master_seed = int(manifest["master_seed"])

        rows = [
            r
            for r in load_scenarios(self.scenarios_path)
            if str(r["session_id"]) == self.replay_of_session
        ]
        if not rows:
            raise ValueError(
                f"replay target session {self.replay_of_session!r} wrote no rows to "
                f"{self.scenarios_path}"
            )
        seen: list[int] = []
        for row in rows:
            key = (int(row["game_seed"]), int(row["draft_position"]))
            if key in self._replay_rows:
                raise ValueError(
                    f"replay target session {self.replay_of_session!r} has two rows for "
                    f"game_seed={key[0]} draft_position={key[1]}; the replay would be "
                    f"ambiguous"
                )
            self._replay_rows[key] = row
            if key[0] not in seen:
                seen.append(key[0])
        self._replay_seed_queue = seen

    def _write_manifest(self, end_time: str | None) -> None:
        manifest: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": _epoch_to_iso(self._start_wall_time),
            "labeler_id": self.labeler_id,
            "scenarios_completed": self.scenarios_completed,
            "master_seed": self._master_seed,
            # D3: the mode + the scorer version live at label time are recorded
            # for BOTH arms, so the gate can identify the ``no_reveal`` control
            # picks by a ``session_id`` join even though those rows deliberately
            # carry no scorer fields of their own.
            "reveal_mode": self.reveal_mode,
            "scorer_version": self.scorer_version,
            "replay_of_session": self.replay_of_session,
            # The boards a replay session presents come from the REPLAYED
            # session's seed, not from ``master_seed`` above (which names the
            # fresh sequence this session generated and then never used). Both
            # are recorded so the provenance link is readable from the manifest
            # alone. ``None`` on a non-replay session.
            "replay_of_master_seed": self._replay_master_seed,
        }
        if end_time is not None:
            manifest["end_time"] = end_time
        self.manifest_path.write_text(json.dumps(manifest, indent=2))


def _validate_pick(scenario: Scenario, settlement_vertex: int, road_edge: int) -> None:
    """Raise if ``(settlement_vertex, road_edge)`` is illegal in ``scenario``.

    Mirrors the validation ``ScenarioGenerator.apply`` does. A replay session
    needs it separately because it advances the draft with the ORIGINAL pick,
    so ``apply`` never sees the owner's new pick — and an unvalidated pick
    would land an illegal row in the corpus.
    """
    if not 0 <= settlement_vertex < 54:
        raise ValueError(f"settlement_vertex_idx out of range: {settlement_vertex}")
    if not bool(scenario.legal_settlement_corners[settlement_vertex]):
        raise ValueError(
            f"illegal settlement vertex {settlement_vertex} at draft position "
            f"{scenario.draft_position}"
        )
    legal_roads = scenario.compute_legal_road_edges(settlement_vertex)
    if not 0 <= road_edge < 72:
        raise ValueError(f"road_edge_idx out of range: {road_edge}")
    if not bool(legal_roads[road_edge]):
        raise ValueError(
            f"illegal road edge {road_edge} at draft position {scenario.draft_position}"
        )


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_to_iso(monotonic_seconds: float) -> str:
    """Convert a session monotonic-start marker to an ISO-ish string.

    We use UTC-now at write time because monotonic timestamps are not
    interpretable as wall-clock. The manifest's start_time is the wall-
    clock time the session was started (approximated by the first
    manifest write).
    """
    return _utcnow_iso()
