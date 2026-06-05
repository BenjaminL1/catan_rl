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

from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.scenario_gen import Scenario, ScenarioGenerator
from catan_rl.labeling.store import (
    SCHEMA_VERSION,
    append_scenario,
    count_scenarios,
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
            session.submit(settlement_vertex=..., road_edge=..., archetype=...)
        session.quit()
    """

    def __init__(
        self,
        data_dir: Path,
        labeler_id: str,
        session_seed: int | None = None,
    ) -> None:
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
        """The current scenario, or None if the session has been quit."""
        if self._quit:
            return None
        if self._gen is None:
            raise RuntimeError("session not started")
        return self._gen.current()

    def submit(
        self,
        settlement_vertex: int,
        road_edge: int,
        archetype: Archetype,
        notes: str = "",
        decision_time_ms: int = 0,
    ) -> None:
        """Record the current scenario and advance.

        Writes a JSONL row, applies the pick to the engine to advance
        the snake-draft state. If the 4th pick of a board is submitted,
        the next call to ``current_scenario()`` returns pick 1 of a
        fresh random board.
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
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "session_id": self.session_id,
            "labeled_at": _utcnow_iso(),
            "labeler_id": self.labeler_id,
            "game_seed": scenario.game_seed,
            "draft_position": scenario.draft_position,
            "acting_player": scenario.acting_player_idx,
            "prior_picks": [p.to_dict() for p in scenario.prior_picks],
            "archetype": archetype.value,
            "settlement_vertex": int(settlement_vertex),
            "road_edge": int(road_edge),
            "decision_time_ms": int(decision_time_ms),
            "notes": notes,
            "quality_flag": "fast" if decision_time_ms and decision_time_ms < 15000 else "",
        }
        # Apply to engine BEFORE persisting — if the pick is illegal,
        # the row never lands.
        self._gen.apply(int(settlement_vertex), int(road_edge))
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

    def _new_generator(self) -> ScenarioGenerator:
        """Build a fresh ScenarioGenerator with the next board seed."""
        seed = self._master_seed + self._next_board_seed_offset
        self._next_board_seed_offset += 1
        return ScenarioGenerator(seed=seed)

    def _write_manifest(self, end_time: str | None) -> None:
        manifest: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": _epoch_to_iso(self._start_wall_time),
            "labeler_id": self.labeler_id,
            "scenarios_completed": self.scenarios_completed,
            "master_seed": self._master_seed,
        }
        if end_time is not None:
            manifest["end_time"] = end_time
        self.manifest_path.write_text(json.dumps(manifest, indent=2))


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
