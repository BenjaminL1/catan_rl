"""Batch-orchestration tests (Stage-3 ``batch`` slice, build brief §4 / §6).

``batch.run_batch`` is the parallel-per-video driver that reads the committed
strength manifest (``data/human/strength_manifest.json``), harvests the
**high + unknown** videos (``excluded`` / manifest-absent dropped — the
scoreboard uses ``high`` only, seeds use ``high`` + ``unknown``), runs an
injected per-video parse callable, and appends the resulting :class:`GameRecord`
rows to the JSONL corpus (accepted) / ``rejected.jsonl`` (rejected, kept for the
§5.6 bias audit) with a per-``(video_id, game_index)`` ledger so a killed run is
resumable with **no duplicate and no corrupt** rows.

The real CV parse (download + ffmpeg + OCR) is out of this slice's scope and is
injected as ``parse_fn`` — ``batch`` owns only the manifest read, parallelism,
ledger, atomic appends, resume, and the accepted/rejected split. CPU-only; no
``gui/`` or training import.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catan_rl.human_data import (
    GameRecord,
    OpeningResult,
    OpponentStrength,
    PlayerOpening,
    load_topology,
)
from catan_rl.human_data.batch import (
    BatchResult,
    HarvestPlan,
    LedgerEntry,
    VideoParseError,
    harvest_plan,
    load_ledger,
    run_batch,
)
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.validate import cross_check

# --- a legal standard game-1 board + openings (mirrors test_validate) --------

_HEXES: tuple[dict[str, Any], ...] = (
    {"hex_id": 0, "resource": "SHEEP", "number": 11},
    {"hex_id": 1, "resource": "BRICK", "number": 9},
    {"hex_id": 2, "resource": "WHEAT", "number": 10},
    {"hex_id": 3, "resource": "WHEAT", "number": 3},
    {"hex_id": 4, "resource": "BRICK", "number": 6},
    {"hex_id": 5, "resource": "SHEEP", "number": 5},
    {"hex_id": 6, "resource": "ORE", "number": 4},
    {"hex_id": 7, "resource": "WOOD", "number": 6},
    {"hex_id": 8, "resource": "SHEEP", "number": 2},
    {"hex_id": 9, "resource": "WOOD", "number": 5},
    {"hex_id": 10, "resource": "BRICK", "number": 8},
    {"hex_id": 11, "resource": "DESERT", "number": None},
    {"hex_id": 12, "resource": "SHEEP", "number": 4},
    {"hex_id": 13, "resource": "WOOD", "number": 11},
    {"hex_id": 14, "resource": "WHEAT", "number": 12},
    {"hex_id": 15, "resource": "ORE", "number": 9},
    {"hex_id": 16, "resource": "ORE", "number": 10},
    {"hex_id": 17, "resource": "WOOD", "number": 8},
    {"hex_id": 18, "resource": "WHEAT", "number": 3},
)
_OPENINGS: dict[str, PlayerOpening] = {
    "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
    "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
}


def _board_read() -> BoardRead:
    return BoardRead(
        hexes=_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=11,
        residual_px=0.77,
        screen_rule_gap=43.5,
        pip_ok=True,
    )


def _make_record(
    video_id: str,
    game_index: int,
    *,
    accepted: bool = True,
) -> GameRecord:
    """Build a real (contract-valid) :class:`GameRecord` via ``cross_check``.

    ``accepted`` toggles the winner-line legality: an accepted record has a real
    winner; a rejected one flips the board+openings desert weld so the
    orientation firewall rejects it (``passed_crosscheck=False`` + a typed
    ``rejection_reason``), exactly as a genuine reject would appear in the corpus.
    """
    board = _board_read()
    result = cross_check(
        video_id=video_id,
        game_index=game_index,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
        board=board,
        openings_desert_hex=11 if accepted else 17,
        opening_result=OpeningResult(openings=dict(_OPENINGS), rejection_reason=None),
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        dice_log=(8, 6, 11, 4),
        winner="ThePhantom",
        resolution=1080,
        residual_px=board.residual_px,
        granted_by_player=None,
        topology=load_topology(),
    )
    assert result.accepted is accepted, "test fixture: unexpected gate verdict"
    return result.record


def _write_manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "strength_manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "videos": rows}, indent=1),
        encoding="utf-8",
    )
    return path


def _manifest_row(video_id: str, strength: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
        "title": "t",
        "strength": strength,
        "source": "tournament" if strength == "high" else "none",
        "evidence": {},
    }


def _read_jsonl(path: Path) -> list[GameRecord]:
    if not path.exists():
        return []
    return [
        GameRecord.from_json_line(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- harvest set: high + unknown, drop excluded + manifest-absent ------------


def test_harvest_covers_high_and_unknown_drops_excluded(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            _manifest_row("vidHIGH00000", "high"),
            _manifest_row("vidUNKNOWN00", "unknown"),
            _manifest_row("vidEXCLUDED0", "excluded"),
        ],
    )
    seen: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        seen.append(video_id)
        return [_make_record(video_id, 1)]

    result = run_batch(
        manifest_path=manifest,
        out_dir=tmp_path / "out",
        parse_fn=parse_fn,
        max_workers=1,
    )
    assert set(seen) == {"vidHIGH00000", "vidUNKNOWN00"}
    assert "vidEXCLUDED0" not in seen
    assert result.videos_processed == 2


# --- accepted -> corpus.jsonl, rejected -> rejected.jsonl --------------------


def test_accepted_and_rejected_split_into_two_files(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [
            _make_record(video_id, 1, accepted=True),
            _make_record(video_id, 2, accepted=False),
        ]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    corpus = _read_jsonl(out / "corpus.jsonl")
    rejected = _read_jsonl(out / "rejected.jsonl")
    assert [r.game_index for r in corpus] == [1]
    assert [r.game_index for r in rejected] == [2]
    assert all(r.passed_crosscheck for r in corpus)
    assert all(not r.passed_crosscheck for r in rejected)
    assert result.records_accepted == 1
    assert result.records_rejected == 1


# --- ledger: one row per (video_id, game_index), done status -----------------


def test_ledger_records_every_game(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1), _make_record(video_id, 2)]

    out = tmp_path / "out"
    run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    ledger = load_ledger(out / "ledger.jsonl")
    assert ("vidHIGH00000", 1) in ledger
    assert ("vidHIGH00000", 2) in ledger
    assert all(e.status == "done" for e in ledger.values())


# --- resume: a second run skips already-done videos --------------------------


def test_resume_skips_done_videos(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [_manifest_row("vidHIGH00000", "high"), _manifest_row("vidUNKNOWN00", "unknown")],
    )
    calls: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        calls.append(video_id)
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert sorted(calls) == ["vidHIGH00000", "vidUNKNOWN00"]

    # Second run: everything is already done -> parse_fn must not run again, and
    # the corpus must NOT gain duplicate rows.
    calls.clear()
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert calls == []
    assert result.videos_processed == 0
    assert result.videos_skipped == 2

    corpus = _read_jsonl(out / "corpus.jsonl")
    assert len(corpus) == 2
    keys = [(r.video_id, r.game_index) for r in corpus]
    assert len(keys) == len(set(keys)), "resume must not duplicate corpus rows"


# --- resume retries a transiently-failed video -------------------------------


def test_resume_retries_transient_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    attempts = itertools.count()

    def parse_fn(video_id: str) -> list[GameRecord]:
        n = next(attempts)
        if n == 0:
            raise VideoParseError("transient download failure")
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    first = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert first.videos_failed == 1
    assert first.records_accepted == 0

    ledger = load_ledger(out / "ledger.jsonl")
    assert ledger[("vidHIGH00000", None)].status == "failed"

    # A failed video is NOT terminal — the resuming run retries it.
    second = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert second.videos_processed == 1
    assert second.records_accepted == 1

    ledger = load_ledger(out / "ledger.jsonl")
    assert ledger[("vidHIGH00000", 1)].status == "done"
    # The append-only ledger keeps the stale video-level ``failed`` marker (a log
    # never rewrites history), but it no longer wins on resume: a THIRD run must
    # skip the now-``done`` video and not re-parse it.
    calls_before = next(attempts)
    third = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert third.videos_skipped == 1
    assert third.videos_processed == 0
    assert next(attempts) == calls_before + 1, "resume must not re-invoke parse_fn"


# --- kill-and-resume: a mid-run crash leaves no dup / no corrupt rows --------


def test_kill_and_resume_no_dup_no_corruption(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            _manifest_row("vidA00000000", "high"),
            _manifest_row("vidB00000000", "high"),
            _manifest_row("vidC00000000", "unknown"),
        ],
    )

    class _Kill(RuntimeError):
        pass

    processed: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        if video_id == "vidC00000000" and "vidB00000000" in processed:
            # Simulate a hard kill (SIGKILL-like) AFTER A and B are fully
            # committed but BEFORE C — an uncaught, non-VideoParseError crash.
            raise _Kill("killed mid-run")
        processed.append(video_id)
        return [_make_record(video_id, 1), _make_record(video_id, 2)]

    out = tmp_path / "out"
    with pytest.raises(_Kill):
        run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    # A + B committed atomically; C never wrote a partial row.
    corpus_after_kill = _read_jsonl(out / "corpus.jsonl")
    keys = sorted((r.video_id, r.game_index) for r in corpus_after_kill)
    assert keys == [
        ("vidA00000000", 1),
        ("vidA00000000", 2),
        ("vidB00000000", 1),
        ("vidB00000000", 2),
    ]
    # Every persisted corpus line is valid JSON (no torn last line).
    for line in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)

    # Resume: C now succeeds; A + B are skipped (no re-parse, no dup).
    processed.clear()
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert processed == ["vidC00000000"]
    assert result.videos_skipped == 2

    corpus = _read_jsonl(out / "corpus.jsonl")
    final_keys = sorted((r.video_id, r.game_index) for r in corpus)
    assert final_keys == [
        ("vidA00000000", 1),
        ("vidA00000000", 2),
        ("vidB00000000", 1),
        ("vidB00000000", 2),
        ("vidC00000000", 1),
        ("vidC00000000", 2),
    ]
    assert len(final_keys) == len(set(final_keys)), "resume duplicated a row"


# --- manifest-absent videos are never harvested ------------------------------


def test_manifest_absent_video_not_harvested(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    seen: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        seen.append(video_id)
        return [_make_record(video_id, 1)]

    run_batch(
        manifest_path=manifest,
        out_dir=tmp_path / "out",
        parse_fn=parse_fn,
        max_workers=1,
        video_ids=["vidHIGH00000", "vidNOTINMANIFEST"],
    )
    assert seen == ["vidHIGH00000"]


# --- parallel workers: all harvested videos get parsed exactly once ----------


def test_parallel_workers_parse_each_video_once(tmp_path: Path) -> None:
    rows = [_manifest_row(f"vid{i:08d}xxx", "high") for i in range(8)]
    manifest = _write_manifest(tmp_path, rows)

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=4)
    assert result.videos_processed == 8

    corpus = _read_jsonl(out / "corpus.jsonl")
    keys = sorted(r.video_id for r in corpus)
    assert keys == sorted(r["video_id"] for r in rows)
    assert len(keys) == len(set(keys))


# --- harvest_plan: manifest partition without parsing ------------------------


def test_harvest_plan_partitions_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            _manifest_row("vidHIGH00001", "high"),
            _manifest_row("vidHIGH00002", "high"),
            _manifest_row("vidUNKNOWN00", "unknown"),
            _manifest_row("vidEXCLUDED0", "excluded"),
        ],
    )
    plan = harvest_plan(manifest)
    assert isinstance(plan, HarvestPlan)
    assert set(plan.harvested) == {"vidHIGH00001", "vidHIGH00002", "vidUNKNOWN00"}
    assert set(plan.scoreboard) == {"vidHIGH00001", "vidHIGH00002"}
    assert plan.excluded_count == 1


# --- ledger typing surface ---------------------------------------------------


def test_ledger_entry_and_result_shapes(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    result = run_batch(
        manifest_path=manifest,
        out_dir=tmp_path / "out",
        parse_fn=parse_fn,
        max_workers=1,
    )
    assert isinstance(result, BatchResult)
    ledger = load_ledger(tmp_path / "out" / "ledger.jsonl")
    entry = ledger[("vidHIGH00000", 1)]
    assert isinstance(entry, LedgerEntry)
    assert entry.status == "done"
    assert entry.error is None
    assert isinstance(entry.ts, float)
