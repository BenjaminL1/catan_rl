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
    _Sink,
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


def _strength_for(manifest_strength: str) -> OpponentStrength:
    """The :class:`OpponentStrength` a real ``parse_fn`` (via ``segment.py``) would
    stamp for a video of the given manifest strength — high→high, unknown→unknown.

    Kept faithful to the manifest so the record's tier is consistent with the
    manifest strength that admitted its video (``batch``'s §5.5 cross-check); a
    ``high``-tier record on an ``unknown`` video would (correctly) be rejected.
    """
    if manifest_strength == "high":
        return OpponentStrength(tier="high", source="tournament", confidence=0.8)
    return OpponentStrength(tier="unknown", source="rank_badge", confidence=0.0)


def _make_record(
    video_id: str,
    game_index: int,
    *,
    accepted: bool = True,
    manifest_strength: str = "high",
) -> GameRecord:
    """Build a real (contract-valid) :class:`GameRecord` via ``cross_check``.

    ``accepted`` toggles the winner-line legality: an accepted record has a real
    winner; a rejected one flips the board+openings desert weld so the
    orientation firewall rejects it (``passed_crosscheck=False`` + a typed
    ``rejection_reason``), exactly as a genuine reject would appear in the corpus.
    ``manifest_strength`` selects the opponent-strength tier the real pipeline
    would derive for that video (high→high, unknown→unknown), so the record's tier
    is consistent with ``batch``'s manifest cross-check.
    """
    board = _board_read()
    result = cross_check(
        video_id=video_id,
        game_index=game_index,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=_strength_for(manifest_strength),
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
    strengths = {"vidHIGH00000": "high", "vidUNKNOWN00": "unknown"}
    calls: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        calls.append(video_id)
        return [_make_record(video_id, 1, manifest_strength=strengths[video_id])]

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

    strengths = {"vidA00000000": "high", "vidB00000000": "high", "vidC00000000": "unknown"}
    processed: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        if video_id == "vidC00000000" and "vidB00000000" in processed:
            # Simulate a hard kill (SIGKILL-like) AFTER A and B are fully
            # committed but BEFORE C — an uncaught, non-VideoParseError crash.
            raise _Kill("killed mid-run")
        processed.append(video_id)
        s = strengths[video_id]
        return [
            _make_record(video_id, 1, manifest_strength=s),
            _make_record(video_id, 2, manifest_strength=s),
        ]

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


# --- mid-video kill: the un-committed tail games are recovered on resume -----


def test_mid_video_kill_recovers_tail_games_on_resume(tmp_path: Path) -> None:
    """A hard kill AFTER game 1 but BEFORE game 2 of a multi-game video must not
    silently drop game 2 (BLOCKER: correlated §5.6 data loss).

    Resume is game-granular: only a video-level ``done`` marker skips a video, and
    that marker is written LAST (after every game). A mid-video kill leaves no
    marker, so the video is re-parsed in full and the on-disk dedup drops game 1
    while re-emitting game 2.
    """
    manifest = _write_manifest(tmp_path, [_manifest_row("vidPARTIAL00", "high")])
    out = tmp_path / "out"

    # Reproduce the torn on-disk state a SIGKILL after game 1 leaves: game 1's
    # record + per-game ledger row are on disk, game 2 and the video-level ``done``
    # marker are NOT. This is byte-identical to a hard kill at that boundary
    # (interleaved per-game appends, no video-level marker yet).
    sink = _Sink(out_dir=out, now_fn=lambda: 0.0)
    game1 = _make_record("vidPARTIAL00", 1)
    from catan_rl.human_data.batch import _append_line

    _append_line(sink.corpus_path, game1.to_json_line())
    _append_line(
        sink.ledger_path,
        LedgerEntry("vidPARTIAL00", 1, "done", None, 0.0).to_json_line(),
    )

    # Before resume: game 1 is committed, but the video is NOT marked done.
    assert ("vidPARTIAL00", None) not in load_ledger(out / "ledger.jsonl")

    calls: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        calls.append(video_id)
        return [_make_record(video_id, 1), _make_record(video_id, 2)]

    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    # The torn video is RE-PARSED (not skipped) and game 2 is recovered.
    assert calls == ["vidPARTIAL00"]
    assert result.videos_processed == 1

    corpus = _read_jsonl(out / "corpus.jsonl")
    keys = sorted((r.video_id, r.game_index) for r in corpus)
    assert keys == [("vidPARTIAL00", 1), ("vidPARTIAL00", 2)], "game 2 must survive the kill"
    assert len(keys) == len(set(keys)), "game 1 must dedup, not duplicate"
    # A second resume now finds the terminal marker and skips.
    calls.clear()
    again = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert calls == []
    assert again.videos_skipped == 1


# --- manifest cross-check: an over-claimed high tier is rejected, not trusted --


def test_unknown_video_high_tier_record_routed_to_rejected(tmp_path: Path) -> None:
    """A ``parse_fn`` bug that stamps ``tier="high"`` on an ``unknown``-manifest
    video must NOT reach the scoreboard corpus (SHOULD-FIX: manifest is the §5.5
    source of truth). ``batch`` cross-checks the record's tier against the manifest
    strength and routes the mismatch to ``rejected.jsonl`` with a typed reason.
    """
    from catan_rl.human_data.batch import STRENGTH_MISMATCH_REASON

    manifest = _write_manifest(tmp_path, [_manifest_row("vidUNKNOWN00", "unknown")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        # BUG: an unknown video stamped high — must be caught by batch.
        return [_make_record(video_id, 1, manifest_strength="high")]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    assert result.records_accepted == 0
    assert result.records_rejected == 1
    corpus = _read_jsonl(out / "corpus.jsonl")
    rejected = _read_jsonl(out / "rejected.jsonl")
    assert corpus == []
    assert [r.game_index for r in rejected] == [1]
    assert rejected[0].rejection_reason == STRENGTH_MISMATCH_REASON
    assert not rejected[0].passed_crosscheck


def test_already_rejected_record_keeps_true_reason_on_strength_mismatch(tmp_path: Path) -> None:
    """A record that is BOTH an over-claimed ``high`` tier on an ``unknown`` video
    AND already a genuine reject (orientation weld) must keep its TRUE, feature-
    correlated ``rejection_reason`` in ``rejected.jsonl`` — the strength-mismatch
    stamp must NOT clobber the real cause (BLOCKER: red-team counterexample).

    The §5.6 rejection-bias audit buckets rejected games by their real cause
    (green-tile subtraction is harder near wood/sheep → rejection is feature-
    correlated); overwriting an ``orientation_mismatch_desert_hex`` cause with the
    strength reason biases that audit's per-archetype acceptance rates.
    """
    from catan_rl.human_data.batch import STRENGTH_MISMATCH_REASON

    manifest = _write_manifest(tmp_path, [_manifest_row("vidUNKNOWN00", "unknown")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        # Compound bug: an unknown video stamped high tier (parse over-claim) AND a
        # genuine orientation-weld reject (board=11 / openings=17). The record
        # already carries the true, feature-correlated rejection_reason.
        return [_make_record(video_id, 1, accepted=False, manifest_strength="high")]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)

    assert result.records_accepted == 0
    assert result.records_rejected == 1
    rejected = _read_jsonl(out / "rejected.jsonl")
    assert [r.game_index for r in rejected] == [1]
    reason = rejected[0].rejection_reason
    assert reason is not None
    # The TRUE orientation cause must survive (not be clobbered by the strength stamp).
    assert "orientation_mismatch_desert_hex:board=11:openings=17" in reason
    # The strength mismatch is appended (compound reason), not overwriting the cause.
    assert STRENGTH_MISMATCH_REASON in reason
    assert not rejected[0].passed_crosscheck


def test_high_video_high_tier_record_accepted(tmp_path: Path) -> None:
    """A ``high``-manifest video legitimately emitting ``tier="high"`` is accepted
    (the cross-check only rejects an OVER-claim, never a consistent one)."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1, manifest_strength="high")]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert result.records_accepted == 1
    assert result.records_rejected == 0


# --- download gate: opt-in parse_fn receives a net-concurrency semaphore -------


def test_download_gate_passed_to_opt_in_parse_fn(tmp_path: Path) -> None:
    """``run_batch`` passes a ``download_gate`` semaphore (bounded by
    ``net_concurrency``) to a ``parse_fn`` that opts in by naming the keyword —
    decoupling the 1—2-wide network phase from the ``max_workers``-wide OCR phase
    (SHOULD-FIX §5.11). A legacy 1-arg ``parse_fn`` is called unchanged.
    """
    import threading as _threading

    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    seen_gate: list[object] = []

    def parse_fn(video_id: str, *, download_gate: _threading.BoundedSemaphore) -> list[GameRecord]:
        with download_gate:  # a real parse_fn holds it only around the download
            seen_gate.append(download_gate)
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        net_concurrency=2,
    )
    assert result.videos_processed == 1
    assert len(seen_gate) == 1
    assert isinstance(seen_gate[0], _threading.BoundedSemaphore)


def test_legacy_single_arg_parse_fn_still_works(tmp_path: Path) -> None:
    """A ``parse_fn`` with no ``download_gate`` keyword is called with just the
    video id (backwards-compatible — the download gate is opt-in)."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    result = run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=1)
    assert result.videos_processed == 1


# --- multi-worker abort: completed in-flight videos are banked, not discarded --


def test_multiworker_abort_banks_completed_inflight(tmp_path: Path) -> None:
    """On an unexpected (non-VideoParseError) abort with ``max_workers>1``, an
    already-COMPLETED in-flight parse must be committed before the batch re-raises,
    so finished work (~10—20 min each) is not thrown away (SHOULD-FIX). The still
    un-committed videos are recovered on resume via the game-granular dedup.
    """
    import threading as _threading
    import time as _time

    rows = [_manifest_row(f"vid{i:08d}xxx", "high") for i in range(4)]
    manifest = _write_manifest(tmp_path, rows)
    out = tmp_path / "out"

    class _Abort(RuntimeError):
        pass

    # A sibling sets this as its LAST act before returning; the crasher waits on
    # it (plus a short settle) so at least one sibling future is ``done()`` when
    # the abort drains — deterministically exercising the bank-before-re-raise path.
    sibling_returned = _threading.Event()

    def parse_fn(video_id: str) -> list[GameRecord]:
        if video_id == "vid00000000xxx":
            assert sibling_returned.wait(timeout=5.0), "sibling never completed"
            _time.sleep(0.05)  # let the sibling future flip to done()
            raise _Abort("unexpected crash")
        recs = [_make_record(video_id, 1)]
        sibling_returned.set()
        return recs

    with pytest.raises(_Abort):
        run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_fn, max_workers=4)

    # At least one completed sibling was banked before the re-raise (not zero):
    corpus = _read_jsonl(out / "corpus.jsonl")
    banked = {r.video_id for r in corpus}
    assert banked, "a completed in-flight parse must be committed before abort"
    assert "vid00000000xxx" not in banked, "the crasher wrote nothing"

    # Resume: the crasher + any un-banked siblings are re-parsed; no dup.
    def parse_ok(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    run_batch(manifest_path=manifest, out_dir=out, parse_fn=parse_ok, max_workers=4)
    final = _read_jsonl(out / "corpus.jsonl")
    keys = sorted(r.video_id for r in final)
    assert keys == sorted(r["video_id"] for r in rows)
    assert len(keys) == len(set(keys)), "resume must not duplicate a banked video"


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
