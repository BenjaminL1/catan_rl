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
from collections import Counter
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
from catan_rl.human_data.glyph_anchor import GlyphValidation, glyph_classifier_fingerprint
from catan_rl.human_data.orientation import (
    GlyphClassifierNotValidated,
    granted_resources_under_orientation,
)
from catan_rl.human_data.validate import (
    GLYPH_MISMATCH_REASON,
    GLYPH_UNREADABLE_REASON,
    cross_check,
)

#: A passing glyph-classifier validation (mirrors the committed
#: ``data/human/glyph_validation.json`` PASS). ``run_batch`` is HARD-GATED on one
#: (expert BLOCKER 1) — every test that expects the batch to actually run supplies
#: this; the gate itself is exercised in
#: ``test_batch_hard_blocks_without_glyph_validation``. The record must carry the
#: CURRENT classifier's fingerprint (expert SHOULD-FIX 2026-07-05: the gate is
#: bound to classifier identity — a ``passed=True`` record without it is exactly
#: the fabricated PASS the gate now rejects, see
#: ``test_batch_hard_blocks_on_unbound_glyph_validation``).
_VALIDATION = GlyphValidation(
    passed=True,
    n_frames=24,
    n_correct=24,
    accuracy=1.0,
    reason=None,
    classifier_fingerprint=glyph_classifier_fingerprint(),
)

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


def _grants(kind: str = "valid") -> dict[str, Counter[str] | None] | None:
    """Grant reads for both players, per firewall scenario.

    ``"valid"`` — readable + orientation-consistent (anchor runs and passes);
    ``"unreadable"`` — ThePhantom's read is ``None`` (anchor cannot run →
    :data:`GLYPH_UNREADABLE_REASON`); ``"mismatch"`` — a confident wrong read
    (anchor runs and fires → :data:`GLYPH_MISMATCH_REASON`); ``"absent"`` — no
    read at all (the old fail-open path, now a typed reject).
    """
    if kind == "absent":
        return None
    topo = load_topology()
    by_hex = {int(h["hex_id"]): str(h["resource"]) for h in _HEXES}
    valid: dict[str, Counter[str] | None] = {
        "ThePhantom": granted_resources_under_orientation(19, by_hex, topo),
        "rayman147": granted_resources_under_orientation(3, by_hex, topo),
    }
    if kind == "unreadable":
        valid["ThePhantom"] = None
    elif kind == "mismatch":
        # No vertex on this board touches 3 ore hexes — a confidently-wrong read.
        valid["ThePhantom"] = Counter({"ORE": 3})
    return valid


def _make_record(
    video_id: str,
    game_index: int,
    *,
    accepted: bool = True,
    manifest_strength: str = "high",
    glyphs: str = "valid",
) -> GameRecord:
    """Build a real (contract-valid) :class:`GameRecord` via ``cross_check``.

    ``accepted`` toggles the winner-line legality: an accepted record has a real
    winner; a rejected one flips the board+openings desert weld so the
    orientation firewall rejects it (``passed_crosscheck=False`` + a typed
    ``rejection_reason``), exactly as a genuine reject would appear in the corpus.
    ``manifest_strength`` selects the opponent-strength tier the real pipeline
    would derive for that video (high→high, unknown→unknown), so the record's tier
    is consistent with ``batch``'s manifest cross-check. ``glyphs`` selects the
    grant-read scenario (:func:`_grants`) — the joint-flip firewall is
    NON-OPTIONAL, so an accepted record always carries readable, matching grants.
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
        granted_by_player=_grants(glyphs),
        topology=load_topology(),
    )
    expect_accept = accepted and glyphs == "valid"
    assert result.accepted is expect_accept, "test fixture: unexpected gate verdict"
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
        glyph_validation=_VALIDATION,
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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

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
    run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

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
    run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert sorted(calls) == ["vidHIGH00000", "vidUNKNOWN00"]

    # Second run: everything is already done -> parse_fn must not run again, and
    # the corpus must NOT gain duplicate rows.
    calls.clear()
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
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
    first = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert first.videos_failed == 1
    assert first.records_accepted == 0

    ledger = load_ledger(out / "ledger.jsonl")
    assert ledger[("vidHIGH00000", None)].status == "failed"

    # A failed video is NOT terminal — the resuming run retries it.
    second = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert second.videos_processed == 1
    assert second.records_accepted == 1

    ledger = load_ledger(out / "ledger.jsonl")
    assert ledger[("vidHIGH00000", 1)].status == "done"
    # The append-only ledger keeps the stale video-level ``failed`` marker (a log
    # never rewrites history), but it no longer wins on resume: a THIRD run must
    # skip the now-``done`` video and not re-parse it.
    calls_before = next(attempts)
    third = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
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
        run_batch(
            manifest_path=manifest,
            out_dir=out,
            parse_fn=parse_fn,
            max_workers=1,
            glyph_validation=_VALIDATION,
        )

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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
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

    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

    # The torn video is RE-PARSED (not skipped) and game 2 is recovered.
    assert calls == ["vidPARTIAL00"]
    assert result.videos_processed == 1

    corpus = _read_jsonl(out / "corpus.jsonl")
    keys = sorted((r.video_id, r.game_index) for r in corpus)
    assert keys == [("vidPARTIAL00", 1), ("vidPARTIAL00", 2)], "game 2 must survive the kill"
    assert len(keys) == len(set(keys)), "game 1 must dedup, not duplicate"
    # A second resume now finds the terminal marker and skips.
    calls.clear()
    again = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert calls == []
    assert again.videos_skipped == 1


# --- torn-line firewall: a partial line from a SIGKILL is not welded on resume --


def test_torn_partial_line_not_welded_on_resume(tmp_path: Path) -> None:
    """A hard kill (SIGKILL) DURING an append can leave a partial line with no
    trailing newline. On resume, the next append must NOT concatenate onto that
    torn tail (welding an unparseable line into the MIDDLE of the file) — it must
    demote the torn tail to a standalone skippable partial line first (red-team
    BLOCKER).

    Reproduces the probe: corpus.jsonl holds a valid game-1 row + a TORN game-2
    row (no ``\\n``); resume re-parses the (un-marked-done) video and every persisted
    line must round-trip through ``GameRecord.from_json_line`` — no mid-file corrupt
    line, no silent drop of the re-parsed game.
    """
    from catan_rl.human_data.batch import _append_line

    manifest = _write_manifest(tmp_path, [_manifest_row("vidTORN00000", "high")])
    out = tmp_path / "out"

    sink = _Sink(out_dir=out, now_fn=lambda: 0.0)
    game1 = _make_record("vidTORN00000", 1)
    # A committed game-1 row + its per-game ledger row (a SIGKILL after game 1).
    _append_line(sink.corpus_path, game1.to_json_line())
    _append_line(
        sink.ledger_path,
        LedgerEntry("vidTORN00000", 1, "done", None, 0.0).to_json_line(),
    )
    # Now simulate the OS committing only PART of game 2's row (no trailing newline)
    # before the SIGKILL — a torn partial line at the tail of corpus.jsonl.
    torn_head = game1.to_json_line()[:50]
    assert not torn_head.endswith("\n")
    with sink.corpus_path.open("a", encoding="utf-8") as fh:
        fh.write(torn_head)  # deliberately NO newline — a torn write

    # Sanity: corpus.jsonl currently ends without a newline (the torn tail).
    assert not sink.corpus_path.read_text(encoding="utf-8").endswith("\n")

    calls: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:
        calls.append(video_id)
        return [_make_record(video_id, 1), _make_record(video_id, 2)]

    run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert calls == ["vidTORN00000"], "the torn (un-marked-done) video must re-parse"

    # No welded corrupt line: every non-blank line either round-trips as a record
    # or is a skippable partial line (the demoted torn head), NEVER a welded hybrid
    # sitting in the middle of the file.
    lines = sink.corpus_path.read_text(encoding="utf-8").splitlines()
    valid = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            GameRecord.from_json_line(line)
            valid += 1
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Only the demoted torn head may fail — and it must be a strict PREFIX
            # of a real row (never a welded head+full-row hybrid).
            assert line == torn_head, f"welded corrupt line in the middle: {line!r}"

    # Both games survive the resume (game 1 dedups, game 2 recovered) — no drop.
    # A tolerant reader (the real corpus consumer contract: skip blank/partial
    # lines) recovers exactly the two full records; only the demoted torn head is
    # skipped, never a welded hybrid that would drop a valid game.
    records: list[GameRecord] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(GameRecord.from_json_line(line))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    keys = sorted((r.video_id, r.game_index) for r in records)
    assert keys == [("vidTORN00000", 1), ("vidTORN00000", 2)]
    assert valid == 2, "both re-parsed games must be readable"


def test_append_line_demotes_torn_tail_across_all_files(tmp_path: Path) -> None:
    """``_append_line`` itself must never weld a new row onto an unterminated tail
    — the guard applies to corpus.jsonl, rejected.jsonl AND ledger.jsonl (a torn
    video-level ``done`` marker welded with the next ledger row would make
    ``load_ledger`` drop a legitimately-written marker and force a needless
    re-parse). Direct unit check of the byte-level guard.
    """
    from catan_rl.human_data.batch import _append_line

    path = tmp_path / "f.jsonl"
    path.write_text('{"a":1}', encoding="utf-8")  # a torn tail: no trailing newline
    _append_line(path, '{"b":2}')

    lines = path.read_text(encoding="utf-8").splitlines()
    # The torn head stayed a standalone (skippable) line; the new row is separate.
    assert lines == ['{"a":1}', '{"b":2}']
    for line in lines:
        json.loads(line)  # both parse cleanly — no welded hybrid
    # And a normal (terminated) file appends without a spurious blank line.
    _append_line(path, '{"c":3}')
    assert path.read_text(encoding="utf-8").splitlines() == ['{"a":1}', '{"b":2}', '{"c":3}']


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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
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
        glyph_validation=_VALIDATION,
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
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
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
        run_batch(
            manifest_path=manifest,
            out_dir=out,
            parse_fn=parse_fn,
            max_workers=4,
            glyph_validation=_VALIDATION,
        )

    # At least one completed sibling was banked before the re-raise (not zero):
    corpus = _read_jsonl(out / "corpus.jsonl")
    banked = {r.video_id for r in corpus}
    assert banked, "a completed in-flight parse must be committed before abort"
    assert "vid00000000xxx" not in banked, "the crasher wrote nothing"

    # Resume: the crasher + any un-banked siblings are re-parsed; no dup.
    def parse_ok(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_ok,
        max_workers=4,
        glyph_validation=_VALIDATION,
    )
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
        glyph_validation=_VALIDATION,
    )
    assert seen == ["vidHIGH00000"]


# --- parallel workers: all harvested videos get parsed exactly once ----------


def test_parallel_workers_parse_each_video_once(tmp_path: Path) -> None:
    rows = [_manifest_row(f"vid{i:08d}xxx", "high") for i in range(8)]
    manifest = _write_manifest(tmp_path, rows)

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [_make_record(video_id, 1)]

    out = tmp_path / "out"
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=4,
        glyph_validation=_VALIDATION,
    )
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
        glyph_validation=_VALIDATION,
    )
    assert isinstance(result, BatchResult)
    ledger = load_ledger(tmp_path / "out" / "ledger.jsonl")
    entry = ledger[("vidHIGH00000", 1)]
    assert isinstance(entry, LedgerEntry)
    assert entry.status == "done"
    assert entry.error is None
    assert isinstance(entry.ts, float)


# --- the harvest is HARD-GATED on a validated glyph classifier (BLOCKER 1) ----


def test_batch_hard_blocks_without_glyph_validation(tmp_path: Path) -> None:
    """``run_batch`` with an ABSENT validation must raise
    :class:`GlyphClassifierNotValidated` before parsing anything — the joint-flip
    firewall can never be silently absent from a harvest run."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    calls: list[str] = []

    def parse_fn(video_id: str) -> list[GameRecord]:  # pragma: no cover - must not run
        calls.append(video_id)
        return [_make_record(video_id, 1)]

    with pytest.raises(GlyphClassifierNotValidated):
        run_batch(
            manifest_path=manifest,
            out_dir=tmp_path / "out",
            parse_fn=parse_fn,
            max_workers=1,
        )
    assert calls == [], "the gate must fire BEFORE any video is parsed"
    assert not (tmp_path / "out" / "corpus.jsonl").exists()


def test_batch_hard_blocks_on_failed_glyph_validation(tmp_path: Path) -> None:
    """A FAILED validation (``passed=False``) blocks exactly like an absent one —
    ``glyph_classifier_is_validated`` is the single switch the gate consults."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    failed = GlyphValidation(
        passed=False, n_frames=4, n_correct=2, accuracy=0.5, reason="below bar"
    )

    def parse_fn(video_id: str) -> list[GameRecord]:  # pragma: no cover - must not run
        raise AssertionError("parse_fn must not be invoked under a failed validation")

    with pytest.raises(GlyphClassifierNotValidated):
        run_batch(
            manifest_path=manifest,
            out_dir=tmp_path / "out",
            parse_fn=parse_fn,
            max_workers=1,
            glyph_validation=failed,
        )


def test_batch_hard_blocks_on_unbound_glyph_validation(tmp_path: Path) -> None:
    """A ``passed=True`` record NOT bound to the current classifier blocks the
    batch (expert SHOULD-FIX 2026-07-05): a FABRICATED pass (no fingerprint —
    e.g. a hand-edited ``glyph_validation.json``) and a STALE pass (a fingerprint
    stamped by an earlier, since-edited classifier) must both fail the gate —
    only :func:`validate_glyph_classifier` run against THIS classifier can
    unblock a harvest."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])
    fabricated = GlyphValidation(passed=True, n_frames=24, n_correct=24, accuracy=1.0, reason=None)
    stale = GlyphValidation(
        passed=True,
        n_frames=24,
        n_correct=24,
        accuracy=1.0,
        reason=None,
        classifier_fingerprint="0" * 64,  # an older classifier's identity
    )

    def parse_fn(video_id: str) -> list[GameRecord]:  # pragma: no cover - must not run
        raise AssertionError("parse_fn must not be invoked under an unbound validation")

    for validation in (fabricated, stale):
        with pytest.raises(GlyphClassifierNotValidated):
            run_batch(
                manifest_path=manifest,
                out_dir=tmp_path / "out",
                parse_fn=parse_fn,
                max_workers=1,
                glyph_validation=validation,
            )
    assert not (tmp_path / "out" / "corpus.jsonl").exists()


# --- firewall telemetry: how often the anchor actually executed ---------------


def test_batch_anchor_telemetry_counts_on_synthetic_mix(tmp_path: Path) -> None:
    """The run summary reports {anchor_ran, anchor_unreadable, anchor_mismatch} +
    grant-read coverage over the committed records: an accepted record (anchor ran
    and passed), a glyph-unreadable reject (anchor could not run), a glyph-mismatch
    reject (anchor ran and fired) and a coarse orientation-weld reject (rejected
    before the glyph gate — anchor never ran)."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return [
            _make_record(video_id, 1, glyphs="valid"),  # accepted; anchor ran
            _make_record(video_id, 2, glyphs="unreadable"),  # reject: unreadable
            _make_record(video_id, 3, glyphs="mismatch"),  # reject: anchor fired
            _make_record(video_id, 4, accepted=False),  # coarse weld reject
        ]

    out = tmp_path / "out"
    result = run_batch(
        manifest_path=manifest,
        out_dir=out,
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )

    assert result.records_accepted == 1
    assert result.records_rejected == 3
    # anchor ran for the accepted record AND the mismatch reject (it executed and
    # fired); not for the unreadable or the coarse-weld reject.
    assert result.anchor_ran == 2
    assert result.anchor_unreadable == 1
    assert result.anchor_mismatch == 1
    assert result.grant_read_coverage == pytest.approx(2 / 4)

    # The typed reasons landed in rejected.jsonl (the audit sees the true causes).
    rejected = _read_jsonl(out / "rejected.jsonl")
    reasons = {r.game_index: r.rejection_reason or "" for r in rejected}
    assert reasons[2] == GLYPH_UNREADABLE_REASON
    assert reasons[3] == GLYPH_MISMATCH_REASON
    assert "orientation_mismatch_desert_hex" in reasons[4]


def test_batch_telemetry_zero_coverage_when_nothing_committed(tmp_path: Path) -> None:
    """Coverage is 0.0 (not a ZeroDivisionError) when a run commits no records."""
    manifest = _write_manifest(tmp_path, [_manifest_row("vidHIGH00000", "high")])

    def parse_fn(video_id: str) -> list[GameRecord]:
        return []

    result = run_batch(
        manifest_path=manifest,
        out_dir=tmp_path / "out",
        parse_fn=parse_fn,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert result.anchor_ran == 0
    assert result.anchor_unreadable == 0
    assert result.anchor_mismatch == 0
    assert result.grant_read_coverage == 0.0
