"""Batch orchestration for the human-data video-parsing pipeline (build brief §4 / §6).

:func:`run_batch` is the parallel-per-video driver that turns ThePhantom's
YouTube corpus into the JSONL dataset. It:

- reads the committed strength manifest (``data/human/strength_manifest.json``)
  — THE opponent-strength source of truth — and **harvests the ``high`` +
  ``unknown`` videos** (``excluded`` and manifest-absent are dropped; the
  scoreboard consumes ``high`` only, seeds consume ``high`` + ``unknown``, per
  :func:`catan_rl.human_data.segment.segment_opponent_strength`);
- runs an **injected** per-video parse callable ``parse_fn(video_id) ->
  list[GameRecord]`` across a thread pool (the real download + ffmpeg + OCR CV
  pipeline is out of this slice's scope; ``batch`` owns only the orchestration);
- appends each accepted record (``passed_crosscheck=True``) to ``corpus.jsonl``
  and each rejected record to ``rejected.jsonl`` (rejected rows are KEPT for the
  §5.6 rejection-bias audit);
- maintains a per-``(video_id, game_index)`` **ledger** (``ledger.jsonl``) so a
  killed run is **resumable with no duplicate and no corrupt row**.

Durability / resume model
-------------------------
Each JSONL append is a single ``write`` + ``flush`` + ``fsync`` in ``"a"`` mode
(atomic for the small line sizes here on POSIX). A video is committed as a unit:
its records are written, then its ledger rows. Resume is made idempotent by
**deduping against the record keys already on disk** — before writing a video's
records, any ``(video_id, game_index)`` already present in ``corpus.jsonl`` /
``rejected.jsonl`` is skipped. So a hard kill between a record append and its
ledger append can never produce a duplicate: the resuming run sees the record
key on disk and does not re-emit it. A ``VideoParseError`` marks the whole video
``failed`` (retried on resume — not terminal). A crash mid-video may leave a
torn *final* line; :func:`load_ledger` and the corpus readers tolerate it by
skipping blank/partial trailing lines, and the video is re-parsed on resume
(its already-committed games dedup out).

CPU-only; never imports ``gui/`` or the training path.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from catan_rl.human_data.record import GameRecord
from catan_rl.human_data.segment import load_strength_manifest, manifest_entry

#: Manifest ``strength`` values that are harvested into the run. ``excluded``
#: (and manifest-absent) videos are dropped. ``high`` also feeds the scoreboard;
#: ``high`` + ``unknown`` feed the seed corpus (build brief §5.5).
HARVEST_STRENGTHS: frozenset[str] = frozenset({"high", "unknown"})

#: Ledger row status. ``done`` = one game's record committed;
#: ``failed`` = the whole video raised a :class:`VideoParseError` (retried on
#: resume — not terminal).
LedgerStatus = Literal["done", "failed"]

#: A video-level ledger row (a whole-video ``failed`` marker) has ``game_index``
#: ``None``; a per-game row has an ``int`` game index.
LedgerKey = tuple[str, int | None]


class VideoParseError(RuntimeError):
    """A recoverable per-video parse failure (bad download, OCR miss, cutoff …).

    Raising this from ``parse_fn`` marks the video ``failed`` in the ledger and
    lets :func:`run_batch` continue with the next video; the failed video is
    **retried** on the next (resuming) run. Any OTHER exception propagates and
    aborts the batch (an unexpected bug should not be silently swallowed).
    """


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One ledger row: the status of a ``(video_id, game_index)`` unit of work."""

    video_id: str
    game_index: int | None
    status: LedgerStatus
    error: str | None
    ts: float

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "video_id": self.video_id,
                "game_index": self.game_index,
                "status": self.status,
                "error": self.error,
                "ts": self.ts,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json_line(cls, line: str) -> LedgerEntry:
        payload = json.loads(line)
        return cls(
            video_id=str(payload["video_id"]),
            game_index=payload["game_index"],
            status=payload["status"],
            error=payload.get("error"),
            ts=float(payload["ts"]),
        )


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Summary of one :func:`run_batch` invocation."""

    videos_processed: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    harvested: tuple[str, ...] = field(default=())


def load_ledger(path: str | Path) -> dict[LedgerKey, LedgerEntry]:
    """Read ``ledger.jsonl`` into a ``{(video_id, game_index): LedgerEntry}`` map.

    Later rows win over earlier ones for the same key (append-only log). A torn /
    partial trailing line (from a hard kill) is tolerated and skipped.
    """
    path = Path(path)
    ledger: dict[LedgerKey, LedgerEntry] = {}
    if not path.exists():
        return ledger
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = LedgerEntry.from_json_line(line)
        except (json.JSONDecodeError, KeyError, ValueError):
            # A partial last line from a hard kill — skip it (resume re-derives).
            continue
        ledger[(entry.video_id, entry.game_index)] = entry
    return ledger


def _load_record_keys(path: Path) -> set[tuple[str, int]]:
    """Return the ``(video_id, game_index)`` keys already committed in a JSONL file.

    Used to make per-video commit idempotent on resume: a record whose key is
    already on disk is never re-appended, so a kill between a record append and
    its ledger append can never duplicate a row. A torn trailing line is skipped.
    """
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn trailing line
        vid = payload.get("video_id")
        gidx = payload.get("game_index")
        if isinstance(vid, str) and isinstance(gidx, int):
            keys.add((vid, gidx))
    return keys


def _append_line(path: Path, line: str) -> None:
    """Atomically append one line to ``path`` (flush + fsync so a kill can't tear it)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _harvest_video_ids(
    manifest: dict[str, object],
    video_ids: Sequence[str] | None,
) -> list[str]:
    """The ordered list of video ids to process: manifest videos whose ``strength``
    is in :data:`HARVEST_STRENGTHS`. If ``video_ids`` is given, restrict to that
    subset (still gated on the manifest — an id absent from the manifest, or one
    whose strength is ``excluded``, is dropped)."""
    videos = manifest.get("videos", [])
    rows: Iterable[object] = videos if isinstance(videos, list) else []
    harvestable: dict[str, None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = row.get("video_id")
        strength = row.get("strength")
        if isinstance(vid, str) and strength in HARVEST_STRENGTHS:
            harvestable[vid] = None
    if video_ids is None:
        return list(harvestable)
    return [vid for vid in video_ids if vid in harvestable]


@dataclass(frozen=True, slots=True)
class HarvestPlan:
    """The manifest-derived work plan for a batch run (no parsing performed).

    ``scoreboard`` (``high``) is the strict subset of ``harvested``
    (``high`` + ``unknown``) that feeds the opening scoreboard; the remainder are
    seed-only. ``excluded_count`` counts the dropped ``excluded`` manifest rows.
    """

    harvested: tuple[str, ...]
    scoreboard: tuple[str, ...]
    excluded_count: int


def harvest_plan(
    manifest_path: str | Path,
    *,
    video_ids: Sequence[str] | None = None,
) -> HarvestPlan:
    """Compute the :class:`HarvestPlan` for a manifest without parsing anything.

    Reads the committed strength manifest and partitions its videos exactly as
    :func:`run_batch` would (``high`` + ``unknown`` harvested, ``high`` also
    scoreboard-eligible, ``excluded`` dropped).
    """
    manifest = load_strength_manifest(manifest_path)
    harvested = _harvest_video_ids(manifest, video_ids)
    scoreboard = [
        vid
        for vid in harvested
        if (entry := manifest_entry(manifest, vid)) is not None and entry.get("strength") == "high"
    ]
    videos = manifest.get("videos", [])
    rows: Iterable[object] = videos if isinstance(videos, list) else []
    excluded = sum(1 for row in rows if isinstance(row, dict) and row.get("strength") == "excluded")
    return HarvestPlan(
        harvested=tuple(harvested),
        scoreboard=tuple(scoreboard),
        excluded_count=excluded,
    )


@dataclass
class _Sink:
    """Serial writer for the corpus / rejected / ledger JSONL files.

    All appends funnel through this single object so the parallel workers write
    through one owner (the main thread drains completed futures in order), which
    keeps every append atomic and dedup-consistent without a cross-file lock.
    """

    out_dir: Path
    now_fn: Callable[[], float]

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.out_dir / "corpus.jsonl"
        self.rejected_path = self.out_dir / "rejected.jsonl"
        self.ledger_path = self.out_dir / "ledger.jsonl"
        self._committed: set[tuple[str, int]] = _load_record_keys(
            self.corpus_path
        ) | _load_record_keys(self.rejected_path)

    def commit_video(self, video_id: str, records: Sequence[GameRecord]) -> tuple[int, int]:
        """Write one video's records + ledger rows. Returns ``(accepted, rejected)``.

        Idempotent: a ``(video_id, game_index)`` already on disk is skipped (so a
        resumed / retried video never duplicates a row).
        """
        accepted = rejected = 0
        for record in records:
            key = (record.video_id, record.game_index)
            if key in self._committed:
                continue
            line = record.to_json_line()
            if record.passed_crosscheck:
                _append_line(self.corpus_path, line)
                accepted += 1
            else:
                _append_line(self.rejected_path, line)
                rejected += 1
            self._committed.add(key)
            _append_line(
                self.ledger_path,
                LedgerEntry(
                    video_id=record.video_id,
                    game_index=record.game_index,
                    status="done",
                    error=None,
                    ts=self.now_fn(),
                ).to_json_line(),
            )
        return accepted, rejected

    def mark_failed(self, video_id: str, error: str) -> None:
        _append_line(
            self.ledger_path,
            LedgerEntry(
                video_id=video_id,
                game_index=None,
                status="failed",
                error=error,
                ts=self.now_fn(),
            ).to_json_line(),
        )


def _done_video_ids(ledger: dict[LedgerKey, LedgerEntry]) -> set[str]:
    """Videos with at least one committed (``done``) game — skip these on resume.

    A stale video-level ``failed`` marker does NOT block resume: once a retried
    video commits any ``done`` game it becomes skippable, and a ``failed``-only
    video (no ``done`` rows) is retried.
    """
    return {
        vid for (vid, gidx), entry in ledger.items() if gidx is not None and entry.status == "done"
    }


def run_batch(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    parse_fn: Callable[[str], list[GameRecord]],
    max_workers: int = 4,
    video_ids: Sequence[str] | None = None,
    now_fn: Callable[[], float] = time.time,
) -> BatchResult:
    """Parse the harvested corpus into ``corpus.jsonl`` + ``rejected.jsonl``.

    Args:
        manifest_path: the committed ``strength_manifest.json`` (source of truth).
        out_dir: destination for ``corpus.jsonl`` / ``rejected.jsonl`` /
            ``ledger.jsonl`` (created if absent). Existing files are appended to
            and resumed.
        parse_fn: injected per-video parser ``video_id -> list[GameRecord]``.
            Raise :class:`VideoParseError` for a recoverable failure (marks the
            video ``failed``, retried on resume); any other exception aborts the
            batch (already-committed videos survive; the in-flight one dedups on
            resume).
        max_workers: thread-pool width (CPU-only; the CV pipeline releases the
            GIL via ffmpeg / OCR subprocesses).
        video_ids: optional subset of ids to process (still manifest-gated).
        now_fn: injectable clock for the ledger timestamp (deterministic tests).

    Returns:
        A :class:`BatchResult` summary. Idempotent under resume: a re-run over an
        already-complete corpus processes nothing and duplicates no rows.
    """
    out_dir = Path(out_dir)
    manifest = load_strength_manifest(manifest_path)
    harvested = _harvest_video_ids(manifest, video_ids)

    ledger = load_ledger(out_dir / "ledger.jsonl")
    done = _done_video_ids(ledger)
    pending = [vid for vid in harvested if vid not in done]
    skipped = len(harvested) - len(pending)

    sink = _Sink(out_dir=out_dir, now_fn=now_fn)

    processed = failed = 0
    accepted_total = rejected_total = 0

    # Bound the pool by the actual pending work so a 1-video run stays single-
    # threaded (deterministic). Submit all, then drain in completion order —
    # every write happens on THIS thread through the single sink.
    workers = max(1, min(max_workers, len(pending))) if pending else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_vid = {pool.submit(parse_fn, vid): vid for vid in pending}
        for future in concurrent.futures.as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                records = future.result()
            except VideoParseError as exc:
                sink.mark_failed(vid, str(exc))
                failed += 1
                continue
            # Any other exception (a hard kill / unexpected bug) propagates and
            # aborts the batch — already-committed videos are durable on disk.
            acc, rej = sink.commit_video(vid, records)
            accepted_total += acc
            rejected_total += rej
            processed += 1

    return BatchResult(
        videos_processed=processed,
        videos_skipped=skipped,
        videos_failed=failed,
        records_accepted=accepted_total,
        records_rejected=rejected_total,
        harvested=tuple(harvested),
    )
