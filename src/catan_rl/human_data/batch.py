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
(atomic for the small line sizes here on POSIX). Commit is **per game** (a game's
record is written, then its ``done`` ledger row), and resume is **game-granular**:
after all of a video's games are committed, a single video-level ``done`` marker
(``game_index=None``) is appended, and ONLY that marker makes the video skippable
on resume (:func:`_done_video_ids`). A video killed mid-commit — after game 1's
rows but before game 2's, whether by a hard SIGKILL or any non-``VideoParseError``
raised during a later append — has NO video-level marker, so the resuming run
**re-parses it in full** and the on-disk ``_committed`` dedup drops the games
already written (game 1 dedups out, game 2 is re-emitted). This closes the §5.6
rejection-bias hazard where a mid-video kill silently dropped a multi-game video's
tail games (its modal shape — ThePhantom videos hold many back-to-back games).

Resume idempotence rests on that same **dedup against the record keys already on
disk**: before writing a game, any ``(video_id, game_index)`` already present in
``corpus.jsonl`` / ``rejected.jsonl`` is skipped, so a hard kill between a record
append and its per-game ledger append can never duplicate a row. A
``VideoParseError`` marks the whole video ``failed`` (retried on resume — not
terminal). A crash mid-video may leave a torn *final* line; :func:`load_ledger`
and the corpus readers tolerate it by skipping blank/partial trailing lines.

CPU-only; never imports ``gui/`` or the training path.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from catan_rl.human_data.record import GameRecord
from catan_rl.human_data.segment import load_strength_manifest, manifest_entry

#: Manifest ``strength`` values that are harvested into the run. ``excluded``
#: (and manifest-absent) videos are dropped. ``high`` also feeds the scoreboard;
#: ``high`` + ``unknown`` feed the seed corpus (build brief §5.5).
HARVEST_STRENGTHS: frozenset[str] = frozenset({"high", "unknown"})

#: Ledger row status. A per-game row (``game_index`` is an ``int``) with
#: ``done`` = that game's record committed. A video-level row (``game_index`` is
#: ``None``): ``done`` = the WHOLE video's games are all committed (the terminal
#: per-video marker that makes the video skippable on resume — see
#: :func:`_done_video_ids`); ``failed`` = the whole video raised a
#: :class:`VideoParseError` (retried on resume — not terminal).
LedgerStatus = Literal["done", "failed"]

#: A video-level ledger row (a whole-video ``done`` / ``failed`` marker) has
#: ``game_index`` ``None``; a per-game row has an ``int`` game index.
LedgerKey = tuple[str, int | None]

#: ``rejection_reason`` stamped on a record whose ``opponent_strength.tier`` is
#: inconsistent with the manifest strength that admitted its video (a
#: ``tier="high"`` record from an ``unknown``-manifest video). The manifest is THE
#: source of truth (§5.5); a ``parse_fn`` bug that over-claims ``high`` would
#: silently contaminate the scoreboard, so ``batch`` routes such a record to
#: ``rejected.jsonl`` with this typed reason rather than trusting the callable.
STRENGTH_MISMATCH_REASON = "manifest_strength_mismatch"


#: The injected per-video parser. ``video_id -> list[GameRecord]``. It MAY also
#: accept a ``download_gate`` keyword — a :class:`threading.BoundedSemaphore` that
#: caps concurrent yt-dlp downloads to :func:`run_batch`'s ``net_concurrency``
#: (§5.11: the pre-resolved googlevideo URL is short-lived and YouTube throttles
#: parallel pulls). A ``parse_fn`` that downloads should acquire it around ONLY the
#: download phase, leaving the CPU/OCR phase to fan out to the full ``max_workers``.
#: ``run_batch`` detects the keyword by signature and omits it for a legacy
#: 1-arg ``parse_fn``, so both shapes work.
ParseFn = Callable[..., list[GameRecord]]


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
    #: ``{video_id: manifest_strength}`` for the harvested corpus. Used to
    #: cross-check each record's ``opponent_strength.tier`` against the manifest
    #: strength that admitted its video (§5.5) — the manifest is the source of
    #: truth, so a ``parse_fn`` that stamps ``tier="high"`` on an ``unknown``
    #: video is rejected here, not trusted.
    manifest_strength: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.out_dir / "corpus.jsonl"
        self.rejected_path = self.out_dir / "rejected.jsonl"
        self.ledger_path = self.out_dir / "ledger.jsonl"
        self._committed: set[tuple[str, int]] = _load_record_keys(
            self.corpus_path
        ) | _load_record_keys(self.rejected_path)

    def _strength_ok(self, record: GameRecord) -> bool:
        """Whether the record's tier is consistent with its video's manifest strength.

        A ``high`` manifest video may emit ``tier="high"`` or ``tier="unknown"``;
        an ``unknown`` manifest video must NOT emit ``tier="high"`` (that would
        silently promote it into the scoreboard, contaminating the §5.5
        mixed-strength filter). A video absent from the strength map (only
        possible in a direct-sink unit test — ``run_batch`` always populates it
        from the manifest it harvested) is not cross-checked.
        """
        manifest_strength = self.manifest_strength.get(record.video_id)
        if manifest_strength is None:
            return True
        return not (manifest_strength != "high" and record.opponent_strength.tier == "high")

    def commit_video(self, video_id: str, records: Sequence[GameRecord]) -> tuple[int, int]:
        """Write one video's per-game rows, then a terminal video-level ``done`` marker.

        Returns ``(accepted, rejected)``. Per game: idempotent — a
        ``(video_id, game_index)`` already on disk is skipped (so a re-parsed /
        resumed video never duplicates a row). A record whose
        ``opponent_strength.tier`` is inconsistent with its manifest strength
        (:meth:`_strength_ok`) is downgraded to a rejected row carrying
        :data:`STRENGTH_MISMATCH_REASON` — a loud failure in ``rejected.jsonl``
        rather than a silent scoreboard contamination.

        The video-level ``done`` marker is appended **only after every game is
        committed** and is the SOLE resume-skip signal (:func:`_done_video_ids`).
        A mid-video kill leaves no such marker, so the resuming run re-parses the
        video in full and the per-game dedup drops the games already written —
        the un-committed tail games are recovered, not silently lost (§5.6).
        """
        accepted = rejected = 0
        for record in records:
            key = (record.video_id, record.game_index)
            if key in self._committed:
                continue
            if not self._strength_ok(record):
                # Manifest is the source of truth (§5.5): an over-claimed ``high``
                # tier is rejected loudly, never written to the scoreboard corpus.
                record = replace(
                    record,
                    passed_crosscheck=False,
                    rejection_reason=STRENGTH_MISMATCH_REASON,
                )
            if record.passed_crosscheck:
                _append_line(self.corpus_path, record.to_json_line())
                accepted += 1
            else:
                _append_line(self.rejected_path, record.to_json_line())
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
        # Terminal per-video marker: written last, so a mid-video kill can never
        # leave it behind (game-granular resume — BLOCKER fix). This is the only
        # row :func:`_done_video_ids` treats as "skip this whole video".
        _append_line(
            self.ledger_path,
            LedgerEntry(
                video_id=video_id,
                game_index=None,
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
    """Videos with a terminal video-level ``done`` marker — skip these on resume.

    Resume is **game-granular** (BLOCKER fix): the skip signal is the video-level
    ``done`` row (``game_index is None``) that :meth:`_Sink.commit_video` writes
    LAST, only after every game of the video is on disk. A per-game ``done`` row
    (``game_index`` is an ``int``) does NOT make the video skippable — so a video
    killed after game 1 but before game 2 (no video-level marker) is re-parsed in
    full, and the on-disk dedup drops game 1 while re-emitting game 2. This closes
    the silent-tail-loss hazard (§5.6) where a per-game skip permanently dropped a
    crashed multi-game video's remaining games.

    A stale video-level ``failed`` marker does NOT block resume: only the
    video-level ``done`` marker skips, and a ``failed``-only video (no video-level
    ``done`` row) is retried.
    """
    return {vid for (vid, gidx), entry in ledger.items() if gidx is None and entry.status == "done"}


def _parse_fn_accepts_download_gate(parse_fn: ParseFn) -> bool:
    """Whether ``parse_fn`` declares a ``download_gate`` parameter (arity probe).

    ``run_batch`` passes the download semaphore only to a ``parse_fn`` that opts
    in by naming the keyword — a legacy 1-arg ``video_id -> list[GameRecord]``
    callable is called unchanged. A callable whose signature can't be introspected
    (a C builtin) is treated as legacy.
    """
    try:
        params = inspect.signature(parse_fn).parameters
    except (TypeError, ValueError):
        return False
    if "download_gate" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def run_batch(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    parse_fn: ParseFn,
    max_workers: int = 4,
    net_concurrency: int = 2,
    video_ids: Sequence[str] | None = None,
    now_fn: Callable[[], float] = time.time,
) -> BatchResult:
    """Parse the harvested corpus into ``corpus.jsonl`` + ``rejected.jsonl``.

    Args:
        manifest_path: the committed ``strength_manifest.json`` (source of truth).
        out_dir: destination for ``corpus.jsonl`` / ``rejected.jsonl`` /
            ``ledger.jsonl`` (created if absent). Existing files are appended to
            and resumed.
        parse_fn: injected per-video parser (:data:`ParseFn`),
            ``video_id -> list[GameRecord]``. Raise :class:`VideoParseError` for a
            recoverable failure (marks the video ``failed``, retried on resume);
            any other exception aborts the batch after the already-COMPLETED
            in-flight parses are committed (so finished work is banked, not thrown
            away), and any un-committed video is re-parsed on resume. May also
            accept a ``download_gate`` keyword (see :data:`ParseFn` /
            ``net_concurrency``).
        max_workers: CPU/OCR fan-out width — the thread-pool width. NOT the
            download width: the CV pipeline releases the GIL via ffmpeg / OCR
            subprocesses, so this governs OCR parallelism only.
        net_concurrency: max concurrent yt-dlp downloads (§5.11: the short-lived
            googlevideo URL throttles under parallel pulls). Passed to ``parse_fn``
            as a ``download_gate`` :class:`threading.BoundedSemaphore` when it opts
            in by naming the keyword; a ``parse_fn`` that downloads acquires it
            around ONLY its download phase, decoupling the 1—2-wide network stage
            from the ``max_workers``-wide OCR stage.
        video_ids: optional subset of ids to process (still manifest-gated).
        now_fn: injectable clock for the ledger timestamp (deterministic tests).

    Returns:
        A :class:`BatchResult` summary. Idempotent under resume: a re-run over an
        already-complete corpus processes nothing and duplicates no rows.
    """
    out_dir = Path(out_dir)
    manifest = load_strength_manifest(manifest_path)
    harvested = _harvest_video_ids(manifest, video_ids)
    strength_map: dict[str, str] = {}
    for vid in harvested:
        entry = manifest_entry(manifest, vid)
        strength = entry.get("strength") if entry is not None else None
        if isinstance(strength, str):
            strength_map[vid] = strength

    ledger = load_ledger(out_dir / "ledger.jsonl")
    done = _done_video_ids(ledger)
    pending = [vid for vid in harvested if vid not in done]
    skipped = len(harvested) - len(pending)

    sink = _Sink(out_dir=out_dir, now_fn=now_fn, manifest_strength=strength_map)

    processed = failed = 0
    accepted_total = rejected_total = 0

    # Split the two concurrency domains (§5.11): ``max_workers`` fans OCR out wide;
    # ``download_gate`` throttles the network phase to ``net_concurrency`` inside
    # parse_fn's download call. A legacy 1-arg parse_fn ignores the gate.
    download_gate = threading.BoundedSemaphore(max(1, net_concurrency))
    pass_gate = _parse_fn_accepts_download_gate(parse_fn)

    def _submit(
        pool: concurrent.futures.ThreadPoolExecutor, vid: str
    ) -> concurrent.futures.Future[list[GameRecord]]:
        if pass_gate:
            return pool.submit(parse_fn, vid, download_gate=download_gate)
        return pool.submit(parse_fn, vid)

    # Bound the pool by the actual pending work so a 1-video run stays single-
    # threaded (deterministic). Submit all, then drain in completion order —
    # every write happens on THIS thread through the single sink.
    workers = max(1, min(max_workers, len(pending))) if pending else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_vid = {_submit(pool, vid): vid for vid in pending}
        pending_futures = set(future_to_vid)
        for future in concurrent.futures.as_completed(future_to_vid):
            vid = future_to_vid[future]
            pending_futures.discard(future)
            try:
                records = future.result()
            except VideoParseError as exc:
                sink.mark_failed(vid, str(exc))
                failed += 1
                continue
            except BaseException:
                # An unexpected (non-VideoParseError) abort. Before re-raising,
                # bank every already-COMPLETED in-flight parse so a multi-worker
                # run does not throw away finished videos (each is ~10—20 min of
                # download + OCR). A still-running future is NOT waited on — it is
                # re-parsed on resume and the on-disk dedup drops any of its games
                # already written. Videos with NO video-level ``done`` marker are
                # re-parsed in full on resume.
                for other in pending_futures:
                    if other.done() and not other.cancelled() and other.exception() is None:
                        acc, rej = sink.commit_video(future_to_vid[other], other.result())
                        accepted_total += acc
                        rejected_total += rej
                        processed += 1
                raise
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
