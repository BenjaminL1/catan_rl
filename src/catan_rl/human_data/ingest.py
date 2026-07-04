"""Stage-1 ingest: yt-dlp download-then-delete + two-pass in-memory frame sampling.

This module turns one ThePhantom YouTube video id into a stream of decoded 1080p
frames, without ever accumulating per-frame PNGs on disk (build brief §5.12) and
without keeping the downloaded video around after the frames are read (brief
§5.11). It has two independent, separately-testable pieces:

1. **Sampling-schedule math** (pure, no I/O) — :func:`build_sampling_schedule`.
   The corpus is ~10 CPU-days at a naive 1 fps (brief §5.10), so we sample in
   **two passes**: a *sparse* pass (one frame every ``sparse_interval_s`` seconds,
   3—5 s) that carries the log OCR + game boundaries + the setup-event timestamps,
   and a *dense* pass (one frame every ``dense_interval_s`` seconds) run **only**
   inside the bounded setup windows the sparse pass discovers (and around the
   winner line). The dense windows are supplied by the caller (Stage-1 log parse
   feeds them back), so this module owns only the arithmetic.

2. **Acquisition** — :func:`download_video` (yt-dlp 1080p video-only DASH,
   download-then-delete, retries / sleep-interval / format-fallback, network
   concurrency capped to 1—2 by the batch driver) and :func:`decode_frames_at`
   (ffmpeg → raw RGB24 over stdout → numpy, one frame per requested timestamp).
   :func:`ingest_video` composes them: download → decode the scheduled frames →
   delete. Reuses the ffmpeg / yt-dlp patterns proven in
   ``scripts/build_strength_manifest.py``.

CPU-only, no ``gui/`` or training-path imports (brief §6). ``node`` satisfies
yt-dlp's nsig JS runtime (brief §5.11); ffmpeg is resolved via
:func:`catan_rl.human_data.ffmpeg.resolve_ffmpeg`.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from catan_rl.human_data.ffmpeg import resolve_ffmpeg

#: 1080p frame geometry (brief §2: 360—480p OCR is garbage, must pull 1080p).
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
#: Bytes per decoded RGB24 frame (3 channels x 8-bit).
_FRAME_NBYTES = FRAME_WIDTH * FRAME_HEIGHT * 3

#: Default sparse-pass cadence — one frame every 4 s (brief §5.10: "1 frame /
#: 3—5 s"). The dense pass runs an order of magnitude finer inside the setup
#: window so the 8 snake-draft placement events are not missed.
DEFAULT_SPARSE_INTERVAL_S = 4.0
DEFAULT_DENSE_INTERVAL_S = 1.0

PassName = Literal["sparse", "dense"]


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A closed ``[start_s, end_s]`` second-interval of a video (inclusive)."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0.0:
            raise ValueError(f"window start_s {self.start_s} must be >= 0")
        if self.end_s < self.start_s:
            raise ValueError(f"window end_s {self.end_s} must be >= start_s {self.start_s}")


@dataclass(frozen=True, slots=True)
class ScheduledFrame:
    """One scheduled sample: a timestamp tagged with which pass requested it."""

    ts: float
    pass_name: PassName


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """A decoded 1080p frame with its source timestamp and pass tag."""

    ts: float
    pass_name: PassName
    #: ``(FRAME_HEIGHT, FRAME_WIDTH, 3)`` ``uint8`` RGB array (in-memory, brief §5.12).
    frame: np.ndarray


def _cadence(start_s: float, end_s: float, interval_s: float) -> list[float]:
    """Timestamps at ``interval_s`` cadence over the closed ``[start_s, end_s]``.

    Endpoint-inclusive, deterministic (no float drift): computed as
    ``start_s + k * interval_s`` for ``k = 0, 1, …`` while ``<= end_s`` (plus a
    tiny epsilon so an exact endpoint is not dropped by rounding).
    """
    if interval_s <= 0.0:
        raise ValueError(f"interval_s {interval_s} must be > 0")
    if end_s < start_s:
        return []
    span = end_s - start_s
    n = int(span / interval_s + 1e-9)
    return [start_s + k * interval_s for k in range(n + 1)]


def build_sampling_schedule(
    duration_s: float,
    dense_windows: Sequence[TimeWindow] = (),
    *,
    sparse_interval_s: float = DEFAULT_SPARSE_INTERVAL_S,
    dense_interval_s: float = DEFAULT_DENSE_INTERVAL_S,
) -> list[ScheduledFrame]:
    """Compute the deterministic two-pass sampling schedule for one video.

    Pass A (sparse): one frame every ``sparse_interval_s`` over the whole video,
    from ``t=0`` to ``t=duration_s``. Pass B (dense): one frame every
    ``dense_interval_s`` inside each supplied ``dense_window`` (the bounded setup
    windows + winner-line region the sparse pass discovered).

    De-duplication: a timestamp that falls on both a sparse and a dense sample is
    emitted **once**, tagged ``"dense"`` (the finer pass wins — the frame is
    decoded a single time). The returned list is sorted by timestamp so a single
    forward ffmpeg pass can honour it. Pure arithmetic — no I/O, unit-tested.

    Raises :class:`ValueError` on a non-positive duration/interval or a window
    that runs past ``duration_s``.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s {duration_s} must be > 0")

    # dense first so its tag wins the de-dup (a shared ts becomes "dense").
    tagged: dict[float, PassName] = {}
    for window in dense_windows:
        if window.end_s > duration_s:
            raise ValueError(f"dense window end_s {window.end_s} runs past duration_s {duration_s}")
        for ts in _cadence(window.start_s, window.end_s, dense_interval_s):
            tagged[round(ts, 6)] = "dense"
    for ts in _cadence(0.0, duration_s, sparse_interval_s):
        tagged.setdefault(round(ts, 6), "sparse")

    return [ScheduledFrame(ts=ts, pass_name=tag) for ts, tag in sorted(tagged.items())]


class VideoDownloadError(RuntimeError):
    """Raised when yt-dlp cannot download a video after its retries."""


#: yt-dlp format selectors, tried in order (brief §5.11): a 1080p video-only DASH
#: stream first (~0.22 GB, no audio needed for CV/OCR), falling back to the best
#: <=1080p muxed stream if the split streams are unavailable for a video.
_FORMAT_FALLBACK: tuple[str, ...] = (
    "bestvideo[height<=1080][ext=mp4]",
    "bestvideo[height<=1080]",
    "best[height<=1080]",
)


def download_video(
    video_id: str,
    dest_dir: Path,
    *,
    retries: int = 3,
    sleep_interval_s: float = 2.0,
    timeout_s: float = 900.0,
) -> Path:
    """Download a video's 1080p video-only DASH stream into ``dest_dir``, return its path.

    Download-then-delete is the caller's job (:func:`ingest_video` /
    ``batch.py``); this only fetches (resumable via yt-dlp's ``--continue``).
    Tries each selector in :data:`_FORMAT_FALLBACK` until one yields a file, with
    yt-dlp's own ``--retries`` / ``--sleep-interval`` for transient network
    errors. ``node`` (present) satisfies yt-dlp's nsig JS runtime; the batch
    driver caps concurrent downloads to 1—2 (brief §5.11).

    Raises :class:`VideoDownloadError` if every format fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = str(dest_dir / f"{video_id}.%(ext)s")

    last_err = ""
    for fmt in _FORMAT_FALLBACK:
        cmd = [
            "yt-dlp",
            "-f",
            fmt,
            "--no-playlist",
            "--continue",
            "--no-part",
            "--retries",
            str(retries),
            "--sleep-interval",
            str(sleep_interval_s),
            "-o",
            out_template,
            url,
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            last_err = (exc.stderr or "").strip()[-500:]
            continue
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_s}s"
            continue

        produced = sorted(dest_dir.glob(f"{video_id}.*"))
        produced = [p for p in produced if p.stat().st_size > 0]
        if produced:
            return produced[0]
        last_err = "yt-dlp reported success but produced no file"

    raise VideoDownloadError(
        f"yt-dlp failed to download {video_id} after all format fallbacks: {last_err}"
    )


def decode_frames_at(
    video_path: Path,
    schedule: Sequence[ScheduledFrame],
    *,
    ffmpeg: str | None = None,
    timeout_s: float = 120.0,
) -> Iterator[DecodedFrame]:
    """Decode exactly the scheduled frames from ``video_path``, one per timestamp.

    Each frame is grabbed by a seek-then-single-frame ffmpeg call (``-ss <ts> -i
    <file> -frames:v 1``) piping **raw RGB24 to stdout** (brief §5.12 — never a
    PNG on disk), scaled to the fixed 1080p geometry, and reshaped to a
    ``(H, W, 3)`` ``uint8`` array. Frames are yielded lazily in schedule order.

    Skips (does not raise on) a timestamp ffmpeg cannot decode — e.g. a seek past
    a truncated download — so one bad frame never aborts a whole video; the caller
    tolerates missing samples. Raises only if ``schedule`` requests nothing.
    """
    if not schedule:
        raise ValueError("empty schedule — nothing to decode")
    ffmpeg_bin = ffmpeg if ffmpeg is not None else resolve_ffmpeg()

    for scheduled in schedule:
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{scheduled.ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=timeout_s,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

        raw = proc.stdout
        if len(raw) != _FRAME_NBYTES:
            # partial/empty decode (seek past EOF, corrupt GOP) — skip this sample.
            continue
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)
        yield DecodedFrame(ts=scheduled.ts, pass_name=scheduled.pass_name, frame=frame)


def ingest_video(
    video_id: str,
    duration_s: float,
    dense_windows: Sequence[TimeWindow] = (),
    *,
    sparse_interval_s: float = DEFAULT_SPARSE_INTERVAL_S,
    dense_interval_s: float = DEFAULT_DENSE_INTERVAL_S,
    ffmpeg: str | None = None,
    work_dir: Path | None = None,
) -> Iterator[DecodedFrame]:
    """Ingest one video end-to-end: download → decode scheduled frames → delete.

    Builds the two-pass schedule (:func:`build_sampling_schedule`), downloads the
    1080p stream into a temp dir (:func:`download_video`), yields the decoded
    frames in schedule order (:func:`decode_frames_at`), then **deletes the
    downloaded video** — no video and no PNGs are left on disk (brief §5.11—12).

    The download is deleted even if the consumer stops early or raises (the temp
    dir is cleaned in a ``finally``). ``work_dir`` overrides the temp-dir parent
    (e.g. a fast scratch volume); it is never populated with per-frame files.
    """
    schedule = build_sampling_schedule(
        duration_s,
        dense_windows,
        sparse_interval_s=sparse_interval_s,
        dense_interval_s=dense_interval_s,
    )
    ffmpeg_bin = ffmpeg if ffmpeg is not None else resolve_ffmpeg()

    tmp = tempfile.mkdtemp(prefix=f"phantom_{video_id}_", dir=str(work_dir) if work_dir else None)
    tmp_path = Path(tmp)
    try:
        video_path = download_video(video_id, tmp_path)
        yield from decode_frames_at(video_path, schedule, ffmpeg=ffmpeg_bin)
    finally:
        for child in tmp_path.glob("*"):
            child.unlink(missing_ok=True)
        tmp_path.rmdir()
