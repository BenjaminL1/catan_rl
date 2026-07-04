"""Unit tests for the human-data ingest slice (build brief §5.10–12).

Two concerns, both without network or ffmpeg on the CI box:

- **Sampling-schedule math** (pure) — two-pass cadence, endpoint inclusivity,
  sparse/dense de-dup (dense tag wins), window validation.
- **A mocked-yt-dlp ingest path** — download-then-delete orchestration with
  ``yt-dlp`` and ``ffmpeg`` subprocess calls stubbed, asserting the temp video is
  cleaned up and the decoded frames flow through in schedule order.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from catan_rl.human_data.ingest import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_RESOLUTION,
    OCR_SECONDS_PER_CROP,
    DecodedFrame,
    ScheduledFrame,
    SubResolutionError,
    TimeWindow,
    VideoDownloadError,
    build_sampling_schedule,
    decode_frames_at,
    download_video,
    estimate_ocr_wall_clock_s,
    ingest_video,
    probe_resolution,
    schedule_ocr_eta_s,
)

_FRAME_NBYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


def _frame_nbytes(width: int, height: int) -> int:
    return width * height * 3


def _native_frame_bytes(fill: int, width: int, height: int) -> bytes:
    return bytes([fill]) * _frame_nbytes(width, height)


# --- sampling-schedule math ----------------------------------------------------


def test_sparse_only_cadence_is_endpoint_inclusive() -> None:
    sched = build_sampling_schedule(12.0, sparse_interval_s=4.0)
    assert [s.ts for s in sched] == [0.0, 4.0, 8.0, 12.0]
    assert {s.pass_name for s in sched} == {"sparse"}


def test_sparse_cadence_stops_before_overrunning_duration() -> None:
    # 10s / 4s -> 0,4,8 (12 would overrun), inclusive of neither past-end nor 10.0.
    sched = build_sampling_schedule(10.0, sparse_interval_s=4.0)
    assert [s.ts for s in sched] == [0.0, 4.0, 8.0]


def test_dense_window_adds_fine_samples_inside_window() -> None:
    sched = build_sampling_schedule(
        30.0,
        [TimeWindow(10.0, 13.0)],
        sparse_interval_s=10.0,
        dense_interval_s=1.0,
    )
    dense = sorted(s.ts for s in sched if s.pass_name == "dense")
    assert dense == [10.0, 11.0, 12.0, 13.0]
    # sparse pass still present at its own cadence (0,10,20,30) minus dedup at 10.
    sparse = sorted(s.ts for s in sched if s.pass_name == "sparse")
    assert sparse == [0.0, 20.0, 30.0]


def test_dense_tag_wins_dedup_on_shared_timestamp() -> None:
    # sparse hits 10.0 and dense also hits 10.0 -> emitted once, tagged dense.
    sched = build_sampling_schedule(
        20.0,
        [TimeWindow(10.0, 10.0)],
        sparse_interval_s=10.0,
        dense_interval_s=1.0,
    )
    at_ten = [s for s in sched if s.ts == 10.0]
    assert len(at_ten) == 1
    assert at_ten[0].pass_name == "dense"


def test_schedule_is_sorted_by_timestamp() -> None:
    sched = build_sampling_schedule(
        60.0,
        [TimeWindow(40.0, 42.0), TimeWindow(5.0, 6.0)],
        sparse_interval_s=20.0,
        dense_interval_s=1.0,
    )
    ts = [s.ts for s in sched]
    assert ts == sorted(ts)


def test_no_dense_windows_yields_pure_sparse() -> None:
    sched = build_sampling_schedule(8.0, sparse_interval_s=2.0)
    assert all(s.pass_name == "sparse" for s in sched)
    assert [s.ts for s in sched] == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_zero_duration_rejected() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        build_sampling_schedule(0.0)


def test_nonpositive_interval_rejected() -> None:
    with pytest.raises(ValueError, match="interval_s"):
        build_sampling_schedule(10.0, sparse_interval_s=0.0)


def test_dense_window_past_duration_rejected() -> None:
    with pytest.raises(ValueError, match="runs past duration_s"):
        build_sampling_schedule(10.0, [TimeWindow(8.0, 12.0)])


def test_time_window_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="end_s"):
        TimeWindow(10.0, 5.0)


def test_time_window_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_s"):
        TimeWindow(-1.0, 5.0)


# --- OCR ETA formula (brief §5.10: fps x 0.58s x n_videos / n_procs) -----------


def test_ocr_seconds_per_crop_is_the_spec_constant() -> None:
    # brief §5.10: measured easyocr cost per 1080p crop.
    assert OCR_SECONDS_PER_CROP == 0.58


def test_estimate_ocr_wall_clock_matches_spec_formula() -> None:
    # fps x 0.58s x n_videos / n_procs, verbatim.
    fps, n_videos, n_procs = 0.25, 300.0, 4
    expected = fps * 0.58 * n_videos / n_procs
    assert estimate_ocr_wall_clock_s(fps, n_videos, n_procs=n_procs) == pytest.approx(expected)


def test_estimate_ocr_wall_clock_defaults_to_single_process() -> None:
    assert estimate_ocr_wall_clock_s(1.0, 1.0) == pytest.approx(0.58)


def test_estimate_ocr_wall_clock_scales_inversely_with_procs() -> None:
    one = estimate_ocr_wall_clock_s(0.5, 100.0, n_procs=1)
    eight = estimate_ocr_wall_clock_s(0.5, 100.0, n_procs=8)
    assert eight == pytest.approx(one / 8.0)


def test_estimate_ocr_wall_clock_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="fps"):
        estimate_ocr_wall_clock_s(0.0, 1.0)
    with pytest.raises(ValueError, match="n_videos"):
        estimate_ocr_wall_clock_s(1.0, 0.0)
    with pytest.raises(ValueError, match="n_procs"):
        estimate_ocr_wall_clock_s(1.0, 1.0, n_procs=0)
    with pytest.raises(ValueError, match="seconds_per_crop"):
        estimate_ocr_wall_clock_s(1.0, 1.0, seconds_per_crop=0.0)


def test_schedule_ocr_eta_derives_fps_from_schedule() -> None:
    # 12s video, 4s sparse cadence -> frames at 0,4,8,12 = 4 frames -> fps = 4/12.
    sched = build_sampling_schedule(12.0, sparse_interval_s=4.0)
    assert len(sched) == 4
    eta = schedule_ocr_eta_s(sched, 12.0, n_videos=1.0)
    assert eta == pytest.approx((4 / 12.0) * 0.58)


def test_schedule_ocr_eta_projects_over_corpus_and_procs() -> None:
    sched = build_sampling_schedule(10.0, sparse_interval_s=2.0)  # 0,2,4,6,8,10 = 6 frames
    fps = 6 / 10.0
    eta = schedule_ocr_eta_s(sched, 10.0, n_videos=300.0, n_procs=6)
    assert eta == pytest.approx(fps * 0.58 * 300.0 / 6)


def test_schedule_ocr_eta_rejects_empty_schedule_and_bad_duration() -> None:
    with pytest.raises(ValueError, match="empty schedule"):
        schedule_ocr_eta_s([], 10.0)
    sched = build_sampling_schedule(8.0, sparse_interval_s=4.0)
    with pytest.raises(ValueError, match="duration_s"):
        schedule_ocr_eta_s(sched, 0.0)


# --- mocked download -----------------------------------------------------------


def test_download_video_returns_produced_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # yt-dlp writes to the -o template; emulate a produced 1080p file.
        out_idx = cmd.index("-o") + 1
        template = cmd[out_idx]
        produced = Path(template.replace("%(ext)s", "mp4"))
        produced.write_bytes(b"\x00" * 1024)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = download_video("abc123", tmp_path)
    assert path.exists()
    assert path.name == "abc123.mp4"


def test_download_video_falls_back_across_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        fmt = cmd[cmd.index("-f") + 1]
        calls.append(fmt)
        if fmt != "best[height>=1080]":  # only the last fallback "succeeds"
            raise subprocess.CalledProcessError(1, cmd, stderr="format unavailable")
        template = cmd[cmd.index("-o") + 1]
        Path(template.replace("%(ext)s", "mp4")).write_bytes(b"\x00" * 512)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = download_video("vid", tmp_path)
    assert path.exists()
    assert len(calls) == 3  # tried all three selectors, last one won


def test_download_video_raises_when_all_formats_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="410 gone")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(VideoDownloadError, match="all format fallbacks"):
        download_video("dead", tmp_path)


# --- native-resolution probe + firewall (BLOCKER fix) --------------------------


def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch, width: int, height: int) -> None:
    """Route the ffprobe branch to report ``width``x``height`` and the ffmpeg
    branch to emit a native-geometry raw frame filled from the seek timestamp."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "ffprobe" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{width}x{height}\n", stderr="")
        ts = float(cmd[cmd.index("-ss") + 1])
        payload = _native_frame_bytes(int(ts) % 256, width, height)
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_probe_resolution_parses_width_and_height(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "ffprobe" in cmd[0]
        return subprocess.CompletedProcess(cmd, 0, stdout="1920x1080\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    width, height = probe_resolution(tmp_path / "v.mp4", ffprobe="ffprobe")
    assert (width, height) == (1920, 1080)


def test_decode_probes_native_resolution_and_does_not_upscale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A genuine >=1080p source that is NOT exactly 1920x1080 (e.g. a 1440p 16:9
    # or a wider crop) must be decoded at its TRUE geometry, not stretched.
    _mock_ffprobe(monkeypatch, width=2560, height=1440)
    schedule = [ScheduledFrame(ts=1.0, pass_name="sparse")]
    frames = list(
        decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg", ffprobe="ffprobe")
    )
    assert len(frames) == 1
    # decoded at native geometry, NOT force-scaled to 1080p.
    assert frames[0].frame.shape == (1440, 2560, 3)
    assert frames[0].native_resolution == 1440


def test_decode_rejects_sub_1080p_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The resolution firewall (brief §2 / FIX-5): a 854x480 source OCRs to garbage
    # and must NOT be admitted (previously it was silently upscaled to 1080p).
    _mock_ffprobe(monkeypatch, width=854, height=480)
    schedule = [ScheduledFrame(ts=1.0, pass_name="sparse")]
    with pytest.raises(SubResolutionError, match="480"):
        list(decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg", ffprobe="ffprobe"))


def test_decode_frame_never_stamps_1080_for_a_480p_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression for the red-team counterexample: an 854x480 source must not be
    # decodable-and-stamped as 1080p. Rejection is the firewall; the caller can
    # never read an honest-but-wrong 1080 off a downgraded stream.
    _mock_ffprobe(monkeypatch, width=854, height=480)
    schedule = [ScheduledFrame(ts=1.0, pass_name="sparse")]
    with pytest.raises(SubResolutionError):
        list(decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg", ffprobe="ffprobe"))


def test_min_resolution_is_1080() -> None:
    assert MIN_RESOLUTION == 1080


def test_ingest_min_resolution_agrees_with_orientation_firewall() -> None:
    # The ingest-side reject bar and the record/orientation FIX-5 gate must never
    # drift apart, or ingest could admit a source the record layer then rejects.
    from catan_rl.human_data.orientation import MIN_RESOLUTION as ORIENTATION_MIN

    assert MIN_RESOLUTION == ORIENTATION_MIN


# --- mocked decode + full ingest orchestration ---------------------------------


def _fake_frame_bytes(fill: int) -> bytes:
    return bytes([fill]) * _FRAME_NBYTES


def test_decode_frames_at_yields_frames_in_schedule_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = [
        ScheduledFrame(ts=1.0, pass_name="sparse"),
        ScheduledFrame(ts=2.0, pass_name="dense"),
    ]
    _mock_ffprobe(monkeypatch, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    frames = list(
        decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg", ffprobe="ffprobe")
    )
    assert [f.ts for f in frames] == [1.0, 2.0]
    assert [f.pass_name for f in frames] == ["sparse", "dense"]
    assert frames[0].frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert frames[0].native_resolution == FRAME_HEIGHT
    assert int(frames[0].frame[0, 0, 0]) == 1


def test_decode_skips_partial_frame_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = [
        ScheduledFrame(ts=1.0, pass_name="sparse"),
        ScheduledFrame(ts=2.0, pass_name="sparse"),
    ]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "ffprobe" in cmd[0]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{FRAME_WIDTH}x{FRAME_HEIGHT}\n", stderr=""
            )
        ts = float(cmd[cmd.index("-ss") + 1])
        # ts=2.0 returns a truncated buffer (seek past EOF) -> skipped.
        payload = _fake_frame_bytes(7) if ts == 1.0 else b"\x00" * 10
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    frames = list(
        decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg", ffprobe="ffprobe")
    )
    assert [f.ts for f in frames] == [1.0]


def test_decode_empty_schedule_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty schedule"):
        list(decode_frames_at(tmp_path / "v.mp4", [], ffmpeg="ffmpeg", ffprobe="ffprobe"))


def test_ingest_video_downloads_decodes_then_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Path] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if cmd[0] == "yt-dlp":
            template = cmd[cmd.index("-o") + 1]
            produced = Path(template.replace("%(ext)s", "mp4"))
            produced.write_bytes(b"\x00" * 2048)
            created.append(produced)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "ffprobe" in cmd[0]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{FRAME_WIDTH}x{FRAME_HEIGHT}\n", stderr=""
            )
        # ffmpeg decode branch
        ts = float(cmd[cmd.index("-ss") + 1])
        payload = _fake_frame_bytes(int(ts) % 256)
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    frames = list(
        ingest_video(
            "vid42",
            duration_s=8.0,
            sparse_interval_s=4.0,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
        )
    )
    # schedule = 0,4,8 -> 3 decoded frames
    assert [f.ts for f in frames] == [0.0, 4.0, 8.0]
    assert all(isinstance(f, DecodedFrame) for f in frames)
    # download-then-delete: the temp video and its temp dir are gone.
    assert created, "yt-dlp mock was never invoked"
    for produced in created:
        assert not produced.exists(), "downloaded video was not deleted"
        assert not produced.parent.exists(), "temp work dir was not cleaned up"


def test_ingest_video_cleans_up_even_if_consumer_stops_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Path] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if cmd[0] == "yt-dlp":
            template = cmd[cmd.index("-o") + 1]
            produced = Path(template.replace("%(ext)s", "mp4"))
            produced.write_bytes(b"\x00" * 2048)
            created.append(produced)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "ffprobe" in cmd[0]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{FRAME_WIDTH}x{FRAME_HEIGHT}\n", stderr=""
            )
        ts = float(cmd[cmd.index("-ss") + 1])
        payload = _fake_frame_bytes(int(ts) % 256)
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    gen = ingest_video(
        "vidEarly", duration_s=40.0, sparse_interval_s=4.0, ffmpeg="ffmpeg", ffprobe="ffprobe"
    )
    first = next(gen)  # consume one frame then abandon the generator
    assert first.ts == 0.0
    gen.close()  # triggers the finally-block cleanup

    assert created
    for produced in created:
        assert not produced.exists()
        assert not produced.parent.exists()
