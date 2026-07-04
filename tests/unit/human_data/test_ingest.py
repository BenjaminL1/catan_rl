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
    DecodedFrame,
    ScheduledFrame,
    TimeWindow,
    VideoDownloadError,
    build_sampling_schedule,
    decode_frames_at,
    download_video,
    ingest_video,
)

_FRAME_NBYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


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
        if fmt != "best[height<=1080]":  # only the last fallback "succeeds"
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

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ts = float(cmd[cmd.index("-ss") + 1])
        return subprocess.CompletedProcess(cmd, 0, stdout=_fake_frame_bytes(int(ts)), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    frames = list(decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg"))
    assert [f.ts for f in frames] == [1.0, 2.0]
    assert [f.pass_name for f in frames] == ["sparse", "dense"]
    assert frames[0].frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert int(frames[0].frame[0, 0, 0]) == 1


def test_decode_skips_partial_frame_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = [
        ScheduledFrame(ts=1.0, pass_name="sparse"),
        ScheduledFrame(ts=2.0, pass_name="sparse"),
    ]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ts = float(cmd[cmd.index("-ss") + 1])
        # ts=2.0 returns a truncated buffer (seek past EOF) -> skipped.
        payload = _fake_frame_bytes(7) if ts == 1.0 else b"\x00" * 10
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    frames = list(decode_frames_at(tmp_path / "v.mp4", schedule, ffmpeg="ffmpeg"))
    assert [f.ts for f in frames] == [1.0]


def test_decode_empty_schedule_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty schedule"):
        list(decode_frames_at(tmp_path / "v.mp4", [], ffmpeg="ffmpeg"))


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
        ts = float(cmd[cmd.index("-ss") + 1])
        payload = _fake_frame_bytes(int(ts) % 256)
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    gen = ingest_video("vidEarly", duration_s=40.0, sparse_interval_s=4.0, ffmpeg="ffmpeg")
    first = next(gen)  # consume one frame then abandon the generator
    assert first.ts == 0.0
    gen.close()  # triggers the finally-block cleanup

    assert created
    for produced in created:
        assert not produced.exists()
        assert not produced.parent.exists()
