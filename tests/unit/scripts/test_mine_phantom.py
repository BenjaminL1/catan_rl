"""Unit tests for scripts/mine_phantom.py — the human-data pipeline CLI.

Pins the ``ingest`` subcommand's OCR-ETA wiring (build brief §5.10: the ETA is
reported as ``fps x 0.58s x n_videos / n_procs``, with ``fps`` derived from the
two-pass schedule). yt-dlp / ffmpeg subprocess calls are stubbed so the test runs
with no network or ffmpeg on the CI box.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from catan_rl.human_data import ingest as ingest_mod
from catan_rl.human_data.ingest import FRAME_HEIGHT, FRAME_WIDTH

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mine_phantom.py"
_spec = importlib.util.spec_from_file_location("mine_phantom_module", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mine = importlib.util.module_from_spec(_spec)
sys.modules["mine_phantom_module"] = mine
_spec.loader.exec_module(mine)

_FRAME_NBYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    if cmd[0] == "yt-dlp":
        template = cmd[cmd.index("-o") + 1]
        Path(template.replace("%(ext)s", "mp4")).write_bytes(b"\x00" * 2048)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if cmd[0] == "ffprobe":
        # Resolution firewall probe: report 1080p so the decoded frame payload
        # (FRAME_WIDTH*FRAME_HEIGHT below) matches the probed geometry.
        geometry = f"{FRAME_WIDTH}x{FRAME_HEIGHT}\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=geometry, stderr="")
    # ffmpeg decode branch
    ts = float(cmd[cmd.index("-ss") + 1])
    payload = bytes([int(ts) % 256]) * _FRAME_NBYTES
    return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr=b"")


def test_ingest_reports_ocr_eta_formula(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run)
    # decode is subprocess-mocked; stub the binary resolvers so no real ffmpeg /
    # ffprobe is needed (the resolution firewall calls resolve_ffprobe()).
    monkeypatch.setattr(ingest_mod, "resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ingest_mod, "resolve_ffprobe", lambda: "ffprobe")

    rc = mine.main(["ingest", "vid42", "--duration", "12", "--sparse-interval", "4"])
    out = capsys.readouterr().out

    assert rc == 0
    # 12s / 4s sparse -> frames at 0,4,8,12 = 4 frames -> fps = 4/12 (context only).
    # WALL-CLOCK ETA keys off TOTAL crops: 4 x 0.58s = 2.32s -> "= 2s".
    assert "OCR ETA" in out
    assert "4 frames/video" in out
    assert "fps=0.3333" in out
    assert "0.58s" in out
    assert "= 2s" in out


def test_ingest_eta_scales_with_n_videos_and_procs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(ingest_mod, "resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ingest_mod, "resolve_ffprobe", lambda: "ffprobe")
    rc = mine.main(
        [
            "ingest",
            "vid42",
            "--duration",
            "12",
            "--sparse-interval",
            "4",
            "--n-videos",
            "300",
            "--n-procs",
            "6",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # 4 crops/video, 300 videos / 6 procs = 4 x 0.58s x 300 / 6 = 116s WALL-CLOCK
    # (NOT the old (4/12)*0.58*300/6 = 9.67s — that dropped the duration factor).
    assert "n_videos=300" in out
    assert "n_procs=6" in out
    assert "= 116s" in out


def test_batch_plan_reports_manifest_harvest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "strength_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "videos": [
                    {"video_id": "vidHIGH00001", "strength": "high", "source": "tournament"},
                    {"video_id": "vidUNKNOWN00", "strength": "unknown", "source": "none"},
                    {"video_id": "vidEXCLUDED0", "strength": "excluded", "source": "none"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Point the CLI at the temp manifest (never the committed one under test).
    monkeypatch.setattr(mine, "MANIFEST", manifest)

    rc = mine.main(["batch-plan"])
    out = capsys.readouterr().out
    assert rc == 0
    # 2 harvested (high+unknown), 1 scoreboard-eligible (high), 1 seed-only, 1 excluded.
    assert "2 videos (high+unknown)" in out
    assert "1 scoreboard-eligible (high)" in out
    assert "1 seed-only (unknown)" in out
    assert "1 excluded" in out
