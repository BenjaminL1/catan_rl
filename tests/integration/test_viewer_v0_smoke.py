"""V0 smoke test: viewer launches under SDL_VIDEODRIVER=dummy and
exits cleanly after N headless frames."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from catan_rl.replay import record_game
from catan_rl.replay.io import save_replay
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec


@pytest.fixture
def tiny_replay(tmp_path: Path) -> Path:
    """A short random vs random recording for smoke-testing the viewer."""
    replay = record_game(
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        seed=42,
        max_turns=20,
    )
    path = tmp_path / "tiny.json"
    save_replay(replay, path)
    return path


def test_viewer_runs_n_frames_and_exits(tiny_replay: Path) -> None:
    # Force dummy SDL driver so this test works in CI / headless.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Local import — viewer initialises pygame on import.
    from catan_rl.replay.viewer import run_viewer

    rc = run_viewer(tiny_replay, window_size=(640, 400), headless_frames=3)
    assert rc == 0
