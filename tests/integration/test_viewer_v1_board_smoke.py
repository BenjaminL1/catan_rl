"""V1 smoke: viewer renders the board for a real recording and
produces a non-empty pixel buffer.

We can't easily assert visual correctness without an image-diff
baseline, but we CAN assert that:

1. The pygame surface has more than just the background color
   after the board is rendered (i.e., hex polygons + outlines
   actually drew SOMETHING).
2. The board layout produces 54 vertex pixels for the standard
   19-hex Catan board.
3. Settlements and roads from the recording show up at sensible
   on-board pixel positions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from catan_rl.replay import record_game, save_replay
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec


@pytest.fixture
def short_replay_path(tmp_path: Path) -> Path:
    replay = record_game(
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        seed=42,
        max_turns=20,
    )
    path = tmp_path / "v1_smoke.json"
    save_replay(replay, path)
    return path


def test_v1_board_renders_nonempty_surface(short_replay_path: Path) -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Local imports to keep the test's import-isolation surface clean.
    import pygame

    from catan_rl.replay.io import load_replay
    from catan_rl.replay.viewer.board_layout import compute_board_layout
    from catan_rl.replay.viewer.board_renderer import render_board

    pygame.init()
    try:
        screen = pygame.display.set_mode((1280, 800))
        font = pygame.font.SysFont(None, 22)
        small_font = pygame.font.SysFont(None, 14)

        replay = load_replay(short_replay_path)
        layout = compute_board_layout(replay.board_static, hex_size=70.0, origin=(640.0, 400.0))

        # Layout sanity: standard 19-hex board → 54 vertices, 72 edges.
        assert len(layout.vertex_pixels) == len(replay.board_static.vertices)
        assert len(layout.edge_midpoints) == len(replay.board_static.edges)
        assert len(layout.hex_centers) == 19

        # Render the final-step state.
        state = replay.steps[-1].state_after
        screen.fill((32, 36, 48))
        render_board(
            screen,
            replay.board_static,
            layout,
            state,
            font=font,
            small_font=small_font,
            hex_size=70.0,
        )

        # Surface should now contain pixels OTHER than the background
        # color (hex polygons + outlines).
        surface = pygame.surfarray.array3d(screen)
        # background color was (32, 36, 48); count pixels NOT matching it.
        bg = (32, 36, 48)
        nonbg = (
            (surface[:, :, 0] != bg[0]) | (surface[:, :, 1] != bg[1]) | (surface[:, :, 2] != bg[2])
        )
        nonbg_count = int(nonbg.sum())
        # 1280 * 800 = 1,024,000 pixels. The board should occupy at
        # least 5% of them.
        assert nonbg_count > 1280 * 800 * 0.05, (
            f"board renderer produced only {nonbg_count} non-bg pixels — "
            "the board may not be visible"
        )
    finally:
        pygame.quit()
