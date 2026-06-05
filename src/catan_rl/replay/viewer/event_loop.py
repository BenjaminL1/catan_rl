"""V0 — pygame skeleton + replay load.

Owns the pygame event loop, window setup, and a metadata-header
placeholder render. Future phases plug board (V1), player panels
(V2), and step-bar / event-log (V3) into the ``_render_frame`` hook.

The viewer reads a replay file via :func:`catan_rl.replay.load_replay`
(``strict=False`` so old-version replays are tolerated) and shuts
down cleanly on ``ESC`` or the window-close button.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Suppress pygame's "Hello from the pygame community" banner at
# import time. Set BEFORE the ``import pygame`` below; environment
# is read once at import. The banner pollutes scripted-use stdout
# (CI logs, future thumbnail-batch jobs) without any user benefit.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

# Import DIRECTLY from io + schema rather than the top-level
# ``catan_rl.replay`` package, because the package's ``__init__.py``
# re-exports the recorder symbols which transitively pull in the
# engine (broadcast events, snapshot accessor, etc.). The viewer's
# dependency-isolation contract (no engine, no env, no torch) is
# asserted in ``tests/unit/replay/test_viewer_import_isolation.py``;
# routing through ``catan_rl.replay`` would silently fail that gate.
from catan_rl.replay.io import load_replay
from catan_rl.replay.schema import Replay, ReplaySchemaError

_LOG = logging.getLogger("catan_rl.replay.viewer")

# Default window dimensions. The recorder's ``intended_hex_size``
# metadata field is the *recommended* canvas the JSON was rendered
# against; the viewer may scale to any size since pixel coords are
# computed via :func:`catan_rl.replay.hex_math.axial_to_pixel`.
DEFAULT_WINDOW_W = 1280
DEFAULT_WINDOW_H = 800

#: Background color (R, G, B) for the empty canvas — desaturated dark
#: blue/grey, matches the v1 GUI aesthetic so screenshots look
#: cohesive in plan docs.
_BG_COLOR = (32, 36, 48)
_FG_COLOR = (220, 224, 232)
_DIM_COLOR = (160, 168, 184)


def run_viewer(
    replay_path: Path,
    *,
    window_size: tuple[int, int] = (DEFAULT_WINDOW_W, DEFAULT_WINDOW_H),
    headless_frames: int | None = None,
) -> int:
    """Launch the pygame viewer for ``replay_path``.

    Args:
        replay_path: path to the JSON replay produced by
            :func:`catan_rl.replay.record_game`.
        window_size: ``(W, H)`` in pixels. Defaults to 1280x800.
        headless_frames: if not ``None``, render exactly that many
            frames then exit cleanly. Used by smoke tests under
            ``SDL_VIDEODRIVER=dummy`` so the pygame loop doesn't
            block waiting for user input.

    Returns the process exit code (``0`` on clean shutdown).
    """
    replay = load_replay(replay_path, strict=False)
    _LOG.info(
        "loaded replay: %s — %d steps, winner=%s, final_vp=%s",
        replay_path,
        replay.metadata.total_steps,
        replay.metadata.winner,
        replay.metadata.final_vp,
    )

    # Heavy pygame import is local so the schema/IO surface stays
    # importable without pygame on the path (tests, recorder).
    import pygame

    pygame.init()
    try:
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(f"Catan Replay — {replay_path.name}")
        font = pygame.font.SysFont(None, 22)
        clock = pygame.time.Clock()

        frame_idx = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            _render_frame(screen, font, replay)
            pygame.display.flip()
            clock.tick(30)  # cap at 30 FPS — the replay is static

            frame_idx += 1
            if headless_frames is not None and frame_idx >= headless_frames:
                running = False
        return 0
    finally:
        pygame.quit()


def _render_frame(screen: Any, font: Any, replay: Replay) -> None:
    """Single-frame draw. V0 just renders a metadata header — future
    versions composite the board (V1), player panels (V2), and
    step-bar (V3) into this hook."""
    screen.fill(_BG_COLOR)

    md = replay.metadata
    title = (
        f"player_a={md.player_a.kind}  vs  player_b={md.player_b.kind}    "
        f"seed={md.seed}  max_turns={md.max_turns}"
    )
    title_surf = font.render(title, True, _FG_COLOR)
    screen.blit(title_surf, (16, 12))

    summary = (
        f"steps={md.total_steps}    "
        f"winner={md.winner or '—'}    "
        f"winner_seat={md.winner_seat if md.winner_seat is not None else '—'}    "
        f"final_vp={md.final_vp}"
    )
    summary_surf = font.render(summary, True, _DIM_COLOR)
    screen.blit(summary_surf, (16, 38))

    # Placeholder for future panels — V0 is intentionally minimal.
    placeholder = font.render(
        "(V0 skeleton — board + panels + step-bar land in V1-V3)",
        True,
        _DIM_COLOR,
    )
    placeholder_rect = placeholder.get_rect(
        center=(screen.get_width() // 2, screen.get_height() // 2)
    )
    screen.blit(placeholder, placeholder_rect)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python scripts/replay_viewer.py <path>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="replay_viewer.py",
        description="Pygame viewer for 1v1 Catan replay JSON.",
    )
    parser.add_argument(
        "replay",
        type=Path,
        help="Path to the JSON replay (output of scripts/record_game.py).",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default=f"{DEFAULT_WINDOW_W}x{DEFAULT_WINDOW_H}",
        help="WxH pixels, e.g. 1600x900. Default matches Metadata.intended_hex_size.",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=None,
        help="(Test-only) Render exactly N frames then exit. Pairs with SDL_VIDEODRIVER=dummy.",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.replay.exists():
        print(f"error: replay file not found: {args.replay}", file=sys.stderr)
        return 2

    try:
        w_str, h_str = args.window_size.lower().split("x", 1)
        window_size = (int(w_str), int(h_str))
    except (ValueError, AttributeError) as exc:
        print(f"error: invalid --window-size {args.window_size!r}: {exc}", file=sys.stderr)
        return 2
    # Range guard: pygame's display.set_mode crashes on non-positive
    # or absurdly large window sizes. 10000 is well above any
    # reasonable monitor; below 1 is meaningless.
    w_, h_ = window_size
    if not (1 <= w_ <= 10000 and 1 <= h_ <= 10000):
        print(
            f"error: --window-size out of range: {w_}x{h_} (must be 1-10000 each)",
            file=sys.stderr,
        )
        return 2

    try:
        return run_viewer(
            args.replay,
            window_size=window_size,
            headless_frames=args.headless_frames,
        )
    except ReplaySchemaError as exc:
        # Surface the schema error as a one-line CLI message rather
        # than a stack trace. Common cases: corrupt JSON, missing
        # ``schema_version``, unknown event kind under strict mode.
        print(f"error: replay file is malformed: {exc}", file=sys.stderr)
        return 2
