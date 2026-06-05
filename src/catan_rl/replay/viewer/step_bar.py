"""V3 — step-bar + event-log renderer + interaction state.

The step bar is a horizontal timeline at the bottom of the window
that shows every step in the replay. Setup steps are colored grey,
main steps colored by acting actor, and the terminal step gets a
distinct gold tint. The current step is highlighted with an outline.

The event log is a right-side scrolling list of the most-recent N
:class:`ReplayStep.log_lines` entries.

This module owns ``ViewerControl`` — the state machine that responds
to key + mouse events and produces the next ``step_idx``. The pygame
event loop calls :func:`handle_event` per input event then
:meth:`tick` per frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import pygame

from catan_rl.replay.schema import Replay

# Reused actor colors for step-bar segments. Inlined rather than
# imported from board_renderer to keep this module standalone (per
# the V2 pattern).
_ACTOR_TINTS: dict[str, tuple[int, int, int]] = {
    "player_a": (228, 96, 96),
    "player_b": (96, 168, 232),
}
_SETUP_TINT = (140, 148, 168)
_TERMINAL_TINT = (240, 196, 96)
_BAR_BG = (28, 32, 44)
_BAR_BORDER = (88, 100, 124)
_CURRENT_OUTLINE = (236, 238, 244)
_LOG_BG = (24, 28, 36)
_LOG_FG = (220, 224, 232)
_LOG_DIM = (148, 156, 172)


@dataclass(frozen=True, slots=True)
class StepBarGeometry:
    """Pixel rectangles for the bottom step bar + the right-side
    event-log panel."""

    bar_rect: pygame.Rect
    log_rect: pygame.Rect


def compute_step_bar_geometry(
    window_size: tuple[int, int],
    *,
    panel_right_edge: int,
    bar_height: int = 36,
    log_width: int = 240,
) -> StepBarGeometry:
    """Place the step bar across the bottom of the window (above the
    bottom margin) and tuck the log panel into the right side
    between the existing right player-panel and the bar."""
    w, h = window_size
    bar_rect = pygame.Rect(0, h - bar_height, w, bar_height)
    # Log panel sits ABOVE the bar, to the right of the centered
    # board area, sharing the right player-panel's column.
    log_rect = pygame.Rect(
        panel_right_edge - log_width,
        64 + 280,  # below the right panel's resource/dev block (~280px)
        log_width,
        h - bar_height - 64 - 280 - 8,
    )
    return StepBarGeometry(bar_rect=bar_rect, log_rect=log_rect)


@dataclass(slots=True)
class ViewerControl:
    """User-facing scrubbing state. ``step_idx`` is the currently
    visible step; ``playing`` advances it on each tick."""

    total_steps: int
    step_idx: int = 0
    playing: bool = False
    play_speed_frames: int = 12  # frames between auto-advances
    _frames_since_tick: int = field(default=0, init=False)

    def step_to(self, idx: int) -> None:
        self.step_idx = max(0, min(self.total_steps - 1, idx))

    def step_by(self, delta: int) -> None:
        self.step_to(self.step_idx + delta)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self._frames_since_tick = 0

    def tick(self) -> None:
        if not self.playing:
            return
        self._frames_since_tick += 1
        if self._frames_since_tick >= self.play_speed_frames:
            self._frames_since_tick = 0
            if self.step_idx >= self.total_steps - 1:
                self.playing = False  # stop at end
                return
            self.step_idx += 1


def handle_event(control: ViewerControl, event: Any, bar_rect: pygame.Rect) -> bool:
    """Update ``control`` based on a single pygame event. Returns
    ``True`` if the event was consumed (so the caller can skip
    further handlers)."""
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            control.step_by(-1)
            return True
        if event.key == pygame.K_RIGHT:
            control.step_by(1)
            return True
        if event.key == pygame.K_PAGEUP:
            control.step_by(-10)
            return True
        if event.key == pygame.K_PAGEDOWN:
            control.step_by(10)
            return True
        if event.key == pygame.K_HOME:
            control.step_to(0)
            return True
        if event.key == pygame.K_END:
            control.step_to(control.total_steps - 1)
            return True
        if event.key == pygame.K_SPACE:
            control.toggle_play()
            return True
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        # Click on the step bar → scrub to that position. We use
        # ``total_steps`` (not ``total_steps - 1``) as the scaling
        # factor so a click near the far edge reliably reaches the
        # last step after ``step_to``'s clamp; otherwise int
        # truncation leaves the last 1/N range unreachable.
        if bar_rect.collidepoint(event.pos):
            frac = (event.pos[0] - bar_rect.left) / max(1, bar_rect.width)
            target = int(frac * control.total_steps)
            control.step_to(target)
            return True
    return False


def render_step_bar(
    screen: Any,
    replay: Replay,
    control: ViewerControl,
    geom: StepBarGeometry,
    *,
    font: Any,
) -> None:
    """Draw the horizontal step bar at the bottom of the window."""
    pygame.draw.rect(screen, _BAR_BG, geom.bar_rect)
    pygame.draw.rect(screen, _BAR_BORDER, geom.bar_rect, width=1)

    if control.total_steps <= 0:
        return

    seg_w = max(1.0, geom.bar_rect.width / control.total_steps)
    for i, step in enumerate(replay.steps):
        x = geom.bar_rect.left + int(i * seg_w)
        next_x = geom.bar_rect.left + int((i + 1) * seg_w)
        width = max(1, next_x - x)
        if step.kind == "setup":
            color = _SETUP_TINT
        elif step.kind == "terminal":
            color = _TERMINAL_TINT
        else:
            color = _ACTOR_TINTS.get(step.actor, (160, 160, 160))
        pygame.draw.rect(
            screen, color, pygame.Rect(x, geom.bar_rect.top + 4, width, geom.bar_rect.height - 8)
        )

    # Current-step outline.
    cur_x = geom.bar_rect.left + int(control.step_idx * seg_w)
    cur_w = max(2, int(seg_w))
    pygame.draw.rect(
        screen,
        _CURRENT_OUTLINE,
        pygame.Rect(cur_x - 1, geom.bar_rect.top + 2, cur_w + 2, geom.bar_rect.height - 4),
        width=2,
    )

    # Position readout (e.g. "step 42 / 513 — main — player_b").
    cur = replay.steps[control.step_idx]
    play_glyph = "▶" if control.playing else ""
    readout = (
        f"step {control.step_idx} / {control.total_steps - 1}    "
        f"{cur.kind}    {cur.actor}    {play_glyph}"
    )
    surf = font.render(readout, True, _CURRENT_OUTLINE)
    screen.blit(surf, (geom.bar_rect.left + 8, geom.bar_rect.top + 8))


def render_event_log(
    screen: Any,
    replay: Replay,
    step_idx: int,
    geom: StepBarGeometry,
    *,
    font: Any,
    small_font: Any,
    max_lines: int = 18,
) -> None:
    """Draw the right-side event log showing the most-recent
    ``max_lines`` log lines, ending at the current step."""
    pygame.draw.rect(screen, _LOG_BG, geom.log_rect)
    pygame.draw.rect(screen, _BAR_BORDER, geom.log_rect, width=1)

    header_surf = font.render("Event log", True, _LOG_DIM)
    screen.blit(header_surf, (geom.log_rect.left + 8, geom.log_rect.top + 6))

    # Walk backwards from ``step_idx`` collecting log lines until we
    # have ``max_lines`` or run out of steps.
    collected: list[tuple[str, bool]] = []  # (line, is_current_step)
    for s_idx in range(step_idx, -1, -1):
        if len(collected) >= max_lines:
            break
        step = replay.steps[s_idx]
        for line in reversed(step.log_lines):
            collected.append((line, s_idx == step_idx))
            if len(collected) >= max_lines:
                break

    # Render newest at top; collected was filled in newest-first order.
    line_h = 14
    y = geom.log_rect.top + 30
    for line, is_current in collected:
        color = _LOG_FG if is_current else _LOG_DIM
        line_surf = small_font.render(line[:32], True, color)
        screen.blit(line_surf, (geom.log_rect.left + 8, y))
        y += line_h
        if y > geom.log_rect.bottom - 16:
            break
