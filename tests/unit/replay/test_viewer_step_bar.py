"""Unit tests for the V3 ViewerControl state machine — scrubbing,
play/pause, bounds clamping. Pygame events are simulated as tiny
NamedTuple-like attrs since we don't need an SDL surface to test
the state logic."""

from __future__ import annotations

from types import SimpleNamespace

import pygame

from catan_rl.replay.viewer.step_bar import ViewerControl, handle_event


class TestViewerControl:
    def test_step_by_clamps_at_bounds(self) -> None:
        c = ViewerControl(total_steps=10, step_idx=5)
        c.step_by(-100)
        assert c.step_idx == 0
        c.step_by(100)
        assert c.step_idx == 9

    def test_step_to_clamps_at_bounds(self) -> None:
        c = ViewerControl(total_steps=10)
        c.step_to(-5)
        assert c.step_idx == 0
        c.step_to(100)
        assert c.step_idx == 9

    def test_tick_advances_only_when_playing(self) -> None:
        c = ViewerControl(total_steps=10, play_speed_frames=2)
        # Not playing: tick is a no-op.
        c.tick()
        assert c.step_idx == 0
        # Playing: advance every play_speed_frames ticks.
        c.toggle_play()
        c.tick()
        assert c.step_idx == 0  # tick 1 — not yet
        c.tick()
        assert c.step_idx == 1  # tick 2 — advanced

    def test_tick_stops_at_end(self) -> None:
        c = ViewerControl(total_steps=2, step_idx=1, play_speed_frames=1)
        c.toggle_play()
        c.tick()
        assert c.step_idx == 1
        # Auto-stops at the final step.
        assert c.playing is False


class TestHandleEvent:
    def _key_event(self, key: int) -> SimpleNamespace:
        return SimpleNamespace(type=pygame.KEYDOWN, key=key)

    def _click_event(self, pos: tuple[int, int]) -> SimpleNamespace:
        return SimpleNamespace(type=pygame.MOUSEBUTTONDOWN, button=1, pos=pos)

    def test_arrow_keys_step(self) -> None:
        c = ViewerControl(total_steps=10, step_idx=5)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        assert handle_event(c, self._key_event(pygame.K_LEFT), bar_rect)
        assert c.step_idx == 4
        assert handle_event(c, self._key_event(pygame.K_RIGHT), bar_rect)
        assert c.step_idx == 5

    def test_page_keys_step_by_10(self) -> None:
        c = ViewerControl(total_steps=100, step_idx=50)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        handle_event(c, self._key_event(pygame.K_PAGEUP), bar_rect)
        assert c.step_idx == 40
        handle_event(c, self._key_event(pygame.K_PAGEDOWN), bar_rect)
        assert c.step_idx == 50

    def test_home_end_jump(self) -> None:
        c = ViewerControl(total_steps=100, step_idx=50)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        handle_event(c, self._key_event(pygame.K_HOME), bar_rect)
        assert c.step_idx == 0
        handle_event(c, self._key_event(pygame.K_END), bar_rect)
        assert c.step_idx == 99

    def test_space_toggles_play(self) -> None:
        c = ViewerControl(total_steps=10)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        assert c.playing is False
        handle_event(c, self._key_event(pygame.K_SPACE), bar_rect)
        assert c.playing is True
        handle_event(c, self._key_event(pygame.K_SPACE), bar_rect)
        assert c.playing is False

    def test_click_scrubs_to_proportional_position(self) -> None:
        c = ViewerControl(total_steps=100)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        # Click at x=50 in a 100-wide bar → 50% → step 50.
        handle_event(c, self._click_event((50, 10)), bar_rect)
        assert c.step_idx == 50
        # Click at far right edge (x=99) → step 99 after clamp.
        handle_event(c, self._click_event((99, 10)), bar_rect)
        assert c.step_idx == 99
        # Click at the start → step 0.
        handle_event(c, self._click_event((0, 10)), bar_rect)
        assert c.step_idx == 0

    def test_click_outside_bar_is_noop(self) -> None:
        c = ViewerControl(total_steps=100, step_idx=42)
        bar_rect = pygame.Rect(0, 0, 100, 20)
        # Click well above the bar.
        handle_event(c, self._click_event((50, 500)), bar_rect)
        assert c.step_idx == 42
