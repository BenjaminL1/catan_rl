"""Unit tests for the unicode emoji probe + polygon fallback (plan §8.2).

The probe runs once per process at module load. These tests force a
fresh cache so we can pin behaviour under controlled font availability.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from catan_rl.gui import render, render_constants  # noqa: E402


@pytest.fixture(autouse=True)
def reset_symbol_cache():
    """Clear the per-process cache before each test so probes re-run."""
    render._SYMBOL_CACHE.clear()
    render._CACHE_INITIALISED = False
    yield
    render._SYMBOL_CACHE.clear()
    render._CACHE_INITIALISED = False


def _make_blank_surface(width: int = 24, height: int = 24) -> pygame.Surface:
    """Surface with no visible content (alpha = 0 everywhere)."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    return surf


def _make_narrow_surface() -> pygame.Surface:
    """Surface visibly populated but only ~3 px wide (below threshold)."""
    surf = pygame.Surface((4, 24), pygame.SRCALPHA)
    surf.fill((255, 0, 0, 255))
    return surf


def _make_wide_surface() -> pygame.Surface:
    """Surface comfortably above the bbox threshold."""
    surf = pygame.Surface((28, 28), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.circle(surf, (255, 128, 64, 255), (14, 14), 10)
    return surf


class TestFallbackWhenGlyphBlank:
    def test_all_resources_fallback_when_probe_returns_none(self) -> None:
        with patch.object(render, "_try_render_emoji", return_value=None):
            render._build_symbol_cache()
        for resource in render_constants.RESOURCE_EMOJI:
            entry = render._SYMBOL_CACHE[resource]
            assert entry.used_fallback is True, f"{resource}: expected fallback, got emoji path"


class TestSurfaceVisibilityHelper:
    def test_blank_surface_judged_invisible(self) -> None:
        assert render._surface_has_visible_glyph(_make_blank_surface()) is False

    def test_narrow_surface_judged_invisible(self) -> None:
        # 4-px wide surface is below EMOJI_PROBE_MIN_BBOX_WIDTH (8).
        assert render._surface_has_visible_glyph(_make_narrow_surface()) is False

    def test_wide_surface_judged_visible(self) -> None:
        assert render._surface_has_visible_glyph(_make_wide_surface()) is True


class TestNoFallbackWhenGlyphRenders:
    def test_all_resources_use_emoji_when_probe_returns_surface(self) -> None:
        wide = _make_wide_surface()
        with patch.object(render, "_try_render_emoji", return_value=wide):
            render._build_symbol_cache()
        for resource in render_constants.RESOURCE_EMOJI:
            entry = render._SYMBOL_CACHE[resource]
            assert entry.used_fallback is False, (
                f"{resource}: emoji surface available but flagged as fallback"
            )


class TestCacheBuiltOnce:
    def test_build_is_idempotent(self) -> None:
        # First call populates the cache.
        render._build_symbol_cache()
        first_cache_size = len(render._SYMBOL_CACHE)
        first_entries = dict(render._SYMBOL_CACHE)
        # Second call should be a no-op.
        render._build_symbol_cache()
        assert len(render._SYMBOL_CACHE) == first_cache_size
        for resource, entry in first_entries.items():
            assert render._SYMBOL_CACHE[resource] is entry, (
                f"{resource}: cache entry replaced on second call"
            )

    def test_draw_resource_symbol_lazy_initializes_cache(self) -> None:
        # Cache should be empty after the fixture's reset.
        assert not render._CACHE_INITIALISED
        screen = pygame.Surface((100, 100))
        render.draw_resource_symbol(screen, (50, 50), "WOOD")
        assert render._CACHE_INITIALISED
        assert "WOOD" in render._SYMBOL_CACHE


class TestUnknownResource:
    def test_unknown_resource_gets_fallback_icon(self) -> None:
        screen = pygame.Surface((100, 100))
        screen.fill((50, 50, 50))
        # Force cache initialisation first so the unknown-resource path
        # exercises the on-the-fly fallback branch.
        render._build_symbol_cache()
        render.draw_resource_symbol(screen, (50, 50), "UNKNOWN_RESOURCE_FOO")
        # After the call, the unknown resource should be cached with
        # used_fallback=True.
        entry = render._SYMBOL_CACHE.get("UNKNOWN_RESOURCE_FOO")
        assert entry is not None
        assert entry.used_fallback is True


class TestFallbackIconShapes:
    """Smoke: each fallback icon renders non-empty pixels in the expected color."""

    @pytest.mark.parametrize("resource", ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT"])
    def test_fallback_icon_has_visible_content(self, resource: str) -> None:
        surf = render._draw_fallback_icon(resource)
        # Bounding rect of opaque pixels should be non-trivial.
        bbox = surf.get_bounding_rect()
        assert bbox.width > 5 and bbox.height > 5, f"{resource}: fallback bbox too small ({bbox})"
