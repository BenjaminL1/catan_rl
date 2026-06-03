"""Unit tests for the shared rendering primitives (plan §A + §8.1).

Headless via ``SDL_VIDEODRIVER=dummy``. Each test boots a small surface,
calls one primitive, asserts something verifiable (pixel color at a
known point with tolerance, polygon bounding box, etc.).

Test budget: ~20 per-primitive tests + a full-board smoke + perf canary.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from catan_rl.engine.board import catanBoard  # noqa: E402
from catan_rl.gui import render, render_constants  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_screen(size: tuple[int, int] = (400, 300)) -> pygame.Surface:
    """Return a fresh in-memory surface (not the display)."""
    return pygame.Surface(size)


def _pixels_close(
    a: tuple[int, int, int, int] | pygame.Color,
    b: tuple[int, int, int],
    tol: int = 10,
) -> bool:
    """Per-channel tolerance comparison for spot-checks."""
    return all(abs(int(a[i]) - int(b[i])) <= tol for i in range(3))


def _seeded_board(seed: int = 2026) -> catanBoard:
    random.seed(seed)
    np.random.seed(seed)
    return catanBoard()


# ---------------------------------------------------------------------------
# draw_water
# ---------------------------------------------------------------------------


class TestDrawWater:
    def test_fills_canvas_with_water_color(self) -> None:
        screen = _new_screen((200, 150))
        render.draw_water(screen, (200, 150))
        for x, y in ((0, 0), (199, 0), (0, 149), (199, 149), (100, 75)):
            assert _pixels_close(screen.get_at((x, y)), render_constants.WATER_COLOR), (
                f"({x},{y}) not water: {screen.get_at((x, y))}"
            )


# ---------------------------------------------------------------------------
# draw_island_outline
# ---------------------------------------------------------------------------


class TestDrawIslandOutline:
    def test_island_covers_every_hex_center(self) -> None:
        screen = _new_screen((1100, 900))
        render.draw_water(screen, (1100, 900))
        board = _seeded_board()
        centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]
        render.draw_island_outline(screen, centers)
        for c in centers:
            assert _pixels_close(
                screen.get_at((int(c.x), int(c.y))), render_constants.SAND_COLOR, tol=20
            ), f"hex center ({int(c.x)},{int(c.y)}) not sand"

    def test_island_does_not_fill_water_corners(self) -> None:
        screen = _new_screen((1100, 900))
        render.draw_water(screen, (1100, 900))
        board = _seeded_board()
        centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]
        render.draw_island_outline(screen, centers)
        # Top-left corner is far from any hex center → still water.
        assert _pixels_close(screen.get_at((10, 10)), render_constants.WATER_COLOR), (
            f"corner pixel polluted: {screen.get_at((10, 10))}"
        )


# ---------------------------------------------------------------------------
# draw_hex_tile
# ---------------------------------------------------------------------------


class TestDrawHexTile:
    def test_centroid_matches_resource_color(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        hex_tile = board.hexTileDict[0]
        render.draw_hex_tile(screen, hex_tile, board, with_bevel=False)
        center = hex_tile.to_pixel(board.flat)
        expected = render_constants.TILE_COLORS[hex_tile.resource_type]
        assert _pixels_close(screen.get_at((int(center.x), int(center.y))), expected, tol=12), (
            f"hex 0 ({hex_tile.resource_type}) center not base color"
        )

    def test_bevel_top_lighter_than_bottom(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        hex_tile = board.hexTileDict[0]
        render.draw_hex_tile(screen, hex_tile, board, with_bevel=True)
        center = hex_tile.to_pixel(board.flat)
        # Sample 12 px above and 12 px below the center.
        above = screen.get_at((int(center.x), int(center.y) - 16))
        below = screen.get_at((int(center.x), int(center.y) + 16))
        # Top half should be lighter (higher RGB sum) than bottom half.
        top_sum = int(above[0]) + int(above[1]) + int(above[2])
        bot_sum = int(below[0]) + int(below[1]) + int(below[2])
        assert top_sum > bot_sum + 12, (
            f"bevel not visible: top={above} ({top_sum}) bot={below} ({bot_sum})"
        )

    def test_unknown_resource_uses_fallback_color(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        hex_tile = board.hexTileDict[0]
        # Mutate resource_type to a string that isn't in the lookup.
        original = hex_tile.resource_type
        hex_tile.resource_type = "UNKNOWN_RESOURCE_X"
        try:
            render.draw_hex_tile(screen, hex_tile, board, with_bevel=False)
        finally:
            hex_tile.resource_type = original
        center = hex_tile.to_pixel(board.flat)
        assert _pixels_close(
            screen.get_at((int(center.x), int(center.y))),
            render_constants.TILE_FALLBACK_COLOR,
            tol=12,
        )


# ---------------------------------------------------------------------------
# draw_number_token
# ---------------------------------------------------------------------------


class TestDrawNumberToken:
    @pytest.mark.parametrize("number", [6, 8])
    def test_red_text_for_six_and_eight(self, number: int) -> None:
        screen = _new_screen((100, 100))
        render.draw_number_token(screen, (50, 50), number)
        # Find a pixel inside the token that's not the cream background.
        # The text is centered; sample around the center.
        found_red = False
        for dx in range(-12, 13):
            for dy in range(-12, 0):
                px = screen.get_at((50 + dx, 50 + dy))
                if px[0] > 150 and px[1] < 80 and px[2] < 80:
                    found_red = True
                    break
            if found_red:
                break
        assert found_red, f"no red pixel found for number {number}"

    @pytest.mark.parametrize("number", [3, 4, 5, 9, 10, 11])
    def test_non_red_text_for_safe_numbers(self, number: int) -> None:
        screen = _new_screen((100, 100))
        render.draw_number_token(screen, (50, 50), number)
        # Confirm no high-red, low-green pixel exists inside the token rect.
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                px = screen.get_at((50 + dx, 50 + dy))
                # Red text would have R high, G+B low. Reject any.
                if px[0] > 170 and px[1] < 60 and px[2] < 60:
                    pytest.fail(f"red pixel found at ({dx},{dy}) for safe number {number}: {px}")

    @pytest.mark.parametrize(
        "number,expected_pips",
        [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)],
    )
    def test_pip_dot_count_matches_2d6(self, number: int, expected_pips: int) -> None:
        screen = _new_screen((100, 100))
        render.draw_number_token(screen, (50, 50), number)
        # Pip dots are below the number. Look at the bottom strip of the
        # token. Count distinct dot pixels by scanning a horizontal line
        # and counting non-background runs.
        # Bottom strip is around y=50 + size/2 - 8 (just inside token bottom).
        token_size = render_constants.NUMBER_TOKEN_SIZE
        scan_y = 50 + token_size // 2 - 7
        # Count transitions from background → dot.
        in_dot = False
        dot_count = 0
        for x in range(50 - token_size // 2 + 2, 50 + token_size // 2 - 2):
            px = screen.get_at((x, scan_y))
            # Dot pixel: dark green or red (high contrast vs cream background).
            r, g, b = int(px[0]), int(px[1]), int(px[2])
            is_dot = (r < 200 and g < 100) or (r > 150 and g < 80 and b < 80)
            if is_dot and not in_dot:
                dot_count += 1
                in_dot = True
            elif not is_dot:
                in_dot = False
        assert dot_count == expected_pips, (
            f"number {number}: expected {expected_pips} pips, found {dot_count}"
        )

    def test_does_not_raise_on_seven(self) -> None:
        # 7 has no token in Catan; the function should no-op gracefully.
        screen = _new_screen((100, 100))
        render.draw_number_token(screen, (50, 50), 7)
        # Center pixel should NOT be cream (we never drew the token).
        center = screen.get_at((50, 50))
        assert not _pixels_close(center, render_constants.NUMBER_TOKEN_BG, tol=5)


# ---------------------------------------------------------------------------
# draw_resource_symbol
# ---------------------------------------------------------------------------


class TestDrawResourceSymbol:
    @pytest.mark.parametrize("resource", list(render_constants.TILE_COLORS.keys()))
    def test_symbol_renders_without_exception(self, resource: str) -> None:
        screen = _new_screen((100, 100))
        # Pre-fill with a known background.
        screen.fill((128, 128, 128))
        render.draw_resource_symbol(screen, (50, 50), resource)
        # Centroid should be different from background — proves something was drawn.
        center = screen.get_at((50, 50))
        # Either emoji rendered (anything) or fallback drew a polygon — either
        # way the center pixel should not match the gray fill exactly.
        # Tolerance: if the rendered pixel is within 5 RGB of background,
        # the primitive didn't draw anything visible.
        background_match = _pixels_close(center, (128, 128, 128), tol=5)
        # For DESERT we draw a cactus shape that may not cover exact center.
        # So instead check the bounding 16x16 region.
        if background_match:
            any_non_bg = False
            for dx in range(-12, 13, 4):
                for dy in range(-12, 13, 4):
                    px = screen.get_at((50 + dx, 50 + dy))
                    if not _pixels_close(px, (128, 128, 128), tol=5):
                        any_non_bg = True
                        break
                if any_non_bg:
                    break
            assert any_non_bg, f"{resource}: nothing drawn"


# ---------------------------------------------------------------------------
# draw_port_ship
# ---------------------------------------------------------------------------


class TestDrawPortShip:
    def test_ship_footprint_visible(self) -> None:
        screen = _new_screen((200, 200))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "2:1", "BRICK", (100, 100))
        # Hull is centered at the anchor — sample below the anchor for hull color.
        hull_y = 100 + 8  # hull is just below anchor center
        px = screen.get_at((100, hull_y))
        assert _pixels_close(px, render_constants.PORT_HULL_COLOR, tol=20), (
            f"hull not at expected position: {px}"
        )

    def test_sail_visible_above_anchor(self) -> None:
        screen = _new_screen((200, 200))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "3:1", None, (100, 100))
        found_sail = False
        for dy in range(-20, -5):
            for dx in range(-8, 9):
                px = screen.get_at((100 + dx, 100 + dy))
                if _pixels_close(px, render_constants.PORT_SAIL_COLOR, tol=20):
                    found_sail = True
                    break
            if found_sail:
                break
        assert found_sail, "no sail color found above ship anchor"

    def test_label_badge_has_cream_bg_at_text_position(self) -> None:
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "2:1", "WHEAT", (150, 150))
        badge_y = 150 + render_constants.PORT_LABEL_VERTICAL_OFFSET
        # Badge layout: ratio text on LEFT, icon on RIGHT. The pixel
        # between them (and at the badge top/bottom edge) is cream.
        px = screen.get_at((150, badge_y - 9))
        assert _pixels_close(px, render_constants.PORT_LABEL_BG_COLOR, tol=20), (
            f"badge bg sample not cream: {px}"
        )

    def test_label_badge_contains_ratio_dark_text(self) -> None:
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "2:1", "WHEAT", (150, 150))
        badge_y = 150 + render_constants.PORT_LABEL_VERTICAL_OFFSET
        # Ratio text sits on the left half of the badge.
        found_text = False
        for dx in range(-30, -5):  # left side only
            for dy in range(-10, 11):
                px = screen.get_at((150 + dx, badge_y + dy))
                if int(px[0]) < 80 and int(px[1]) < 80 and int(px[2]) < 80:
                    found_text = True
                    break
            if found_text:
                break
        assert found_text, "no dark text pixel found on left side of badge"

    def test_label_badge_contains_resource_icon_pixels(self) -> None:
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "2:1", "BRICK", (150, 150))
        badge_y = 150 + render_constants.PORT_LABEL_VERTICAL_OFFSET
        # Icon sits on the right side of the badge. Look for any non-cream,
        # non-water pixel proving an icon was blitted.
        found_icon = False
        for dx in range(2, 30):  # right side
            for dy in range(-12, 13):
                px = screen.get_at((150 + dx, badge_y + dy))
                is_cream = _pixels_close(px, render_constants.PORT_LABEL_BG_COLOR, tol=20)
                is_water = _pixels_close(px, render_constants.WATER_COLOR, tol=20)
                if not is_cream and not is_water:
                    found_icon = True
                    break
            if found_icon:
                break
        assert found_icon, "no icon pixel found on right side of badge"

    def test_generic_port_uses_question_mark(self) -> None:
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "3:1", None, (150, 150))
        badge_y = 150 + render_constants.PORT_LABEL_VERTICAL_OFFSET
        # The right side of the badge must contain dark pixels from "?".
        found_glyph = False
        for dx in range(2, 25):
            for dy in range(-12, 13):
                px = screen.get_at((150 + dx, badge_y + dy))
                if int(px[0]) < 80 and int(px[1]) < 80 and int(px[2]) < 80:
                    found_glyph = True
                    break
            if found_glyph:
                break
        assert found_glyph, "no '?' glyph pixels in generic-port badge"

    def test_unknown_resource_falls_back_to_question_mark(self) -> None:
        # Passing a resource_type that isn't in the icon cache should
        # render the generic ? glyph without raising.
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        render.draw_port_ship(screen, "2:1", "GIBBERISH", (150, 150))
        # Just verify it didn't crash and the badge bg is present.
        badge_y = 150 + render_constants.PORT_LABEL_VERTICAL_OFFSET
        px = screen.get_at((150, badge_y - 9))
        assert _pixels_close(px, render_constants.PORT_LABEL_BG_COLOR, tol=20)


class TestNumberTokenAndSymbolDoNotOverlap:
    """With the new offsets, the symbol and token must not collide.

    Symbol center sits at RESOURCE_SYMBOL_VERTICAL_OFFSET (negative,
    above), token center at NUMBER_TOKEN_VERTICAL_OFFSET (positive,
    below). Verify the symbol bottom is at or above the token top.
    """

    def test_symbol_above_token_no_overlap(self) -> None:
        sym_offset = render_constants.RESOURCE_SYMBOL_VERTICAL_OFFSET
        tok_offset = render_constants.NUMBER_TOKEN_VERTICAL_OFFSET
        # Symbol surface bounding height ≈ RESOURCE_SYMBOL_FONT_SIZE + 6.
        sym_half = (render_constants.RESOURCE_SYMBOL_FONT_SIZE + 6) // 2
        sym_bottom = sym_offset + sym_half
        tok_top = tok_offset - render_constants.NUMBER_TOKEN_SIZE // 2
        assert sym_bottom <= tok_top, (
            f"symbol bottom {sym_bottom} overlaps token top {tok_top} "
            f"(sym_offset={sym_offset}, tok_offset={tok_offset})"
        )


# ---------------------------------------------------------------------------
# draw_port_planks
# ---------------------------------------------------------------------------


class TestDrawPortPlanks:
    def test_two_planks_drawn_to_each_vertex(self) -> None:
        screen = _new_screen((300, 300))
        screen.fill(render_constants.WATER_COLOR)
        anchor = (150, 50)
        v1 = (100, 200)
        v2 = (200, 200)
        render.draw_port_planks(screen, anchor, v1, v2)
        # Sample along each plank — midpoint should be plank-tan.
        for vx in (v1, v2):
            mid = ((anchor[0] + vx[0]) // 2, (anchor[1] + vx[1]) // 2)
            px = screen.get_at(mid)
            assert _pixels_close(px, render_constants.PORT_PLANK_COLOR, tol=15), (
                f"plank midpoint {mid} not tan: {px}"
            )


# ---------------------------------------------------------------------------
# draw_vertex_marker
# ---------------------------------------------------------------------------


class TestDrawVertexMarker:
    @pytest.mark.parametrize("state", render_constants.VERTEX_STATES)
    def test_marker_renders_state_color(self, state: str) -> None:
        screen = _new_screen((100, 100))
        screen.fill((30, 30, 35))
        render.draw_vertex_marker(screen, (50, 50), state)
        expected = render_constants.VERTEX_COLORS[state]
        px = screen.get_at((50, 50))
        assert _pixels_close(px, expected, tol=15), f"state {state}: expected {expected}, got {px}"


# ---------------------------------------------------------------------------
# draw_robber_pawn
# ---------------------------------------------------------------------------


class TestDrawRobberPawn:
    def test_robber_appears_on_correct_hex_center(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        # Find the desert (or whichever hex has has_robber=True; engine
        # places robber on the desert at construction).
        robber_hex = next(t for t in board.hexTileDict.values() if t.has_robber)
        # Render the hex first so we have a background to draw on.
        render.draw_hex_tile(screen, robber_hex, board, with_bevel=False)
        render.draw_robber_pawn(screen, robber_hex, board)
        center = robber_hex.to_pixel(board.flat)
        # Sample slightly below the center where the robber pawn body sits.
        px = screen.get_at((int(center.x), int(center.y) + 2))
        assert _pixels_close(px, render_constants.ROBBER_COLOR, tol=25), (
            f"robber not at hex center: {px}"
        )

    def test_robber_does_not_render_on_non_robber_hex(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        non_robber = next(t for t in board.hexTileDict.values() if not t.has_robber)
        render.draw_hex_tile(screen, non_robber, board, with_bevel=False)
        render.draw_robber_pawn(screen, non_robber, board)
        center = non_robber.to_pixel(board.flat)
        px = screen.get_at((int(center.x), int(center.y)))
        # Should still be the tile color, not robber gray.
        expected = render_constants.TILE_COLORS.get(
            non_robber.resource_type, render_constants.TILE_FALLBACK_COLOR
        )
        assert _pixels_close(px, expected, tol=15)


# ---------------------------------------------------------------------------
# Full-board integration smoke + perf canary
# ---------------------------------------------------------------------------


class TestFullBoardRender:
    def test_render_complete_scene_no_exceptions(self) -> None:
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]

        render.draw_water(screen, (1100, 900))
        render.draw_island_outline(screen, centers)
        for h_idx in range(19):
            hex_tile = board.hexTileDict[h_idx]
            render.draw_hex_tile(screen, hex_tile, board, with_bevel=True)
            center = hex_tile.to_pixel(board.flat)
            render.draw_resource_symbol(
                screen, (int(center.x), int(center.y) + 24), hex_tile.resource_type
            )
            if hex_tile.number_token and hex_tile.number_token != 0:
                render.draw_number_token(
                    screen, (int(center.x), int(center.y)), hex_tile.number_token
                )
        render.draw_robber_pawn(
            screen,
            next(t for t in board.hexTileDict.values() if t.has_robber),
            board,
        )
        # Should reach here without exceptions.
        assert True

    def test_full_board_render_under_perf_budget(self) -> None:
        """Single frame render < 50 ms on M1 Pro (33 ms is the 30 FPS budget)."""
        screen = _new_screen((1100, 900))
        board = _seeded_board()
        centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]

        # Warm up font/emoji caches first.
        render.draw_water(screen, (1100, 900))
        render.draw_island_outline(screen, centers)
        for h_idx in range(19):
            hex_tile = board.hexTileDict[h_idx]
            render.draw_hex_tile(screen, hex_tile, board, with_bevel=True)
            center = hex_tile.to_pixel(board.flat)
            render.draw_resource_symbol(
                screen, (int(center.x), int(center.y) + 24), hex_tile.resource_type
            )

        # Measure 5 frames, take median to absorb GC jitter.
        timings: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            render.draw_water(screen, (1100, 900))
            render.draw_island_outline(screen, centers)
            for h_idx in range(19):
                hex_tile = board.hexTileDict[h_idx]
                render.draw_hex_tile(screen, hex_tile, board, with_bevel=True)
                center = hex_tile.to_pixel(board.flat)
                render.draw_resource_symbol(
                    screen,
                    (int(center.x), int(center.y) + 24),
                    hex_tile.resource_type,
                )
                if hex_tile.number_token and hex_tile.number_token != 0:
                    render.draw_number_token(
                        screen,
                        (int(center.x), int(center.y)),
                        hex_tile.number_token,
                    )
            timings.append(time.perf_counter() - t0)
        timings.sort()
        median = timings[len(timings) // 2]
        # Soft cap: 50 ms. If this fires, profile + cache the island outline.
        assert median < 0.050, f"median frame time {median * 1000:.1f}ms > 50ms"
