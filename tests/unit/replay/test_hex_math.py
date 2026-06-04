"""Tests for `catan_rl.replay.hex_math`."""

from __future__ import annotations

import math

import pytest

from catan_rl.replay.hex_math import axial_to_pixel, edge_midpoint, hex_corners


class TestAxialToPixel:
    def test_origin_maps_to_origin(self) -> None:
        x, y = axial_to_pixel(0, 0, hex_size=80.0, origin=(500.0, 400.0))
        assert x == pytest.approx(500.0)
        assert y == pytest.approx(400.0)

    def test_east_neighbour_spacing(self) -> None:
        # Moving (q, r) -> (q+1, r) shifts east by size * sqrt(3).
        ax, ay = axial_to_pixel(0, 0, hex_size=10.0, origin=(0.0, 0.0))
        bx, by = axial_to_pixel(1, 0, hex_size=10.0, origin=(0.0, 0.0))
        assert bx - ax == pytest.approx(10.0 * math.sqrt(3.0))
        assert ay == pytest.approx(by)

    def test_southeast_neighbour_spacing(self) -> None:
        # (q, r) -> (q, r+1) shifts south-east by (size*sqrt(3)/2, size*1.5).
        ax, ay = axial_to_pixel(0, 0, hex_size=20.0, origin=(0.0, 0.0))
        bx, by = axial_to_pixel(0, 1, hex_size=20.0, origin=(0.0, 0.0))
        assert bx - ax == pytest.approx(20.0 * math.sqrt(3.0) / 2.0)
        assert by - ay == pytest.approx(20.0 * 1.5)


class TestHexCorners:
    def test_returns_six_corners(self) -> None:
        corners = hex_corners(0, 0, hex_size=50.0, origin=(0.0, 0.0))
        assert len(corners) == 6

    def test_all_corners_equidistant_from_center(self) -> None:
        size = 37.5
        cx, cy = axial_to_pixel(0, 0, hex_size=size, origin=(100.0, 200.0))
        corners = hex_corners(0, 0, hex_size=size, origin=(100.0, 200.0))
        for x, y in corners:
            d = math.hypot(x - cx, y - cy)
            assert d == pytest.approx(size, abs=1e-9)

    def test_corner_ordering_clockwise_from_upper_right(self) -> None:
        # The first corner is at angle 30° (upper-right); the next is
        # at 90° (right tip). Verify by checking that corner 1.x >=
        # corner 0.x (it's the rightmost) and corner 0.y >= corner 1.y
        # in screen coords (y grows down).
        size = 10.0
        corners = hex_corners(0, 0, hex_size=size, origin=(0.0, 0.0))
        # Corner 0 is upper-right; corner 1 is rightmost (3 o'clock).
        # Corner 0: (size*cos(30°), size*sin(30°)) = (8.66, 5.0)
        # Corner 1: (size*cos(90°), size*sin(90°)) = (0.0, 10.0)
        assert corners[0][0] == pytest.approx(size * math.cos(math.radians(30.0)))
        assert corners[0][1] == pytest.approx(size * math.sin(math.radians(30.0)))
        assert corners[1][0] == pytest.approx(size * math.cos(math.radians(90.0)), abs=1e-12)
        assert corners[1][1] == pytest.approx(size * math.sin(math.radians(90.0)))

    def test_ordering_is_stable_across_calls(self) -> None:
        c1 = hex_corners(2, -1, hex_size=42.0, origin=(50.0, 50.0))
        c2 = hex_corners(2, -1, hex_size=42.0, origin=(50.0, 50.0))
        assert c1 == c2


class TestEdgeMidpoint:
    def test_arithmetic_mean(self) -> None:
        mid = edge_midpoint((10.0, 20.0), (30.0, 50.0))
        assert mid == (20.0, 35.0)

    def test_same_point_returns_same_point(self) -> None:
        mid = edge_midpoint((7.5, -3.0), (7.5, -3.0))
        assert mid == (7.5, -3.0)
