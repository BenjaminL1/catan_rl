"""Axial → pixel math for hex grids (pointy-top orientation).

This module is intentionally tiny and self-contained. It will be
duplicated under :mod:`catan_rl.replay.viewer.hex_math` once the
viewer package lands, so the viewer can render without any package
import that transitively pulls in torch / env / engine.

Conventions:

* **Pointy-top hexagons**: flat sides on left/right, points at top/bottom.
* **Axial coords**: ``(q, r)`` where ``q`` increments east and ``r``
  increments south-east. Origin ``(0, 0)`` maps to ``origin_xy`` in
  pixel space.
* **Hex size**: distance from hex center to any corner, in pixels.
  The horizontal hex-to-hex spacing is ``size * sqrt(3)``; vertical is
  ``size * 1.5``.
* **Corner ordering**: starting at corner at angle ``30°`` from the
  +x axis. In standard math coordinates (y-up) this is the upper-right
  corner; **in screen coordinates with y growing down (pygame's
  convention), this is the lower-right corner**. Subsequent corners
  proceed clockwise on screen (counter-clockwise in math coords).

Reference: https://www.redblobgames.com/grids/hexagons/ (axial coords
section).
"""

from __future__ import annotations

import math

_SQRT3 = math.sqrt(3.0)

#: Six pointy-top corner offset angles, in degrees, starting at the
#: upper-right corner and rotating clockwise. Pre-converted to radians
#: at module import to avoid a per-call ``math.radians`` cost.
_CORNER_ANGLES_RAD: tuple[float, ...] = tuple(math.radians(30.0 + 60.0 * i) for i in range(6))


def axial_to_pixel(
    q: float, r: float, hex_size: float, origin: tuple[float, float]
) -> tuple[float, float]:
    """Map axial ``(q, r)`` to a pixel ``(x, y)`` for the hex center.

    With pointy-top orientation: ``x = origin.x + hex_size * sqrt(3) *
    (q + r/2)``, ``y = origin.y + hex_size * 1.5 * r``. The function
    is a pure transform — no state, no rounding."""
    x = origin[0] + hex_size * _SQRT3 * (q + r / 2.0)
    y = origin[1] + hex_size * 1.5 * r
    return (x, y)


def hex_corners(
    q: float, r: float, hex_size: float, origin: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    """Return the 6 pixel corners of the hex at ``(q, r)``.

    Corner 0 sits at angle 30° from the +x axis. In screen
    coordinates (y-down) that is the lower-right corner; subsequent
    corners proceed clockwise. See the module docstring for the
    ordering rationale."""
    cx, cy = axial_to_pixel(q, r, hex_size, origin)
    return tuple(
        (cx + hex_size * math.cos(a), cy + hex_size * math.sin(a)) for a in _CORNER_ANGLES_RAD
    )


def edge_midpoint(v1_xy: tuple[float, float], v2_xy: tuple[float, float]) -> tuple[float, float]:
    """Arithmetic midpoint between two vertex pixels. Used for road
    label / icon placement."""
    return ((v1_xy[0] + v2_xy[0]) / 2.0, (v1_xy[1] + v2_xy[1]) / 2.0)
