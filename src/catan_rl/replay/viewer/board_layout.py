"""Pure-geometry helpers that turn axial-coord ``BoardStatic`` into
pixel-space layout data the renderer consumes.

The viewer does NOT import the engine, so the standard-board vertex
adjacency from :mod:`catan_rl.replay.schema.BoardStatic` is the only
input we have. We derive vertex pixels by intersecting the 6 corners
of each adjacent hex:

* **Interior 3-hex vertex** — the 3 adjacent hexes share exactly 1
  corner point. We pick it by voting across all hex-corner sets and
  taking the location with max-overlap count (within a small
  tolerance scaled by ``hex_size``).
* **Coastal 2-hex vertex** — the 2 adjacent hexes share 2 corners
  (the endpoints of their shared edge). Without more info we pick
  the corner farther from the board centroid (i.e., the outermost
  of the 2 candidates).
* **1-hex tip vertex** — only 6 corners of one hex; all candidates
  have overlap count 1. We pick the corner farthest from the board
  centroid. This is an MVP heuristic — for the standard 19-hex
  layout it places these corners visually plausibly. A future schema
  bump could carry explicit corner indices to make this exact.

Edges are placed at the midpoint of their two vertex pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

from catan_rl.replay.hex_math import axial_to_pixel, edge_midpoint, hex_corners
from catan_rl.replay.schema import BoardStatic


@dataclass(frozen=True, slots=True)
class BoardLayout:
    """Pixel positions for every static board element at a given
    ``hex_size`` and ``origin``."""

    hex_centers: tuple[tuple[float, float], ...]
    hex_corners_by_idx: tuple[tuple[tuple[float, float], ...], ...]
    """For each hex_idx, the 6 corner pixels in pointy-top order."""

    vertex_pixels: tuple[tuple[float, float], ...]
    edge_midpoints: tuple[tuple[float, float], ...]
    port_anchors: tuple[tuple[float, float], ...]
    """Midpoint of each port's two adjacent vertices."""


def compute_board_layout(
    board: BoardStatic,
    *,
    hex_size: float,
    origin: tuple[float, float],
) -> BoardLayout:
    """Materialise a :class:`BoardLayout` for the given ``hex_size`` /
    ``origin``. Pure function — call it once per viewer init or
    whenever the window size changes."""
    # Hex centers + corner pixels (cached so vertex resolution can
    # walk them).
    centers: list[tuple[float, float]] = []
    corners_by_idx: list[tuple[tuple[float, float], ...]] = []
    for h in board.hexes:
        centers.append(axial_to_pixel(h.q, h.r, hex_size, origin))
        corners_by_idx.append(hex_corners(h.q, h.r, hex_size, origin))

    # Board centroid — used to disambiguate coastal vertex positions
    # by picking the candidate corner farthest from the centroid.
    centroid = (
        sum(c[0] for c in centers) / len(centers),
        sum(c[1] for c in centers) / len(centers),
    )

    # Tolerance for "same corner across hexes" comparison. The hex
    # corner radius is ``hex_size``, so a tolerance of ~5% of that is
    # tight enough to avoid false matches and loose enough to absorb
    # floating-point noise.
    tol = hex_size * 0.05

    vertex_pixels = tuple(
        _resolve_vertex_pixel(
            vertex_adj=v.adjacent_hex_indices,
            corners_by_idx=corners_by_idx,
            centroid=centroid,
            tol=tol,
        )
        for v in board.vertices
    )

    edge_midpoints = tuple(
        edge_midpoint(vertex_pixels[e.v1_idx], vertex_pixels[e.v2_idx]) for e in board.edges
    )

    port_anchors = tuple(
        edge_midpoint(vertex_pixels[p.vertex_idx_pair[0]], vertex_pixels[p.vertex_idx_pair[1]])
        for p in board.ports
    )

    return BoardLayout(
        hex_centers=tuple(centers),
        hex_corners_by_idx=tuple(corners_by_idx),
        vertex_pixels=vertex_pixels,
        edge_midpoints=edge_midpoints,
        port_anchors=port_anchors,
    )


def _resolve_vertex_pixel(
    *,
    vertex_adj: tuple[int, ...],
    corners_by_idx: list[tuple[tuple[float, float], ...]],
    centroid: tuple[float, float],
    tol: float,
) -> tuple[float, float]:
    """Pick the vertex's pixel by majority-vote across adjacent-hex
    corner sets, breaking ties by distance-from-centroid (outermost
    wins for coastal vertices)."""
    # Gather all candidate corner pixels with their overlap counts.
    candidates: list[tuple[tuple[float, float], int]] = []
    for hex_idx in vertex_adj:
        for corner in corners_by_idx[hex_idx]:
            # Look for an existing candidate within ``tol``.
            merged = False
            for i, (existing, count) in enumerate(candidates):
                if _close(existing, corner, tol):
                    candidates[i] = (existing, count + 1)
                    merged = True
                    break
            if not merged:
                candidates.append((corner, 1))

    # Highest overlap first; ties broken by max distance from centroid.
    candidates.sort(
        key=lambda c: (-c[1], -((c[0][0] - centroid[0]) ** 2 + (c[0][1] - centroid[1]) ** 2))
    )
    return candidates[0][0]


def _close(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
