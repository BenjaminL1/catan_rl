"""Unit tests for the viewer's board-layout geometry.

The board renderer needs pixel positions for every vertex, edge, and
port. The layout module derives these from axial-coord
:class:`BoardStatic` data without ever touching the engine. These
tests check the layout produces sensible output for a synthetic
mini-board and that the relationship between vertex pixels matches
the underlying adjacency.
"""

from __future__ import annotations

import math

from catan_rl.replay.schema import (
    BoardStatic,
    EdgeStatic,
    HexStatic,
    PortStatic,
    VertexStatic,
)
from catan_rl.replay.viewer.board_layout import compute_board_layout


def _tiny_board() -> BoardStatic:
    """A 3-hex L-shape with 3 shared vertices — easier to reason about
    than the 19-hex production board for layout-correctness checks."""
    hexes = (
        HexStatic(
            hex_idx=0, q=0, r=0, resource="DESERT", number_token=None, has_robber_initial=True
        ),
        HexStatic(hex_idx=1, q=1, r=0, resource="WOOD", number_token=8, has_robber_initial=False),
        HexStatic(hex_idx=2, q=0, r=1, resource="BRICK", number_token=5, has_robber_initial=False),
    )
    # 3 corners are shared between pairs of hexes; the rest are 1-hex.
    # We only construct enough vertices to test the layout logic — not
    # the full 18 corners of the L-shape.
    vertices = (
        # Interior 3-hex vertex (shared by hex 0, 1, 2 — at axial (0,0,1))
        VertexStatic(vertex_idx=0, adjacent_hex_indices=(0, 1, 2)),
        # 2-hex coastal vertex (shared by hex 0 + hex 1)
        VertexStatic(vertex_idx=1, adjacent_hex_indices=(0, 1)),
        # 1-hex tip vertex (only hex 0)
        VertexStatic(vertex_idx=2, adjacent_hex_indices=(0,)),
    )
    edges = (EdgeStatic(edge_idx=0, v1_idx=0, v2_idx=1),)
    ports = (PortStatic(port_idx=0, vertex_idx_pair=(0, 1), ratio="3:1", resource=None),)
    return BoardStatic(hexes=hexes, vertices=vertices, edges=edges, ports=ports)


class TestComputeBoardLayout:
    def test_hex_centers_have_expected_count(self) -> None:
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        assert len(layout.hex_centers) == 3
        assert len(layout.hex_corners_by_idx) == 3
        for corners in layout.hex_corners_by_idx:
            assert len(corners) == 6

    def test_axial_origin_maps_to_origin_pixel(self) -> None:
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        # Hex 0 is at axial (0, 0) → pixel = origin.
        assert layout.hex_centers[0] == (100.0, 100.0)

    def test_interior_3hex_vertex_is_shared_corner(self) -> None:
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        # The 3-hex interior vertex must coincide with a corner of all
        # three adjacent hexes. Verify by checking the resolved
        # vertex pixel lies within ~1px of a corner of each hex.
        v0 = layout.vertex_pixels[0]
        for h_idx in (0, 1, 2):
            distances = [
                math.hypot(c[0] - v0[0], c[1] - v0[1]) for c in layout.hex_corners_by_idx[h_idx]
            ]
            assert min(distances) < 1.0, (
                f"interior vertex 0 should coincide with a corner of "
                f"hex {h_idx}; min distance = {min(distances)}"
            )

    def test_edge_midpoint_is_midpoint_of_vertex_pixels(self) -> None:
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        v0 = layout.vertex_pixels[0]
        v1 = layout.vertex_pixels[1]
        mid = layout.edge_midpoints[0]
        assert abs(mid[0] - (v0[0] + v1[0]) / 2.0) < 0.01
        assert abs(mid[1] - (v0[1] + v1[1]) / 2.0) < 0.01

    def test_port_anchor_matches_edge_of_pair(self) -> None:
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        v0 = layout.vertex_pixels[0]
        v1 = layout.vertex_pixels[1]
        anchor = layout.port_anchors[0]
        assert abs(anchor[0] - (v0[0] + v1[0]) / 2.0) < 0.01
        assert abs(anchor[1] - (v0[1] + v1[1]) / 2.0) < 0.01

    def test_coastal_vertex_picks_outermost_corner(self) -> None:
        # The 1-hex tip vertex (vertex_idx=2) must land on one of the
        # 6 corners of hex 0. The disambiguation rule is "farthest
        # from board centroid" — verify the chosen corner is indeed
        # the one farthest from centroid.
        layout = compute_board_layout(_tiny_board(), hex_size=20.0, origin=(100.0, 100.0))
        v2 = layout.vertex_pixels[2]
        # v2 should be one of hex 0's corners.
        corners = layout.hex_corners_by_idx[0]
        min_dist = min(math.hypot(c[0] - v2[0], c[1] - v2[1]) for c in corners)
        assert min_dist < 1.0
        # And it should be the farthest from the centroid of the 3
        # hex centers.
        centroid = (
            sum(c[0] for c in layout.hex_centers) / 3,
            sum(c[1] for c in layout.hex_centers) / 3,
        )
        distances_to_centroid = [
            (math.hypot(c[0] - centroid[0], c[1] - centroid[1]), c) for c in corners
        ]
        farthest = max(distances_to_centroid, key=lambda x: x[0])[1]
        assert math.hypot(farthest[0] - v2[0], farthest[1] - v2[1]) < 1.0
