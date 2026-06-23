"""Export Torevan board topology tables from the reference catanBoard.

Phase 1b of the Torevan MVP build. Dumps the static integer-ID topology of a
standard Catan board to JSON in the reference engine's deterministic ordering,
so the TypeScript engine (@torevan/engine) can bake it as a committed fixture
and verify byte-compatibility rather than re-deriving float pixel math.

The topology is purely geometric (vertex/edge/corner IDs do NOT depend on the
random resource/number/port assignment), so any constructed board yields the
same tables. We still seed numpy for full determinism.

Run:
    PYTHONPATH=src python3 scripts/export_topology.py > /tmp/topology.fixture.json

Ordering reproduced (matches the live RL engine):
  * vertex_idx: first-occurrence dedup order while walking hex 0..18,
    corner 0..5 (corners are CW from 30deg E via BoardVertex pixel rounding).
  * edge_idx:   CatanEnv._build_index_maps order — iterate boardGraph by
    vertex-pixel insertion order (== vertex index order), then each vertex's
    .neighbors list, dedup by lex-sorted pixel-string key (_edge_key).
  * port slots: the fixed (hex, corner1, corner2) table in board.updatePorts,
    resolved to vertex pairs through the hex-corner geometry.
"""

from __future__ import annotations

import json
import os
import sys

# Suppress pygame's stdout banner so the JSON we emit on stdout stays clean.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from catan_rl.engine.board import catanBoard


def _edge_key(v1: object, v2: object) -> tuple[str, str]:
    """Replica of CatanEnv._edge_key — lex-sort pixel-string representations."""
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


# The fixed port (hex, corner1, corner2) table — copied from
# catanBoard.updatePorts so we resolve vertex pairs the same way the engine does.
PORT_HEX_CORNERS: list[tuple[int, int, int]] = [
    (7, 2, 3),
    (8, 1, 2),
    (10, 1, 2),
    (11, 0, 1),
    (12, 5, 0),
    (14, 0, 5),
    (15, 4, 5),
    (16, 3, 4),
    (18, 3, 4),
]


def main() -> None:
    np.random.seed(0)
    board = catanBoard()

    # pixel -> vertex_idx (the canonical map)
    pixel_to_idx: dict[object, int] = {
        px: idx for idx, px in board.vertex_index_to_pixel_dict.items()
    }
    assert len(pixel_to_idx) == 54, f"expected 54 vertices, got {len(pixel_to_idx)}"

    # --- hexCornerToVertex[hex][corner] -> vertex_idx -------------------------
    hex_corner_to_vertex: list[list[int]] = []
    for hex_idx in range(19):
        tile = board.hexTileDict[hex_idx]
        corners = tile.get_corners(board.flat)  # 6 pixel Points, CW from 30E
        row: list[int] = []
        for corner_pt in corners:
            # Match the corner pixel back to a vertex via exact-pixel distance,
            # reproducing updatePorts.get_vertex_index (round(dist) == 0).
            found = None
            for v_px, v_idx in pixel_to_idx.items():
                dx = corner_pt.x - v_px.x
                dy = corner_pt.y - v_px.y
                if round((dx * dx + dy * dy) ** 0.5) == 0:
                    found = v_idx
                    break
            assert found is not None, f"hex {hex_idx} corner pixel {corner_pt} unmatched"
            row.append(found)
        assert len(row) == 6
        hex_corner_to_vertex.append(row)

    # --- vertexAdjacentHexes[vertex] -> [hex, ...] ---------------------------
    vertex_adjacent_hexes: list[list[int]] = [[] for _ in range(54)]
    for v_idx in range(54):
        px = board.vertex_index_to_pixel_dict[v_idx]
        vobj = board.boardGraph[px]
        vertex_adjacent_hexes[v_idx] = [int(h) for h in vobj.adjacent_hex_indices]

    # --- edge list in engine-canonical _build_index_maps order ---------------
    seen: set[tuple[str, str]] = set()
    edge_order: list[tuple[int, int]] = []  # (v1_idx, v2_idx) in engine ID order
    for v_px, v_obj in board.boardGraph.items():
        for nb_px in v_obj.neighbors:
            key = _edge_key(v_px, nb_px)
            if key in seen:
                continue
            seen.add(key)
            a = pixel_to_idx[v_px]
            b = pixel_to_idx[nb_px]
            lo, hi = (a, b) if a < b else (b, a)
            edge_order.append((lo, hi))
    assert len(edge_order) == 72, f"expected 72 edges, got {len(edge_order)}"

    # edge_vertices: same list, each pair already ascending (v1 < v2).
    edge_vertices: list[list[int]] = [[lo, hi] for (lo, hi) in edge_order]

    # --- vertexNeighbors[vertex] -> [vertex, ...] ---------------------------
    vertex_neighbors: list[list[int]] = [[] for _ in range(54)]
    for v_idx in range(54):
        px = board.vertex_index_to_pixel_dict[v_idx]
        vobj = board.boardGraph[px]
        nbrs = [pixel_to_idx[nb_px] for nb_px in vobj.neighbors]
        vertex_neighbors[v_idx] = nbrs

    # --- port slots ----------------------------------------------------------
    port_slots: list[dict[str, object]] = []
    for slot, (h_idx, c1, c2) in enumerate(PORT_HEX_CORNERS):
        v1 = hex_corner_to_vertex[h_idx][c1]
        v2 = hex_corner_to_vertex[h_idx][c2]
        port_slots.append(
            {
                "slot": slot,
                "hex": h_idx,
                "corners": [c1, c2],
                "vertices": [v1, v2],
            }
        )

    # --- sanity invariants before emitting -----------------------------------
    flat_vertices = {v for row in hex_corner_to_vertex for v in row}
    assert flat_vertices == set(range(54)), "hexCornerToVertex must cover 0..53"

    # degree histogram of vertexAdjacentHexes (number of hexes each vertex touches)
    hist: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for adj in vertex_adjacent_hexes:
        hist[len(adj)] += 1
    assert hist == {1: 18, 2: 12, 3: 24}, f"unexpected hex-adjacency histogram {hist}"
    # 18 perimeter vertices touch 1 hex, 12 touch 2, 24 interior touch 3.

    payload = {
        "hexCornerToVertex": hex_corner_to_vertex,
        "vertexAdjacentHexes": vertex_adjacent_hexes,
        "edgeVertices": edge_vertices,
        "vertexNeighbors": vertex_neighbors,
        "portSlots": port_slots,
    }

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
