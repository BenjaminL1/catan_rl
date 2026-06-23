"""Loader for the committed engine board-topology fixture.

``topology.json`` is generated once by ``scripts/export_topology.py`` from the
reference ``catanBoard`` (purely geometric — vertex/edge/corner IDs do not depend
on the random resource/number assignment) and committed as a **package fixture**.
The pipeline reads it via :func:`importlib.resources` (never ``/tmp``) so parsed
records map onto the same engine integer IDs (19 hex / 54 vertex / 72 edge) the RL
stack uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

#: Standard-board structural counts the committed fixture must satisfy.
NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72
NUM_PORTS = 9


@dataclass(frozen=True, slots=True)
class Topology:
    """Static integer-ID board topology in the engine's canonical ordering."""

    hex_corner_to_vertex: tuple[tuple[int, ...], ...]
    vertex_adjacent_hexes: tuple[tuple[int, ...], ...]
    edge_vertices: tuple[tuple[int, int], ...]
    vertex_neighbors: tuple[tuple[int, ...], ...]
    port_slots: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if len(self.hex_corner_to_vertex) != NUM_HEXES:
            raise ValueError(f"expected {NUM_HEXES} hexes, got {len(self.hex_corner_to_vertex)}")
        if len(self.vertex_adjacent_hexes) != NUM_VERTICES:
            raise ValueError(
                f"expected {NUM_VERTICES} vertices, got {len(self.vertex_adjacent_hexes)}"
            )
        if len(self.edge_vertices) != NUM_EDGES:
            raise ValueError(f"expected {NUM_EDGES} edges, got {len(self.edge_vertices)}")
        if len(self.port_slots) != NUM_PORTS:
            raise ValueError(f"expected {NUM_PORTS} port slots, got {len(self.port_slots)}")


def load_topology() -> Topology:
    """Load the committed ``topology.json`` package fixture."""
    resource = files("catan_rl.human_data").joinpath("topology.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return Topology(
        hex_corner_to_vertex=tuple(tuple(row) for row in payload["hexCornerToVertex"]),
        vertex_adjacent_hexes=tuple(tuple(row) for row in payload["vertexAdjacentHexes"]),
        edge_vertices=tuple((int(a), int(b)) for a, b in payload["edgeVertices"]),
        vertex_neighbors=tuple(tuple(row) for row in payload["vertexNeighbors"]),
        port_slots=tuple(dict(slot) for slot in payload["portSlots"]),
    )
