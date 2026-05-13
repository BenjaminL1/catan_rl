"""Precomputed D6 permutation tables for the Catan hex board.

The 12 elements of the dihedral group D6 are encoded as integers 0..11:

  * ``g = 0..5``  → rotation by ``g · 60°`` counter-clockwise.
  * ``g = 6..11`` → reflection across the q-axis (i.e. ``r ↔ -q-r``)
    followed by rotation by ``(g - 6) · 60°``.

Tile, vertex (corner), and edge permutations are computed once from a
freshly-instantiated ``catanBoard`` and cached. The board's static
topology (axial coords, vertex pixel positions, edge endpoints) is what
matters; the random resource shuffle does not affect the symmetry tables.

Math derivation (kept here so future-us can audit):

  * Axial rotation by 60° CCW is the involution ``(q, r) → (-r, q + r)``.
  * Reflection across the q-axis is ``(q, r) → (q, -q - r)``. All D6
    reflections are this single mirror composed with a rotation.
  * Vertex permutations are derived by rotating the *centered* pixel
    coordinate and finding the nearest existing vertex; tolerance is 1px
    (the rotation should be exact, so any closer-than-1px hit is the
    target).
  * Edge ``(a, b)`` under permutation ``v_perm`` maps to
    ``(v_perm[a], v_perm[b])``.
  * Within-tile corner / edge slot permutations are *global* (the same
    for every tile because the board is rigid under D6) — derived once
    from the center tile.

The math is identical to v1 Phase 1.5 (which has been training-validated
through >18M PPO steps on the v1 codebase). The port mainly adds typing
discipline and tighter docstrings.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TypedDict

import numpy as np

#: 12 elements: 6 rotations + 6 reflections-then-rotations.
D6_GROUP_SIZE: int = 12

#: Identity element index.
D6_IDENTITY: int = 0


# ---------------------------------------------------------------------------
# Group structure
# ---------------------------------------------------------------------------


def rotation_steps(g: int) -> int:
    """Number of 60° CCW rotation steps in element ``g``.

    For the rotation half (``g < 6``) this is simply ``g``. For the
    reflection half (``g ∈ {6..11}``) the convention is *reflect first,
    then rotate by ``(g − 6) · 60°``*, keeping the within-tile slot
    permutations consistent.
    """
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return g if g < 6 else g - 6


def is_reflection(g: int) -> bool:
    """``True`` iff ``g`` includes the q-axis reflection (``g ∈ {6..11}``)."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return g >= 6


def D6_INVERSE(g: int) -> int:
    """Return the index of the group inverse of ``g``.

    Derivation: brute-force composition test using the corner permutation
    as a discriminator. The inverse table is built once and cached.
    """
    return _build_d6_inverse_table()[g]


# ---------------------------------------------------------------------------
# Axial-coord primitives
# ---------------------------------------------------------------------------


def _rotate_axial(q: int, r: int, k: int) -> tuple[int, int]:
    """Rotate hex axial coords ``k`` steps of 60° CCW."""
    k %= 6
    for _ in range(k):
        q, r = -r, q + r
    return q, r


def _reflect_axial(q: int, r: int) -> tuple[int, int]:
    """Reflect hex axial coords across the q-axis (``r ↔ -q-r``)."""
    return q, -q - r


def _apply_axial(g: int, q: int, r: int) -> tuple[int, int]:
    """Apply D6 element ``g`` to axial coordinates."""
    if is_reflection(g):
        q, r = _reflect_axial(q, r)
    return _rotate_axial(q, r, rotation_steps(g))


# ---------------------------------------------------------------------------
# Per-board data extraction (cached)
# ---------------------------------------------------------------------------


class _BoardData(TypedDict):
    """Static topology data extracted from one ``catanBoard`` instance."""

    tile_qr: dict[int, tuple[int, int]]
    qr_to_tile: dict[tuple[int, int], int]
    vertex_xy: dict[int, tuple[float, float]]
    n_vertices: int
    edges: list[tuple[int, int]]
    edge_to_idx: dict[tuple[int, int], int]
    n_edges: int
    tile_corners_vidx: dict[int, list[int]]


@lru_cache(maxsize=1)
def _board_data() -> _BoardData:
    """Build the static topology lookups needed to derive D6 perms.

    Cached because we pay the ~50ms construction cost once per process.
    Resource shuffle does not affect any of this output.
    """
    # Local import to keep module-import-time cheap (avoids pulling pygame
    # in at import time on the BC training path).
    from catan_rl.engine.board import catanBoard

    board = catanBoard()
    cx, cy = board.width / 2.0, board.height / 2.0

    tile_qr: dict[int, tuple[int, int]] = {
        i: (int(board.hexTileDict[i].q), int(board.hexTileDict[i].r)) for i in range(19)
    }
    qr_to_tile: dict[tuple[int, int], int] = {qr: i for i, qr in tile_qr.items()}

    vertex_xy: dict[int, tuple[float, float]] = {
        v_idx: (px.x - cx, px.y - cy) for v_idx, px in board.vertex_index_to_pixel_dict.items()
    }
    n_vertices = len(vertex_xy)

    edge_set: set[tuple[int, int]] = set()
    for _v_px, v_obj in board.boardGraph.items():
        a = v_obj.vertex_index
        for nb_px in v_obj.neighbors:
            b = board.boardGraph[nb_px].vertex_index
            edge_set.add((min(a, b), max(a, b)))
    edges = sorted(edge_set)
    edge_to_idx: dict[tuple[int, int], int] = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    tile_corners_vidx: dict[int, list[int]] = {}
    for tile_i in range(19):
        tile = board.hexTileDict[tile_i]
        corners_px = tile.get_corners(board.flat)
        tile_corners_vidx[tile_i] = [board.boardGraph[px].vertex_index for px in corners_px]

    return _BoardData(
        tile_qr=tile_qr,
        qr_to_tile=qr_to_tile,
        vertex_xy=vertex_xy,
        n_vertices=n_vertices,
        edges=edges,
        edge_to_idx=edge_to_idx,
        n_edges=n_edges,
        tile_corners_vidx=tile_corners_vidx,
    )


# ---------------------------------------------------------------------------
# Pixel-rotation primitives + nearest-vertex matcher
# ---------------------------------------------------------------------------


def _rotate_pixel(x: float, y: float, theta_rad: float, *, mirror: bool) -> tuple[float, float]:
    """Rotate (and optionally pre-mirror) a centred pixel coordinate.

    The mirror is across the y-axis (``x → -x``); composed with rotation
    this generates all 6 reflection axes through hex vertex pairs.
    """
    if mirror:
        x = -x
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y


def _nearest_vertex_index(
    target_xy: tuple[float, float],
    vertex_xy: dict[int, tuple[float, float]],
    eps: float = 1.0,
) -> int:
    """Find the vertex_index whose centred pixel is closest to ``target_xy``.

    Raises ``RuntimeError`` if no vertex is within ``eps`` pixels — the
    rotation should be exact in pixel space, so any farther hit means a
    coordinate that doesn't correspond to a real vertex (which is a bug).
    """
    best_idx, best_d2 = -1, float("inf")
    tx, ty = target_xy
    for v_idx, (vx, vy) in vertex_xy.items():
        d2 = (vx - tx) ** 2 + (vy - ty) ** 2
        if d2 < best_d2:
            best_d2, best_idx = d2, v_idx
    if best_d2 > eps * eps:
        raise RuntimeError(
            f"vertex mapping: nearest vertex {best_idx} is {math.sqrt(best_d2):.2f}px "
            f"away from target {target_xy}; expected exact match (eps={eps})"
        )
    return best_idx


# ---------------------------------------------------------------------------
# Build all permutations (cached)
# ---------------------------------------------------------------------------


class _AllPerms(TypedDict):
    tile: list[np.ndarray]
    corner: list[np.ndarray]
    edge: list[np.ndarray]
    within_corner: list[np.ndarray]
    within_edge: list[np.ndarray]


@lru_cache(maxsize=1)
def _all_perms() -> _AllPerms:
    """Compute and cache the full set of permutation tables for D6."""
    data = _board_data()
    tile_qr = data["tile_qr"]
    qr_to_tile = data["qr_to_tile"]
    vertex_xy = data["vertex_xy"]
    n_vertices = data["n_vertices"]
    edges = data["edges"]
    edge_to_idx = data["edge_to_idx"]
    n_edges = data["n_edges"]
    tile_corners_vidx = data["tile_corners_vidx"]

    tile_perms: list[np.ndarray] = []
    corner_perms: list[np.ndarray] = []
    edge_perms: list[np.ndarray] = []
    within_tile_c_perms: list[np.ndarray] = []
    within_tile_e_perms: list[np.ndarray] = []

    for g in range(D6_GROUP_SIZE):
        # ---- Tile perm via axial coords (exact integer arithmetic) -----
        tile_p = np.zeros(19, dtype=np.int64)
        for i in range(19):
            q, r = tile_qr[i]
            new_q, new_r = _apply_axial(g, q, r)
            tile_p[i] = qr_to_tile[(new_q, new_r)]
        tile_perms.append(tile_p)

        # ---- Vertex perm via pixel rotation + nearest neighbour --------
        theta = math.radians(60.0 * rotation_steps(g))
        mirror = is_reflection(g)
        v_perm = np.zeros(n_vertices, dtype=np.int64)
        for v_idx, (vx, vy) in vertex_xy.items():
            new_xy = _rotate_pixel(vx, vy, theta, mirror=mirror)
            v_perm[v_idx] = _nearest_vertex_index(new_xy, vertex_xy)
        corner_perms.append(v_perm)

        # ---- Edge perm: (a, b) → (v_perm[a], v_perm[b]) ----------------
        e_perm = np.zeros(n_edges, dtype=np.int64)
        for old_idx, (a, b) in enumerate(edges):
            na, nb = int(v_perm[a]), int(v_perm[b])
            new_e = (min(na, nb), max(na, nb))
            e_perm[old_idx] = edge_to_idx[new_e]
        edge_perms.append(e_perm)

        # ---- Within-tile corner slot perm ------------------------------
        # The board is globally rigid under D6 — derive the within-tile
        # slot perm once from the centre tile and use it for every tile.
        center_corners = tile_corners_vidx[0]
        rotated_center_corners = [int(v_perm[v]) for v in center_corners]
        slot_perm = np.zeros(6, dtype=np.int64)
        for new_slot in range(6):
            target_v = rotated_center_corners[new_slot]
            try:
                slot_perm[new_slot] = center_corners.index(target_v)
            except ValueError as e:
                raise RuntimeError(
                    f"D6 element {g}: rotated centre corner {new_slot} "
                    f"(v_idx={target_v}) not in original centre corners {center_corners}"
                ) from e
        within_tile_c_perms.append(slot_perm)

        # ---- Within-tile edge slot perm --------------------------------
        # Edge slot k spans corner slots k and (k+1) % 6. Under the corner
        # slot perm, edge slot k goes to whichever new edge slot connects
        # the new positions of those two corners.
        edge_slot_perm = np.zeros(6, dtype=np.int64)
        inv_slot = np.argsort(slot_perm)
        for old_e_slot in range(6):
            a_old, b_old = old_e_slot, (old_e_slot + 1) % 6
            a_new, b_new = int(inv_slot[a_old]), int(inv_slot[b_old])
            for new_e_slot in range(6):
                pair = {new_e_slot, (new_e_slot + 1) % 6}
                if pair == {a_new, b_new}:
                    edge_slot_perm[old_e_slot] = new_e_slot
                    break
        within_tile_e_perms.append(edge_slot_perm)

    return _AllPerms(
        tile=tile_perms,
        corner=corner_perms,
        edge=edge_perms,
        within_corner=within_tile_c_perms,
        within_edge=within_tile_e_perms,
    )


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def tile_perm(g: int) -> np.ndarray:
    """Permutation of 19 hex tile indices under D6 element ``g``.

    Convention: ``tile_perm(g)[i]`` is the *new* tile index that *old* tile
    ``i`` lands on after applying ``g``. To invert, use ``np.argsort``.
    """
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return _all_perms()["tile"][g].copy()


def corner_perm(g: int) -> np.ndarray:
    """Permutation of 54 vertex / corner indices under ``g``."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return _all_perms()["corner"][g].copy()


def edge_perm(g: int) -> np.ndarray:
    """Permutation of 72 edge indices under ``g``."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return _all_perms()["edge"][g].copy()


def within_tile_corner_perm(g: int) -> np.ndarray:
    """Permutation of the 6 within-tile vertex slots under ``g``."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return _all_perms()["within_corner"][g].copy()


def within_tile_edge_perm(g: int) -> np.ndarray:
    """Permutation of the 6 within-tile edge slots under ``g``."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return _all_perms()["within_edge"][g].copy()


# ---------------------------------------------------------------------------
# Inverse table
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_d6_inverse_table() -> tuple[int, ...]:
    """For each ``g`` find ``g_inv`` such that applying both yields identity.

    Uses the corner permutation as the discriminator: ``g_inv`` is the
    element whose corner permutation un-shuffles ``g``'s corner perm.
    """
    perms = _all_perms()["corner"]
    inverse = [-1] * D6_GROUP_SIZE
    n_vertices = perms[0].shape[0]
    identity_perm = np.arange(n_vertices, dtype=np.int64)
    for g in range(D6_GROUP_SIZE):
        for h in range(D6_GROUP_SIZE):
            # Composition: apply h then g. Final mapping is perms[g][perms[h][i]].
            composed = perms[g][perms[h]]
            if np.array_equal(composed, identity_perm):
                inverse[g] = h
                break
        if inverse[g] == -1:
            raise RuntimeError(f"D6 inverse not found for element {g}")
    return tuple(inverse)
