"""Precomputed D6 permutation tables for Catan board symmetries (Phase 1.5).

The 12 elements of the dihedral group D6 are encoded as integers 0..11:
  - g = 0..5     : rotation by ``g · 60°``  (counter-clockwise)
  - g = 6..11    : reflection across an axis through vertex pair ``g − 6``,
                   followed by rotation by ``(g − 6) · 60°``.

Using the hex-axial rotation rule ``(q, r) → (-r, q + r)`` (one 60° step), we
build deterministic permutations of:

  - the 19 hex tile indices (``tile_perm[g][i]`` = tile index where tile ``i``
    lands after applying ``g``);
  - the 54 vertex / corner indices (``corner_perm[g]``);
  - the 72 edge indices (``edge_perm[g]``);
  - the within-tile vertex slot order (``within_tile_corner_perm[g]``;
    each tile carries 6 vertex feature slots that cyclically rotate or
    reverse with the rest of the board);
  - the within-tile edge slot order (``within_tile_edge_perm[g]``).

The tables are computed once on first access from a freshly-instantiated
``catanBoard``. They are deterministic given the board's static topology
(coordinates, vertex pixel positions, edge endpoints) — the random resource
shuffle does not affect them.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

# Public constants. ``D6_GROUP_SIZE = 12`` covers the rotations 0..5 followed
# by the same six rotations composed with a single mirror reflection.
D6_GROUP_SIZE: int = 12
D6_IDENTITY: int = 0


# ── Group structure ──────────────────────────────────────────────────────────


def rotation_steps(g: int) -> int:
    """Number of 60° counter-clockwise rotation steps in element ``g``.

    For the rotation half (``g < 6``) this is simply ``g``. For the
    reflection half (``g ∈ {6..11}``) the convention is *reflect first, then
    rotate by ``(g − 6) · 60°``*, which keeps the within-tile slot
    permutations consistent.
    """
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return g if g < 6 else g - 6


def is_reflection(g: int) -> bool:
    """``True`` iff ``g`` includes a reflection (``g`` in 6..11)."""
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    return g >= 6


def D6_INVERSE(g: int) -> int:
    """Return the index of the group inverse of ``g``.

    For pure rotations the inverse is ``(6 − g) % 6``. Reflections are
    self-inverse iff the rotation component is 0; for ``g = 6 + k`` with
    ``k > 0`` the inverse is the *same* reflection composed with the
    inverse rotation, which lands at ``g`` itself only when ``k = 0`` and
    otherwise at the element that undoes it. Easiest derivation: solve
    ``g · g_inv = identity`` over the cached composition table built once
    in ``_build_d6_inverse_table``.
    """
    return _build_d6_inverse_table()[g]


# ── Axial-coord rotation primitive ──────────────────────────────────────────


def _rotate_axial(q: int, r: int, k: int) -> tuple[int, int]:
    """Rotate hex axial coords ``k`` steps of 60° counter-clockwise."""
    k %= 6
    for _ in range(k):
        q, r = -r, q + r
    return q, r


def _reflect_axial(q: int, r: int) -> tuple[int, int]:
    """Reflect hex axial coords across the q-axis (a fixed mirror line).

    Equivalent to ``s ↔ r`` since ``s = -q - r``: produces ``(q, -q - r)``.
    All other D6 reflections compose this single mirror with a rotation.
    """
    return q, -q - r


def _apply_axial(g: int, q: int, r: int) -> tuple[int, int]:
    """Apply D6 element ``g`` to axial coordinates."""
    if is_reflection(g):
        q, r = _reflect_axial(q, r)
    return _rotate_axial(q, r, rotation_steps(g))


# ── Build perms from a temporary board ──────────────────────────────────────


@lru_cache(maxsize=1)
def _board_data() -> dict:
    """Build the static topology lookups we need from one ``catanBoard``.

    Resource shuffle is irrelevant — only positions matter. Cached so we pay
    the construction cost (≈ 50 ms) once per process.
    """
    # Local import to keep the module-import-time cheap (and avoid pulling in
    # pygame at module import time).
    from catan_rl.engine.board import catanBoard

    board = catanBoard()
    cx, cy = board.width / 2.0, board.height / 2.0

    # Tile axial coords keyed by index.
    tile_qr = {i: (board.hexTileDict[i].q, board.hexTileDict[i].r) for i in range(19)}
    qr_to_tile = {qr: i for i, qr in tile_qr.items()}

    # Vertex pixel centers, indexed by canonical vertex id.
    vertex_xy = {
        v_idx: (px.x - cx, px.y - cy) for v_idx, px in board.vertex_index_to_pixel_dict.items()
    }
    n_vertices = len(vertex_xy)

    # Edge list: deduplicated unordered pairs of vertex indices.
    edge_set: set[tuple[int, int]] = set()
    for v_px, v_obj in board.boardGraph.items():
        a = v_obj.vertex_index
        for nb_px in v_obj.neighbors:
            b = board.boardGraph[nb_px].vertex_index
            edge_set.add((min(a, b), max(a, b)))
    edges = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    # Each tile's six corner pixel positions, in their canonical 0..5 order
    # (geometry.HexCoordinates.get_corners). We store the vertex_index that
    # each canonical corner of each tile maps to.
    tile_corners_vidx: dict[int, list[int]] = {}
    for tile_i in range(19):
        tile = board.hexTileDict[tile_i]
        corners_px = tile.get_corners(board.flat)
        slot_to_vidx: list[int] = []
        for px in corners_px:
            v = board.boardGraph[px].vertex_index
            slot_to_vidx.append(v)
        tile_corners_vidx[tile_i] = slot_to_vidx

    return {
        "tile_qr": tile_qr,
        "qr_to_tile": qr_to_tile,
        "vertex_xy": vertex_xy,
        "n_vertices": n_vertices,
        "edges": edges,
        "edge_to_idx": edge_to_idx,
        "n_edges": n_edges,
        "tile_corners_vidx": tile_corners_vidx,
    }


def _rotate_pixel(x: float, y: float, theta_rad: float, *, mirror: bool) -> tuple[float, float]:
    """Rotate (and optionally pre-mirror) a centered pixel coordinate.

    The mirror is across the y-axis (i.e., ``x → -x``); composed with the
    rotation this yields all 6 reflection axes through hex vertex pairs.
    """
    if mirror:
        x = -x
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y


def _nearest_vertex_index(target_xy: tuple[float, float], vertex_xy: dict, eps: float = 1.0) -> int:
    """Find the vertex_index whose centered pixel is closest to ``target_xy``.

    Asserts the closest is within ``eps`` pixels — anything farther means the
    rotation produced a coordinate that doesn't correspond to any vertex,
    which is a bug.
    """
    best_idx, best_d2 = -1, float("inf")
    tx, ty = target_xy
    for v_idx, (vx, vy) in vertex_xy.items():
        d2 = (vx - tx) ** 2 + (vy - ty) ** 2
        if d2 < best_d2:
            best_d2, best_idx = d2, v_idx
    if best_d2 > eps * eps:
        raise RuntimeError(
            f"vertex mapping: nearest vertex {best_idx} is {math.sqrt(best_d2):.2f}px away "
            f"from target {target_xy}; expected exact match (eps={eps})"
        )
    return best_idx


@lru_cache(maxsize=1)
def _all_perms() -> dict:
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
        # ── Tile permutation via axial coordinates (exact) ──────────────
        tile_perm = np.zeros(19, dtype=np.int64)
        for i in range(19):
            q, r = tile_qr[i]
            new_q, new_r = _apply_axial(g, q, r)
            tile_perm[i] = qr_to_tile[(new_q, new_r)]
        tile_perms.append(tile_perm)

        # ── Vertex permutation via pixel rotation + nearest neighbor ────
        theta = math.radians(60.0 * rotation_steps(g))
        mirror = is_reflection(g)
        v_perm = np.zeros(n_vertices, dtype=np.int64)
        for v_idx, (vx, vy) in vertex_xy.items():
            new_xy = _rotate_pixel(vx, vy, theta, mirror=mirror)
            v_perm[v_idx] = _nearest_vertex_index(new_xy, vertex_xy)
        corner_perms.append(v_perm)

        # ── Edge permutation: each edge (a, b) maps to (v_perm[a], v_perm[b]) ─
        e_perm = np.zeros(n_edges, dtype=np.int64)
        for old_idx, (a, b) in enumerate(edges):
            na, nb = int(v_perm[a]), int(v_perm[b])
            new_e = (min(na, nb), max(na, nb))
            e_perm[old_idx] = edge_to_idx[new_e]
        edge_perms.append(e_perm)

        # ── Within-tile corner slot order ───────────────────────────────
        # Tile feature dims 19..54 hold 6 vertex feature blocks in canonical
        # corner order. After applying g, the new tile sitting at slot i
        # took the place of old tile ``inv_tile[i]``; its corners' positions
        # have themselves been permuted. We compute that permutation by
        # checking, for each canonical corner slot ``c`` of the rotated tile,
        # which canonical corner of the source tile maps to it.
        # The within-tile permutation is the same for every tile (board is
        # globally rigid under D6) — derive it from tile 0 (the center).
        center_corners = tile_corners_vidx[0]  # 6 vertex indices
        rotated_center_corners = [int(v_perm[v]) for v in center_corners]
        # New slot c at the rotated center should point to the v_idx that
        # the source slot ``slot_perm[c]`` originally held.
        slot_perm = np.zeros(6, dtype=np.int64)
        for new_slot in range(6):
            target_v = rotated_center_corners[new_slot]
            # Find the original slot in tile 0 that holds this vertex.
            try:
                slot_perm[new_slot] = center_corners.index(target_v)
            except ValueError as e:
                raise RuntimeError(
                    f"D6 element {g}: rotated center corner {new_slot} "
                    f"(v_idx={target_v}) not in original center corners "
                    f"{center_corners}"
                ) from e
        within_tile_c_perms.append(slot_perm)

        # Edges between adjacent corners: edge slot k joins corner slot k and
        # (k+1) % 6. Under within-tile slot permutation, edge slot k goes to
        # the edge between (slot_perm^-1[k]) and (slot_perm^-1[k+1]).
        # Easier: reuse the corner permutation — edge slot mapping is the
        # same cycle structure when we go between consecutive corners.
        # Compute by enumeration:
        edge_slot_perm = np.zeros(6, dtype=np.int64)
        # Reverse the slot permutation so we're answering "where did slot k go".
        inv_slot = np.argsort(slot_perm)
        for old_e_slot in range(6):
            a_old, b_old = old_e_slot, (old_e_slot + 1) % 6
            a_new, b_new = int(inv_slot[a_old]), int(inv_slot[b_old])
            # An edge slot is canonically the smaller corner index that
            # starts it (i.e., slot k spans corners k and k+1). Find which
            # new edge slot matches the unordered pair (a_new, b_new).
            for new_e_slot in range(6):
                pair = {new_e_slot, (new_e_slot + 1) % 6}
                if pair == {a_new, b_new}:
                    edge_slot_perm[old_e_slot] = new_e_slot
                    break
        within_tile_e_perms.append(edge_slot_perm)

    return {
        "tile": tile_perms,
        "corner": corner_perms,
        "edge": edge_perms,
        "within_corner": within_tile_c_perms,
        "within_edge": within_tile_e_perms,
    }


# ── Public accessors ────────────────────────────────────────────────────────


def tile_perm(g: int) -> np.ndarray:
    """Permutation of 19 hex tile indices under D6 element ``g``.

    Convention: ``tile_perm(g)[i]`` is the *new* tile index that the *old*
    tile ``i`` lands on after applying ``g``. Use ``np.argsort(tile_perm(g))``
    to invert (i.e., to ask "which old tile occupies new slot ``j``").
    """
    return _all_perms()["tile"][g].copy()


def corner_perm(g: int) -> np.ndarray:
    """Permutation of 54 vertex / corner action indices under ``g``."""
    return _all_perms()["corner"][g].copy()


def edge_perm(g: int) -> np.ndarray:
    """Permutation of 72 edge action indices under ``g``."""
    return _all_perms()["edge"][g].copy()


def within_tile_corner_perm(g: int) -> np.ndarray:
    """Permutation of the 6 within-tile vertex slots under ``g``."""
    return _all_perms()["within_corner"][g].copy()


def within_tile_edge_perm(g: int) -> np.ndarray:
    """Permutation of the 6 within-tile edge slots under ``g``."""
    return _all_perms()["within_edge"][g].copy()


# ── Group inverse table ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _build_d6_inverse_table() -> list[int]:
    """For each ``g`` find ``g_inv`` such that applying both yields identity.

    Uses the corner permutation as the discriminator: ``g_inv`` is the unique
    element whose corner permutation un-shuffles ``g``'s corner permutation.
    """
    perms = _all_perms()["corner"]
    n = D6_GROUP_SIZE
    inverse = [-1] * n
    for g in range(n):
        comp = perms[g]
        for h in range(n):
            # Composition: apply h then g. The final corner mapping is
            # ``perms[g][perms[h][i]]``. Identity if it equals ``i`` for all i.
            composed = comp[perms[h]]
            if np.array_equal(composed, np.arange(perms[g].shape[0])):
                inverse[g] = h
                break
        if inverse[g] == -1:
            raise RuntimeError(f"D6 inverse not found for element {g}")
    return inverse
