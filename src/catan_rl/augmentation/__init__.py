"""D6 dihedral-group augmentation for the v2 Catan obs / action / mask schema.

The 19-tile hex board has the dihedral symmetry of the regular hexagon
(group D6, 12 elements = 6 rotations × 2 reflections). Permuting an
``(obs, action, mask)`` triple by a D6 element produces a strategically
equivalent state — augmentation multiplies effective training data by up to
12× without violating game-rule correctness.

This package exports:

  * :func:`tile_perm`, :func:`corner_perm`, :func:`edge_perm` — board-wide
    permutations of the 19 tiles, 54 vertices, and 72 edges.
  * :func:`within_tile_corner_perm`, :func:`within_tile_edge_perm` —
    per-tile reordering of the 6 corner / 6 edge feature blocks inside
    each ``tile_representations`` row.
  * :func:`apply_symmetry` — entry point that applies one D6 element to a
    full minibatch dict.
  * :func:`sample_d6_element` — uniform sampler over the non-identity
    elements (BC loader's default).
  * :data:`D6_GROUP_SIZE`, :data:`D6_IDENTITY`, :func:`D6_INVERSE`,
    :func:`is_reflection`, :func:`rotation_steps` — group-structure
    primitives.

Important invariant: every D6 transformation is applied to **both the obs
and the action together** through these tables. State-only augmentation
would teach inconsistent labels under the engine's deterministic
tiebreakers (faculty review of the BC plan, 2026-05-13). The loader is
required to call :func:`apply_symmetry` on the joint tuple, not on either
piece in isolation.
"""

from catan_rl.augmentation.dihedral import apply_symmetry, sample_d6_element
from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    D6_INVERSE,
    corner_perm,
    edge_perm,
    is_reflection,
    rotation_steps,
    tile_perm,
    within_tile_corner_perm,
    within_tile_edge_perm,
)

__all__ = [
    "D6_GROUP_SIZE",
    "D6_IDENTITY",
    "D6_INVERSE",
    "apply_symmetry",
    "corner_perm",
    "edge_perm",
    "is_reflection",
    "rotation_steps",
    "sample_d6_element",
    "tile_perm",
    "within_tile_corner_perm",
    "within_tile_edge_perm",
]
