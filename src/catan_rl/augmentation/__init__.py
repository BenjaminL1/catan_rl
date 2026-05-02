"""Phase 1.5: data augmentation under board symmetries.

The 1v1 standard Catan board has the dihedral symmetry of the regular hexagon
(group D6, 12 elements: 6 rotations × 2 reflections). Permuting an
``(obs, action, masks)`` tuple by a D6 element produces a strategically
equivalent state — we can use this to multiply effective training data.

This package provides:
  - ``symmetry_tables``: precomputed permutations of the 19 hex tile slots,
    the 54 vertex (corner) action slots, and the 72 edge action slots.
  - ``dihedral.apply_symmetry``: applies a chosen D6 element to a minibatch
    of observations, actions, and masks.

Phase 1.5 ships D6 only. The Z₂ player-swap augmentation is deferred to
Phase 2.5 (Option B / PPG aux phase) per the roadmap, since Z₂-swapped
samples are only valid under the value-distillation loss, not the policy
gradient.
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
