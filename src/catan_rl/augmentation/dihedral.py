"""Apply D6 dihedral-group augmentation to a (obs, action, masks) minibatch.

Phase 1.5 of the roadmap. Hooked into ``CompositeRolloutBuffer.get_batches``
so each yielded minibatch is, with probability ``aug_prob``, transformed by
a single non-identity D6 element. The transformation:

  - permutes the 19 hex *tile slots* in ``tile_representations``;
  - within each tile, cyclically (or reflectively) permutes the 6 vertex
    feature blocks (dims 19..54) and the 6 edge feature blocks (dims 55..78);
  - permutes the corner action axis (54 vertex slots) in ``corner_*`` masks
    and in ``actions[:, 1]``;
  - permutes the edge action axis (72 edge slots) in ``edge`` mask and
    ``actions[:, 2]``;
  - permutes the tile action axis (19 hex slots) in ``tile`` mask and
    ``actions[:, 3]``.

The resource one-hot, number-token one-hot, robber, dot count, dev cards,
phase flags, dice, and Karma fields are intrinsic to the tile / player and
do not move under D6 — only the geometric layout shifts.

Old log-probs stored in the buffer are kept unchanged. The PPO ratio formed
from those old log-probs and the *new* log-probs computed on the permuted
state is biased relative to a strictly equivariant policy, but that bias is
exactly what makes augmentation a useful regularizer (rather than a strict
invariance constraint that the network must satisfy).
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    corner_perm,
    edge_perm,
    tile_perm,
    within_tile_corner_perm,
    within_tile_edge_perm,
)

# Slice positions within the per-tile feature vector. Dims 0..18 are intrinsic
# (resource, number-token, has_robber, dot count) — they do NOT permute.
# Dims 19..54 are 6 corner blocks of 6 dims each. Dims 55..78 are 6 edge
# blocks of 4 dims each.
_TILE_INTRINSIC_DIM = 19
_TILE_CORNER_BLOCKS = 6
_TILE_CORNER_BLOCK_WIDTH = 6
_TILE_CORNER_START = 19
_TILE_CORNER_END = 19 + _TILE_CORNER_BLOCKS * _TILE_CORNER_BLOCK_WIDTH  # = 55

_TILE_EDGE_BLOCKS = 6
_TILE_EDGE_BLOCK_WIDTH = 4
_TILE_EDGE_START = _TILE_CORNER_END  # = 55
_TILE_EDGE_END = _TILE_EDGE_START + _TILE_EDGE_BLOCKS * _TILE_EDGE_BLOCK_WIDTH  # = 79


def sample_d6_element(
    rng: np.random.Generator | None = None, *, exclude_identity: bool = True
) -> int:
    """Sample a D6 element uniformly from {1..11} (or {0..11} if identity is allowed).

    Excluding identity by default keeps every augmented minibatch genuinely
    permuted; the no-augmentation case is handled by the caller's
    ``aug_prob`` gate, not by drawing identity.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if exclude_identity:
        return int(rng.integers(1, D6_GROUP_SIZE))
    return int(rng.integers(0, D6_GROUP_SIZE))


def _torch_perm(perm: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a numpy int permutation array to a long-dtype tensor on ``device``."""
    return torch.from_numpy(perm).to(device=device, dtype=torch.long)


def _permute_tile_features(
    tile_features: torch.Tensor,
    g: int,
) -> torch.Tensor:
    """Apply tile slot + within-tile corner/edge permutations to per-tile features.

    Args:
        tile_features: ``(B, 19, 79)`` float tensor. Last axis is the documented
            per-tile schema (intrinsic 0..18; 6 corner blocks 19..54;
            6 edge blocks 55..78).
        g: D6 element index 0..11.

    Returns:
        Permuted tensor with the same shape and dtype.
    """
    if g == D6_IDENTITY:
        return tile_features

    device = tile_features.device
    B, n_tiles, n_dims = tile_features.shape
    if n_tiles != 19:
        raise ValueError(f"expected 19 tiles, got {n_tiles}")
    if n_dims != _TILE_EDGE_END:
        raise ValueError(f"expected {_TILE_EDGE_END}-dim per-tile feature, got {n_dims}")

    # 1) Tile-slot permutation: tile that was at slot i moves to slot tile_perm(g)[i].
    #    To produce the new tensor in slot ``j``, we need the old tile whose new
    #    location is ``j``, i.e., index by argsort(tile_perm(g)).
    tile_p = tile_perm(g)
    inv_tile = np.argsort(tile_p)
    inv_tile_t = _torch_perm(inv_tile, device)
    out = tile_features.index_select(dim=1, index=inv_tile_t)

    # 2) Within-tile corner block reorder. Corner blocks span dims [19..55).
    corner_p = within_tile_corner_perm(g)
    inv_corner = np.argsort(corner_p)
    # Build the dim-axis permutation for the corner section by expanding the
    # 6-slot permutation over the per-slot block width (6 dims each).
    corner_dim_perm = (
        inv_corner[:, None] * _TILE_CORNER_BLOCK_WIDTH + np.arange(_TILE_CORNER_BLOCK_WIDTH)
    ).reshape(-1)  # length 36
    corner_dim_perm_full = corner_dim_perm + _TILE_CORNER_START

    # 3) Within-tile edge block reorder. Edge blocks span dims [55..79).
    edge_p = within_tile_edge_perm(g)
    inv_edge = np.argsort(edge_p)
    edge_dim_perm = (
        inv_edge[:, None] * _TILE_EDGE_BLOCK_WIDTH + np.arange(_TILE_EDGE_BLOCK_WIDTH)
    ).reshape(-1)  # length 24
    edge_dim_perm_full = edge_dim_perm + _TILE_EDGE_START

    # Compose into a single dim-axis permutation, leaving intrinsic dims fixed.
    full_dim_perm = np.concatenate(
        [
            np.arange(_TILE_INTRINSIC_DIM),
            corner_dim_perm_full,
            edge_dim_perm_full,
        ]
    )
    full_dim_perm_t = _torch_perm(full_dim_perm, device)
    return out.index_select(dim=2, index=full_dim_perm_t)


def _permute_axis(tensor: torch.Tensor, perm: np.ndarray, *, axis: int) -> torch.Tensor:
    """Permute one axis of ``tensor`` so new position ``j`` holds old slot ``inv[j]``.

    Mirrors the convention used by the action heads: ``perm[i]`` is where
    *old* slot ``i`` lands, so the slot now holding value-of-old-``i`` is
    ``perm[i]``. To reorder the tensor we use ``argsort(perm)`` to ask
    "which old slot is now at position j?".
    """
    inv = np.argsort(perm)
    return tensor.index_select(dim=axis, index=_torch_perm(inv, tensor.device))


def _permute_action_indices(
    actions: torch.Tensor, perm: np.ndarray, *, head_idx: int
) -> torch.Tensor:
    """Map the integer index in ``actions[:, head_idx]`` through ``perm``.

    ``perm[i] = j`` means "the choice old-``i`` is now choice ``j``". So an
    action that originally selected ``i`` should now select ``perm[i]``.
    """
    perm_t = _torch_perm(perm, actions.device)
    actions = actions.clone()
    actions[:, head_idx] = perm_t[actions[:, head_idx].long()]
    return actions


def apply_symmetry(
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    masks: dict[str, torch.Tensor],
    g: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Apply D6 element ``g`` to a minibatch.

    Args:
        obs: dict of batched obs tensors. ``tile_representations`` is permuted;
            ``current_player_main`` / ``next_player_main`` and the dev-card
            sequences carry no spatial axis and pass through unchanged.
        actions: ``(B, 6)`` int tensor (action heads: type, corner, edge,
            tile, resource1, resource2). Heads 1, 2, 3 are remapped through
            ``corner_perm(g)``, ``edge_perm(g)``, ``tile_perm(g)`` respectively.
            Type and resource heads are spatially invariant.
        masks: dict of batched mask tensors. ``corner_settlement``,
            ``corner_city`` are permuted along the corner axis;
            ``edge`` along the edge axis; ``tile`` along the tile axis.
            ``type``, ``resource1_*``, ``resource2_*`` pass through unchanged.
        g: D6 element index 0..11. ``0`` is identity (returns inputs unchanged).

    Returns:
        ``(new_obs, new_actions, new_masks)``.
    """
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    if g == D6_IDENTITY:
        return obs, actions, masks

    corner_p = corner_perm(g)
    edge_p = edge_perm(g)
    tile_p = tile_perm(g)

    # ── Observations ─────────────────────────────────────────────────────
    new_obs = dict(obs)
    if "tile_representations" in obs:
        new_obs["tile_representations"] = _permute_tile_features(obs["tile_representations"], g)

    # ── Actions ──────────────────────────────────────────────────────────
    new_actions = actions
    new_actions = _permute_action_indices(new_actions, corner_p, head_idx=1)
    new_actions = _permute_action_indices(new_actions, edge_p, head_idx=2)
    new_actions = _permute_action_indices(new_actions, tile_p, head_idx=3)

    # ── Masks ────────────────────────────────────────────────────────────
    new_masks = dict(masks)
    if "corner_settlement" in masks:
        new_masks["corner_settlement"] = _permute_axis(masks["corner_settlement"], corner_p, axis=1)
    if "corner_city" in masks:
        new_masks["corner_city"] = _permute_axis(masks["corner_city"], corner_p, axis=1)
    if "edge" in masks:
        new_masks["edge"] = _permute_axis(masks["edge"], edge_p, axis=1)
    if "tile" in masks:
        new_masks["tile"] = _permute_axis(masks["tile"], tile_p, axis=1)

    return new_obs, new_actions, new_masks
