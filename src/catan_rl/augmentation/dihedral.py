"""Apply D6 dihedral-group augmentation to a v2 (obs, action, mask) batch.

This is the runtime entry point used by:

  * The BC loader's ``__getitem__`` — augments single examples at training
    time (batch=1 is fine; the implementation broadcasts).
  * The PPO rollout buffer's minibatch sampler — augments full
    minibatches before each gradient step.

Permutations applied for a chosen D6 element ``g``:

  * ``tile_representations`` (B, 19, 79): per-tile slot permutation along
    axis 1 plus the within-tile corner-block (dims 19..55) and edge-block
    (dims 55..79) reordering along axis 2.
  * ``hex_features`` (B, 19, F) — v2 GNN input — tile-axis permutation only.
  * ``vertex_features`` (B, 54, F) — v2 GNN input — corner-axis permutation.
  * ``edge_features`` (B, 72, F) — v2 GNN input — edge-axis permutation.
  * Actions ``(B, 6)``: corner-head (index 1), edge-head (index 2),
    tile-head (index 3) remapped through the corresponding D6 perm. The
    type / resource1 / resource2 heads are spatially invariant.
  * Masks: ``corner_settlement``, ``corner_city`` along the corner axis;
    ``edge`` along the edge axis; ``tile`` along the tile axis. Type and
    resource masks pass through unchanged.

Per-player scalar feature vectors (``current_player_main``,
``next_player_main``), dev-card count vectors, opp-id scalars, and the
``belief_target`` carry no spatial axis and pass through unchanged.

Faculty correctness note (BC plan, 2026-05-13): augmentation **must
transform both state and action**. State-only augmentation under the
engine's deterministic tiebreakers would produce inconsistent labels
(``(T(s), a)`` is generally not a sample of the heuristic's policy).
This module's :func:`apply_symmetry` is the *only* sanctioned way to
augment; callers must not roll their own state-only transformer.
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

# Slice positions within the per-tile feature vector. Dims 0..18 are
# intrinsic (resource, number-token, has_robber, dot count) and do NOT
# permute. Dims 19..55 are 6 corner blocks of 6 dims each. Dims 55..79
# are 6 edge blocks of 4 dims each.
_TILE_INTRINSIC_DIM = 19
_TILE_CORNER_BLOCKS = 6
_TILE_CORNER_BLOCK_WIDTH = 6
_TILE_CORNER_START = 19
_TILE_CORNER_END = _TILE_CORNER_START + _TILE_CORNER_BLOCKS * _TILE_CORNER_BLOCK_WIDTH  # 55

_TILE_EDGE_BLOCKS = 6
_TILE_EDGE_BLOCK_WIDTH = 4
_TILE_EDGE_START = _TILE_CORNER_END  # 55
_TILE_EDGE_END = _TILE_EDGE_START + _TILE_EDGE_BLOCKS * _TILE_EDGE_BLOCK_WIDTH  # 79


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_d6_element(
    rng: np.random.Generator | None = None,
    *,
    exclude_identity: bool = True,
) -> int:
    """Sample a D6 element uniformly.

    Default excludes identity: when ``aug_prob`` fires in the loader, we
    want a *real* permutation. The no-augmentation case is handled by the
    caller's probability gate, not by drawing identity here.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if exclude_identity:
        return int(rng.integers(1, D6_GROUP_SIZE))
    return int(rng.integers(0, D6_GROUP_SIZE))


def _torch_perm(perm: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(perm).to(device=device, dtype=torch.long)


def _permute_axis(tensor: torch.Tensor, perm: np.ndarray, *, axis: int) -> torch.Tensor:
    """Permute one axis so the new position ``j`` holds old slot ``inv[j]``.

    Mirrors the action-head convention: ``perm[i] = j`` means "old slot
    ``i`` maps to new slot ``j``"; to reorder the tensor we ask "which old
    slot lives at new position ``j``?", which is ``np.argsort(perm)[j]``.
    """
    inv = np.argsort(perm)
    return tensor.index_select(dim=axis, index=_torch_perm(inv, tensor.device))


def _permute_action_indices(
    actions: torch.Tensor, perm: np.ndarray, *, head_idx: int
) -> torch.Tensor:
    """Map the integer index in ``actions[:, head_idx]`` through ``perm``.

    ``perm[i] = j`` means "the choice old-``i`` is now choice ``j``", so an
    action that originally selected ``i`` should now select ``perm[i]``.
    """
    perm_t = _torch_perm(perm, actions.device)
    actions = actions.clone()
    actions[:, head_idx] = perm_t[actions[:, head_idx].long()]
    return actions


def _permute_tile_features(tile_features: torch.Tensor, g: int) -> torch.Tensor:
    """Apply tile-slot + within-tile corner/edge perms to per-tile features.

    Args:
        tile_features: ``(B, 19, 79)`` float tensor. Last axis is the
            documented per-tile schema: intrinsic 0..18, 6 corner blocks
            of 6 dims each (19..54), 6 edge blocks of 4 dims each (55..78).
        g: D6 element 0..11.
    """
    if g == D6_IDENTITY:
        return tile_features

    if tile_features.dim() != 3:
        raise ValueError(
            f"tile_features must be (B, 19, 79); got shape {tuple(tile_features.shape)}"
        )
    _, n_tiles, n_dims = tile_features.shape
    if n_tiles != 19:
        raise ValueError(f"expected 19 tiles, got {n_tiles}")
    if n_dims != _TILE_EDGE_END:
        raise ValueError(f"expected {_TILE_EDGE_END}-dim per-tile feature, got {n_dims}")

    device = tile_features.device

    # 1) Tile-slot permutation.
    tile_p = tile_perm(g)
    out = _permute_axis(tile_features, tile_p, axis=1)

    # 2) Within-tile corner block reorder.
    corner_p = within_tile_corner_perm(g)
    inv_corner = np.argsort(corner_p)
    corner_dim_perm = (
        inv_corner[:, None] * _TILE_CORNER_BLOCK_WIDTH + np.arange(_TILE_CORNER_BLOCK_WIDTH)
    ).reshape(-1)
    corner_dim_perm_full = corner_dim_perm + _TILE_CORNER_START

    # 3) Within-tile edge block reorder.
    edge_p = within_tile_edge_perm(g)
    inv_edge = np.argsort(edge_p)
    edge_dim_perm = (
        inv_edge[:, None] * _TILE_EDGE_BLOCK_WIDTH + np.arange(_TILE_EDGE_BLOCK_WIDTH)
    ).reshape(-1)
    edge_dim_perm_full = edge_dim_perm + _TILE_EDGE_START

    # Compose into a single dim-axis permutation; intrinsic dims stay put.
    full_dim_perm = np.concatenate(
        [
            np.arange(_TILE_INTRINSIC_DIM),
            corner_dim_perm_full,
            edge_dim_perm_full,
        ]
    )
    return out.index_select(dim=2, index=_torch_perm(full_dim_perm, device))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_symmetry(
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    masks: dict[str, torch.Tensor],
    g: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Apply D6 element ``g`` to a v2 (obs, actions, masks) minibatch.

    Args:
        obs: dict of batched obs tensors. Spatial keys permuted:
            ``tile_representations``, ``hex_features``,
            ``vertex_features``, ``edge_features``. Non-spatial keys pass
            through (player main vectors, dev counts, opp-id, belief
            targets).
        actions: ``(B, 6)`` int tensor. Heads 1/2/3 (corner/edge/tile)
            remapped through the matching D6 perm; heads 0/4/5
            (type/res1/res2) are spatially invariant.
        masks: dict of batched mask tensors. ``corner_settlement``,
            ``corner_city`` along axis 1 (corner); ``edge`` along axis 1
            (edge); ``tile`` along axis 1 (tile). ``type`` and
            ``resource*`` masks are spatially invariant.
        g: D6 element 0..11. Identity short-circuits to a no-op (returns
            inputs unchanged).

    Returns:
        ``(new_obs, new_actions, new_masks)`` — fresh dicts and a fresh
        action tensor so the caller can safely mutate. Tensor data inside
        ``new_obs`` and ``new_masks`` references new memory created by
        ``torch.index_select``; the originals are not aliased.
    """
    if not 0 <= g < D6_GROUP_SIZE:
        raise ValueError(f"D6 element out of range: {g!r}")
    if g == D6_IDENTITY:
        return obs, actions, masks

    if actions.dim() != 2 or actions.shape[-1] != 6:
        raise ValueError(f"actions must be (B, 6) int tensor; got shape {tuple(actions.shape)}")

    corner_p = corner_perm(g)
    edge_p = edge_perm(g)
    tile_p = tile_perm(g)

    # ---- Observations -----------------------------------------------------
    new_obs = dict(obs)
    if "tile_representations" in obs:
        new_obs["tile_representations"] = _permute_tile_features(obs["tile_representations"], g)
    if "hex_features" in obs:
        new_obs["hex_features"] = _permute_axis(obs["hex_features"], tile_p, axis=1)
    if "vertex_features" in obs:
        new_obs["vertex_features"] = _permute_axis(obs["vertex_features"], corner_p, axis=1)
    if "edge_features" in obs:
        new_obs["edge_features"] = _permute_axis(obs["edge_features"], edge_p, axis=1)

    # ---- Actions ----------------------------------------------------------
    new_actions = _permute_action_indices(actions, corner_p, head_idx=1)
    new_actions = _permute_action_indices(new_actions, edge_p, head_idx=2)
    new_actions = _permute_action_indices(new_actions, tile_p, head_idx=3)

    # ---- Masks ------------------------------------------------------------
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
