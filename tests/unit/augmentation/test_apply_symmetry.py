"""Tests for ``apply_symmetry`` round-trip and shape preservation (Phase 1.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.augmentation import apply_symmetry
from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    D6_INVERSE,
    corner_perm,
    edge_perm,
    tile_perm,
)


def _fake_obs(B: int = 4) -> dict[str, torch.Tensor]:
    return {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "next_player_main": torch.randn(B, 173),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }


def _fake_masks(B: int = 4) -> dict[str, torch.Tensor]:
    return {
        "type": torch.ones(B, 13, dtype=torch.bool),
        "corner_settlement": torch.rand(B, 54) > 0.5,
        "corner_city": torch.rand(B, 54) > 0.5,
        "edge": torch.rand(B, 72) > 0.5,
        "tile": torch.rand(B, 19) > 0.5,
        "resource1_trade": torch.ones(B, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(B, 5, dtype=torch.bool),
        "resource1_default": torch.ones(B, 5, dtype=torch.bool),
        "resource2_default": torch.ones(B, 5, dtype=torch.bool),
    }


def _fake_actions(B: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(0)
    actions = np.zeros((B, 6), dtype=np.int64)
    actions[:, 0] = rng.integers(0, 13, size=B)  # type
    actions[:, 1] = rng.integers(0, 54, size=B)  # corner
    actions[:, 2] = rng.integers(0, 72, size=B)  # edge
    actions[:, 3] = rng.integers(0, 19, size=B)  # tile
    actions[:, 4] = rng.integers(0, 5, size=B)  # res1
    actions[:, 5] = rng.integers(0, 5, size=B)  # res2
    return torch.from_numpy(actions)


def test_identity_is_passthrough() -> None:
    """Applying the identity element returns the inputs untouched (up to identity)."""
    obs, masks = _fake_obs(), _fake_masks()
    actions = _fake_actions()
    new_obs, new_actions, new_masks = apply_symmetry(obs, actions, masks, D6_IDENTITY)
    torch.testing.assert_close(new_obs["tile_representations"], obs["tile_representations"])
    torch.testing.assert_close(new_actions, actions)
    for k in masks:
        torch.testing.assert_close(new_masks[k], masks[k])


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_round_trip_via_inverse(g: int) -> None:
    """Applying ``g`` then ``g_inv`` recovers the original obs/action/masks."""
    obs, masks = _fake_obs(), _fake_masks()
    actions = _fake_actions()
    g_inv = D6_INVERSE(g)
    obs1, actions1, masks1 = apply_symmetry(obs, actions, masks, g)
    obs2, actions2, masks2 = apply_symmetry(obs1, actions1, masks1, g_inv)
    torch.testing.assert_close(obs2["tile_representations"], obs["tile_representations"])
    torch.testing.assert_close(actions2, actions)
    for k in ("corner_settlement", "corner_city", "edge", "tile"):
        torch.testing.assert_close(masks2[k], masks[k])


def test_shape_preserved() -> None:
    """All shapes are preserved under any group element."""
    for g in range(D6_GROUP_SIZE):
        obs, masks = _fake_obs(), _fake_masks()
        actions = _fake_actions()
        new_obs, new_actions, new_masks = apply_symmetry(obs, actions, masks, g)
        for k, v in obs.items():
            assert new_obs[k].shape == v.shape, f"g={g} key={k}"
        assert new_actions.shape == actions.shape
        for k, v in masks.items():
            assert new_masks[k].shape == v.shape, f"g={g} mask {k}"


def test_action_indices_mapped_through_corner_perm() -> None:
    """The corner head action is remapped through ``corner_perm(g)``."""
    # Use a single-sample batch with a known action so we can verify the mapping.
    obs = _fake_obs(B=1)
    masks = _fake_masks(B=1)
    actions = torch.tensor([[0, 7, 11, 3, 1, 2]], dtype=torch.long)  # corner=7, edge=11, tile=3
    g = 1  # 60° rotation
    _new_obs, new_actions, _new_masks = apply_symmetry(obs, actions, masks, g)
    cp = corner_perm(g)
    ep = edge_perm(g)
    tp = tile_perm(g)
    assert int(new_actions[0, 1]) == int(cp[7])
    assert int(new_actions[0, 2]) == int(ep[11])
    assert int(new_actions[0, 3]) == int(tp[3])
    # Type and resource heads are spatially invariant.
    assert int(new_actions[0, 0]) == 0
    assert int(new_actions[0, 4]) == 1
    assert int(new_actions[0, 5]) == 2


def test_intrinsic_tile_features_unchanged_across_slots() -> None:
    """Resource/number/robber/dot dims (0..18) are intrinsic to the tile —
    they should follow the tile to its new slot but not change *within*."""
    obs = _fake_obs(B=1)
    actions = _fake_actions(B=1)
    masks = _fake_masks(B=1)
    g = 1
    new_obs, _, _ = apply_symmetry(obs, actions, masks, g)

    # For each old tile i, its intrinsic features (dims 0..18) should appear
    # at the new slot tile_perm(g)[i] in the rotated tensor.
    tp = tile_perm(g)
    for i in range(19):
        new_slot = int(tp[i])
        torch.testing.assert_close(
            new_obs["tile_representations"][0, new_slot, :19],
            obs["tile_representations"][0, i, :19],
        )


def test_player_features_pass_through() -> None:
    """current_player_main / next_player_main have no spatial axis to permute."""
    obs = _fake_obs()
    actions = _fake_actions()
    masks = _fake_masks()
    g = 5  # 300° rotation
    new_obs, _, _ = apply_symmetry(obs, actions, masks, g)
    torch.testing.assert_close(new_obs["current_player_main"], obs["current_player_main"])
    torch.testing.assert_close(new_obs["next_player_main"], obs["next_player_main"])
