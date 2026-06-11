"""Unit tests for apply_symmetry — the runtime entry point.

The most important contract: applying ``g`` then ``D6_INVERSE(g)`` to a
(obs, action, mask) triple recovers the original. This catches subtle bugs
in axis numbering, inverse-vs-forward convention, and within-tile slot
composition that the table-level tests can't see.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.augmentation import apply_symmetry, sample_d6_element
from catan_rl.augmentation.dihedral import _permute_tile_features
from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    D6_INVERSE,
    corner_perm,
    edge_perm,
    tile_perm,
)
from catan_rl.policy.obs_schema import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
)

# Synthetic obs / action / mask builders ------------------------------------


def _make_obs(batch_size: int, seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return {
        "tile_representations": torch.from_numpy(
            rng.standard_normal((batch_size, 19, 79)).astype(np.float32)
        ),
        "current_player_main": torch.from_numpy(
            rng.standard_normal((batch_size, 54)).astype(np.float32)
        ),
        "next_player_main": torch.from_numpy(
            rng.standard_normal((batch_size, 61)).astype(np.float32)
        ),
        "current_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (batch_size, 5)).astype(np.float32)
        ),
        "next_played_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (batch_size, 5)).astype(np.float32)
        ),
        "hex_features": torch.from_numpy(
            rng.standard_normal((batch_size, N_TILES, 19)).astype(np.float32)
        ),
        "vertex_features": torch.from_numpy(
            rng.standard_normal((batch_size, N_VERTICES, 16)).astype(np.float32)
        ),
        "edge_features": torch.from_numpy(
            rng.standard_normal((batch_size, N_EDGES, 16)).astype(np.float32)
        ),
    }


def _make_actions(batch_size: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a = np.zeros((batch_size, 6), dtype=np.int64)
    a[:, 0] = rng.integers(0, 13, batch_size)
    a[:, 1] = rng.integers(0, 54, batch_size)
    a[:, 2] = rng.integers(0, 72, batch_size)
    a[:, 3] = rng.integers(0, 19, batch_size)
    a[:, 4] = rng.integers(0, 5, batch_size)
    a[:, 5] = rng.integers(0, 5, batch_size)
    return torch.from_numpy(a)


def _make_masks(batch_size: int, seed: int = 2) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return {
        "type": torch.from_numpy(rng.integers(0, 2, (batch_size, 13)).astype(bool)),
        "corner_settlement": torch.from_numpy(rng.integers(0, 2, (batch_size, 54)).astype(bool)),
        "corner_city": torch.from_numpy(rng.integers(0, 2, (batch_size, 54)).astype(bool)),
        "edge": torch.from_numpy(rng.integers(0, 2, (batch_size, 72)).astype(bool)),
        "tile": torch.from_numpy(rng.integers(0, 2, (batch_size, 19)).astype(bool)),
        "resource1_trade": torch.from_numpy(rng.integers(0, 2, (batch_size, 5)).astype(bool)),
        "resource1_discard": torch.from_numpy(rng.integers(0, 2, (batch_size, 5)).astype(bool)),
        "resource1_default": torch.from_numpy(rng.integers(0, 2, (batch_size, 5)).astype(bool)),
        "resource2_default": torch.from_numpy(rng.integers(0, 2, (batch_size, 5)).astype(bool)),
    }


# ---------------------------------------------------------------------------
# Identity passes through
# ---------------------------------------------------------------------------


def test_identity_is_noop() -> None:
    obs, actions, masks = _make_obs(2), _make_actions(2), _make_masks(2)
    new_obs, new_actions, new_masks = apply_symmetry(obs, actions, masks, D6_IDENTITY)
    assert new_obs is obs  # identity short-circuit returns same dict
    assert torch.equal(new_actions, actions)
    assert new_masks is masks


# ---------------------------------------------------------------------------
# Round-trip — g then inverse(g) recovers original
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_round_trip_recovers_obs(g: int) -> None:
    obs = _make_obs(3)
    actions = _make_actions(3)
    masks = _make_masks(3)
    new_obs, new_actions, new_masks = apply_symmetry(obs, actions, masks, g)
    back_obs, back_actions, back_masks = apply_symmetry(
        new_obs, new_actions, new_masks, D6_INVERSE(g)
    )

    # Spatial obs keys must round-trip
    for key in ("tile_representations", "hex_features", "vertex_features", "edge_features"):
        assert torch.allclose(back_obs[key], obs[key], atol=1e-6), (
            f"round-trip failed for obs[{key}] under D6 element {g}"
        )
    # Non-spatial obs passes through unchanged on both legs
    for key in (
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
    ):
        assert torch.equal(back_obs[key], obs[key])

    # Actions round-trip exactly (integer indices)
    assert torch.equal(back_actions, actions), (
        f"action round-trip failed under D6 element {g}: "
        f"original={actions.tolist()}, recovered={back_actions.tolist()}"
    )

    # Masks round-trip on spatial keys
    for key in ("corner_settlement", "corner_city", "edge", "tile"):
        assert torch.equal(back_masks[key], masks[key]), (
            f"mask round-trip failed for {key} under D6 element {g}"
        )


# ---------------------------------------------------------------------------
# State + action consistency — the critical correctness property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_action_legal_under_mask_stays_legal_after_aug(g: int) -> None:
    """If the agent's action was legal under the original mask, it must remain
    legal under the augmented mask. This is the joint-transform invariant
    that makes augmentation a valid supervision signal."""
    # Construct deterministic batch where each row's action is legal.
    obs = _make_obs(4, seed=10)
    actions = torch.zeros((4, 6), dtype=torch.int64)
    masks = {
        "type": torch.zeros((4, 13), dtype=torch.bool),
        "corner_settlement": torch.zeros((4, 54), dtype=torch.bool),
        "corner_city": torch.zeros((4, 54), dtype=torch.bool),
        "edge": torch.zeros((4, 72), dtype=torch.bool),
        "tile": torch.zeros((4, 19), dtype=torch.bool),
        "resource1_trade": torch.ones((4, 5), dtype=torch.bool),
        "resource1_discard": torch.ones((4, 5), dtype=torch.bool),
        "resource1_default": torch.ones((4, 5), dtype=torch.bool),
        "resource2_default": torch.ones((4, 5), dtype=torch.bool),
    }
    # Row 0: BuildSettlement at corner 17. Row 1: BuildRoad at edge 33.
    # Row 2: MoveRobber at tile 7. Row 3: BuildCity at corner 42.
    actions[0] = torch.tensor([0, 17, 0, 0, 0, 0])
    masks["type"][0, 0] = True
    masks["corner_settlement"][0, 17] = True

    actions[1] = torch.tensor([2, 0, 33, 0, 0, 0])
    masks["type"][1, 2] = True
    masks["edge"][1, 33] = True

    actions[2] = torch.tensor([4, 0, 0, 7, 0, 0])
    masks["type"][2, 4] = True
    masks["tile"][2, 7] = True

    actions[3] = torch.tensor([1, 42, 0, 0, 0, 0])
    masks["type"][3, 1] = True
    masks["corner_city"][3, 42] = True

    new_obs, new_actions, new_masks = apply_symmetry(obs, actions, masks, g)
    del new_obs

    # Row 0: settle at new corner; corner_settlement mask must still be True there.
    assert new_masks["corner_settlement"][0, int(new_actions[0, 1])]
    # Row 1: road at new edge.
    assert new_masks["edge"][1, int(new_actions[1, 2])]
    # Row 2: robber at new tile.
    assert new_masks["tile"][2, int(new_actions[2, 3])]
    # Row 3: city at new corner.
    assert new_masks["corner_city"][3, int(new_actions[3, 1])]


# ---------------------------------------------------------------------------
# Non-spatial obs / masks / action heads pass through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_non_spatial_obs_keys_unchanged(g: int) -> None:
    obs = _make_obs(2)
    new_obs, _, _ = apply_symmetry(obs, _make_actions(2), _make_masks(2), g)
    for key in (
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
    ):
        assert torch.equal(new_obs[key], obs[key])


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_non_spatial_masks_unchanged(g: int) -> None:
    masks = _make_masks(2)
    _, _, new_masks = apply_symmetry(_make_obs(2), _make_actions(2), masks, g)
    for key in (
        "type",
        "resource1_trade",
        "resource1_discard",
        "resource1_default",
        "resource2_default",
    ):
        assert torch.equal(new_masks[key], masks[key])


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_non_spatial_action_heads_unchanged(g: int) -> None:
    actions = _make_actions(4)
    _, new_actions, _ = apply_symmetry(_make_obs(4), actions, _make_masks(4), g)
    # Type, res1, res2 heads (0, 4, 5) are spatially invariant.
    assert torch.equal(new_actions[:, 0], actions[:, 0])
    assert torch.equal(new_actions[:, 4], actions[:, 4])
    assert torch.equal(new_actions[:, 5], actions[:, 5])


# ---------------------------------------------------------------------------
# Spatial-axis permutation matches the table for each individual obs key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_vertex_features_permuted_by_corner_perm(g: int) -> None:
    """vertex_features axis 1 should be reindexed by argsort(corner_perm(g))."""
    obs = _make_obs(1, seed=42)
    new_obs, _, _ = apply_symmetry(obs, _make_actions(1), _make_masks(1), g)
    inv = np.argsort(corner_perm(g))
    expected = obs["vertex_features"][:, inv, :]
    assert torch.equal(new_obs["vertex_features"], expected)


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_edge_features_permuted_by_edge_perm(g: int) -> None:
    obs = _make_obs(1, seed=43)
    new_obs, _, _ = apply_symmetry(obs, _make_actions(1), _make_masks(1), g)
    inv = np.argsort(edge_perm(g))
    expected = obs["edge_features"][:, inv, :]
    assert torch.equal(new_obs["edge_features"], expected)


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_hex_features_permuted_by_tile_perm(g: int) -> None:
    obs = _make_obs(1, seed=44)
    new_obs, _, _ = apply_symmetry(obs, _make_actions(1), _make_masks(1), g)
    inv = np.argsort(tile_perm(g))
    expected = obs["hex_features"][:, inv, :]
    assert torch.equal(new_obs["hex_features"], expected)


# ---------------------------------------------------------------------------
# Tile-features within-tile reorder is invertible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
def test_tile_features_within_block_roundtrip(g: int) -> None:
    """Forward then inverse on the tile features must recover the original."""
    obs = _make_obs(1, seed=100)
    forward = _permute_tile_features(obs["tile_representations"], g)
    back = _permute_tile_features(forward, D6_INVERSE(g))
    assert torch.allclose(back, obs["tile_representations"], atol=1e-6)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def test_sample_d6_element_excludes_identity_by_default() -> None:
    rng = np.random.default_rng(0)
    samples = {sample_d6_element(rng) for _ in range(200)}
    assert 0 not in samples
    assert samples <= set(range(1, D6_GROUP_SIZE))


def test_sample_d6_element_can_include_identity() -> None:
    rng = np.random.default_rng(0)
    samples = {sample_d6_element(rng, exclude_identity=False) for _ in range(500)}
    assert 0 in samples


# ---------------------------------------------------------------------------
# Bad inputs raise
# ---------------------------------------------------------------------------


def test_apply_symmetry_rejects_bad_g() -> None:
    obs = _make_obs(1)
    a = _make_actions(1)
    m = _make_masks(1)
    with pytest.raises(ValueError):
        apply_symmetry(obs, a, m, -1)
    with pytest.raises(ValueError):
        apply_symmetry(obs, a, m, D6_GROUP_SIZE)


def test_apply_symmetry_rejects_bad_actions_shape() -> None:
    obs = _make_obs(1)
    m = _make_masks(1)
    with pytest.raises(ValueError):
        apply_symmetry(obs, torch.zeros((1, 5), dtype=torch.int64), m, 1)
    with pytest.raises(ValueError):
        apply_symmetry(obs, torch.zeros((6,), dtype=torch.int64), m, 1)


def test_apply_symmetry_rejects_bad_tile_features_shape() -> None:
    obs = _make_obs(1)
    obs["tile_representations"] = torch.zeros((1, 19, 78), dtype=torch.float32)
    with pytest.raises(ValueError):
        apply_symmetry(obs, _make_actions(1), _make_masks(1), 1)


def test_within_tile_edge_block_follows_corner_block() -> None:
    """Edges are rigidly attached to corners: edge slot e sits between corner
    slots e and (e+1)%6, so after ANY D6 transform the output edge at slot s must
    be the original edge that joined the two original corners now occupying output
    slots s and s+1. This pins ABSOLUTE within-tile edge geometry against the
    (independently test-pinned) corner block — a round-trip test cannot catch a
    directional inversion, which is how the edge-block argsort inversion hid.
    """
    from catan_rl.augmentation import dihedral as D

    cs, cw = D._TILE_CORNER_START, D._TILE_CORNER_BLOCK_WIDTH
    es, ew = D._TILE_EDGE_START, D._TILE_EDGE_BLOCK_WIDTH
    feats = torch.zeros((1, 19, 79), dtype=torch.float32)
    for t in range(19):
        for c in range(6):
            feats[0, t, cs + c * cw] = c + 1  # corner slot identity marker
        for e in range(6):
            feats[0, t, es + e * ew] = e + 1  # edge slot identity marker

    def old_edge_joining(a: int, b: int) -> int:
        if (a + 1) % 6 == b:
            return a
        if (b + 1) % 6 == a:
            return b
        raise AssertionError(f"corners {a},{b} are not tile-adjacent")

    for g in range(12):
        out = D._permute_tile_features(feats, g)
        cmark = [int(out[0, 0, cs + s * cw].item()) - 1 for s in range(6)]
        emark = [int(out[0, 0, es + s * ew].item()) - 1 for s in range(6)]
        for s in range(6):
            expected = old_edge_joining(cmark[s], cmark[(s + 1) % 6])
            assert emark[s] == expected, (
                f"g={g}: output edge slot {s} holds old edge {emark[s]}, but the "
                f"corners now there ({cmark[s]},{cmark[(s + 1) % 6]}) joined old edge "
                f"{expected} — edge block out of sync with corner block"
            )
