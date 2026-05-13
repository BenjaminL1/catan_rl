"""Gates for v2 design Step 2: param count, forward pass, gradient flow.

The v2 design doc Step 2 states three falsifiable gates:

  1. Parameter count is ~1.5M (we accept 1.0M-2.0M; the design rationale is
     "smaller than v1's 2.74M, larger than Charlesworth's 1.2M").
  2. Forward pass produces valid masked log-probs (every sampled action is
     legal under its mask).
  3. Backward updates every parameter that is supposed to be trained.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.policy import CatanPolicy
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_EDGES,
    N_OPP_KINDS,
    N_OPP_POLICY_SLOTS,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    TILE_DIM,
    ActionType,
)


def _make_obs(batch_size: int, *, with_opp_id: bool = True) -> dict[str, torch.Tensor]:
    """Synthetic obs respecting the policy's expected shapes / dtypes."""
    rng = np.random.default_rng(0)
    obs: dict[str, torch.Tensor] = {
        "tile_representations": torch.from_numpy(
            rng.standard_normal((batch_size, N_TILES, TILE_DIM)).astype(np.float32)
        ),
        "current_player_main": torch.from_numpy(
            rng.standard_normal((batch_size, CURR_PLAYER_DIM)).astype(np.float32)
        ),
        "next_player_main": torch.from_numpy(
            rng.standard_normal((batch_size, NEXT_PLAYER_DIM)).astype(np.float32)
        ),
        "current_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (batch_size, N_DEV_TYPES)).astype(np.float32)
        ),
        "next_played_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (batch_size, N_DEV_TYPES)).astype(np.float32)
        ),
        # Graph-encoder inputs (vertex/edge feature dim defaults to 16).
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
    if with_opp_id:
        obs["opponent_kind"] = torch.from_numpy(
            rng.integers(0, N_OPP_KINDS, batch_size).astype(np.int64)
        )
        obs["opponent_policy_id"] = torch.from_numpy(
            rng.integers(0, N_OPP_POLICY_SLOTS, batch_size).astype(np.int64)
        )
    return obs


def _make_masks(batch_size: int, force_type: int | None = None) -> dict[str, torch.Tensor]:
    """All-ones masks (everything legal) except optionally force the type slot."""
    masks: dict[str, torch.Tensor] = {
        "type": torch.ones(batch_size, 13, dtype=torch.bool),
        "corner_settlement": torch.ones(batch_size, N_VERTICES, dtype=torch.bool),
        "corner_city": torch.ones(batch_size, N_VERTICES, dtype=torch.bool),
        "edge": torch.ones(batch_size, N_EDGES, dtype=torch.bool),
        "tile": torch.ones(batch_size, N_TILES, dtype=torch.bool),
        "resource1_trade": torch.ones(batch_size, N_RESOURCES, dtype=torch.bool),
        "resource1_discard": torch.ones(batch_size, N_RESOURCES, dtype=torch.bool),
        "resource1_default": torch.ones(batch_size, N_RESOURCES, dtype=torch.bool),
        "resource2_default": torch.ones(batch_size, N_RESOURCES, dtype=torch.bool),
    }
    if force_type is not None:
        masks["type"] = torch.zeros(batch_size, 13, dtype=torch.bool)
        masks["type"][:, force_type] = True
    return masks


# ---------------------------------------------------------------------------
# Gate 1: param count
# ---------------------------------------------------------------------------


def test_param_count_in_target_range() -> None:
    policy = CatanPolicy()
    n = policy.num_parameters()
    # Design doc target: ~1.5M. Hard band 1.0M-2.0M; warn band 1.2M-1.8M.
    assert 1_000_000 <= n <= 2_000_000, f"param count {n:,} outside 1.0M-2.0M target band"


def test_param_count_breakdown_smoke() -> None:
    """Spot-check the per-component param counts to catch silent regressions."""
    policy = CatanPolicy()
    counts = {
        name: sum(p.numel() for p in mod.parameters()) for name, mod in policy.named_children()
    }
    # The fusion + action heads should each be substantial; tile encoder smaller.
    assert counts["fusion"] > 100_000
    assert counts["action_heads"] > 100_000
    assert counts["value_head"] > 50_000


# ---------------------------------------------------------------------------
# Gate 2: forward pass + valid masked actions
# ---------------------------------------------------------------------------


def test_forward_returns_value_and_belief() -> None:
    policy = CatanPolicy().eval()
    obs = _make_obs(batch_size=4)
    out = policy(obs)
    assert out["value"].shape == (4,)
    assert out["belief_logits"].shape == (4, N_DEV_TYPES)
    assert torch.isfinite(out["value"]).all()
    assert torch.isfinite(out["belief_logits"]).all()


@pytest.mark.parametrize(
    "forced_type",
    [
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.END_TURN,
        ActionType.PLAY_YOP,
        ActionType.BANK_TRADE,
    ],
)
def test_sample_respects_type_mask(forced_type: int) -> None:
    policy = CatanPolicy().eval()
    torch.manual_seed(0)
    obs = _make_obs(batch_size=8)
    masks = _make_masks(8, force_type=forced_type)
    out = policy.sample(obs, masks)
    action = out["action"]
    assert action.shape == (8, 6)
    # Type head must respect the forced mask.
    assert (action[:, 0] == forced_type).all()
    # Log-prob must be finite (not log(0) from a degenerate softmax).
    assert torch.isfinite(out["log_prob"]).all()


def test_sample_respects_per_head_mask_specific() -> None:
    """When only some corner indices are legal, the sample must land there."""
    policy = CatanPolicy().eval()
    torch.manual_seed(1)
    obs = _make_obs(batch_size=16)
    masks = _make_masks(16, force_type=ActionType.BUILD_SETTLEMENT)
    corner_legal = {3, 17, 42}
    masks["corner_settlement"] = torch.zeros(16, N_VERTICES, dtype=torch.bool)
    for v in corner_legal:
        masks["corner_settlement"][:, v] = True
    out = policy.sample(obs, masks)
    assert set(out["action"][:, 1].tolist()) <= corner_legal


# ---------------------------------------------------------------------------
# Gate 3: gradient flow updates every (trainable) parameter
# ---------------------------------------------------------------------------


def test_backward_updates_every_parameter() -> None:
    policy = CatanPolicy()
    obs = _make_obs(batch_size=8)
    masks = _make_masks(8)
    # Spread the action types across the batch so every head gets relevance.
    action_types = torch.tensor(
        [
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_CITY,
            ActionType.BUILD_ROAD,
            ActionType.MOVE_ROBBER,
            ActionType.PLAY_YOP,
            ActionType.BANK_TRADE,
            ActionType.DISCARD,
            ActionType.END_TURN,
        ],
        dtype=torch.long,
    )
    action = torch.stack(
        [
            action_types,
            torch.full((8,), 3, dtype=torch.long),  # corner
            torch.full((8,), 5, dtype=torch.long),  # edge
            torch.full((8,), 7, dtype=torch.long),  # tile
            torch.full((8,), 2, dtype=torch.long),  # res1
            torch.full((8,), 4, dtype=torch.long),  # res2
        ],
        dim=-1,
    )
    out = policy.evaluate_actions(obs, action, masks)
    loss = (
        -out["log_prob"].mean()
        + 0.5 * out["value"].pow(2).mean()
        + 0.05
        * torch.nn.functional.cross_entropy(out["belief_logits"], torch.zeros(8, dtype=torch.long))
    )
    loss.backward()

    missing: list[str] = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or p.grad.abs().sum().item() == 0.0:
            missing.append(name)
    assert not missing, f"these parameters did not receive a non-zero gradient: {missing}"
