"""Phase 0 per-head entropy logging contract.

The trainer needs per-head entropy diagnostics to detect silent collapse on
individual heads. This test verifies that ``CatanPolicy.evaluate_actions``
with ``return_per_head=True`` returns a dict with the documented shape.
"""

from __future__ import annotations

import torch

from catan_rl.models.action_heads_module import MultiActionHeads
from catan_rl.models.build_agent_model import build_agent_model
from catan_rl.models.utils import (
    CURR_PLAYER_DIM,
    MAX_DEV_SEQ,
    N_TILES,
    NEXT_PLAYER_DIM,
    OBS_TILE_DIM,
)


def _fake_batch(batch_size: int = 4):
    obs = {
        "tile_representations": torch.zeros(batch_size, N_TILES, OBS_TILE_DIM),
        "current_player_main": torch.zeros(batch_size, CURR_PLAYER_DIM),
        "next_player_main": torch.zeros(batch_size, NEXT_PLAYER_DIM),
        "current_player_hidden_dev": torch.zeros(batch_size, MAX_DEV_SEQ, dtype=torch.long),
        "current_player_played_dev": torch.zeros(batch_size, MAX_DEV_SEQ, dtype=torch.long),
        "next_player_played_dev": torch.zeros(batch_size, MAX_DEV_SEQ, dtype=torch.long),
    }
    masks = {
        "type": torch.ones(batch_size, 13, dtype=torch.bool),
        "corner_settlement": torch.ones(batch_size, 54, dtype=torch.bool),
        "corner_city": torch.ones(batch_size, 54, dtype=torch.bool),
        "edge": torch.ones(batch_size, 72, dtype=torch.bool),
        "tile": torch.ones(batch_size, 19, dtype=torch.bool),
        "resource1_trade": torch.ones(batch_size, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(batch_size, 5, dtype=torch.bool),
        "resource1_default": torch.ones(batch_size, 5, dtype=torch.bool),
        "resource2_default": torch.ones(batch_size, 5, dtype=torch.bool),
    }
    return obs, masks


def test_head_names_are_canonical() -> None:
    """Stable per-head identifiers; do not rename without updating tests + TB scalars."""
    assert MultiActionHeads.HEAD_NAMES == (
        "type",
        "corner",
        "edge",
        "tile",
        "resource1",
        "resource2",
    )


def test_evaluate_actions_returns_per_head_dict() -> None:
    """``return_per_head=True`` returns the documented dict shape."""
    policy = build_agent_model(device="cpu")
    obs, masks = _fake_batch(batch_size=4)
    # Sample some actions to evaluate.
    actions, _, _ = policy.act(obs, masks)
    value, log_prob, entropy, per_head = policy.evaluate_actions(
        obs, masks, actions, return_per_head=True
    )

    assert value.shape == (4, 1)
    assert log_prob.shape == (4,)
    assert entropy.dim() == 0  # scalar

    assert set(per_head.keys()) == set(MultiActionHeads.HEAD_NAMES)
    for name in MultiActionHeads.HEAD_NAMES:
        entry = per_head[name]
        assert {"entropy_per_sample", "weight", "log_prob"} <= set(entry.keys())
        assert entry["entropy_per_sample"].shape == (4,)
        assert entry["weight"].shape == (4,)
        assert entry["log_prob"].shape == (4,)


def test_evaluate_actions_default_unchanged() -> None:
    """Without ``return_per_head``, the API is unchanged (3-tuple)."""
    policy = build_agent_model(device="cpu")
    obs, masks = _fake_batch(batch_size=2)
    actions, _, _ = policy.act(obs, masks)
    out = policy.evaluate_actions(obs, masks, actions)
    assert len(out) == 3
    value, _log_prob, _entropy = out
    assert value.shape == (2, 1)


def test_type_head_weight_is_unconditional() -> None:
    """The type head is always relevant: weight==1 for every sample."""
    policy = build_agent_model(device="cpu")
    obs, masks = _fake_batch(batch_size=8)
    actions, _, _ = policy.act(obs, masks)
    _, _, _, per_head = policy.evaluate_actions(obs, masks, actions, return_per_head=True)
    type_weight = per_head["type"]["weight"].detach()
    assert torch.allclose(type_weight, torch.ones(8))
