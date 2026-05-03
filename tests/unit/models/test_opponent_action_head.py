"""Tests for Phase 2.5c opponent-action auxiliary head."""

from __future__ import annotations

import torch

from catan_rl.models.opponent_action_head import N_ACTION_TYPES, OpponentActionHead


def test_opponent_action_head_output_shape() -> None:
    """Logits shape: (B, 13)."""
    head = OpponentActionHead(input_dim=512)
    out = head(torch.randn(4, 512))
    assert out.shape == (4, N_ACTION_TYPES)


def test_masked_cross_entropy_returns_none_when_no_valid() -> None:
    """All-False mask → None (trainer skips the loss term)."""
    logits = torch.randn(4, N_ACTION_TYPES)
    target = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros(4, dtype=torch.bool)
    assert OpponentActionHead.masked_cross_entropy(logits, target, mask) is None


def test_masked_cross_entropy_only_uses_valid_rows() -> None:
    """Valid rows produce the loss; invalid rows are silently dropped.

    We construct a batch where only one row has a target and verify the
    loss equals the per-row CE on that single row.
    """
    torch.manual_seed(0)
    logits = torch.randn(3, N_ACTION_TYPES)
    target = torch.tensor([5, 99, 0], dtype=torch.long)  # 99 is out-of-range; would crash
    mask = torch.tensor([True, False, False])
    loss = OpponentActionHead.masked_cross_entropy(logits, target, mask)
    expected = torch.nn.functional.cross_entropy(logits[:1], target[:1])
    assert loss is not None
    torch.testing.assert_close(loss, expected)


def test_masked_cross_entropy_clamps_out_of_range() -> None:
    """Even on a valid row, an out-of-range target gets clamped (defensive)."""
    logits = torch.randn(2, N_ACTION_TYPES)
    target = torch.tensor([99, -5], dtype=torch.long)
    mask = torch.tensor([True, True])
    loss = OpponentActionHead.masked_cross_entropy(logits, target, mask)
    assert loss is not None
    # No exception, loss is a finite scalar.
    assert torch.isfinite(loss).item()


def test_init_near_uniform_predictions() -> None:
    """gain=0.01 final layer keeps initial predictions near uniform.

    Loss starts at log(13) ≈ 2.565 — well-defined and shared across
    initializations, so we don't see wild swings on the loss curve at
    step 0.
    """
    torch.manual_seed(0)
    head = OpponentActionHead(input_dim=512)
    head.eval()
    logits = head(torch.randn(64, 512))
    probs = torch.softmax(logits, dim=-1)
    # No single class should dominate at init.
    assert probs.max().item() < 0.5


def test_evaluate_actions_returns_opp_action_logits() -> None:
    """``return_opp_action_logits=True`` appends logits to the return tuple."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy(use_opponent_action_head=True)
    p.eval()
    B = 2
    obs = {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_main": torch.randn(B, 173),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }
    masks = {
        "type": torch.ones(B, 13, dtype=torch.bool),
        "corner_settlement": torch.ones(B, 54, dtype=torch.bool),
        "corner_city": torch.ones(B, 54, dtype=torch.bool),
        "edge": torch.ones(B, 72, dtype=torch.bool),
        "tile": torch.ones(B, 19, dtype=torch.bool),
        "resource1_default": torch.ones(B, 5, dtype=torch.bool),
        "resource1_trade": torch.ones(B, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(B, 5, dtype=torch.bool),
        "resource2_default": torch.ones(B, 5, dtype=torch.bool),
    }
    actions = torch.zeros(B, 6, dtype=torch.long)
    out = p.evaluate_actions(obs, masks, actions, return_opp_action_logits=True)
    assert len(out) == 4  # value, log_prob, entropy, opp_action_logits
    opp_logits = out[-1]
    assert opp_logits.shape == (B, N_ACTION_TYPES)


def test_belief_and_opp_action_compose() -> None:
    """Both aux heads can be returned simultaneously, in stable order."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy(use_belief_head=True, use_opponent_action_head=True)
    p.eval()
    B = 2
    obs = {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_main": torch.randn(B, 173),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }
    masks = {
        "type": torch.ones(B, 13, dtype=torch.bool),
        "corner_settlement": torch.ones(B, 54, dtype=torch.bool),
        "corner_city": torch.ones(B, 54, dtype=torch.bool),
        "edge": torch.ones(B, 72, dtype=torch.bool),
        "tile": torch.ones(B, 19, dtype=torch.bool),
        "resource1_default": torch.ones(B, 5, dtype=torch.bool),
        "resource1_trade": torch.ones(B, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(B, 5, dtype=torch.bool),
        "resource2_default": torch.ones(B, 5, dtype=torch.bool),
    }
    actions = torch.zeros(B, 6, dtype=torch.long)
    out = p.evaluate_actions(
        obs, masks, actions, return_belief_logits=True, return_opp_action_logits=True
    )
    # value, log_prob, entropy, belief_logits, opp_action_logits
    assert len(out) == 5
    assert out[3].shape == (B, 5)  # belief
    assert out[4].shape == (B, N_ACTION_TYPES)  # opp action


def test_no_opp_action_head_when_flag_off() -> None:
    """Default policy has no opponent-action head module."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy()
    assert p.use_opponent_action_head is False
    assert p.opponent_action_head is None
