"""Tests for Phase 2.5b opponent-belief head."""

from __future__ import annotations

import torch

from catan_rl.models.belief_head import N_DEV_CARD_TYPES, BeliefHead


def test_belief_head_output_shape() -> None:
    """Logits shape: (B, 5)."""
    head = BeliefHead(input_dim=512)
    out = head(torch.randn(4, 512))
    assert out.shape == (4, N_DEV_CARD_TYPES)


def test_soft_cross_entropy_zero_when_target_matches_softmax() -> None:
    """When target == softmax(logits), loss equals entropy of that distribution.

    Soft CE is `-Σ p log q`. When `q = p` it reduces to `H(p)`. Since target
    is uniform here, expected loss = log(5).
    """
    logits = torch.zeros(2, 5)  # softmax → uniform 1/5
    target = torch.full((2, 5), 1.0 / 5)
    loss = BeliefHead.soft_cross_entropy(logits, target)
    expected = float(torch.tensor(5.0).log())
    assert abs(loss.item() - expected) < 1e-5


def test_soft_cross_entropy_grad_flows() -> None:
    """Gradient flows through logits."""
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.softmax(torch.randn(3, 5), dim=-1)
    loss = BeliefHead.soft_cross_entropy(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_soft_cross_entropy_lower_when_predict_target() -> None:
    """Predicting the target should lower the loss compared to uniform logits."""
    target = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1]])
    aligned_logits = target.log()  # softmax of log(p) recovers p
    misaligned = torch.zeros(1, 5)  # uniform
    aligned_loss = BeliefHead.soft_cross_entropy(aligned_logits, target).item()
    misaligned_loss = BeliefHead.soft_cross_entropy(misaligned, target).item()
    assert aligned_loss < misaligned_loss


def test_belief_head_init_near_uniform_predictions() -> None:
    """gain=0.01 final layer → output logits near zero → softmax near uniform.

    Helps training stability: until the encoder learns useful structure,
    the belief loss starts near `log(5)` instead of arbitrarily large.
    """
    torch.manual_seed(0)
    head = BeliefHead(input_dim=512)
    head.eval()
    logits = head(torch.randn(64, 512))
    probs = torch.softmax(logits, dim=-1)
    # Close to uniform: max prob shouldn't dominate.
    assert probs.max().item() < 0.5


def test_evaluate_actions_returns_belief_logits() -> None:
    """CatanPolicy.evaluate_actions(return_belief_logits=True) shape check."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy(use_belief_head=True)
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
    out = p.evaluate_actions(obs, masks, actions, return_belief_logits=True)
    assert len(out) == 4  # value, log_prob, entropy, belief_logits
    belief_logits = out[-1]
    assert belief_logits.shape == (B, N_DEV_CARD_TYPES)


def test_no_belief_head_when_flag_off() -> None:
    """Default policy has no belief head module."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy()
    assert p.use_belief_head is False
    assert p.belief_head is None
