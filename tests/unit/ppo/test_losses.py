"""Tests for `ppo/losses.py`."""

from __future__ import annotations

import math

import pytest
import torch

from catan_rl.ppo.losses import (
    compute_belief_loss,
    compute_entropy_bonus,
    compute_kl_approximation,
    compute_policy_loss,
    compute_value_loss,
)

# ---------------------------------------------------------------------------
# Policy loss — clipped surrogate
# ---------------------------------------------------------------------------


class TestPolicyLoss:
    def test_ratio_one_is_negative_advantage_mean(self) -> None:
        # If new_logp == old_logp → ratio == 1 → loss = -E[advantages]
        new = torch.zeros(8)
        old = torch.zeros(8)
        adv = torch.tensor([1.0, 2.0, -1.0, 0.5, 0.0, -2.0, 1.5, -0.5])
        loss, stats = compute_policy_loss(
            new_log_prob=new, old_log_prob=old, advantages=adv, clip_range=0.2
        )
        assert loss.item() == pytest.approx(-adv.mean().item(), abs=1e-6)
        assert stats["clip_frac"].item() == 0.0
        assert stats["ratio_mean"].item() == pytest.approx(1.0)

    def test_clip_caps_positive_advantage_loss(self) -> None:
        # Large new_logp boost: ratio = e^1 ≈ 2.72. With clip=0.2,
        # the clipped surrogate is (1.2 * adv); the min of (2.72*adv, 1.2*adv)
        # for positive adv is 1.2*adv, so loss = -1.2 * E[adv].
        new = torch.ones(4)
        old = torch.zeros(4)
        adv = torch.ones(4)  # all positive
        loss, _ = compute_policy_loss(
            new_log_prob=new, old_log_prob=old, advantages=adv, clip_range=0.2
        )
        assert loss.item() == pytest.approx(-1.2, abs=1e-5)

    def test_clip_does_not_cap_negative_advantage_loss(self) -> None:
        # For negative advantage with ratio > 1, the *unclipped* arm is
        # smaller (more negative) so PPO uses it, NOT the clipped one.
        # The whole point of taking the min: don't let the policy escape
        # bad updates via the clip.
        new = torch.ones(4)
        old = torch.zeros(4)
        adv = torch.full((4,), -1.0)
        loss, _ = compute_policy_loss(
            new_log_prob=new, old_log_prob=old, advantages=adv, clip_range=0.2
        )
        # unclipped = e * -1 ≈ -2.72; clipped = 1.2 * -1 = -1.2
        # min(-2.72, -1.2) = -2.72; loss = +2.72
        assert loss.item() == pytest.approx(math.e, abs=1e-3)

    def test_clip_frac_counts_correctly(self) -> None:
        # Half the batch has ratio inside the clip; half outside.
        new = torch.tensor([0.0, 0.0, 1.0, 1.0])
        old = torch.zeros(4)
        adv = torch.ones(4)
        _, stats = compute_policy_loss(
            new_log_prob=new, old_log_prob=old, advantages=adv, clip_range=0.2
        )
        assert stats["clip_frac"].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Value loss — PPO2-style
# ---------------------------------------------------------------------------


class TestValueLoss:
    def test_unclipped_is_half_mse(self) -> None:
        new = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([0.0, 0.0, 0.0])
        loss, _ = compute_value_loss(
            new_value=new,
            old_value=new,
            returns=returns,
            clip_range_vf=0.2,
            use_value_clipping=False,
        )
        # 0.5 * mean(1^2 + 2^2 + 3^2) = 0.5 * 14/3
        assert loss.item() == pytest.approx(0.5 * 14 / 3, abs=1e-5)

    def test_clipped_uses_max_branch(self) -> None:
        # When the value head moves far from old_v in the WRONG direction
        # (relative to returns), the clipped branch should be selected by
        # max(MSE_unclipped, MSE_clipped) and prevent the loss from
        # benefiting from the clip.
        old = torch.tensor([0.0])
        # new_value moves toward returns; without the clip the loss would
        # be smaller. With clipping, the max() picks the larger
        # mse_clipped (because v_clipped is closer to the bad old_v),
        # so the loss is BIGGER than the unclipped version would suggest.
        new = torch.tensor([2.0])  # change of +2.0 vs old=0
        returns = torch.tensor([3.0])
        loss_clipped, stats = compute_value_loss(
            new_value=new,
            old_value=old,
            returns=returns,
            clip_range_vf=0.2,
            use_value_clipping=True,
        )
        loss_unclipped, _ = compute_value_loss(
            new_value=new,
            old_value=old,
            returns=returns,
            clip_range_vf=0.2,
            use_value_clipping=False,
        )
        assert loss_clipped.item() >= loss_unclipped.item()
        # v_clipped = 0 + clip(2, -0.2, 0.2) = 0.2 → mse_clipped = (0.2-3)^2 = 7.84
        # mse_unclipped = (2-3)^2 = 1.0 → max = 7.84 → loss = 0.5 * 7.84
        assert loss_clipped.item() == pytest.approx(0.5 * 7.84, abs=1e-5)
        assert "value_loss_clipped" in stats


# ---------------------------------------------------------------------------
# KL approximations
# ---------------------------------------------------------------------------


class TestKL:
    def test_k1_is_negative_log_ratio_mean(self) -> None:
        new = torch.tensor([0.1, 0.2])
        old = torch.tensor([0.0, 0.0])
        kl = compute_kl_approximation(new_log_prob=new, old_log_prob=old, estimator="k1")
        # k1 = -mean(new - old) = -0.15
        assert kl.item() == pytest.approx(-0.15, abs=1e-5)

    def test_k3_non_negative(self) -> None:
        new = torch.tensor([0.1, -0.1, 0.5, -0.5])
        old = torch.tensor([0.0, 0.0, 0.0, 0.0])
        kl = compute_kl_approximation(new_log_prob=new, old_log_prob=old, estimator="k3")
        assert kl.item() >= 0.0

    def test_k2_non_negative(self) -> None:
        new = torch.tensor([0.1, -0.1])
        old = torch.tensor([0.0, 0.0])
        kl = compute_kl_approximation(new_log_prob=new, old_log_prob=old, estimator="k2")
        assert kl.item() >= 0.0

    def test_unknown_estimator_raises(self) -> None:
        with pytest.raises(ValueError, match="k1/k2/k3"):
            compute_kl_approximation(
                new_log_prob=torch.zeros(1),
                old_log_prob=torch.zeros(1),
                estimator="k4",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Aux
# ---------------------------------------------------------------------------


class TestEntropyBonus:
    def test_passthrough_mean(self) -> None:
        ent = torch.tensor([1.0, 2.0, 3.0])
        assert compute_entropy_bonus(joint_entropy=ent).item() == pytest.approx(2.0)


class TestBeliefLoss:
    def test_one_hot_target_matches_cross_entropy(self) -> None:
        # Soft CE with a one-hot target == standard cross-entropy.
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.0, 1.0, 0.0]])
        soft = compute_belief_loss(belief_logits=logits, belief_target=target)
        std = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        assert soft.item() == pytest.approx(std.item(), abs=1e-5)

    def test_uniform_target_matches_logsumexp_minus_mean(self) -> None:
        # E[-log_softmax] under uniform target = entropy of softmax
        # under uniform — equivalent to logsumexp - log(K).
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.full((1, 3), 1 / 3)
        loss = compute_belief_loss(belief_logits=logits, belief_target=target)
        log_q = torch.nn.functional.log_softmax(logits, dim=-1)
        expected = -(target * log_q).sum().item()
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_all_zero_target_rows_excluded_from_mean(self) -> None:
        # The "useful, not noisy" guarantee: an all-zero target (opponent holds
        # no hidden dev cards) is dropped from the mean, so the loss equals the
        # mean over ONLY the valid rows — not diluted by the empty ones.
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [2.0, 1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        loss = compute_belief_loss(belief_logits=logits, belief_target=target)
        # Manual: mean of rows 0 and 2 only (row 1 is empty -> excluded).
        log_q = torch.nn.functional.log_softmax(logits, dim=-1)
        per = -(target * log_q).sum(-1)
        expected = (per[0] + per[2]) / 2
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_all_empty_batch_is_exactly_zero(self) -> None:
        # No valid rows -> loss is exactly 0 (no gradient injected), never NaN.
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.zeros(4, 3)
        loss = compute_belief_loss(belief_logits=logits, belief_target=target)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_empty_rows_inject_no_gradient(self) -> None:
        # Empty-target rows must contribute zero gradient even when valid rows
        # are present (so they neither dilute magnitude nor add noise).
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        compute_belief_loss(belief_logits=logits, belief_target=target).backward()
        assert logits.grad is not None
        assert torch.allclose(logits.grad[1], torch.zeros(3))  # empty row -> no grad
        assert not torch.allclose(logits.grad[0], torch.zeros(3))  # valid row -> grad
