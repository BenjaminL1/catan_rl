"""Tests for `algorithms/pikl/anchor.py`.

Pins:
1. Wrapping freezes every parameter.
2. evaluate_actions returns detached tensors (no grad).
3. Backward through KL using anchor output reaches student but NOT anchor.
4. train(True) is a no-op — anchor stays in eval forever.
5. forward() raises.
6. Wrapping a module without evaluate_actions raises.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from catan_rl.algorithms.pikl.anchor import AnchorPolicy, AnchorPolicyError
from catan_rl.algorithms.pikl.loss import pikl_kl_loss


class _MockPolicy(nn.Module):
    """Mimics the surface of CatanPolicy.evaluate_actions."""

    def __init__(self) -> None:
        super().__init__()
        # Per-head bias scalars + one dummy linear so we have BN-free
        # parameters that backward can reach.
        self.head_bias = nn.Parameter(torch.zeros(6))
        self.dummy = nn.Linear(8, 4)

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B = action.shape[0]
        # Per-head log-probs determined by head_bias so we can train it.
        per_head = self.head_bias.expand(B, -1).clone()
        relevance = torch.ones(B, 6)
        log_prob = per_head.sum(dim=-1)
        return {
            "log_prob": log_prob,
            "per_head_log_prob": per_head,
            "per_head_entropy": torch.zeros(B, 6),
            "entropy": torch.zeros(B),
            "relevance": relevance,
        }


class TestFreezing:
    def test_params_have_requires_grad_false(self) -> None:
        p = _MockPolicy()
        for param in p.parameters():
            assert param.requires_grad is True
        anchor = AnchorPolicy(p)
        for param in anchor.parameters():
            assert param.requires_grad is False

    def test_module_is_in_eval_mode(self) -> None:
        p = _MockPolicy()
        p.train()
        anchor = AnchorPolicy(p)
        assert not anchor._inner.training


class TestEvaluateActions:
    def _make_inputs(self, B: int = 4) -> tuple[dict, torch.Tensor, dict]:
        obs = {"dummy": torch.zeros(B, 8)}
        action = torch.zeros(B, 6, dtype=torch.long)
        masks: dict[str, torch.Tensor] = {}
        return obs, action, masks

    def test_returned_tensors_are_detached(self) -> None:
        anchor = AnchorPolicy(_MockPolicy())
        obs, action, masks = self._make_inputs()
        out = anchor.evaluate_actions(obs, action, masks)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert v.requires_grad is False, k
                assert v.grad_fn is None, k

    def test_backward_does_not_touch_anchor_params(self) -> None:
        # Build a student that mimics CatanPolicy and an anchor that
        # also mimics. The KL loss must produce gradients on the
        # student's head_bias but NOT on the anchor's.
        student = _MockPolicy()
        anchor_inner = _MockPolicy()
        # Make them initially different.
        with torch.no_grad():
            anchor_inner.head_bias.fill_(0.5)
        anchor = AnchorPolicy(anchor_inner)

        obs = {"dummy": torch.zeros(4, 8)}
        action = torch.zeros(4, 6, dtype=torch.long)
        masks: dict[str, torch.Tensor] = {}
        s_out = student.evaluate_actions(obs, action, masks)
        a_out = anchor.evaluate_actions(obs, action, masks)
        loss = pikl_kl_loss(
            student_per_head_log_prob=s_out["per_head_log_prob"],
            anchor_per_head_log_prob=a_out["per_head_log_prob"],
            relevance=s_out["relevance"],
            estimator="k3",
            lambda_kl=1.0,
        ).loss
        loss.backward()
        # Student should have grads on head_bias.
        assert student.head_bias.grad is not None
        assert student.head_bias.grad.abs().sum().item() > 0
        # Anchor's inner has no grads accumulated (requires_grad=False).
        assert anchor_inner.head_bias.grad is None


class TestLocked:
    def test_train_true_is_noop(self) -> None:
        p = _MockPolicy()
        anchor = AnchorPolicy(p)
        anchor.train(True)
        assert not anchor.training
        assert not anchor._inner.training

    def test_forward_raises(self) -> None:
        anchor = AnchorPolicy(_MockPolicy())
        with pytest.raises(AnchorPolicyError, match="evaluate_actions"):
            anchor.forward()


class TestConstruction:
    def test_missing_evaluate_actions_raises(self) -> None:
        class _Bad(nn.Module):
            pass

        with pytest.raises(AnchorPolicyError, match="evaluate_actions"):
            AnchorPolicy(_Bad())


class TestStateDictTransparency:
    def test_inner_state_dict_has_no_inner_prefix(self) -> None:
        # AnchorPolicy.inner_state_dict returns the bare inner dict
        # so its keys load directly into a fresh _MockPolicy.
        p = _MockPolicy()
        anchor = AnchorPolicy(p)
        sd = anchor.inner_state_dict()
        assert "head_bias" in sd
        assert "dummy.weight" in sd
        assert all(not k.startswith("_inner.") for k in sd)

    def test_inner_state_dict_loads_into_bare_policy(self) -> None:
        # Round-trip: grab the inner dict via the helper, load it
        # into a fresh _MockPolicy, params match.
        original = _MockPolicy()
        with torch.no_grad():
            original.head_bias.fill_(0.42)
        anchor = AnchorPolicy(original)

        fresh = _MockPolicy()
        fresh.load_state_dict(anchor.inner_state_dict())
        assert torch.equal(fresh.head_bias, original.head_bias)

    def test_load_inner_state_dict_forwards_to_inner(self) -> None:
        # Symmetric: load_inner_state_dict accepts a bare
        # CatanPolicy-shaped state dict.
        anchor = AnchorPolicy(_MockPolicy())
        other = _MockPolicy()
        with torch.no_grad():
            other.head_bias.fill_(0.99)
        anchor.load_inner_state_dict(other.state_dict())
        assert torch.equal(anchor._inner.head_bias, other.head_bias)
