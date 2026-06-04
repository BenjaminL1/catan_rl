"""Tests for `algorithms/pikl/loss.py`.

Pins:
1. Both estimators are 0 when student == anchor elementwise.
2. k3 is non-negative on every sample (k1 is not — that's OK).
3. Both estimators agree in expectation under a low-variance MC.
4. Relevance masking zeros irrelevant heads.
5. lambda_kl scales the loss but not the kl_mean diagnostic.
6. Non-detached anchor raises a clear error.
7. Shape / sign validators.
"""

from __future__ import annotations

import math

import pytest
import torch

from catan_rl.algorithms.pikl.loss import (
    PiKLLossOutput,
    pikl_kl_loss,
    pikl_kl_per_head,
)


def _rand_log_probs(batch: int, seed: int) -> torch.Tensor:
    """Random per-head log-probs that sum-exp to ~1 within each head
    — i.e., realistic softmax outputs. ``(B, 6)``."""
    torch.manual_seed(seed)
    # Random logits per head, log_softmax to get one log-prob per head.
    # We treat each head as having 5 "actions" and pick action 0 as the
    # one we evaluate (any column works).
    logits = torch.randn(batch, 6, 5)
    return torch.log_softmax(logits, dim=-1)[:, :, 0]  # (B, 6)


class TestPerHead:
    def test_zero_when_identical(self) -> None:
        lp = _rand_log_probs(8, seed=0)
        for est in ("k1", "k3"):
            out = pikl_kl_per_head(
                student_log_prob=lp,
                anchor_log_prob=lp,
                estimator=est,  # type: ignore[arg-type]
            )
            assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_k3_non_negative_on_every_sample(self) -> None:
        torch.manual_seed(42)
        student = _rand_log_probs(64, seed=1)
        anchor = _rand_log_probs(64, seed=2)
        out = pikl_kl_per_head(
            student_log_prob=student,
            anchor_log_prob=anchor,
            estimator="k3",
        )
        # Mathematical guarantee: r - 1 - log_r >= 0 for r > 0.
        assert (out >= -1e-7).all()

    def test_k1_can_be_negative_on_a_sample(self) -> None:
        # The unbiased estimator goes negative when the anchor places
        # higher probability on the sampled action than the student.
        student = torch.full((1, 6), -1.0)  # log p = -1
        anchor = torch.full((1, 6), -0.5)  # log q = -0.5 (higher prob)
        out = pikl_kl_per_head(
            student_log_prob=student,
            anchor_log_prob=anchor,
            estimator="k1",
        )
        # log_p - log_q = -1 - (-0.5) = -0.5 (negative).
        assert torch.allclose(out, torch.full_like(out, -0.5))

    def test_unknown_estimator_raises(self) -> None:
        lp = _rand_log_probs(2, seed=0)
        with pytest.raises(ValueError, match="unknown estimator"):
            pikl_kl_per_head(
                student_log_prob=lp,
                anchor_log_prob=lp,
                estimator="bogus",  # type: ignore[arg-type]
            )

    def test_shape_mismatch_raises(self) -> None:
        a = torch.zeros(4, 6)
        b = torch.zeros(4, 5)
        with pytest.raises(ValueError, match="same"):
            pikl_kl_per_head(student_log_prob=a, anchor_log_prob=b)


class TestScalarLoss:
    def _build(
        self, *, student_seed: int, anchor_seed: int, relevance_pattern: str = "all"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = _rand_log_probs(16, seed=student_seed).requires_grad_(True)
        a = _rand_log_probs(16, seed=anchor_seed).detach()
        if relevance_pattern == "all":
            r = torch.ones_like(s)
        elif relevance_pattern == "type_only":
            r = torch.zeros_like(s)
            r[:, 0] = 1.0
        elif relevance_pattern == "none":
            r = torch.zeros_like(s)
        else:
            raise ValueError(relevance_pattern)
        return s, a, r

    def test_identical_distribution_yields_zero_loss(self) -> None:
        lp = _rand_log_probs(8, seed=0).requires_grad_(True)
        rel = torch.ones_like(lp)
        for est in ("k1", "k3"):
            out = pikl_kl_loss(
                student_per_head_log_prob=lp,
                anchor_per_head_log_prob=lp.detach(),
                relevance=rel,
                estimator=est,  # type: ignore[arg-type]
                lambda_kl=0.1,
            )
            assert isinstance(out, PiKLLossOutput)
            assert out.loss.abs().item() < 1e-7
            assert out.kl_mean.abs().item() < 1e-7

    def test_irrelevant_heads_zero_contribution(self) -> None:
        # Even with maximally-different distributions, if relevance is
        # all zero the loss is exactly 0.
        s, a, _ = self._build(student_seed=10, anchor_seed=20)
        rel = torch.zeros_like(s)
        out = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=rel,
            estimator="k3",
            lambda_kl=1.0,
        )
        assert out.loss.item() == 0.0
        assert out.n_active_heads_mean.item() == 0.0

    def test_lambda_scales_loss_but_not_kl_mean(self) -> None:
        s, a, r = self._build(student_seed=10, anchor_seed=20)
        out_a = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=r,
            estimator="k3",
            lambda_kl=1.0,
        )
        out_b = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=r,
            estimator="k3",
            lambda_kl=0.05,
        )
        # Loss is scaled, kl_mean is invariant.
        assert torch.allclose(out_b.loss, out_a.loss * 0.05)
        assert torch.allclose(out_b.kl_mean, out_a.kl_mean)

    def test_negative_lambda_rejected(self) -> None:
        s, a, r = self._build(student_seed=0, anchor_seed=1)
        with pytest.raises(ValueError, match="lambda_kl"):
            pikl_kl_loss(
                student_per_head_log_prob=s,
                anchor_per_head_log_prob=a,
                relevance=r,
                lambda_kl=-0.01,
            )

    def test_non_detached_anchor_rejected(self) -> None:
        s = _rand_log_probs(2, seed=0).requires_grad_(True)
        a = _rand_log_probs(2, seed=1).requires_grad_(True)
        r = torch.ones_like(s)
        with pytest.raises(ValueError, match="detached"):
            pikl_kl_loss(
                student_per_head_log_prob=s,
                anchor_per_head_log_prob=a,
                relevance=r,
            )

    def test_loss_carries_gradient_through_student(self) -> None:
        s, a, r = self._build(student_seed=3, anchor_seed=4)
        out = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=r,
            estimator="k3",
            lambda_kl=0.1,
        )
        out.loss.backward()
        assert s.grad is not None
        assert (s.grad.abs().sum() > 0).item()

    def test_k3_loss_is_non_negative(self) -> None:
        # Across many random batches, the k3 mean is non-negative.
        for seed in range(5):
            s, a, r = self._build(student_seed=seed, anchor_seed=seed + 100)
            out = pikl_kl_loss(
                student_per_head_log_prob=s,
                anchor_per_head_log_prob=a,
                relevance=r,
                estimator="k3",
                lambda_kl=1.0,
            )
            assert out.loss.item() >= -1e-7

    def test_k3_clamps_log_r_to_avoid_overflow(self) -> None:
        # When log(q/p) ~= +50 (anchor places much more mass than
        # student), the unclamped k3 would compute exp(50) ~= 5e21
        # and overflow. The clamp at ±20 caps it at exp(20) ~= 4.85e8.
        s = torch.full((1, 6), -50.0).requires_grad_(True)  # student log p
        a = torch.zeros(1, 6).detach()  # anchor log q (log_r = +50)
        r = torch.ones_like(s)
        out = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=r,
            estimator="k3",
            lambda_kl=1.0,
        )
        # Loss is finite (not inf, not nan) thanks to the clamp.
        assert torch.isfinite(out.loss).item()
        # Sanity ceiling: clamp at log_r=20 → 6 heads × (exp(20) - 1 - 20).
        ceiling = 6.0 * (math.exp(20.0) - 1.0 - 20.0)
        assert out.loss.item() <= ceiling * 1.01

    def test_device_mismatch_raises(self) -> None:
        # CPU/CPU/CPU is fine; we test the validator triggers when
        # devices are constructed differently. (No GPU on CI so we
        # patch the device attribute.)
        s = torch.zeros(2, 6).requires_grad_(True)
        a = torch.zeros(2, 6)
        r = torch.ones(2, 6)
        # Make ``a.device`` look different from the others.
        with pytest.raises(ValueError, match="share a device"):
            # Forge by reusing a CPU tensor under a different device
            # context. Easiest: build a CUDA-like meta tensor.
            a_meta = a.to("meta")
            pikl_kl_loss(
                student_per_head_log_prob=s,
                anchor_per_head_log_prob=a_meta,
                relevance=r,
            )

    def test_anchor_relevance_mismatch_raises(self) -> None:
        s = torch.zeros(2, 6).requires_grad_(True)
        a = torch.zeros(2, 6)
        rel_s = torch.ones(2, 6)
        rel_a = torch.zeros(2, 6)
        rel_a[:, :3] = 1.0  # only first 3 heads — disagrees with student
        with pytest.raises(ValueError, match="relevance"):
            pikl_kl_loss(
                student_per_head_log_prob=s,
                anchor_per_head_log_prob=a,
                relevance=rel_s,
                anchor_relevance=rel_a,
            )

    def test_anchor_relevance_match_passes(self) -> None:
        s = torch.zeros(2, 6).requires_grad_(True)
        a = torch.zeros(2, 6)
        rel = torch.ones(2, 6)
        out = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=rel,
            anchor_relevance=rel.clone(),
            lambda_kl=0.1,
        )
        assert out.loss.item() == 0.0

    def test_lambda_zero_short_circuits(self) -> None:
        # lambda_kl=0 returns zero tensors without going through k3.
        # We assert the loss is exactly 0 and that the diagnostic
        # tensors still carry the active-head count for TB.
        s = torch.randn(4, 6).requires_grad_(True)
        a = torch.randn(4, 6).detach()
        rel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)  # 1 active head per sample
        out = pikl_kl_loss(
            student_per_head_log_prob=s,
            anchor_per_head_log_prob=a,
            relevance=rel,
            lambda_kl=0.0,
        )
        assert out.loss.item() == 0.0
        assert out.kl_mean.item() == 0.0
        assert out.n_active_heads_mean.item() == 1.0

    def test_k3_estimator_matches_true_kl_under_real_sampling(self) -> None:
        # Real test of the k3 estimator's correctness: sample actions
        # under the student distribution, look up student/anchor
        # log-probs at those samples, and confirm the k3 mean
        # converges to the closed-form categorical KL.
        torch.manual_seed(0)
        # Student and anchor are two different categorical
        # distributions over 5 actions, per-head independent.
        n_samples = 50_000
        student_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
        anchor_logits = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])  # uniform
        student_logp = torch.log_softmax(student_logits, dim=-1)  # (1, 5)
        anchor_logp = torch.log_softmax(anchor_logits, dim=-1)  # (1, 5)

        # Closed-form KL(student || anchor).
        closed_form = (student_logp.exp() * (student_logp - anchor_logp)).sum().item()
        # MC samples a ~ student.
        probs = student_logp.exp().squeeze(0)
        samples = torch.multinomial(probs, n_samples, replacement=True)
        s_lp = student_logp[0, samples].view(-1, 1).expand(-1, 6)
        a_lp = anchor_logp[0, samples].view(-1, 1).expand(-1, 6)
        rel = torch.zeros_like(s_lp)
        rel[:, 0] = 1.0  # only one head active
        s_lp = s_lp.detach().requires_grad_(True)
        a_lp = a_lp.detach()
        out_k3 = pikl_kl_loss(
            student_per_head_log_prob=s_lp,
            anchor_per_head_log_prob=a_lp,
            relevance=rel,
            estimator="k3",
            lambda_kl=1.0,
        )
        # k3 has lower variance than the closed form — converges
        # quickly. 5% tolerance at 50k samples is comfortable.
        assert math.fabs(out_k3.loss.item() - closed_form) / closed_form < 0.05
