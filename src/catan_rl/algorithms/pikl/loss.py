"""piKL KL-anchor loss computation.

Two estimators of :math:`\\mathrm{KL}(\\pi_\\theta\\ ||\\ \\pi_{anchor})` over
the 6-head autoregressive Catan policy. Both take per-head
log-probabilities + a relevance mask and return a scalar suitable to
add to the PPO total loss.

* **k1 (unbiased Monte Carlo)**:
  :math:`\\widehat{KL}_1 = \\log p - \\log q`. Unbiased estimator of
  the true KL, but can go negative on individual samples and has
  high variance.
* **k3 (Schulman 2020)**:
  :math:`\\widehat{KL}_3 = r - 1 - \\log r` where
  :math:`r = q / p = \\exp(\\log q - \\log p)`. Non-negative on every
  sample, lower variance, drop-in replacement.

Why per-head, not joint: the Catan policy factors as
:math:`\\log \\pi(a|s) = \\sum_h r_h(s)\\,\\log \\pi_h(a_h | s, a_{<h})`
where :math:`r_h` is the head's relevance mask (0 or 1) and the heads
are sampled autoregressively. The KL factorises the same way:

.. math::

    \\mathrm{KL}(\\pi_\\theta\\ ||\\ \\pi_{anchor})
    = \\sum_h \\mathbb{E}_{\\pi_\\theta}\\big[
        r_h(s)\\,\\big(\\log \\pi_{\\theta,h} - \\log \\pi_{anchor,h}\\big)
      \\big]

so we compute per-head, mask out irrelevant heads with relevance, and
sum. Averaging over the batch gives the scalar loss.

The k1 estimator is used in PPO's `target_kl` early-stop. k3 is used
inside the gradient because it's non-negative and so directly
penalises drift without sometimes rewarding it.

**Sign convention vs PPO's k1**: the PPO loss in
:mod:`catan_rl.ppo.losses` samples ``a ~ π_old`` (off-policy
importance) and computes ``approx_kl ≈ (old_log_prob -
new_log_prob).mean()`` — an estimator of
:math:`\\mathrm{KL}(\\pi_{old}\\ ||\\ \\pi_{new})`. piKL samples
``a ~ π_θ`` (on-policy student rollouts) and computes ``student -
anchor`` — an estimator of
:math:`\\mathrm{KL}(\\pi_\\theta\\ ||\\ \\pi_{anchor})`. The two signs
are *opposite* by sampling-distribution convention; both are correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from catan_rl.algorithms.pikl.config import KLEstimator


@dataclass(frozen=True)
class PiKLLossOutput:
    """Aggregated KL stats for one minibatch.

    Attributes:
        loss: Scalar tensor to add to the PPO total loss (with
            ``lambda_kl`` already applied if the caller wants).
        kl_mean: Per-sample mean of the relevance-weighted KL
            (k1 or k3, matching the estimator the caller asked for).
            For logging only; should NOT be multiplied by lambda
            again before going to TB.
        kl_max: Maximum per-sample KL across the batch — TB scalar
            for spotting outliers.
        n_active_heads_mean: Average number of relevant heads per
            sample, for TB. ~1.5-2.5 in practice (most actions
            engage 1-3 heads).
    """

    loss: torch.Tensor
    kl_mean: torch.Tensor
    kl_max: torch.Tensor
    n_active_heads_mean: torch.Tensor


#: Maximum absolute value of ``log_r = log_anchor - log_student``
#: before k3 is clamped. ``exp(20) ≈ 4.85e8`` — still loud enough to
#: spot in TB but safe from fp32 overflow + keeps gradients bounded
#: on stale off-policy rows where the recorded action sits in a
#: near-zero region of the live student's distribution.
_K3_LOG_R_CLAMP = 20.0


def pikl_kl_per_head(
    *,
    student_log_prob: torch.Tensor,
    anchor_log_prob: torch.Tensor,
    estimator: KLEstimator = "k3",
) -> torch.Tensor:
    """Per-sample per-head KL contribution.

    Args:
        student_log_prob: ``(B, 6)`` per-head log-probability of the
            sampled actions under the live policy. Must carry
            gradients.
        anchor_log_prob: ``(B, 6)`` per-head log-probability of the
            same actions under the frozen anchor. Must be detached.
        estimator: ``"k1"`` or ``"k3"``.

    Returns:
        ``(B, 6)`` tensor of per-head KL contributions. NOT yet
        relevance-masked — caller multiplies by the relevance vector
        before summing.

    Math:

    * k1: ``log_p - log_q``
    * k3: ``exp(log_q - log_p) - 1 - (log_q - log_p)``, with
      ``log(q/p)`` clamped to ``[-20, 20]`` to prevent fp32 overflow
      on stale off-policy minibatch rows where ``log q - log p`` can
      hit ~50 (the anchor places near-zero mass on an action the
      student is currently confident about).

    Both reduce to 0 when ``log_p == log_q`` elementwise.
    """
    if student_log_prob.shape != anchor_log_prob.shape:
        raise ValueError(
            "student_log_prob and anchor_log_prob must have the same "
            f"shape; got {tuple(student_log_prob.shape)} vs "
            f"{tuple(anchor_log_prob.shape)}"
        )
    if estimator == "k1":
        return student_log_prob - anchor_log_prob
    if estimator == "k3":
        log_r = anchor_log_prob - student_log_prob
        # Clamp BOTH directions: a huge positive log_r blows up
        # exp(log_r), a huge negative log_r is harmless for exp but
        # produces a huge -log_r contribution to the estimator.
        log_r = log_r.clamp(min=-_K3_LOG_R_CLAMP, max=_K3_LOG_R_CLAMP)
        # exp(log_r) - 1 - log_r >= 0 for all real log_r, with equality
        # iff log_r == 0. Cheap to verify analytically.
        r = torch.exp(log_r)
        return r - 1.0 - log_r
    raise ValueError(f"unknown estimator {estimator!r}; expected 'k1' or 'k3'")


def pikl_kl_loss(
    *,
    student_per_head_log_prob: torch.Tensor,
    anchor_per_head_log_prob: torch.Tensor,
    relevance: torch.Tensor,
    estimator: KLEstimator = "k3",
    lambda_kl: float = 1.0,
    anchor_relevance: torch.Tensor | None = None,
) -> PiKLLossOutput:
    """Scalar piKL anchor loss for one minibatch.

    Args:
        student_per_head_log_prob: ``(B, 6)`` from the live policy's
            :meth:`evaluate_actions`. Carries gradients.
        anchor_per_head_log_prob: ``(B, 6)`` from the frozen anchor's
            :meth:`evaluate_actions`. Detached.
        relevance: ``(B, 6)`` head relevance from the live policy.
            The autoregressive factorisation assumes the anchor's
            relevance is identical (two ``CatanPolicy`` instances
            share the hardcoded ``head_relevance`` buffer) — pass
            ``anchor_relevance`` to assert it explicitly when the
            anchor is a non-``CatanPolicy`` baseline.
        estimator: ``"k1"`` or ``"k3"``.
        lambda_kl: Scalar weight applied to the mean loss. The
            returned ``loss`` field is already scaled; ``kl_mean``
            is NOT (so TB shows the raw KL estimate, not the
            weighted one). Setting ``lambda_kl=0.0`` short-circuits
            the math and returns zero tensors — but the anchor
            forward pass already happened by then, so the trainer
            should ALSO skip the call at the call site to save the
            wasted compute (typical during warmup).
        anchor_relevance: Optional ``(B, 6)`` relevance from the
            anchor. When provided, asserts elementwise equality
            with ``relevance``; if a future hand-shaped baseline
            anchor uses a different action-type→relevance mapping,
            the joint KL factorisation breaks and this gate catches
            it loudly.

    Returns:
        :class:`PiKLLossOutput` with loss + diagnostics.
    """
    if student_per_head_log_prob.shape != anchor_per_head_log_prob.shape:
        raise ValueError(
            "student and anchor per-head log-prob shapes mismatch: "
            f"{tuple(student_per_head_log_prob.shape)} vs "
            f"{tuple(anchor_per_head_log_prob.shape)}"
        )
    if relevance.shape != student_per_head_log_prob.shape:
        raise ValueError(
            "relevance shape must match per-head log-prob shape; got "
            f"{tuple(relevance.shape)} vs "
            f"{tuple(student_per_head_log_prob.shape)}"
        )
    if (
        student_per_head_log_prob.device != anchor_per_head_log_prob.device
        or student_per_head_log_prob.device != relevance.device
    ):
        raise ValueError(
            "student / anchor / relevance must share a device; got "
            f"student={student_per_head_log_prob.device}, "
            f"anchor={anchor_per_head_log_prob.device}, "
            f"relevance={relevance.device}. Call "
            "anchor.to(student.device) after load_pikl_anchor()."
        )
    if anchor_per_head_log_prob.requires_grad:
        # The anchor must be frozen — if a caller wires a non-detached
        # anchor we'd silently push gradients into the anchor's params.
        # Catch it loudly here.
        raise ValueError(
            "anchor_per_head_log_prob must be detached; got "
            "requires_grad=True. Did you forget to wrap the anchor "
            "policy with AnchorPolicy?"
        )
    if lambda_kl < 0.0:
        raise ValueError(f"lambda_kl must be >= 0, got {lambda_kl}")
    if anchor_relevance is not None and not torch.equal(relevance, anchor_relevance):
        # Mismatched relevance would silently misweight per-head KLs.
        # See loss docstring for why this matters once the anchor is
        # a non-CatanPolicy baseline.
        raise ValueError(
            "student relevance and anchor relevance differ — joint KL "
            "factorisation requires both policies to agree on which "
            "heads are active for each sampled action type."
        )

    # Short-circuit when the caller didn't bother to skip a zero-weight
    # KL term. The anchor forward already burned compute upstream, but
    # at least skip the math + tensor allocations here.
    if lambda_kl == 0.0:
        zero = torch.zeros((), device=student_per_head_log_prob.device)
        n_active = relevance.sum(dim=-1).mean()
        return PiKLLossOutput(
            loss=zero,
            kl_mean=zero,
            kl_max=zero,
            n_active_heads_mean=n_active.detach(),
        )

    per_head = pikl_kl_per_head(
        student_log_prob=student_per_head_log_prob,
        anchor_log_prob=anchor_per_head_log_prob,
        estimator=estimator,
    )
    # Mask out irrelevant heads — they're noise w.r.t. the joint KL.
    weighted = per_head * relevance
    per_sample_kl = weighted.sum(dim=-1)  # (B,)
    n_active = relevance.sum(dim=-1)  # (B,) per-sample head count

    kl_mean = per_sample_kl.mean()
    kl_max = per_sample_kl.max()
    n_active_mean = n_active.mean()

    loss = lambda_kl * kl_mean

    return PiKLLossOutput(
        loss=loss,
        kl_mean=kl_mean.detach(),
        kl_max=kl_max.detach(),
        n_active_heads_mean=n_active_mean.detach(),
    )
