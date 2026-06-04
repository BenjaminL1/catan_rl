"""piKL config dataclass.

Pinned in :mod:`catan_rl.ppo.arguments` once the trainer wires the
loss in (Phase 10). For Phase 9 the dataclass lives standalone so
tests can validate it in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: KL estimator names accepted by :func:`pikl_kl_loss`.
KLEstimator = Literal["k1", "k3"]


@dataclass(frozen=True)
class PiKLConfig:
    """Configuration for piKL anchor regularisation.

    Disabled by default; switch ``enabled=True`` and set
    ``anchor_checkpoint_path`` to a frozen BC checkpoint to activate
    the regularisation once Phase 10 wires it into the trainer.

    Attributes:
        enabled: Master switch. When False, the trainer skips the
            anchor forward pass entirely (no compute cost).
        lambda_kl: Coefficient on the KL penalty in the PPO total
            loss. ``0.0`` is degenerate (disables the term without
            changing ``enabled``); meaningful values are typically
            in [1e-3, 1.0] — too high suppresses learning, too low
            doesn't anchor.
        anchor_checkpoint_path: Path to the frozen anchor's Phase 8
            checkpoint. Required when ``enabled=True``; ignored
            otherwise.
        kl_estimator: ``"k1"`` (unbiased Monte Carlo,
            ``E[log_p - log_q]``) or ``"k3"`` (low-variance,
            non-negative). ``"k3"`` is the recommended default per
            Schulman 2020.
        warmup_updates: Linearly ramps ``lambda_kl`` from 0 to its
            target over the first N updates. ``0`` = no warmup.
            Useful when the anchor is much weaker than the student
            at start and a hard kick from a strong KL term would
            destabilise the policy.
    """

    enabled: bool = False
    lambda_kl: float = 0.0
    anchor_checkpoint_path: str | None = None
    kl_estimator: KLEstimator = "k3"
    warmup_updates: int = 0

    def __post_init__(self) -> None:
        if self.lambda_kl < 0.0:
            raise ValueError(f"lambda_kl must be >= 0, got {self.lambda_kl}")
        if self.warmup_updates < 0:
            raise ValueError(f"warmup_updates must be >= 0, got {self.warmup_updates}")
        if self.kl_estimator not in ("k1", "k3"):
            raise ValueError(f"kl_estimator must be 'k1' or 'k3', got {self.kl_estimator!r}")
        if self.enabled and not self.anchor_checkpoint_path:
            raise ValueError("enabled=True requires anchor_checkpoint_path to be set")

    def coef_at(self, update_idx: int) -> float:
        """Return the effective ``lambda_kl`` at PPO update ``update_idx``.

        Implements the linear warmup. ``update_idx < 0`` is treated as
        ``0`` (caller fed a pre-training step).
        """
        if not self.enabled:
            return 0.0
        if self.warmup_updates <= 0:
            return self.lambda_kl
        u = max(0, update_idx)
        frac = min(1.0, u / float(self.warmup_updates))
        return self.lambda_kl * frac
