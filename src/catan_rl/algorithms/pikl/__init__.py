"""piKL anchor KL regularisation for v2 PPO.

Phase 9 ships the primitives needed to compute a piKL-style anchor
penalty against the live training policy:

* :class:`AnchorPolicy` — a frozen wrapper around a CatanPolicy-shaped
  module. Detaches outputs so KL terms can't leak gradients into the
  anchor's params; ``requires_grad=False`` is set on every param.
* :func:`pikl_kl_loss` — relevance-weighted KL estimator over the 6
  autoregressive heads. Two estimators supported: ``"k1"`` (unbiased
  Monte Carlo, higher variance) and ``"k3"`` (low-variance,
  non-negative, Schulman 2020). Both return scalars suitable to add
  to the PPO total loss.
* :class:`PiKLConfig` — dataclass carrying the loss weight + anchor
  path + estimator choice. Frozen, validated, YAML-roundtrippable.
* :func:`load_pikl_anchor` — loads a frozen anchor from a Phase 8
  checkpoint and wraps it.

**This phase is a stub.** Nothing in the live PPO trainer calls
:func:`pikl_kl_loss` yet — wiring it through ``PPOTrainer._sgd_step``
is deferred to Phase 10 (or later, once the BC anchor has been
trained to a usable WR). Shipping the primitives and tests now means
Phase 10 can flip a single flag in :class:`PiKLConfig` to enable it
without scrambling for a new loss term.

**Why piKL, not vanilla anchor**: the canonical KL-anchored RL
objective (RLHF-style) treats the anchor as the *reference* and
penalises drift in either direction. piKL is asymmetric — it
penalises the student's drift away from the anchor while still
letting it improve via the value signal. The estimators below
implement that asymmetry (KL(student || anchor), not the reverse).
"""

from catan_rl.algorithms.pikl.anchor import AnchorPolicy, AnchorPolicyError
from catan_rl.algorithms.pikl.config import KLEstimator, PiKLConfig
from catan_rl.algorithms.pikl.loader import load_pikl_anchor
from catan_rl.algorithms.pikl.loss import (
    PiKLLossOutput,
    pikl_kl_loss,
    pikl_kl_per_head,
)

__all__ = [
    "AnchorPolicy",
    "AnchorPolicyError",
    "KLEstimator",
    "PiKLConfig",
    "PiKLLossOutput",
    "load_pikl_anchor",
    "pikl_kl_loss",
    "pikl_kl_per_head",
]
