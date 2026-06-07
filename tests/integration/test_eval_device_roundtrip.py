"""Regression tests for the eval CPU-pin + eval-mode toggle.

These guard the two integration concerns introduced by the wall-clock
audit (2026-06) when eval was pinned to CPU while the learner trains on
MPS/CUDA:

* ``EvalHarness.run`` must switch the policy to ``eval()`` mode for the
  round (the encoder carries dropout) and restore the prior mode after.
* When the policy lives on a non-CPU device, ``run`` must move it to the
  harness device for the round and restore it afterwards WITHOUT
  breaking a live optimiser (params + AdamW moment buffers must stay
  co-located so the next ``step()`` succeeds). This path is MPS-gated;
  it only runs on a machine where MPS is available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl.eval.harness import EvalHarness


class _ModeRecordingPolicy(nn.Module):
    """Stub that records ``self.training`` the first time ``sample`` runs.

    Same action contract as the unit-test stub: pick a uniformly-random
    legal type, zeros for the other five heads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rng = np.random.default_rng(0)
        self.training_during_sample: bool | None = None

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.training_during_sample is None:
            self.training_during_sample = self.training
        B = next(iter(obs.values())).shape[0]
        type_mask = masks["type"].cpu().numpy()
        action = np.zeros((B, 6), dtype=np.int64)
        for i in range(B):
            legal = np.flatnonzero(type_mask[i])
            action[i, 0] = int(self._rng.choice(legal)) if legal.size else 3
        device = next(iter(obs.values())).device
        return {
            "action": torch.as_tensor(action, device=device),
            "log_prob": torch.zeros(B, device=device),
            "value": torch.zeros(B, device=device),
            "per_head_log_prob": torch.zeros((B, 6), device=device),
            "entropy": torch.zeros(B, device=device) + 1.5,
        }


def _tiny_harness() -> EvalHarness:
    return EvalHarness(
        opponent_types=("heuristic",),
        n_games_per_seat=1,  # 2 games total — enough to drive the policy
        seed=0,
        device=torch.device("cpu"),
        max_turns=40,  # cap length; the round-trip, not the game, is the point
    )


def test_eval_switches_to_eval_mode_and_restores() -> None:
    """A training-mode policy is evaluated in eval() mode, then restored."""
    policy = _ModeRecordingPolicy()
    policy.train()
    assert policy.training is True

    _tiny_harness().run(policy)

    # eval() was active during inference...
    assert policy.training_during_sample is False
    # ...and train() mode is restored for the next SGD update.
    assert policy.training is True


def test_eval_leaves_eval_mode_policy_in_eval_mode() -> None:
    """A policy already in eval() mode stays in eval() mode afterwards."""
    policy = _ModeRecordingPolicy()
    policy.eval()

    _tiny_harness().run(policy)

    assert policy.training_during_sample is False
    assert policy.training is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS-only: validates the CPU-pin device round-trip + optimizer",
)
def test_eval_cpu_pin_roundtrip_preserves_mps_optimizer() -> None:
    """Policy on MPS -> eval on CPU -> restored to MPS; optimizer survives.

    Uses the real :class:`CatanPolicy` (the optimizer-state concern is
    real) but synthetic grads so no obs schema is needed for the steps.
    """
    from catan_rl.policy.network import CatanPolicy

    mps = torch.device("mps")
    policy = CatanPolicy().to(mps)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    def param_device_types() -> set[str]:
        return {p.device.type for p in policy.parameters()}

    def moment_device_types() -> set[str]:
        # AdamW keeps `step` on CPU by design; only moment buffers must
        # stay co-located with the params.
        return {
            st[k].device.type
            for st in opt.state.values()
            for k in ("exp_avg", "exp_avg_sq")
            if k in st
        }

    def fake_step() -> None:
        for p in policy.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Seed AdamW moment buffers on MPS.
    fake_step()
    assert param_device_types() == {"mps"}
    assert moment_device_types() == {"mps"}

    # Eval round pinned to CPU; policy starts on MPS.
    report = _tiny_harness().run(policy)
    assert report.n_games_total == 2

    # Policy restored to MPS, and a subsequent optimiser step still works.
    assert param_device_types() == {"mps"}, "policy not restored to MPS after eval"
    fake_step()
    assert moment_device_types() == {"mps"}
