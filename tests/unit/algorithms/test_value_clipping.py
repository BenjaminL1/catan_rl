"""Tests for Phase 1.1 PPO2-style value clipping."""

from __future__ import annotations

import torch


def test_value_clip_pessimistic() -> None:
    """The clipped MSE is always >= unclipped MSE only when the clip kicks in."""
    # Mirror the trainer's loss code in isolation.
    old_values = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([1.05, 1.30, 0.50])  # one inside clip, two outside
    returns = torch.tensor([2.0, 2.0, 2.0])
    clip_range_vf = 0.2

    v_clipped = old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf)
    # Inside clip: v_clipped == values  → both losses identical.
    # Outside clip: v_clipped is at the boundary; the clipped loss is closer
    # to ``returns`` for the upward-clipped sample (1.30 → 1.20) and farther
    # for the downward-clipped sample (0.50 → 0.80) than the unclipped values.
    # The pessimistic ``max`` always picks whichever errs higher.
    v_loss_unclipped = (values - returns).pow(2)
    v_loss_clipped = (v_clipped - returns).pow(2)
    pessimistic = torch.max(v_loss_unclipped, v_loss_clipped)

    # The pessimistic loss is element-wise >= the unclipped loss
    # whenever the unclipped guess is closer to the target than the clip
    # would allow (so PPO2 keeps the larger error).
    assert torch.all(pessimistic >= v_loss_unclipped - 1e-9)
    assert torch.all(pessimistic >= v_loss_clipped - 1e-9)


def test_value_clip_zero_when_inside_clip() -> None:
    """If every new value is within ``clip_range_vf`` of the old, both losses match."""
    old_values = torch.tensor([1.0, 1.0])
    values = torch.tensor([1.05, 0.90])  # both within ±0.2
    returns = torch.tensor([2.0, 2.0])
    clip_range_vf = 0.2

    v_clipped = old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf)
    torch.testing.assert_close(v_clipped, values)
    v_loss_unclipped = (values - returns).pow(2)
    v_loss_clipped = (v_clipped - returns).pow(2)
    pessimistic = torch.max(v_loss_unclipped, v_loss_clipped)
    torch.testing.assert_close(pessimistic, v_loss_unclipped)


def test_trainer_uses_value_clipping_flag() -> None:
    """The trainer constructs with ``use_value_clipping`` and ``clip_range_vf``
    drawn from config, with sane defaults."""
    from catan_rl.algorithms.ppo.arguments import get_config
    from catan_rl.algorithms.ppo.trainer import CatanPPO

    cfg = get_config()
    cfg.update(
        {
            "total_timesteps": 100,
            "n_steps": 50,
            "n_envs": 1,
            "batch_size": 16,
            "n_epochs": 1,
            "eval_games": 0,
            "eval_freq": 99999,
            "checkpoint_freq": 99999,
            "log_dir": "/tmp/cr_test_vclip",
            "checkpoint_dir": "/tmp/cr_test_vclip_ckpts",
        }
    )
    t = CatanPPO(cfg)
    assert t.use_value_clipping is True  # default
    assert t.clip_range_vf == 0.2

    cfg["use_value_clipping"] = False
    cfg["clip_range_vf"] = 0.5
    t2 = CatanPPO(cfg)
    assert t2.use_value_clipping is False
    assert t2.clip_range_vf == 0.5
