"""D4 pin: the auxiliary value-target head is byte-neutral at coef=0.

The trainer folds the aux-value MSE into the total loss ONLY when
``aux_value_coef != 0`` (mirroring the belief / setup-entropy guards). With
coef=0 the objective — and therefore every parameter gradient — is identical
whether or not the head emits ``aux_value``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from catan_rl.ppo.arguments import (
    GAEConfig,
    LossConfig,
    OptimizerConfig,
    PPOConfig,
    RolloutConfig,
    TrainConfig,
)
from catan_rl.ppo.buffer import CompositeRolloutBuffer, MaskSpec, ObsSpec
from catan_rl.ppo.trainer import PPOTrainer


class _StubPolicy(nn.Module):
    def __init__(self, *, emit_aux: bool) -> None:
        super().__init__()
        self.backbone = nn.Linear(8, 32)
        self.value_head = nn.Linear(32, 1)
        self.emit_aux = emit_aux

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del action, masks
        x = torch.relu(self.backbone(obs["current_player_main"]))
        value = self.value_head(x).squeeze(-1)
        B = value.shape[0]
        out = {
            "value": value,
            "log_prob": torch.zeros(B),
            "entropy": torch.full((B,), 1.5),
            "per_head_log_prob": torch.zeros((B, 6)),
            "per_head_entropy": torch.zeros((B, 6)),
            "relevance": torch.zeros((B, 6)),
        }
        if self.emit_aux:
            # A trunk-derived scalar (would be folded when coef != 0).
            out["aux_value"] = value * 0.5 + 0.1
        return out


def _config(aux_coef: float) -> TrainConfig:
    return TrainConfig(
        rollout=RolloutConfig(n_envs=4, n_steps=4, vec_env_mode="serial"),
        ppo=PPOConfig(n_epochs=1, batch_size=16, clip_range=0.2, target_kl=10.0),
        gae=GAEConfig(),
        loss=LossConfig(
            value_coef=0.5,
            entropy_coef_start=0.0,
            entropy_coef_end=0.0,
            aux_value_coef=aux_coef,
        ),
        optimizer=OptimizerConfig(lr_start=1e-3, lr_end=1e-4, lr_anneal_total_updates=10),
        total_steps=16,
        run_name="test",
    )


def _buffer(cfg: TrainConfig) -> CompositeRolloutBuffer:
    buf = CompositeRolloutBuffer(
        n_steps=cfg.rollout.n_steps,
        n_envs=cfg.rollout.n_envs,
        obs_spec={"current_player_main": ObsSpec(shape=(8,), dtype=np.dtype(np.float32))},
        mask_spec={"type": MaskSpec(shape=(13,))},
    )
    rng = np.random.default_rng(0)
    N = buf.n_envs
    for _ in range(buf.n_steps):
        buf.add(
            obs={"current_player_main": rng.standard_normal((N, 8)).astype(np.float32)},
            action=rng.integers(0, 5, (N, 6)).astype(np.int64),
            per_head_log_prob=rng.standard_normal((N, 6)).astype(np.float32),
            log_prob=(rng.standard_normal((N,)) * 0.1).astype(np.float32),
            value=rng.standard_normal((N,)).astype(np.float32),
            reward=rng.standard_normal((N,)).astype(np.float32),
            terminated=np.zeros(N, dtype=bool),
            truncated=np.zeros(N, dtype=bool),
            masks={"type": np.ones((N, 13), dtype=bool)},
        )
    buf.compute_returns_and_advantages(
        last_values=np.zeros(N, dtype=np.float32),
        gamma=cfg.gae.gamma,
        gae_lambda=cfg.gae.gae_lambda,
    )
    return buf


def _run_one_step_backbone_grad(emit_aux: bool) -> torch.Tensor:
    torch.manual_seed(123)
    policy = _StubPolicy(emit_aux=emit_aux)
    cfg = _config(aux_coef=0.0)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
    trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")
    buf = _buffer(cfg)
    batch = buf.get_batch(np.arange(cfg.rollout.n_envs * cfg.rollout.n_steps), device="cpu")
    stats = trainer._sgd_step(batch=batch, entropy_coef=0.0)
    return stats["total_loss"].detach()


def test_aux_coef_zero_is_byte_identical_to_no_head() -> None:
    loss_with_aux = _run_one_step_backbone_grad(emit_aux=True)
    loss_without_aux = _run_one_step_backbone_grad(emit_aux=False)
    assert torch.equal(loss_with_aux, loss_without_aux)
