"""Integration smoke for :class:`PPOTrainer`.

Uses a tiny stub policy (≠ real CatanPolicy) so the test focuses on
trainer mechanics: scheduling, minibatch iteration, KL early stop,
metric averaging, gradient flow. The real policy is too heavy to
construct in every unit-test run; Phase 10's sanity training run will
exercise the real-policy path end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
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
from catan_rl.ppo.trainer import PPOTrainer, UpdateMetrics

# ---------------------------------------------------------------------------
# Tiny stub policy that exposes evaluate_actions in the expected shape
# ---------------------------------------------------------------------------


class _StubPolicy(nn.Module):
    """Minimum-surface stand-in for CatanPolicy.

    Maps the ``current_player_main`` obs through a small MLP into a
    joint logit / value / belief surface. Only used to verify the
    trainer's plumbing — NOT a real Catan policy.
    """

    def __init__(self, *, obs_dim: int = 8, n_belief: int = 5) -> None:
        super().__init__()
        self.backbone = nn.Linear(obs_dim, 32)
        self.value_head = nn.Linear(32, 1)
        self.belief_head = nn.Linear(32, n_belief)
        # Scalar "log-prob bias" trained per-step — just enough to give
        # the policy something to update.
        self.logp_bias = nn.Parameter(torch.zeros(1))

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del action, masks  # stub doesn't use these
        x = self.backbone(obs["current_player_main"])
        x = torch.relu(x)
        value = self.value_head(x).squeeze(-1)
        belief_logits = self.belief_head(x)
        B = value.shape[0]
        # Joint log-prob: small + a trainable shift; advantage path uses
        # ratio = exp(new_lp - old_lp) so we want gradient on logp.
        log_prob = torch.zeros(B, device=value.device) + self.logp_bias
        entropy = torch.full((B,), 1.5, device=value.device)
        per_head = torch.zeros((B, 6), device=value.device)
        return {
            "value": value,
            "belief_logits": belief_logits,
            "log_prob": log_prob,
            "entropy": entropy,
            "per_head_log_prob": per_head,
            "per_head_entropy": per_head,
            "relevance": per_head,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> TrainConfig:
    # Small config that satisfies all Phase 1 validators.
    return TrainConfig(
        rollout=RolloutConfig(n_envs=4, n_steps=4, vec_env_mode="serial"),
        ppo=PPOConfig(n_epochs=2, batch_size=8, clip_range=0.2, target_kl=10.0),
        gae=GAEConfig(),
        loss=LossConfig(value_coef=0.5, entropy_coef_start=0.0, entropy_coef_end=0.0),
        optimizer=OptimizerConfig(lr_start=1e-3, lr_end=1e-4, lr_anneal_total_updates=10),
        total_steps=16,  # divides 4*4=16 → 1 update
        run_name="test",
    )


def _make_buffer(cfg: TrainConfig, with_belief: bool = False) -> CompositeRolloutBuffer:
    return CompositeRolloutBuffer(
        n_steps=cfg.rollout.n_steps,
        n_envs=cfg.rollout.n_envs,
        obs_spec={"current_player_main": ObsSpec(shape=(8,), dtype=np.dtype(np.float32))},
        mask_spec={"type": MaskSpec(shape=(13,))},
        belief_target_dim=5 if with_belief else None,
    )


def _fill_random(buffer: CompositeRolloutBuffer, with_belief: bool = False) -> None:
    rng = np.random.default_rng(0)
    N = buffer.n_envs
    for _ in range(buffer.n_steps):
        kwargs = {
            "obs": {"current_player_main": rng.standard_normal((N, 8)).astype(np.float32)},
            "action": rng.integers(0, 5, (N, 6)).astype(np.int64),
            "per_head_log_prob": rng.standard_normal((N, 6)).astype(np.float32),
            "log_prob": rng.standard_normal((N,)).astype(np.float32) * 0.1,
            "value": rng.standard_normal((N,)).astype(np.float32),
            "reward": rng.standard_normal((N,)).astype(np.float32),
            "terminated": np.zeros(N, dtype=bool),
            "truncated": np.zeros(N, dtype=bool),
            "masks": {"type": np.ones((N, 13), dtype=bool)},
        }
        if with_belief:
            target = rng.dirichlet(np.ones(5), size=N).astype(np.float32)
            kwargs["belief_target"] = target
        buffer.add(**kwargs)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


class TestSchedules:
    def test_lr_anneals_across_update_index(self) -> None:
        cfg = _make_config()
        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")
        assert trainer.lr_at(0) == pytest.approx(cfg.optimizer.lr_start)
        assert trainer.lr_at(9) == pytest.approx(cfg.optimizer.lr_end)
        # Past total: holds at end
        assert trainer.lr_at(100) == pytest.approx(cfg.optimizer.lr_end)

    def test_entropy_coef_anneals(self) -> None:
        # Construct a config with a non-trivial entropy anneal window.
        cfg = TrainConfig(
            rollout=RolloutConfig(n_envs=4, n_steps=4, vec_env_mode="serial"),
            ppo=PPOConfig(n_epochs=2, batch_size=8, target_kl=10.0),
            gae=GAEConfig(),
            loss=LossConfig(
                entropy_coef_start=0.04,
                entropy_coef_end=0.005,
                entropy_anneal_start_update=10,
                entropy_anneal_end_update=20,
            ),
            optimizer=OptimizerConfig(lr_start=1e-3, lr_end=1e-4, lr_anneal_total_updates=10),
            total_steps=16,
        )
        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")
        # Holds at start before the anneal window.
        assert trainer.entropy_coef_at(0) == pytest.approx(0.04)
        assert trainer.entropy_coef_at(10) == pytest.approx(0.04)
        # End of anneal window.
        assert trainer.entropy_coef_at(20) == pytest.approx(0.005)
        # Past the anneal window: holds at coef_end.
        assert trainer.entropy_coef_at(500) == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_runs_one_pass_and_returns_metrics(self) -> None:
        cfg = _make_config()
        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        buf = _make_buffer(cfg)
        _fill_random(buf)
        buf.compute_returns_and_advantages(
            last_values=np.zeros(cfg.rollout.n_envs, dtype=np.float32),
            gamma=cfg.gae.gamma,
            gae_lambda=cfg.gae.gae_lambda,
            advantage_norm="rollout",
        )

        rng = np.random.default_rng(0)
        metrics = trainer.update(buf, update_idx=0, rng=rng)
        assert isinstance(metrics, UpdateMetrics)
        # n_epochs=2, batch_size=8, total=16 → 2 batches/epoch → 4 SGD steps total
        assert metrics.n_sgd_steps == 4
        assert metrics.n_epochs_run == 2
        assert torch.isfinite(torch.tensor(metrics.policy_loss))
        assert torch.isfinite(torch.tensor(metrics.value_loss))
        # lr at update_idx=0 == lr_start
        assert metrics.lr == pytest.approx(cfg.optimizer.lr_start)

    def test_update_mutates_policy_parameters(self) -> None:
        cfg = _make_config()
        policy = _StubPolicy()
        snapshot = {k: v.clone() for k, v in policy.state_dict().items()}
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        buf = _make_buffer(cfg)
        _fill_random(buf)
        buf.compute_returns_and_advantages(
            last_values=np.zeros(cfg.rollout.n_envs, dtype=np.float32),
            gamma=cfg.gae.gamma,
            gae_lambda=cfg.gae.gae_lambda,
            advantage_norm="rollout",
        )

        rng = np.random.default_rng(0)
        trainer.update(buf, update_idx=0, rng=rng)
        after = policy.state_dict()
        # At least one parameter should differ
        diffs = [not torch.equal(snapshot[k], after[k]) for k in snapshot]
        assert any(diffs), "no policy parameter moved across update"

    def test_unfinalised_buffer_rejected(self) -> None:
        cfg = _make_config()
        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        buf = _make_buffer(cfg)
        _fill_random(buf)
        # Don't call compute_returns_and_advantages.
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="compute_returns_and_advantages"):
            trainer.update(buf, update_idx=0, rng=rng)

    def test_per_batch_advantage_normalization(self) -> None:
        # Reviewer fix (HIGH): advantage_norm="batch" must actually
        # normalise per minibatch in the trainer. Otherwise the buffer
        # leaves raw advantages and the policy gradient scales by
        # O(advantage) instead of O(1).
        cfg_dict = _make_config().to_dict()
        cfg_dict["ppo"]["advantage_norm"] = "batch"
        cfg = TrainConfig._from_dict(cfg_dict)

        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        # Verify trainer renormalises by inspecting the advantage tensor
        # that reaches compute_policy_loss. Monkeypatch the loss function
        # to capture the actual advantages it received.
        captured: list[torch.Tensor] = []
        import catan_rl.ppo.trainer as trainer_mod

        original_policy_loss = trainer_mod.compute_policy_loss

        def _spy(**kwargs):
            captured.append(kwargs["advantages"].detach().clone())
            return original_policy_loss(**kwargs)

        trainer_mod.compute_policy_loss = _spy  # type: ignore[assignment]
        try:
            buf = _make_buffer(cfg)
            rng = np.random.default_rng(0)
            N = buf.n_envs
            for _ in range(buf.n_steps):
                buf.add(
                    obs={"current_player_main": rng.standard_normal((N, 8)).astype(np.float32)},
                    action=rng.integers(0, 5, (N, 6)).astype(np.int64),
                    per_head_log_prob=rng.standard_normal((N, 6)).astype(np.float32),
                    log_prob=rng.standard_normal((N,)).astype(np.float32),
                    # Large value magnitudes → large raw advantages.
                    value=(rng.standard_normal((N,)) * 100).astype(np.float32),
                    reward=(rng.standard_normal((N,)) * 100).astype(np.float32),
                    terminated=np.zeros(N, dtype=bool),
                    truncated=np.zeros(N, dtype=bool),
                    masks={"type": np.ones((N, 13), dtype=bool)},
                )
            buf.compute_returns_and_advantages(
                last_values=np.zeros(N, dtype=np.float32),
                gamma=cfg.gae.gamma,
                gae_lambda=cfg.gae.gae_lambda,
                advantage_norm="batch",  # buffer leaves raw
            )
            # Sanity: buffer's stored advantages have large magnitude under
            # "batch" mode (they're raw GAE outputs from a high-variance
            # value/reward sequence).
            assert float(buf.advantages.std()) > 1.0, (
                "fixture should have non-normalised raw advantages"
            )
            rng2 = np.random.default_rng(0)
            trainer.update(buf, update_idx=0, rng=rng2)
        finally:
            trainer_mod.compute_policy_loss = original_policy_loss

        # Every captured advantage batch should be ~zero-mean / unit-std
        # because the trainer applied per-batch renorm before calling
        # compute_policy_loss.
        assert len(captured) > 0
        for adv in captured:
            assert abs(float(adv.mean())) < 1e-5, (
                f"trainer did not renormalise: batch mean={float(adv.mean())}"
            )
            assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-3, (
                f"trainer did not renormalise: batch std={float(adv.std())}"
            )

    def test_kl_early_stop(self) -> None:
        # Set target_kl very low → after first epoch, KL exceeds it,
        # second epoch is skipped.
        cfg_dict = _make_config().to_dict()
        cfg_dict["ppo"]["target_kl"] = 1e-8  # essentially "stop immediately"
        cfg = TrainConfig._from_dict(cfg_dict)

        policy = _StubPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        buf = _make_buffer(cfg)
        _fill_random(buf)
        buf.compute_returns_and_advantages(
            last_values=np.zeros(cfg.rollout.n_envs, dtype=np.float32),
            gamma=cfg.gae.gamma,
            gae_lambda=cfg.gae.gae_lambda,
            advantage_norm="rollout",
        )
        rng = np.random.default_rng(0)
        metrics = trainer.update(buf, update_idx=0, rng=rng)
        assert metrics.n_epochs_run == 1  # 2nd epoch skipped


# ---------------------------------------------------------------------------
# Belief loss integration
# ---------------------------------------------------------------------------


class TestBeliefLossIntegration:
    def test_belief_loss_nonzero_when_target_present(self) -> None:
        cfg_dict = _make_config().to_dict()
        cfg_dict["loss"]["belief_coef"] = 0.5
        cfg = TrainConfig._from_dict(cfg_dict)

        policy = _StubPolicy(n_belief=5)
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

        buf = _make_buffer(cfg, with_belief=True)
        _fill_random(buf, with_belief=True)
        buf.compute_returns_and_advantages(
            last_values=np.zeros(cfg.rollout.n_envs, dtype=np.float32),
            gamma=cfg.gae.gamma,
            gae_lambda=cfg.gae.gae_lambda,
            advantage_norm="rollout",
        )
        rng = np.random.default_rng(0)
        metrics = trainer.update(buf, update_idx=0, rng=rng)
        # With random untrained belief head, belief loss should be >0
        # (entropy bound: log(5) ≈ 1.609).
        assert metrics.belief_loss > 0.0
