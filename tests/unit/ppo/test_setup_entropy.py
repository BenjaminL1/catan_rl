"""Setup-phase-only entropy bonus (step6 human-corpus plan §2.2 / §5).

Covers the ``loss.setup_entropy_coef`` slice end to end:

* config default (0.0), validation, and round-trip;
* the buffer's per-transition ``is_setup`` flag round-trips through
  ``add`` / ``get_batch``;
* ``setup_entropy_coef=0.0`` leaves the PPO total loss BYTE-IDENTICAL to
  the pre-slice four-term objective regardless of the ``is_setup`` flags;
* ``setup_entropy_coef>0`` perturbs the loss ONLY through the setup rows'
  entropy — non-setup contributions (policy / value / entropy / belief)
  are bit-for-bit unchanged;
* the ``openings/setup_head_entropy`` TB scalar is emitted (and NOT
  double-logged under ``train/``).

The stub policy returns a CONSTANT joint entropy (1.5), so the setup term
has zero gradient and the arithmetic is exact and deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

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
from catan_rl.ppo.training_loop import _log_update_metrics
from tests.unit.ppo.test_trainer import _fill_random, _StubPolicy

_STUB_ENTROPY = 1.5  # _StubPolicy returns this constant joint entropy.


# ---------------------------------------------------------------------------
# Config: default / validation / round-trip
# ---------------------------------------------------------------------------


class TestSetupEntropyConfig:
    def test_default_is_zero(self) -> None:
        assert LossConfig().setup_entropy_coef == 0.0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="setup_entropy_coef"):
            LossConfig(setup_entropy_coef=-0.1)

    def test_round_trip_via_dict(self) -> None:
        cfg = TrainConfig(loss=LossConfig(setup_entropy_coef=0.02))
        cfg2 = TrainConfig._from_dict(cfg.to_dict())
        assert cfg2.loss.setup_entropy_coef == pytest.approx(0.02)

    def test_round_trip_via_yaml(self, tmp_path: Any) -> None:
        cfg = TrainConfig(loss=LossConfig(setup_entropy_coef=0.03))
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)
        cfg2 = TrainConfig.from_yaml(path)
        assert cfg2.loss.setup_entropy_coef == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Buffer: is_setup round-trips
# ---------------------------------------------------------------------------


class TestBufferIsSetup:
    def test_defaults_false_and_gathered(self) -> None:
        b = CompositeRolloutBuffer(
            n_steps=2,
            n_envs=3,
            obs_spec={"current_player_main": ObsSpec(shape=(8,), dtype=np.dtype(np.float32))},
            mask_spec={"type": MaskSpec(shape=(13,))},
        )
        _fill_two_steps(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(3, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
            advantage_norm="none",
        )
        batch = b.get_batch(np.arange(6), device="cpu")
        assert "is_setup" in batch
        assert batch["is_setup"].dtype == torch.bool
        assert not bool(batch["is_setup"].any())

    def test_stored_flag_round_trips(self) -> None:
        b = CompositeRolloutBuffer(
            n_steps=2,
            n_envs=3,
            obs_spec={"current_player_main": ObsSpec(shape=(8,), dtype=np.dtype(np.float32))},
            mask_spec={"type": MaskSpec(shape=(13,))},
        )
        flags = np.array([True, False, True], dtype=bool)
        _fill_two_steps(b, is_setup=flags)
        np.testing.assert_array_equal(b.is_setup[0], flags)
        np.testing.assert_array_equal(b.is_setup[1], flags)


# ---------------------------------------------------------------------------
# Trainer: byte-identical at coef 0; isolated at coef > 0
# ---------------------------------------------------------------------------


def _cfg(setup_coef: float) -> TrainConfig:
    # Single SGD step (n_epochs=1, one full-buffer batch) so the returned
    # total_loss equals that step's loss exactly — no multi-step averaging.
    return TrainConfig(
        rollout=RolloutConfig(n_envs=4, n_steps=4, vec_env_mode="serial"),
        ppo=PPOConfig(n_epochs=1, batch_size=16, clip_range=0.2, target_kl=10.0),
        gae=GAEConfig(),
        loss=LossConfig(
            value_coef=0.5,
            entropy_coef_start=0.01,
            entropy_coef_end=0.01,
            setup_entropy_coef=setup_coef,
        ),
        optimizer=OptimizerConfig(lr_start=1e-3, lr_end=1e-4, lr_anneal_total_updates=10),
        total_steps=16,
        run_name="test",
    )


def _run_once(cfg: TrainConfig, *, is_setup: np.ndarray | None, init_state: dict) -> UpdateMetrics:
    policy = _StubPolicy()
    policy.load_state_dict(init_state)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr_start)
    trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="cpu")

    buf = CompositeRolloutBuffer(
        n_steps=cfg.rollout.n_steps,
        n_envs=cfg.rollout.n_envs,
        obs_spec={"current_player_main": ObsSpec(shape=(8,), dtype=np.dtype(np.float32))},
        mask_spec={"type": MaskSpec(shape=(13,))},
    )
    _fill_random(buf)
    if is_setup is not None:
        buf.is_setup[:] = is_setup
    buf.compute_returns_and_advantages(
        last_values=np.zeros(cfg.rollout.n_envs, dtype=np.float32),
        gamma=cfg.gae.gamma,
        gae_lambda=cfg.gae.gae_lambda,
        advantage_norm="rollout",
    )
    return trainer.update(buf, update_idx=0, rng=np.random.default_rng(0))


class TestSetupEntropyLoss:
    def test_coef_zero_byte_identical_regardless_of_flags(self) -> None:
        """coef=0 ⇒ total loss independent of is_setup, bit-for-bit."""
        init = {k: v.clone() for k, v in _StubPolicy().state_dict().items()}
        cfg = _cfg(0.0)
        all_false = np.zeros((cfg.rollout.n_steps, cfg.rollout.n_envs), dtype=bool)
        all_true = np.ones((cfg.rollout.n_steps, cfg.rollout.n_envs), dtype=bool)

        m_false = _run_once(cfg, is_setup=all_false, init_state=init)
        m_true = _run_once(cfg, is_setup=all_true, init_state=init)

        # Exact equality: the setup term is fully gated out at coef 0.
        assert m_false.total_loss == m_true.total_loss
        assert m_false.policy_loss == m_true.policy_loss
        assert m_false.value_loss == m_true.value_loss
        assert m_false.entropy_bonus == m_true.entropy_bonus

    def test_coef_zero_matches_no_flag_path(self) -> None:
        """coef=0 with flags set == coef=0 with is_setup left at default."""
        init = {k: v.clone() for k, v in _StubPolicy().state_dict().items()}
        cfg = _cfg(0.0)
        all_true = np.ones((cfg.rollout.n_steps, cfg.rollout.n_envs), dtype=bool)

        m_flagged = _run_once(cfg, is_setup=all_true, init_state=init)
        m_default = _run_once(cfg, is_setup=None, init_state=init)
        assert m_flagged.total_loss == m_default.total_loss

    def test_coef_positive_shifts_loss_by_setup_entropy_only(self) -> None:
        """coef>0 changes total loss ONLY by ``coef * mean(H over setup rows)``;
        the non-setup contributions are bit-for-bit unchanged."""
        init = {k: v.clone() for k, v in _StubPolicy().state_dict().items()}
        coef = 0.1
        cfg = _cfg(coef)
        all_false = np.zeros((cfg.rollout.n_steps, cfg.rollout.n_envs), dtype=bool)
        all_true = np.ones((cfg.rollout.n_steps, cfg.rollout.n_envs), dtype=bool)

        m_none = _run_once(cfg, is_setup=all_false, init_state=init)
        m_setup = _run_once(cfg, is_setup=all_true, init_state=init)

        # No setup rows → no bonus → setup diagnostic is 0.
        assert m_none.setup_head_entropy == pytest.approx(0.0)
        # All setup rows → masked mean of the constant stub entropy.
        assert m_setup.setup_head_entropy == pytest.approx(_STUB_ENTROPY)

        # Non-setup contributions identical (constant entropy ⇒ zero setup
        # gradient ⇒ policy/value paths evolve identically).
        assert m_setup.policy_loss == pytest.approx(m_none.policy_loss)
        assert m_setup.value_loss == pytest.approx(m_none.value_loss)
        assert m_setup.entropy_bonus == pytest.approx(m_none.entropy_bonus)

        # Total loss shifts by exactly -coef * setup_entropy.
        assert m_none.total_loss - m_setup.total_loss == pytest.approx(coef * _STUB_ENTROPY)

    def test_coef_positive_no_setup_rows_matches_coef_zero(self) -> None:
        """coef>0 but zero setup rows ⇒ loss identical to coef=0."""
        init = {k: v.clone() for k, v in _StubPolicy().state_dict().items()}
        all_false = np.zeros((4, 4), dtype=bool)
        m_zero = _run_once(_cfg(0.0), is_setup=all_false, init_state=init)
        m_pos = _run_once(_cfg(0.5), is_setup=all_false, init_state=init)
        assert m_zero.total_loss == pytest.approx(m_pos.total_loss)


# ---------------------------------------------------------------------------
# TB: openings/setup_head_entropy scalar emitted
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars[tag] = float(value)


def _metrics(**over: float) -> UpdateMetrics:
    base: dict[str, Any] = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy_bonus": 1.4,
        "setup_head_entropy": 1.23,
        "belief_loss": 0.0,
        "aux_value_loss": 0.0,
        "total_loss": 0.3,
        "approx_kl": 0.01,
        "clip_frac": 0.05,
        "ratio_mean": 1.0,
        "grad_norm": 0.5,
        "lr": 3e-4,
        "entropy_coef": 0.01,
        "n_epochs_run": 1,
        "n_sgd_steps": 4,
    }
    base.update(over)
    return UpdateMetrics(**base)


class TestSetupEntropyTB:
    def test_scalar_emitted_under_openings_namespace(self) -> None:
        w = _FakeWriter()
        _log_update_metrics(w, _metrics(setup_head_entropy=0.77), global_step=100)
        assert w.scalars["openings/setup_head_entropy"] == pytest.approx(0.77)
        # Namespaced under openings/, NOT double-logged under train/.
        assert "train/setup_head_entropy" not in w.scalars
        # The other fields still flow under train/.
        assert "train/policy_loss" in w.scalars


# ---------------------------------------------------------------------------
# Local buffer-fill helper (2-step, matches the buffer's add contract)
# ---------------------------------------------------------------------------


def _fill_two_steps(buf: CompositeRolloutBuffer, *, is_setup: np.ndarray | None = None) -> None:
    rng = np.random.default_rng(1)
    n = buf.n_envs
    for _ in range(buf.n_steps):
        buf.add(
            obs={"current_player_main": rng.standard_normal((n, 8)).astype(np.float32)},
            action=rng.integers(0, 5, (n, 6)).astype(np.int64),
            per_head_log_prob=rng.standard_normal((n, 6)).astype(np.float32),
            log_prob=rng.standard_normal(n).astype(np.float32),
            value=rng.standard_normal(n).astype(np.float32),
            reward=rng.standard_normal(n).astype(np.float32),
            terminated=np.zeros(n, dtype=bool),
            truncated=np.zeros(n, dtype=bool),
            masks={"type": np.ones((n, 13), dtype=bool)},
            is_setup=is_setup,
        )
