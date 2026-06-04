"""Tests for `algorithms/pikl/config.py`.

Pins:
1. Default is disabled with lambda_kl=0.
2. lambda_kl<0 / warmup<0 / bad estimator / enabled-without-path raise.
3. coef_at handles disabled, no-warmup, and the linear ramp correctly.
"""

from __future__ import annotations

import pytest

from catan_rl.algorithms.pikl.config import PiKLConfig


class TestConstruction:
    def test_default_is_disabled(self) -> None:
        c = PiKLConfig()
        assert c.enabled is False
        assert c.lambda_kl == 0.0
        assert c.anchor_checkpoint_path is None
        assert c.kl_estimator == "k3"
        assert c.warmup_updates == 0

    def test_enabled_requires_path(self) -> None:
        with pytest.raises(ValueError, match="anchor_checkpoint_path"):
            PiKLConfig(enabled=True, lambda_kl=0.05)

    def test_negative_lambda_rejected(self) -> None:
        with pytest.raises(ValueError, match="lambda_kl"):
            PiKLConfig(lambda_kl=-0.1)

    def test_negative_warmup_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_updates"):
            PiKLConfig(warmup_updates=-1)

    def test_bad_estimator_rejected(self) -> None:
        with pytest.raises(ValueError, match="kl_estimator"):
            PiKLConfig(kl_estimator="k99")  # type: ignore[arg-type]


class TestCoefAt:
    def test_disabled_is_zero(self) -> None:
        c = PiKLConfig(enabled=False, lambda_kl=0.5)
        assert c.coef_at(0) == 0.0
        assert c.coef_at(1000) == 0.0

    def test_no_warmup_is_constant(self) -> None:
        c = PiKLConfig(
            enabled=True,
            lambda_kl=0.05,
            anchor_checkpoint_path="dummy",
            warmup_updates=0,
        )
        assert c.coef_at(0) == 0.05
        assert c.coef_at(123) == 0.05

    def test_warmup_ramps_linearly(self) -> None:
        c = PiKLConfig(
            enabled=True,
            lambda_kl=1.0,
            anchor_checkpoint_path="dummy",
            warmup_updates=10,
        )
        assert c.coef_at(0) == 0.0
        assert c.coef_at(5) == pytest.approx(0.5)
        assert c.coef_at(10) == 1.0
        assert c.coef_at(20) == 1.0  # clamped at the target

    def test_negative_update_idx_treated_as_zero(self) -> None:
        c = PiKLConfig(
            enabled=True,
            lambda_kl=1.0,
            anchor_checkpoint_path="dummy",
            warmup_updates=10,
        )
        assert c.coef_at(-1) == 0.0
