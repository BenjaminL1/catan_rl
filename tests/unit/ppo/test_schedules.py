"""Tests for `ppo/schedules.py`."""

from __future__ import annotations

import pytest

from catan_rl.ppo.schedules import linear_entropy_coef_schedule, linear_lr_schedule


class TestLinearLR:
    def test_start_at_lr_start(self) -> None:
        assert linear_lr_schedule(
            update_idx=0, lr_start=3e-4, lr_end=1e-5, total_updates=100
        ) == pytest.approx(3e-4)

    def test_end_at_lr_end(self) -> None:
        assert linear_lr_schedule(
            update_idx=99, lr_start=3e-4, lr_end=1e-5, total_updates=100
        ) == pytest.approx(1e-5)

    def test_holds_at_lr_end_after_total(self) -> None:
        assert linear_lr_schedule(
            update_idx=500, lr_start=3e-4, lr_end=1e-5, total_updates=100
        ) == pytest.approx(1e-5)

    def test_midpoint_linear(self) -> None:
        # Halfway between 0 and 99 → ~(start + end) / 2
        v = linear_lr_schedule(update_idx=49, lr_start=3e-4, lr_end=1e-5, total_updates=100)
        # Idx 49 / 99 ≈ 0.4949; close to but not exactly midpoint.
        expected = 3e-4 + (1e-5 - 3e-4) * (49 / 99)
        assert v == pytest.approx(expected, abs=1e-9)

    def test_total_updates_one_holds_at_start(self) -> None:
        # Single update → no anneal possible; stays at lr_start.
        assert linear_lr_schedule(update_idx=0, lr_start=1.0, lr_end=0.0, total_updates=1) == 1.0
        assert linear_lr_schedule(update_idx=10, lr_start=1.0, lr_end=0.0, total_updates=1) == 1.0

    def test_invalid_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_updates"):
            linear_lr_schedule(update_idx=0, lr_start=1.0, lr_end=0.0, total_updates=0)
        with pytest.raises(ValueError, match="update_idx"):
            linear_lr_schedule(update_idx=-1, lr_start=1.0, lr_end=0.0, total_updates=10)


class TestLinearEntropyCoef:
    def test_holds_at_start_before_anneal(self) -> None:
        for idx in (0, 10, 49, 50):
            assert linear_entropy_coef_schedule(
                update_idx=idx,
                coef_start=0.04,
                coef_end=0.005,
                start_update=50,
                end_update=200,
            ) == pytest.approx(0.04)

    def test_at_end_returns_coef_end(self) -> None:
        assert linear_entropy_coef_schedule(
            update_idx=200,
            coef_start=0.04,
            coef_end=0.005,
            start_update=50,
            end_update=200,
        ) == pytest.approx(0.005)
        # And past
        assert linear_entropy_coef_schedule(
            update_idx=500,
            coef_start=0.04,
            coef_end=0.005,
            start_update=50,
            end_update=200,
        ) == pytest.approx(0.005)

    def test_midpoint_linear(self) -> None:
        # Midway through the anneal window
        v = linear_entropy_coef_schedule(
            update_idx=125,
            coef_start=0.04,
            coef_end=0.005,
            start_update=50,
            end_update=200,
        )
        expected = 0.04 + (0.005 - 0.04) * 0.5
        assert v == pytest.approx(expected, abs=1e-9)

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="end_update"):
            linear_entropy_coef_schedule(
                update_idx=0,
                coef_start=0.04,
                coef_end=0.005,
                start_update=200,
                end_update=50,
            )
