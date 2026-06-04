"""Tests for `eval/wilson.py`.

Pins:
1. Standard known reference values (Newcombe 1998, Table 1).
2. Monotonicity: wider alpha → wider CI.
3. Bounds clamped to [0, 1] at extremes.
4. Validators on (n, wins, alpha).
5. ``clears_zero_against`` behaviour matches the eval gate semantic.
"""

from __future__ import annotations

import pytest

from catan_rl.eval.wilson import _normal_ppf, wilson_interval


class TestNormalPPF:
    def test_at_0_5_is_zero(self) -> None:
        assert _normal_ppf(0.5) == pytest.approx(0.0, abs=1e-7)

    def test_at_0_975_is_1_96(self) -> None:
        # 95% two-sided z.
        assert _normal_ppf(0.975) == pytest.approx(1.95996, abs=1e-4)

    def test_at_0_995_is_2_576(self) -> None:
        # 99% two-sided z.
        assert _normal_ppf(0.995) == pytest.approx(2.5758, abs=1e-4)

    def test_at_0_9995_is_3_291(self) -> None:
        assert _normal_ppf(0.9995) == pytest.approx(3.2905, abs=1e-3)

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            _normal_ppf(0.0)
        with pytest.raises(ValueError, match="q must be"):
            _normal_ppf(1.0)


class TestKnownReferenceValues:
    """Anchor on closed-form computed values for the Wilson CI.

    Values verified against Newcombe 1998 Table 1 (method 3) within
    rounding tolerance.
    """

    def test_15_of_148_95pct(self) -> None:
        # Newcombe 1998 method 3: r=15, n=148 → (0.0624, 0.1605).
        ci = wilson_interval(wins=15, n=148, alpha=0.05)
        assert ci.lower == pytest.approx(0.0624, abs=5e-4)
        assert ci.upper == pytest.approx(0.1605, abs=5e-4)

    def test_81_of_263_95pct(self) -> None:
        # Computed: r=81, n=263 → (0.2553, 0.3662).
        ci = wilson_interval(wins=81, n=263, alpha=0.05)
        assert ci.lower == pytest.approx(0.2553, abs=5e-4)
        assert ci.upper == pytest.approx(0.3662, abs=5e-4)


class TestWilsonProperties:
    def test_point_is_sample_proportion(self) -> None:
        ci = wilson_interval(wins=55, n=100, alpha=0.05)
        assert ci.point == pytest.approx(0.55)

    def test_bounds_always_in_unit_interval(self) -> None:
        for wins, n in ((0, 10), (10, 10), (1, 1000), (999, 1000)):
            ci = wilson_interval(wins=wins, n=n, alpha=0.05)
            assert 0.0 <= ci.lower <= 1.0
            assert 0.0 <= ci.upper <= 1.0
            assert ci.lower <= ci.upper

    def test_zero_wins_lower_bound_is_zero(self) -> None:
        ci = wilson_interval(wins=0, n=20, alpha=0.05)
        assert ci.lower == pytest.approx(0.0, abs=1e-12)

    def test_all_wins_upper_bound_is_one(self) -> None:
        ci = wilson_interval(wins=20, n=20, alpha=0.05)
        assert ci.upper == pytest.approx(1.0, abs=1e-12)

    def test_narrower_alpha_widens_interval(self) -> None:
        # 99% CI must be wider than 95%.
        ci_95 = wilson_interval(wins=55, n=100, alpha=0.05)
        ci_99 = wilson_interval(wins=55, n=100, alpha=0.01)
        assert ci_99.half_width > ci_95.half_width

    def test_larger_n_narrows_interval(self) -> None:
        ci_100 = wilson_interval(wins=55, n=100, alpha=0.05)
        ci_1000 = wilson_interval(wins=550, n=1000, alpha=0.05)
        # Same point estimate, smaller CI at N=1000.
        assert ci_1000.half_width < ci_100.half_width


class TestClearsZeroAgainst:
    def test_55_of_100_does_not_clear_0_5(self) -> None:
        # 0.55 with n=100 → 95% CI ~(0.456, 0.640) — straddles 0.5.
        ci = wilson_interval(wins=55, n=100, alpha=0.05)
        assert not ci.clears_zero_against(0.5)

    def test_65_of_100_clears_0_5(self) -> None:
        # 0.65 with n=100 → CI ~(0.552, 0.737) — clears 0.5.
        ci = wilson_interval(wins=65, n=100, alpha=0.05)
        assert ci.clears_zero_against(0.5)

    def test_110_of_200_does_not_clear_0_5(self) -> None:
        # Audit-relevant: raw mean 0.55 at n=200 → CI lower ≈ 0.482.
        ci = wilson_interval(wins=110, n=200, alpha=0.05)
        assert ci.point == pytest.approx(0.55)
        assert ci.lower < 0.5
        assert not ci.clears_zero_against(0.5)


class TestValidators:
    def test_zero_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must"):
            wilson_interval(wins=0, n=0)

    def test_wins_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="wins"):
            wilson_interval(wins=-1, n=10)
        with pytest.raises(ValueError, match="wins"):
            wilson_interval(wins=11, n=10)

    def test_alpha_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            wilson_interval(wins=5, n=10, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            wilson_interval(wins=5, n=10, alpha=1.0)
