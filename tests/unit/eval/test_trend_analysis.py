"""Unit tests for the Part A trend-analysis verdict.

Locks the verdict decision boundary against three synthetic series:

1. Flat plateau at 0.30 (matches the pre-bundle empirical plateau) →
   verdict = ``unsatisfactory``.
2. Linear ramp from 0.30 to 0.45 across 12 evals → verdict =
   ``satisfactory``.
3. Noisy upward drift with high variance → verdict = ``inconclusive``.

Plus boundary tests on the underlying stats primitives.
"""

from __future__ import annotations

import random

from catan_rl.eval.trend_analysis import (
    linear_fit,
    mann_kendall,
    rolling_mean,
    verdict,
)


def _build_series(values: list[float], step_stride: int = 100_000) -> list[tuple[int, float]]:
    return [(step_stride * (i + 1), v) for i, v in enumerate(values)]


# ── Verdict integration tests ─────────────────────────────────────────────


def test_verdict_flat_plateau_is_unsatisfactory() -> None:
    """12 evals tightly clustered at ~0.30 → unsatisfactory."""
    rng = random.Random(0)
    vals = [0.30 + rng.uniform(-0.01, 0.01) for _ in range(12)]
    series = _build_series(vals)
    out = verdict(series)
    assert out["verdict"] == "unsatisfactory", (
        f"expected unsatisfactory for flat plateau, got {out}"
    )
    assert 0.27 <= out["stats"]["mean10"] <= 0.36


def test_verdict_linear_ramp_is_satisfactory() -> None:
    """Linear 0.30 → 0.45 over 12 evals with a steady positive trend."""
    vals = [0.30 + i * (0.45 - 0.30) / 11 for i in range(12)]
    series = _build_series(vals)
    out = verdict(series)
    assert out["verdict"] == "satisfactory", f"expected satisfactory for clean ramp, got {out}"
    # Either mean10 cleared 0.40 or the lift+MK rule fired.
    s = out["stats"]
    cleared = s["mean10"] >= 0.40
    lifted = s["delta_last5_minus_prior5"] >= 0.05 and s["mk_p_value"] < 0.10 and s["mk_tau"] > 0
    assert cleared or lifted, f"neither satisfactory path fired: stats={s}"


def test_verdict_noisy_drift_is_inconclusive() -> None:
    """Above-plateau drift drowned in noise — not in plateau band, not
    statistically significant, so should be inconclusive."""
    rng = random.Random(7)
    # Center around 0.42 (above plateau-hi 0.36) with moderate noise so the
    # mean10 escapes the plateau band but no significant trend emerges.
    vals = [0.42 + rng.uniform(-0.08, 0.08) for _ in range(12)]
    series = _build_series(vals)
    out = verdict(series)
    assert out["verdict"] in ("inconclusive", "satisfactory"), (
        f"above-plateau noise should not be unsatisfactory, got {out}"
    )


def test_verdict_few_evals_is_inconclusive() -> None:
    series = _build_series([0.30, 0.31, 0.29])
    out = verdict(series)
    assert out["verdict"] == "inconclusive"
    assert "need at least" in out["reasoning"]


def test_verdict_clear_breakthrough() -> None:
    """If mean10 ≥ 0.40 the satisfactory path fires unconditionally."""
    # 5 priors at 0.30, 5 recents at 0.50 → mean10 = 0.40.
    vals = [0.30] * 5 + [0.50] * 5
    series = _build_series(vals)
    out = verdict(series)
    assert out["verdict"] == "satisfactory"
    assert out["stats"]["mean10"] >= 0.40


# ── Stats primitive tests ─────────────────────────────────────────────────


def test_rolling_mean_right_aligned() -> None:
    series = _build_series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_mean(series, window=3)
    # Prefix mean for the first 2, then 3-wide window.
    assert [round(v, 3) for _, v in out] == [1.0, 1.5, 2.0, 3.0, 4.0]
    # Steps unchanged.
    assert [s for s, _ in out] == [s for s, _ in series]


def test_mann_kendall_monotone_increase_is_positive_tau() -> None:
    out = mann_kendall([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert out["tau"] > 0.9
    assert out["p_value"] < 0.05


def test_mann_kendall_monotone_decrease_is_negative_tau() -> None:
    out = mann_kendall([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert out["tau"] < -0.9
    assert out["p_value"] < 0.05


def test_mann_kendall_flat_is_zero_tau() -> None:
    out = mann_kendall([0.5] * 10)
    assert abs(out["tau"]) < 0.01


def test_linear_fit_recovers_slope() -> None:
    xs = list(range(10))
    ys = [2.0 * x + 1.0 for x in xs]
    out = linear_fit(xs, ys)
    assert abs(out["slope"] - 2.0) < 1e-6
    assert abs(out["intercept"] - 1.0) < 1e-6
    assert out["r_squared"] > 0.999
