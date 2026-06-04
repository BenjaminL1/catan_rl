"""Wilson score interval for binomial proportions.

Why Wilson and not normal-approx (``p ± z * sqrt(p(1-p)/n)``):

* The normal approximation under-covers at small N (it gives an
  interval narrower than reality), so a ``p=0.55, n=200`` raw-mean
  delta could falsely "clear zero" when in fact it doesn't.
* The Wilson interval is asymmetric and pulls the bounds toward 0.5,
  which is the correct behaviour for evidence-of-a-coin-bias tests.

This is the same CI Phase A.4 / B.5 acceptance gates require and the
plan doc cites. Kept in its own module so the eval harness, future
diagnostic scripts, and league rating updates can all share one
implementation.

References
----------
* Wilson, E. B. (1927). "Probable inference, the law of succession,
  and statistical inference." JASA 22(158): 209-212.
* Newcombe, R. G. (1998). "Two-sided confidence intervals for the
  single proportion: comparison of seven methods." Stat. Med. 17(8):
  857-872. (Wilson is method 3.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WilsonInterval:
    """Closed-form Wilson CI for a binomial proportion."""

    point: float
    """Sample proportion (``wins / n``)."""

    lower: float
    """Lower bound of the CI."""

    upper: float
    """Upper bound of the CI."""

    n: int
    """Number of trials."""

    alpha: float
    """Significance level — the CI is ``(1 - alpha) * 100%``."""

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def clears_zero_against(self, baseline: float) -> bool:
        """``True`` iff the CI strictly excludes ``baseline``.

        Used by the eval gate: ``EvalHarness`` declares a "significant
        improvement vs heuristic" when the WR CI's lower bound is
        strictly above 0.5.
        """
        return self.lower > baseline


def wilson_interval(
    *,
    wins: int,
    n: int,
    alpha: float = 0.05,
) -> WilsonInterval:
    """Wilson score CI for a binomial proportion.

    Args:
        wins: number of successes.
        n: number of trials. Must be > 0.
        alpha: significance level; default 0.05 → 95% CI.

    Returns:
        A :class:`WilsonInterval` with the point estimate and bounds.

    Examples:
        At ``wins=110, n=200, alpha=0.05``: 95% CI is roughly
        ``(0.482, 0.615)`` — the lower bound does NOT clear 0.5, so
        the gate should NOT promote the candidate.

        At ``wins=130, n=200, alpha=0.05``: 95% CI is roughly
        ``(0.583, 0.708)`` — clears 0.5.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not (0 <= wins <= n):
        raise ValueError(f"wins ({wins}) must be in [0, n] for n={n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p = wins / n
    # z for two-sided test at significance ``alpha``.
    # phi_inv(1 - alpha/2) via the std-normal CDF inverse.
    z = _normal_ppf(1.0 - alpha / 2.0)

    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) / denom
    lower = centre - margin
    upper = centre + margin
    # Clamp to [0, 1] — Wilson keeps these in range analytically but
    # float arithmetic can drift by ~1e-16.
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))
    return WilsonInterval(point=p, lower=lower, upper=upper, n=n, alpha=alpha)


# ---------------------------------------------------------------------------
# Std-normal PPF — rational approximation per Beasley-Springer-Moro.
# Avoids a scipy dependency for this single inverse.
# ---------------------------------------------------------------------------


def _normal_ppf(q: float) -> float:
    """Inverse of the std-normal CDF.

    Beasley-Springer-Moro rational approximation; max absolute error is
    well under 1e-7 for the alpha values we care about (alpha = 0.05,
    0.01, 0.001 → z = 1.96, 2.576, 3.291).
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    # Coefficients from Wichura (1988) / Moro (1995).
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if q < p_low:
        r = math.sqrt(-2.0 * math.log(q))
        num = ((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]
        den = (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0
        return num / den
    if q <= p_high:
        r = q - 0.5
        r2 = r * r
        num = (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * r
        den = ((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1.0
        return num / den
    r = math.sqrt(-2.0 * math.log(1.0 - q))
    num = ((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]
    den = (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0
    return -num / den
