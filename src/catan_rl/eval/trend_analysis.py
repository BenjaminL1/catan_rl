"""Trend-analysis statistics for evaluation-trajectory verdicts.

Pure-function module exporting trend-detection primitives (rolling mean,
Mann-Kendall, linear regression) and a single mechanical ``verdict()``
that consumes a (step, win_rate) timeseries and returns one of
``satisfactory`` / ``unsatisfactory`` / ``inconclusive``.

The verdict logic encodes the Part A protocol from the plateau-breaking
plan: after a config bundle lands, run ~5M more training steps and check
whether the WR-vs-heuristic eval mean has actually shifted upward, or
whether the agent is genuinely stuck in the plateau band.

Designed to be testable in isolation — no TB I/O here. The CLI wrapper
``scripts/analyze_trend.py`` is responsible for reading the events file
and passing the (step, wr) pairs to ``verdict()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# ── TB I/O (kept here so callers don't have to recreate the boilerplate) ──


def read_tb_scalar(events_path: str, tag: str) -> list[tuple[int, float]]:
    """Read a single scalar tag from a TF events file.

    Returns a list of ``(step, value)`` pairs in the order written. Empty
    list if the tag isn't present (so callers can detect "this scalar
    wasn't logged" without an exception path).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(events_path, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]


# ── Rolling statistics ────────────────────────────────────────────────────


def rolling_mean(series: Sequence[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    """Right-aligned rolling mean. Output is the same length as input.

    For positions ``i < window-1`` the mean is taken over the available
    prefix ``series[:i+1]`` — no padding, no NaNs.
    """
    out: list[tuple[int, float]] = []
    if not series:
        return out
    vals = [v for _, v in series]
    for i in range(len(series)):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        out.append((series[i][0], sum(chunk) / len(chunk)))
    return out


# ── Trend tests ───────────────────────────────────────────────────────────


def mann_kendall(values: Sequence[float]) -> dict[str, float]:
    """Two-sided Mann-Kendall trend test.

    Returns ``{'tau', 'p_value', 'n'}``. ``tau`` is Kendall's tau-b in
    [-1, +1]. Positive = increasing trend. ``p_value`` is from the normal
    approximation (valid for n >= 8); below that, it's still returned but
    the test is underpowered.

    Uses ``scipy.stats.kendalltau`` when available (handles ties properly);
    otherwise falls back to a hand-rolled implementation that matches
    SciPy on the no-ties case.
    """
    n = len(values)
    result = {"tau": 0.0, "p_value": 1.0, "n": float(n)}
    if n < 3:
        return result
    try:
        from scipy.stats import kendalltau

        steps = list(range(n))
        tau, p_value = kendalltau(steps, list(values))
        result["tau"] = float(tau) if not math.isnan(tau) else 0.0
        result["p_value"] = float(p_value) if not math.isnan(p_value) else 1.0
        return result
    except ImportError:
        pass
    # Manual fallback: count concordant/discordant pairs, no tie correction.
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if values[j] > values[i]:
                s += 1
            elif values[j] < values[i]:
                s -= 1
    denom = n * (n - 1) // 2
    tau = s / denom if denom > 0 else 0.0
    # Normal-approximation p-value with variance n(n-1)(2n+5)/18.
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s > 0:
        z = (s - (1 if s > 0 else (-1 if s < 0 else 0))) / math.sqrt(var_s)
        # Two-sided p from the standard normal CDF (math.erf-based).
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    else:
        p_value = 1.0
    result["tau"] = tau
    result["p_value"] = p_value
    return result


def linear_fit(steps: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """OLS linear regression ``values ~ slope * steps + intercept``.

    Returns ``{'slope', 'intercept', 'r_squared', 'p_value', 'std_err', 'n'}``.
    Uses ``scipy.stats.linregress`` when available; otherwise a manual OLS
    fit with an approximate t-test p-value.
    """
    n = len(steps)
    result = {
        "slope": 0.0,
        "intercept": 0.0,
        "r_squared": 0.0,
        "p_value": 1.0,
        "std_err": 0.0,
        "n": float(n),
    }
    if n < 3:
        return result
    try:
        from scipy.stats import linregress

        fit = linregress(list(steps), list(values))
        result["slope"] = float(fit.slope)
        result["intercept"] = float(fit.intercept)
        result["r_squared"] = float(fit.rvalue) ** 2
        result["p_value"] = float(fit.pvalue)
        result["std_err"] = float(fit.stderr)
        return result
    except ImportError:
        pass
    # Manual OLS.
    sx = sum(steps)
    sy = sum(values)
    mx = sx / n
    my = sy / n
    num = sum((x - mx) * (y - my) for x, y in zip(steps, values, strict=False))
    den = sum((x - mx) ** 2 for x in steps)
    if den <= 0:
        return result
    slope = num / den
    intercept = my - slope * mx
    y_hat = [slope * x + intercept for x in steps]
    ss_tot = sum((y - my) ** 2 for y in values)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(values, y_hat, strict=False))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    result["slope"] = slope
    result["intercept"] = intercept
    result["r_squared"] = r2
    # Approximate p-value via t-test on slope; conservative when scipy missing.
    if n > 2 and ss_res > 0:
        se = math.sqrt(ss_res / (n - 2)) / math.sqrt(den)
        result["std_err"] = se
        # Two-sided p ≈ 2 * (1 - Φ(|t|)) for large n.
        t = abs(slope) / se if se > 0 else 0.0
        result["p_value"] = math.erfc(t / math.sqrt(2.0))
    return result


# ── Verdict ───────────────────────────────────────────────────────────────


def verdict(
    series: Sequence[tuple[int, float]],
    *,
    prior_plateau_lo: float = 0.27,
    prior_plateau_hi: float = 0.36,
    satisfactory_mean_threshold: float = 0.40,
    lift_threshold: float = 0.05,
    mk_alpha: float = 0.10,
    delta_band: float = 0.02,
    min_evals: int = 10,
) -> dict[str, Any]:
    """Apply the Part A verdict rule to a (step, wr) timeseries.

    Args:
        series: List of ``(step, win_rate)`` pairs in step-ascending order.
        prior_plateau_lo: Lower bound of the "stuck" band. Defaults to the
            empirical plateau observed before the plateau-breaking bundle.
        prior_plateau_hi: Upper bound of the "stuck" band.
        satisfactory_mean_threshold: 10-eval rolling-mean WR above which the
            verdict is unconditionally ``satisfactory``.
        lift_threshold: Required positive ``mean_last5 - mean_prior5`` lift
            for a "satisfactory" verdict on the trend path.
        mk_alpha: Mann-Kendall p-value threshold. Below this = significant.
        delta_band: ``|mean_last5 - mean_prior5|`` band below which the
            change is treated as no-change (used in the unsatisfactory path).
        min_evals: Required number of eval points before any verdict
            besides ``inconclusive`` is allowed.

    Returns:
        ``{'verdict', 'reasoning', 'stats'}``. ``stats`` carries the
        intermediate values so the caller can log / plot them.
    """
    n = len(series)
    if n < min_evals:
        return {
            "verdict": "inconclusive",
            "reasoning": f"only {n} eval points; need at least {min_evals}",
            "stats": {"n": n},
        }

    vals = [v for _, v in series]
    last10 = vals[-10:]
    last5 = vals[-5:]
    prior5 = vals[-10:-5]
    mean10 = sum(last10) / 10
    mean_last5 = sum(last5) / 5
    mean_prior5 = sum(prior5) / 5
    delta = mean_last5 - mean_prior5
    mk = mann_kendall(last10)

    stats = {
        "n": n,
        "mean10": mean10,
        "mean_last5": mean_last5,
        "mean_prior5": mean_prior5,
        "delta_last5_minus_prior5": delta,
        "mk_tau": mk["tau"],
        "mk_p_value": mk["p_value"],
    }

    # Satisfactory path.
    if mean10 >= satisfactory_mean_threshold:
        return {
            "verdict": "satisfactory",
            "reasoning": (
                f"mean10={mean10:.3f} ≥ {satisfactory_mean_threshold}: "
                f"agent has cleared the plateau."
            ),
            "stats": stats,
        }
    if delta >= lift_threshold and mk["p_value"] < mk_alpha and mk["tau"] > 0:
        return {
            "verdict": "satisfactory",
            "reasoning": (
                f"Δ(last5−prior5)={delta:.3f} ≥ {lift_threshold}, "  # noqa: RUF001
                f"MK τ={mk['tau']:.2f} (p={mk['p_value']:.3f}): "
                f"statistically significant upward trend."
            ),
            "stats": stats,
        }

    # Unsatisfactory path.
    if (
        prior_plateau_lo <= mean10 <= prior_plateau_hi
        and mk["p_value"] > mk_alpha
        and abs(delta) < delta_band
    ):
        return {
            "verdict": "unsatisfactory",
            "reasoning": (
                f"mean10={mean10:.3f} in plateau band "
                f"[{prior_plateau_lo}, {prior_plateau_hi}], "
                f"MK p={mk['p_value']:.3f} ≥ {mk_alpha} (no trend), "
                f"|Δ|={abs(delta):.3f} < {delta_band}: stuck."
            ),
            "stats": stats,
        }

    # Otherwise: trend exists but not strong enough, or mean above plateau
    # without a clean lift signal. Need more data.
    return {
        "verdict": "inconclusive",
        "reasoning": (
            f"mean10={mean10:.3f}, Δ={delta:.3f}, MK p={mk['p_value']:.3f}: "
            f"signal present but insufficient for either verdict — collect more evals."
        ),
        "stats": stats,
    }
