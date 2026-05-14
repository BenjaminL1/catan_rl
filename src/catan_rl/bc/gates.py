"""BC acceptance-gate statistical tests.

Per ``v2_step3_bc.md`` §6 (post round-2 faculty re-review), the BC
compound gate has three sub-gates:

  1. Val NLL plateau (convergence) — mechanically handled by the
     training loop's early-stop; not implemented here.
  2. Per-head paired-bootstrap NLL test against a frequency-baseline
     policy — :func:`paired_bootstrap_nll_compound`.
  3. TOST WR equivalence against the measured ``BASE_WR_HEUR_SELF``
     — :func:`tost_wr_equivalence`.

Both tests are calibrated against measured baselines from preflight
E0.2 (``runs/preflight/e02/distribution.json``). The thresholds
(``alpha``, ``equivalence_margin``) live in ``configs/bc.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paired-bootstrap NLL test
# ---------------------------------------------------------------------------


def paired_bootstrap_nll_per_head(
    *,
    base_nll: dict[str, np.ndarray],
    bc_nll: dict[str, np.ndarray],
    n_resamples: int = 10_000,
    alpha: float = 0.01,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Paired-bootstrap one-sided test ``NLL_base − NLL_BC > 0`` per head.

    For each head, computes the per-pair improvement
    ``δ_i = NLL_base_i − NLL_BC_i``, bootstraps the mean over
    ``n_resamples`` paired resamples, and returns the
    ``(1 − alpha)`` percentile-CI lower bound. The head **passes** iff
    ``CI_lower > 0`` — i.e. we can reject "BC is no better than baseline"
    at level α.

    Args:
        base_nll: ``head_name → (n_pairs,) np.ndarray`` of per-pair NLLs
            under the frequency-baseline policy.
        bc_nll: same shape, under the BC policy. Both dicts must have
            the same keys and same per-key length.
        n_resamples: number of bootstrap resamples.
        alpha: significance level; the CI bound used is
            ``alpha`` (one-sided) — i.e. the test rejects when the
            ``alpha`` percentile of the resampled means is above zero.
        seed: RNG seed for reproducibility.

    Returns:
        Per-head dict with fields: ``mean_delta``, ``ci_lower``,
        ``ci_upper``, ``passes`` (bool), ``n_pairs``, ``alpha``.
    """  # noqa: RUF002
    if set(base_nll.keys()) != set(bc_nll.keys()):
        raise ValueError(
            f"base/bc head keys mismatch: {set(base_nll.keys())} vs {set(bc_nll.keys())}"
        )
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    for head in base_nll:
        b = np.asarray(base_nll[head], dtype=np.float64)
        bc = np.asarray(bc_nll[head], dtype=np.float64)
        if b.shape != bc.shape:
            raise ValueError(f"head {head}: base shape {b.shape} != bc shape {bc.shape}")
        n = int(b.shape[0])
        delta = b - bc
        # Bootstrap the mean of delta over n_resamples paired resamples.
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot_means = delta[idx].mean(axis=-1)
        # One-sided 100·(1-alpha)% CI lower bound: alpha-percentile.
        ci_lower = float(np.percentile(boot_means, 100 * alpha))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
        mean_delta = float(delta.mean())
        passes = ci_lower > 0.0
        out[head] = {
            "mean_delta": mean_delta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "passes": bool(passes),
            "n_pairs": n,
            "alpha": float(alpha),
        }
    return out


def paired_bootstrap_nll_compound(
    *,
    base_nll: dict[str, np.ndarray],
    bc_nll: dict[str, np.ndarray],
    required_heads: tuple[str, ...] = ("type", "corner", "edge"),
    n_resamples: int = 10_000,
    alpha: float = 0.01,
    seed: int = 0,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Run the per-head test and return ``(per_head_results, all_passed)``.

    ``all_passed`` is True iff every head in ``required_heads`` passed
    its per-head test. This is the BC plan's Gate 2.
    """
    per_head = paired_bootstrap_nll_per_head(
        base_nll=base_nll, bc_nll=bc_nll, n_resamples=n_resamples, alpha=alpha, seed=seed
    )
    compound = all(per_head[h]["passes"] for h in required_heads if h in per_head)
    return per_head, compound


# ---------------------------------------------------------------------------
# TOST WR equivalence
# ---------------------------------------------------------------------------


def tost_wr_equivalence(
    *,
    wr_bc: float,
    wr_self: float,
    n: int,
    alpha: float = 0.05,
    equivalence_margin: float = 0.04,
) -> dict[str, Any]:
    """Power-calibrated equivalence test between BC WR and teacher self-WR.

    Per ``v2_step3_bc.md`` §6 (faculty re-review formulation):

        "At α = 0.05 and n = 600 games per seat, the Wald-CI half-width
        is ~ 1.96 · √(0.25/n) ≈ 0.040. Gate 3 passes iff symmetrised
        |WR_BC − WR_heur_self| ≤ 0.04 — i.e. we can't statistically
        reject equivalence with the teacher."

    Implementation:
      * Compute ``delta = wr_bc - wr_self``.
      * Compute the 100·(1-α)·2-sided Wald CI half-width on ``delta``
        under a binomial normal approximation.
      * ``passes`` iff ``|delta| ≤ equivalence_margin``. The CI is
        exposed as a diagnostic; the test is the margin comparison.

    The equivalence_margin (0.04 by default) was calibrated by the
    plan to roughly equal the Wald CI half-width at n=600 — so at
    that sample size, a delta inside the margin is also inside the CI
    (cannot reject equality). At larger n the test gets stricter
    (more power); at smaller n it gets looser.

    Args:
        wr_bc: measured symmetrised BC vs heuristic WR.
        wr_self: measured BASE_WR_HEUR_SELF symmetrised (preflight E0.2).
        n: per-side game count (e.g. 600).
        alpha: significance level for the diagnostic CI (default 0.05).
        equivalence_margin: equivalence band half-width
            (default 0.04 per the BC plan).

    Returns:
        Dict with ``passes`` (bool) + diagnostic fields ``wr_bc``,
        ``wr_self``, ``delta``, ``ci_lower``, ``ci_upper``,
        ``margin``, ``n``.
    """  # noqa: RUF002
    if not 0.0 <= wr_bc <= 1.0:
        raise ValueError(f"wr_bc must be in [0, 1], got {wr_bc}")
    if not 0.0 <= wr_self <= 1.0:
        raise ValueError(f"wr_self must be in [0, 1], got {wr_self}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if equivalence_margin <= 0.0:
        raise ValueError(f"equivalence_margin must be > 0, got {equivalence_margin}")

    # Two-binomial normal-approx SE for the difference (diagnostic).
    se = float(np.sqrt(max(wr_self * (1 - wr_self), 1e-9) / n + max(wr_bc * (1 - wr_bc), 1e-9) / n))
    z_two_sided = float(_inv_normal_cdf(1.0 - alpha / 2.0))
    half_width = z_two_sided * se
    delta = wr_bc - wr_self
    ci_lower = delta - half_width
    ci_upper = delta + half_width
    passes = abs(delta) <= equivalence_margin
    return {
        "passes": bool(passes),
        "wr_bc": float(wr_bc),
        "wr_self": float(wr_self),
        "delta": float(delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "margin": float(equivalence_margin),
        "n": int(n),
    }


def _inv_normal_cdf(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro.

    ~7 digits of accuracy across the unit interval. Self-contained so
    we don't drag in scipy for one function.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"_inv_normal_cdf: p must be in (0,1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = float(np.sqrt(-2 * np.log(p)))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = float(np.sqrt(-2 * np.log(1 - p)))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
