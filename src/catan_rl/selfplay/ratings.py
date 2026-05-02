"""Skill rating system for league policies (Phase 0/3).

Uses Microsoft TrueSkill if the optional ``trueskill`` package is installed;
falls back to a tiny in-house Glicko-2 implementation otherwise so the
``ratings`` extra is genuinely optional. Both backends expose the same API:

  rating = system.create()
  rating, opponent = system.update_match(rating, opponent, result)
  prob = system.expected_win_prob(a, b)

The wrapper is a class so the choice of backend is recorded once and used
consistently. ``RatingTable`` stores per-policy ratings keyed by policy ID
and provides the bulk operations the league + eval harness need (round-robin
scoring, top-K, summary stats).

Phase 0 only constructs and stores ratings; Phase 3 wires PFSP-hard sampling
to consume them. We ship the wrapper now so checkpoint format / TB scalars
are stable across the two phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Detect TrueSkill at import time. If absent, fall back to Glicko-2.
try:
    import trueskill  # type: ignore

    _TRUESKILL_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by the fallback path
    _TRUESKILL_AVAILABLE = False


# ── Backend interface ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rating:
    """Backend-agnostic rating snapshot.

    ``mu`` is the point estimate of skill on a TrueSkill-like scale; ``sigma``
    is the standard deviation (uncertainty). For Glicko-2 these map onto rating
    and rating-deviation respectively, scaled to comparable magnitude.
    """

    mu: float
    sigma: float

    @property
    def conservative(self) -> float:
        """Conservative skill = ``mu - 3·sigma`` (TrueSkill convention)."""
        return self.mu - 3.0 * self.sigma


class RatingSystem:
    """Backend-agnostic rating updates."""

    def __init__(self, *, default_mu: float = 25.0, default_sigma: float = 25.0 / 3.0):
        self.default_mu = default_mu
        self.default_sigma = default_sigma
        self.backend = "trueskill" if _TRUESKILL_AVAILABLE else "glicko2-fallback"
        if _TRUESKILL_AVAILABLE:
            # TrueSkill draw probability is irrelevant for 1v1 zero-sum
            # (no draws in our reward shape — even truncated games have a
            # winner by VP margin), but we set it small to be safe.
            self._env = trueskill.TrueSkill(  # type: ignore[attr-defined]
                mu=default_mu,
                sigma=default_sigma,
                draw_probability=0.0,
            )

    def create(self) -> Rating:
        """Create a fresh rating with the system's defaults."""
        return Rating(self.default_mu, self.default_sigma)

    def update_match(self, a: Rating, b: Rating, *, a_won: bool) -> tuple[Rating, Rating]:
        """Update both ratings given the binary outcome ``a_won``.

        Returns the new (a, b) ratings; the inputs are not mutated.
        """
        if _TRUESKILL_AVAILABLE:
            ts_a = self._env.create_rating(a.mu, a.sigma)  # type: ignore[attr-defined]
            ts_b = self._env.create_rating(b.mu, b.sigma)  # type: ignore[attr-defined]
            if a_won:
                (new_a,), (new_b,) = self._env.rate(  # type: ignore[attr-defined]
                    [(ts_a,), (ts_b,)], ranks=[0, 1]
                )
            else:
                (new_a,), (new_b,) = self._env.rate(  # type: ignore[attr-defined]
                    [(ts_a,), (ts_b,)], ranks=[1, 0]
                )
            return Rating(new_a.mu, new_a.sigma), Rating(new_b.mu, new_b.sigma)
        return _glicko2_update(a, b, a_won=a_won)

    def expected_win_prob(self, a: Rating, b: Rating) -> float:
        """Expected probability that ``a`` beats ``b``.

        Uses the TrueSkill performance-difference formula in both backends so
        we get consistent numbers regardless of which is in use.
        """
        ts_beta = self.default_sigma / 2.0  # TrueSkill default
        denom = math.sqrt(2.0 * ts_beta * ts_beta + a.sigma * a.sigma + b.sigma * b.sigma)
        if denom < 1e-9:
            return 0.5
        return _normal_cdf((a.mu - b.mu) / denom)


# ── Glicko-2 fallback (single-step, simplified) ─────────────────────────────


_GLICKO_Q = math.log(10.0) / 400.0  # Glicko-1 scaling, used inside the fallback


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * _GLICKO_Q * _GLICKO_Q * rd * rd / (math.pi * math.pi))


def _glicko2_update(a: Rating, b: Rating, *, a_won: bool) -> tuple[Rating, Rating]:
    """One-game Glicko-style update.

    Faithful to Glicko-1 (Glickman 1995) but operating on the (mu, sigma) scale
    used by TrueSkill so the two backends produce comparable numbers. This is
    intentionally simpler than full Glicko-2 because we only need approximate
    skill ratings — the league sampler is robust to small inaccuracies.
    """
    expected_a = 1.0 / (1.0 + math.exp(-_GLICKO_Q * _g(b.sigma) * (a.mu - b.mu)))
    expected_b = 1.0 - expected_a
    score_a = 1.0 if a_won else 0.0
    score_b = 1.0 - score_a

    # d² for each player.
    d2_a = 1.0 / (_GLICKO_Q * _GLICKO_Q * _g(b.sigma) ** 2 * expected_a * (1.0 - expected_a) + 1e-9)
    d2_b = 1.0 / (_GLICKO_Q * _GLICKO_Q * _g(a.sigma) ** 2 * expected_b * (1.0 - expected_b) + 1e-9)

    new_mu_a = a.mu + (_GLICKO_Q / (1.0 / (a.sigma * a.sigma) + 1.0 / d2_a)) * _g(b.sigma) * (
        score_a - expected_a
    )
    new_mu_b = b.mu + (_GLICKO_Q / (1.0 / (b.sigma * b.sigma) + 1.0 / d2_b)) * _g(a.sigma) * (
        score_b - expected_b
    )

    new_sigma_a = math.sqrt(1.0 / (1.0 / (a.sigma * a.sigma) + 1.0 / d2_a))
    new_sigma_b = math.sqrt(1.0 / (1.0 / (b.sigma * b.sigma) + 1.0 / d2_b))

    # Floor sigma so a flood of identical match outcomes can't lock skill
    # estimates to artificially tight bounds.
    new_sigma_a = max(new_sigma_a, 0.5)
    new_sigma_b = max(new_sigma_b, 0.5)

    return Rating(new_mu_a, new_sigma_a), Rating(new_mu_b, new_sigma_b)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error-function trick."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Per-population rating table ─────────────────────────────────────────────


@dataclass
class RatingTable:
    """Map of policy ID → Rating, with bulk operations.

    Used by Phase 3's PFSP-hard sampler and by the eval harness's
    ``--mode league-rating`` round-robin.
    """

    system: RatingSystem = field(default_factory=RatingSystem)
    ratings: dict[Any, Rating] = field(default_factory=dict)

    def get(self, key: Any) -> Rating:
        """Return the rating for ``key``, creating one if absent."""
        if key not in self.ratings:
            self.ratings[key] = self.system.create()
        return self.ratings[key]

    def record_match(self, a_key: Any, b_key: Any, *, a_won: bool) -> None:
        """Record a single match between two policies."""
        a = self.get(a_key)
        b = self.get(b_key)
        new_a, new_b = self.system.update_match(a, b, a_won=a_won)
        self.ratings[a_key] = new_a
        self.ratings[b_key] = new_b

    def expected_win_prob(self, a_key: Any, b_key: Any) -> float:
        """Predicted win probability that ``a_key`` beats ``b_key``."""
        return self.system.expected_win_prob(self.get(a_key), self.get(b_key))

    def top_k(self, k: int = 10) -> list[tuple[Any, Rating]]:
        """Return up to ``k`` policies with the highest conservative rating."""
        sorted_items = sorted(self.ratings.items(), key=lambda kv: kv[1].conservative, reverse=True)
        return sorted_items[:k]

    def to_dict(self) -> dict[Any, dict[str, float]]:
        """Serialize to a JSON-friendly dict (the keys are coerced via ``str``)."""
        return {
            str(k): {"mu": v.mu, "sigma": v.sigma, "conservative": v.conservative}
            for k, v in self.ratings.items()
        }
