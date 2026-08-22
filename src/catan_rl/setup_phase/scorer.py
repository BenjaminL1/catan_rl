"""The fitted setup scorer: board + prior picks -> vertex / edge scores (D1, D7).

A :class:`ScorerWeights` is a linear model over the D1 feature block, plus the
standardisation constants the fit used. Scoring is
``(features - mean) / scale @ weights``, with illegal candidates forced to
``-inf`` so an argmax can never select one.

**Vehicle neutrality is structural, not a promise** (D7). This module imports
numpy and the engine-facing feature block and nothing from the TRAINING stack —
no ``catan_rl.bc``, no fine-tune code, no ``catan_rl.gui``, no checkpoint
loader. Scoring needs a board, not a policy. The artifact is therefore droppable
into either downstream vehicle (synthetic-corpus fine-tune, or setup-node search
priors) and the choice stays a post-gate owner decision. An import-graph test
pins this.

``torch`` (and, through ``catan_rl.policy.__init__``, the ``CatanPolicy`` class)
is reachable transitively, because the shared schema constants live under
``catan_rl.policy`` — exactly the route ``setup_phase.analytic_value`` already
took. That is a property of the package layout, not a dependency of the scorer:
nothing here builds a network or loads a checkpoint. The import-graph test names
what it actually enforces rather than asserting something the tree does not
support.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_rl.labeling.scenario_gen import Pick
from catan_rl.setup_phase.scorer_features import (
    FEATURE_VERSION,
    N_EDGES,
    N_VERTICES,
    ROAD_FEATURE_NAMES,
    SETTLEMENT_FEATURE_NAMES,
    SetupContext,
    all_road_features,
    all_settlement_features,
)


class ScorerVersionError(ValueError):
    """Raised when a weights artifact does not match :data:`FEATURE_VERSION`."""


class ScorerOverwriteError(ValueError):
    """Raised when a save would replace an artifact carrying the SAME ``version``
    with DIFFERENT weights (spec D6: "each refit bumps the artifact version")."""


@dataclass(frozen=True)
class ScorerWeights:
    """One head's fitted linear weights over a named, ordered feature block."""

    feature_names: tuple[str, ...]
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_version: str = FEATURE_VERSION

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        for name, arr in (("weights", self.weights), ("mean", self.mean), ("scale", self.scale)):
            if np.asarray(arr).shape != (n,):
                raise ValueError(
                    f"{name} must have one entry per feature ({n}), got {np.asarray(arr).shape}"
                )
        if np.any(np.asarray(self.scale) <= 0.0):
            raise ValueError("scale entries must be strictly positive")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score a ``(n_candidates, n_features)`` design block."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"features must be (n, {len(self.feature_names)}), got {x.shape}")
        return ((x - self.mean) / self.scale) @ self.weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "mean": [float(m) for m in self.mean],
            "scale": [float(s) for s in self.scale],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScorerWeights:
        version = str(payload.get("feature_version", ""))
        if version != FEATURE_VERSION:
            raise ScorerVersionError(
                f"weights were fitted against feature_version {version!r} but this "
                f"build computes {FEATURE_VERSION!r}. The design matrix has changed; "
                f"refit rather than scoring an old weight vector against new columns."
            )
        return cls(
            feature_names=tuple(str(n) for n in payload["feature_names"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            feature_version=version,
        )


@dataclass(frozen=True)
class SetupScorer:
    """The two fitted heads plus the provenance stamped alongside them."""

    settlement: ScorerWeights
    road: ScorerWeights
    version: str
    provenance: dict[str, Any]

    def score_vertices(
        self,
        board: Any,
        prior_picks: Sequence[Pick],
        acting_player: int,
        legal_settlements: np.ndarray,
    ) -> np.ndarray:
        """``(54,)`` scores; illegal vertices are ``-inf``."""
        ctx = SetupContext.build(board, prior_picks, acting_player, legal_settlements)
        return self.score_vertices_for(ctx)

    def score_vertices_for(self, ctx: SetupContext) -> np.ndarray:
        raw = self.settlement.score(all_settlement_features(ctx))
        out = np.full(N_VERTICES, -np.inf, dtype=np.float64)
        legal = ctx.legal_settlements
        out[legal] = raw[legal]
        return out

    def score_edges(
        self,
        board: Any,
        prior_picks: Sequence[Pick],
        acting_player: int,
        legal_settlements: np.ndarray,
        settlement: int,
        legal_edges: np.ndarray,
    ) -> np.ndarray:
        """``(72,)`` scores for the setup road after ``settlement``; illegal
        edges are ``-inf``."""
        ctx = SetupContext.build(board, prior_picks, acting_player, legal_settlements)
        return self.score_edges_for(ctx, settlement, legal_edges)

    def score_edges_for(
        self, ctx: SetupContext, settlement: int, legal_edges: np.ndarray
    ) -> np.ndarray:
        legal = np.asarray(legal_edges, dtype=bool)
        raw = self.road.score(all_road_features(ctx, settlement, legal))
        out = np.full(N_EDGES, -np.inf, dtype=np.float64)
        out[legal] = raw[legal]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_version": FEATURE_VERSION,
            "settlement": self.settlement.to_dict(),
            "road": self.road.to_dict(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SetupScorer:
        return cls(
            settlement=ScorerWeights.from_dict(payload["settlement"]),
            road=ScorerWeights.from_dict(payload["road"]),
            version=str(payload["version"]),
            provenance=dict(payload.get("provenance", {})),
        )


def _model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The parts of an artifact that ARE the model.

    Provenance is excluded on purpose: it carries ``fit_date`` and ``git_sha``,
    so comparing whole files would refuse a bit-identical re-run of the same fit
    — which is precisely the harmless case. What D6 is protecting is the
    weights, and the weights are here.
    """
    return {
        "feature_version": payload.get("feature_version"),
        "settlement": payload.get("settlement"),
        "road": payload.get("road"),
    }


def save_weights(scorer: SetupScorer, path: Path, *, overwrite: bool = False) -> None:
    """Write a fitted artifact, REFUSING a same-version content change (D6).

    D6 makes the artifact version the identity of a fit: "each refit bumps the
    artifact version; agreement is always reported against the scorer version
    live at label time". A label row stamps ``scorer_version``, so an artifact
    that changes its weights without changing its version retroactively rewrites
    what every already-labeled pick was graded by — and the D4 exam would go on
    reporting the stamp as if it still identified something.

    Re-saving byte-identical weights under the same version is allowed (a
    deterministic re-run is not a refit). ``overwrite=True`` is the escape hatch;
    callers that use it are expected to record the fact in the artifact's
    provenance, as ``scripts/fit_setup_scorer.py`` does.
    """
    path = Path(path)
    payload = scorer.to_dict()
    if path.exists() and not overwrite:
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and str(existing.get("version", "")) == scorer.version
            and _model_payload(existing) != _model_payload(payload)
        ):
            raise ScorerOverwriteError(
                f"{path} already holds DIFFERENT weights stamped version "
                f"{scorer.version!r}. D6 requires each refit to bump the artifact "
                f"version — pass --version <new> (and a new --out path), or "
                f"--overwrite to replace it deliberately."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_weights(path: Path) -> SetupScorer:
    """Load a fitted artifact. Raises :class:`ScorerVersionError` on a
    ``feature_version`` mismatch."""
    return SetupScorer.from_dict(json.loads(Path(path).read_text()))


def rank_of(scores: np.ndarray, choice: int, *, legal: np.ndarray | None = None) -> int:
    """1-based rank of ``choice`` among the LEGAL candidates (ties by index).

    Used for the D3 reveal's ``scorer_rank_of_pick``: rank 1 means the owner
    and the scorer agreed, and the rank distribution is what tells a near-miss
    apart from a disagreement the scorer never even considered.
    """
    s = np.asarray(scores, dtype=np.float64)
    if not 0 <= choice < s.shape[0]:
        raise ValueError(f"choice {choice} out of range for scores of shape {s.shape}")
    mask = legal_mask(s, legal)
    if not mask[choice]:
        raise ValueError(f"choice {choice} is not a legal candidate under these scores")
    better = np.sum(mask & (s > s[choice]))
    ties_before = np.sum(mask & (s == s[choice]) & (np.arange(s.shape[0]) < choice))
    return int(better + ties_before) + 1


@dataclass(frozen=True)
class PickGrade:
    """How one grader scored one of the owner's picks (D4 exam v2).

    The SAME record is produced for the scorer (from its masked-softmax vertex
    scores) and for the u500 baseline (from its ``log_dist/corner`` head), which
    is what makes the paired comparison a comparison of like with like: both
    graders emit a distribution over the identical legal-vertex mask, and every
    number below is a function of that distribution and the owner's pick alone.
    """

    log_prob: float
    """Log-probability the grader put on the owner's pick. D4's PRIMARY metric
    (a proper scoring rule — a genuine k-way tie is best answered ~1/k each)."""

    top1: int
    """The grader's argmax candidate."""

    agree: bool
    """``top1 == the owner's pick``. D4 v2 keeps this ONLY for the ``clear``
    strictness bar and for continuity with the pre-amendment reports; it is no
    longer the primary metric (measured ~35% labeler ceiling)."""

    in_top3: bool
    """Whether the owner's pick is in the grader's top 3 — the reported rate on
    ``close`` picks."""

    margin: float
    """Top-1 minus top-2 probability: the grader's confidence, read against the
    owner's clarity tag in the calibration report."""

    rank: int
    """1-based rank of the owner's pick under the grader's scores."""


def legal_mask(scores: np.ndarray, legal: np.ndarray | None = None) -> np.ndarray:
    """Resolve which candidates count as legal for every grading helper here.

    ``legal=None`` falls back to ``isfinite``, which is right for the scorer's
    own heads: :meth:`SetupScorer.score_vertices` writes ``-inf`` into every
    illegal slot itself.

    It is NOT right for a grader whose mask is applied with a large finite
    negative — ``-1e9`` is finite, so ``isfinite`` would call an illegal vertex
    legal and let it win an argmax, silently reporting an impossible top-1 as
    the baseline's answer. Any caller that HAS the mask should pass it: the
    boolean mask is the statement of legality, the score value is not.
    """
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    if legal is None:
        return finite
    m = np.asarray(legal, dtype=bool)
    if m.shape != s.shape:
        raise ValueError(f"legal mask must have shape {s.shape}, got {m.shape}")
    return m & finite


def grade_scores(scores: np.ndarray, choice: int, *, legal: np.ndarray | None = None) -> PickGrade:
    """Grade one pick against one grader's scores.

    ``legal`` is the explicit legality mask over ``scores``. Pass it whenever
    the caller has one (see :func:`legal_mask`).
    """
    mask = legal_mask(scores, legal)
    top3 = top_k(scores, 3, legal=mask)
    return PickGrade(
        log_prob=log_prob_of(scores, choice, legal=mask),
        top1=int(top3[0]) if top3 else -1,
        agree=bool(top3 and top3[0] == choice),
        in_top3=bool(choice in top3),
        margin=top1_margin(scores, legal=mask),
        rank=rank_of(scores, choice, legal=mask),
    )


def probabilities(scores: np.ndarray, *, legal: np.ndarray | None = None) -> np.ndarray:
    """Masked softmax over ``scores``; illegal candidates get 0.0.

    The scorer's heads ARE masked-softmax models — that is what
    :mod:`catan_rl.setup_phase.fit` maximises — so the probability vector is the
    model's own output, not a post-hoc calibration. D4 v2 grades distributions
    (paired mean log-probability of the owner's pick), and D3 reveals the
    scorer's CONFIDENCE rather than a bare pick, so both need this.
    """
    s = np.asarray(scores, dtype=np.float64)
    mask = legal_mask(s, legal)
    out = np.zeros_like(s)
    if not np.any(mask):
        return out
    z = s[mask] - s[mask].max()
    e = np.exp(z)
    out[mask] = e / e.sum()
    return out


def log_prob_of(scores: np.ndarray, choice: int, *, legal: np.ndarray | None = None) -> float:
    """Log of the masked-softmax probability the scorer puts on ``choice``.

    Computed in log space (log-sum-exp), not as ``log(probabilities(...))``, so
    a pick the model is very confident is WRONG returns a large finite negative
    number instead of underflowing to ``-inf`` and poisoning a mean.
    """
    s = np.asarray(scores, dtype=np.float64)
    if not 0 <= choice < s.shape[0]:
        raise ValueError(f"choice {choice} out of range for scores of shape {s.shape}")
    mask = legal_mask(s, legal)
    if not mask[choice]:
        raise ValueError(f"choice {choice} is not a legal candidate under these scores")
    m = s[mask].max()
    return float(s[choice] - m - np.log(np.exp(s[mask] - m).sum()))


def top1_margin(scores: np.ndarray, *, legal: np.ndarray | None = None) -> float:
    """Top-1 minus top-2 PROBABILITY: the scorer's confidence in its own pick.

    D4's calibration report reads this against the owner's ``pick_clarity``
    tags. It is a probability gap, not a score gap, so it is comparable across
    positions with different numbers of legal candidates. A position with a
    single legal candidate has margin 1.0.
    """
    mask = legal_mask(scores, legal)
    p = probabilities(scores, legal=mask)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    if idx.size == 1:
        return 1.0
    ordered = np.sort(p[idx])[::-1]
    return float(ordered[0] - ordered[1])


def top_k(scores: np.ndarray, k: int, *, legal: np.ndarray | None = None) -> list[int]:
    """The ``k`` highest-scoring legal candidates, best first."""
    s = np.asarray(scores, dtype=np.float64)
    idx = np.flatnonzero(legal_mask(s, legal))
    order = idx[np.argsort(-s[idx], kind="stable")]
    return [int(i) for i in order[:k]]


__all__ = [
    "FEATURE_VERSION",
    "ROAD_FEATURE_NAMES",
    "SETTLEMENT_FEATURE_NAMES",
    "PickGrade",
    "ScorerOverwriteError",
    "ScorerVersionError",
    "ScorerWeights",
    "SetupScorer",
    "grade_scores",
    "legal_mask",
    "load_weights",
    "log_prob_of",
    "probabilities",
    "rank_of",
    "save_weights",
    "top1_margin",
    "top_k",
]
