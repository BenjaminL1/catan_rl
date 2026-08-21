"""Masked-softmax fit of the setup scorer against the owner's labels (spec D2).

Two heads, fit independently:

* **settlement** — a softmax over the LEGAL vertices at each decision point,
  trained to put mass on the vertex the owner picked;
* **road** — a softmax over the legal setup edges given the owner's settlement.

The road head is FIT, not asserted. The "point at the expansion target" rule
the owner's theory suggests ships as :data:`ROAD_NULL_FEATURE` — a one-feature
null model the fit is REPORTED against. If the fitted head cannot beat it, that
is a finding about the theory, not a number to bury.

**Replay rows are excluded, by identity.** A D0 replay row carries
``replay_of``; it re-labels a position that is already in the corpus, so
fitting on both would double-count that position and quietly weight it by how
often the owner happened to replay it. The exclusion is a filter on
``replay_of is not None`` AND a hard refusal on any remaining duplicate
``(game_seed, draft_position)`` — the guard against a free-replay session
written before ``replay_of`` existed.

That refusal FIRES on the corpus as it stands today: the owner ran a free
replay before this code existed, so there are un-annotated duplicate positions.
The escape hatch is a named, recorded choice — ``duplicate_policy="first-labeled"``,
surfaced by ``scripts/fit_setup_scorer.py --on-duplicate first-labeled`` and
stamped into the artifact's provenance — never a silent de-duplication.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from catan_rl.labeling.scenario_gen import Pick, Scenario, ScenarioGenerator
from catan_rl.setup_phase.scorer import (
    PickGrade,
    ScorerWeights,
    SetupScorer,
    grade_scores,
)
from catan_rl.setup_phase.scorer_features import (
    N_ROAD_FEATURES,
    ROAD_FEATURE_NAMES,
    SETTLEMENT_FEATURE_NAMES,
    SetupContext,
    all_road_features,
    all_settlement_features,
)

ROAD_NULL_FEATURE: str = "opens_best_vertex_value"
"""The single road feature that encodes "point at the expansion target". The
fitted road head is reported against a model that uses only this column."""


class FitError(ValueError):
    """Raised when the label corpus cannot be turned into a fit."""


@dataclass(frozen=True)
class Example:
    """One decision point as two masked-softmax problems."""

    scenario_id: str
    game_seed: int
    draft_position: int
    settlement_x: np.ndarray
    settlement_candidates: np.ndarray
    settlement_target: int
    road_x: np.ndarray
    road_candidates: np.ndarray
    road_target: int
    road_null_x: np.ndarray


@dataclass
class FitResult:
    """The fitted scorer plus everything the artifact reports."""

    scorer: SetupScorer
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------
DUPLICATE_POLICIES: tuple[str, ...] = ("refuse", "first-labeled")


def training_rows(
    rows: Iterable[dict[str, Any]], *, duplicate_policy: str = "refuse"
) -> list[dict[str, Any]]:
    """Non-replay label rows, with duplicated decision points handled explicitly.

    ``duplicate_policy="refuse"`` (default) raises on any repeated
    ``(game_seed, draft_position)``. ``"first-labeled"`` keeps the earliest
    ``labeled_at`` row for each and drops the rest — the ONLY sanctioned way to
    fit over the pre-``replay_of`` free replay, and the caller is expected to
    record that it did so.
    """
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise FitError(
            f"duplicate_policy must be one of {DUPLICATE_POLICIES}, got {duplicate_policy!r}"
        )
    kept = [r for r in rows if r.get("replay_of") is None]
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    dropped: list[str] = []
    for row in kept:
        key = (int(row["game_seed"]), int(row["draft_position"]))
        prior = seen.get(key)
        if prior is None:
            seen[key] = row
            continue
        if duplicate_policy == "refuse":
            raise FitError(
                f"corpus holds two NON-replay rows for game_seed={key[0]} "
                f"draft_position={key[1]} ({prior['scenario_id']!r} and "
                f"{row['scenario_id']!r}). A re-labeled position must carry "
                f"``replay_of`` so the fit can exclude it and the consistency report "
                f"can pair it; fitting on both would double-count the position. Pass "
                f"duplicate_policy='first-labeled' to keep the earliest of each pair."
            )
        earlier, later = sorted(
            (prior, row), key=lambda r: (str(r["labeled_at"]), str(r["scenario_id"]))
        )
        seen[key] = earlier
        dropped.append(str(later["scenario_id"]))
    ordered = [r for r in kept if r["scenario_id"] not in set(dropped)]
    return ordered


def replay_scenario(row: dict[str, Any]) -> tuple[ScenarioGenerator, Scenario]:
    """Rebuild the exact position a label row describes."""
    gen = ScenarioGenerator(seed=int(row["game_seed"]))
    for pick in row["prior_picks"]:
        p = Pick.from_dict(pick)
        gen.apply(p.settlement_vertex, p.road_edge)
    scenario = gen.current()
    if scenario is None:
        raise FitError(f"row {row['scenario_id']!r}: prior_picks exhausted the draft")
    if scenario.draft_position != int(row["draft_position"]):
        raise FitError(
            f"row {row['scenario_id']!r}: replayed to draft position "
            f"{scenario.draft_position} but the row claims {row['draft_position']}"
        )
    return gen, scenario


def context_for_row(row: dict[str, Any]) -> tuple[SetupContext, Scenario]:
    """``SetupContext`` + live scenario for one label row."""
    gen, scenario = replay_scenario(row)
    ctx = SetupContext.build(
        gen._board,
        scenario.prior_picks,
        int(scenario.acting_player_idx),
        scenario.legal_settlement_corners,
    )
    return ctx, scenario


def _subset_columns(feature_names: Sequence[str], subset: Sequence[str] | None) -> np.ndarray:
    if subset is None:
        return np.arange(len(feature_names))
    unknown = [n for n in subset if n not in feature_names]
    if unknown:
        raise FitError(f"unknown feature names in subset: {unknown}")
    return np.asarray([list(feature_names).index(n) for n in subset], dtype=np.int64)


def build_examples(
    rows: Iterable[dict[str, Any]],
    *,
    settlement_feature_subset: Sequence[str] | None = None,
) -> list[Example]:
    """Turn label rows into masked-softmax examples.

    ``settlement_feature_subset`` selects a COLUMN SUBSET of the current design
    matrix (used by acceptance criterion 3 to refit the pilot's 10 features on
    the pilot's split). It is a subset, never a second feature function.
    """
    cols = _subset_columns(SETTLEMENT_FEATURE_NAMES, settlement_feature_subset)
    null_col = ROAD_FEATURE_NAMES.index(ROAD_NULL_FEATURE)
    out: list[Example] = []
    for row in rows:
        ctx, scenario = context_for_row(row)
        vertex = int(row["settlement_vertex"])
        edge = int(row["road_edge"])
        if not bool(ctx.legal_settlements[vertex]):
            raise FitError(
                f"row {row['scenario_id']!r}: labeled settlement {vertex} is illegal "
                f"in the reconstructed position"
            )
        v_candidates = np.flatnonzero(ctx.legal_settlements)
        settle_x = all_settlement_features(ctx)[v_candidates][:, cols]

        legal_edges = np.asarray(scenario.compute_legal_road_edges(vertex), dtype=bool)
        if not bool(legal_edges[edge]):
            raise FitError(
                f"row {row['scenario_id']!r}: labeled road {edge} is illegal after "
                f"the labeled settlement at {vertex}"
            )
        e_candidates = np.flatnonzero(legal_edges)
        road_all = all_road_features(ctx, vertex, legal_edges)
        road_x = road_all[e_candidates]

        out.append(
            Example(
                scenario_id=str(row["scenario_id"]),
                game_seed=int(row["game_seed"]),
                draft_position=int(row["draft_position"]),
                settlement_x=settle_x,
                settlement_candidates=v_candidates,
                settlement_target=int(np.flatnonzero(v_candidates == vertex)[0]),
                road_x=road_x,
                road_candidates=e_candidates,
                road_target=int(np.flatnonzero(e_candidates == edge)[0]),
                road_null_x=road_x[:, [null_col]],
            )
        )
    return out


# ---------------------------------------------------------------------------
# The fit itself
# ---------------------------------------------------------------------------
def _standardise(blocks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-column mean / scale over every candidate row in the corpus.

    Standardising is not cosmetic here: the raw columns span pip counts (0-15)
    and 0/1 indicators, and a single L2 penalty over unscaled columns silently
    penalises the indicator weights ~100x less than the pip weights, which makes
    the "relational features got weight w" report unreadable.
    """
    pooled = np.concatenate(list(blocks), axis=0)
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    return mean, scale


def _fit_head(
    blocks: Sequence[np.ndarray],
    targets: Sequence[int],
    *,
    l2: float,
    iters: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-batch Adam on the masked-softmax NLL. Deterministic given inputs."""
    if not blocks:
        raise FitError("cannot fit a head with zero examples")
    mean, scale = _standardise(blocks)
    z = [(b - mean) / scale for b in blocks]
    n_features = z[0].shape[1]
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1e-3, size=n_features)

    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = float(len(z))
    for step in range(1, iters + 1):
        grad = 2.0 * l2 * w
        for x, target in zip(z, targets, strict=True):
            logits = x @ w
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            grad += (x.T @ p - x[target]) / n
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad * grad
        w = w - lr * (m / (1.0 - b1**step)) / (np.sqrt(v / (1.0 - b2**step)) + eps)
    return w, mean, scale


def _nll_and_agreement(
    weights: ScorerWeights, blocks: Sequence[np.ndarray], targets: Sequence[int]
) -> tuple[float, list[bool]]:
    nll = 0.0
    agree: list[bool] = []
    for x, target in zip(blocks, targets, strict=True):
        s = weights.score(x)
        m = s.max()
        nll += float(-(s[target] - m - np.log(np.exp(s - m).sum())))
        agree.append(int(np.argmax(s)) == int(target))
    return nll / max(1, len(blocks)), agree


def _by_position(examples: Sequence[Example], agree: Sequence[bool]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pos in (1, 2, 3, 4):
        idx = [i for i, ex in enumerate(examples) if ex.draft_position == pos]
        out[str(pos)] = {
            "n": len(idx),
            "agreement": (float(np.mean([agree[i] for i in idx])) if idx else None),
        }
    return out


def fit_scorer(
    rows: Iterable[dict[str, Any]],
    *,
    version: str,
    seed: int = 0,
    l2: float = 1e-3,
    iters: int = 1500,
    lr: float = 0.05,
    settlement_feature_subset: Sequence[str] | None = None,
    duplicate_policy: str = "refuse",
    provenance: dict[str, Any] | None = None,
) -> FitResult:
    """Fit both heads on the non-replay labels in ``rows``.

    Deterministic given ``(rows, seed)`` — the only stochastic element is the
    1e-3 weight init, which is drawn from ``default_rng(seed)``.
    """
    kept = training_rows(rows, duplicate_policy=duplicate_policy)
    if not kept:
        raise FitError("no non-replay label rows to fit on")
    examples = build_examples(kept, settlement_feature_subset=settlement_feature_subset)

    settle_names = tuple(
        SETTLEMENT_FEATURE_NAMES if settlement_feature_subset is None else settlement_feature_subset
    )
    settle_blocks = [ex.settlement_x for ex in examples]
    settle_targets = [ex.settlement_target for ex in examples]
    w, mean, scale = _fit_head(settle_blocks, settle_targets, l2=l2, iters=iters, lr=lr, seed=seed)
    settlement = ScorerWeights(feature_names=settle_names, weights=w, mean=mean, scale=scale)

    road_blocks = [ex.road_x for ex in examples]
    road_targets = [ex.road_target for ex in examples]
    rw, rmean, rscale = _fit_head(
        road_blocks, road_targets, l2=l2, iters=iters, lr=lr, seed=seed + 1
    )
    road = ScorerWeights(feature_names=ROAD_FEATURE_NAMES, weights=rw, mean=rmean, scale=rscale)

    null_blocks = [ex.road_null_x for ex in examples]
    nw, nmean, nscale = _fit_head(
        null_blocks, road_targets, l2=l2, iters=iters, lr=lr, seed=seed + 2
    )
    road_null = ScorerWeights(
        feature_names=(ROAD_NULL_FEATURE,), weights=nw, mean=nmean, scale=nscale
    )

    settle_nll, settle_agree = _nll_and_agreement(settlement, settle_blocks, settle_targets)
    road_nll, road_agree = _nll_and_agreement(road, road_blocks, road_targets)
    null_nll, null_agree = _nll_and_agreement(road_null, null_blocks, road_targets)

    metrics: dict[str, Any] = {
        "n_labels": len(examples),
        "duplicate_policy": duplicate_policy,
        "n_settlement_features": len(settle_names),
        "n_road_features": N_ROAD_FEATURES,
        "settlement": {
            "nll": settle_nll,
            "agreement": float(np.mean(settle_agree)),
            "by_position": _by_position(examples, settle_agree),
            "weights": dict(zip(settle_names, [float(x) for x in w], strict=True)),
        },
        "road": {
            "nll": road_nll,
            "agreement": float(np.mean(road_agree)),
            "by_position": _by_position(examples, road_agree),
            "weights": dict(zip(ROAD_FEATURE_NAMES, [float(x) for x in rw], strict=True)),
        },
        "road_null_baseline": {
            "feature": ROAD_NULL_FEATURE,
            "nll": null_nll,
            "agreement": float(np.mean(null_agree)),
            "beaten_by_fit": float(np.mean(road_agree)) > float(np.mean(null_agree)),
        },
    }

    scorer = SetupScorer(
        settlement=settlement,
        road=road,
        version=version,
        provenance=dict(provenance or {}) | {"fit_metrics": metrics},
    )
    return FitResult(scorer=scorer, metrics=metrics)


def settlement_grades(
    scorer: SetupScorer,
    rows: Iterable[dict[str, Any]],
    *,
    settlement_feature_subset: Sequence[str] | None = None,
) -> list[PickGrade]:
    """Per-row :class:`~catan_rl.setup_phase.scorer.PickGrade` for a fitted scorer.

    Grading happens in CANDIDATE space (the legal-vertex block the fit itself
    uses), so the distribution is the masked softmax over exactly the vertices
    that were legal at that decision point — the same mask the u500 baseline is
    read under. Only ``top1`` is mapped back to a board vertex index, because it
    is the only field whose value means a vertex rather than a probability.
    """
    examples = build_examples(rows, settlement_feature_subset=settlement_feature_subset)
    out: list[PickGrade] = []
    for ex in examples:
        scores = scorer.settlement.score(ex.settlement_x)
        grade = grade_scores(scores, ex.settlement_target)
        out.append(replace(grade, top1=int(ex.settlement_candidates[grade.top1])))
    return out


def top1_agreement(
    scorer: SetupScorer,
    rows: Iterable[dict[str, Any]],
    *,
    settlement_feature_subset: Sequence[str] | None = None,
) -> list[bool]:
    """Per-row top-1 settlement agreement for an already-fitted scorer.

    Kept as the thin view over :func:`settlement_grades` that the pilot-
    continuity reports read; D4 v2's exam grades distributions, not this.
    """
    return [
        g.agree
        for g in settlement_grades(
            scorer, rows, settlement_feature_subset=settlement_feature_subset
        )
    ]
