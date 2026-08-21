"""Owner-vs-owner self-consistency over D0 replay labels.

Spec ``setup-scorer-and-blind-reveal`` D0. This module turns re-labeled
positions into the **labeler-noise ceiling**: the rate at which the owner
reproduces their own top-1 settlement and road, per draft position and overall,
with Wilson CIs.

Why it is reported PER POSITION and not only in aggregate: the premise review
established that u500 already matches the owner ~55% on pick 1 and 0-18% on
picks 2-4, so the owner's edge — and therefore every later comparison — lives
in positions 2-4. An aggregate self-agreement number would average the owner's
easiest decision together with their hardest and read as a ceiling that applies
to neither.

Nothing here is a gate. It is the number every later bar is READ AGAINST: if
self-agreement on picks 2-4 comes back near the bar the scorer is being asked
to clear, the bar structure is recalibrated before anything else is trusted.

THREE ESTIMATORS, NAMED
=======================
The corpus contains re-labels of two different kinds, and they are NOT the same
measurement. The report therefore always names the estimator that produced its
headline and publishes all three side by side under ``estimators`` so no reader
has to reconcile a number against a differently-derived one.

``linked`` (:data:`ESTIMATOR_LINKED`, :func:`pair_replay_rows`)
    Rows carrying ``replay_of``, written by ``--replay-session``. That mode is a
    FORCED-ORIGINAL replay: the draft is advanced with the ORIGINAL pick after
    every submit, so pick *k* of the replay is the same POSITION as pick *k* of
    the original regardless of what the owner just chose. This is the unbiased
    estimator and the one D0 asks for. The pre-2026-08-21 corpus has none of
    these — ``replay_of`` did not exist when the free replay was run.

``free_replay`` (:data:`ESTIMATOR_FREE_REPLAY`, :func:`free_replay_pairs`)
    Un-annotated re-labels recovered by joining on ``(game_seed,
    draft_position)`` alone. **This is the estimator that produced the banked
    D0 RESULT** — on the 292-row store it yields exactly 7/20 = 35% overall,
    pos1 4/5 degrading to pos4 0/5, road-given-same-settlement 3/7. It is biased
    DOWNWARD on picks 2-4: a free replay advances with its OWN picks, so once a
    pick disagrees, the later picks share a draft NUMBER with the original while
    standing on a different board position. Disagreement there mixes labeler
    noise with draft divergence.

``same_position`` (:data:`ESTIMATOR_SAME_POSITION`, :func:`legacy_pairs`)
    ``free_replay`` narrowed to pairs with identical ``prior_picks``. That
    filter removes the divergence contamination but replaces it with SELECTION
    ON THE OUTCOME: a pair survives only where every earlier pick in the draft
    already agreed, so the surviving rate is biased UPWARD and cannot reach
    draft position 4 at all. On the same store it reads 6/8 = 75% — a 40pp gap
    against the banked figure, which is what makes publishing either one
    unlabeled unacceptable.

``estimator="auto"`` (the default) prefers ``linked`` when any linked pair
exists and falls back to ``free_replay`` — the banked D0 estimator — otherwise.
It never silently returns the upward-biased ``same_position`` rate.

**Road agreement is CONDITIONAL.** A setup road must be incident to the
settlement just placed, so when the replay picks a different settlement the
original road edge is not even in the replay's legal set. Comparing the two edge
indices there is not a labeler-noise measurement, it is a comparison across two
different choice sets. The report therefore scores roads only on pairs whose
settlement agreed — the ``road_given_same_settlement`` quantity D0 itself
reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from catan_rl.eval.wilson import wilson_interval

DRAFT_POSITIONS: tuple[int, ...] = (1, 2, 3, 4)

ESTIMATOR_LINKED = "linked"
ESTIMATOR_FREE_REPLAY = "free_replay"
ESTIMATOR_SAME_POSITION = "same_position"
ESTIMATOR_AUTO = "auto"
ESTIMATORS: tuple[str, ...] = (
    ESTIMATOR_LINKED,
    ESTIMATOR_FREE_REPLAY,
    ESTIMATOR_SAME_POSITION,
)

#: One line per estimator saying which way it is biased. Copied verbatim into
#: every report so a number can never travel without its caveat.
ESTIMATOR_BIAS: dict[str, str] = {
    ESTIMATOR_LINKED: (
        "unbiased: ``replay_of`` rows come from the forced-original replay, which "
        "advances the draft with the ORIGINAL pick, so every replay pick stands on "
        "the same position as the pick it is compared against."
    ),
    ESTIMATOR_FREE_REPLAY: (
        "biased DOWNWARD on picks 2-4, and the estimator the banked D0 RESULT "
        "(7/20 = 35%) was computed with: pairs are joined on (game_seed, "
        "draft_position) alone, and a free replay advances with its OWN picks, so "
        "after any disagreement the later picks share a draft NUMBER with the "
        "original while standing on a different board position."
    ),
    ESTIMATOR_SAME_POSITION: (
        "biased UPWARD, and cannot reach draft position 4: free-replay pairs are "
        "narrowed to identical prior_picks, which selects on the outcome — a pair "
        "survives only where every earlier pick in that draft already agreed."
    ),
}


class ConsistencyError(ValueError):
    """Raised when replay rows cannot be paired against their originals."""


@dataclass(frozen=True)
class AgreementStat:
    """A binomial agreement rate with its Wilson interval."""

    n: int
    n_agree: int
    rate: float | None
    ci_lower: float | None
    ci_upper: float | None

    @classmethod
    def from_counts(cls, *, n_agree: int, n: int, alpha: float = 0.05) -> AgreementStat:
        if n <= 0:
            # An empty cell is reported as an explicit zero-n row rather than
            # omitted: a missing draft position in the report is indistinguishable
            # from a position that was labeled and never agreed. The rate is
            # ``None``, not NaN — ``json.dumps`` emits a bare ``NaN`` token that
            # is not valid JSON, and this report is written to disk.
            return cls(n=0, n_agree=0, rate=None, ci_lower=None, ci_upper=None)
        ci = wilson_interval(wins=n_agree, n=n, alpha=alpha)
        return cls(n=n, n_agree=n_agree, rate=ci.point, ci_lower=ci.lower, ci_upper=ci.upper)


@dataclass(frozen=True)
class ReplayPair:
    """One original row and the replay row that re-labeled the same position."""

    original: dict[str, Any]
    replay: dict[str, Any]

    @property
    def draft_position(self) -> int:
        return int(self.original["draft_position"])

    @property
    def settlement_agrees(self) -> bool:
        return int(self.original["settlement_vertex"]) == int(self.replay["settlement_vertex"])

    @property
    def road_agrees(self) -> bool:
        return int(self.original["road_edge"]) == int(self.replay["road_edge"])


def pair_replay_rows(rows: Iterable[dict[str, Any]]) -> list[ReplayPair]:
    """Pair every ``replay_of``-carrying row with the row it points at.

    Raises:
        ConsistencyError: if a replay row points at a ``scenario_id`` that is
            not in ``rows``, or if two rows share a ``scenario_id``. Both are
            corpus defects — a self-agreement number computed over a silently
            dropped subset is exactly the kind of quiet optimism this report
            exists to rule out.
    """
    rows = list(rows)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row["scenario_id"])
        if sid in by_id:
            raise ConsistencyError(f"duplicate scenario_id in corpus: {sid!r}")
        by_id[sid] = row

    pairs: list[ReplayPair] = []
    for row in rows:
        target = row.get("replay_of")
        if target is None:
            continue
        original = by_id.get(str(target))
        if original is None:
            raise ConsistencyError(
                f"replay row {row['scenario_id']!r} points at scenario_id "
                f"{str(target)!r}, which is not in the corpus"
            )
        pairs.append(ReplayPair(original=original, replay=row))
    return pairs


def _duplicate_groups(
    rows: Iterable[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Non-replay rows grouped by ``(game_seed, draft_position)``, size >= 2.

    Each group is sorted by ``labeled_at`` so ``[0]`` is the original.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("replay_of") is not None:
            continue
        groups.setdefault((int(row["game_seed"]), int(row["draft_position"])), []).append(row)
    return [
        sorted(members, key=lambda r: (str(r["labeled_at"]), str(r["scenario_id"])))
        for members in groups.values()
        if len(members) >= 2
    ]


def free_replay_pairs(rows: Iterable[dict[str, Any]]) -> list[ReplayPair]:
    """Recover un-annotated re-labels by ``(game_seed, draft_position)`` alone.

    **This is the estimator behind the banked D0 RESULT.** On the shipped label
    store it recovers 20 pairs, 7 of which agree — 7/20 = 35%, with pos1 4/5
    down to pos4 0/5 and road-given-same-settlement 3/7, reproducing every
    figure the spec banks under D0.

    No ``prior_picks`` filter is applied, deliberately: filtering would change
    the estimand (see :func:`legacy_pairs`). The cost is that a free replay
    advances with its OWN picks, so after a disagreement the later picks share a
    draft NUMBER with the original while standing on a different board position
    — the rate is biased DOWNWARD on picks 2-4. That direction is stated in
    :data:`ESTIMATOR_BIAS` and carried in every report.
    """
    pairs: list[ReplayPair] = []
    for members in _duplicate_groups(rows):
        original = members[0]
        for later in members[1:]:
            pairs.append(ReplayPair(original=original, replay=later))
    return pairs


def legacy_pairs(rows: Iterable[dict[str, Any]]) -> tuple[list[ReplayPair], int]:
    """Free-replay pairs narrowed to the SAME position: ``(pairs, n_dropped)``.

    Two rows form a pair iff they share ``(game_seed, draft_position)`` AND
    ``prior_picks`` — the latter is what makes them the same POSITION rather
    than merely the same draft number. The earliest ``labeled_at`` row is
    treated as the original; ``n_dropped`` counts the free-replay pairs the
    filter removed.

    The ``prior_picks`` filter selects on the outcome: a pair survives only if
    every earlier pick in that draft already agreed, so the rate over these
    pairs is an UPWARD-BIASED estimate of the labeler ceiling and cannot reach
    draft position 4. On the shipped store it reads 6/8 = 75% against the banked
    7/20 = 35%. It is never the report's headline.
    """
    pairs: list[ReplayPair] = []
    n_dropped = 0
    for members in _duplicate_groups(rows):
        original = members[0]
        for later in members[1:]:
            if later["prior_picks"] == original["prior_picks"]:
                pairs.append(ReplayPair(original=original, replay=later))
            else:
                n_dropped += 1
    return pairs, n_dropped


def _breakdown(pairs: Sequence[ReplayPair], *, agrees: str, alpha: float) -> dict[str, Any]:
    def rate(subset: Sequence[ReplayPair]) -> AgreementStat:
        return AgreementStat.from_counts(
            n_agree=sum(1 for pr in subset if getattr(pr, agrees)),
            n=len(subset),
            alpha=alpha,
        )

    return {
        "overall": asdict(rate(pairs)),
        "picks_2_4": asdict(rate([pr for pr in pairs if pr.draft_position >= 2])),
        "by_position": {
            str(pos): asdict(rate([pr for pr in pairs if pr.draft_position == pos]))
            for pos in DRAFT_POSITIONS
        },
    }


def consistency_report(
    rows: Iterable[dict[str, Any]],
    *,
    alpha: float = 0.05,
    estimator: str = ESTIMATOR_AUTO,
) -> dict[str, Any]:
    """Owner-vs-owner top-1 agreement, under a NAMED estimator.

    ``rows`` is the WHOLE label store (originals and replays together) — the
    pairing needs both sides present.

    ``estimator`` is one of :data:`ESTIMATORS`, or ``"auto"`` (the default):
    ``linked`` when the corpus has any ``replay_of`` row, else
    ``free_replay`` — the estimator the banked D0 RESULT was computed with.
    Whatever is chosen, the report states it under ``estimator``, states its
    bias direction under ``estimator_bias``, and publishes all three overall
    rates side by side under ``estimators`` so the headline can always be
    reconciled against the other two.
    """
    if estimator not in ESTIMATORS and estimator != ESTIMATOR_AUTO:
        raise ConsistencyError(
            f"estimator must be one of {(*ESTIMATORS, ESTIMATOR_AUTO)}, got {estimator!r}"
        )
    rows = list(rows)
    linked = pair_replay_rows(rows)
    free = free_replay_pairs(rows)
    same_position, n_filtered_out = legacy_pairs(rows)
    by_name: dict[str, list[ReplayPair]] = {
        ESTIMATOR_LINKED: linked,
        ESTIMATOR_FREE_REPLAY: free,
        ESTIMATOR_SAME_POSITION: same_position,
    }

    chosen = estimator
    if chosen == ESTIMATOR_AUTO:
        chosen = ESTIMATOR_LINKED if linked else ESTIMATOR_FREE_REPLAY
    pairs = by_name[chosen]

    same_settlement = [pr for pr in pairs if pr.settlement_agrees]
    return {
        "estimator": chosen,
        "estimator_requested": estimator,
        "estimator_bias": ESTIMATOR_BIAS[chosen],
        # All three overall rates in one place. The banked D0 RESULT is the
        # ``free_replay`` row; a rerun that quotes ``same_position`` (6/8 = 75%
        # on the shipped store) is NOT reproducing it, and neither is a rerun
        # that quotes ``linked`` before any forced-original replay has been run.
        "estimators": {
            name: {
                "n_pairs": len(group),
                "settlement_overall": asdict(
                    AgreementStat.from_counts(
                        n_agree=sum(1 for pr in group if pr.settlement_agrees),
                        n=len(group),
                        alpha=alpha,
                    )
                ),
                "bias": ESTIMATOR_BIAS[name],
            }
            for name, group in by_name.items()
        },
        "n_pairs": len(pairs),
        "n_linked_pairs": len(linked),
        "n_free_replay_pairs": len(free),
        "n_same_position_pairs": len(same_position),
        "n_free_replay_pairs_filtered_out": n_filtered_out,
        "n_replay_sessions": len({str(pr.replay["session_id"]) for pr in pairs}),
        "alpha": alpha,
        "settlement": _breakdown(pairs, agrees="settlement_agrees", alpha=alpha),
        # Roads are scored ONLY where the settlement agreed: a road incident to
        # a different settlement was never in the same choice set (see the
        # module docstring). This is D0's ``road-given-same-settlement``.
        "road_given_same_settlement": _breakdown(same_settlement, agrees="road_agrees", alpha=alpha)
        | {"conditioned_on": "settlement_agrees", "n_conditioning_pairs": len(same_settlement)},
    }
