"""Pentanomial SPRT gate (spec 008 STAGE-A, FR-001 / FR-006).

The confirmation infrastructure every STAGE-A strength claim runs through. A
Sequential Probability Ratio Test decides ``PROMOTE`` / ``REJECT`` /
``INCONCLUSIVE`` between two hypotheses on the Elo difference of two configs,
consuming *materially fewer* games than a fixed-n eval at equal error rates
(SC-002).

Design (all copied from the vetted Stockfish-fishtest recipe, NOT hand-derived):

* **Pentanomial pairing.** The atomic sample is a seat-swapped block scored in
  ``{0, 0.5, 1, 1.5, 2}`` (a "pair"), NOT a single game. Pairing seat-swapped
  common-seed games removes the colour/opening variance that inflates a naive
  binomial test — the whole point of pentanomial.
* **GSPRT / score-based LLR.** With per-pair sample mean ``m`` and sample
  variance ``v`` (floored), the log-likelihood ratio of H1 (Elo=elo1) vs H0
  (Elo=elo0) is the Wald score form
  ``LLR = n*(t1-t0)/v * (m - (t0+t1)/2)`` where ``t{0,1}`` are the per-pair
  expected scores for elo{0,1}. This is the standard practical SPRT with an
  estimated variance (Michel Van den Bergh's GSPRT, as used by fishtest).
* **Bounds.** ``A = log(beta/(1-alpha))`` (REJECT), ``B = log((1-beta)/alpha)`` (PROMOTE); at
  the default alpha=beta=0.05 these are ∓2.944. ``INCONCLUSIVE`` = the max-pairs cap is
  hit before either bound.
* **Hypotheses (FR-001).** H0: Elo <= elo0 (default 0), H1: Elo >= elo1 (default
  +10). PROMOTE accepts H1 (config A is stronger); REJECT accepts H0.

The gate is pure/streaming (``PentanomialSPRT``) so it is fully unit-testable on
a synthetic pair stream, and a thin real-game driver
(``run_sprt_config_vs_config``) wraps ``eval_search`` for on-net comparisons.
Matched total sim budget (FR-006) is asserted by ``assert_matched_budget`` — a
mismatch RAISES.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import SnapshotOpponent

__all__ = [
    "PENTANOMIAL_SCORES",
    "PentanomialSPRT",
    "SPRTConfig",
    "SPRTResult",
    "assert_matched_budget",
    "config_total_sim_budget",
    "elo_to_score",
    "run_sprt_config_vs_config",
    "score_to_elo",
]

#: The five possible seat-swapped-block (pair) scores.
PENTANOMIAL_SCORES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
#: Regularization pseudo-count added to empty pentanomial buckets so the MLE-tilt
#: GSPRT never sees a degenerate (single-outcome) empirical distribution — the
#: fishtest ``regularize`` trick that keeps the log-likelihood ratio finite and
#: bounded on an all-wins / all-losses streak.
_REGULARIZE = 1e-3
#: Score clamp for the Elo<->score maps (a 0/1 score is ±inf Elo).
_SCORE_EPS = 1e-12


def elo_to_score(elo: float) -> float:
    """Logistic expected *per-game* score in (0,1) for an Elo advantage."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Inverse of :func:`elo_to_score` — per-game score in (0,1) -> Elo."""
    s = min(1.0 - _SCORE_EPS, max(_SCORE_EPS, score))
    return -400.0 * math.log10(1.0 / s - 1.0)


def _tilt_pdf(support: tuple[float, ...], pdf: list[float], theta: float) -> list[float]:
    """Exponential tilt ``q_i ∝ pdf_i * exp(θ*a_i)`` (normalised)."""
    w = [p * math.exp(theta * a) for p, a in zip(pdf, support, strict=True)]
    z = sum(w)
    return [wi / z for wi in w]


def _tilt_to_mean(support: tuple[float, ...], pdf: list[float], target: float) -> list[float]:
    """MLE distribution closest to ``pdf`` (min-KL) with mean ``target``.

    Fits the single exponential-tilt parameter θ by bisection (the tilted mean is
    monotone increasing in θ). ``target`` must lie strictly inside the support,
    which the caller guarantees by regularizing every bucket to a positive mass.
    """
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        m = sum(a * q for a, q in zip(support, _tilt_pdf(support, pdf, mid), strict=True))
        if m < target:
            lo = mid
        else:
            hi = mid
    return _tilt_pdf(support, pdf, 0.5 * (lo + hi))


@dataclass(frozen=True)
class SPRTConfig:
    """SPRT hypotheses + error rates (FR-001)."""

    #: H0 boundary Elo of A over B (default 0 — "no improvement").
    elo0: float = 0.0
    #: H1 boundary Elo (default +10; the spec's elo1 in the 5-10 band).
    elo1: float = 10.0
    #: Type-I / Type-II error rates (default 0.05 each -> bounds ∓2.944).
    alpha: float = 0.05
    beta: float = 0.05
    #: Hard cap on pentanomial pairs (seat-swapped blocks) -> INCONCLUSIVE.
    max_pairs: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be in (0,1), got {self.beta}")
        if self.elo1 <= self.elo0:
            raise ValueError(f"elo1 ({self.elo1}) must exceed elo0 ({self.elo0})")
        if self.max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {self.max_pairs}")

    @property
    def upper_bound(self) -> float:
        """PROMOTE bound B = log((1-beta)/alpha)."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower_bound(self) -> float:
        """REJECT bound A = log(beta/(1-alpha))."""
        return math.log(self.beta / (1.0 - self.alpha))


class PentanomialSPRT:
    """Streaming pentanomial GSPRT accumulator.

    Feed seat-swapped-block scores (in :data:`PENTANOMIAL_SCORES`) via
    :meth:`update`; read :meth:`llr` / :meth:`decision` after each. Pure — no I/O,
    no game play — so it is exhaustively testable on a synthetic stream.
    """

    def __init__(self, cfg: SPRTConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SPRTConfig()
        #: Bucket counts indexed by 2*score (0..4).
        self.counts: list[int] = [0, 0, 0, 0, 0]

    # -- accumulation --------------------------------------------------------

    def update(self, pair_score: float) -> None:
        """Record one pair's score; must be one of :data:`PENTANOMIAL_SCORES`."""
        idx = round(pair_score * 2.0)
        if idx < 0 or idx > 4 or abs(idx * 0.5 - pair_score) > 1e-9:
            raise ValueError(f"pair_score must be one of {PENTANOMIAL_SCORES}, got {pair_score}")
        self.counts[idx] += 1

    @property
    def n_pairs(self) -> int:
        return sum(self.counts)

    def mean_var(self) -> tuple[float, float]:
        """Sample mean and (population) variance of the per-pair score."""
        n = self.n_pairs
        if n == 0:
            return 0.0, 0.0
        total = 0.0
        total_sq = 0.0
        for idx, c in enumerate(self.counts):
            s = idx * 0.5
            total += s * c
            total_sq += s * s * c
        mean = total / n
        var = total_sq / n - mean * mean
        return mean, max(0.0, var)

    # -- test statistic ------------------------------------------------------

    def llr(self) -> float:
        """GSPRT log-likelihood ratio of H1 (elo1) vs H0 (elo0).

        Van den Bergh / fishtest generalised LLR: tilt the *regularized* empirical
        pentanomial distribution to the min-KL member of its exponential family
        whose per-pair mean matches each hypothesis (``t0`` for elo0, ``t1`` for
        elo1), then ``LLR = Σ_i count_i * log(q1_i / q0_i)``. Bounded by the data —
        it never explodes on a zero-variance streak (unlike the naive
        sample-variance score form). Returns 0.0 until >=2 pairs.
        """
        n = self.n_pairs
        if n < 2:
            return 0.0
        # Regularize empty buckets to a small positive mass -> a non-degenerate
        # empirical pdf over the full [0,2] support (so the tilt can hit t0/t1).
        raw = [float(c) + _REGULARIZE for c in self.counts]
        z = sum(raw)
        pdf = [r / z for r in raw]
        # Per-pair expected scores (a pair maxes at 2 -> 2 * per-game score).
        t0 = 2.0 * elo_to_score(self.cfg.elo0)
        t1 = 2.0 * elo_to_score(self.cfg.elo1)
        q0 = _tilt_to_mean(PENTANOMIAL_SCORES, pdf, t0)
        q1 = _tilt_to_mean(PENTANOMIAL_SCORES, pdf, t1)
        return sum(self.counts[i] * math.log(q1[i] / q0[i]) for i in range(len(self.counts)))

    def elo_estimate(self) -> float:
        """Point Elo of A over B from the observed per-pair mean score."""
        mean, _ = self.mean_var()
        return score_to_elo(mean / 2.0)

    def decision(self) -> str:
        """One of ``PROMOTE`` / ``REJECT`` / ``INCONCLUSIVE`` / ``CONTINUE``."""
        stat = self.llr()
        if stat >= self.cfg.upper_bound:
            return "PROMOTE"
        if stat <= self.cfg.lower_bound:
            return "REJECT"
        if self.n_pairs >= self.cfg.max_pairs:
            return "INCONCLUSIVE"
        return "CONTINUE"

    def result(self) -> SPRTResult:
        return SPRTResult(
            decision=self.decision(),
            llr=self.llr(),
            n_pairs=self.n_pairs,
            counts=list(self.counts),
            elo_estimate=self.elo_estimate(),
            upper_bound=self.cfg.upper_bound,
            lower_bound=self.cfg.lower_bound,
            elo0=self.cfg.elo0,
            elo1=self.cfg.elo1,
        )


@dataclass
class SPRTResult:
    """Terminal (or snapshot) state of an SPRT run."""

    decision: str
    llr: float
    n_pairs: int
    counts: list[int]
    elo_estimate: float
    upper_bound: float
    lower_bound: float
    elo0: float
    elo1: float
    #: 4 games per pentanomial pair (A/B x two seats) when produced by the
    #: config-vs-config driver; ``None`` for a raw synthetic-stream run.
    n_games: int | None = None
    #: Free-form provenance (config budgets, ckpts, seed) for the JSON record.
    meta: dict[str, object] = field(default_factory=dict)


def run_sprt_stream(cfg: SPRTConfig, pair_scores: Iterable[float]) -> SPRTResult:
    """Drive an SPRT over a pre-computed stream of pair scores (test helper)."""
    sprt = PentanomialSPRT(cfg)
    for score in pair_scores:
        sprt.update(score)
        if sprt.decision() != "CONTINUE":
            break
    return sprt.result()


# ---------------------------------------------------------------------------
# Matched-budget guard (FR-006)
# ---------------------------------------------------------------------------


def config_total_sim_budget(cfg: SearchConfig) -> int:
    """Total leaf-eval budget of a search config = Σ sims across determinizations.

    With the ``split_sims_across_determinizations`` split ON, each of the K trees
    runs ``max(1, sims//K)`` for a total ~``sims``; OFF, each runs the full
    ``sims`` for a total ``K*sims``. Only defined for the ``sims_per_move`` mode
    (time-budget mode is not bit-reproducible — FR-006/FR-010).
    """
    if cfg.sims_per_move is None:
        raise ValueError("matched-budget accounting requires sims_per_move mode")
    k = max(1, cfg.n_determinizations)
    if cfg.split_sims_across_determinizations and k > 1:
        return max(1, cfg.sims_per_move // k) * k
    return cfg.sims_per_move * k


def assert_matched_budget(cfg_a: SearchConfig, cfg_b: SearchConfig) -> int:
    """Assert two configs run at the SAME total sim budget; RAISE on mismatch.

    The load-bearing control (FR-006): a strength A/B is meaningless unless both
    sides spend equal leaf evaluations. Also requires equal ``n_determinizations``
    (the pentanomial pairing assumes symmetric per-move compute). Returns the
    shared budget on success.
    """
    if cfg_a.n_determinizations != cfg_b.n_determinizations:
        raise ValueError(
            "matched-budget violation: n_determinizations differ "
            f"({cfg_a.n_determinizations} != {cfg_b.n_determinizations})"
        )
    budget_a = config_total_sim_budget(cfg_a)
    budget_b = config_total_sim_budget(cfg_b)
    if budget_a != budget_b:
        raise ValueError(
            f"matched-budget violation: total sim budget differs ({budget_a} != {budget_b})"
        )
    return budget_a


# ---------------------------------------------------------------------------
# Real-game driver (config A vs config B, common-reference differential)
# ---------------------------------------------------------------------------


class _ChoosesActions(Protocol):
    """The decision surface the driver needs (SearchAgent satisfies it)."""

    def choose_action(self, env: CatanEnv) -> np.ndarray: ...


def _seat_swapped_block_score(
    env: CatanEnv,
    agent_a: _ChoosesActions,
    agent_b: _ChoosesActions,
    *,
    seed0: int,
    seed1: int,
    audit_rules: bool,
) -> float:
    """One pentanomial pair: A and B each play both seats vs the SAME reference.

    Differential common-reference design (no search-on-both-seats needed): A and
    B face the identical frozen reference on identical seeds (common random
    numbers), so ``(win_A - win_B)`` cancels shared opening/dice variance. Summed
    over the two seats the differential lands in ``{-2..+2}`` -> a pentanomial
    ``{0,0.5,1,1.5,2}`` pair scored from A's perspective.
    """
    from catan_rl.search.eval_search import _play_search_game

    a0 = _play_search_game(env, agent_a, seed=seed0, agent_seat=0, audit_rules=audit_rules).won
    a1 = _play_search_game(env, agent_a, seed=seed1, agent_seat=1, audit_rules=audit_rules).won
    b0 = _play_search_game(env, agent_b, seed=seed0, agent_seat=0, audit_rules=audit_rules).won
    b1 = _play_search_game(env, agent_b, seed=seed1, agent_seat=1, audit_rules=audit_rules).won
    diff = (int(a0) - int(b0)) + (int(a1) - int(b1))
    return (diff + 2) / 2.0


def run_sprt_agents_vs_reference(
    agent_a: _ChoosesActions,
    agent_b: _ChoosesActions,
    *,
    reference: SnapshotOpponent,
    sprt_cfg: SPRTConfig,
    seed: int = 0,
    max_turns: int = 400,
    audit_rules: bool = False,
    on_pair: Callable[[int, float, PentanomialSPRT], None] | None = None,
) -> SPRTResult:
    """Sequentially compare two decision agents A vs B against a common reference.

    Plays seat-swapped common-seed blocks (4 games/pair) until the SPRT decides
    or the pair cap is hit. RNG is snapshotted/restored so the run is
    reproducible and side-effect-free (mirrors ``eval_search``). Both agents wrap
    the SAME frozen net; only their decision rule differs.
    """
    import numpy as np
    import torch

    from catan_rl.env.catan_env import CatanEnv

    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    sprt = PentanomialSPRT(sprt_cfg)
    n_games = 0
    try:
        env = CatanEnv(opponent_type="snapshot", max_turns=max_turns)
        env.set_snapshot_opponent(reference)
        try:
            base = (seed * 1_000_003 + 2_246_822_519) % (2**31 - 1)
            for i in range(sprt_cfg.max_pairs):
                seed0 = (base + i * 2) % (2**31 - 1)
                seed1 = (base + i * 2 + 1) % (2**31 - 1)
                score = _seat_swapped_block_score(
                    env, agent_a, agent_b, seed0=seed0, seed1=seed1, audit_rules=audit_rules
                )
                sprt.update(score)
                n_games += 4
                if on_pair is not None:
                    on_pair(i + 1, score, sprt)
                if sprt.decision() != "CONTINUE":
                    break
        finally:
            env.close()
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)

    result = sprt.result()
    result.n_games = n_games
    return result


def run_sprt_config_vs_config(
    cfg_a: SearchConfig,
    cfg_b: SearchConfig,
    *,
    ckpt: str,
    reference_ckpt: str,
    sprt_cfg: SPRTConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    max_turns: int = 400,
    audit_rules: bool = False,
    on_pair: Callable[[int, float, PentanomialSPRT], None] | None = None,
) -> SPRTResult:
    """SPRT-compare two SEARCH configs on the frozen ``ckpt`` (matched budget).

    Both configs wrap the same net and play a common frozen reference
    (``reference_ckpt``). Asserts a matched total sim budget (FR-006) and RAISES
    on mismatch before any game is played.
    """
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    budget = assert_matched_budget(cfg_a, cfg_b)
    sprt_cfg = sprt_cfg if sprt_cfg is not None else SPRTConfig()

    from typing import cast

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=str(ckpt)), seed=seed, device=device),
    )
    agent_a = SearchAgent(actor.policy, cfg_a, device=actor.device)
    agent_b = SearchAgent(actor.policy, cfg_b, device=actor.device)

    ref_actor = cast(
        "_PolicyActor",
        build_actor(
            PlayerSpec(kind="policy", ckpt_path=str(reference_ckpt)), seed=seed, device=device
        ),
    )
    reference = FrozenSnapshotOpponent(ref_actor.policy, device=ref_actor.device, seed=seed)

    result = run_sprt_agents_vs_reference(
        agent_a,
        agent_b,
        reference=reference,
        sprt_cfg=sprt_cfg,
        seed=seed,
        max_turns=max_turns,
        audit_rules=audit_rules,
        on_pair=on_pair,
    )
    result.meta = {
        "ckpt": str(ckpt),
        "reference_ckpt": str(reference_ckpt),
        "total_sim_budget": budget,
        "seed": seed,
    }
    return result
