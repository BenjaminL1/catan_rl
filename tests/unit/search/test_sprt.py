"""Pentanomial SPRT gate (spec 008 STAGE-A, FR-001 / FR-006).

Covers the required guarantees, all on a SYNTHETIC pair stream (no game play):
  (a) a clearly-stronger (+30 Elo) stream -> PROMOTE in materially fewer games
      than a fixed n>=600 eval at equal error rates (SC-002);
  (b) a clearly-weaker (-50 Elo) stream -> REJECT;
  (c) an equal-strength (0 Elo) stream -> NOT PROMOTE (REJECT or, under a small
      cap, INCONCLUSIVE) — the gate does not false-positive;
  (d) the LLR is finite/bounded on a zero-variance (all-wins) streak (the
      regularized MLE-tilt GSPRT, not the naive sample-variance form);
  (e) matched-total-sim-budget is asserted and a mismatch RAISES (FR-006).
"""

from __future__ import annotations

import math
import random

import pytest

from catan_rl.search.config import SearchConfig
from catan_rl.search.sprt import (
    PAIRING_COMMON_REF_DIFF,
    PentanomialSPRT,
    SPRTConfig,
    assert_matched_budget,
    config_total_sim_budget,
    elo_from_pair_mean,
    elo_to_score,
    pair_mean_for_elo,
    run_sprt_stream,
    score_to_elo,
)


def _stream_decision(elo: float, seed: int, cfg: SPRTConfig) -> tuple[str, int]:
    """Run the SPRT on an iid pentanomial stream at a true Elo; (decision, pairs)."""
    p = elo_to_score(elo)
    rng = random.Random(seed)
    sprt = PentanomialSPRT(cfg)
    pairs = 0
    while sprt.decision() == "CONTINUE":
        w0 = 1 if rng.random() < p else 0
        w1 = 1 if rng.random() < p else 0
        sprt.update(float(w0 + w1))
        pairs += 1
    return sprt.decision(), pairs


# --- (a) strong stream -> PROMOTE, fewer games than fixed n ----------------


def _fixed_n_games(p0: float, p1: float, alpha: float, beta: float) -> float:
    """Equal-error fixed-sample-size (per game) to separate p0 from p1.

    One-sided normal approximation ``N = ((z_α·σ0 + z_β·σ1)/(p1−p0))²``. This is
    the honest baseline SC-002 means by "a fixed n eval at equal error rates" —
    for the tight [0,+10] Elo band it is ~13k games, so the sequential test's
    ~10³ games is *materially* fewer.
    """
    # 0.95-quantile of the standard normal (alpha=beta=0.05).
    z = 1.6448536269514722
    s0 = math.sqrt(p0 * (1.0 - p0))
    s1 = math.sqrt(p1 * (1.0 - p1))
    return ((z * s0 + z * s1) / (p1 - p0)) ** 2


def test_strong_stream_promotes_faster_than_fixed_n() -> None:
    # A +30-Elo stream must PROMOTE, and the AVERAGE games-to-decision must be far
    # below the equal-error fixed-sample-size test for the SAME [elo0, elo1]
    # hypotheses — the whole value of sequential testing (SC-002). Averaged over
    # seeds so it is not a lucky-streak artifact.
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=20000)
    p0, p1 = elo_to_score(cfg.elo0), elo_to_score(cfg.elo1)
    fixed_n_games = _fixed_n_games(p0, p1, cfg.alpha, cfg.beta)
    assert fixed_n_games >= 600, "sanity: the equal-error fixed-n baseline is >= the spec's floor"

    decisions = []
    games = []
    for seed in range(40):
        d, pr = _stream_decision(30.0, 5000 + seed, cfg)
        decisions.append(d)
        games.append(2 * pr)  # 2 games per pentanomial pair
    promote_rate = decisions.count("PROMOTE") / len(decisions)
    assert promote_rate >= 0.9, f"+30 Elo should almost always PROMOTE, got {promote_rate}"
    avg_games = sum(games) / len(games)
    assert avg_games < fixed_n_games, (
        f"sequential test used {avg_games:.0f} games on average, not fewer than the "
        f"equal-error fixed-n {fixed_n_games:.0f}"
    )


# --- (b) weak stream -> REJECT ---------------------------------------------


def test_weak_stream_rejects() -> None:
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=4000)
    rejects = sum(_stream_decision(-50.0, 6000 + s, cfg)[0] == "REJECT" for s in range(40))
    assert rejects / 40 >= 0.9, "a -50 Elo stream should almost always REJECT"


# --- (c) equal stream -> not a false PROMOTE -------------------------------


def test_equal_stream_does_not_false_promote() -> None:
    # At the H0 boundary (0 Elo) the gate's Type-I (false PROMOTE) rate must be
    # small — well under half — over independent seeds. We do NOT require exactly
    # INCONCLUSIVE (with H0 elo<=0 the boundary case mostly REJECTs), only that it
    # rarely PROMOTEs.
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=4000)
    promotes = sum(_stream_decision(0.0, 7000 + s, cfg)[0] == "PROMOTE" for s in range(60))
    assert promotes / 60 <= 0.2, f"Type-I (false PROMOTE) too high: {promotes}/60"


def test_equal_stream_inconclusive_under_small_cap() -> None:
    # With a small pair cap an equal-strength stream that has not crossed either
    # bound returns INCONCLUSIVE (the cap outcome), never PROMOTE.
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=20)
    d, pr = _stream_decision(0.0, 42, cfg)
    assert d in ("INCONCLUSIVE", "REJECT")
    assert pr <= 20


# --- (d) numerical robustness on a degenerate streak -----------------------


def test_llr_finite_on_all_wins_streak() -> None:
    # A run of identical (all-wins) pairs has ZERO sample variance; the naive
    # score-SPRT would divide by ~0 and explode. The regularized MLE-tilt GSPRT
    # must stay finite and grow only gradually.
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=4000)
    sprt = PentanomialSPRT(cfg)
    for _ in range(50):
        sprt.update(2.0)
    assert math.isfinite(sprt.llr())
    # 50 identical low-information pairs must not have produced runaway evidence.
    assert abs(sprt.llr()) < 50.0


def test_update_rejects_out_of_grid_scores() -> None:
    sprt = PentanomialSPRT()
    with pytest.raises(ValueError, match="pair_score"):
        sprt.update(0.3)
    for s in (0.0, 0.5, 1.0, 1.5, 2.0):
        sprt.update(s)  # all valid


def test_bounds_are_symmetric_at_equal_error_rates() -> None:
    cfg = SPRTConfig(alpha=0.05, beta=0.05)
    assert math.isclose(cfg.upper_bound, -cfg.lower_bound)
    assert math.isclose(cfg.upper_bound, math.log(0.95 / 0.05))


def test_elo_score_roundtrip() -> None:
    for elo in (-200.0, -10.0, 0.0, 10.0, 200.0):
        assert math.isclose(score_to_elo(elo_to_score(elo)), elo, abs_tol=1e-6)


def test_run_sprt_stream_helper() -> None:
    cfg = SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=2000)
    res = run_sprt_stream(cfg, [2.0] * 4000)  # a strong stream from a list
    assert res.decision == "PROMOTE"
    assert res.n_pairs <= 2000


# --- (e) matched-budget guard (FR-006) -------------------------------------


def test_matched_budget_equal_passes() -> None:
    a = SearchConfig(sims_per_move=100, seed=0)
    b = SearchConfig(sims_per_move=100, seed=1, final_move_mode="lcb")
    assert assert_matched_budget(a, b) == 100


def test_matched_budget_sims_mismatch_raises() -> None:
    a = SearchConfig(sims_per_move=100)
    b = SearchConfig(sims_per_move=200)
    with pytest.raises(ValueError, match="total sim budget differs"):
        assert_matched_budget(a, b)


def test_matched_budget_ndet_mismatch_raises() -> None:
    a = SearchConfig(sims_per_move=100, n_determinizations=1)
    b = SearchConfig(sims_per_move=100, n_determinizations=2)
    with pytest.raises(ValueError, match="n_determinizations differ"):
        assert_matched_budget(a, b)


# --- (f) differential common-reference pairing calibration (BLOCKER) --------


def _differential_pair_stream(elo_ab: float, seed: int, n: int) -> list[float]:
    """A and B each play a COMMON reference; A-over-B Elo split symmetrically.

    Per game the reference-relative win probs are ``elo_to_score(±elo_ab/2)``; a
    pentanomial pair is two seat-swapped games scored ``(Σ(win_A - win_B) + 2)/2``
    — exactly what ``_seat_swapped_block_score`` produces in the real driver.
    """
    p_a = elo_to_score(elo_ab / 2.0)
    p_b = elo_to_score(-elo_ab / 2.0)
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        diff = 0
        for _seat in range(2):
            a = 1 if rng.random() < p_a else 0
            b = 1 if rng.random() < p_b else 0
            diff += a - b
        out.append((diff + 2) / 2.0)
    return out


def test_differential_pairing_recovers_true_elo() -> None:
    # A differential common-reference stream at a KNOWN +80-Elo A-over-B gap: the
    # differential pairing must recover ~+80, while the head-to-head map (the bug)
    # under-reads it by ~2x (~+40). The two estimates differ by EXACTLY a factor
    # of two because elo_from_pair_mean(diff) == 2 * elo_from_pair_mean(hh).
    elo_true = 80.0
    stream = _differential_pair_stream(elo_true, seed=2024, n=20000)

    diff_sprt = PentanomialSPRT(SPRTConfig(pairing=PAIRING_COMMON_REF_DIFF, max_pairs=10**9))
    hh_sprt = PentanomialSPRT(SPRTConfig(max_pairs=10**9))  # default head_to_head
    for s in stream:
        diff_sprt.update(s)
        hh_sprt.update(s)

    diff_est = diff_sprt.elo_estimate()
    hh_est = hh_sprt.elo_estimate()
    # Calibrated: the differential estimate lands near the true gap.
    assert abs(diff_est - elo_true) < 20.0, f"differential mis-reads Elo: {diff_est:.1f}"
    # The head-to-head map (the BLOCKER) is ~2x too small -> biased toward NO-GO.
    assert abs(hh_est - elo_true / 2.0) < 12.0, f"head-to-head not ~half: {hh_est:.1f}"
    # And the miscalibration is precisely the factor of two.
    assert math.isclose(diff_est, 2.0 * hh_est, rel_tol=1e-9)


def test_pair_mean_and_elo_roundtrip_per_pairing() -> None:
    for pairing in ("head_to_head", PAIRING_COMMON_REF_DIFF):
        for elo in (-40.0, 0.0, 25.0):
            m = pair_mean_for_elo(elo, pairing)
            assert 0.0 < m < 2.0
            assert math.isclose(elo_from_pair_mean(m, pairing), elo, abs_tol=1e-6)
    # elo=0 is a dead-even pair mean of 1.0 under both pairings.
    assert math.isclose(pair_mean_for_elo(0.0, "head_to_head"), 1.0)
    assert math.isclose(pair_mean_for_elo(0.0, PAIRING_COMMON_REF_DIFF), 1.0)
    # Differential has HALF the head-to-head Elo sensitivity around parity.
    d_hh = pair_mean_for_elo(4.0, "head_to_head") - 1.0
    d_diff = pair_mean_for_elo(4.0, PAIRING_COMMON_REF_DIFF) - 1.0
    assert math.isclose(d_diff, 0.5 * d_hh, rel_tol=1e-3)


def test_invalid_pairing_raises() -> None:
    with pytest.raises(ValueError, match="pairing must be one of"):
        SPRTConfig(pairing="gauntlet")


def test_config_total_budget_split_vs_unsplit() -> None:
    # Split ON: K trees each run sims//K -> total ~sims. Split OFF: total K*sims.
    split = SearchConfig(
        sims_per_move=120, n_determinizations=4, split_sims_across_determinizations=True
    )
    unsplit = SearchConfig(sims_per_move=120, n_determinizations=4)
    assert config_total_sim_budget(split) == 120  # (120//4)*4
    assert config_total_sim_budget(unsplit) == 480  # 120*4
    # And a split K=4 at sims=120 is matched-budget with a single-tree sims=120.
    single = SearchConfig(sims_per_move=120, n_determinizations=1)
    with pytest.raises(ValueError, match="n_determinizations differ"):
        assert_matched_budget(split, single)
