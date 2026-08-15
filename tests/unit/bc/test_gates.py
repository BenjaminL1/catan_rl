"""TDD tests for bc/gates.py.

Pin the statistical correctness of:
  * ``paired_bootstrap_nll`` — the per-head NLL paired-bootstrap test.
  * ``tost_wr_equivalence`` — TOST (two one-sided test) equivalence
    test against the heuristic's self-WR.

Both are the BC-plan §6 compound-gate components (post faculty
re-review). Tests use hand-constructed numerical scenarios so we can
check the boolean pass / fail decisions deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# paired_bootstrap_nll
# ---------------------------------------------------------------------------


def test_paired_bootstrap_passes_when_bc_strictly_better() -> None:
    """If BC has lower NLL than baseline on EVERY pair, the test passes
    (CI lower bound > 0) at any reasonable α."""
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    rng = np.random.default_rng(0)
    n = 1000
    # baseline NLL > BC NLL by 0.5 nats on every pair.
    base = rng.uniform(1.0, 2.0, size=n)
    bc = base - 0.5
    res = paired_bootstrap_nll_per_head(
        base_nll={"type": base},
        bc_nll={"type": bc},
        n_resamples=2000,
        alpha=0.01,
        seed=0,
    )
    assert res["type"]["ci_lower"] > 0.0
    assert res["type"]["passes"] is True


def test_paired_bootstrap_fails_when_bc_strictly_worse() -> None:
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    rng = np.random.default_rng(0)
    n = 500
    base = rng.uniform(1.0, 2.0, size=n)
    bc = base + 0.5  # BC is worse than baseline.
    res = paired_bootstrap_nll_per_head(
        base_nll={"type": base},
        bc_nll={"type": bc},
        n_resamples=2000,
        alpha=0.01,
        seed=0,
    )
    assert res["type"]["ci_lower"] < 0.0
    assert res["type"]["passes"] is False


def test_paired_bootstrap_fails_when_bc_indistinguishable() -> None:
    """When BC and baseline have the same NLL distribution (within noise),
    the bootstrap CI should straddle zero and the gate should NOT pass."""
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    rng = np.random.default_rng(0)
    n = 500
    base = rng.uniform(1.0, 2.0, size=n)
    bc = base + rng.normal(0, 0.01, size=n)  # essentially equal
    res = paired_bootstrap_nll_per_head(
        base_nll={"type": base},
        bc_nll={"type": bc},
        n_resamples=2000,
        alpha=0.01,
        seed=0,
    )
    # Mean improvement is ~0; either side of zero is acceptable; but
    # passes must be False (we require LOWER bound > 0).
    assert res["type"]["passes"] is False


def test_paired_bootstrap_handles_multiple_heads() -> None:
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    rng = np.random.default_rng(0)
    n = 400
    base_type = rng.uniform(1.0, 2.0, size=n)
    bc_type = base_type - 0.3
    base_corner = rng.uniform(3.0, 4.0, size=n)
    bc_corner = base_corner + 0.2  # BC worse on corner

    res = paired_bootstrap_nll_per_head(
        base_nll={"type": base_type, "corner": base_corner},
        bc_nll={"type": bc_type, "corner": bc_corner},
        n_resamples=2000,
        alpha=0.01,
        seed=0,
    )
    assert res["type"]["passes"] is True
    assert res["corner"]["passes"] is False


def test_paired_bootstrap_returns_required_fields() -> None:
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    rng = np.random.default_rng(0)
    n = 200
    base = rng.uniform(1.0, 2.0, size=n)
    bc = base - 0.3
    res = paired_bootstrap_nll_per_head(
        base_nll={"type": base},
        bc_nll={"type": bc},
        n_resamples=500,
        alpha=0.01,
        seed=0,
    )
    head_res = res["type"]
    for key in ("mean_delta", "ci_lower", "ci_upper", "passes", "n_pairs", "alpha"):
        assert key in head_res


def test_paired_bootstrap_rejects_mismatched_shapes() -> None:
    from catan_rl.bc.gates import paired_bootstrap_nll_per_head

    with pytest.raises(ValueError):
        paired_bootstrap_nll_per_head(
            base_nll={"type": np.zeros(100)},
            bc_nll={"type": np.zeros(99)},
            n_resamples=100,
            alpha=0.01,
            seed=0,
        )


def test_paired_bootstrap_compound_gate_requires_all_heads_pass() -> None:
    """compound_pass=True iff every named head passes."""
    from catan_rl.bc.gates import paired_bootstrap_nll_compound

    rng = np.random.default_rng(0)
    n = 400
    base_type = rng.uniform(1.0, 2.0, size=n)
    bc_type = base_type - 0.3
    base_corner = rng.uniform(3.0, 4.0, size=n)
    bc_corner = base_corner - 0.3
    base_edge = rng.uniform(3.0, 4.0, size=n)
    bc_edge = base_edge + 0.5  # edge fails

    res, compound = paired_bootstrap_nll_compound(
        base_nll={"type": base_type, "corner": base_corner, "edge": base_edge},
        bc_nll={"type": bc_type, "corner": bc_corner, "edge": bc_edge},
        required_heads=("type", "corner", "edge"),
        n_resamples=1000,
        alpha=0.01,
        seed=0,
    )
    assert compound is False
    assert res["type"]["passes"]
    assert res["corner"]["passes"]
    assert not res["edge"]["passes"]


# ---------------------------------------------------------------------------
# TOST WR equivalence
# ---------------------------------------------------------------------------


def test_tost_passes_when_wr_matches_teacher() -> None:
    """BC clone WR equal to teacher self-WR → passes."""
    from catan_rl.bc.gates import tost_wr_equivalence

    res = tost_wr_equivalence(wr_bc=0.42, wr_self=0.42, n=600, alpha=0.05, equivalence_margin=0.04)
    assert res["passes"] is True


def test_tost_fails_when_bc_too_strong() -> None:
    """BC clone significantly outperforms the teacher → fails (a real
    clone shouldn't exceed the teacher's symmetrised self-WR)."""
    from catan_rl.bc.gates import tost_wr_equivalence

    res = tost_wr_equivalence(wr_bc=0.62, wr_self=0.42, n=600, alpha=0.05, equivalence_margin=0.04)
    assert res["passes"] is False


def test_tost_fails_when_bc_too_weak() -> None:
    from catan_rl.bc.gates import tost_wr_equivalence

    res = tost_wr_equivalence(wr_bc=0.20, wr_self=0.42, n=600, alpha=0.05, equivalence_margin=0.04)
    assert res["passes"] is False


def test_tost_returns_diagnostic_fields() -> None:
    from catan_rl.bc.gates import tost_wr_equivalence

    res = tost_wr_equivalence(wr_bc=0.41, wr_self=0.42, n=600, alpha=0.05, equivalence_margin=0.04)
    for k in ("passes", "wr_bc", "wr_self", "delta", "ci_lower", "ci_upper", "margin", "n"):
        assert k in res


def test_tost_rejects_invalid_inputs() -> None:
    from catan_rl.bc.gates import tost_wr_equivalence

    with pytest.raises(ValueError):
        tost_wr_equivalence(wr_bc=1.5, wr_self=0.5, n=600)  # WR out of [0,1]
    with pytest.raises(ValueError):
        tost_wr_equivalence(wr_bc=0.5, wr_self=0.5, n=0)  # n must be > 0
    with pytest.raises(ValueError):
        tost_wr_equivalence(wr_bc=0.5, wr_self=0.5, n=600, equivalence_margin=-0.04)


# ---------------------------------------------------------------------------
# D7 — human-opening fine-tune gates
# ---------------------------------------------------------------------------


class _Game:
    def __init__(self, seed: int, agent_seat: int, won: bool) -> None:
        self.seed = seed
        self.agent_seat = agent_seat
        self.won = won


class _Result:
    def __init__(self, games) -> None:  # type: ignore[no-untyped-def]
        self.games = tuple(games)


def _paired(n: int, ft_wins, pre_wins):  # type: ignore[no-untyped-def]
    ft = _Result([_Game(i, i % 2, bool(ft_wins[i])) for i in range(n)])
    pre = _Result([_Game(i, i % 2, bool(pre_wins[i])) for i in range(n)])
    return ft, pre


class TestSetupAgreementGate:
    def test_bar_is_the_calibrated_max_not_a_flat_floor(self) -> None:
        from catan_rl.bc.gates import setup_agreement_gate

        # A weak baseline leaves the 0.30 floor binding.
        low = setup_agreement_gate(agreements=[True] * 13 + [False] * 27, baseline=0.10)
        assert low["bar"] == pytest.approx(0.30)
        # A strong baseline raises the bar to baseline + 0.10.
        high = setup_agreement_gate(agreements=[True] * 13 + [False] * 27, baseline=0.45)
        assert high["bar"] == pytest.approx(0.55)

    def test_passes_on_the_point_estimate_and_reports_the_ci(self) -> None:
        from catan_rl.bc.gates import setup_agreement_gate

        got = setup_agreement_gate(agreements=[True] * 20 + [False] * 20, baseline=0.20)
        assert got["n"] == 40
        assert got["point"] == pytest.approx(0.5)
        assert got["bar"] == pytest.approx(0.30)
        assert got["passes"] is True
        assert got["ci_lower"] < got["point"] < got["ci_upper"]
        # At 40 held-out scenarios the estimate is worth roughly +-14pp; the gate
        # reports that so a marginal result reads "more labels", not "lower bar".
        assert 0.10 < got["noise_half_width"] < 0.20

    def test_a_candidate_below_the_bar_fails(self) -> None:
        from catan_rl.bc.gates import setup_agreement_gate

        got = setup_agreement_gate(agreements=[True] * 8 + [False] * 32, baseline=0.20)
        assert got["passes"] is False
        assert got["margin_over_bar"] < 0

    def test_empty_held_out_set_is_refused(self) -> None:
        from catan_rl.bc.gates import setup_agreement_gate

        with pytest.raises(ValueError, match="no held-out scenarios"):
            setup_agreement_gate(agreements=[], baseline=0.2)


class TestPairedWrNonInferiority:
    def test_a_neutral_candidate_passes(self) -> None:
        """The whole point of D7.2: an identical candidate must not fail ~50% of
        the time the way a point-estimate ``WR_ft >= WR_pre`` gate does."""
        from catan_rl.bc.gates import paired_wr_non_inferiority

        wins = [i % 2 == 0 for i in range(200)]
        ft, pre = _paired(200, wins, wins)
        got = paired_wr_non_inferiority(finetuned=ft, pre=pre)
        assert got["delta"] == pytest.approx(0.0)
        assert got["passes"] is True
        assert got["n_pairs"] == 200

    def test_a_ten_point_regression_fails(self) -> None:
        from catan_rl.bc.gates import paired_wr_non_inferiority

        pre_wins = [True] * 120 + [False] * 80
        ft_wins = [True] * 100 + [False] * 100  # -10pp, all on shared seeds
        ft, pre = _paired(200, ft_wins, pre_wins)
        got = paired_wr_non_inferiority(finetuned=ft, pre=pre)
        assert got["delta"] == pytest.approx(-0.10)
        assert got["ci_lower"] < -0.05
        assert got["passes"] is False

    def test_arithmetic_matches_a_hand_computed_ci(self) -> None:
        from catan_rl.bc.gates import _inv_normal_cdf, paired_wr_non_inferiority

        pre_wins = [True] * 50 + [False] * 50
        ft_wins = [True] * 48 + [False] * 52
        ft, pre = _paired(100, ft_wins, pre_wins)
        got = paired_wr_non_inferiority(finetuned=ft, pre=pre)
        deltas = np.array(
            [float(ft_wins[i]) - float(pre_wins[i]) for i in range(100)], dtype=np.float64
        )
        se = deltas.std(ddof=1) / np.sqrt(100)
        z = _inv_normal_cdf(0.975)
        assert got["delta"] == pytest.approx(deltas.mean())
        assert got["ci_lower"] == pytest.approx(deltas.mean() - z * se)
        assert got["ci_upper"] == pytest.approx(deltas.mean() + z * se)

    def test_mismatched_seed_plans_raise_rather_than_unpair(self) -> None:
        from catan_rl.bc.gates import UnpairableEvalError, paired_wr_non_inferiority

        ft = _Result([_Game(i, 0, True) for i in range(10)])
        pre = _Result([_Game(i + 5, 0, True) for i in range(10)])
        with pytest.raises(UnpairableEvalError, match="not seed-paired"):
            paired_wr_non_inferiority(finetuned=ft, pre=pre)

    def test_duplicate_seed_seat_keys_raise(self) -> None:
        from catan_rl.bc.gates import UnpairableEvalError, paired_wr_non_inferiority

        dup = _Result([_Game(1, 0, True), _Game(1, 0, False)])
        with pytest.raises(UnpairableEvalError, match="duplicate"):
            paired_wr_non_inferiority(finetuned=dup, pre=dup)

    def test_the_same_seed_played_from_both_seats_pairs_separately(self) -> None:
        """``(seed, agent_seat)`` — not seed alone. The harness plays every seed
        from both seats, and collapsing them would halve the pair count."""
        from catan_rl.bc.gates import paired_wr_non_inferiority

        ft = _Result([_Game(1, 0, True), _Game(1, 1, False)])
        pre = _Result([_Game(1, 0, True), _Game(1, 1, True)])
        got = paired_wr_non_inferiority(finetuned=ft, pre=pre)
        assert got["n_pairs"] == 2


class TestPairingAgainstRealEvalRuns:
    """The synthetic tests above pin the arithmetic; this one pins the PREMISE.

    ``paired_wr_non_inferiority`` raises unless both runs cover the same
    ``(seed, agent_seat)`` keys, and the whole D7.2 gate rests on two real
    ``EvalHarness`` rounds at one seed producing exactly that. Nothing in the
    synthetic doubles would notice if the harness stopped deriving its seeds
    from ``self.seed`` + the opponent label.
    """

    @staticmethod
    def _tiny_report():  # type: ignore[no-untyped-def]
        import torch

        from catan_rl.eval.harness import EvalHarness
        from catan_rl.policy import CatanPolicy
        from catan_rl.policy.board_geometry import build_geometry

        torch.manual_seed(0)
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        policy.eval()
        harness = EvalHarness(
            opponent_types=("random",),
            n_games_per_seat=1,
            seed=17,
            max_turns=8,
            audit_rules=False,
        )
        return harness.run(policy).by_opponent("random")

    def test_two_real_harness_runs_at_one_seed_pair_cleanly(self) -> None:
        from catan_rl.bc.gates import paired_wr_non_inferiority

        first = self._tiny_report()
        second = self._tiny_report()
        assert first is not None and second is not None
        got = paired_wr_non_inferiority(finetuned=first, pre=second, margin=0.05)
        assert got["n_pairs"] == 2  # one game per seat
        assert got["delta"] == pytest.approx(0.0)
