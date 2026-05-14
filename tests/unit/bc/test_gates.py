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
