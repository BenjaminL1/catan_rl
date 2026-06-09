"""T021 — policy-vs-policy eval (US2).

The champion is evaluated head-to-head against a loaded opponent checkpoint;
the opponent is seated via the in-env snapshot-opponent driver. Pins: a finite
WR + Wilson CI, seat-symmetrization, ``EvalMatchupResult`` extends ``EvalResult``
and carries the opponent ref, and bit-for-bit determinism on CPU (SC-004).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from catan_rl.checkpoint.manager import save_checkpoint
from catan_rl.eval.harness import EvalMatchupResult, EvalResult, evaluate_policy_vs_policy
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy


def _policy(seed: int) -> CatanPolicy:
    torch.manual_seed(seed)
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    return policy


def _save_ckpt(path: Path, policy: CatanPolicy) -> Path:
    return save_checkpoint(
        path,
        config={},
        policy=policy,
        optimizer=None,
        update_idx=0,
        global_step=0,
        capture_rng=False,
    )


def test_evaluate_policy_vs_policy_returns_wr_and_ci(tmp_path: Path) -> None:
    ckpt = _save_ckpt(tmp_path / "opp.pt", _policy(1))
    result = evaluate_policy_vs_policy(
        _policy(0), str(ckpt), n_games=4, seed=0, device="cpu", max_turns=50
    )
    assert isinstance(result, EvalMatchupResult)
    assert isinstance(result, EvalResult)  # extends, not forks
    assert result.opponent_ref == str(ckpt)
    assert result.opponent_type == "snapshot"
    assert result.n == 4
    assert 0.0 <= result.wr <= 1.0
    assert math.isfinite(result.ci.lower) and math.isfinite(result.ci.upper)
    # Seat-symmetrized: 2 from seat 0, 2 from seat 1.
    assert result.n_seat0 == 2
    assert result.n_seat1 == 2


def test_evaluate_policy_vs_policy_is_deterministic(tmp_path: Path) -> None:
    ckpt = _save_ckpt(tmp_path / "opp.pt", _policy(1))
    r1 = evaluate_policy_vs_policy(
        _policy(0), str(ckpt), n_games=4, seed=0, device="cpu", max_turns=50
    )
    # Perturb the global torch RNG between runs — the eval saves/restores it, so
    # this must NOT affect the second run (proves the containment, not just luck).
    _ = torch.rand(1024)
    r2 = evaluate_policy_vs_policy(
        _policy(0), str(ckpt), n_games=4, seed=0, device="cpu", max_turns=50
    )
    assert r1.wins == r2.wins
    assert [g.won for g in r1.games] == [g.won for g in r2.games]
    assert [g.final_vp_agent for g in r1.games] == [g.final_vp_agent for g in r2.games]
    assert [g.final_vp_opp for g in r1.games] == [g.final_vp_opp for g in r2.games]
    # n_turns is the sensitive fingerprint: it diverges under ANY sampling-stream
    # drift, unlike the saturated win/VP fields (untrained policies mostly truncate).
    assert [g.n_turns for g in r1.games] == [g.n_turns for g in r2.games]


def test_evaluate_policy_vs_policy_different_seed_differs(tmp_path: Path) -> None:
    """Negative control: a different seed yields a different trajectory — proving
    the determinism test can actually distinguish sampling streams."""
    ckpt = _save_ckpt(tmp_path / "opp.pt", _policy(1))
    a = evaluate_policy_vs_policy(
        _policy(0), str(ckpt), n_games=4, seed=0, device="cpu", max_turns=50
    )
    b = evaluate_policy_vs_policy(
        _policy(0), str(ckpt), n_games=4, seed=1, device="cpu", max_turns=50
    )
    assert [g.n_turns for g in a.games] != [g.n_turns for g in b.games]
