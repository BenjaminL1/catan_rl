"""Oracle-root SPRT probe: RNG-contract + reference-reset guards (spec 008 FR-004).

Covers the two SHOULD-FIX resolutions on the kill-probe apparatus:
  (a) ``OracleRootAgent.choose_action`` upholds the global-RNG snapshot/restore
      contract ``SearchAgent`` does — its candidate rollouts must not perturb the
      surrounding game's torch/numpy/stdlib streams (FR-006);
  (b) the env honours an explicit per-game ``opponent_seed`` so the SPRT
      differential driver can reset the reference opponent to a BYTE-IDENTICAL
      stream in the A-side and B-side game of a pair (its variance cancellation).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv

from .conftest import drive_to_decision, make_policy

# ``probe_oracle_sprt`` lives under scripts/dev (not an installed package) — load
# it by path so the test can exercise the real OracleRootAgent.
_PROBE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "dev" / "probe_oracle_sprt.py"
_spec = importlib.util.spec_from_file_location("probe_oracle_sprt", _PROBE_PATH)
assert _spec is not None and _spec.loader is not None
probe_oracle_sprt = importlib.util.module_from_spec(_spec)
sys.modules["probe_oracle_sprt"] = probe_oracle_sprt
_spec.loader.exec_module(probe_oracle_sprt)


def _rng_fingerprint() -> tuple[Any, Any, Any]:
    return (
        torch.random.get_rng_state().clone(),
        np.random.get_state(),
        random.getstate(),
    )


def test_oracle_choose_action_preserves_global_rng() -> None:
    # The oracle scores candidates by full MC rollouts (reseeding torch/numpy/
    # stdlib + advancing the opponent stream). It MUST leave the surrounding
    # game's RNG exactly as it found it — otherwise it desyncs the common random
    # numbers the SPRT differential relies on. Small max_turns keeps rollouts fast.
    policy = make_policy()
    device = torch.device("cpu")
    env = CatanEnv(opponent_type="heuristic", max_turns=60)
    env.reset(seed=7)
    assert drive_to_decision(env)

    oracle = probe_oracle_sprt.OracleRootAgent(policy, device=device, sims=4, rollouts=1, seed=0)

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    before = _rng_fingerprint()

    action = oracle.choose_action(env)
    assert isinstance(action, np.ndarray) and action.shape == (6,)

    t_after, np_after, py_after = _rng_fingerprint()
    assert torch.equal(before[0], t_after), "oracle perturbed the torch RNG stream"
    # numpy get_state() -> (name, keys, pos, has_gauss, cached); compare the keys+pos.
    assert before[1][0] == np_after[0]
    assert np.array_equal(before[1][1], np_after[1])
    assert before[1][2] == np_after[2]
    assert before[2] == py_after, "oracle perturbed the stdlib RNG stream"


def test_oracle_scores_candidates_on_common_random_numbers(monkeypatch: Any) -> None:
    # The resolution fix (BLOCKER): the oracle must score every candidate on the
    # SAME per-rollout seeds (common random numbers) so the shared dice/opponent
    # randomness cancels in the pairwise q(a)-q(b) the argmax depends on. We stub
    # _rollout_win to record its seeds; candidates are processed sequentially, each
    # doing `rollouts` calls, so the seed stream must tile: chunk it into groups of
    # `rollouts` and every group must be identical across candidates.
    policy = make_policy()
    device = torch.device("cpu")
    env = CatanEnv(opponent_type="heuristic", max_turns=60)
    env.reset(seed=7)
    assert drive_to_decision(env)

    rollouts = 3
    oracle = probe_oracle_sprt.OracleRootAgent(
        policy, device=device, sims=4, rollouts=rollouts, seed=0
    )

    seen_seeds: list[int] = []

    def _fake_rollout_win(*_args: Any, seed: int, **_kwargs: Any) -> float:
        seen_seeds.append(int(seed))
        return 0.0

    monkeypatch.setattr(probe_oracle_sprt, "_rollout_win", _fake_rollout_win)

    oracle.choose_action(env)

    n_candidates = len(oracle.last_diagnostics.get("visit_counts", {}))
    assert n_candidates >= 2, "need >1 candidate to test CRN pairing"
    assert len(seen_seeds) == n_candidates * rollouts, (n_candidates, rollouts, len(seen_seeds))
    chunks = [seen_seeds[i * rollouts : (i + 1) * rollouts] for i in range(n_candidates)]
    for c in chunks[1:]:
        assert c == chunks[0], f"candidate rollout seeds not paired (CRN broken): {chunks}"
    # And the shared seeds within a chunk are distinct (rollouts are not identical).
    assert len(set(chunks[0])) == rollouts, "per-rollout seeds must differ within a candidate"


class _RecordingOpponent:
    """Minimal SnapshotOpponent stub that records every ``reset_rng`` seed."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset_rng(self, seed: int | None = None) -> None:
        self.seeds.append(seed)

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> torch.Tensor:  # pragma: no cover - not exercised in reset-only test
        raise AssertionError("sample must not be called during reset")


def test_env_honours_explicit_opponent_seed() -> None:
    # With an explicit opponent_seed, the reference is reset to that SAME seed
    # every time (identical stream for A's and B's game of a pair). Two resets at
    # the same opponent_seed but DIFFERENT game seeds must still pin the same
    # reference seed -> the variance-cancellation guarantee.
    opp = _RecordingOpponent()
    env = CatanEnv(opponent_type="snapshot", max_turns=40)
    env.set_snapshot_opponent(opp)

    env.reset(seed=11, options={"agent_seat": 0, "opponent_seed": 999})
    env.reset(seed=22, options={"agent_seat": 0, "opponent_seed": 999})
    assert opp.seeds == [999, 999], opp.seeds

    env.reset(seed=11, options={"agent_seat": 0, "opponent_seed": 1000})
    assert opp.seeds[-1] == 1000


def test_env_default_opponent_seed_is_env_drawn_and_deterministic() -> None:
    # Without opponent_seed the env draws a per-game seed from its own RNG (the
    # unchanged default). It is a deterministic function of the GAME seed, so two
    # resets at the same game seed pin the same reference seed; different game
    # seeds generally differ (no explicit pin passed -> byte-identical to before).
    opp = _RecordingOpponent()
    env = CatanEnv(opponent_type="snapshot", max_turns=40)
    env.set_snapshot_opponent(opp)

    env.reset(seed=101, options={"agent_seat": 0})
    env.reset(seed=101, options={"agent_seat": 0})
    assert opp.seeds[0] == opp.seeds[1] is not None
    env.reset(seed=202, options={"agent_seat": 0})
    assert opp.seeds[2] != opp.seeds[0]
