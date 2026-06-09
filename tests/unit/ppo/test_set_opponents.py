"""T024 — mid-rollout opponent swap (US3).

``SerialVecEnv.set_opponents`` threads each env's opponent assignment into its
next reset: snapshot kinds inject a resolved frozen opponent (or fall back when
the resolver returns None, FR-011); non-snapshot kinds clear it. Together with
``League.build_env_opponent_assignments`` this is the "fresh snapshot enters
play" path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.policy.obs_schema import OPP_KIND_LEAGUE
from catan_rl.ppo.arguments import LeagueConfig
from catan_rl.ppo.vec_env import SerialVecEnv
from catan_rl.selfplay.league import OPPONENT_KIND_SNAPSHOT, League, OpponentAssignment


class _Stub:
    device = torch.device("cpu")

    def reset_rng(self, seed: int | None = None) -> None:
        pass

    def sample(self, obs: dict, masks: dict) -> torch.Tensor:
        return torch.zeros((1, 6), dtype=torch.int64)


def _vec(n: int) -> SerialVecEnv:
    return SerialVecEnv(env_kwargs_list=[{"opponent_type": "heuristic"} for _ in range(n)], seed=0)


def test_set_opponents_injects_snapshot_on_next_reset() -> None:
    vec = _vec(2)
    assigns = [
        OpponentAssignment(kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=5),
        OpponentAssignment(kind="random"),
    ]
    vec.set_opponents(assigns, snapshot_resolver=lambda _sid, _i: _Stub())
    vec.reset_all(seeds=[0, 1])
    assert vec.envs[0].has_snapshot_opponent
    assert int(vec.envs[0]._opp_kind) == OPP_KIND_LEAGUE  # agent sees a league opp
    assert not vec.envs[1].has_snapshot_opponent


def test_set_opponents_falls_back_when_resolver_returns_none() -> None:
    vec = _vec(2)
    assigns = [OpponentAssignment(kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=0)] * 2
    vec.set_opponents(assigns, snapshot_resolver=lambda _sid, _i: None)
    vec.reset_all(seeds=[0, 1])
    assert not vec.envs[0].has_snapshot_opponent  # None -> heuristic body fallback


def test_set_opponents_length_mismatch_raises() -> None:
    vec = _vec(2)
    with pytest.raises(ValueError, match="assignments length"):
        vec.set_opponents([OpponentAssignment(kind="random")])


def test_fresh_snapshot_enters_play_via_league() -> None:
    # Empty pool -> heuristic assignments; after add_snapshot the league can
    # hand out the new id and set_opponents seats it for the next rollout.
    empty = League(LeagueConfig(random_weight=0.0, heuristic_weight=1.0, snapshot_weight=1.0))
    a0 = empty.build_env_opponent_assignments(n_envs=4, rng=np.random.default_rng(0))
    assert all(a.kind != OPPONENT_KIND_SNAPSHOT for a in a0)

    lg = League(
        LeagueConfig(
            random_weight=0.0,
            heuristic_weight=0.0,
            snapshot_weight=1.0,
            require_heuristic_floor=False,
        )
    )
    sid = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=4)
    a1 = lg.build_env_opponent_assignments(n_envs=4, rng=np.random.default_rng(0))
    assert all(a.kind == OPPONENT_KIND_SNAPSHOT and a.snapshot_id == sid for a in a1)

    vec = _vec(4)
    vec.set_opponents(a1, snapshot_resolver=lambda _sid, _i: _Stub())
    vec.reset_all(seeds=[0, 1, 2, 3])
    assert all(env.has_snapshot_opponent for env in vec.envs)


def test_set_opponents_clears_snapshot_when_reassigned() -> None:
    vec = _vec(2)
    vec.set_opponents(
        [OpponentAssignment(kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=0)] * 2,
        snapshot_resolver=lambda _sid, _i: _Stub(),
    )
    vec.reset_all(seeds=[0, 1])
    assert all(e.has_snapshot_opponent for e in vec.envs)
    # Re-assign a non-snapshot kind -> the snapshot body is cleared on next reset.
    vec.set_opponents([OpponentAssignment(kind="heuristic")] * 2)
    vec.reset_all(seeds=[0, 1])
    assert not any(e.has_snapshot_opponent for e in vec.envs)


def test_resolver_gets_distinct_env_idx_per_env() -> None:
    # BLOCKER regression: two envs sharing one snapshot_id must get SEPARATE
    # RNG-bearing wrappers (resolver called once per env with its env_idx).
    seen: list[tuple[int, int]] = []

    def _resolver(sid: int, env_idx: int) -> _Stub:
        seen.append((sid, env_idx))
        return _Stub()  # a fresh instance each call

    vec = _vec(3)
    vec.set_opponents(
        [OpponentAssignment(kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=7)] * 3,
        snapshot_resolver=_resolver,
    )
    assert seen == [(7, 0), (7, 1), (7, 2)]
    vec.reset_all(seeds=[0, 1, 2])
    bodies = [e._snapshot_opponent for e in vec.envs]
    assert len({id(b) for b in bodies}) == 3  # three distinct wrapper instances
