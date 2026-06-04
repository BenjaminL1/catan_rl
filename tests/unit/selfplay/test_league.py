"""Tests for `selfplay/league.py` — minimal opponent league.

Pins:

1. Sampling weights — uniform random across many samples → ratios
   converge to the configured weights.
2. Snapshot retention — deque is bounded by maxlen, oldest entries
   evicted FIFO.
3. Snapshot cloning — modifying the source state dict after
   ``add_snapshot`` does NOT mutate the stored snapshot.
4. Snapshot path is currently gated — sampling raises
   NotImplementedError when snapshot_weight>0 AND the pool is non-empty.
5. add_snapshot_every_n_updates schedule.
6. Validators reject negative / all-zero weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.ppo.arguments import LeagueConfig
from catan_rl.selfplay.league import (
    OPPONENT_KIND_HEURISTIC,
    OPPONENT_KIND_RANDOM,
    League,
    LeagueSnapshot,
)

# ---------------------------------------------------------------------------
# Config validators
# ---------------------------------------------------------------------------


class TestLeagueConfigValidators:
    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="random_weight"):
            LeagueConfig(random_weight=-0.1)

    def test_all_zero_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one league weight"):
            LeagueConfig(
                random_weight=0.0,
                heuristic_weight=0.0,
                snapshot_weight=0.0,
            )

    def test_maxlen_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="maxlen"):
            LeagueConfig(maxlen=0)

    def test_add_every_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="add_snapshot_every"):
            LeagueConfig(add_snapshot_every_n_updates=0)


# ---------------------------------------------------------------------------
# Snapshot retention
# ---------------------------------------------------------------------------


class TestSnapshotRetention:
    def test_empty_on_construction(self) -> None:
        lg = League(LeagueConfig())
        assert lg.n_snapshots() == 0

    def test_append_and_count(self) -> None:
        lg = League(LeagueConfig())
        lg.add_snapshot({"layer1.weight": torch.zeros(3)}, update_idx=10)
        lg.add_snapshot({"layer1.weight": torch.ones(3)}, update_idx=20)
        assert lg.n_snapshots() == 2

    def test_maxlen_bounds_pool(self) -> None:
        lg = League(LeagueConfig(maxlen=3))
        for i in range(5):
            lg.add_snapshot(
                {"layer1.weight": torch.full((1,), float(i))},
                update_idx=i,
            )
        assert lg.n_snapshots() == 3
        # Newest snapshots survive (FIFO eviction of the oldest).
        assert lg.peek_snapshot(0).update_idx == 2
        assert lg.peek_snapshot(-1).update_idx == 4

    def test_peek_out_of_range_raises(self) -> None:
        lg = League(LeagueConfig())
        with pytest.raises(IndexError, match="no snapshots"):
            lg.peek_snapshot(0)
        lg.add_snapshot({"x": torch.zeros(1)}, update_idx=4)
        with pytest.raises(IndexError):
            lg.peek_snapshot(99)


class TestSnapshotCloning:
    def test_source_mutation_does_not_affect_stored(self) -> None:
        # Reviewer-aware foot-gun pre-emption: if add_snapshot stored
        # aliased tensors, the next PPO update on the live policy
        # would silently mutate every "past" snapshot in the league.
        src = {"w": torch.zeros(3)}
        lg = League(LeagueConfig())
        lg.add_snapshot(src, update_idx=0)
        # Mutate the source AFTER the add.
        src["w"].fill_(99.0)
        stored = lg.peek_snapshot(0)
        assert torch.equal(stored.state_dict["w"], torch.zeros(3)), (
            "add_snapshot should clone tensors; the live source "
            "must not be observable in the stored snapshot"
        )

    def test_clones_to_cpu(self) -> None:
        # Reviewer-caught HIGH: snapshots used to clone on source
        # device → ~560MB GPU resident at maxlen=100. Pool now lives
        # on CPU regardless of where the live policy is.
        src = {"w": torch.zeros(3)}
        lg = League(LeagueConfig())
        lg.add_snapshot(src, update_idx=10)
        stored = lg.peek_snapshot(0)
        assert stored.state_dict["w"].device.type == "cpu"

    def test_non_tensor_values_pass_through(self) -> None:
        # State dicts can include non-tensor entries (e.g., int counters)
        # — the cloner should pass those through unchanged.
        lg = League(LeagueConfig())
        lg.add_snapshot({"step": 42}, update_idx=10)
        assert lg.peek_snapshot(0).state_dict["step"] == 42


class TestSnapshotIdStability:
    def test_id_increments_monotonically(self) -> None:
        lg = League(LeagueConfig())
        ids = [lg.add_snapshot({"w": torch.zeros(1)}, update_idx=u) for u in (4, 8, 12, 16)]
        assert ids == [0, 1, 2, 3]

    def test_peek_by_id_survives_eviction(self) -> None:
        # Reviewer-caught HIGH: positional indices reshuffle on FIFO
        # eviction, so the vec-env opponent path must look up by
        # stable id instead. Verify the lookup works pre- and post-
        # eviction.
        lg = League(LeagueConfig(maxlen=3))
        ids = []
        for u in (4, 8, 12, 16, 20):
            ids.append(lg.add_snapshot({"w": torch.full((1,), float(u))}, update_idx=u))
        # First two ids are evicted (FIFO).
        assert lg.peek_by_id(ids[0]) is None
        assert lg.peek_by_id(ids[1]) is None
        # Last three survive and are retrievable BY STABLE ID even
        # though their positional index in the deque has shifted.
        for i in (2, 3, 4):
            snap = lg.peek_by_id(ids[i])
            assert snap is not None
            assert snap.snapshot_id == ids[i]

    def test_peek_by_id_returns_none_for_unknown_id(self) -> None:
        lg = League(LeagueConfig())
        assert lg.peek_by_id(999) is None


# ---------------------------------------------------------------------------
# add_snapshot_every_n_updates schedule
# ---------------------------------------------------------------------------


class TestSnapshotSchedule:
    def test_every_n_default_4(self) -> None:
        lg = League(LeagueConfig())  # default add_snapshot_every=4
        # Update 4 / 8 / 12 → snapshot. Update 0 is SKIPPED
        # (reviewer-caught HIGH: random-init policy is not worth
        # snapshotting).
        for u in (4, 8, 12):
            assert lg.should_snapshot_this_update(u)
        for u in (0, 1, 2, 3, 5, 6, 7, 9):
            assert not lg.should_snapshot_this_update(u)

    def test_skip_update_zero(self) -> None:
        # Explicit regression: update 0 must NOT snapshot the random-init
        # policy. See LeagueSnapshot docstring + the v1 audit's
        # warm-up notes.
        lg = League(LeagueConfig(add_snapshot_every_n_updates=1))
        assert not lg.should_snapshot_this_update(0)
        assert lg.should_snapshot_this_update(1)

    def test_custom_period(self) -> None:
        lg = League(LeagueConfig(add_snapshot_every_n_updates=10))
        assert not lg.should_snapshot_this_update(0)  # update 0 skip
        assert lg.should_snapshot_this_update(10)
        assert not lg.should_snapshot_this_update(5)


# ---------------------------------------------------------------------------
# Opponent-mix sampling
# ---------------------------------------------------------------------------


class TestOpponentMix:
    def test_n_envs_must_be_positive(self) -> None:
        lg = League(LeagueConfig())
        with pytest.raises(ValueError, match="n_envs"):
            lg.build_env_opponent_mix(n_envs=0, rng=np.random.default_rng(0))

    def test_pure_heuristic_default(self) -> None:
        # Default cfg → all heuristic.
        lg = League(LeagueConfig())
        mix = lg.build_env_opponent_mix(n_envs=8, rng=np.random.default_rng(0))
        assert all(k == OPPONENT_KIND_HEURISTIC for k in mix)

    def test_50_50_mix_converges_to_weights(self) -> None:
        lg = League(LeagueConfig(random_weight=1.0, heuristic_weight=1.0))
        rng = np.random.default_rng(0)
        mix = lg.build_env_opponent_mix(n_envs=2000, rng=rng)
        n_random = sum(1 for k in mix if k == OPPONENT_KIND_RANDOM)
        # Expect ~50%; SE under fair binomial at N=2000 is ~0.011,
        # so ±0.03 is a generous window.
        assert 0.47 < n_random / 2000 < 0.53

    def test_weighted_mix(self) -> None:
        # 75/25 split heuristic/random.
        lg = League(LeagueConfig(random_weight=1.0, heuristic_weight=3.0))
        rng = np.random.default_rng(0)
        mix = lg.build_env_opponent_mix(n_envs=4000, rng=rng)
        n_heur = sum(1 for k in mix if k == OPPONENT_KIND_HEURISTIC)
        assert 0.72 < n_heur / 4000 < 0.78

    def test_snapshot_weight_with_empty_pool_falls_back(self) -> None:
        # snapshot_weight > 0 but pool empty → renormalise over
        # remaining (heuristic+random) kinds rather than raising.
        lg = League(
            LeagueConfig(
                random_weight=0.0,
                heuristic_weight=1.0,
                snapshot_weight=1.0,
            )
        )
        rng = np.random.default_rng(0)
        mix = lg.build_env_opponent_mix(n_envs=8, rng=rng)
        assert all(k == OPPONENT_KIND_HEURISTIC for k in mix)

    def test_snapshot_weight_with_populated_pool_raises(self) -> None:
        # Phase 6 gates the snapshot opponent path with a loud error.
        # Wait for Phase 8 (checkpoint loading) before flipping it on.
        lg = League(
            LeagueConfig(
                random_weight=0.0,
                heuristic_weight=0.0,
                snapshot_weight=1.0,
            )
        )
        lg.add_snapshot({"w": torch.zeros(1)}, update_idx=4)
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError, match="snapshot opponent path"):
            lg.build_env_opponent_mix(n_envs=4, rng=rng)


# ---------------------------------------------------------------------------
# Repr (smoke)
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_does_not_crash(self) -> None:
        lg = League(LeagueConfig())
        s = repr(lg)
        assert "League(" in s
        assert "n_snapshots=0" in s


# ---------------------------------------------------------------------------
# LeagueSnapshot dataclass
# ---------------------------------------------------------------------------


class TestSnapshotEnvSentinel:
    """Pin the contract between :data:`OPPONENT_KIND_SNAPSHOT` and
    :class:`CatanEnv` (reviewer-caught CRITICAL).

    Until Phase 8 wires the actual snapshot opponent inference path,
    ``CatanEnv(opponent_type="snapshot")`` must raise
    ``NotImplementedError`` — NOT the generic ``ValueError`` that the
    previous "supported: 'random' or 'heuristic'" whitelist produced.
    """

    def test_catan_env_recognises_snapshot_sentinel(self) -> None:
        from catan_rl.env.catan_env import CatanEnv
        from catan_rl.selfplay.league import OPPONENT_KIND_SNAPSHOT

        with pytest.raises(NotImplementedError, match="snapshot"):
            CatanEnv(opponent_type=OPPONENT_KIND_SNAPSHOT)

    def test_catan_env_rejects_unknown_opponent_type(self) -> None:
        # The whitelist still rejects arbitrary strings (regression
        # guard: don't accidentally widen too far).
        from catan_rl.env.catan_env import CatanEnv

        with pytest.raises(ValueError, match="opponent_type"):
            CatanEnv(opponent_type="mcts")


class TestLeagueSnapshot:
    def test_metadata_default_empty(self) -> None:
        snap = LeagueSnapshot(state_dict={}, update_idx=0)
        assert snap.metadata == {}

    def test_metadata_is_independent(self) -> None:
        # Default-factory dict per instance — not shared.
        a = LeagueSnapshot(state_dict={}, update_idx=0)
        b = LeagueSnapshot(state_dict={}, update_idx=1)
        a.metadata["x"] = 1
        assert b.metadata == {}
