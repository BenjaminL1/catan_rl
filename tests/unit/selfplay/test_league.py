"""Tests for `selfplay/league.py` — minimal opponent league.

Pins:

1. Sampling weights — uniform random across many samples → ratios
   converge to the configured weights.
2. Snapshot retention — deque is bounded by maxlen, oldest entries
   evicted FIFO.
3. Snapshot cloning — modifying the source state dict after
   ``add_snapshot`` does NOT mutate the stored snapshot.
4. Snapshot sampling is wired — a non-empty pool yields snapshot kinds and
   assignments carry a stable snapshot_id; an empty pool falls back (FR-011).
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
    OPPONENT_KIND_SNAPSHOT,
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

    def test_snapshot_weight_requires_heuristic_floor(self) -> None:
        # PG-2 guard (T027): self-play with no heuristic weight is rejected.
        with pytest.raises(ValueError, match="requires heuristic_weight"):
            LeagueConfig(random_weight=0.0, heuristic_weight=0.0, snapshot_weight=1.0)

    def test_heuristic_floor_can_be_opted_out(self) -> None:
        LeagueConfig(
            random_weight=0.0,
            heuristic_weight=0.0,
            snapshot_weight=1.0,
            require_heuristic_floor=False,
        )


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

    def test_snapshot_weight_with_populated_pool_yields_snapshots(self) -> None:
        # Snapshot path is wired (T017): a populated pool yields snapshot kinds,
        # not a raise.
        lg = League(
            LeagueConfig(
                random_weight=0.0,
                heuristic_weight=0.0,
                snapshot_weight=1.0,
                require_heuristic_floor=False,
            )
        )
        lg.add_snapshot({"w": torch.zeros(1)}, update_idx=4)
        rng = np.random.default_rng(0)
        mix = lg.build_env_opponent_mix(n_envs=4, rng=rng)
        assert all(k == OPPONENT_KIND_SNAPSHOT for k in mix)

    def test_assignments_carry_stable_snapshot_id(self) -> None:
        lg = League(
            LeagueConfig(
                random_weight=0.0,
                heuristic_weight=0.0,
                snapshot_weight=1.0,
                require_heuristic_floor=False,
            )
        )
        sid = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=4)
        rng = np.random.default_rng(0)
        assigns = lg.build_env_opponent_assignments(n_envs=4, rng=rng)
        assert all(a.kind == OPPONENT_KIND_SNAPSHOT for a in assigns)
        assert all(a.snapshot_id == sid for a in assigns)
        # The sampled id resolves back through the stable-id lookup.
        assert lg.peek_by_id(sid) is not None

    def test_assignments_fall_back_when_pool_empty(self) -> None:
        # snapshot_weight>0 but empty pool -> heuristic, no snapshot id (FR-011).
        lg = League(LeagueConfig(random_weight=0.0, heuristic_weight=1.0, snapshot_weight=1.0))
        rng = np.random.default_rng(0)
        assigns = lg.build_env_opponent_assignments(n_envs=4, rng=rng)
        assert all(a.kind == OPPONENT_KIND_HEURISTIC for a in assigns)
        assert all(a.snapshot_id is None for a in assigns)


class TestAnchor:
    def test_set_anchor_resolves_and_reports_id(self) -> None:
        lg = League(LeagueConfig())
        aid = lg.set_anchor({"w": torch.ones(2)}, update_idx=99)
        assert lg.anchor_id() == aid
        snap = lg.peek_by_id(aid)
        assert snap is not None and torch.equal(snap.state_dict["w"], torch.ones(2))
        assert aid in lg.snapshot_ids()

    def test_anchor_never_evicted(self) -> None:
        lg = League(LeagueConfig(maxlen=3))
        aid = lg.set_anchor({"w": torch.zeros(1)}, update_idx=0)
        for i in range(10):  # overflow the maxlen=3 pool many times over
            lg.add_snapshot({"w": torch.zeros(1)}, update_idx=i + 1)
        assert lg.n_snapshots() == 3  # pool stays capped
        assert lg.peek_by_id(aid) is not None  # anchor survived all evictions
        assert aid in lg.snapshot_ids()

    def test_anchor_gets_reserved_weight(self) -> None:
        # 50/50 anchor vs pool -> anchor id appears ~half the assignments.
        lg = League(
            LeagueConfig(
                heuristic_weight=0.0,
                snapshot_weight=1.0,
                anchor_weight=1.0,
                require_heuristic_floor=False,
            )
        )
        aid = lg.set_anchor({"w": torch.zeros(1)}, update_idx=0)
        pid = lg.add_snapshot({"w": torch.ones(1)}, update_idx=4)
        assigns = lg.build_env_opponent_assignments(n_envs=200, rng=np.random.default_rng(0))
        assert all(a.kind == OPPONENT_KIND_SNAPSHOT for a in assigns)
        n_anchor = sum(a.snapshot_id == aid for a in assigns)
        assert sum(a.snapshot_id == pid for a in assigns) + n_anchor == 200
        assert 70 < n_anchor < 130  # ~100 expected; generous band

    def test_anchor_absent_when_weight_zero(self) -> None:
        # Anchor set but anchor_weight=0 (default) -> never assigned.
        lg = League(LeagueConfig(heuristic_weight=1.0, snapshot_weight=1.0))
        aid = lg.set_anchor({"w": torch.zeros(1)}, update_idx=0)
        lg.add_snapshot({"w": torch.ones(1)}, update_idx=4)
        assigns = lg.build_env_opponent_assignments(n_envs=100, rng=np.random.default_rng(0))
        assert all(a.snapshot_id != aid for a in assigns)


class TestPFSP:
    def _pool_cfg(self, **kw):
        return LeagueConfig(
            heuristic_weight=0.0, snapshot_weight=1.0, require_heuristic_floor=False, **kw
        )

    def test_record_outcome_ema_and_games(self) -> None:
        lg = League(self._pool_cfg(pfsp_ema_alpha=0.5))
        sid = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)
        lg.record_outcome(sid, agent_won=True)  # seed: p=1.0, games=1
        assert lg.opponent_win_rate(sid) == (1.0, 1)
        lg.record_outcome(sid, agent_won=False)  # p = 0.5*1 + 0.5*0 = 0.5, games=2
        p, g = lg.opponent_win_rate(sid)
        assert p == 0.5 and g == 2

    def test_record_outcome_ignores_unknown_id(self) -> None:
        lg = League(self._pool_cfg())
        lg.record_outcome(999, agent_won=True)  # no such snapshot -> no-op
        assert lg.opponent_win_rate(999) == (0.0, 0)

    def test_stats_pruned_on_eviction(self) -> None:
        lg = League(self._pool_cfg(maxlen=2))
        s0 = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)
        lg.record_outcome(s0, agent_won=True)
        lg.add_snapshot({"w": torch.zeros(1)}, update_idx=1)
        lg.add_snapshot({"w": torch.zeros(1)}, update_idx=2)  # evicts s0
        assert lg.peek_by_id(s0) is None
        assert lg.opponent_win_rate(s0) == (0.0, 0)  # stats pruned with the snapshot

    def test_hard_curve_prefers_harder_opponent(self) -> None:
        # SC-001: a 0.3-WR opponent is drawn >=2x as often as a 0.7-WR one.
        lg = League(self._pool_cfg(pfsp_enabled=True, pfsp_curve="hard", pfsp_min_games=1))
        easy = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)  # agent wins -> high WR
        hard = lg.add_snapshot({"w": torch.ones(1)}, update_idx=4)  # agent loses -> low WR
        for _ in range(10):
            lg.record_outcome(easy, agent_won=True)  # p~1.0 (after EMA)
            lg.record_outcome(hard, agent_won=False)  # p~0.0
        assigns = lg.build_env_opponent_assignments(n_envs=400, rng=np.random.default_rng(0))
        n_hard = sum(a.snapshot_id == hard for a in assigns)
        n_easy = sum(a.snapshot_id == easy for a in assigns)
        assert n_hard >= 2 * n_easy

    def test_cold_start_snapshot_sampled(self) -> None:
        # SC-002: a 0-game snapshot still gets a share alongside an established one.
        lg = League(self._pool_cfg(pfsp_enabled=True, pfsp_curve="hard", pfsp_min_games=5))
        old = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)
        for _ in range(10):
            lg.record_outcome(old, agent_won=True)  # established, easy -> low weight
        fresh = lg.add_snapshot({"w": torch.ones(1)}, update_idx=4)  # 0 games -> cold-start
        assigns = lg.build_env_opponent_assignments(n_envs=200, rng=np.random.default_rng(0))
        assert sum(a.snapshot_id == fresh for a in assigns) > 0

    def test_equal_winrate_is_uniform(self) -> None:
        # FR-008: equal WRs -> ~uniform (no degeneracy).
        lg = League(self._pool_cfg(pfsp_enabled=True, pfsp_curve="hard", pfsp_min_games=1))
        a = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)
        b = lg.add_snapshot({"w": torch.ones(1)}, update_idx=4)
        for _ in range(6):
            lg.record_outcome(a, agent_won=True)
            lg.record_outcome(b, agent_won=True)  # both p~1.0 -> both floored equally
        assigns = lg.build_env_opponent_assignments(n_envs=400, rng=np.random.default_rng(1))
        na = sum(x.snapshot_id == a for x in assigns)
        assert 150 < na < 250  # ~200, roughly uniform

    def test_off_is_byte_identical(self) -> None:
        # SC-003/FR-005: pfsp off (or curve uniform) reproduces the exact draw.
        base = LeagueConfig(
            heuristic_weight=0.0, snapshot_weight=1.0, require_heuristic_floor=False
        )
        pfsp_off = LeagueConfig(
            heuristic_weight=0.0,
            snapshot_weight=1.0,
            require_heuristic_floor=False,
            pfsp_enabled=True,
            pfsp_curve="uniform",
        )
        for cfg in (base, pfsp_off):
            lg = League(cfg)
            ids = [lg.add_snapshot({"w": torch.zeros(1)}, update_idx=i) for i in range(3)]
            for sid in ids:  # give them WR so a 'hard' curve WOULD differ
                lg.record_outcome(sid, agent_won=(sid % 2 == 0))
            seq = [
                a.snapshot_id
                for a in lg.build_env_opponent_assignments(n_envs=20, rng=np.random.default_rng(7))
            ]
            if cfg is base:
                baseline = seq
            else:
                assert seq == baseline  # identical to the non-PFSP draw

    def test_opponent_stats_round_trip(self) -> None:
        # SC-004: stats serialise + restore exactly.
        lg = League(self._pool_cfg(pfsp_ema_alpha=0.3))
        sid = lg.add_snapshot({"w": torch.zeros(1)}, update_idx=0)
        lg.record_outcome(sid, agent_won=True)
        lg.record_outcome(sid, agent_won=False)
        state = lg.opponent_stats_state()
        lg2 = League(self._pool_cfg())
        lg2.add_snapshot({"w": torch.zeros(1)}, update_idx=0)  # same id space (0)
        lg2.load_opponent_stats(state)
        assert lg2.opponent_win_rate(sid) == lg.opponent_win_rate(sid)


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
    :class:`CatanEnv`.

    The snapshot opponent path is now wired (US1): constructing a snapshot env
    must NOT raise — until a frozen policy is injected via
    ``set_snapshot_opponent``, it falls back to the heuristic body (FR-011).
    """

    def test_catan_env_accepts_snapshot_and_falls_back(self) -> None:
        from catan_rl.env.catan_env import CatanEnv
        from catan_rl.selfplay.league import OPPONENT_KIND_SNAPSHOT

        env = CatanEnv(opponent_type=OPPONENT_KIND_SNAPSHOT)  # no raise
        assert not env.has_snapshot_opponent  # empty pool -> heuristic fallback
        obs, _ = env.reset(seed=0)
        assert obs is not None  # plays via the heuristic body without error

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
