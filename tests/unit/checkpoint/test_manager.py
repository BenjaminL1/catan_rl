"""Tests for `checkpoint/manager.py`.

Covered:
1. Round-trip preserves policy params bit-exact.
2. Round-trip preserves Adam moment buffers.
3. Round-trip preserves PRNG state (next draw matches).
4. Round-trip preserves league snapshots + next_snapshot_id.
5. Atomic write: tmp file removed after success; corrupted dest
   doesn't appear on a simulated mid-write crash.
6. Schema mismatch → CheckpointError.
7. Corrupt file → CheckpointError.
8. Pruning keeps exactly keep_last_n.
9. list_checkpoints ignores junk files in the directory.
10. apply_to_league preserves id monotonicity after eviction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl.checkpoint.manager import (
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
    checkpoint_filename,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from catan_rl.ppo.arguments import LeagueConfig
from catan_rl.selfplay.league import League

# ---------------------------------------------------------------------------
# Tiny live objects
# ---------------------------------------------------------------------------


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)


def _fresh_policy_and_optimizer(seed: int = 0) -> tuple[_TinyPolicy, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    policy = _TinyPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    return policy, opt


def _take_one_optim_step(policy: _TinyPolicy, optimizer: torch.optim.Optimizer) -> None:
    # Run one forward+backward+step so the optimizer accumulates Adam
    # moment buffers; without this the optimiser state is empty and
    # the round-trip is vacuous.
    x = torch.randn(2, 8)
    y = policy.fc(x).sum()
    y.backward()
    optimizer.step()
    optimizer.zero_grad()


def _params_equal(a: nn.Module, b: nn.Module) -> bool:
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if na != nb:
            return False
        if not torch.equal(pa, pb):
            return False
    return True


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_policy_params_bit_exact(self, tmp_path: Path) -> None:
        policy_a, opt_a = _fresh_policy_and_optimizer(seed=0)
        _take_one_optim_step(policy_a, opt_a)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={"foo": 1},
            policy=policy_a,
            optimizer=opt_a,
            update_idx=10,
            global_step=1000,
        )

        # Different seed → different params at construction.
        policy_b, _opt_b = _fresh_policy_and_optimizer(seed=1)
        assert not _params_equal(policy_a, policy_b)
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        payload.apply_to_policy(policy_b)
        assert _params_equal(policy_a, policy_b)
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.update_idx == 10
        assert payload.global_step == 1000

    def test_optimizer_state_roundtrip(self, tmp_path: Path) -> None:
        policy_a, opt_a = _fresh_policy_and_optimizer(seed=0)
        _take_one_optim_step(policy_a, opt_a)
        # Verify Adam state has accumulated.
        adam_state_a = {id(p): dict(opt_a.state[p]) for p in policy_a.parameters()}
        assert any(v for v in adam_state_a.values())  # at least one moment

        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy_a,
            optimizer=opt_a,
            update_idx=0,
            global_step=0,
        )

        policy_b, opt_b = _fresh_policy_and_optimizer(seed=1)
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        payload.apply_to_policy(policy_b)
        payload.apply_to_optimizer(opt_b)
        # Compare moment buffers across param-key positions.
        for (_pa, sa), (_pb, sb) in zip(opt_a.state.items(), opt_b.state.items(), strict=True):
            for k in ("exp_avg", "exp_avg_sq", "step"):
                if k in sa:
                    assert torch.equal(sa[k], sb[k]), k

    def test_rng_roundtrip_next_draw_matches(self, tmp_path: Path) -> None:
        np.random.seed(123)
        random.seed(456)
        torch.manual_seed(789)
        # Burn a couple draws to exercise the state.
        _ = np.random.random(3)
        _ = random.random()
        _ = torch.randn(2)

        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
            capture_rng=True,
        )
        # Reference: next draws from the current RNG state.
        np_next_ref = np.random.random(4).tolist()
        py_next_ref = [random.random() for _ in range(3)]
        torch_next_ref = torch.randn(5).tolist()

        # Pollute the global state.
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        payload = load_checkpoint(tmp_path / "ckpt.pt")
        payload.apply_rng_state()
        assert np.random.random(4).tolist() == np_next_ref
        assert [random.random() for _ in range(3)] == py_next_ref
        assert torch.randn(5).tolist() == torch_next_ref

    def test_capture_rng_false_omits_state(self, tmp_path: Path) -> None:
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
            capture_rng=False,
        )
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        assert payload.rng_state is None
        # apply_rng_state is a no-op when state is None.
        payload.apply_rng_state()


# ---------------------------------------------------------------------------
# League round-trip
# ---------------------------------------------------------------------------


class TestLeagueRoundTrip:
    def _build_league_with_snapshots(self, n: int = 3) -> League:
        league = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
        for i in range(n):
            policy, _ = _fresh_policy_and_optimizer(seed=i)
            league.add_snapshot(
                policy.state_dict(),
                update_idx=(i + 1) * 10,
                metadata={"wr": 0.5 + 0.05 * i},
            )
        return league

    def test_snapshots_and_id_cursor_preserved(self, tmp_path: Path) -> None:
        league_a = self._build_league_with_snapshots(3)
        policy, opt = _fresh_policy_and_optimizer(seed=99)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=42,
            global_step=4242,
            league=league_a,
        )
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        assert payload.league_state is not None
        assert len(payload.league_state["snapshots"]) == 3
        assert payload.league_state["next_snapshot_id"] == 3

        league_b = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
        payload.apply_to_league(league_b)
        assert league_b.n_snapshots() == 3
        # Snapshot IDs preserved bit-for-bit.
        assert [s.snapshot_id for s in league_b._snapshots] == [0, 1, 2]
        # Metadata preserved.
        assert league_b._snapshots[2].metadata == {"wr": 0.6}
        # Next id cursor preserved so adding another snapshot gets id=3.
        new_id = league_b.add_snapshot(policy.state_dict(), update_idx=100)
        assert new_id == 3

    def test_apply_to_league_clears_existing(self, tmp_path: Path) -> None:
        league_a = self._build_league_with_snapshots(2)
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
            league=league_a,
        )
        # Target league already has snapshots → must be wiped.
        league_b = self._build_league_with_snapshots(5)
        assert league_b.n_snapshots() == 5
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        payload.apply_to_league(league_b)
        assert league_b.n_snapshots() == 2


# ---------------------------------------------------------------------------
# Atomic + error paths
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_tmp_file_removed_after_success(self, tmp_path: Path) -> None:
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        dest = tmp_path / "ckpt.pt"
        save_checkpoint(
            dest,
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        assert dest.exists()
        # The .tmp file must have been os.replace'd away.
        assert not (tmp_path / "ckpt.pt.tmp").exists()

    def test_existing_dest_survives_mid_write_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First, write a valid checkpoint.
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        dest = tmp_path / "ckpt.pt"
        save_checkpoint(
            dest,
            config={"v": "good"},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        good_bytes = dest.read_bytes()

        # Patch torch.save to raise mid-write. The atomic guard means
        # the original ``ckpt.pt`` should remain intact.
        def _boom(*_args, **_kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr("catan_rl.checkpoint.manager.torch.save", _boom)
        with pytest.raises(OSError, match="simulated disk full"):
            save_checkpoint(
                dest,
                config={"v": "bad"},
                policy=policy,
                optimizer=opt,
                update_idx=1,
                global_step=1,
            )
        # Original file untouched.
        assert dest.read_bytes() == good_bytes
        # Tmp file from the failed save must be cleaned up.
        assert not (tmp_path / "ckpt.pt.tmp").exists()

    def test_fsync_parent_dir_called(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure save_checkpoint invokes _fsync_dir on the parent
        # directory (best-effort durability on POSIX). We monkeypatch
        # the helper to observe the call.
        called: list[Path] = []
        from catan_rl.checkpoint import manager as mgr_mod

        original = mgr_mod._fsync_dir

        def _spy(path: Path) -> None:
            called.append(Path(path))
            original(path)

        monkeypatch.setattr(mgr_mod, "_fsync_dir", _spy)
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        assert called == [tmp_path]


class TestVecEnvRoundTrip:
    def test_per_env_rng_state_preserved(self, tmp_path: Path) -> None:
        # Build a minimal stand-in for SerialVecEnv's _reset_rngs field.
        # Capture each Generator's state, save, mutate, restore, and
        # verify the next draws match the captured reference.
        ss = np.random.SeedSequence(12345)
        child = ss.spawn(4)
        vec_env_a = type("Vec", (), {"_reset_rngs": [np.random.default_rng(s) for s in child]})()
        # Burn each Generator a different amount so they hold distinct
        # internal states.
        for i, gen in enumerate(vec_env_a._reset_rngs):
            _ = gen.integers(0, 2**31 - 1, size=i + 1)

        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
            vec_env=vec_env_a,
        )
        # Reference: next draws.
        ref = [int(gen.integers(0, 2**31 - 1)) for gen in vec_env_a._reset_rngs]

        # New vec env with the same shape but fresh state.
        vec_env_b = type(
            "Vec",
            (),
            {"_reset_rngs": [np.random.default_rng(99) for _ in range(4)]},
        )()
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        assert payload.vec_env_state is not None
        assert len(payload.vec_env_state["reset_rng_states"]) == 4
        payload.apply_to_vec_env(vec_env_b)
        got = [int(gen.integers(0, 2**31 - 1)) for gen in vec_env_b._reset_rngs]
        assert got == ref

    def test_n_envs_mismatch_raises(self, tmp_path: Path) -> None:
        vec_env_a = type(
            "Vec",
            (),
            {"_reset_rngs": [np.random.default_rng(s) for s in np.random.SeedSequence(0).spawn(4)]},
        )()
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
            vec_env=vec_env_a,
        )
        vec_env_b = type(
            "Vec",
            (),
            {"_reset_rngs": [np.random.default_rng(0) for _ in range(2)]},
        )()
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        with pytest.raises(CheckpointError, match="n_envs mismatch"):
            payload.apply_to_vec_env(vec_env_b)

    def test_no_vec_env_section_is_noop(self, tmp_path: Path) -> None:
        # Saved without vec_env=; restoring is a no-op.
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        payload = load_checkpoint(tmp_path / "ckpt.pt")
        assert payload.vec_env_state is None
        # Should not raise even though we pass a vec_env to apply_to.
        fake_vec = type("Vec", (), {"_reset_rngs": [np.random.default_rng(0)]})()
        payload.apply_to_vec_env(fake_vec)


class TestApplyAll:
    def test_calls_every_section(self, tmp_path: Path) -> None:
        # End-to-end: a single apply_all call restores policy +
        # optimizer + league + vec_env + RNGs.
        np.random.seed(11)
        random.seed(22)
        torch.manual_seed(33)
        _ = np.random.random(2)
        _ = random.random()
        _ = torch.randn(1)

        policy_a, opt_a = _fresh_policy_and_optimizer(seed=0)
        _take_one_optim_step(policy_a, opt_a)
        league_a = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
        league_a.add_snapshot(policy_a.state_dict(), update_idx=1)
        vec_env_a = type(
            "Vec",
            (),
            {"_reset_rngs": [np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(3)]},
        )()
        for gen in vec_env_a._reset_rngs:
            _ = gen.integers(0, 1000)

        save_checkpoint(
            tmp_path / "ckpt.pt",
            config={},
            policy=policy_a,
            optimizer=opt_a,
            update_idx=5,
            global_step=500,
            league=league_a,
            vec_env=vec_env_a,
        )
        np_ref = np.random.random(3).tolist()

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        policy_b, opt_b = _fresh_policy_and_optimizer(seed=1)
        league_b = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
        vec_env_b = type(
            "Vec",
            (),
            {"_reset_rngs": [np.random.default_rng(0) for _ in range(3)]},
        )()

        payload = load_checkpoint(tmp_path / "ckpt.pt")
        payload.apply_all(
            policy=policy_b,
            optimizer=opt_b,
            league=league_b,
            vec_env=vec_env_b,
        )
        assert _params_equal(policy_a, policy_b)
        assert league_b.n_snapshots() == 1
        assert np.random.random(3).tolist() == np_ref


class TestErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "does_not_exist.pt")

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.pt"
        bad.write_bytes(b"not a torch payload")
        with pytest.raises(CheckpointError, match=r"torch\.load"):
            load_checkpoint(bad)

    def test_non_dict_payload_raises(self, tmp_path: Path) -> None:
        wrong = tmp_path / "wrong.pt"
        torch.save([1, 2, 3], wrong)
        with pytest.raises(CheckpointError, match="not a dict"):
            load_checkpoint(wrong)

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        flat = tmp_path / "flat.pt"
        torch.save({"foo": "bar"}, flat)
        with pytest.raises(CheckpointError, match="missing 'schema_version'"):
            load_checkpoint(flat)


# ---------------------------------------------------------------------------
# Listing + pruning
# ---------------------------------------------------------------------------


class TestListAndPrune:
    def _write_ckpts(self, dir_: Path, updates: list[int]) -> None:
        for u in updates:
            (dir_ / checkpoint_filename(u)).write_bytes(b"placeholder")

    def test_list_returns_sorted_by_update_idx(self, tmp_path: Path) -> None:
        self._write_ckpts(tmp_path, [5, 1, 100, 10])
        files = list_checkpoints(tmp_path)
        names = [p.name for p in files]
        assert names == [
            "ckpt_000000001.pt",
            "ckpt_000000005.pt",
            "ckpt_000000010.pt",
            "ckpt_000000100.pt",
        ]

    def test_list_ignores_unrelated_files(self, tmp_path: Path) -> None:
        self._write_ckpts(tmp_path, [1, 2])
        (tmp_path / "notes.md").write_text("not a checkpoint")
        (tmp_path / "ckpt_old.pt").write_bytes(b"wrong name")
        (tmp_path / "ckpt_abc.pt").write_bytes(b"non-numeric")
        files = list_checkpoints(tmp_path)
        assert len(files) == 2

    def test_prune_keeps_last_n(self, tmp_path: Path) -> None:
        self._write_ckpts(tmp_path, [1, 2, 3, 4, 5])
        deleted = prune_checkpoints(tmp_path, keep_last_n=2)
        assert [p.name for p in deleted] == [
            "ckpt_000000001.pt",
            "ckpt_000000002.pt",
            "ckpt_000000003.pt",
        ]
        remaining = [p.name for p in list_checkpoints(tmp_path)]
        assert remaining == ["ckpt_000000004.pt", "ckpt_000000005.pt"]

    def test_prune_zero_disables(self, tmp_path: Path) -> None:
        self._write_ckpts(tmp_path, [1, 2, 3])
        deleted = prune_checkpoints(tmp_path, keep_last_n=0)
        assert deleted == []
        assert len(list_checkpoints(tmp_path)) == 3

    def test_prune_noop_when_under_cap(self, tmp_path: Path) -> None:
        self._write_ckpts(tmp_path, [1, 2])
        deleted = prune_checkpoints(tmp_path, keep_last_n=5)
        assert deleted == []


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_and_latest(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, keep_last_n=3)
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        for u in (10, 20, 30, 40, 50):
            mgr.save(
                config={},
                policy=policy,
                optimizer=opt,
                update_idx=u,
                global_step=u * 100,
            )
        latest = mgr.latest()
        assert latest is not None and latest.name == "ckpt_000000050.pt"
        # Pruning kept only the last 3.
        assert [p.name for p in mgr.list()] == [
            "ckpt_000000030.pt",
            "ckpt_000000040.pt",
            "ckpt_000000050.pt",
        ]

    def test_latest_none_on_empty_dir(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path)
        assert mgr.latest() is None

    def test_save_without_optimizer_state(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path, save_optimizer_state=False)
        policy, opt = _fresh_policy_and_optimizer(seed=0)
        _take_one_optim_step(policy, opt)
        path = mgr.save(
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        payload = load_checkpoint(path)
        assert payload.optimizer_state_dict is None

    def test_keep_last_n_negative_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="keep_last_n"):
            CheckpointManager(tmp_path, keep_last_n=-1)
