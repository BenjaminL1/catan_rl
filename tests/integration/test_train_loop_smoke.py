"""End-to-end smoke for the Phase 10 training loop.

These tests run the actual ``run_training`` entry point on a tiny
config (n_envs=2, n_steps=8, ~few PPO updates) against a real
``CatanPolicy`` + ``CatanEnv``. They are integration-scoped because
they exercise every Phase 0-8 module together; unit-level coverage
of the individual pieces lives under ``tests/unit/``.

Pins:

1. Loop runs N PPO updates without crashing.
2. Checkpoints appear on disk at the configured cadence.
3. League grows snapshots at the configured cadence.
4. Resume from latest checkpoint advances update_idx and global_step.
5. TrainingState survives ``run_training`` exit (vec env closed, tb
   writer closed) — operator can introspect post-run.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.training_loop import (
    TrainingState,
    build_training_state,
    maybe_resume_from_checkpoint,
    run_training,
    run_training_loop,
)


def _tiny_cfg(*, total_updates: int = 2) -> TrainConfig:
    """Smallest viable config that still exercises the full plumbing."""
    cfg = TrainConfig.default()
    return replace(
        cfg,
        rollout=replace(cfg.rollout, n_envs=2, n_steps=8, max_turns=25),
        ppo=replace(cfg.ppo, batch_size=4, n_epochs=1, target_kl=0.0),
        checkpoint=replace(cfg.checkpoint, save_every_updates=1, keep_last_n=4),
        eval=replace(
            cfg.eval,
            eval_every_updates=10_000,  # disable eval — keeps the smoke fast
            eval_games=2,
        ),
        league=replace(
            cfg.league,
            add_snapshot_every_n_updates=1,
            maxlen=4,
            heuristic_weight=1.0,
        ),
        total_steps=total_updates * 2 * 8,
    )


@pytest.fixture
def silent_logger() -> logging.Logger:
    log = logging.getLogger("catan_rl.train.smoke")
    log.setLevel(logging.CRITICAL)
    return log


class TestEndToEndSmoke:
    def test_two_updates_complete(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        cfg = _tiny_cfg(total_updates=2)
        state = run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        assert isinstance(state, TrainingState)
        assert state.update_idx == 2
        # 2 envs * 8 steps * 2 updates = 32 transitions.
        assert state.global_step == 32

    def test_checkpoints_saved_at_cadence(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        cfg = _tiny_cfg(total_updates=3)
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=3,
            open_tb=False,
        )
        ckpt_dir = tmp_path / "checkpoints"
        files = sorted(p.name for p in ckpt_dir.iterdir())
        # save_every_updates=1 → one ckpt per update.
        assert files == [
            "ckpt_000000000.pt",
            "ckpt_000000001.pt",
            "ckpt_000000002.pt",
        ]

    def test_terminal_save_on_non_cadence_final(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        # 3 updates @ save_every=2: cadence saves only update_idx 1 ((1+1)%2==0).
        # The terminal save must capture the FINAL update_idx 2 ((2+1)%2!=0), so the
        # last partial window isn't dropped (the exploiter-probe footgun, 005 review).
        cfg = replace(
            _tiny_cfg(total_updates=3),
            checkpoint=replace(_tiny_cfg().checkpoint, save_every_updates=2, keep_last_n=8),
        )
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=3,
            open_tb=False,
        )
        files = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
        assert "ckpt_000000002.pt" in files, (
            "terminal save did not write the FINAL update's checkpoint"
        )
        assert "ckpt_000000001.pt" in files, "cadence save missing"
        assert "ckpt_000000000.pt" not in files, (
            "update 0 is neither on cadence nor final — should not save"
        )

    def test_league_snapshots_added(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        # add_snapshot_every_n_updates=1 BUT the league skips update 0
        # (Phase 6 reviewer fix), so after 3 updates we expect 2
        # snapshots (updates 1 and 2).
        cfg = _tiny_cfg(total_updates=3)
        state = run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=3,
            open_tb=False,
        )
        assert state.league.n_snapshots() == 2


class TestResume:
    def test_resume_advances_update_idx(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        cfg = _tiny_cfg(total_updates=4)
        # First half: 2 updates, saves checkpoints.
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        # Second half: 2 more updates. The resume picks up at
        # update_idx=2 (saved ckpt was for update 1, so next = 2).
        state = run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        assert state.resumed_from is not None
        assert state.resumed_from.name == "ckpt_000000001.pt"
        assert state.update_idx == 4
        assert state.global_step == 64  # 4 updates * 16 transitions

    def test_resume_preserves_league_snapshots(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        cfg = _tiny_cfg(total_updates=4)
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        state = run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        # After 4 total updates with add_every=1 and skip-update-0,
        # the league should have snapshots for updates 1, 2, 3.
        assert state.league.n_snapshots() == 3


class TestBuildOnly:
    """Test the constructor path without running the loop."""

    def test_build_training_state_minimal(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        cfg = _tiny_cfg(total_updates=2)
        state = build_training_state(
            cfg, run_dir=tmp_path, device_label="cpu", logger=silent_logger
        )
        try:
            assert state.policy.num_parameters() > 0
            assert state.vec_env.n_envs == 2
            assert state.buffer.n_envs == 2
            assert state.buffer.n_steps == 8
            assert state.update_idx == 0
            assert state.global_step == 0
            # ckpt dir created.
            assert (tmp_path / "checkpoints").exists()
        finally:
            state.vec_env.close()

    def test_resume_noop_on_empty_dir(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        cfg = _tiny_cfg(total_updates=1)
        state = build_training_state(
            cfg, run_dir=tmp_path, device_label="cpu", logger=silent_logger
        )
        try:
            result = maybe_resume_from_checkpoint(state, logger=silent_logger)
            assert result is None
            assert state.update_idx == 0
            assert state.resumed_from is None
        finally:
            state.vec_env.close()


class TestLoopHonoursMaxUpdates:
    def test_max_updates_caps_loop(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        # n_updates_total in cfg is 4, but we cap at 1.
        cfg = _tiny_cfg(total_updates=4)
        state = build_training_state(
            cfg, run_dir=tmp_path, device_label="cpu", logger=silent_logger
        )
        try:
            run_training_loop(
                state,
                run_dir=tmp_path,
                max_updates=1,
                logger=silent_logger,
                open_tb=False,
            )
            assert state.update_idx == 1
            assert state.global_step == 16
        finally:
            state.vec_env.close()


class TestRNGHygiene:
    def test_reset_all_called_with_explicit_seeds(
        self, tmp_path: Path, silent_logger: logging.Logger, monkeypatch
    ) -> None:
        # Reviewer-caught HIGH: ``state.vec_env.reset_all(seeds=None)``
        # would silently draw per-env seeds from ``np.random.randint``
        # and desynchronise the resumed rollout stream. Spy on the
        # reset_all call to assert the loop ALWAYS passes an explicit
        # ``seeds`` list.
        cfg = _tiny_cfg(total_updates=1)
        state = build_training_state(
            cfg, run_dir=tmp_path, device_label="cpu", logger=silent_logger
        )
        seen_seeds: list = []
        original = state.vec_env.reset_all

        def _spy_reset_all(*, seeds=None):
            seen_seeds.append(seeds)
            return original(seeds=seeds)

        monkeypatch.setattr(state.vec_env, "reset_all", _spy_reset_all)
        try:
            run_training_loop(
                state,
                run_dir=tmp_path,
                max_updates=1,
                logger=silent_logger,
                open_tb=False,
            )
        finally:
            state.vec_env.close()

        # Exactly one reset_all call (the initial one before the loop
        # body), with an explicit list of n_envs seeds derived from
        # cfg.seed — never None.
        assert len(seen_seeds) == 1
        assert seen_seeds[0] is not None
        assert len(seen_seeds[0]) == cfg.rollout.n_envs
        assert all(isinstance(s, int) for s in seen_seeds[0])


class TestNonFiniteLossGuard:
    def test_nan_policy_loss_raises(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        # Patch trainer.update to emit a NaN policy_loss; the loop
        # must raise immediately before the next checkpoint write.
        from catan_rl.ppo.training_loop import build_training_state

        cfg = _tiny_cfg(total_updates=2)
        state = build_training_state(
            cfg, run_dir=tmp_path, device_label="cpu", logger=silent_logger
        )
        try:
            original_update = state.trainer.update

            class _NaNMetrics:
                policy_loss = float("nan")
                value_loss = 0.5
                entropy_bonus = 1.0
                belief_loss = 0.0
                total_loss = float("nan")
                approx_kl = 0.01
                clip_frac = 0.1
                ratio_mean = 1.0
                grad_norm = 0.5
                lr = 3e-4
                entropy_coef = 0.01
                n_epochs_run = 1
                n_sgd_steps = 4

            def _broken_update(*args, **kwargs):
                _ = original_update(*args, **kwargs)
                return _NaNMetrics()

            state.trainer.update = _broken_update  # type: ignore[assignment]

            with pytest.raises(RuntimeError, match="non-finite loss"):
                run_training_loop(
                    state,
                    run_dir=tmp_path,
                    max_updates=1,
                    logger=silent_logger,
                    open_tb=False,
                )
        finally:
            state.vec_env.close()


class TestMetricsJSONL:
    def _eval_cfg(self, total_updates: int = 2) -> TrainConfig:
        # Override the base tiny cfg's disabled eval so we exercise both train
        # and eval lines. Pin eval_opponents explicitly (the dataclass default
        # was trimmed to ("heuristic",) by the wall-clock audit) so the
        # 2-opponent eval-line count asserted below is self-contained.
        cfg = _tiny_cfg(total_updates=total_updates)
        return replace(
            cfg,
            eval=replace(cfg.eval, eval_every_updates=1, eval_opponents=("random", "heuristic")),
        )

    def test_file_appears_with_expected_schema(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        import json

        from catan_rl.ppo.training_loop import METRICS_FILENAME

        cfg = self._eval_cfg(total_updates=2)
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        path = tmp_path / METRICS_FILENAME
        assert path.exists()
        records = [json.loads(line) for line in path.read_text().splitlines()]

        train = [r for r in records if r["kind"] == "train"]
        evals = [r for r in records if r["kind"] == "eval"]
        # 2 PPO updates → 2 train lines.
        assert len(train) == 2
        # 2 updates × 2 opponents (random, heuristic) × eval_every=1 → 4 eval lines.
        assert len(evals) == 4

        # Train schema.
        required_train = {
            "kind",
            "wall_time",
            "update_idx",
            "global_step",
            "policy_loss",
            "value_loss",
            "entropy_bonus",
            "total_loss",
            "approx_kl",
            "clip_frac",
            "grad_norm",
            "lr",
        }
        assert required_train.issubset(train[0].keys())
        # Eval schema.
        required_eval = {
            "kind",
            "wall_time",
            "update_idx",
            "global_step",
            "opponent_type",
            "wr",
            "ci_low",
            "ci_high",
            "n_games",
            "n_wins",
            "n_truncated",
            "wr_seat0",
            "wr_seat1",
        }
        assert required_eval.issubset(evals[0].keys())
        # CI is well-formed.
        for e in evals:
            assert 0.0 <= e["ci_low"] <= e["ci_high"] <= 1.0
            assert e["opponent_type"] in ("random", "heuristic")

    def test_resume_appends_does_not_clobber(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        # First run writes 2 lines, second run resumes and appends 2 more.
        from catan_rl.ppo.training_loop import METRICS_FILENAME

        cfg = _tiny_cfg(total_updates=4)
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        first_lines = (tmp_path / METRICS_FILENAME).read_text().splitlines()
        assert len(first_lines) == 2  # 2 train, no eval (eval disabled in _tiny_cfg)

        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        all_lines = (tmp_path / METRICS_FILENAME).read_text().splitlines()
        # The first 2 lines are unchanged; resume only appended.
        assert all_lines[:2] == first_lines
        assert len(all_lines) == 4

    def test_disabled_via_flag(self, tmp_path: Path, silent_logger: logging.Logger) -> None:
        from catan_rl.ppo.training_loop import METRICS_FILENAME

        cfg = _tiny_cfg(total_updates=1)
        run_training(
            cfg,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=1,
            open_tb=False,
            open_metrics_jsonl=False,
        )
        assert not (tmp_path / METRICS_FILENAME).exists()

    def test_numpy_scalar_serialisation(
        self, tmp_path: Path, silent_logger: logging.Logger
    ) -> None:
        # The writer must cope with numpy scalars (np.float32 etc.) that
        # ``json.dumps`` would otherwise reject. _write_jsonl unit-tests
        # cover this directly without the full loop overhead.
        import io
        import json

        import numpy as np

        from catan_rl.ppo.training_loop import _write_jsonl

        buf = io.StringIO()
        _write_jsonl(
            buf,
            {
                "kind": "train",
                "wall_time": "2026-06-03T22:00:00Z",
                "update_idx": np.int64(5),
                "global_step": np.int32(80),
                "policy_loss": np.float32(-0.04),
                "value_loss": np.float64(0.5),
            },
        )
        line = buf.getvalue().strip()
        rec = json.loads(line)
        assert rec["update_idx"] == 5
        assert rec["global_step"] == 80
        assert rec["policy_loss"] == pytest.approx(-0.04, abs=1e-6)
        assert rec["value_loss"] == pytest.approx(0.5, abs=1e-12)


class TestResumeConfigDiff:
    def test_diff_emitted_on_critical_field_mismatch(
        self, tmp_path: Path, silent_logger: logging.Logger, caplog
    ) -> None:
        # First half: save a checkpoint with gamma=0.99.
        from dataclasses import replace

        cfg_a = _tiny_cfg(total_updates=4)
        cfg_a = replace(cfg_a, gae=replace(cfg_a.gae, gamma=0.99))
        run_training(
            cfg_a,
            run_dir=tmp_path,
            device_label="cpu",
            logger=silent_logger,
            max_updates=2,
            open_tb=False,
        )
        # Second half: resume with gamma=0.995 (the v2 default).
        cfg_b = replace(cfg_a, gae=replace(cfg_a.gae, gamma=0.995))
        with caplog.at_level(logging.WARNING, logger="catan_rl.train.smoke"):
            run_training(
                cfg_b,
                run_dir=tmp_path,
                device_label="cpu",
                logger=silent_logger,
                max_updates=1,
                open_tb=False,
            )
        # Find the warning about gae.gamma.
        warning_msgs = [r.getMessage() for r in caplog.records]
        assert any("gae.gamma" in m and "0.99" in m for m in warning_msgs), (
            f"expected gae.gamma diff warning; got {warning_msgs!r}"
        )
