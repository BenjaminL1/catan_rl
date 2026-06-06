"""Tests for the PPO training entry point.

Pins:
1. CLI parses every flag with the documented type + default behavior.
2. ``resolve_config`` respects precedence: defaults < YAML < env < CLI.
3. ``setup_seeding`` seeds all three globals so dice rolls are
   reproducible across runs.
4. ``build_run_directory`` creates a timestamped sub-dir under
   ``output_dir`` and never collides on repeat invocations.
5. ``snapshot_config`` writes a YAML that round-trips back to the
   original ``TrainConfig`` (so the snapshot is a faithful reproducer).
6. ``main(--dry-run)`` exits 0 without invoking the trainer.
7. ``main()`` without ``--dry-run`` exits 2 (trainer not yet wired).

Implementation note: the entry point lived at ``scripts/train.py``
pre-cutover and was loaded via importlib's spec loader because the
script lived outside the package tree. Post maturin-sole-backend
cutover the body moved to :mod:`catan_rl.cli.train` (with
``scripts/train.py`` retained as a thin back-compat shim wired to
the canonical module), so the fixture now does a normal package
import.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture(scope="module")
def train_mod():
    """Import the canonical entry-point module.

    Returns :mod:`catan_rl.cli.train`. Tests below access its
    module-level helpers (``build_parser``, ``resolve_config``,
    ``setup_seeding``, ``build_run_directory``, ``snapshot_config``,
    ``main``).
    """
    import catan_rl.cli.train as mod

    return mod


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_no_args_uses_dataclass_defaults(self, train_mod) -> None:
        args = train_mod.build_parser().parse_args([])
        assert args.config is None
        assert args.run_name is None
        assert args.seed is None
        assert args.device is None
        assert args.total_steps is None
        assert args.n_envs is None
        assert args.dry_run is False
        assert args.verbose is False

    def test_all_overrides_parse(self, train_mod) -> None:
        args = train_mod.build_parser().parse_args(
            [
                "--config",
                "ppo.yaml",
                "--run-name",
                "exp1",
                "--output-dir",
                "runs/foo",
                "--seed",
                "7",
                "--device",
                "cpu",
                "--total-steps",
                "32768",
                "--n-envs",
                "16",
                "--dry-run",
                "--verbose",
            ]
        )
        assert args.config == Path("ppo.yaml")
        assert args.run_name == "exp1"
        assert args.output_dir == Path("runs/foo")
        assert args.seed == 7
        assert args.device == "cpu"
        assert args.total_steps == 32768
        assert args.n_envs == 16
        assert args.dry_run is True
        assert args.verbose is True

    def test_invalid_device_choice_rejected(self, train_mod) -> None:
        with pytest.raises(SystemExit):
            train_mod.build_parser().parse_args(["--device", "tpu"])


# ---------------------------------------------------------------------------
# Config resolution precedence
# ---------------------------------------------------------------------------


class TestResolveConfigPrecedence:
    def test_defaults_when_no_overrides(self, train_mod) -> None:
        args = train_mod.build_parser().parse_args([])
        cfg = train_mod.resolve_config(args)
        # Audit defaults pinned by Phase 1 tests; spot-check a few here.
        assert cfg.rollout.n_envs == 128
        assert cfg.seed == 42

    def test_yaml_overrides_defaults(self, train_mod, tmp_path: Path) -> None:
        path = tmp_path / "ppo.yaml"
        path.write_text(yaml.safe_dump({"rollout": {"n_envs": 64}}))
        args = train_mod.build_parser().parse_args(["--config", str(path)])
        cfg = train_mod.resolve_config(args)
        assert cfg.rollout.n_envs == 64

    def test_cli_overrides_yaml(self, train_mod, tmp_path: Path) -> None:
        path = tmp_path / "ppo.yaml"
        path.write_text(yaml.safe_dump({"rollout": {"n_envs": 64}}))
        args = train_mod.build_parser().parse_args(
            [
                "--config",
                str(path),
                "--n-envs",
                "32",
            ]
        )
        cfg = train_mod.resolve_config(args)
        assert cfg.rollout.n_envs == 32

    def test_cli_overrides_top_level(self, train_mod) -> None:
        # 32768 is a valid total_steps multiple at the default rollout
        # (32768 % (128*256) == 0).
        args = train_mod.build_parser().parse_args(["--total-steps", "32768"])
        cfg = train_mod.resolve_config(args)
        assert cfg.total_steps == 32768

    def test_cli_total_steps_validates_against_rollout(self, train_mod) -> None:
        # 50_000_000 doesn't divide 128*256 — Phase 1 validator must fire.
        args = train_mod.build_parser().parse_args(["--total-steps", "50000000"])
        with pytest.raises(ValueError, match=r"total_steps.*multiple"):
            train_mod.resolve_config(args)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_seeds_all_three_globals(self, train_mod) -> None:
        train_mod.setup_seeding(42)
        a_np = np.random.random()
        a_rand = random.random()
        import torch

        a_torch = torch.rand(1).item()

        train_mod.setup_seeding(42)
        b_np = np.random.random()
        b_rand = random.random()
        b_torch = torch.rand(1).item()

        assert a_np == b_np, "numpy not seeded"
        assert a_rand == b_rand, "stdlib random not seeded (StackedDice breaks)"
        assert a_torch == b_torch, "torch not seeded"


# ---------------------------------------------------------------------------
# Run directory + config snapshot
# ---------------------------------------------------------------------------


class TestRunDirectory:
    def test_creates_timestamped_subdir(self, train_mod, tmp_path: Path) -> None:
        from dataclasses import replace

        from catan_rl.ppo.arguments import TrainConfig

        cfg = replace(TrainConfig.default(), output_dir=str(tmp_path), run_name="exp")
        # Force a known timestamp (epoch=0 → 1969-12-31 16:00:00 local)
        run_dir = train_mod.build_run_directory(cfg, now=0.0)
        assert run_dir.exists()
        assert run_dir.parent == tmp_path
        assert run_dir.name.startswith("exp_")

    def test_distinct_timestamps_produce_distinct_dirs(self, train_mod, tmp_path: Path) -> None:
        from dataclasses import replace

        from catan_rl.ppo.arguments import TrainConfig

        cfg = replace(TrainConfig.default(), output_dir=str(tmp_path), run_name="exp")
        dir_a = train_mod.build_run_directory(cfg, now=0.0)
        dir_b = train_mod.build_run_directory(cfg, now=3600.0)
        assert dir_a != dir_b


class TestSnapshotConfig:
    def test_snapshot_round_trips(self, train_mod, tmp_path: Path) -> None:
        from catan_rl.ppo.arguments import TrainConfig

        cfg = TrainConfig.default()
        path = train_mod.snapshot_config(cfg, tmp_path)
        assert path.exists()
        assert path.name == "config.yaml"
        reloaded = TrainConfig.from_yaml(path)
        assert reloaded == cfg


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


class TestMain:
    def test_dry_run_exits_zero(self, train_mod, tmp_path: Path) -> None:
        rc = train_mod.main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        assert rc == 0

    def test_no_dry_run_returns_non_retriable_exit_code(self, train_mod, tmp_path: Path) -> None:
        # Trainer raises NotImplementedError until Phase 4; main() catches
        # and returns EXIT_TRAINER_NOT_WIRED = 64 (EX_USAGE), the
        # non-retriable exit code so SkyPilot / Modal don't infinite-respawn.
        rc = train_mod.main(
            [
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        assert rc == train_mod.EXIT_TRAINER_NOT_WIRED
        assert rc == 64

    def test_dry_run_writes_config_snapshot(self, train_mod, tmp_path: Path) -> None:
        rc = train_mod.main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        assert rc == 0
        # The run dir is timestamped; find it.
        run_dirs = sorted(tmp_path.iterdir())
        assert len(run_dirs) == 1
        snapshot = run_dirs[0] / "config.yaml"
        assert snapshot.exists()

    def test_dry_run_writes_train_log(self, train_mod, tmp_path: Path) -> None:
        train_mod.main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        run_dirs = sorted(tmp_path.iterdir())
        log_path = run_dirs[0] / "train.log"
        assert log_path.exists()
        contents = log_path.read_text()
        assert "config snapshot" in contents
        assert "rollout shape" in contents

    def test_invalid_total_steps_via_cli_raises(self, train_mod, tmp_path: Path) -> None:
        # Operator typos a non-multiple total_steps — must fail BEFORE
        # the trainer is invoked.
        with pytest.raises(ValueError, match=r"total_steps.*multiple"):
            train_mod.main(
                [
                    "--total-steps",
                    "50000000",
                    "--output-dir",
                    str(tmp_path),
                    "--device",
                    "cpu",
                ]
            )

    def test_dry_run_writes_run_metadata(self, train_mod, tmp_path: Path) -> None:
        train_mod.main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        run_dir = next(tmp_path.iterdir())
        meta_path = run_dir / "run_metadata.yaml"
        assert meta_path.exists()
        meta = yaml.safe_load(meta_path.read_text())
        assert "git" in meta
        assert set(meta["git"].keys()) == {"sha", "branch", "dirty"}
        assert meta["resolved_device"] == "cpu"
        assert isinstance(meta["launch_cmd"], list)
        assert "--dry-run" in meta["launch_cmd"]
        assert "hostname" in meta
        assert "start_utc" in meta


# ---------------------------------------------------------------------------
# Path normalization (reviewer fix: tilde + relative CLI paths)
# ---------------------------------------------------------------------------


class TestPathNormalization:
    def test_tilde_output_dir_expanded(
        self, train_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point $HOME at tmp_path so we can verify the literal "~" is NOT
        # used as a directory name.
        monkeypatch.setenv("HOME", str(tmp_path))
        train_mod.main(
            [
                "--dry-run",
                "--output-dir",
                "~/p2_tilde",
                "--device",
                "cpu",
            ]
        )
        # Real expansion: tmp_path/p2_tilde/<timestamp>/...
        expanded = tmp_path / "p2_tilde"
        assert expanded.exists(), (
            f"expected {expanded} to be created; got contents={list(tmp_path.iterdir())}"
        )
        # No literal "~" directory anywhere.
        assert not (Path.cwd() / "~").exists()

    def test_relative_config_path_resolved_against_cwd(
        self, train_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Write a relative-path config, then chdir into tmp_path and
        # invoke with the relative name. The script must resolve it.
        (tmp_path / "my.yaml").write_text(yaml.safe_dump({"rollout": {"n_envs": 32}}))
        monkeypatch.chdir(tmp_path)
        args = train_mod.build_parser().parse_args(
            [
                "--config",
                "my.yaml",
                "--n-envs",
                "64",
            ]
        )
        cfg = train_mod.resolve_config(args)
        # CLI flag wins; verify the resolve still worked
        assert cfg.rollout.n_envs == 64

    def test_missing_config_path_raises_clear_error(self, train_mod, tmp_path: Path) -> None:
        args = train_mod.build_parser().parse_args(
            [
                "--config",
                str(tmp_path / "nonexistent.yaml"),
            ]
        )
        with pytest.raises(FileNotFoundError, match=r"--config path"):
            train_mod.resolve_config(args)


# ---------------------------------------------------------------------------
# Run-dir collision detection (reviewer fix)
# ---------------------------------------------------------------------------


class TestRunDirCollision:
    def test_collision_at_same_timestamp_yields_suffix(self, train_mod, tmp_path: Path) -> None:
        from dataclasses import replace

        from catan_rl.ppo.arguments import TrainConfig

        cfg = replace(TrainConfig.default(), output_dir=str(tmp_path), run_name="exp")
        # First call: creates exp_<stamp>
        dir_a = train_mod.build_run_directory(cfg, now=0.0)
        # Second call at the SAME timestamp: must NOT merge — appends _2.
        dir_b = train_mod.build_run_directory(cfg, now=0.0)
        dir_c = train_mod.build_run_directory(cfg, now=0.0)
        assert dir_a != dir_b != dir_c
        # The 2nd & 3rd land at <name>_2 / <name>_3
        assert dir_b.name.endswith("_2")
        assert dir_c.name.endswith("_3")


# ---------------------------------------------------------------------------
# Logger handler ownership (reviewer fix)
# ---------------------------------------------------------------------------


class TestLoggerOwnership:
    def test_setup_logging_does_not_remove_foreign_handlers(
        self, train_mod, tmp_path: Path
    ) -> None:
        import logging

        # Install a foreign handler BEFORE main() runs.
        foreign = logging.NullHandler()
        foreign.name = "foreign_supervisor"
        logging.getLogger("catan_rl.train").addHandler(foreign)
        try:
            train_mod.main(
                [
                    "--dry-run",
                    "--output-dir",
                    str(tmp_path),
                    "--device",
                    "cpu",
                ]
            )
            handlers = logging.getLogger("catan_rl.train").handlers
            # Foreign handler survives; our owned handlers are also present.
            assert foreign in handlers, "foreign handler was incorrectly removed"
        finally:
            logging.getLogger("catan_rl.train").removeHandler(foreign)

    def test_setup_logging_closes_prior_owned_file_handlers(
        self, train_mod, tmp_path: Path
    ) -> None:
        # Two main() calls in the same process: prior file handlers must
        # be closed (no FD leak under Modal-style respawn supervisors).
        for _ in range(3):
            train_mod.main(
                [
                    "--dry-run",
                    "--output-dir",
                    str(tmp_path),
                    "--device",
                    "cpu",
                ]
            )
        import logging

        owned = [
            h
            for h in logging.getLogger("catan_rl.train").handlers
            if h.name == "catan_rl.train.entry"
        ]
        # After the last main(), only the latest pair survives (1 stream + 1 file).
        assert len(owned) == 2
