"""Tests for ``scripts/bench_engine.py`` — the Phase 1 throughput
benchmark harness.

Pins:

1. Importing the script does not invoke ``main()``.
2. ``_stack_rust_obs`` correctly squeezes the ``(1,)``-shaped Rust
   ``opponent_kind`` / ``opponent_policy_id`` arrays into ``(N,)``.
3. ``_run_py_backend`` returns a positive wall-time and exercises
   the policy forward path when ``include_policy=True``.
4. ``_run_rust_no_op_backend`` returns a positive wall-time when
   the Rust extension is built.
5. ``_run_rust_with_opp_backend`` raises ``NotImplementedError``
   with a message that cites Phase 5 of the remediation plan.
6. CSV row schema is stable and includes the ``include_policy``
   column (regression guard against the architect's #2 must-do).
7. ``main(--all)`` writes a CSV and a JSON manifest with the
   expected fields.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Load ``scripts/bench_engine.py`` as a module without invoking main.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bench_engine.py"
_spec = importlib.util.spec_from_file_location("bench_engine_module", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bench_engine = importlib.util.module_from_spec(_spec)
sys.modules["bench_engine_module"] = bench_engine
_spec.loader.exec_module(bench_engine)


class TestObsStacking:
    def test_stack_rust_obs_squeezes_single_element_arrays(self) -> None:
        # Rust returns opponent_kind / opponent_policy_id as (1,)
        # arrays; the stacker must squeeze them into (N,) so the
        # frozen policy's embedding sees a 1-D index vector.
        obs_list = [
            {
                "current_player_main": np.zeros(54, dtype=np.float32),
                "opponent_kind": np.array([0], dtype=np.int64),
                "opponent_policy_id": np.array([5], dtype=np.int64),
            },
            {
                "current_player_main": np.ones(54, dtype=np.float32),
                "opponent_kind": np.array([1], dtype=np.int64),
                "opponent_policy_id": np.array([6], dtype=np.int64),
            },
        ]
        stacked = bench_engine._stack_rust_obs(obs_list)
        assert stacked["current_player_main"].shape == (2, 54)
        assert stacked["opponent_kind"].shape == (2,)
        assert stacked["opponent_policy_id"].shape == (2,)
        assert stacked["opponent_kind"].tolist() == [0, 1]
        assert stacked["opponent_policy_id"].tolist() == [5, 6]

    def test_stack_rust_obs_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty obs_list"):
            bench_engine._stack_rust_obs([])

    def test_stack_rust_obs_only_squeezes_allow_listed_scalar_keys(self) -> None:
        # Architect non-blocking #7: the squeeze is allow-listed to
        # ``opponent_kind`` / ``opponent_policy_id``. A future Rust
        # change that adds a different ``(1,)``-shaped field must
        # NOT be silently squeezed — it stacks as ``(N, 1)`` and
        # the consumer can decide what to do with it.
        obs_list = [
            {
                "opponent_kind": np.array([0], dtype=np.int64),
                "future_one_dim_field": np.array([0.5], dtype=np.float32),
            },
            {
                "opponent_kind": np.array([1], dtype=np.int64),
                "future_one_dim_field": np.array([0.25], dtype=np.float32),
            },
        ]
        stacked = bench_engine._stack_rust_obs(obs_list)
        assert stacked["opponent_kind"].shape == (2,)
        # Non-allow-listed (1,) field keeps its inner dim.
        assert stacked["future_one_dim_field"].shape == (2, 1)


class TestRustWithOppPlaceholder:
    def test_raises_with_phase5_pointer(self) -> None:
        # Architect must-do #6: the placeholder must point at the
        # phase that lands the opponent injection contract so
        # future readers know where the gap is closed.
        with pytest.raises(NotImplementedError, match="Phase 5"):
            bench_engine._run_rust_with_opp_backend(
                n_envs=2, n_steps=4, seed=0, policy=None, include_policy=True
            )

    def test_raises_with_actual_state_doc_pointer(self) -> None:
        with pytest.raises(NotImplementedError, match="rust_engine_actual_state"):
            bench_engine._run_rust_with_opp_backend(
                n_envs=2, n_steps=4, seed=0, policy=None, include_policy=True
            )


class TestPyBackend:
    def test_returns_positive_wall_time(self) -> None:
        policy = bench_engine._build_frozen_policy()
        wall_s = bench_engine._run_py_backend(
            n_envs=2, n_steps=2, seed=42, policy=policy, include_policy=True
        )
        assert wall_s > 0.0
        # Sanity: shouldn't take more than a few seconds at n_envs=2, n_steps=2.
        assert wall_s < 30.0

    def test_exclude_policy_completes(self) -> None:
        # The exclude-policy branch must also work end-to-end so
        # the env-only sanity comparison number is producible.
        policy = bench_engine._build_frozen_policy()
        wall_s = bench_engine._run_py_backend(
            n_envs=2, n_steps=2, seed=42, policy=policy, include_policy=False
        )
        assert wall_s > 0.0


class TestRustNoOpBackend:
    def test_returns_positive_wall_time(self) -> None:
        pytest.importorskip("catan_engine")
        policy = bench_engine._build_frozen_policy()
        wall_s = bench_engine._run_rust_no_op_backend(
            n_envs=2, n_steps=2, seed=42, policy=policy, include_policy=True
        )
        assert wall_s > 0.0
        assert wall_s < 10.0


class TestCsvSchema:
    def test_bench_row_has_include_policy_column(self) -> None:
        # Architect's #2 must-do: the CSV schema must distinguish
        # rows with/without the policy in the loop so a reader can
        # tell which numbers are comparable. This is a regression
        # guard against silent removal of the column.
        from dataclasses import fields

        row_fields = {f.name for f in fields(bench_engine.BenchRow)}
        assert "include_policy" in row_fields
        assert "env_steps_per_sec" in row_fields
        assert "wall_s" in row_fields
        assert "backend" in row_fields
        assert "n_envs" in row_fields


class TestBackendsContractPin:
    def test_backends_enum_matches_driver_table(self) -> None:
        # If someone adds a 4th backend (Phase 5 will likely
        # promote ``rust_with_opp`` into a real driver) and forgets
        # to wire it into ``_BACKEND_DRIVERS``, the bench will
        # silently ``KeyError`` at runtime. This test forces the
        # two sources of truth to stay in sync.
        assert set(bench_engine.BACKENDS) == set(bench_engine._BACKEND_DRIVERS.keys())


class TestMainWritesArtifacts:
    def test_main_all_writes_csv_and_manifest(self, tmp_path: Path) -> None:
        # Drive ``main`` end-to-end at a tiny grid so the test
        # finishes in a few seconds. Uses ``--exclude-policy`` so the
        # CPU cost stays trivial; the include_policy path is exercised
        # by ``TestPyBackend.test_returns_positive_wall_time``.
        out_dir = tmp_path / "results"
        rc = bench_engine.main(
            [
                "--backend",
                "py",
                "--n-envs",
                "2",
                "--n-steps",
                "2",
                "--repeat",
                "1",
                "--exclude-policy",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        csv_files = list(out_dir.glob("bench_*.csv"))
        json_files = list(out_dir.glob("bench_*.json"))
        assert len(csv_files) == 1, csv_files
        assert len(json_files) == 1, json_files

        # CSV has a header + at least one row.
        with csv_files[0].open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["backend"] == "py"
        assert rows[0]["n_envs"] == "2"
        assert rows[0]["include_policy"] == "False"
        assert float(rows[0]["wall_s"]) > 0.0

        # Manifest has the expected top-level fields.
        with json_files[0].open() as f:
            manifest = json.load(f)
        assert "git_sha" in manifest
        assert "hardware" in manifest
        assert manifest["include_policy"] is False
        assert "py@n_envs=2" in manifest["cells"]

    def test_main_with_policy_in_loop_writes_artifacts(self, tmp_path: Path) -> None:
        # Architect must-do #5: the include_policy=True end-to-end
        # path must be test-covered, otherwise the architect's #2
        # must-do (policy-in-loop in the bench) is only half-guarded.
        # Tiny grid keeps it fast — n_envs=2, n_steps=2, repeat=1.
        out_dir = tmp_path / "results"
        rc = bench_engine.main(
            [
                "--backend",
                "py",
                "--n-envs",
                "2",
                "--n-steps",
                "2",
                "--repeat",
                "1",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        csv_files = list(out_dir.glob("bench_*.csv"))
        with csv_files[0].open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        # The headline guard: the row must record include_policy=True.
        assert rows[0]["include_policy"] == "True"
        assert float(rows[0]["wall_s"]) > 0.0
        with csv_files[0].with_suffix(".json").open() as f:
            manifest = json.load(f)
        assert manifest["include_policy"] is True
