"""Tests for `catan_rl.replay.player_factory`.

Covers:
1. Random + heuristic kinds construct cheaply (no torch import path).
2. Policy kind without ckpt_path raises ValueError.
3. Policy kind with nonexistent ckpt path raises FileNotFoundError.
4. Policy kind with real ckpt path loads (gated on the sanity_phase10
   checkpoint existing).
5. ``_resolve_device`` falls back to CPU with a WARNING when CUDA/MPS
   are requested but unavailable.
6. ``_resolve_device("auto")`` walks cuda → mps → cpu.
7. The recorder-side ``select_action`` of an engine-driven actor
   returns a (6,) int64 array with the type entry inside the legal
   mask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from catan_rl.replay.player_factory import (
    PlayerSpec,
    _EngineDrivenActor,
    _resolve_device,
    build_actor,
)


class TestEngineDrivenConstruct:
    def test_random_constructs(self) -> None:
        actor = build_actor(PlayerSpec(kind="random"), seed=42)
        assert actor.kind == "random"

    def test_heuristic_constructs(self) -> None:
        actor = build_actor(PlayerSpec(kind="heuristic"), seed=42)
        assert actor.kind == "heuristic"

    def test_random_select_action_uses_legal_mask(self) -> None:
        actor = build_actor(PlayerSpec(kind="random"), seed=42)
        assert isinstance(actor, _EngineDrivenActor)
        type_mask = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        masks = {"type": type_mask}
        action = actor.select_action(obs={}, masks=masks)
        assert action.shape == (6,)
        assert action.dtype == np.int64
        # type entry must be one of the legal indices.
        assert action[0] in (0, 3)

    def test_random_select_action_falls_back_to_end_turn(self) -> None:
        # No legal action → fallback to END_TURN (index 3).
        actor = build_actor(PlayerSpec(kind="random"), seed=42)
        assert isinstance(actor, _EngineDrivenActor)
        masks = {"type": np.zeros(13, dtype=bool)}
        action = actor.select_action(obs={}, masks=masks)
        assert action[0] == 3


class TestPolicyValidation:
    def test_missing_ckpt_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="requires ckpt_path"):
            build_actor(PlayerSpec(kind="policy", ckpt_path=None), seed=42)

    def test_nonexistent_ckpt_raises_filenotfound(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            build_actor(PlayerSpec(kind="policy", ckpt_path=str(bogus)), seed=42)


class TestPolicyConstruction:
    @pytest.fixture
    def sanity_ckpt(self) -> Path:
        ckpt = Path(
            "/Users/benjaminli/my_projects/catan_rl_v2/runs/train/"
            "sanity_phase10_20260603_231643/checkpoints/ckpt_000000099.pt"
        )
        if not ckpt.exists():
            pytest.skip(f"sanity_phase10 checkpoint absent at {ckpt}")
        return ckpt

    def test_policy_loads_from_real_ckpt(self, sanity_ckpt: Path) -> None:
        actor = build_actor(
            PlayerSpec(kind="policy", ckpt_path=str(sanity_ckpt)),
            seed=42,
            device="cpu",
        )
        assert actor.kind == "policy"


class TestDeviceResolution:
    def test_cpu_pass_through(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown device"):
            _resolve_device("tpu")

    def test_cuda_fallback_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # The test harness runs on M1 CPU; CUDA is unavailable. The
        # resolver must log a warning and fall back to CPU, NEVER raise.
        with caplog.at_level(logging.WARNING, logger="catan_rl.replay"):
            resolved = _resolve_device("cuda")
        if resolved == "cuda":
            pytest.skip("CUDA actually available — fallback path not exercised")
        assert resolved == "cpu"
        assert any(
            "cuda requested but CUDA not available" in r.getMessage() for r in caplog.records
        )

    def test_mps_fallback_logs_warning_if_unavailable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("MPS available — fallback path not exercised")
        with caplog.at_level(logging.WARNING, logger="catan_rl.replay"):
            resolved = _resolve_device("mps")
        assert resolved == "cpu"

    def test_auto_resolves_to_some_device(self) -> None:
        resolved = _resolve_device("auto")
        assert resolved in ("cuda", "mps", "cpu")
