"""Tests for `scripts/bc_to_checkpoint.py` — the BC -> schema'd bridge.

Pins the BC->bootstrap handoff that `init_policy_checkpoint` / `build_actor`
depend on:

1. A bare BC payload (no `schema_version`) is NOT loadable via `load_checkpoint`
   (the documented BLOCKER this converter exists to fix).
2. Round-trip: bare BC save -> convert -> `load_checkpoint` ->
   `apply_to_policy(strict=True)` succeeds and the weights are byte-equal.
3. The bridged file is a schema'd, policy-only checkpoint.
4. `main()` defaults `--out` to a sibling `*_ckpt.pt`.
5. Missing input / already-schema'd input exit 1.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from catan_rl.checkpoint import SCHEMA_VERSION, CheckpointError, load_checkpoint
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy

# Import the CLI module directly (mirrors tests/unit/checkpoint/test_migrate_cli.py)
# so we can call `convert_bc_checkpoint` / `main` without spawning a subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
bc_to_checkpoint = importlib.import_module("bc_to_checkpoint")


def _fresh_policy() -> CatanPolicy:
    p = CatanPolicy()
    p.set_board_geometry(build_geometry().as_dict_of_tensors())
    return p


def _write_bare_bc(
    path: Path, policy: CatanPolicy, *, step: int = 123, val_nll: float = 3.2
) -> None:
    """Write a payload in the exact shape `bc.train._save_checkpoint` emits."""
    torch.save(
        {"policy_state_dict": policy.state_dict(), "step": step, "val_nll": val_nll},
        path,
    )


class TestBridge:
    def test_bare_bc_is_not_loadable_then_bridge_fixes_it(self, tmp_path: Path) -> None:
        p0 = _fresh_policy()
        bc_path = tmp_path / "best.pt"
        _write_bare_bc(bc_path, p0, step=9500, val_nll=3.2046)

        # The BLOCKER: the bare BC file is rejected by the canonical loader.
        with pytest.raises(CheckpointError):
            load_checkpoint(bc_path)

        out = tmp_path / "best_ckpt.pt"
        written = bc_to_checkpoint.convert_bc_checkpoint(bc_path, out)
        assert written == out
        assert out.exists()

        # Now loadable via the standard path, as a policy-only schema'd checkpoint.
        payload = load_checkpoint(out)
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.metadata.get("kind") == "policy_only"
        assert payload.global_step == 9500  # carried from BC "step"

        # apply_to_policy(strict=True) must succeed against a fresh pointer-arch policy.
        p1 = _fresh_policy()
        payload.apply_to_policy(p1, strict=True)

        # Byte-equal to the original BC weights.
        sd0, sd1 = p0.state_dict(), p1.state_dict()
        assert sd0.keys() == sd1.keys()
        for k in sd0:
            assert torch.equal(sd0[k].cpu(), sd1[k].cpu()), f"weight drift at {k}"

    def test_main_defaults_out_to_sibling(self, tmp_path: Path) -> None:
        p0 = _fresh_policy()
        bc_path = tmp_path / "best.pt"
        _write_bare_bc(bc_path, p0)

        rc = bc_to_checkpoint.main(["--in", str(bc_path)])
        assert rc == 0

        bridged = tmp_path / "best_ckpt.pt"  # default_out_path(best.pt)
        assert bridged.exists()
        # Sanity: the default-out file loads via the canonical path.
        load_checkpoint(bridged).apply_to_policy(_fresh_policy(), strict=True)

    def test_missing_input_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = bc_to_checkpoint.main(["--in", str(tmp_path / "nope.pt")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_already_schemad_input_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A schema'd checkpoint is not a BC bridge input — reject rather than
        # double-wrap it.
        src = tmp_path / "already.pt"
        torch.save({"schema_version": SCHEMA_VERSION, "policy_state_dict": {}}, src)
        rc = bc_to_checkpoint.main(["--in", str(src), "--out", str(tmp_path / "o.pt")])
        assert rc == 1
        assert "schema_version" in capsys.readouterr().err
