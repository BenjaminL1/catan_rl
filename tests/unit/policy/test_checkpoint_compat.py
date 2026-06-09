"""T028 — checkpoint back-compat (Constitution III / FR-008).

The self-play feature introduced NO obs/action-head shape change — it feeds the
existing ``_OppIdEmbedding`` real values without resizing it. So every v2
checkpoint, including the in-flight ``bootstrap_v1`` self-play seed, must still
load strictly into a current ``CatanPolicy`` (no shape mismatch).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catan_rl.checkpoint.manager import load_checkpoint, save_checkpoint
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy


def _policy() -> CatanPolicy:
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    return policy


def test_current_policy_roundtrips_strict(tmp_path: Path) -> None:
    src = _policy()
    save_checkpoint(
        tmp_path / "c.pt",
        config={},
        policy=src,
        optimizer=None,
        update_idx=0,
        global_step=0,
        capture_rng=False,
    )
    fresh = _policy()
    # strict=True raises on any missing/unexpected/mismatched parameter.
    load_checkpoint(tmp_path / "c.pt").apply_to_policy(fresh, strict=True)


def test_bootstrap_v1_seed_still_loads() -> None:
    ckpt = Path("runs/train/bootstrap_v1_20260607_233931/checkpoints/ckpt_000000799.pt")
    if not ckpt.exists():
        pytest.skip(f"bootstrap_v1 u799 checkpoint absent at {ckpt}")
    load_checkpoint(ckpt).apply_to_policy(_policy(), strict=True)
