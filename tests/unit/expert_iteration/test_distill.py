"""Warm-started distillation (T007): produces a v2-lineage-loadable checkpoint."""

from __future__ import annotations

from pathlib import Path

from catan_rl.checkpoint.manager import load_checkpoint
from catan_rl.expert_iteration.config import DistillConfig
from catan_rl.expert_iteration.distill import distill


def test_distill_produces_v2_lineage_loadable_ckpt(
    tiny_ckpt: Path, tiny_labels: Path, tmp_path: Path
) -> None:
    cfg = DistillConfig(
        data_dir=str(tiny_labels),
        out_dir=str(tmp_path / "distill"),
        init_ckpt=str(tiny_ckpt),
        peak_lr=1e-4,
        max_epochs=1,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    distilled = distill(cfg)
    assert distilled.exists()

    # FR-004: loads via the existing checkpoint manager (the distilled ckpt is
    # re-saved schema'd, unlike train_bc's bare best.pt).
    payload = load_checkpoint(distilled)
    assert payload.policy_state_dict

    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    payload.apply_to_policy(policy, strict=True)  # must not raise (shape-compatible, v2-lineage)
