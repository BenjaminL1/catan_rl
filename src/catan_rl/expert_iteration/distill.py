"""Distillation = a warm-started BC fine-tune on search-labeled shards (contract C2).

Runs the existing BC trainer warm-started from the round base (v6) on the
search-labeled dataset, then RE-SAVES the trained policy as a proper schema'd
v2-lineage checkpoint (``train_bc`` writes a bare ``{policy_state_dict}`` best.pt
that ``load_checkpoint`` can't read — it requires ``schema_version``). The returned
``distilled.pt`` loads via the existing checkpoint manager / ``build_actor`` (FR-004).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from catan_rl.expert_iteration.config import DistillConfig


def distill(cfg: DistillConfig) -> Path:
    """Fine-tune the base policy toward the search labels; return the distilled ckpt."""
    import torch

    from catan_rl.bc.train import train_bc
    from catan_rl.checkpoint.manager import save_checkpoint
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    out_dir = Path(cfg.out_dir)
    train_bc(
        data_dir=Path(cfg.data_dir),
        out_dir=out_dir,
        init_ckpt=Path(cfg.init_ckpt),  # warm-start from the round base
        peak_lr=cfg.peak_lr,
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        value_weight=cfg.value_weight,
        belief_weight=cfg.belief_weight,
        seed=cfg.seed,
        device=cfg.device,
    )

    # train_bc writes best.pt (best val) + last.pt (always). Prefer best.
    best = out_dir / "best.pt"
    src = best if best.exists() else out_dir / "last.pt"
    raw: dict[str, Any] = torch.load(src, map_location="cpu", weights_only=False)

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.load_state_dict(raw["policy_state_dict"], strict=True)

    distilled = out_dir / "distilled.pt"
    save_checkpoint(
        distilled,
        config={"source": "exit-distill", "base_ckpt": str(cfg.init_ckpt)},
        policy=policy,
        optimizer=None,
        update_idx=0,
        global_step=int(raw.get("step", 0)),
        capture_rng=False,
        metadata={"lineage": "v6->exit-distill", "data_dir": str(cfg.data_dir), "bc_src": src.name},
    )
    return distilled
