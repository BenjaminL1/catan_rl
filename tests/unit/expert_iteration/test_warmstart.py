"""train_bc warm-start (T002/T003): init_ckpt loads weights before training.

Lives here (not tests/unit/bc/) to reuse the tiny v2-ckpt + tiny-labels fixtures; it
exercises the `train_bc(init_ckpt=...)` path that expert-iteration distillation needs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from catan_rl.bc.train import train_bc
from catan_rl.checkpoint.manager import load_checkpoint


def test_train_bc_init_ckpt_warm_starts(tiny_ckpt: Path, tiny_labels: Path, tmp_path: Path) -> None:
    init_sd = load_checkpoint(tiny_ckpt).policy_state_dict
    # a trunk weight that should be loaded from the init, not randomly re-initialised
    key = next(k for k in init_sd if k.endswith(".weight") and init_sd[k].ndim >= 2)

    out = tmp_path / "warm"
    # peak_lr ~ 0 + warmup so the params barely move; the trained net then equals
    # the init iff init_ckpt was actually loaded (a fresh random init would differ).
    train_bc(
        data_dir=tiny_labels,
        out_dir=out,
        init_ckpt=tiny_ckpt,
        max_epochs=1,
        batch_size=8,
        peak_lr=1e-9,
        warmup_steps=1,
        val_pct=0.1,
        seed=0,
        device="cpu",
    )
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(
        last["policy_state_dict"][key].float(), init_sd[key].float(), atol=1e-4
    ), "warm-start did not load the init checkpoint's weights"
