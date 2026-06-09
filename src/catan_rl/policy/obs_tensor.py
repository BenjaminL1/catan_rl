"""Shared obs/mask → torch conversion.

One definition used by the rollout collector (``game_manager``, already batched
across envs) and the in-env snapshot-opponent driver (single env, needs a batch
dim). Keeping it in one place avoids a silent train/opponent skew if a dtype or
device fix ever lands in only one copy (senior-SWE review, Phase 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def obs_to_torch(
    obs: dict[str, np.ndarray], device: torch.device, *, add_batch: bool = False
) -> dict[str, torch.Tensor]:
    if add_batch:
        return {k: torch.as_tensor(np.expand_dims(v, 0), device=device) for k, v in obs.items()}
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def masks_to_torch(
    masks: dict[str, Any], device: torch.device, *, add_batch: bool = False
) -> dict[str, torch.Tensor]:
    if add_batch:
        return {
            k: torch.as_tensor(np.expand_dims(v, 0), device=device, dtype=torch.bool)
            for k, v in masks.items()
        }
    return {k: torch.as_tensor(v, device=device, dtype=torch.bool) for k, v in masks.items()}
