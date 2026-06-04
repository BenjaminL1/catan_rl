"""Build an :class:`AnchorPolicy` from a Phase 8 checkpoint.

The loader delegates to :func:`catan_rl.checkpoint.load_checkpoint`
for the file-format details (atomic, versioned, RNG capture) and
then wraps the resulting policy in :class:`AnchorPolicy`.

The caller supplies a ``policy_factory`` callable that builds a
fresh, randomly-initialised policy of the right architecture. The
loader applies the saved ``policy_state_dict`` to that policy and
wraps it. This indirection lets piKL anchors be ANY policy shape
(BC pretrain, an earlier PPO snapshot, a hand-shaped baseline) as
long as the factory matches the checkpoint's architecture.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from catan_rl.algorithms.pikl.anchor import AnchorPolicy
from catan_rl.checkpoint import load_checkpoint


def load_pikl_anchor(
    checkpoint_path: str | Path,
    *,
    policy_factory: Callable[[], nn.Module],
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> AnchorPolicy:
    """Load a frozen anchor policy from a Phase 8 checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint written by
            :func:`catan_rl.checkpoint.save_checkpoint`.
        policy_factory: Callable returning a fresh policy module
            whose ``state_dict`` shape matches the saved one.
            Typically ``lambda: CatanPolicy.from_config(cfg)``.
        map_location: Forwarded to ``torch.load`` via the checkpoint
            module. Default ``"cpu"`` matches the save-side CPU
            cloning.
        strict: ``True`` (the default) raises if the factory's
            policy has different keys from the saved state. Set to
            ``False`` only for partial-restore scenarios (e.g.
            loading a pretrained encoder into a larger model).

    Returns:
        :class:`AnchorPolicy` — already frozen + in eval mode.
    """
    payload = load_checkpoint(checkpoint_path, map_location=map_location)
    fresh_policy = policy_factory()
    fresh_policy.load_state_dict(payload.policy_state_dict, strict=strict)
    return AnchorPolicy(fresh_policy)
