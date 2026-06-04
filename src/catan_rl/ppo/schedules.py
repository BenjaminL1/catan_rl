"""Pure scheduling helpers for the PPO trainer.

Kept separate from :class:`catan_rl.ppo.trainer.PPOTrainer` so the math
is unit-testable and the trainer just calls them by `update_idx`.

* :func:`linear_lr_schedule` — linear from ``lr_start`` to ``lr_end`` over
  the documented total-update count. After the end, holds at ``lr_end``.
* :func:`linear_entropy_coef_schedule` — linear anneal from
  ``coef_start`` to ``coef_end`` between two update indices.
"""

from __future__ import annotations


def linear_lr_schedule(
    *,
    update_idx: int,
    lr_start: float,
    lr_end: float,
    total_updates: int,
) -> float:
    """Return the LR for the given ``update_idx``.

    Linear interpolation from ``lr_start`` at update 0 to ``lr_end`` at
    update ``total_updates - 1``. Holds at ``lr_end`` past the boundary.

    Args:
        update_idx: 0-indexed update counter.
        lr_start: LR at update 0.
        lr_end: LR at update ``total_updates - 1`` and beyond.
        total_updates: number of updates over which to anneal. Must be
            positive. Set to 1 to disable annealing (LR stays at
            ``lr_start``).
    """
    if total_updates <= 0:
        raise ValueError(f"total_updates must be > 0, got {total_updates}")
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    if total_updates == 1:
        return lr_start
    progress = min(1.0, update_idx / max(1, total_updates - 1))
    return lr_start + (lr_end - lr_start) * progress


def linear_entropy_coef_schedule(
    *,
    update_idx: int,
    coef_start: float,
    coef_end: float,
    start_update: int,
    end_update: int,
) -> float:
    """Linear-anneal entropy coef from ``coef_start`` (at ``start_update``)
    to ``coef_end`` (at ``end_update`` and beyond).

    Holds at ``coef_start`` before ``start_update``. This mirrors the
    Charlesworth recipe — burn-in at high entropy, anneal to a low
    floor after the policy starts converging.
    """
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    if end_update < start_update:
        raise ValueError(f"end_update ({end_update}) must be >= start_update ({start_update})")
    if update_idx <= start_update:
        return coef_start
    if update_idx >= end_update:
        return coef_end
    span = end_update - start_update
    if span == 0:
        return coef_end
    progress = (update_idx - start_update) / span
    return coef_start + (coef_end - coef_start) * progress
