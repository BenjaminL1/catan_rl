"""Frozen-policy snapshot opponent (T008).

Loads a league snapshot's state-dict into an inference-only ``CatanPolicy`` and
samples actions for the opponent seat during self-play. Two hard invariants:

* **Inference-only (D6).** The policy is ``eval()``, every parameter has
  ``requires_grad_(False)``, and sampling runs under ``no_grad`` — a snapshot is
  NEVER trained and never enters the optimizer.
* **RNG isolation (FR-006).** The opponent's stochastic draw is seeded
  deterministically per call and the global RNG state is snapshotted/restored
  around it, so the opponent's randomness is reproducible AND does not advance
  the learner's rollout RNG stream. We restore state rather than thread a
  ``torch.Generator`` through the heads because the heads' ``Categorical.sample``
  is the *shared learner* path — it must stay untouched (no checkpoint risk).

The ``sample`` surface mirrors ``CatanPolicy.sample`` exactly (batched obs/mask
tensors in), so the caller (the env turn-driver / game manager) can batch the
opponent forward across envs without this helper caring about batching.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

from catan_rl.policy.network import CatanPolicy

__all__ = [
    "FrozenSnapshotOpponent",
    "SnapshotOpponent",
    "build_snapshot_opponent",
    "load_frozen_policy",
]


@runtime_checkable
class SnapshotOpponent(Protocol):
    """Structural interface the env's turn-driver depends on (typing seam).

    ``FrozenSnapshotOpponent`` satisfies this, as do the test stubs — so the
    driver type-checks against the surface (``device`` / ``reset_rng`` /
    ``sample``) instead of ``Any``.
    """

    @property
    def device(self) -> torch.device: ...

    def reset_rng(self, seed: int | None = None) -> None: ...

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> torch.Tensor: ...


class FrozenSnapshotOpponent:
    """An inference-only frozen policy that samples opponent actions."""

    def __init__(self, policy: CatanPolicy, *, device: torch.device, seed: int) -> None:
        self._policy = policy.to(device)
        self._policy.eval()
        for param in self._policy.parameters():
            param.requires_grad_(False)
        self._device = device
        self._seed = int(seed)
        self._call_count = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def policy(self) -> CatanPolicy:
        return self._policy

    def reset_rng(self, seed: int | None = None) -> None:
        """Restart the opponent's deterministic action stream (e.g. per game)."""
        if seed is not None:
            self._seed = int(seed)
        self._call_count = 0

    def _snapshot_rng(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu_state = torch.random.get_rng_state()
        dev_state: torch.Tensor | None = None
        if self._device.type == "cuda":
            dev_state = torch.cuda.get_rng_state(self._device)
        elif self._device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            dev_state = torch.mps.get_rng_state()
        return cpu_state, dev_state

    def _restore_rng(self, state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        cpu_state, dev_state = state
        torch.random.set_rng_state(cpu_state)
        if dev_state is not None:
            if self._device.type == "cuda":
                torch.cuda.set_rng_state(dev_state, self._device)
            elif self._device.type == "mps":
                torch.mps.set_rng_state(dev_state)

    @torch.no_grad()
    def sample(self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample a (batched) action tensor; learner RNG is left unperturbed."""
        rng_state = self._snapshot_rng()
        try:
            # Deterministic per-call seed -> reproducible opponent sequence.
            torch.manual_seed(self._seed + self._call_count)
            out = self._policy.sample(obs, masks)
        finally:
            self._restore_rng(rng_state)
        self._call_count += 1
        return out["action"]


def load_frozen_policy(state_dict: Mapping[str, Any], *, geometry: dict[str, Any]) -> CatanPolicy:
    """Load a snapshot's state-dict into a fresh ``CatanPolicy`` (no wrapper).

    Mirrors ``replay/player_factory.build_actor``'s load order: construct a
    ``CatanPolicy``, wire board geometry, then load the snapshot strictly
    (raises on any shape mismatch — guards Constitution III back-compat). The
    returned policy is STATELESS (weights only) and so is safe to SHARE across
    many per-env ``FrozenSnapshotOpponent`` wrappers — the RNG-bearing state
    lives in the wrapper, never the policy.
    """
    policy = CatanPolicy()
    policy.set_board_geometry(geometry)
    policy.load_state_dict(dict(state_dict), strict=True)
    return policy


def build_snapshot_opponent(
    state_dict: Mapping[str, Any],
    *,
    geometry: dict[str, Any],
    device: torch.device,
    seed: int,
) -> FrozenSnapshotOpponent:
    """Load a snapshot and wrap it in a single frozen opponent (one env)."""
    return FrozenSnapshotOpponent(
        load_frozen_policy(state_dict, geometry=geometry), device=device, seed=seed
    )
