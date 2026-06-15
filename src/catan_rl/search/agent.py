"""``SearchAgent`` — the offline search decision surface (contract C3).

``choose_action(env)`` runs determinized PUCT search on the LIVE env (which it
clones internally — it never mutates the passed env) and returns a legal action
6-tuple as an ``np.ndarray``. A forced state (<=1 representative legal action)
short-circuits inside the MCTS without spending the simulation budget.

The opponent model inside the tree is a frozen clone of the wrapped policy
(exact in the bake-off, where the real opponent IS this policy; a reasonable
self-model otherwise). RNG hygiene: the global torch/numpy/stdlib streams are
snapshotted and restored around each search, so it NEVER perturbs the surrounding
game's RNG (FR-006). Determinism + per-simulation determinization seeding lives in
``MCTS`` (a search is a reproducible function of ``(cfg.seed, env state)``); the
agent only owns the outer save/restore.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from catan_rl.search.mcts import MCTS

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.search.config import SearchConfig


def _snapshot_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "np": np.random.get_state(),
        "py": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["torch"])
    np.random.set_state(state["np"])
    random.setstate(state["py"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


class SearchAgent:
    """Wraps a frozen ``CatanPolicy`` with determinized PUCT lookahead."""

    def __init__(
        self,
        policy: CatanPolicy,
        cfg: SearchConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

        if device is None:
            device = next(policy.parameters()).device
        self.policy = policy
        self.cfg = cfg
        self.device = device
        # Opponent model = a frozen clone of the wrapped policy (shared net,
        # isolated RNG stream). Reset per search for reproducibility.
        self.opponent = FrozenSnapshotOpponent(policy, device=device, seed=cfg.seed)
        self._mcts = MCTS(policy, cfg, self.opponent, device)
        self.last_diagnostics: dict[str, Any] = {}

    def choose_action(self, env: CatanEnv) -> np.ndarray:
        # Snapshot/restore the global RNG so the search never perturbs the live
        # game's stream; MCTS reseeds per-simulation internally for determinism.
        rng = _snapshot_rng()
        try:
            action, diagnostics = self._mcts.run(env)
        finally:
            _restore_rng(rng)
        self.last_diagnostics = diagnostics
        return np.asarray(action, dtype=np.int64)
