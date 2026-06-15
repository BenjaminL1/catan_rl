"""Minimal determinized PUCT-MCTS (T009).

A max-only tree over agent decision points (the env folds the opponent's turn +
dice into the agent's ``EndTurn`` transition). One simulation = descend by PUCT
on the SQUASHED leaf value, expand exactly one new child (clone the parent's env
and ``step`` the action — the clone faithfully + independently continues the
parent's dice future, so the line's stochasticity is fixed), evaluate the new
leaf with the value head (or the true outcome if terminal), and back the value up
the path. No rollouts, no progressive widening yet (that is US2 hardening).

Determinism + determinization: the tree logic is pure. The opponent model's
sampling (inside a folded ``EndTurn``) and the engine's dev-card/steal/YoP draws
go through the process-global RNG; ``run`` reseeds them PER SIMULATION from a seed
derived from ``(cfg.seed, root state)``, so each expanded node gets a clean,
reproducible opponent/chance sample independent of tree-traversal order (open-loop
determinized MCTS — one fresh sample per expanded node). The dice are per-line
faithful via the env clone (Rust RNG, deep-copied). The *agent* (``SearchAgent``)
snapshots/restores the global RNG around a whole search so it never perturbs the
live game's stream (FR-006). Closed-loop per-line replay + N-determinization
averaging is the US2 hardening (T017).
"""

from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.search.node import SearchNode, build_node

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.search.config import SearchConfig
    from catan_rl.search.priors import ActionTuple
    from catan_rl.selfplay.snapshot_opponent import SnapshotOpponent


def clone_env(env: CatanEnv, opponent: SnapshotOpponent | None) -> CatanEnv:
    """Deep-copy ``env`` for simulation, attaching ``opponent`` as the model.

    The live env's snapshot opponent (which wraps a heavy policy net) is detached
    BEFORE the deepcopy so we never clone the network, then restored — the live
    env is left byte-identical. The clone gets ``opponent`` (the search's own
    opponent model) so its folded ``EndTurn`` steps simulate the modelled
    opponent. ``env._snapshot_opponent`` is read/written via the public
    ``set_snapshot_opponent`` setter (no getter exists; the one private read is
    the save).
    """
    saved = env._snapshot_opponent
    env.set_snapshot_opponent(None)
    try:
        clone = copy.deepcopy(env)
    finally:
        env.set_snapshot_opponent(saved)
    clone.set_snapshot_opponent(opponent)
    return clone


def _puct_select(node: SearchNode, c_puct: float) -> ActionTuple:
    """Pick the action maximising Q + c_puct * P * sqrt(ΣN) / (1 + N)."""
    sqrt_total = math.sqrt(max(1, node.total_N))
    best_action: ActionTuple | None = None
    best_score = -math.inf
    for action, prior in node.priors.items():
        q = node.q_value(action)
        u = c_puct * prior * sqrt_total / (1 + node.child_N.get(action, 0))
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    assert best_action is not None  # non-terminal nodes always have >= 1 prior
    return best_action


class MCTS:
    """Determinized PUCT search driven by a frozen policy (priors + value leaf)."""

    def __init__(
        self,
        policy: CatanPolicy,
        cfg: SearchConfig,
        opponent: SnapshotOpponent | None,
        device: Any,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.opponent = opponent
        self.device = device

    def _make_node(
        self, env: CatanEnv, obs: dict[str, np.ndarray], *, terminal: bool
    ) -> SearchNode:
        return build_node(
            self.policy,
            env,
            obs,
            terminal=terminal,
            device=self.device,
            a=self.cfg.value_squash_a,
            b=self.cfg.value_squash_b,
        )

    def _search_base_seed(self, root_env: CatanEnv) -> int:
        """Per-search seed = f(cfg.seed, root state).

        Makes the search a deterministic function of ``(cfg.seed, root state)``
        (FR-006) while giving distinct states (different moves/games) distinct
        opponent + chance streams — so the modelled opponent is not seed-locked
        to one realization across a whole game / the bake-off.
        """
        assert root_env.agent_player is not None and root_env.opponent_player is not None
        sig = (
            self.cfg.seed * 1_000_003
            + root_env._turn_count * 9_973
            + int(root_env.agent_player.victoryPoints) * 131
            + int(root_env.opponent_player.victoryPoints) * 17
        )
        return sig & 0x7FFF_FFFF

    def _reseed(self, s: int) -> None:
        """Reseed the opponent model + the engine's global RNG streams.

        Called once per simulation. Each simulation expands exactly one new node;
        reseeding here makes that expansion's opponent reply + dev-card/steal/YoP
        draws (which go through the process-global ``np.random`` / stdlib
        ``random``) a clean, reproducible sample INDEPENDENT of how many sims ran
        before it — removing the cross-simulation traversal-order coupling (a
        single global stream sliced arbitrarily by tree descent). The dice stay
        per-line faithful via the env clone (Rust RNG, deep-copied, untouched by
        these seeds). This is open-loop determinized MCTS (one fresh
        opponent/chance sample per expanded node); closed-loop per-line replay +
        N-determinization averaging is the US2 hardening (T017).
        """
        s32 = s & 0x7FFF_FFFF
        if self.opponent is not None:
            self.opponent.reset_rng(seed=s32)
        np.random.seed(s32)
        random.seed(s32)

    def run(self, root_env: CatanEnv) -> tuple[ActionTuple, dict[str, Any]]:
        """Search from ``root_env`` (the live env, NOT mutated) and return the
        chosen action 6-tuple + diagnostics. A forced root (<=1 representative
        legal action) short-circuits without spending the simulation budget."""
        root = self._make_node(
            clone_env(root_env, self.opponent), root_env._get_obs(), terminal=False
        )

        sims_run = 0
        if len(root.legal_actions) > 1:
            if self.cfg.sims_per_move is None:
                raise NotImplementedError(
                    "time-budget search is US2 (T021); set sims_per_move for the minimal MCTS"
                )
            base_seed = self._search_base_seed(root_env)
            for i in range(self.cfg.sims_per_move):
                self._reseed(base_seed + i)
                self._simulate(root)
                sims_run += 1

        best = self._best_action(root)
        diagnostics: dict[str, Any] = {
            "root_value": root.value,
            "root_visits": root.total_N,
            "sims_run": sims_run,
            "n_legal_actions": len(root.legal_actions),
            "best_action": best,
            "best_q": root.q_value(best),
            "best_visits": root.child_N.get(best, 0),
            "forced": len(root.legal_actions) <= 1,
        }
        return best, diagnostics

    def _best_action(self, root: SearchNode) -> ActionTuple:
        """Most-visited action (robust child), tie-broken by mean value then prior."""
        return max(
            root.legal_actions,
            key=lambda a: (root.child_N.get(a, 0), root.q_value(a), root.priors[a]),
        )

    def _simulate(self, node: SearchNode) -> float:
        """One playout from ``node``; returns the agent-POV value backed up."""
        if node.is_terminal:
            return node.outcome

        action = _puct_select(node, self.cfg.c_puct)
        if action not in node.children:
            # Expand exactly one new child: clone this node's env (fixing the
            # line's dice future) and step the action through the real engine.
            assert node.env is not None
            child_env = clone_env(node.env, self.opponent)
            _obs, _r, terminated, truncated, _info = child_env.step(
                np.asarray(action, dtype=np.int64)
            )
            terminal = bool(terminated or truncated)
            child = self._make_node(child_env, _obs, terminal=terminal)
            node.children[action] = child
            value = child.outcome if terminal else child.value
        else:
            value = self._simulate(node.children[action])

        node.record(action, value)
        return value
