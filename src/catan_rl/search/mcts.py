"""Determinized PUCT-MCTS.

A max-only tree over agent decision points (the env folds the opponent's turn +
dice into the agent's ``EndTurn`` transition). One simulation = descend by PUCT
on the SQUASHED leaf value, expand exactly one new child (clone the parent's env
and ``step`` the action — the clone faithfully + independently continues the
parent's dice future, so the line's stochasticity is fixed), evaluate the new
leaf with the value head (or the true outcome if terminal), and back the value up
the path.

US2 hardening (T017) layered on the minimal core:
- **Progressive widening** on the type head (OFF by default — ``cfg.
  progressive_widening``): when on, a node only exposes its top
  ``max(2, ceil(pw_c * N^pw_alpha))`` legal types (by prior) to PUCT, revealing
  more as the node's visits grow. Default-off keeps the US1 (gated) all-types
  behavior, since the type head's 2-6 branching doesn't need taming.
- **N-determinization aggregation**: run ``n_determinizations`` independent trees
  (distinct determinization base seeds), aggregate the root visit/value across
  them, and pick the globally most-visited action (variance reduction; N=1 is the
  gated US1 behavior, unchanged).
- **Anytime ``time_budget_s``**: simulate until a wall-clock deadline and return
  the best action found so far (the budget is split evenly across determinizations).
- **``max_depth``**: optional depth cut — beyond it a node returns its static leaf
  value instead of expanding.

Determinism + determinization: the tree logic is pure. The opponent model's
sampling (inside a folded ``EndTurn``) and the engine's dev-card/steal/YoP draws
go through the process-global RNG; each simulation reseeds them from a seed
derived from ``(cfg.seed, root state, determinization, sim index)``, so each
expanded node gets a clean, reproducible opponent/chance sample independent of
tree-traversal order (open-loop determinized MCTS). The dice are per-line faithful
via the env clone (Rust RNG, deep-copied). The *agent* (``SearchAgent``) snapshots/
restores the global RNG around a whole search so it never perturbs the live game's
stream (FR-006).
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.search.node import SearchNode, build_node

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.search.config import SearchConfig
    from catan_rl.search.priors import ActionTuple
    from catan_rl.selfplay.snapshot_opponent import SnapshotOpponent

#: Stride between determinization base seeds. Keeps per-det sim-seed ranges
#: (``det_base + i`` for i in [0, sims)) disjoint for any sims < this stride —
#: true for all realistic budgets.
_DET_SEED_STRIDE = 1_000_003
#: Action-type head width (the 13-way type head); the soft policy target's length.
_N_ACTION_TYPES = 13
#: END_TURN action fallback for a degenerate state with no legal actions.
_END_TURN_ACTION: tuple[int, int, int, int, int, int] = (3, 0, 0, 0, 0, 0)


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


def _puct_select(
    node: SearchNode,
    c_puct: float,
    candidates: list[ActionTuple] | None = None,
) -> ActionTuple:
    """Pick the action maximising Q + c_puct * P * sqrt(ΣN) / (1 + N).

    ``candidates`` restricts the selectable actions (progressive widening);
    ``None`` considers all legal actions (the minimal/US1 behavior).
    """
    actions = candidates if candidates is not None else list(node.priors.keys())
    sqrt_total = math.sqrt(max(1, node.total_N))
    best_action: ActionTuple | None = None
    best_score = -math.inf
    for action in actions:
        prior = node.priors[action]
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
            sub_actions_per_type=self.cfg.sub_actions_per_type,
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

        Called once per simulation so that simulation's expansion draws a clean,
        reproducible opponent reply + dev-card/steal/YoP sample INDEPENDENT of how
        many sims ran before it (no cross-sim traversal-order coupling). Dice stay
        per-line faithful via the env clone (Rust RNG, deep-copied, untouched).
        """
        s32 = s & 0x7FFF_FFFF
        if self.opponent is not None:
            self.opponent.reset_rng(seed=s32)
        np.random.seed(s32)
        random.seed(s32)

    def _widened_actions(self, node: SearchNode) -> list[ActionTuple]:
        """Progressive widening on the type head: the top-k legal types by prior.

        ``k = max(2, ceil(pw_c * N^pw_alpha))`` grows with the node's visit count;
        a node with <=2 legal actions exposes them all. As ``N`` grows ``k`` reaches
        ``len(priors)``, so widening front-loads compute on high-prior types
        without ever capping the action set.

        Disabled by default (``cfg.progressive_widening`` False) — then PUCT sees
        ALL legal types, identical to the US1 (gated) behavior.
        """
        actions = node.legal_actions
        if not self.cfg.progressive_widening or len(actions) <= 2:
            return actions
        k = max(2, math.ceil(self.cfg.pw_c * (node.total_N**self.cfg.pw_alpha)))
        if k >= len(actions):
            return actions
        return sorted(actions, key=lambda a: node.priors[a], reverse=True)[:k]

    def run(self, root_env: CatanEnv) -> tuple[ActionTuple, dict[str, Any]]:
        """Search from ``root_env`` (the live env, NOT mutated) and return the
        chosen action 6-tuple + diagnostics. A forced root (<=1 representative
        legal action) short-circuits without spending the simulation budget;
        otherwise ``n_determinizations`` trees are run and their root statistics
        aggregated."""
        obs = root_env._get_obs()
        base = self._search_base_seed(root_env)
        n_det = max(1, self.cfg.n_determinizations)

        legal_actions: list[ActionTuple] | None = None
        priors: dict[ActionTuple, float] = {}
        root_value = 0.0
        forced = False
        total_sims = 0
        agg_n: dict[ActionTuple, int] = {}
        agg_w: dict[ActionTuple, float] = {}

        for d in range(n_det):
            root = self._make_node(clone_env(root_env, self.opponent), obs, terminal=False)
            if legal_actions is None:
                legal_actions = root.legal_actions
                priors = root.priors
                root_value = root.value
                forced = len(legal_actions) <= 1
            if forced:
                break
            total_sims += self._run_one_tree(
                root, (base + d * _DET_SEED_STRIDE) & 0x7FFF_FFFF, n_det
            )
            for action in root.legal_actions:
                agg_n[action] = agg_n.get(action, 0) + root.child_N.get(action, 0)
                agg_w[action] = agg_w.get(action, 0.0) + root.child_W.get(action, 0.0)

        assert legal_actions is not None
        if forced or not agg_n:
            best = legal_actions[0] if legal_actions else _END_TURN_ACTION
        else:
            best = max(
                legal_actions,
                key=lambda a: (
                    agg_n.get(a, 0),
                    agg_w.get(a, 0.0) / max(1, agg_n.get(a, 0)),
                    priors[a],
                ),
            )

        best_n = agg_n.get(best, 0)
        # Per-type visit distribution over the 13-way type head (the search's
        # post-lookahead "how to distribute across action types") — the soft
        # expert-iteration policy target. Forced/no-sims -> one-hot at best's type.
        type_visits = np.zeros(_N_ACTION_TYPES, dtype=np.float64)
        for action, n in agg_n.items():
            type_visits[action[0]] += n
        tv_total = float(type_visits.sum())
        if tv_total > 0.0:
            type_visit_dist = type_visits / tv_total
        else:
            type_visit_dist = np.zeros(_N_ACTION_TYPES, dtype=np.float64)
            type_visit_dist[best[0]] = 1.0
        diagnostics: dict[str, Any] = {
            "root_value": root_value,
            "root_visits": sum(agg_n.values()),
            "sims_run": total_sims,
            "n_determinizations": n_det,
            "n_legal_actions": len(legal_actions),
            "best_action": best,
            "best_q": (agg_w.get(best, 0.0) / best_n) if best_n > 0 else 0.0,
            "best_visits": best_n,
            "forced": forced,
            "type_visit_dist": type_visit_dist,
            # Full-action soft target: search visit counts + the policy prior, on
            # the same (k-expanded) legal-action support. The soft-distillation
            # labeler reads visit_counts as the policy target; the probe compares
            # the two to measure the distillable where-to-build gap.
            "visit_counts": dict(agg_n),
            "priors": dict(priors),
        }
        return best, diagnostics

    def _run_one_tree(self, root: SearchNode, det_base: int, n_det: int) -> int:
        """Run one tree's simulations (fixed sim budget or anytime wall-clock).

        Returns the number of simulations run. For ``time_budget_s`` the per-move
        budget is split evenly across the ``n_det`` determinization trees.
        """
        sims_run = 0
        if self.cfg.time_budget_s is not None:
            deadline = time.monotonic() + (self.cfg.time_budget_s / n_det)
            i = 0
            # Run at least one simulation per tree even if a single sim already
            # exceeds the (split) budget — never return a zero-lookahead move while
            # reporting it as searched.
            while sims_run == 0 or time.monotonic() < deadline:
                self._reseed(det_base + i)
                self._simulate(root)
                i += 1
                sims_run += 1
        else:
            assert self.cfg.sims_per_move is not None
            for i in range(self.cfg.sims_per_move):
                self._reseed(det_base + i)
                self._simulate(root)
                sims_run += 1
        return sims_run

    def _best_action(self, root: SearchNode) -> ActionTuple:
        """Most-visited action of a SINGLE tree (robust child), tie-broken by mean
        value then prior. Aggregation across determinizations happens in ``run``."""
        return max(
            root.legal_actions,
            key=lambda a: (root.child_N.get(a, 0), root.q_value(a), root.priors[a]),
        )

    def _simulate(self, node: SearchNode, depth: int = 0) -> float:
        """One playout from ``node`` at tree ``depth``; returns the backed-up value."""
        if node.is_terminal:
            return node.outcome
        if self.cfg.max_depth is not None and depth >= self.cfg.max_depth:
            # Depth cut: use the static squashed leaf value, do not expand deeper.
            return node.value

        action = _puct_select(node, self.cfg.c_puct, self._widened_actions(node))
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
            value = self._simulate(node.children[action], depth + 1)

        node.record(action, value)
        return value
