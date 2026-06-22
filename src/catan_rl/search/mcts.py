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
#: Variance floor for the LCB stderr (avoids sqrt of a tiny-negative float from
#: catastrophic cancellation, and a 0/0 for a single-visit child) — FR-002.
_LCB_VAR_EPS = 1e-8


def _lcb_select(
    actions: list[ActionTuple],
    agg_n: dict[ActionTuple, int],
    agg_w: dict[ActionTuple, float],
    agg_q2: dict[ActionTuple, float],
    priors: dict[ActionTuple, float],
    z: float,
) -> ActionTuple:
    """Pick the root child maximising the lower-confidence bound on its value.

    ``lcb = mean_Q - z * stderr`` where ``mean_Q = W/N`` and
    ``stderr = sqrt(max(eps, Q2/N - mean_Q²) / N)`` (sample variance / N, with an
    eps variance floor). This favours moves that are BOTH high-value AND
    well-explored: a many-times-visited high-Q child has stderr≈0 (LCB→mean_Q),
    while a high-mean-but-rarely-visited child is penalised by its large stderr.

    UNVISITED children (N==0) are non-competitive (assigned -inf). Tie-break by
    (lcb, visits, prior) descending. Operates on the SAME aggregated root child
    stats max-visit selects on (the determinization aggregation), so the two rules
    differ only in the scoring function, not the data.
    """
    best_action: ActionTuple | None = None
    best_key: tuple[float, int, float] = (-math.inf, -1, -math.inf)
    for action in actions:
        n = agg_n.get(action, 0)
        if n <= 0:
            continue  # unvisited -> non-competitive (lcb = -inf)
        mean_q = agg_w.get(action, 0.0) / n
        var = max(_LCB_VAR_EPS, agg_q2.get(action, 0.0) / n - mean_q * mean_q)
        stderr = math.sqrt(var / n)
        lcb = mean_q - z * stderr
        key = (lcb, n, priors.get(action, 0.0))
        if key > best_key:
            best_key = key
            best_action = action
    # If every candidate was unvisited (no sims landed) fall back to the first
    # action (the caller's max-visit path handles the empty-agg case before us).
    return best_action if best_action is not None else actions[0]


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
    fpu: float = 0.0,
) -> ActionTuple:
    """Pick the action maximising Q + c_puct * P * sqrt(ΣN) / (1 + N).

    ``candidates`` restricts the selectable actions (progressive widening);
    ``None`` considers all legal actions (the minimal/US1 behavior). ``fpu`` is the
    First-Play-Urgency value used for an *unvisited* edge's Q (default 0.0 = the
    shipped behavior; with squashed leaf values ~0.5-0.9 that starves siblings,
    so "parent"-mode passes the node's own value here to keep them competitive).
    """
    actions = candidates if candidates is not None else list(node.priors.keys())
    sqrt_total = math.sqrt(max(1, node.total_N))
    best_action: ActionTuple | None = None
    best_score = -math.inf
    for action in actions:
        prior = node.priors[action]
        n = node.child_N.get(action, 0)
        q = (node.child_W[action] / n) if n > 0 else fpu
        u = c_puct * prior * sqrt_total / (1 + n)
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

    def _fpu(self, node: SearchNode) -> float:
        """First-Play-Urgency value for an unvisited edge of ``node``.

        "zero" (shipped) -> 0.0; "parent" -> the node's own squashed leaf value, so
        an unvisited sibling stays competitive with the visited prior-argmax instead
        of sitting at Q=0 far below it (the visit-collapse fix).
        """
        return node.value if self.cfg.fpu_mode == "parent" else 0.0

    def _apply_root_noise(self, root: SearchNode, seed: int) -> None:
        """Mix Dir(alpha) noise into ``root``'s priors (AlphaZero root exploration).

        No-op unless ``cfg.root_dirichlet_alpha`` is set (default off = shipped). A
        LOCAL ``Generator(seed)`` draws the noise so it is reproducible and does NOT
        perturb the process-global RNG that ``_reseed`` uses for opponent/chance
        sampling. Mutates this tree's root priors only (each det tree is fresh)."""
        alpha = self.cfg.root_dirichlet_alpha
        if alpha is None:
            return
        actions = root.legal_actions
        if len(actions) <= 1:
            return
        rng = np.random.default_rng(seed & 0x7FFF_FFFF)
        noise = rng.dirichlet([alpha] * len(actions))
        frac = self.cfg.root_dirichlet_fraction
        for action, e in zip(actions, noise, strict=True):
            root.priors[action] = (1.0 - frac) * root.priors[action] + frac * float(e)

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
        agg_q2: dict[ActionTuple, float] = {}

        for d in range(n_det):
            root = self._make_node(clone_env(root_env, self.opponent), obs, terminal=False)
            if legal_actions is None:
                legal_actions = root.legal_actions
                # Capture a COPY of the clean (pre-noise) policy prior for diagnostics
                # — root noise (if on) mutates root.priors in place below.
                priors = dict(root.priors)
                root_value = root.value
                forced = len(legal_actions) <= 1
            if forced:
                break
            # Root exploration noise (default off): perturb THIS tree's root priors
            # with a stream offset from the per-sim reseed stream.
            self._apply_root_noise(root, (base + d * _DET_SEED_STRIDE + 7919) & 0x7FFF_FFFF)
            total_sims += self._run_one_tree(
                root, (base + d * _DET_SEED_STRIDE) & 0x7FFF_FFFF, n_det
            )
            for action in root.legal_actions:
                agg_n[action] = agg_n.get(action, 0) + root.child_N.get(action, 0)
                agg_w[action] = agg_w.get(action, 0.0) + root.child_W.get(action, 0.0)
                # Aggregate the second moment too (cheap, additive). Read ONLY by
                # the "lcb" final-move path below — never feeds the max-visit math.
                agg_q2[action] = agg_q2.get(action, 0.0) + root.child_Q2.get(action, 0.0)

        assert legal_actions is not None
        if forced or not agg_n:
            best = legal_actions[0] if legal_actions else _END_TURN_ACTION
        elif self.cfg.final_move_mode == "lcb":
            best = _lcb_select(legal_actions, agg_n, agg_w, agg_q2, priors, self.cfg.lcb_z)
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
            # the two to measure the distillable where-to-build gap. ``action_q`` is
            # the backed-up mean value per visited action (for value-margin metrics).
            "visit_counts": dict(agg_n),
            "priors": dict(priors),
            "action_q": {a: agg_w[a] / n for a, n in agg_n.items() if n > 0},
            # Aggregated per-action second moment Σ value² (additive diagnostic;
            # NEW key only). Lets a consumer reconstruct the LCB pick from a SINGLE
            # search (mean_Q from action_q + variance from action_q2) without a
            # second search pass. Search-internal in-memory stats; never persisted.
            "action_q2": {a: agg_q2[a] for a, n in agg_n.items() if n > 0},
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

        action = _puct_select(
            node, self.cfg.c_puct, self._widened_actions(node), fpu=self._fpu(node)
        )
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
