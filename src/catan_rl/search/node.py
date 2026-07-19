"""``SearchNode`` — one agent decision point in the determinized search tree.

The tree is MAX-ONLY: ``CatanEnv.step`` folds the opponent's whole turn + dice
into the agent's ``EndTurn`` transition, so every node is an *agent* decision and
all values are agent-POV (no per-ply sign flip — see ``mcts.py`` / ``value.py``).

A node owns the cloned ``CatanEnv`` at its state (so a child can be expanded by
cloning + stepping it), the agent-POV obs, the squashed leaf value, and the
PUCT visit/value statistics keyed by the legal action 6-tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from catan_rl.search.priors import ActionTuple, priors_from_trunk
from catan_rl.search.value import squash_value

if TYPE_CHECKING:
    import numpy as np

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy


def agent_outcome(env: CatanEnv) -> float:
    """Agent-POV terminal outcome: 1.0 iff the agent won, else 0.0.

    Mirrors the eval harness's win rule (``agent_vp >= maxPoints AND > opp_vp``);
    a truncation / opponent-win / tie is 0.0. Uses true ``victoryPoints`` (the
    win condition is on full VP, as in ``_terminal_reward``).
    """
    assert env.agent_player is not None and env.opponent_player is not None
    assert env.game is not None
    agent_vp = int(env.agent_player.victoryPoints)
    opp_vp = int(env.opponent_player.victoryPoints)
    return 1.0 if (agent_vp >= env.game.maxPoints and agent_vp > opp_vp) else 0.0


@dataclass
class SearchNode:
    """A decision node + its per-action PUCT statistics."""

    #: Cloned env at this state — kept for non-terminal nodes (to expand children
    #: by cloning + stepping); ``None`` for terminal leaves (nothing to expand).
    env: CatanEnv | None
    #: Agent-POV observation at this state (as returned by reset/step).
    obs: dict[str, np.ndarray]
    is_terminal: bool
    #: Agent-POV outcome in {0.0, 1.0}; valid iff ``is_terminal``.
    outcome: float
    #: Squashed leaf win-probability in (0,1); valid iff not terminal.
    value: float
    #: Prior over one representative legal action per legal type (sums to 1).
    priors: dict[ActionTuple, float]

    children: dict[ActionTuple, SearchNode] = field(default_factory=dict)
    child_N: dict[ActionTuple, int] = field(default_factory=dict)
    child_W: dict[ActionTuple, float] = field(default_factory=dict)
    #: IN-MEMORY per-child second moment Σ value² accumulated in lock-step with
    #: ``child_W`` (spec 008 FR-002). Powers the LCB final-move rule's per-child
    #: variance/stderr. Search-internal tree state ONLY — never persisted to any
    #: checkpoint/state-dict (only the policy net is serialized). Computed always
    #: (it is cheap) but read ONLY by the "lcb" final-move path, so its presence
    #: leaves the value/visit math + RNG draws byte-identical to before.
    child_Q2: dict[ActionTuple, float] = field(default_factory=dict)
    total_N: int = 0

    @property
    def legal_actions(self) -> list[ActionTuple]:
        return list(self.priors.keys())

    def q_value(self, action: ActionTuple) -> float:
        """Mean backed-up value for ``action`` (FPU = 0 for an unvisited edge)."""
        n = self.child_N.get(action, 0)
        if n == 0:
            return 0.0
        return self.child_W[action] / n

    def record(self, action: ActionTuple, value: float) -> None:
        self.child_N[action] = self.child_N.get(action, 0) + 1
        self.child_W[action] = self.child_W.get(action, 0.0) + value
        # Second-moment accumulator (LCB stderr). Parallel to child_W; does not
        # feed into child_N / child_W / total_N, so it cannot perturb existing
        # selection or backup math (byte-identity when final_move_mode="max_visit").
        self.child_Q2[action] = self.child_Q2.get(action, 0.0) + value * value
        self.total_N += 1


@torch.no_grad()
def build_node(
    policy: CatanPolicy,
    env: CatanEnv,
    obs: dict[str, np.ndarray],
    *,
    terminal: bool,
    device: torch.device,
    a: float,
    b: float,
    sub_actions_per_type: int = 1,
) -> SearchNode:
    """Construct a node for ``env``'s current state.

    Terminal: read the true agent-POV outcome and drop the env (no expansion).
    Non-terminal: ONE ``policy.forward`` yields both the squashed leaf value and
    the action priors (shared trunk) — the NN-bound hot path. ``sub_actions_per_type``
    (default 1 = one modal action per legal type) lets the priors expand multiple
    WHERE sub-actions per placement type so search can explore where-to-build.
    """
    if terminal:
        return SearchNode(
            env=None,
            obs=obs,
            is_terminal=True,
            outcome=agent_outcome(env),
            value=0.0,
            priors={},
        )

    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

    obs_t = obs_to_torch(obs, device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    out = policy.forward(obs_t)
    v = squash_value(out["value"], a, b)
    assert isinstance(v, torch.Tensor)
    nodes = {"v": out["_node_v"], "e": out["_node_e"], "h": out["_node_h"]}
    priors = priors_from_trunk(
        policy.action_heads,
        out["trunk"],
        masks_t,
        nodes,
        sub_actions_per_type,
        out.get("_is_setup"),
    )
    return SearchNode(
        env=env,
        obs=obs,
        is_terminal=False,
        outcome=0.0,
        value=float(v.item()),
        priors=priors,
    )
