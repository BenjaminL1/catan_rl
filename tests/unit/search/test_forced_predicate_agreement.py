"""AC-4 — the shared ``forced`` predicate must agree with search's own
independent enumeration.

``search/mcts.py`` decides ``forced = len(legal_actions) <= 1`` off
``priors_from_trunk``'s enumeration, which is derived from the masks by a
completely separate code path (it expands the WHERE head per legal type). This
test walks real env states and cross-checks the two rather than adding a fourth
definition of the rule.

Where they DIFFER is documented and intentional: search expands one
representative per legal type for the RESOURCE types (it has no resource-head
widening), so a singleton resource type reads as "forced" to search while
``is_forced_decision`` — which is what decides whether a row is worth cloning —
sees the open resource head. The agreement pinned here is therefore
(a) ``is_forced_decision ⟹ search-forced`` everywhere, and
(b) exact equality on the placement types, which is where D1's restoration
    (SETUP + ROBBER) actually lives.
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.policy.obs_schema import ActionType, is_forced_decision
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
from catan_rl.search.priors import priors_from_trunk

from .conftest import legal_action, make_policy

_PLACEMENT_TYPES = {
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUILD_ROAD,
    ActionType.MOVE_ROBBER,
    ActionType.PLAY_KNIGHT,
}
_WIDE = 64  # enumerate enough sub-actions that a real choice cannot hide


@torch.no_grad()
def _search_n_legal(policy, env: CatanEnv, obs: dict[str, np.ndarray]) -> int:
    device = torch.device("cpu")
    obs_t = obs_to_torch(obs, device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    out = policy.forward(obs_t)
    nodes = {"v": out["_node_v"], "e": out["_node_e"], "h": out["_node_h"]}
    priors = priors_from_trunk(
        policy.action_heads, out["trunk"], masks_t, nodes, _WIDE, out.get("_is_setup")
    )
    return len(priors)


def test_forced_predicate_agrees_with_search_enumeration() -> None:
    policy = make_policy()
    rng = np.random.default_rng(0)
    checked_placement = 0
    for seed in range(3):
        env = CatanEnv(opponent_type="heuristic", max_turns=60)
        obs, _ = env.reset(seed=seed)
        for _ in range(120):
            masks = env.get_action_masks()
            legal_types = np.flatnonzero(masks["type"])
            if legal_types.size == 0:
                break
            n_search = _search_n_legal(policy, env, obs)
            ours = is_forced_decision(masks)
            search_forced = n_search <= 1

            # (a) our predicate is never MORE permissive than search's.
            assert not (ours and not search_forced), (
                f"is_forced=True but search enumerated {n_search} actions"
            )
            # (b) exact equality on the placement types (SETUP + ROBBER scope).
            if legal_types.size == 1 and int(legal_types[0]) in _PLACEMENT_TYPES:
                checked_placement += 1
                assert ours == search_forced, (
                    f"disagreement on type {int(legal_types[0])}: "
                    f"is_forced={ours}, search n_legal={n_search}"
                )

            action = legal_action(env)
            if rng.random() < 0.5:
                t = int(rng.choice(legal_types))
                action[0] = t
            obs, _r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    assert checked_placement > 0, "no singleton placement state visited — test is vacuous"
