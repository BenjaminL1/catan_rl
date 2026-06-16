"""Action priors over the 6-head autoregressive action space (contract C2).

The action space is ``MultiDiscrete([type, corner, edge, tile, res1, res2])``
sampled autoregressively: a 13-way TYPE head, then sub-heads conditioned on the
sampled type. The 13-way type head is the real branching factor (only ~2-6 types
legal mid-game); corner/edge/tile/resource are conditional refinements.

For the minimal search we expand ONE representative child per legal type — the
*modal* (argmax) sub-action of the policy's own conditional heads over the masked
sub-head (the mode, not a sample; the policy samples at rollout). Prior(type) is
the masked type-head probability, so priors sum to 1 over the legal types.
Progressive widening (US2) splits a type's mass across multiple sub-actions.

We deliberately reuse ``CatanActionHeads``' own context/mask helpers
(``_corner_context``, ``_corner_mask``, ...) rather than re-deriving the FiLM
contexts and mask-selection logic here — duplicating that logic would risk
silent drift from how the policy actually samples (the worse bug).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from catan_rl.policy.heads import masked_log_softmax

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.heads import CatanActionHeads
    from catan_rl.policy.network import CatanPolicy

#: A legal action as a plain python 6-tuple (hashable tree-node / dict key).
ActionTuple = tuple[int, int, int, int, int, int]


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> int:
    """Argmax over a (1, D) logit row restricted to ``mask`` (a (1, D) bool).

    Returns 0 if the mask is empty (the sub-choice is then irrelevant/degenerate;
    the env ignores an irrelevant sub-field, mirroring the random actor's
    ``np.zeros(6)`` convention).

    Invariant relied upon: ``env/masks.py`` only marks a TYPE legal when its
    relevant sub-collection is non-empty (and the discard/robber sub-masks
    self-fill), so for any legal type the relevant sub-mask is non-empty and the
    empty-mask branch never fires on the search path. If that invariant ever
    changes, this would silently emit index-0 priors instead of the policy's
    random-uniform fallback — re-audit here.
    """
    if not bool(mask.any()):
        return 0
    masked = logits.masked_fill(~mask, float("-inf"))
    return int(masked.argmax(dim=-1)[0].item())


def _representative_action(
    heads: CatanActionHeads,
    trunk: torch.Tensor,
    masks_t: dict[str, torch.Tensor],
    type_id: int,
) -> ActionTuple:
    """Build the policy's argmax sub-action for a fixed action ``type_id``.

    Only the heads relevant to ``type_id`` (per the relevance buffer) are queried;
    irrelevant sub-fields stay 0.
    """
    type_idx = torch.tensor([type_id], device=trunk.device)
    # head_relevance is a 0/1 float buffer; == 1.0 reads as the explicit gate it is.
    relevance = heads.head_relevance[type_id]  # (6,) — 1.0 where the head matters
    corner = edge = tile = res1 = res2 = 0

    if relevance[1].item() == 1.0:  # corner (settlement / city)
        logits = heads.corner_head(trunk, heads._corner_context(type_idx))
        corner = _masked_argmax(logits, heads._corner_mask(type_idx, masks_t))
    if relevance[2].item() == 1.0:  # edge (road)
        edge = _masked_argmax(heads.edge_head(trunk), masks_t["edge"])
    if relevance[3].item() == 1.0:  # tile (robber / knight)
        tile = _masked_argmax(heads.tile_head(trunk), masks_t["tile"])
    if relevance[4].item() == 1.0:  # resource1 (YoP / monopoly / trade-give / discard)
        logits = heads.resource1_head(trunk, heads._resource1_context(type_idx))
        res1 = _masked_argmax(logits, heads._resource1_mask(type_idx, masks_t))
    if relevance[5].item() == 1.0:  # resource2 (YoP-2nd / trade-receive)
        res1_idx = torch.tensor([res1], device=trunk.device)
        logits = heads.resource2_head(trunk, heads._resource2_context(type_idx, res1_idx))
        res2 = _masked_argmax(logits, heads._resource2_mask(type_idx, res1_idx, masks_t))

    return (type_id, corner, edge, tile, res1, res2)


#: type -> (primary "where" sub-head slot index, mask key, FiLM?) for multi-rep
#: expansion. settle/city -> corner (FiLM, type-conditioned); road -> edge;
#: robber/knight -> tile. Others (res / no-sub types) get a single argmax rep.
_WHERE_HEAD = {0: 1, 1: 1, 2: 2, 4: 3, 6: 3}  # type_id -> head_relevance slot (corner/edge/tile)


def _subaction_candidates(
    heads: CatanActionHeads,
    trunk: torch.Tensor,
    masks_t: dict[str, torch.Tensor],
    type_id: int,
    k: int,
) -> list[tuple[ActionTuple, float]]:
    """Up to ``k`` (action, conditional-prob) candidates for ``type_id``.

    For a placement type, the top-``k`` values of its primary WHERE head (corner/
    edge/tile) by the policy's conditional prob, each with the OTHER sub-fields at
    their argmax; conditional probs are renormalised within the top-k so they sum
    to 1 (k=1 -> [1.0], i.e. prior == P(type), preserving the US1 behavior). Types
    without a WHERE head (resources / EndTurn / ...) get a single argmax rep.
    """
    base = _representative_action(heads, trunk, masks_t, type_id)
    slot = _WHERE_HEAD.get(type_id)
    if slot is None or k <= 1:
        return [(base, 1.0)]

    type_idx = torch.tensor([type_id], device=trunk.device)
    if slot == 1:  # corner (FiLM, type-conditioned)
        logits = heads.corner_head(trunk, heads._corner_context(type_idx))
        mask = heads._corner_mask(type_idx, masks_t)
    elif slot == 2:  # edge
        logits = heads.edge_head(trunk)
        mask = masks_t["edge"]
    else:  # slot == 3, tile
        logits = heads.tile_head(trunk)
        mask = masks_t["tile"]

    if not bool(mask.any()):
        return [(base, 1.0)]
    probs = masked_log_softmax(logits, mask).exp()[0]  # (D,)
    kk = min(k, int(mask[0].sum().item()))
    topk = torch.topk(probs, kk)
    idxs = topk.indices.tolist()
    vals = topk.values.tolist()
    denom = float(sum(vals)) or 1.0
    out: list[tuple[ActionTuple, float]] = []
    for idx, p in zip(idxs, vals, strict=True):
        action = list(base)
        action[slot] = int(idx)
        out.append((tuple(action), float(p) / denom))  # type: ignore[arg-type]
    return out


def priors_from_trunk(
    heads: CatanActionHeads,
    trunk: torch.Tensor,
    masks_t: dict[str, torch.Tensor],
    sub_actions_per_type: int = 1,
) -> dict[ActionTuple, float]:
    """Priors over representative legal actions, from a trunk.

    With ``sub_actions_per_type==1`` (default): one modal action per legal type,
    prior == P(type) — the US1 behavior, unchanged. With k>1: each placement
    type's P(type) is split across its top-k WHERE sub-actions by the conditional
    head probs, so search can explore *where* to build, not only *which* type.
    The hot-path core (no forward here): the MCTS node builder shares one forward.
    """
    type_logp = masked_log_softmax(heads.type_head(trunk), masks_t["type"])
    type_p = type_logp.exp()[0]  # (13,)

    legal_types = torch.nonzero(masks_t["type"][0], as_tuple=False).flatten().tolist()
    priors: dict[ActionTuple, float] = {}
    for type_id in legal_types:
        p_type = float(type_p[type_id].item())
        for action, sub_p in _subaction_candidates(
            heads, trunk, masks_t, int(type_id), sub_actions_per_type
        ):
            priors[action] = priors.get(action, 0.0) + p_type * sub_p

    total = sum(priors.values())
    if total > 0.0:
        priors = {a: p / total for a, p in priors.items()}
    return priors


@torch.no_grad()
def action_priors(
    policy: CatanPolicy,
    env: CatanEnv,
    *,
    device: torch.device | None = None,
    sub_actions_per_type: int = 1,
) -> dict[ActionTuple, float]:
    """Prior distribution over representative legal actions (standalone C2 surface).

    Keys are legal 6-tuples consistent with ``env.get_action_masks()``; values are
    non-negative and sum to 1. No illegal type ever appears. The MCTS hot path uses
    :func:`priors_from_trunk` to share a single forward with the value head.
    """
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

    if device is None:
        device = next(policy.parameters()).device

    obs_t = obs_to_torch(env._get_obs(), device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    trunk = policy.forward(obs_t)["trunk"]
    return priors_from_trunk(policy.action_heads, trunk, masks_t, sub_actions_per_type)
