"""Tests for :func:`catan_rl.policy.obs_schema.action_masked_legal`.

The BC writer's legality gate used to consult the TYPE head alone, so a row
whose chosen corner / edge / tile / resource index was masked OFF was still
written to the corpus — supervising the network on an action it could not take
under the same mask at inference time. ``action_masked_legal`` is the
torch-free predicate that closes that hole; it must agree EXACTLY with the
torch masking the policy applies at sample time
(:meth:`catan_rl.policy.heads.CatanActionHeads._resource2_mask` is the hard
case, being autoregressive).
"""

from __future__ import annotations

import numpy as np
import torch

import catan_rl.policy.obs_schema as S
from catan_rl.policy.heads import CatanActionHeads

N_R = S.N_RESOURCES


def _empty_masks() -> dict[str, np.ndarray]:
    masks = {
        "type": np.zeros(S.N_ACTION_TYPES, dtype=bool),
        "corner_settlement": np.zeros(S.N_VERTICES, dtype=bool),
        "corner_city": np.zeros(S.N_VERTICES, dtype=bool),
        "edge": np.zeros(S.N_EDGES, dtype=bool),
        "tile": np.zeros(S.N_TILES, dtype=bool),
    }
    for key in (
        "resource1_trade",
        "resource1_discard",
        "resource1_default",
        "resource1_yop",
        "resource2_yop",
        "resource2_yop_same",
        "resource2_trade",
    ):
        masks[key] = np.zeros(N_R, dtype=bool)
    return masks


def _act(action_type: int, **kw: int) -> np.ndarray:
    a = np.zeros(6, dtype=np.int64)
    a[0] = action_type
    for head, value in kw.items():
        a[int(head[1:])] = value
    return a


# ---------------------------------------------------------------------------
# head 5 — cross-check against the torch implementation
# ---------------------------------------------------------------------------


def test_action_masked_legal_head5_matches_heads_resource2_mask() -> None:
    """Exhaustive (r1, r2) grid x sampled mask vectors, both action types."""
    rng = np.random.default_rng(0)
    for action_type in (S.ActionType.BANK_TRADE, S.ActionType.PLAY_YOP):
        for _ in range(64):
            masks = _empty_masks()
            masks["type"][action_type] = True
            masks["resource1_trade"][:] = rng.random(N_R) < 0.7
            masks["resource1_yop"][:] = rng.random(N_R) < 0.7
            masks["resource2_trade"][:] = rng.random(N_R) < 0.7
            masks["resource2_yop"][:] = rng.random(N_R) < 0.7
            masks["resource2_yop_same"][:] = rng.random(N_R) < 0.7
            torch_masks = {k: torch.from_numpy(v).unsqueeze(0) for k, v in masks.items()}
            for r1 in range(N_R):
                # Make the head-4 gate pass so head 5 is what is under test.
                key1 = S.MASK_KEY_FOR[action_type][4]
                assert key1 is not None
                masks[key1][r1] = True
                torch_masks[key1] = torch.from_numpy(masks[key1]).unsqueeze(0)
                ref = CatanActionHeads._resource2_mask(
                    torch.tensor([int(action_type)]),
                    torch.tensor([r1]),
                    torch_masks,
                )[0]
                for r2 in range(N_R):
                    ours = S.action_masked_legal(masks, _act(action_type, h4=r1, h5=r2))
                    assert ours == bool(ref[r2]), (
                        f"type={int(action_type)} r1={r1} r2={r2}: "
                        f"predicate={ours} torch={bool(ref[r2])}"
                    )


# ---------------------------------------------------------------------------
# the other heads
# ---------------------------------------------------------------------------


def test_illegal_type_is_rejected() -> None:
    masks = _empty_masks()
    masks["type"][S.ActionType.END_TURN] = True
    assert S.action_masked_legal(masks, _act(S.ActionType.END_TURN)) is True
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_ROAD)) is False


def test_offmask_corner_is_rejected_per_type() -> None:
    masks = _empty_masks()
    masks["type"][S.ActionType.BUILD_SETTLEMENT] = True
    masks["type"][S.ActionType.BUILD_CITY] = True
    masks["corner_settlement"][3] = True
    masks["corner_city"][7] = True
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_SETTLEMENT, h1=3)) is True
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_SETTLEMENT, h1=7)) is False
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_CITY, h1=7)) is True
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_CITY, h1=3)) is False


def test_offmask_tile_is_rejected_for_knight_and_robber() -> None:
    masks = _empty_masks()
    masks["type"][S.ActionType.MOVE_ROBBER] = True
    masks["type"][S.ActionType.PLAY_KNIGHT] = True
    masks["tile"][5] = True
    assert S.action_masked_legal(masks, _act(S.ActionType.MOVE_ROBBER, h3=5)) is True
    assert S.action_masked_legal(masks, _act(S.ActionType.MOVE_ROBBER, h3=6)) is False
    # PLAY_KNIGHT is tile-relevant too (HEAD_RELEVANCE[PLAY_KNIGHT][3] == 1).
    assert S.action_masked_legal(masks, _act(S.ActionType.PLAY_KNIGHT, h3=5)) is True
    assert S.action_masked_legal(masks, _act(S.ActionType.PLAY_KNIGHT, h3=6)) is False


def test_irrelevant_heads_are_ignored() -> None:
    """END_TURN carries junk in every downstream head and is still legal."""
    masks = _empty_masks()
    masks["type"][S.ActionType.END_TURN] = True
    a = _act(S.ActionType.END_TURN, h1=11, h2=22, h3=13, h4=4, h5=2)
    assert S.action_masked_legal(masks, a) is True


def test_out_of_range_index_is_rejected_not_raised() -> None:
    masks = _empty_masks()
    masks["type"][S.ActionType.BUILD_ROAD] = True
    masks["edge"][0] = True
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_ROAD, h2=S.N_EDGES)) is False
    assert S.action_masked_legal(masks, _act(S.ActionType.BUILD_ROAD, h2=-1)) is False
