"""The shared head-relevance / mask-routing tables (spec D1 scaffolding).

``HEAD_RELEVANCE`` + ``MASK_KEY_FOR`` live in the torch-free schema module so
the network, the BC writer and the ExIt labeler all read ONE table. Before the
consolidation the rule existed in three independent copies and the BC writer
used a fourth, type-head-only approximation.
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.policy import obs_schema as S
from catan_rl.policy.heads import CatanActionHeads


def test_head_relevance_table_matches_the_network_buffer() -> None:
    heads = CatanActionHeads(trunk_dim=32, hidden_dim=8, node_dim=8)
    expected = torch.tensor(S.HEAD_RELEVANCE, dtype=torch.float32)
    assert heads.head_relevance.shape == (S.N_ACTION_TYPES, len(S.HEAD_DIMS))
    assert torch.equal(heads.head_relevance, expected)


def test_mask_key_routing_is_consistent_with_relevance() -> None:
    for t in range(S.N_ACTION_TYPES):
        for head in range(len(S.HEAD_DIMS)):
            key = S.MASK_KEY_FOR[t][head]
            if S.HEAD_RELEVANCE[t][head]:
                assert key in S.MASK_KEYS, (t, head, key)
            else:
                assert key is None, (t, head, key)


def test_type_head_is_always_relevant() -> None:
    assert all(row[0] == 1 for row in S.HEAD_RELEVANCE)


def test_resource2_routing_is_split_per_type() -> None:
    assert S.MASK_KEY_FOR[S.ActionType.PLAY_YOP][5] == "resource2_yop"
    assert S.MASK_KEY_FOR[S.ActionType.BANK_TRADE][5] == "resource2_trade"


# ---------------------------------------------------------------------------
# is_forced_decision
# ---------------------------------------------------------------------------


def _mask(**over: np.ndarray) -> dict[str, np.ndarray]:
    sizes = {
        "type": S.N_ACTION_TYPES,
        "corner_settlement": S.N_VERTICES,
        "corner_city": S.N_VERTICES,
        "edge": S.N_EDGES,
        "tile": S.N_TILES,
        "resource1_trade": S.N_RESOURCES,
        "resource1_discard": S.N_RESOURCES,
        "resource1_default": S.N_RESOURCES,
        "resource1_yop": S.N_RESOURCES,
        "resource2_yop": S.N_RESOURCES,
        "resource2_yop_same": S.N_RESOURCES,
        "resource2_trade": S.N_RESOURCES,
    }
    out = {k: np.zeros(n, dtype=bool) for k, n in sizes.items()}
    out.update(over)
    return out


def _one_hot(n: int, *idx: int) -> np.ndarray:
    a = np.zeros(n, dtype=bool)
    for i in idx:
        a[i] = True
    return a


def test_roll_dice_is_forced() -> None:
    m = _mask(type=_one_hot(S.N_ACTION_TYPES, S.ActionType.ROLL_DICE))
    assert S.is_forced_decision(m) is True


def test_setup_settlement_with_many_corners_is_not_forced() -> None:
    """THE regression: one legal type, ~24 legal corners. The old type-head-only
    rule deleted every one of these at write time."""
    m = _mask(
        type=_one_hot(S.N_ACTION_TYPES, S.ActionType.BUILD_SETTLEMENT),
        corner_settlement=_one_hot(S.N_VERTICES, 3, 9, 17),
    )
    assert S.is_forced_decision(m) is False


def test_setup_settlement_with_one_corner_is_forced() -> None:
    m = _mask(
        type=_one_hot(S.N_ACTION_TYPES, S.ActionType.BUILD_SETTLEMENT),
        corner_settlement=_one_hot(S.N_VERTICES, 3),
    )
    assert S.is_forced_decision(m) is True


def test_robber_placement_with_many_tiles_is_not_forced() -> None:
    m = _mask(
        type=_one_hot(S.N_ACTION_TYPES, S.ActionType.MOVE_ROBBER),
        tile=_one_hot(S.N_TILES, 0, 4, 11),
    )
    assert S.is_forced_decision(m) is False


def test_multiple_legal_types_is_never_forced() -> None:
    m = _mask(type=_one_hot(S.N_ACTION_TYPES, S.ActionType.END_TURN, S.ActionType.BUY_DEV_CARD))
    assert S.is_forced_decision(m) is False


def test_autoregressive_resource2_head_is_never_a_singleton() -> None:
    """Head 5's legal set depends on the resource1 pick, so a decision that
    reaches it always carries a genuine choice."""
    m = _mask(
        type=_one_hot(S.N_ACTION_TYPES, S.ActionType.BANK_TRADE),
        resource1_trade=_one_hot(S.N_RESOURCES, 0),
        resource2_trade=_one_hot(S.N_RESOURCES, 1),
    )
    assert S.is_forced_decision(m) is False


def test_discard_with_several_holdings_is_not_forced() -> None:
    """Relevance-aware forcing WOULD classify discards as real decisions —
    which is exactly why ``bc.dataset`` excludes them explicitly rather than
    relying on the filter (their recorded resource label is fabricated)."""
    m = _mask(
        type=_one_hot(S.N_ACTION_TYPES, S.ActionType.DISCARD),
        resource1_discard=_one_hot(S.N_RESOURCES, 0, 2, 4),
    )
    assert S.is_forced_decision(m) is False


# ---------------------------------------------------------------------------
# D3 — player-block dim arithmetic must not fork any checkpoint
# ---------------------------------------------------------------------------


def test_player_block_dim_arithmetic_is_pinned() -> None:
    assert S.PLAYER_BASE_DIM == 54
    assert S.CURR_EXTRA_DIM == 6
    assert S.CURR_RESERVED_SLOTS == 7
    assert S.OPP_EXTRA_DIM == 7
    # RESERVED_PLAYER_SLOTS feeds BOTH blocks; shrinking it would move
    # NEXT_PLAYER_DIM and fork every checkpoint.
    assert S.RESERVED_PLAYER_SLOTS == 8
    assert S.PLAYER_BASE_DIM + S.CURR_EXTRA_DIM + S.CURR_RESERVED_SLOTS == S.CURR_PLAYER_DIM == 67
    assert S.PLAYER_BASE_DIM + S.OPP_EXTRA_DIM + S.RESERVED_PLAYER_SLOTS == S.NEXT_PLAYER_DIM == 69


# ---------------------------------------------------------------------------
# D5 — masked_log_dists is the single source of truth for evaluate_actions
# ---------------------------------------------------------------------------


def test_evaluate_actions_gathers_from_masked_log_dists() -> None:
    """The additive accessor and the per-head log-probs cannot drift: the second
    is a ``gather`` of the first, and the pin says so numerically."""
    import torch

    from catan_rl.policy import CatanPolicy
    from catan_rl.policy.obs_schema import ActionType
    from tests.unit.bc.test_loss import _synth_batch

    policy = CatanPolicy().eval()
    batch = _synth_batch(
        6,
        [
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_ROAD,
            ActionType.END_TURN,
            ActionType.BUILD_CITY,
            ActionType.BUY_DEV_CARD,
            ActionType.MOVE_ROBBER,
        ],
    )
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    names = ("type", "corner", "edge", "tile", "resource1", "resource2")
    for head, name in enumerate(names):
        dist = out[f"log_dist/{name}"]
        idx = batch["action"][:, head]
        gathered = dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        assert torch.equal(gathered, out["per_head_log_prob"][:, head]), name
        # A masked log-distribution normalises over its legal support.
        assert torch.allclose(dist.exp().sum(dim=-1), torch.ones(dist.shape[0]), atol=1e-5), name
