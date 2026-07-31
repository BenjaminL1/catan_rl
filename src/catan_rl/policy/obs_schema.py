"""Observation-schema and action-space constants for the v2 policy.

This module is the single source of truth for tensor shapes. Both the env's
obs-construction code (Step 3+) and the network (Step 2) import these
constants, so any change here is automatically picked up by both sides.

Notes on the dim choices (vs v1 Phase 1.3 compact obs):
  * ``CURR_PLAYER_DIM = 67`` — 54 legacy compact base + 6 current-player-only
    extras (own hand-total, discard-pressure, own played YoP/Mono/RB,
    remaining-owed discard count) + 7 reserved strict-0.0 slots (pointer-arch
    fork D3; the sixth extra consumed one reserved slot, so the total is
    unchanged and no checkpoint forks).
  * ``NEXT_PLAYER_DIM = 69`` — 54 legacy compact base + 7 opponent extras
    (6-bin hidden-dev one-hot + 1 total-resources scalar) + 8 reserved
    strict-0.0 slots.
  * ``GLOBAL_DIM = 14`` — POV-neutral block: 5 finite-bank-remaining + 5
    public-reveal-derived dev-deck-remaining + 4 reserved (D3.3).
  * ``TILE_DIM = 79`` — same as v1; per-tile features cover resource +
    number token + dot count + robber + per-corner ownership + per-edge
    ownership + per-corner port.
  * ``DEV_TYPES = 5`` — KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY (matches the
    belief-head target order).
  * Charlesworth resource order is used for the obs (different from the
    engine's alphabetical ``RESOURCES``); see ``RESOURCES_CW``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ---------------------------------------------------------------------------
# Resources & dev cards
# ---------------------------------------------------------------------------

#: Charlesworth resource order (used in obs + 6-head action codec).
RESOURCES_CW: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")
N_RESOURCES = len(RESOURCES_CW)

#: Dev-card ordering used by the belief head and the bincount encoder.
DEV_CARD_ORDER: tuple[str, ...] = ("KNIGHT", "VP", "ROADBUILDER", "YEAROFPLENTY", "MONOPOLY")
N_DEV_TYPES = len(DEV_CARD_ORDER)

# ---------------------------------------------------------------------------
# Board geometry (fixed by Catan rules; never change in 1v1)
# ---------------------------------------------------------------------------

N_TILES = 19
N_VERTICES = 54
N_EDGES = 72
N_CORNERS_PER_TILE = 6
N_EDGES_PER_TILE = 6

# ---------------------------------------------------------------------------
# Observation tensor dims
# ---------------------------------------------------------------------------

TILE_DIM = 79

# --- Per-player scalar blocks (pointer-arch fork, D3) -----------------------
# The legacy shared block (54 dims) is preserved BYTE-FOR-BYTE at the FRONT of
# both player vectors so a migration can copy the old encoder columns verbatim
# and zero-pad only the appended tail. New signals are appended AFTER the base.
PLAYER_BASE_DIM = 54  # legacy compact block (unchanged ordering)

# Current-player-only honest additions (D3.1 + D3.2):
#   own hand-total (1) + discard-pressure (1) + own played YoP/Mono/RB (3)
#   + remaining-owed discard count (1, spec bc-coverage-and-bank-legality D3).
CURR_EXTRA_DIM = 6
# Opponent-only extras (legacy): 6-bin hidden-dev-count one-hot + total-res.
OPP_EXTRA_DIM = 7
# Reserved strict-0.0 headroom, per player block (D3.4). Keeps future scalars
# shape-stable so a new signal fills a slot mid-lineage with no fork / BC-regen.
#
# ``RESERVED_PLAYER_SLOTS`` feeds BOTH player blocks, so it must NOT shrink when
# only one block consumes a slot — decrementing it would move the opponent
# block's dim and fork every checkpoint. Instead the current-player block gets
# its own reserved count, one smaller, keeping CURR_PLAYER_DIM invariant.
RESERVED_PLAYER_SLOTS = 8
CURR_RESERVED_SLOTS = 7

CURR_PLAYER_DIM = PLAYER_BASE_DIM + CURR_EXTRA_DIM + CURR_RESERVED_SLOTS  # 67
NEXT_PLAYER_DIM = PLAYER_BASE_DIM + OPP_EXTRA_DIM + RESERVED_PLAYER_SLOTS  # 69

# --- POV-neutral GLOBAL block (D3.3) ----------------------------------------
# Lives OUTSIDE the current/next player pair; built once per obs. Public,
# honest signals only (bank + public-reveal-derived dev-deck) + reserved slots.
GLOBAL_BANK_DIM = N_RESOURCES  # finite bank remaining, bank[r]/19
GLOBAL_DEVDECK_DIM = N_DEV_TYPES  # public-reveal-derived per-type deck remaining
GLOBAL_RESERVED_SLOTS = 4
GLOBAL_DIM = GLOBAL_BANK_DIM + GLOBAL_DEVDECK_DIM + GLOBAL_RESERVED_SLOTS  # 14

#: Finite resource bank capacity per resource (spec-009); used for bank[r]/CAP.
BANK_CAPACITY = 19

#: Initial dev-deck composition over DEV_CARD_ORDER (KNIGHT, VP, ROADBUILDER,
#: YEAROFPLENTY, MONOPOLY). Standard Catan deck = 14/5/2/2/2 = 25 cards. This is
#: a PUBLIC constant (not engine deck truth); the honest per-type-remaining
#: feature derives from it minus own-cards minus publicly-played (see D3.3).
DEV_DECK_INITIAL: tuple[int, ...] = (14, 5, 2, 2, 2)

# ---------------------------------------------------------------------------
# Opponent identity embedding (Phase 3.6 carryover — keeps the policy able to
# switch personas when the env reveals the opponent kind during training)
# ---------------------------------------------------------------------------

#: Opponent "kind" indices for the embedding table. ``UNKNOWN`` is reserved
#: for eval-time games where opponent identity is intentionally hidden.
OPP_KIND_UNKNOWN = 0
OPP_KIND_RANDOM = 1
OPP_KIND_HEURISTIC = 2
OPP_KIND_SELF_LATEST = 3
OPP_KIND_LEAGUE = 4
OPP_KIND_EXPLOITER = 5
N_OPP_KINDS = 6

#: League-policy id slot count. Phase 3.6 league maxlen + 1 unknown sentinel.
LEAGUE_MAXLEN_DEFAULT = 100
N_OPP_POLICY_SLOTS = LEAGUE_MAXLEN_DEFAULT + 1  # last index = "unknown"

# ---------------------------------------------------------------------------
# Action space (6 autoregressive heads)
# ---------------------------------------------------------------------------

N_ACTION_TYPES = 13


class ActionType:
    """Symbolic indices into the type head."""

    BUILD_SETTLEMENT = 0
    BUILD_CITY = 1
    BUILD_ROAD = 2
    END_TURN = 3
    MOVE_ROBBER = 4
    BUY_DEV_CARD = 5
    PLAY_KNIGHT = 6
    PLAY_YOP = 7
    PLAY_MONOPOLY = 8
    PLAY_ROAD_BUILDER = 9
    BANK_TRADE = 10
    DISCARD = 11
    ROLL_DICE = 12


#: Per-head output sizes (matches MultiDiscrete in CatanEnv.action_space).
HEAD_DIMS: tuple[tuple[str, int], ...] = (
    ("type", N_ACTION_TYPES),
    ("corner", N_VERTICES),
    ("edge", N_EDGES),
    ("tile", N_TILES),
    ("resource1", N_RESOURCES),
    ("resource2", N_RESOURCES),
)

#: Which action types each "context-using" head conditions on.
#: The corner head needs to know whether this is a settlement (0) or city (1)
#: placement; the resource heads need the type of dev-card / trade in flight.
#: The other heads (type, edge, tile) take no per-action-type context.
CORNER_CONTEXT_DIM = 3  # settlement, city, is_setup (D2 snake-draft modulation)
RESOURCE_CONTEXT_DIM = 4  # which "resource-consuming" action: YoP, Mono, Trade, Discard
RESOURCE2_RES1_CONTEXT_DIM = N_RESOURCES  # one-hot over the first resource


# ---------------------------------------------------------------------------
# Mask keys (must match CatanEnv.get_action_masks() output)
# ---------------------------------------------------------------------------

MASK_KEYS: tuple[str, ...] = (
    "type",
    "corner_settlement",
    "corner_city",
    "edge",
    "tile",
    "resource1_trade",
    "resource1_discard",
    "resource1_default",
    # D4 (spec bc-coverage-and-bank-legality): the resource heads used to share
    # ONE ``resource1_default`` key across Year-of-Plenty and Monopoly and ONE
    # ``resource2_default`` key across Year-of-Plenty and BankTrade, which made
    # bank-awareness inexpressible. YoP is legal only when ``bank[first] >= 2``
    # (doubled pick) or ``bank[first] >= 1 and bank[second] >= 1``; BankTrade
    # needs ``bank[r2] >= 1``. Monopoly is bank-independent and keeps
    # ``resource1_default``, so YoP takes its own first-pick key and its
    # receive side splits into a different-pick vector (``bank >= 1``) and a
    # doubled-pick vector (``bank >= 2``) that the AUTOREGRESSIVE resource2
    # head selects between using the sampled ``res1``.
    "resource1_yop",
    "resource2_yop",
    "resource2_yop_same",
    "resource2_trade",
)


# ---------------------------------------------------------------------------
# Head relevance + per-type mask-key routing (single source of truth)
# ---------------------------------------------------------------------------
#
# These two tables describe, for every action type, WHICH of the six heads
# contributes to the joint log-prob and WHICH mask key that head reads. They
# live here — the torch-free schema module — so that:
#
#   * :mod:`catan_rl.policy.heads` builds its ``head_relevance`` buffer from
#     them instead of re-declaring the rule in torch, and
#   * :mod:`catan_rl.bc.dataset` / :mod:`catan_rl.expert_iteration.labeler`
#     (both deliberately torch-free) can compute a relevance-aware ``forced``
#     predicate from the same table.
#
# Before this consolidation the rule existed in three independent copies
# (heads.py buffer, replay/recorder_loop.py routing, search/mcts.py
# enumeration) and the BC writer used a fourth, type-head-only approximation.

#: Ordered head names, index-aligned with the 6-head MultiDiscrete action.
HEAD_NAMES: tuple[str, ...] = tuple(name for name, _ in HEAD_DIMS)

#: The resource2 head is AUTOREGRESSIVE — its mask depends on the resource1
#: index sampled one step earlier (BankTrade forbids ``r2 == r1``; YoP swaps in
#: ``resource2_yop_same`` at that index, since a doubled pick needs
#: ``bank >= 2``), so its legal-count is not knowable from the mask dict alone.
#: Callers that reason about "is there really a choice here?" must treat it as
#: never-singleton. ``MASK_KEY_FOR[..][5]`` therefore names only the BASE key;
#: ``resource2_yop_same`` is a refinement read by ``heads._resource2_mask``.
AUTOREGRESSIVE_HEAD = 5


def _build_head_tables() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[str | None, ...], ...]]:
    relevance: list[list[int]] = [[0] * len(HEAD_DIMS) for _ in range(N_ACTION_TYPES)]
    keys: list[list[str | None]] = [[None] * len(HEAD_DIMS) for _ in range(N_ACTION_TYPES)]

    def _set(action_type: int, head: int, mask_key: str) -> None:
        relevance[action_type][head] = 1
        keys[action_type][head] = mask_key

    for t in range(N_ACTION_TYPES):
        _set(t, 0, "type")  # the type head is always relevant
    _set(ActionType.BUILD_SETTLEMENT, 1, "corner_settlement")
    _set(ActionType.BUILD_CITY, 1, "corner_city")
    _set(ActionType.BUILD_ROAD, 2, "edge")
    _set(ActionType.MOVE_ROBBER, 3, "tile")
    _set(ActionType.PLAY_KNIGHT, 3, "tile")  # knight triggers a robber move
    _set(ActionType.PLAY_YOP, 4, "resource1_yop")
    _set(ActionType.PLAY_MONOPOLY, 4, "resource1_default")
    _set(ActionType.BANK_TRADE, 4, "resource1_trade")
    _set(ActionType.DISCARD, 4, "resource1_discard")
    _set(ActionType.PLAY_YOP, 5, "resource2_yop")
    _set(ActionType.BANK_TRADE, 5, "resource2_trade")
    return (
        tuple(tuple(row) for row in relevance),
        tuple(tuple(row) for row in keys),
    )


#: ``HEAD_RELEVANCE[action_type][head] == 1`` iff that head's log-prob is part
#: of the joint log-prob for the action type. Mirrored into a torch buffer by
#: :class:`catan_rl.policy.heads.CatanActionHeads`.
#: ``MASK_KEY_FOR[action_type][head]`` is the mask-dict key that head reads for
#: that type (``None`` where the head is irrelevant).
HEAD_RELEVANCE, MASK_KEY_FOR = _build_head_tables()


#: Bumped whenever :func:`is_forced_decision` changes semantics. Stamped into
#: the BC manifest and enforced by :mod:`catan_rl.bc.loader` (D5) so a corpus
#: written under an older rule cannot be silently trained on.
#:
#: 1 — type-head-only (``mask["type"].sum() <= 1``): deleted every setup and
#:     robber decision at write time because their type mask is a singleton.
#: 2 — relevance-aware (this module's :func:`is_forced_decision`).
FORCED_RULE_VERSION = 2

#: Revision of the 1v1 ruleset + legality contract a BC corpus was written
#: under. Stamped into the manifest alongside :data:`FORCED_RULE_VERSION` and
#: enforced by :mod:`catan_rl.bc.loader` (D5).
#:
#: 1 — pre-spec-009 (infinite bank), 9-key mask dict.
#: 2 — spec-009 finite bank + D4 bank-gated YoP / BankTrade legality:
#:     ``resource1_yop`` split off ``resource1_default`` and
#:     ``resource2_default`` split into ``resource2_yop`` /
#:     ``resource2_yop_same`` / ``resource2_trade`` (12-key mask dict).
#: 3 — BC WRITE-TIME contract change (F-A / F-B). The mask KEYS are unchanged
#:     from 2, but what a corpus CONTAINS is not:
#:     * ``PLAY_KNIGHT`` rows now carry the hex the teacher is about to rob,
#:       under a populated robber-branch ``tile`` mask. Under 2 they carried
#:       the default ``tile_idx = 0`` under an all-False tile mask, so the
#:       tile head got no gradient on exactly those rows.
#:     * the write-time legality gate is relevance-aware
#:       (:func:`action_masked_legal`) rather than type-head-only, so a row
#:       whose corner / edge / tile / resource index is masked OFF is dropped.
#:     A version-2 corpus loaded into this tree would train the tile head on
#:     fabricated knight labels, and nothing in the shard arrays can reveal
#:     that — hence the bump. Owner-signed-off 2026-07-31, acknowledging that
#:     the next BC train needs a regeneration.
#:
#: SCOPE NOTE — the ExIt labeler INHERITS this constant.
#: ``expert_iteration/labeler.py`` stamps the same ``RULESET_VERSION`` into its
#: manifests, but its write contract did NOT change at 3: it does not record
#: ``PLAY_KNIGHT`` tile labels the way ``bc/dataset.py`` does, so the two
#: bullets above describe BC corpora only. An ExIt corpus stamped 3 is stamped
#: with a version whose documented semantics do not describe it. That is
#: accepted (the stamp is still monotonic and still refuses genuinely stale
#: data); it is recorded here so a future reader does not infer an ExIt
#: contract change that never happened.
RULESET_VERSION = 3


def is_forced_decision(masks: object) -> bool:
    """Return True iff ``masks`` describes a decision with no real choice.

    Relevance-aware: a decision is forced only when the type head is a
    singleton AND every *relevant* downstream head is a singleton too. The
    old type-head-only rule called a setup settlement placement "forced"
    because ``BUILD_SETTLEMENT`` was the only legal type — while the corner
    head still offered ~24 distinct vertices.

    The autoregressive resource2 head (:data:`AUTOREGRESSIVE_HEAD`) is never
    treated as a singleton: its legal set depends on the resource1 pick, so a
    decision that reaches it always carries a genuine choice.

    Args:
        masks: mapping of mask key -> boolean sequence (numpy array or list).
    """
    masks_map = cast("Mapping[str, Sequence[bool]]", masks)
    type_mask = masks_map["type"]
    legal_types = [i for i, legal in enumerate(type_mask) if legal]
    if len(legal_types) > 1:
        return False
    if not legal_types:
        # No legal type at all — degenerate; nothing to learn from it.
        return True
    action_type = legal_types[0]
    for head in range(1, len(HEAD_DIMS)):
        if not HEAD_RELEVANCE[action_type][head]:
            continue
        if head == AUTOREGRESSIVE_HEAD:
            return False
        key = MASK_KEY_FOR[action_type][head]
        assert key is not None  # relevance implies a routed mask key
        if sum(1 for legal in masks_map[key] if legal) > 1:
            return False
    return True


def _index_legal(mask: Sequence[bool], idx: int) -> bool:
    """``mask[idx]`` with out-of-range treated as illegal rather than raising."""
    if idx < 0 or idx >= len(mask):
        return False
    return bool(mask[idx])


def action_masked_legal(masks: object, action: object) -> bool:
    """Return True iff every RELEVANT head of ``action`` is on its own mask.

    The torch-free twin of the masking :class:`catan_rl.policy.heads.
    CatanActionHeads` applies at sample time. Writers of offline corpora (BC,
    expert iteration) must gate on this and not on ``masks["type"][a[0]]``
    alone: a type-legal row whose corner / edge / tile / resource index is
    masked OFF supervises the network on an action it can never sample under
    the same mask, which is exactly the train/serve skew BC exists to avoid.

    Head 5 is AUTOREGRESSIVE and mirrors ``heads._resource2_mask``:

    * ``BANK_TRADE`` — ``resource2_trade[r2]`` and ``r2 != r1`` (the engine
      accepts a same-resource trade as a strict loss; the policy never
      samples one).
    * ``PLAY_YOP`` — ``resource2_yop_same[r2]`` when ``r2 == r1`` (a doubled
      pick needs ``bank >= 2``), else ``resource2_yop[r2]``.

    Args:
        masks: mapping of mask key -> boolean sequence (numpy array or list).
        action: the 6-tuple ``[type, corner, edge, tile, res1, res2]`` (numpy
            array, list or tuple).
    """
    masks_map = cast("Mapping[str, Sequence[bool]]", masks)
    act = cast("Sequence[int]", action)
    action_type = int(act[0])
    if not _index_legal(masks_map["type"], action_type):
        return False
    for head in range(1, len(HEAD_DIMS)):
        if not HEAD_RELEVANCE[action_type][head]:
            continue
        idx = int(act[head])
        if head == AUTOREGRESSIVE_HEAD:
            res1 = int(act[4])
            if action_type == ActionType.BANK_TRADE:
                if idx == res1 or not _index_legal(masks_map["resource2_trade"], idx):
                    return False
            else:  # PLAY_YOP — the only other resource2-relevant type
                key2 = "resource2_yop_same" if idx == res1 else "resource2_yop"
                if not _index_legal(masks_map[key2], idx):
                    return False
            continue
        key = MASK_KEY_FOR[action_type][head]
        assert key is not None  # relevance implies a routed mask key
        if not _index_legal(masks_map[key], idx):
            return False
    return True
