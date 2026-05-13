"""Observation-schema and action-space constants for the v2 policy.

This module is the single source of truth for tensor shapes. Both the env's
obs-construction code (Step 3+) and the network (Step 2) import these
constants, so any change here is automatically picked up by both sides.

Notes on the dim choices (vs v1 Phase 1.3 compact obs):
  * ``CURR_PLAYER_DIM = 54`` — same as v1 compact.
  * ``NEXT_PLAYER_DIM = 61`` — same as v1 compact (54 base + 6 hidden-dev
    one-hot + 1 total-resources scalar).
  * ``TILE_DIM = 79`` — same as v1; per-tile features cover resource +
    number token + dot count + robber + per-corner ownership + per-edge
    ownership + per-corner port.
  * ``DEV_TYPES = 5`` — KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY (matches the
    belief-head target order).
  * Charlesworth resource order is used for the obs (different from the
    engine's alphabetical ``RESOURCES``); see ``RESOURCES_CW``.
"""

from __future__ import annotations

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
CURR_PLAYER_DIM = 54
NEXT_PLAYER_DIM = 61

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
CORNER_CONTEXT_DIM = 2  # settlement vs city
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
    "resource2_default",
)
