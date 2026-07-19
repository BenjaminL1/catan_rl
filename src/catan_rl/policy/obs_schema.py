"""Observation-schema and action-space constants for the v2 policy.

This module is the single source of truth for tensor shapes. Both the env's
obs-construction code (Step 3+) and the network (Step 2) import these
constants, so any change here is automatically picked up by both sides.

Notes on the dim choices (vs v1 Phase 1.3 compact obs):
  * ``CURR_PLAYER_DIM = 67`` — 54 legacy compact base + 5 current-player-only
    extras (own hand-total, discard-pressure, own played YoP/Mono/RB) + 8
    reserved strict-0.0 slots (pointer-arch fork, D3).
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
#   own hand-total (1) + discard-pressure (1) + own played YoP/Mono/RB (3).
CURR_EXTRA_DIM = 5
# Opponent-only extras (legacy): 6-bin hidden-dev-count one-hot + total-res.
OPP_EXTRA_DIM = 7
# Reserved strict-0.0 headroom, per player block (D3.4). Keeps future scalars
# shape-stable so a new signal fills a slot mid-lineage with no fork / BC-regen.
RESERVED_PLAYER_SLOTS = 8

CURR_PLAYER_DIM = PLAYER_BASE_DIM + CURR_EXTRA_DIM + RESERVED_PLAYER_SLOTS  # 67
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
    "resource2_default",
)
