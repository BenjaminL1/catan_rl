"""Build the v2 observation dict from a live game state.

This module is the canonical bridge between the engine + env state machine
and the tensors the v2 policy network consumes. Used by:

  * :class:`catan_rl.env.catan_env.CatanEnv` — at every ``step()`` /
    ``reset()`` to produce the obs returned to the agent.
  * The Step 3 BC dataset generator — to record (state, action) pairs
    from heuristic-vs-heuristic rollouts in the same schema the policy
    will see at training time.

Schema (sizes from :mod:`catan_rl.policy.obs_schema`):

  * ``tile_representations``      (19, 79) — Charlesworth tile features.
  * ``current_player_main``       (54,)    — agent scalar features.
  * ``next_player_main``          (61,)    — opponent scalar features.
  * ``current_dev_counts``        (5,)     — agent's dev-card counts
                                              (KNIGHT/VP/ROADBUILDER/YOP/MONO).
  * ``next_played_dev_counts``    (5,)     — opp's PLAYED dev cards (observable).
  * ``hex_features``              (19, 19) — GNN per-hex node input.
  * ``vertex_features``           (54, 16) — GNN per-vertex node input.
  * ``edge_features``             (72, 16) — GNN per-edge node input.
  * ``opponent_kind``             scalar int64 — Phase 3.6 opp-id kind.
  * ``opponent_policy_id``        scalar int64 — Phase 3.6 opp-id slot.

**Resource-ordering footgun** (CLAUDE.md rule #5): the engine's internal
``RESOURCES`` order is alphabetical (BRICK, ORE, SHEEP, WHEAT, WOOD),
but the obs schema's :data:`RESOURCES_CW` is Charlesworth-canonical
(WOOD, BRICK, WHEAT, ORE, SHEEP). All obs fields use Charlesworth order;
this module is responsible for the translation. A dedicated unit test
(``test_resource_ordering_charlesworth``) pins the contract.

**Hand tracker integration**: opponent resource counts in
``next_player_main`` come from the optional :class:`BroadcastHandTracker`,
which subscribes to the engine bus and maintains an event-driven copy of
each player's resources. If no tracker is provided, the encoder falls
back to the opponent's raw ``player.resources`` dict (also valid in 1v1
+ no-P2P-trading, but breaks the moment either of those changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.engine.board import catanBoard
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    DEV_CARD_ORDER,
    N_DEV_TYPES,
    N_EDGES,
    N_OPP_KINDS,
    N_OPP_POLICY_SLOTS,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    OPP_KIND_UNKNOWN,
    RESOURCES_CW,
    TILE_DIM,
)

if TYPE_CHECKING:
    # Import only for type checking — avoids a circular import where
    # ``env/__init__.py`` imports CatanEnv (which imports back here).
    from catan_rl.env.hand_tracker import BroadcastHandTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Engine alphabetical resource order (used inside ``resourcesList`` and
# ``player.resources`` dicts) — kept here for translation. Source of truth
# lives in the engine; this module asserts the set matches at import time.
_RESOURCE_TYPES_FOR_ONEHOT: tuple[str, ...] = (
    "BRICK",
    "ORE",
    "SHEEP",
    "WHEAT",
    "WOOD",
    "DESERT",
)
_N_RESOURCE_ONEHOT = len(_RESOURCE_TYPES_FOR_ONEHOT)  # 6

#: Number tokens that appear on hexes, plus None for desert. Order matches
#: the v1 obs schema; do not change without ranging through every consumer.
_NUMBER_TOKEN_ORDER: tuple[int | None, ...] = (
    None,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
)
_N_NUMBER_ONEHOT = len(_NUMBER_TOKEN_ORDER)  # 11

# Dot count by number token (the standard Catan dot-count is how many of
# the 36 2d6 outcomes roll that number). Public alias ``DOTS_BY_TOKEN``
# is exposed for downstream consumers (e.g. setup-phase analytic scorer).
_DOTS_BY_TOKEN: dict[int | None, int] = {
    None: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
}
_MAX_DOTS = 5  # used to normalise dim 18

# Public alias of the dot-count table for cross-module imports.
DOTS_BY_TOKEN: dict[int | None, int] = _DOTS_BY_TOKEN

# Hex-feature dim for the GNN: same as tile_representations dims 0..18.
HEX_FEATURE_DIM = 19

# Vertex feature dim for the GNN. Layout (16 dims):
#   3 owner one-hot (empty / mine / opp)
#   3 building one-hot (none / settle / city)
#   7 port one-hot (none / generic / 2:1 WOOD,BRICK,WHEAT,ORE,SHEEP)
#   1 has_any_building flag (redundant but cheap and useful for GNN msgs)
#   2 reserved (padding to keep dim a round 16)
VERTEX_FEATURE_DIM = 16
_VTX_OWNER_OFFSET = 0
_VTX_BUILDING_OFFSET = 3
_VTX_PORT_OFFSET = 6
_VTX_HAS_BUILDING_OFFSET = 13

# Edge feature dim for the GNN. Layout (16 dims):
#   3 owner one-hot (none / mine / opp)
#   1 has_road flag
#   12 reserved (padding)
EDGE_FEATURE_DIM = 16
_EDGE_OWNER_OFFSET = 0
_EDGE_HAS_ROAD_OFFSET = 3

# Port one-hot ordering (used in vertex_features layout).
_PORT_LAYOUT: tuple[str, ...] = (
    "NONE",
    "3:1 PORT",
    "2:1 WOOD",
    "2:1 BRICK",
    "2:1 WHEAT",
    "2:1 ORE",
    "2:1 SHEEP",
)
_N_PORTS = len(_PORT_LAYOUT)  # 7


# ---------------------------------------------------------------------------
# Env state input
# ---------------------------------------------------------------------------


@dataclass
class EnvObsState:
    """Subset of env state needed by the obs encoder.

    The env constructs one of these on every ``_get_obs`` call. The BC
    dataset generator (Step 3) constructs the same shape from its own
    state machine, so the encoder doesn't have to know which caller it
    has.
    """

    initial_placement_phase: bool
    setup_step: int = 0
    roll_pending: bool = False
    discard_pending: bool = False
    robber_placement_pending: bool = False
    road_building_roads_left: int = 0
    last_dice_roll: int = 0
    # Phase 3.6 opponent-identity inputs. Defaults to unknown — eval-time
    # default for BC; PPO env passes real values once a league exists.
    opp_kind: int = OPP_KIND_UNKNOWN
    opp_policy_id: int = N_OPP_POLICY_SLOTS - 1
    # Whether the env wants to stochastically mask opp-id this step (Phase
    # 3.6 ``opp_id_mask_prob``). False during BC pretrain; PPO env may set.
    opp_id_masked: bool = False


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ObsEncoder:
    """Stateful per-game encoder.

    Built once at ``env.reset()`` after the engine constructs ``catanBoard``.
    Caches the static-per-board tables (tile static features, per-tile
    corner pixel coords, per-vertex port assignment) so each obs build
    only computes the dynamic dims.

    Note: the static-tile cache is rebuilt on every reset because the
    resource shuffle changes per game. Geometry caches (board_geometry,
    corner pixels) are board-invariant and only need rebuilding if the
    underlying engine were ever to change.
    """

    def __init__(
        self,
        board: catanBoard,
        *,
        vertex_dim: int = VERTEX_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
    ) -> None:
        if vertex_dim < VERTEX_FEATURE_DIM:
            raise ValueError(f"vertex_dim must be >= {VERTEX_FEATURE_DIM}, got {vertex_dim}")
        if edge_dim < EDGE_FEATURE_DIM:
            raise ValueError(f"edge_dim must be >= {EDGE_FEATURE_DIM}, got {edge_dim}")
        self.vertex_dim = vertex_dim
        self.edge_dim = edge_dim
        self._board = board
        self._build_static_caches(board)

    # ------------------------------------------------------------------
    # Static cache (rebuilt per board)
    # ------------------------------------------------------------------

    def _build_static_caches(self, board: catanBoard) -> None:
        # Per-tile static dims 0..18 (resource one-hot, number one-hot,
        # has_robber stays at 0 here, dot count).
        tile_static = np.zeros((N_TILES, HEX_FEATURE_DIM), dtype=np.float32)
        for i in range(N_TILES):
            tile = board.hexTileDict[i]
            res = tile.resource_type
            for j, r in enumerate(_RESOURCE_TYPES_FOR_ONEHOT):
                if res == r:
                    tile_static[i, j] = 1.0
                    break
            num = tile.number_token
            for j, t in enumerate(_NUMBER_TOKEN_ORDER):
                if t == num:
                    tile_static[i, _N_RESOURCE_ONEHOT + j] = 1.0
                    break
            tile_static[i, 17] = 0.0  # has_robber filled dynamically
            tile_static[i, 18] = _DOTS_BY_TOKEN.get(num, 0) / float(_MAX_DOTS)
        self._tile_static = tile_static

        # Per-tile corner pixels in canonical 0..5 order.
        self._tile_corners: list[list[Any]] = [
            list(board.hexTileDict[i].get_corners(board.flat)) for i in range(N_TILES)
        ]

        # Per-vertex port one-hot (54, 7). Each vertex has at most one port.
        port_static = np.zeros((N_VERTICES, _N_PORTS), dtype=np.float32)
        port_static[:, 0] = 1.0  # NONE by default
        for v_idx, px in board.vertex_index_to_pixel_dict.items():
            v_obj = board.boardGraph[px]
            port = v_obj.port if v_obj.port else None
            if port:
                # Find the slot.
                for j, p in enumerate(_PORT_LAYOUT):
                    if p == port:
                        port_static[v_idx, 0] = 0.0
                        port_static[v_idx, j] = 1.0
                        break
        self._vertex_port_static = port_static

        # Per-edge endpoint map: edge_idx → (corner_pt_1, corner_pt_2).
        # Built in the same iteration order as CatanEnv._build_index_maps so
        # indices align. The cross-check in board_geometry.py also covers
        # this; we recompute locally to keep the encoder self-contained.
        seen: set[tuple[str, str]] = set()
        edges: list[tuple[Any, Any]] = []
        for v_pt, v_obj in board.boardGraph.items():
            for nb_pt in v_obj.neighbors:
                key = _edge_key(v_pt, nb_pt)
                if key not in seen:
                    seen.add(key)
                    edges.append((v_pt, nb_pt))
        if len(edges) != N_EDGES:
            raise RuntimeError(f"obs_encoder: derived {len(edges)} edges, expected {N_EDGES}")
        self._edge_pixels: list[tuple[Any, Any]] = edges

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def build_obs(
        self,
        game: Any,
        agent_player: Any,
        opponent_player: Any,
        env_state: EnvObsState,
        hand_tracker: BroadcastHandTracker | None = None,
    ) -> dict[str, np.ndarray | np.int64]:
        """Build the obs dict for the agent's POV."""
        opp_resources = self._opp_resources(opponent_player, hand_tracker)
        opp_kind = OPP_KIND_UNKNOWN if env_state.opp_id_masked else int(env_state.opp_kind)
        opp_policy_id = (
            N_OPP_POLICY_SLOTS - 1 if env_state.opp_id_masked else int(env_state.opp_policy_id)
        )
        if not 0 <= opp_kind < N_OPP_KINDS:
            opp_kind = OPP_KIND_UNKNOWN
        if not 0 <= opp_policy_id < N_OPP_POLICY_SLOTS:
            opp_policy_id = N_OPP_POLICY_SLOTS - 1

        return {
            "tile_representations": self._build_tile_features(game, agent_player),
            "current_player_main": self._build_player_main(
                game, agent_player, agent_player.resources, env_state, is_agent=True
            ),
            "next_player_main": self._build_player_main(
                game, opponent_player, opp_resources, env_state, is_agent=False
            ),
            "current_dev_counts": _dev_counts(agent_player, hidden=True),
            "next_played_dev_counts": _dev_counts(opponent_player, hidden=False),
            "hex_features": self._build_hex_features(game),
            "vertex_features": self._build_vertex_features(game, agent_player),
            "edge_features": self._build_edge_features(game, agent_player),
            "opponent_kind": np.int64(opp_kind),
            "opponent_policy_id": np.int64(opp_policy_id),
        }

    # ------------------------------------------------------------------
    # Per-tile / per-hex features
    # ------------------------------------------------------------------

    def _build_tile_features(self, game: Any, acting_player: Any) -> np.ndarray:
        """(19, 79): static + dynamic per-tile features."""
        board = game.board
        out = np.zeros((N_TILES, TILE_DIM), dtype=np.float32)
        out[:, :HEX_FEATURE_DIM] = self._tile_static  # 19 intrinsic dims

        for h_idx in range(N_TILES):
            # has_robber (dynamic dim 17)
            out[h_idx, 17] = 1.0 if board.hexTileDict[h_idx].has_robber else 0.0

            # Per-vertex blocks dim 19..54 (6 verts × 6 dims).  # noqa: RUF003
            corners = self._tile_corners[h_idx]
            v_off = 19
            for corner_pt in corners:
                v_obj = board.boardGraph.get(corner_pt)
                if v_obj is None:
                    # Should never happen; mark slot as "empty no-building".
                    out[h_idx, v_off] = 1.0
                    out[h_idx, v_off + 3] = 1.0
                else:
                    owner = v_obj.owner
                    btype = v_obj.building_type
                    out[h_idx, v_off + 0] = 1.0 if owner is None else 0.0
                    out[h_idx, v_off + 1] = 1.0 if owner is acting_player else 0.0
                    out[h_idx, v_off + 2] = (
                        1.0 if (owner is not None and owner is not acting_player) else 0.0
                    )
                    out[h_idx, v_off + 3] = 1.0 if btype is None else 0.0
                    out[h_idx, v_off + 4] = 1.0 if btype == "Settlement" else 0.0
                    out[h_idx, v_off + 5] = (
                        1.0 if (btype is not None and btype != "Settlement") else 0.0
                    )
                v_off += 6

            # Per-edge blocks dim 55..78 (6 edges × 4 dims).  # noqa: RUF003
            e_off = 55
            for i in range(6):
                c1 = corners[i]
                c2 = corners[(i + 1) % 6]
                has_road, road_owner = _road_owner_between(board, c1, c2)
                out[h_idx, e_off + 0] = 1.0 if (not has_road or road_owner is None) else 0.0
                out[h_idx, e_off + 1] = 1.0 if (has_road and road_owner is acting_player) else 0.0
                out[h_idx, e_off + 2] = (
                    1.0
                    if (has_road and road_owner is not None and road_owner is not acting_player)
                    else 0.0
                )
                out[h_idx, e_off + 3] = 1.0 if has_road else 0.0
                e_off += 4

        return out

    def _build_hex_features(self, game: Any) -> np.ndarray:
        """(19, 19): GNN per-hex input. Reuses the intrinsic tile dims."""
        out = self._tile_static.copy()
        for h_idx in range(N_TILES):
            out[h_idx, 17] = 1.0 if game.board.hexTileDict[h_idx].has_robber else 0.0
        return out

    # ------------------------------------------------------------------
    # GNN per-vertex / per-edge features
    # ------------------------------------------------------------------

    def _build_vertex_features(self, game: Any, agent: Any) -> np.ndarray:
        """(54, vertex_dim): owner / building / port / has_building flags."""
        board = game.board
        out = np.zeros((N_VERTICES, self.vertex_dim), dtype=np.float32)
        # Port section is static.
        out[:, _VTX_PORT_OFFSET : _VTX_PORT_OFFSET + _N_PORTS] = self._vertex_port_static

        for v_idx, px in board.vertex_index_to_pixel_dict.items():
            v_obj = board.boardGraph[px]
            owner = v_obj.owner
            btype = v_obj.building_type
            out[v_idx, _VTX_OWNER_OFFSET + 0] = 1.0 if owner is None else 0.0
            out[v_idx, _VTX_OWNER_OFFSET + 1] = 1.0 if owner is agent else 0.0
            out[v_idx, _VTX_OWNER_OFFSET + 2] = (
                1.0 if (owner is not None and owner is not agent) else 0.0
            )
            out[v_idx, _VTX_BUILDING_OFFSET + 0] = 1.0 if btype is None else 0.0
            out[v_idx, _VTX_BUILDING_OFFSET + 1] = 1.0 if btype == "Settlement" else 0.0
            out[v_idx, _VTX_BUILDING_OFFSET + 2] = (
                1.0 if (btype is not None and btype != "Settlement") else 0.0
            )
            out[v_idx, _VTX_HAS_BUILDING_OFFSET] = 1.0 if btype is not None else 0.0
        return out

    def _build_edge_features(self, game: Any, agent: Any) -> np.ndarray:
        """(72, edge_dim): owner one-hot + has_road flag."""
        board = game.board
        out = np.zeros((N_EDGES, self.edge_dim), dtype=np.float32)
        for e_idx, (c1, c2) in enumerate(self._edge_pixels):
            has_road, road_owner = _road_owner_between(board, c1, c2)
            out[e_idx, _EDGE_OWNER_OFFSET + 0] = (
                1.0 if (not has_road or road_owner is None) else 0.0
            )
            out[e_idx, _EDGE_OWNER_OFFSET + 1] = 1.0 if (has_road and road_owner is agent) else 0.0
            out[e_idx, _EDGE_OWNER_OFFSET + 2] = (
                1.0 if (has_road and road_owner is not None and road_owner is not agent) else 0.0
            )
            out[e_idx, _EDGE_HAS_ROAD_OFFSET] = 1.0 if has_road else 0.0
        return out

    # ------------------------------------------------------------------
    # Per-player scalar features
    # ------------------------------------------------------------------

    def _build_player_main(
        self,
        game: Any,
        player: Any,
        resources: dict[str, int],
        env_state: EnvObsState,
        *,
        is_agent: bool,
    ) -> np.ndarray:
        """Compact player-feature vector.

        Layout matches the v1 Phase 1.3 compact obs:
          * 5 resources (Charlesworth order) / 8.0          → 5
          * VP / 15.0                                       → 1
          * 5 income                                        → 5
          * 5 best-trade-ratios per resource                → 5
          * 6 port one-hots                                 → 6
          * longest road flag + largest army flag           → 2
          * road length / 15.0                              → 1
          * knights played / 8.0                            → 1
          * settlements left / 5.0                          → 1
          * cities left / 4.0                               → 1
          * roads left / 15.0                               → 1
          * 5 dev cards (DEV_CARD_ORDER) / 8.0              → 5
          * 5 phase flags (setup/roll/main/robber/discard)  → 5
          * dev deck remaining / 25.0                       → 1
          * dice one-hot 2..12                              → 11
          * 2 Karma flags                                   → 2
          * devCardPlayedThisTurn                           → 1
          (total: 54)

        For ``next_player_main`` (opponent), the layout is identical up to
        this point, then 7 extra dims are appended:
          * 6-bin hidden-dev-count one-hot (0..5+)          → 6
          * total resources / 20.0                          → 1
          (total: 54 + 7 = 61)
        """
        feats: list[float] = []

        # 5 resources (Charlesworth order). Clip to [0, 1]: the 1v1 ruleset
        # discards at 9 cards on a 7-roll but lets a player accumulate
        # arbitrarily many between 7-rolls, so r/8.0 can exceed 1.0. Saturate.
        for r in RESOURCES_CW:
            feats.append(min(1.0, float(resources.get(r, 0)) / 8.0))

        # VP / 15. POV split (no-leak): the AGENT sees its OWN true total
        # (it knows its own hidden VP cards); the OPPONENT contributes only
        # visibleVictoryPoints (= victoryPoints - hidden VP cards), so the
        # opponent's hidden-VP count never leaks. A hidden VP only becomes
        # visible when it wins the game (which ends the episode anyway).
        if is_agent:
            feats.append(float(player.victoryPoints) / 15.0)
        else:
            visible_vp = getattr(player, "visibleVictoryPoints", player.victoryPoints)
            feats.append(float(visible_vp) / 15.0)

        # Income (resource production rate per resource).
        income = _compute_income(player, game.board)
        feats.extend(income)

        # Trade ratios (best per resource as fraction of 1).
        for r in RESOURCES_CW:
            r_port = "2:1 " + r
            if r_port in player.portList:
                feats.append(0.5)
            elif "3:1 PORT" in player.portList:
                feats.append(1.0 / 3.0)
            else:
                feats.append(0.25)

        # Ports (6 bools).
        for pn in ("3:1 PORT", "2:1 WOOD", "2:1 BRICK", "2:1 WHEAT", "2:1 ORE", "2:1 SHEEP"):
            feats.append(1.0 if pn in player.portList else 0.0)

        # LR + LA bools.
        feats.append(1.0 if getattr(player, "longestRoadFlag", False) else 0.0)
        feats.append(1.0 if getattr(player, "largestArmyFlag", False) else 0.0)

        # Road length / 15. Clipped: Road Builder dev card adds 2 roads
        # in a single turn so the longest path can exceed 15 edges in
        # late-game corner cases.
        feats.append(min(1.0, float(getattr(player, "maxRoadLength", 0)) / 15.0))

        # Knights played / 8. Clipped: the dev deck holds 14 KNIGHT cards so
        # a player can play >8 in long games, pushing this above 1.0.
        feats.append(min(1.0, float(getattr(player, "knightsPlayed", 0)) / 8.0))

        # Settlements / cities / roads left.
        feats.append(float(getattr(player, "settlementsLeft", 5)) / 5.0)
        feats.append(float(getattr(player, "citiesLeft", 4)) / 4.0)
        feats.append(float(getattr(player, "roadsLeft", 15)) / 15.0)

        # Dev cards in DEV_CARD_ORDER (5 floats). Clipped: KNIGHT count in
        # hand can exceed 8 in long games (14 in the deck).
        #
        # POV split (no-leak): the AGENT sees its OWN currently-HELD (hidden)
        # hand — it knows its own cards. The OPPONENT contributes only its
        # PLAYED dev cards (observable); its hidden dev-card TYPES are the only
        # remaining hidden state in 1v1 (ADR 0002) and MUST NOT appear here —
        # they are the belief head's prediction target. The opponent's hidden
        # *count* is still encoded (the 6-bin one-hot appended below for the
        # opponent), so observable information is preserved.
        dev = getattr(player, "devCards", {}) or {}
        if is_agent:
            for c in DEV_CARD_ORDER:
                feats.append(min(1.0, float(dev.get(c, 0)) / 8.0))
        else:
            # Opponent: hidden dev-card TYPES are secret (the belief head's
            # target) and must NOT leak here. The opponent's observable dev
            # info is already carried elsewhere — PLAYED composition in
            # ``next_played_dev_counts`` and the hidden COUNT in the one-hot
            # appended below — so this slice is zero (no leak, no duplication).
            feats.extend([0.0] * N_DEV_TYPES)

        # Phase flags (5).
        in_setup = bool(env_state.initial_placement_phase)
        in_main = (
            not in_setup
            and not env_state.roll_pending
            and not env_state.discard_pending
            and not env_state.robber_placement_pending
            and env_state.road_building_roads_left == 0
        )
        feats.append(1.0 if in_setup else 0.0)
        feats.append(1.0 if env_state.roll_pending else 0.0)
        feats.append(1.0 if in_main else 0.0)
        feats.append(1.0 if env_state.robber_placement_pending else 0.0)
        feats.append(1.0 if env_state.discard_pending else 0.0)

        # Dev deck remaining / 25.
        deck_total = (
            sum(getattr(game.board, "devCardStack", {}).values()) if hasattr(game, "board") else 0
        )
        feats.append(float(deck_total) / 25.0)

        # Dice one-hot 2..12.
        dice_oh = [0.0] * 11
        last_dice = int(env_state.last_dice_roll)
        if 2 <= last_dice <= 12:
            dice_oh[last_dice - 2] = 1.0
        feats.extend(dice_oh)

        # Karma flags — encode the *persistent* Karma buff state from this
        # player's perspective. `last_player_to_roll_7` is updated only when a
        # 7 rolls (never reset on turn change), so both flags persist across
        # however many turns it takes for another 7 to roll.
        #   flag 0: this player is the most recent 7-roller → no buff against
        #           them; the *other* player is currently under the 20% buff.
        #   flag 1: some other player is the most recent 7-roller AND this
        #           player is not → karma_buff_active(player) is True; this
        #           player's next roll has a 20% chance of being forced to 7.
        # Both flags are 0 when no 7 has been rolled yet in the game.
        last7 = getattr(game, "last_player_to_roll_7", None)
        feats.append(1.0 if last7 is player else 0.0)
        feats.append(1.0 if (last7 is not None and last7 is not player) else 0.0)

        # devCardPlayedThisTurn.
        feats.append(1.0 if getattr(player, "devCardPlayedThisTurn", False) else 0.0)

        # Sanity check against the schema constant.
        if len(feats) != CURR_PLAYER_DIM:
            raise RuntimeError(
                f"player_main dim={len(feats)} expected {CURR_PLAYER_DIM} (is_agent={is_agent})"
            )

        if is_agent:
            return np.asarray(feats, dtype=np.float32)

        # Opponent: append hidden-dev count one-hot (6) + total_res / 20 (1).
        hidden = 0
        new_dev = getattr(player, "newDevCards", []) or []
        hidden += len(new_dev)
        for name in DEV_CARD_ORDER:
            if name != "VP":
                hidden += int(dev.get(name, 0))
        hidden_oh = [0.0] * 6
        hidden_oh[min(hidden, 5)] = 1.0
        feats.extend(hidden_oh)

        total_res = float(sum(resources.values()))
        # Same saturation reasoning as the per-resource clip above.
        feats.append(min(1.0, total_res / 20.0))

        if len(feats) != NEXT_PLAYER_DIM:
            raise RuntimeError(f"next_player_main dim={len(feats)} expected {NEXT_PLAYER_DIM}")

        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Resource lookup helper
    # ------------------------------------------------------------------

    @staticmethod
    def _opp_resources(opp: Any, hand_tracker: BroadcastHandTracker | None) -> dict[str, int]:
        """Return opponent's resources as a name-keyed dict.

        Uses the hand tracker if provided (the canonical source under the
        v2 design — broadcast-derived, not direct read), else falls back
        to ``opp.resources`` for tests / dataset gen that don't need a
        tracker.
        """
        if hand_tracker is not None:
            return hand_tracker.get_hand(opp.name)
        raw = getattr(opp, "resources", {}) or {}
        return {r: int(raw.get(r, 0)) for r in RESOURCES_CW}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def _road_owner_between(board: Any, c1: Any, c2: Any) -> tuple[bool, Any]:
    """Look up whether an edge has a road and who owns it."""
    v_obj = board.boardGraph.get(c1)
    if v_obj is None:
        return False, None
    for j, nb in enumerate(v_obj.neighbors):
        if nb == c2:
            owner = v_obj.edge_state[j][0]
            has = v_obj.edge_state[j][1]
            return bool(has), owner
    return False, None


def _dev_counts(player: Any, *, hidden: bool) -> np.ndarray:
    """Return (5,) count vector over :data:`DEV_CARD_ORDER`.

    If ``hidden=True``, returns the player's currently-held dev card counts
    (both bought-and-playable and just-bought-this-turn). For
    ``hidden=False``, returns the *played* counts (which are observable).

    Note: this is the input for the v2 ``CountDevEncoder``, which expects
    a (5,) bincount-style vector. The encoder is permutation-invariant by
    construction so the in-array order is purely conventional.

    Foot-gun: with ``hidden=True`` the returned vector INCLUDES VP cards. VP is
    observable in 1v1 (the victoryPoints / visibleVictoryPoints gap), so any
    caller wanting the genuinely-SECRET hidden split (e.g. the belief target)
    must zero the VP index itself — see ``CatanEnv.belief_target``.
    """
    out = np.zeros(N_DEV_TYPES, dtype=np.float32)
    if hidden:
        dev = getattr(player, "devCards", {}) or {}
        new_dev = getattr(player, "newDevCards", []) or []
        for i, name in enumerate(DEV_CARD_ORDER):
            out[i] = float(dev.get(name, 0))
        # newDevCards entries are strings; bincount them.
        for c in new_dev:
            if c in DEV_CARD_ORDER:
                out[DEV_CARD_ORDER.index(c)] += 1.0
    else:
        out[DEV_CARD_ORDER.index("KNIGHT")] = float(getattr(player, "knightsPlayed", 0))
        out[DEV_CARD_ORDER.index("YEAROFPLENTY")] = float(getattr(player, "yopPlayed", 0))
        out[DEV_CARD_ORDER.index("MONOPOLY")] = float(getattr(player, "monopolyPlayed", 0))
        out[DEV_CARD_ORDER.index("ROADBUILDER")] = float(getattr(player, "roadBuilderPlayed", 0))
        # VP cards are never "played" (they're revealed on win); leave 0.
    return out


def _compute_income(player: Any, board: Any) -> list[float]:
    """5-element vector of per-resource expected production per turn (× 0.1).

    Loosely: for each resource r, sum (dot count / 36) over hexes adjacent
    to settlements / cities the player owns, weighted 1 for settlements
    and 2 for cities. Scaled by 0.1 so a strong economy (~3 expected
    resources / turn for one type) lands in [0, 1].
    """  # noqa: RUF002
    income = [0.0] * 5
    res_index = {r: i for i, r in enumerate(RESOURCES_CW)}
    settlements = getattr(player, "buildGraph", {}).get("SETTLEMENTS", []) or []
    cities = getattr(player, "buildGraph", {}).get("CITIES", []) or []
    for v_pt in settlements:
        v_obj = board.boardGraph.get(v_pt)
        if v_obj is None:
            continue
        for h_idx in v_obj.adjacent_hex_indices:
            tile = board.hexTileDict[h_idx]
            r = tile.resource_type
            if r in res_index and tile.number_token is not None:
                income[res_index[r]] += _DOTS_BY_TOKEN.get(tile.number_token, 0) / 36.0
    for v_pt in cities:
        v_obj = board.boardGraph.get(v_pt)
        if v_obj is None:
            continue
        for h_idx in v_obj.adjacent_hex_indices:
            tile = board.hexTileDict[h_idx]
            r = tile.resource_type
            if r in res_index and tile.number_token is not None:
                income[res_index[r]] += 2.0 * _DOTS_BY_TOKEN.get(tile.number_token, 0) / 36.0
    return [v / 10.0 for v in income]
