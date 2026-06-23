"""Snake-draft scenario generator for labeling (plan §A).

Drives the engine through the four-pick 1v1 snake draft, yielding one
``Scenario`` per user-decision point. The caller (the UI) reads the
current scenario, prompts the user for a settlement-vertex + road-edge,
and calls ``apply(...)`` to advance to the next pick.

Design notes:
- API is a **stateful class**, not a Python ``Iterator``. The plan §A
  draft mentioned ``Iterator[Scenario]`` but iterators don't compose
  with user-driven choice injection without ``.send(...)`` gymnastics.
  A class is clearer for UI code.
- Determinism: ``ScenarioGenerator(seed=k)`` produces a bit-identical
  board (resource shuffle, token shuffle, port shuffle) on every
  invocation. Pinned by ``test_scenario_gen.py::test_same_seed_*``.
- Both sides of the snake draft are labeled by the same user (plan §2:
  "user labels both sides"). ``acting_player_idx`` tells the UI which
  side they are playing this turn.
- Snake-draft order in 1v1: ``[P1, P2, P2, P1]``. Verified against
  ``engine.game.catanGame.build_initial_settlements`` (forward pass +
  reverse pass).
- ``Scenario.legal_settlement_corners`` is computed up front; the road
  mask depends on which settlement was chosen, so it is exposed via
  ``Scenario.compute_legal_road_edges(settlement_idx)``.
"""

from __future__ import annotations

import queue
import random
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from catan_rl.engine.board import catanBoard
from catan_rl.engine.game import catanGame
from catan_rl.engine.player import player as Player
from catan_rl.env.masks import compute_action_masks
from catan_rl.policy.obs_encoder import EnvObsState

# Snake-draft order: P1, P2, P2, P1 (0-indexed: 0, 1, 1, 0).
_SNAKE_ORDER: tuple[int, ...] = (0, 1, 1, 0)
_SETUP_STEPS_SETTLE: tuple[int, ...] = (0, 0, 2, 2)
_SETUP_STEPS_ROAD: tuple[int, ...] = (1, 1, 3, 3)


@dataclass
class Pick:
    """A single completed pick (settlement + road) by one player."""

    player: int  # 0 (P1) or 1 (P2)
    settlement_vertex: int  # 0..53
    road_edge: int  # 0..71

    def to_dict(self) -> dict[str, int]:
        return {
            "player": self.player,
            "settlement_vertex": self.settlement_vertex,
            "road_edge": self.road_edge,
        }

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> Pick:
        return cls(
            player=int(d["player"]),
            settlement_vertex=int(d["settlement_vertex"]),
            road_edge=int(d["road_edge"]),
        )


@dataclass
class Scenario:
    """A single user-decision point in the snake draft."""

    scenario_id: str
    game_seed: int
    draft_position: int  # 1..4
    acting_player_idx: int  # 0 or 1
    prior_picks: list[Pick]
    legal_settlement_corners: np.ndarray  # bool (54,)
    # Live engine handle for the UI to render (NOT serialised).
    game: Any
    # Internal — needed to compute the post-settlement road mask.
    _acting_player: Any
    _setup_step_road: int
    _vertex_to_idx: dict[Any, int]
    _edge_to_idx: dict[tuple[str, str], int]
    _idx_to_vertex_pixel: dict[int, Any]
    _idx_to_edge_pixel_pair: dict[int, tuple[Any, Any]]

    def compute_legal_road_edges(self, settlement_vertex_idx: int) -> np.ndarray:
        """Return the (72,) bool mask of legal road edges *after*
        placing a settlement at the given vertex index.

        Implementation: temporarily applies the settlement to the engine,
        calls ``compute_action_masks`` in setup_step ∈ {1, 3}, then
        reverts the engine state. The revert restores every mutable
        attribute touched by ``player.build_settlement`` + the engine's
        ``updateBoardGraph_settlement``: vertex owner / building_type,
        player buildGraph / settlementsLeft / victoryPoints / portList.

        Build_settlement with is_free=True emits no broadcast event,
        so no event-bus rollback is needed.
        """
        if not (0 <= settlement_vertex_idx < 54):
            raise ValueError(f"settlement_vertex_idx out of range: {settlement_vertex_idx}")
        if not bool(self.legal_settlement_corners[settlement_vertex_idx]):
            raise ValueError(f"settlement_vertex_idx {settlement_vertex_idx} is not legal")
        v_px = self._idx_to_vertex_pixel[settlement_vertex_idx]
        board = self.game.board
        vertex_obj = board.boardGraph[v_px]
        acting = self._acting_player

        # Snapshot mutable state.
        prev_owner = getattr(vertex_obj, "owner", None)
        prev_building_type = getattr(vertex_obj, "building_type", None)
        prev_settlements = list(acting.buildGraph["SETTLEMENTS"])
        prev_settlementsLeft = acting.settlementsLeft
        prev_vp = acting.victoryPoints
        prev_portList = list(acting.portList)

        try:
            acting.build_settlement(v_px, board, is_free=True)
            env_state = EnvObsState(
                initial_placement_phase=True,
                setup_step=self._setup_step_road,
            )
            masks = compute_action_masks(
                self.game,
                acting,
                env_state,
                self._vertex_to_idx,
                self._edge_to_idx,
            )
            return np.asarray(masks["edge"], dtype=bool)
        finally:
            # Revert in reverse order of mutation.
            vertex_obj.owner = prev_owner
            vertex_obj.building_type = prev_building_type
            acting.buildGraph["SETTLEMENTS"] = prev_settlements
            acting.settlementsLeft = prev_settlementsLeft
            acting.victoryPoints = prev_vp
            acting.portList = prev_portList


class ScenarioGenerator:
    """Stateful snake-draft state machine for the labeling UI.

    Usage::

        gen = ScenarioGenerator(seed=42)
        while (scenario := gen.current()) is not None:
            # render scenario.legal_settlement_corners
            ... user picks settlement_idx ...
            edge_mask = scenario.compute_legal_road_edges(settlement_idx)
            ... user picks road_idx ...
            gen.apply(settlement_idx, road_idx)
    """

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._done = False
        self._draft_step = 0  # 0..3 (yields draft_position 1..4)
        self._prior_picks: list[Pick] = []

        # Engine state: deterministic via `random.seed` + `np.random.seed`.
        random.seed(self._seed)
        np.random.seed(self._seed)
        self._board = catanBoard()
        self._game = catanGame(render_mode=None)
        # Replace the game's auto-generated board with our seeded one.
        self._game.board = self._board

        # Two human players (we drive their picks via build_settlement/build_road).
        # Names match the BC dataset convention (P1 / P2).
        self._players = [Player("P1", "black"), Player("P2", "darkslateblue")]
        for p in self._players:
            p.game = self._game

        # Wire up the game's player queue (needed by some engine code paths).
        self._game.playerQueue = queue.Queue(2)
        for p in self._players:
            self._game.playerQueue.put(p)
        self._game.gameSetup = True
        self._game.currentPlayer = self._players[0]

        # Build index maps (canonical ordering, matches BC dataset).
        self._vertex_to_idx, self._edge_to_idx = _build_index_maps(self._board)
        self._idx_to_vertex_pixel: dict[int, Any] = {
            idx: px for px, idx in self._vertex_to_idx.items()
        }
        self._idx_to_edge_pixel_pair: dict[int, tuple[Any, Any]] = _build_idx_to_edge_pixel_pair(
            self._board, self._edge_to_idx
        )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def current(self) -> Scenario | None:
        """Return the current scenario or None if all 4 picks are done."""
        if self._done:
            return None

        acting_idx = _SNAKE_ORDER[self._draft_step]
        acting = self._players[acting_idx]
        self._game.currentPlayer = acting
        setup_step_settle = _SETUP_STEPS_SETTLE[self._draft_step]
        setup_step_road = _SETUP_STEPS_ROAD[self._draft_step]
        env_state = EnvObsState(
            initial_placement_phase=True,
            setup_step=setup_step_settle,
        )
        masks = compute_action_masks(
            self._game, acting, env_state, self._vertex_to_idx, self._edge_to_idx
        )
        legal_corners = np.asarray(masks["corner_settlement"], dtype=bool)

        return Scenario(
            scenario_id=str(uuid.uuid4()),
            game_seed=self._seed,
            draft_position=self._draft_step + 1,
            acting_player_idx=acting_idx,
            prior_picks=list(self._prior_picks),
            legal_settlement_corners=legal_corners,
            game=self._game,
            _acting_player=acting,
            _setup_step_road=setup_step_road,
            _vertex_to_idx=self._vertex_to_idx,
            _edge_to_idx=self._edge_to_idx,
            _idx_to_vertex_pixel=self._idx_to_vertex_pixel,
            _idx_to_edge_pixel_pair=self._idx_to_edge_pixel_pair,
        )

    def apply(self, settlement_vertex_idx: int, road_edge_idx: int) -> None:
        """Advance the state machine by one snake-draft position.

        Validates legality before applying. Grants initial resources
        for reverse-pass settlements (picks 3 and 4 of the snake draft).
        """
        if self._done:
            raise RuntimeError("scenario generator is done (all 4 picks applied)")

        scenario = self.current()
        if scenario is None:  # Defensive — should not reach here if not done.
            raise RuntimeError("no current scenario to apply")

        if not (0 <= settlement_vertex_idx < 54):
            raise ValueError(f"settlement_vertex_idx out of range: {settlement_vertex_idx}")
        if not bool(scenario.legal_settlement_corners[settlement_vertex_idx]):
            raise ValueError(
                f"illegal settlement vertex {settlement_vertex_idx} at "
                f"draft position {scenario.draft_position}"
            )

        # Apply settlement.
        v_px = self._idx_to_vertex_pixel[settlement_vertex_idx]
        acting = scenario._acting_player
        acting.build_settlement(v_px, self._board, is_free=True)

        # Validate road legality against the post-settlement state.
        env_state = EnvObsState(
            initial_placement_phase=True,
            setup_step=scenario._setup_step_road,
        )
        road_masks = compute_action_masks(
            self._game, acting, env_state, self._vertex_to_idx, self._edge_to_idx
        )
        legal_roads = np.asarray(road_masks["edge"], dtype=bool)
        if not (0 <= road_edge_idx < 72):
            raise ValueError(f"road_edge_idx out of range: {road_edge_idx}")
        if not bool(legal_roads[road_edge_idx]):
            raise ValueError(
                f"illegal road edge {road_edge_idx} at draft position {scenario.draft_position}"
            )

        # Apply road.
        v1, v2 = self._idx_to_edge_pixel_pair[road_edge_idx]
        acting.build_road(v1, v2, self._board, is_free=True)

        # Grant initial resources for reverse-pass picks (snake positions
        # 3 and 4 = draft positions 3 and 4 in 1v1 snake [P1, P2, P2, P1]).
        if self._draft_step >= 2:
            _grant_setup_resources(self._game, acting)

        # Record + advance.
        self._prior_picks.append(
            Pick(
                player=scenario.acting_player_idx,
                settlement_vertex=settlement_vertex_idx,
                road_edge=road_edge_idx,
            )
        )
        self._draft_step += 1
        if self._draft_step >= 4:
            self._done = True


# ---------------------------------------------------------------------------
# Helpers (replicas of bc/dataset.py private helpers — kept local to avoid
# importing the BC package from the labeling package).
# ---------------------------------------------------------------------------


def _build_index_maps(
    board: catanBoard,
) -> tuple[dict[Any, int], dict[tuple[str, str], int]]:
    """Same canonical ordering as ``CatanEnv._build_index_maps`` and
    ``bc.dataset._build_index_maps``."""
    vertex_to_idx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
    seen: set[tuple[str, str]] = set()
    edge_to_idx: dict[tuple[str, str], int] = {}
    for v_pt, v_obj in board.boardGraph.items():
        for nb_pt in v_obj.neighbors:
            s1, s2 = str(v_pt), str(nb_pt)
            key = (s1, s2) if s1 < s2 else (s2, s1)
            if key not in seen:
                seen.add(key)
                edge_to_idx[key] = len(edge_to_idx)
    return vertex_to_idx, edge_to_idx


def _build_idx_to_edge_pixel_pair(
    board: catanBoard,
    edge_to_idx: dict[tuple[str, str], int],
) -> dict[int, tuple[Any, Any]]:
    """Inverse of ``edge_to_idx``: idx → (v1_pixel, v2_pixel)."""
    out: dict[int, tuple[Any, Any]] = {}
    for v_pt, v_obj in board.boardGraph.items():
        for nb_pt in v_obj.neighbors:
            s1, s2 = str(v_pt), str(nb_pt)
            key = (s1, s2) if s1 < s2 else (s2, s1)
            idx = edge_to_idx.get(key)
            if idx is not None and idx not in out:
                # Order matches the lexicographic key.
                out[idx] = (v_pt, nb_pt) if s1 < s2 else (nb_pt, v_pt)
    return out


def _grant_setup_resources(game: catanGame, p: Any) -> None:
    """Grant the initial resources for a reverse-pass settlement.

    Identical to ``bc.dataset._grant_setup_resources``; replicated here
    to avoid a labeling → bc import.
    """
    if not p.buildGraph["SETTLEMENTS"]:
        return
    last = p.buildGraph["SETTLEMENTS"][-1]
    for adj_hex in game.board.boardGraph[last].adjacent_hex_indices:
        res = game.board.hexTileDict[adj_hex].resource_type
        if res != "DESERT":
            p.resources[res] = p.resources.get(res, 0) + 1
            game.board.bank_draw({res: 1})  # spec 009: setup grant draws from bank
            if hasattr(game, "broadcast"):
                game.broadcast.resource_change(p.name, {res: 1}, "SETUP")
