"""Theory-shaped setup-phase features for the fitted scorer (spec D1).

Pure, vehicle-neutral (D7): a board plus the picks made so far in, a float
matrix out. No ``bc``, no ``gui``, no checkpoint, no policy forward pass — so
the same feature block feeds either downstream vehicle (synthetic-corpus
fine-tune, or setup-node search priors) without dragging a training stack along.
(The shared schema constants come from ``catan_rl.policy.obs_schema`` /
``obs_encoder``, whose package ``__init__`` pulls torch transitively — the same
route ``analytic_value`` already took. Nothing here calls a network.)

Feature design follows the owner's opening theory as captured 2026-08-20:

* **The port penalty is a confound.** A port corner is usually a 1-2-hex corner,
  so a raw "has port" flag reads as "less production" and the fit learns to
  avoid ports for the wrong reason. The confound is removed by carrying
  ``n_adjacent_hexes`` and ``n_hexes_x_second`` alongside the port flags, so the
  hex-count effect has its own weights to live in and the port flags carry only
  the residual. The conditional part of port value is spelled separately as
  ``port_2to1_matched`` (a 2:1 port for a resource this seat actually produces).
* **Denial is relational.** ``opponent_new_resources``, ``opponent_best_margin``,
  ``adjacency_block`` and ``scarcity_starve`` describe the candidate in terms of
  what it does to the OTHER seat, which is the only place picks 2-4 logic can be
  expressed at all.

**Circularity break.** The relational features need "the opponent's best
remaining vertex". That must NOT be the scorer being fit, or the features would
be a function of the weights they are used to estimate. It is instead computed
from a PINNED base: :func:`catan_rl.setup_phase.analytic_value.vertex_yield`
under :data:`~catan_rl.setup_phase.resource_weights.CHARLESWORTH_V0`. The base is
a constant function of the board, so the design matrix is fixed before the fit
starts. Pinned by fixture.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from catan_rl.labeling.scenario_gen import Pick, build_index_maps
from catan_rl.policy.obs_encoder import DOTS_BY_TOKEN
from catan_rl.policy.obs_schema import RESOURCES_CW
from catan_rl.setup_phase.analytic_value import vertex_yield
from catan_rl.setup_phase.resource_weights import CHARLESWORTH_V0

N_VERTICES = 54
N_EDGES = 72
N_HEXES = 19

FEATURE_VERSION: str = "v1"
"""Stamped into every fitted artifact. A scorer whose ``feature_version`` does
not match this constant REFUSES to load (see
:func:`catan_rl.setup_phase.scorer.load_weights`): silently scoring an old
weight vector against a re-ordered design matrix is a wrong number that looks
like a right one."""

SETTLEMENT_FEATURE_NAMES: tuple[str, ...] = (
    "pips_wood",
    "pips_brick",
    "pips_wheat",
    "pips_ore",
    "pips_sheep",
    "pips_total",
    "n_distinct_resources",
    "n_new_resources",
    "n_adjacent_hexes",
    "n_hexes_x_second",
    "port_any",
    "port_2to1_matched",
    "port_3to1",
    "expansion_value",
    "opponent_new_resources",
    "opponent_best_margin",
    "adjacency_block",
    "scarcity_starve",
)
"""D1's settlement feature list, in design-matrix column order.

D1 asks for "opponent value of the candidate (their pips + their missing
resources)". The *pips* half is byte-for-byte ``pips_total`` — a vertex yields
the same dots whoever settles it — so carrying it twice would be an exactly
collinear column that makes the fit's weights unidentifiable while adding
nothing. Only the missing-resources half gets its own column.
"""

ROAD_FEATURE_NAMES: tuple[str, ...] = (
    "opens_best_vertex_value",
    "blocks_opponent_target",
    "toward_port",
)
"""D1's road feature list. The road model is FIT over these; the
"point at the expansion target" rule is reported as a null baseline it must
beat, never asserted (see :mod:`catan_rl.setup_phase.fit`)."""

PILOT_FEATURE_NAMES: tuple[str, ...] = (
    "pips_wood",
    "pips_brick",
    "pips_wheat",
    "pips_ore",
    "pips_sheep",
    "pips_total",
    "n_distinct_resources",
    "n_new_resources",
    "opponent_new_resources",
    "port_any",
)
"""The 2026-08-15 pilot's 10 features, as a SUBSET of the current list.

Acceptance criterion 3 (regression continuity) refits exactly these on the
pilot's 168-label split and reads the result against the pilot's reported
34.1% held-out / 45.2% train. Expressing the pilot as a subset — rather than a
second feature function — is what makes that comparison mean anything."""

N_SETTLEMENT_FEATURES: int = len(SETTLEMENT_FEATURE_NAMES)
N_ROAD_FEATURES: int = len(ROAD_FEATURE_NAMES)

_BASE_WEIGHTS: Mapping[str, float] = CHARLESWORTH_V0
"""The PINNED resource weighting behind every "value of a vertex" the relational
features reference. Deliberately not configurable: a caller who could swap it
could make the design matrix depend on a tuned quantity again."""


# ---------------------------------------------------------------------------
# Board adapters
# ---------------------------------------------------------------------------
def edge_vertex_pairs(board: Any) -> dict[int, tuple[int, int]]:
    """``edge_index -> (vertex_index, vertex_index)`` in the canonical order.

    Derived from :func:`catan_rl.labeling.scenario_gen.build_index_maps`, which
    is the same ordering the labels were written in.
    """
    vertex_to_idx, edge_to_idx = build_index_maps(board)
    by_str = {str(px): idx for px, idx in vertex_to_idx.items()}
    return {idx: (by_str[a], by_str[b]) for (a, b), idx in edge_to_idx.items()}


def _vertex_object(board: Any, vertex_idx: int) -> Any:
    return board.boardGraph[board.vertex_index_to_pixel_dict[vertex_idx]]


def vertex_resource_pips(board: Any) -> np.ndarray:
    """``(54, 5)`` dot-count per vertex per resource, in Charlesworth order."""
    out = np.zeros((N_VERTICES, len(RESOURCES_CW)), dtype=np.float64)
    res_col = {r: i for i, r in enumerate(RESOURCES_CW)}
    for v in range(N_VERTICES):
        for h_idx in _vertex_object(board, v).adjacent_hex_indices:
            tile = board.hexTileDict[h_idx]
            col = res_col.get(str(tile.resource_type))
            if col is None:  # DESERT
                continue
            out[v, col] += float(DOTS_BY_TOKEN.get(tile.number_token, 0))
    return out


def board_resource_pips(board: Any) -> np.ndarray:
    """``(5,)`` total dot-count on the board per resource (scarcity source)."""
    out = np.zeros(len(RESOURCES_CW), dtype=np.float64)
    res_col = {r: i for i, r in enumerate(RESOURCES_CW)}
    for h_idx in range(N_HEXES):
        tile = board.hexTileDict[h_idx]
        col = res_col.get(str(tile.resource_type))
        if col is None:
            continue
        out[col] += float(DOTS_BY_TOKEN.get(tile.number_token, 0))
    return out


def vertex_adjacency(board: Any) -> list[list[int]]:
    """``vertex_index -> [neighbouring vertex_index, ...]``."""
    return [
        sorted(board.boardGraph[px].vertex_index for px in _vertex_object(board, v).neighbors)
        for v in range(N_VERTICES)
    ]


def _hop_distances(adjacency: Sequence[Sequence[int]]) -> np.ndarray:
    """All-pairs road-hop distance over the vertex graph (``(54, 54)`` int)."""
    n = len(adjacency)
    dist = np.full((n, n), -1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adjacency[cur]:
                if dist[src, nb] == -1:
                    dist[src, nb] = dist[src, cur] + 1
                    q.append(nb)
    return dist


def _port_of(board: Any, vertex_idx: int) -> str | None:
    port = getattr(_vertex_object(board, vertex_idx), "port", False)
    return str(port) if port else None


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SetupContext:
    """Everything a candidate's features are computed against, precomputed once.

    Built per decision point (board + prior picks + acting seat + the legal
    settlement mask). Holding it explicitly is what lets the relational features
    reference the opponent's best remaining spot without recomputing a 54-vertex
    scan per candidate.
    """

    board: Any
    acting_player: int
    legal_settlements: np.ndarray
    pips: np.ndarray
    base_value: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]
    distances: np.ndarray
    own_vertices: tuple[int, ...]
    opp_vertices: tuple[int, ...]
    own_pips: np.ndarray
    opp_pips: np.ndarray
    scarce_mask: np.ndarray
    opponent_best_vertex: int | None
    is_second_settlement: bool
    edge_vertices: dict[int, tuple[int, int]]

    @classmethod
    def build(
        cls,
        board: Any,
        prior_picks: Sequence[Pick],
        acting_player: int,
        legal_settlements: np.ndarray,
    ) -> SetupContext:
        legal = np.asarray(legal_settlements, dtype=bool)
        if legal.shape != (N_VERTICES,):
            raise ValueError(f"legal_settlements must be shape (54,), got {legal.shape}")
        if acting_player not in (0, 1):
            raise ValueError(f"acting_player must be 0 or 1, got {acting_player}")

        pips = vertex_resource_pips(board)
        base_value = np.asarray(
            [vertex_yield(board, v, _BASE_WEIGHTS) for v in range(N_VERTICES)],
            dtype=np.float64,
        )
        adjacency = vertex_adjacency(board)
        distances = _hop_distances(adjacency)

        own = tuple(p.settlement_vertex for p in prior_picks if p.player == acting_player)
        opp = tuple(p.settlement_vertex for p in prior_picks if p.player != acting_player)
        own_pips = pips[list(own)].sum(axis=0) if own else np.zeros(len(RESOURCES_CW))
        opp_pips = pips[list(opp)].sum(axis=0) if opp else np.zeros(len(RESOURCES_CW))

        board_pips = board_resource_pips(board)
        scarce = board_pips == board_pips.min()

        legal_idx = np.flatnonzero(legal)
        opponent_best: int | None = (
            int(legal_idx[int(np.argmax(base_value[legal_idx]))]) if legal_idx.size else None
        )

        return cls(
            board=board,
            acting_player=acting_player,
            legal_settlements=legal,
            pips=pips,
            base_value=base_value,
            adjacency=tuple(tuple(nb) for nb in adjacency),
            distances=distances,
            own_vertices=own,
            opp_vertices=opp,
            own_pips=own_pips,
            opp_pips=opp_pips,
            scarce_mask=scarce,
            opponent_best_vertex=opponent_best,
            is_second_settlement=len(own) >= 1,
            edge_vertices=edge_vertex_pairs(board),
        )


# ---------------------------------------------------------------------------
# Settlement features
# ---------------------------------------------------------------------------
def settlement_features(ctx: SetupContext, vertex: int) -> np.ndarray:
    """The ``(18,)`` feature row for one candidate settlement vertex."""
    if not 0 <= vertex < N_VERTICES:
        raise ValueError(f"vertex out of range: {vertex}")
    pips = ctx.pips[vertex]
    n_hexes = float(len(_vertex_object(ctx.board, vertex).adjacent_hex_indices))
    produces = pips > 0.0
    new_for_me = float(np.count_nonzero(produces & (ctx.own_pips <= 0.0)))
    new_for_opp = float(np.count_nonzero(produces & (ctx.opp_pips <= 0.0)))

    port = _port_of(ctx.board, vertex)
    port_any = 1.0 if port else 0.0
    port_3to1 = 1.0 if port == "3:1 PORT" else 0.0
    port_matched = 0.0
    if port is not None and port.startswith("2:1 "):
        res = port.split(" ", 1)[1]
        if res in RESOURCES_CW:
            col = RESOURCES_CW.index(res)
            port_matched = 1.0 if (pips[col] > 0.0 or ctx.own_pips[col] > 0.0) else 0.0

    # ``hops=(2,)`` is D1's "within road distance 1-2" with the redundant half
    # dropped, not a narrowing: ``exclude=(vertex,)`` already forbids every
    # distance-1 vertex (the placement's own neighbours are un-settleable), so
    # hop 1 can contribute no candidate. Written explicitly to stop a future
    # reader "restoring" a hop that would only ever be empty.
    #
    # DEVIATION FROM D1, recorded rather than silently taken: the spec words
    # this feature "own-yield-scored", and it is scored with ``ctx.base_value``,
    # the seat-NEUTRAL pinned ``vertex_yield`` — so ``expansion_value`` is
    # identical for either seat at the same decision point and cannot express
    # the owner's stated theory that expansion is about "building to" a missing
    # resource. That is a REAL gap in the 18-column block (the relational half of
    # the theory got its columns; the own-need half did not) and it is not
    # justified by the circularity break, which only requires the OPPONENT-facing
    # reference to be pinned. It is left as-is here because changing the feature
    # arithmetic moves ``FEATURE_VERSION``, every hand-computed fixture and the
    # banked pilot regression numbers that acceptance criterion 3 pins — a
    # scope the owner has to price, not a fix to slip into a review pass. An
    # own-need-weighted expansion term is the first candidate for the next
    # refit.
    expansion = _best_value_at_hops(ctx, origin=vertex, hops=(2,), exclude=(vertex,))

    if ctx.opponent_best_vertex is None:
        opp_margin = 0.0
        block = 0.0
    else:
        opp_margin = float(ctx.base_value[vertex] - ctx.base_value[ctx.opponent_best_vertex])
        block = 1.0 if int(ctx.distances[vertex, ctx.opponent_best_vertex]) == 1 else 0.0

    starve = 1.0 if bool(np.any(produces & ctx.scarce_mask & (ctx.opp_pips <= 0.0))) else 0.0

    return np.asarray(
        [
            pips[0],
            pips[1],
            pips[2],
            pips[3],
            pips[4],
            float(pips.sum()),
            float(np.count_nonzero(produces)),
            new_for_me,
            n_hexes,
            n_hexes if ctx.is_second_settlement else 0.0,
            port_any,
            port_matched,
            port_3to1,
            expansion,
            new_for_opp,
            opp_margin,
            block,
            starve,
        ],
        dtype=np.float64,
    )


def all_settlement_features(ctx: SetupContext) -> np.ndarray:
    """``(54, 18)``. Rows for ILLEGAL vertices are all-zero and must never be
    read — :func:`catan_rl.setup_phase.scorer.score_vertices` masks them to
    ``-inf`` rather than letting a zero row compete."""
    out = np.zeros((N_VERTICES, N_SETTLEMENT_FEATURES), dtype=np.float64)
    for v in np.flatnonzero(ctx.legal_settlements):
        out[v] = settlement_features(ctx, int(v))
    return out


def _best_value_at_hops(
    ctx: SetupContext, *, origin: int, hops: tuple[int, ...], exclude: tuple[int, ...]
) -> float:
    """Best PINNED-base yield among legal vertices ``hops`` roads from ``origin``.

    ``exclude`` drops the origin settlement and anything the distance rule will
    forbid once it is placed (its direct neighbours), so an "expansion target"
    can never be a vertex the placement itself just killed.
    """
    forbidden = set(exclude)
    for v in exclude:
        forbidden.update(ctx.adjacency[v])
    best = 0.0
    for cand in range(N_VERTICES):
        if cand in forbidden or not ctx.legal_settlements[cand]:
            continue
        if int(ctx.distances[origin, cand]) not in hops:
            continue
        best = max(best, float(ctx.base_value[cand]))
    return best


# ---------------------------------------------------------------------------
# Road features
# ---------------------------------------------------------------------------
def road_far_endpoint(ctx: SetupContext, settlement: int, edge: int) -> int:
    """The endpoint of ``edge`` that is not ``settlement``."""
    if edge not in ctx.edge_vertices:
        raise ValueError(f"edge out of range: {edge}")
    v1, v2 = ctx.edge_vertices[edge]
    if settlement == v1:
        return v2
    if settlement == v2:
        return v1
    raise ValueError(f"edge {edge} is not incident to settlement vertex {settlement}")


def road_features(ctx: SetupContext, settlement: int, edge: int) -> np.ndarray:
    """The ``(3,)`` feature row for one candidate setup road."""
    far = road_far_endpoint(ctx, settlement, edge)
    opens = _best_value_at_hops(ctx, origin=far, hops=(1, 2), exclude=(settlement,))
    if ctx.opponent_best_vertex is None:
        blocks = 0.0
    else:
        blocks = 1.0 if int(ctx.distances[far, ctx.opponent_best_vertex]) <= 1 else 0.0
    toward_port = 1.0 if _port_of(ctx.board, far) else 0.0
    return np.asarray([opens, blocks, toward_port], dtype=np.float64)


def all_road_features(ctx: SetupContext, settlement: int, legal_edges: np.ndarray) -> np.ndarray:
    """``(72, 3)``; rows for illegal edges are all-zero (masked by the scorer)."""
    legal = np.asarray(legal_edges, dtype=bool)
    if legal.shape != (N_EDGES,):
        raise ValueError(f"legal_edges must be shape (72,), got {legal.shape}")
    out = np.zeros((N_EDGES, N_ROAD_FEATURES), dtype=np.float64)
    for e in np.flatnonzero(legal):
        out[e] = road_features(ctx, settlement, int(e))
    return out
