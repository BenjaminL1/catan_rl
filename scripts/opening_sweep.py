"""Opening-quality sweep of a frozen champion policy over the SETUP phase.

Measurement only. This script scores every *legal alternative* at each of the
policy's four setup decisions (settlement, road, settlement, road) against a
fixed set of hand-defined board metrics, and records where the candidate the
policy actually chose sits in that distribution. It draws no conclusions; the
report it writes is a numbers-only artifact.

No game is played past the setup phase, so no search, no dice and no reward is
involved: the action-type mask at setup forces BUILD_SETTLEMENT / BUILD_ROAD
(``catan_rl.env.masks.compute_action_masks``), leaving exactly one open head per
decision. The policy's choice is taken as the **deterministic argmax** of that
head under the legal mask (never a sample), so the whole sweep is reproducible.

--------------------------------------------------------------------------
SWEEP DESIGN
--------------------------------------------------------------------------
* 200 boards (env seeds ``0..199``); the env seeds ``numpy.random`` +
  ``random`` on reset, so one seed == one fixed board layout.
* Both seats: ``agent_seat=0`` (policy drafts first) and ``agent_seat=1``
  (policy drafts second) in the 1->2->2->1 snake.
* Two opponent conditions:
  - ``greedy``: the opponent seat is the same champion policy, argmax on
    every head.
  - ``diverse``: identical except that the opponent's *setup settlement*
    is drawn uniformly from its top-8 legal vertices by corner-head logit
    (own numpy Generator, seeded per game -> reproducible). This exists to
    inject design variance into the board states the policy is scored on;
    at the champion's setup entropy the greedy condition alone leaves the
    covariates nearly constant.

Cells: 200 seeds x 2 seats x 2 conditions = 800 games x 4 decisions = 3200
decisions (1600 settlement, 1600 road).

Decisions 0, 1 and 2 are applied to the env; decision 3 (the second road) is
scored from its mask without being applied, because nothing downstream of it is
measured and applying it would trigger a full opponent main turn.

--------------------------------------------------------------------------
METRIC DEFINITIONS
--------------------------------------------------------------------------
``dots(h)``  - the standard Catan dot count of hex ``h``'s number token
               (2/12=1, 3/11=2, 4/10=3, 5/9=4, 6/8=5; desert / no token = 0).
               Table imported from ``catan_rl.policy.obs_encoder.DOTS_BY_TOKEN``
               (the public alias), not re-derived here.
``adj(v)``   - the hexes adjacent to vertex ``v`` (1-3; coastal vertices have
               fewer).
``N(v)``     - the vertices joined to ``v`` by one board edge (i.e. by one
               road). "road-distance" below always means the number of edges on
               a shortest path in this graph.

SETTLEMENT-CANDIDATE METRICS (computed for every legal vertex at the decision)

  pip_sum              sum of dots(h) over h in adj(v).                [high=more production]
  ore_pips             sum of dots(h) over ORE hexes in adj(v).        [high=more ore]
  has_ore              1 if ore_pips > 0 else 0.
  wheat_pips           sum of dots(h) over WHEAT hexes in adj(v).      [high=more wheat]
  robber_robustness    pip_sum - max(dots(h) for h in adj(v)); i.e. the
                       production that survives if the opponent parks the
                       robber on this vertex's single best hex. 0 if adj(v)
                       is empty.                                       [high=more robust]
  exp_d2/d3/d4         count of LEGAL FUTURE SETTLEMENT SITES at road-distance
                       exactly 2 / 3 / 4 from the candidate. A vertex u is a
                       legal future site iff u is unoccupied and no vertex in
                       N(u) is occupied (Catan's distance rule), evaluated
                       against the board as it stands at the decision PLUS the
                       candidate hypothetically placed at v. Distance is BFS
                       from v over the vertex graph, and the BFS does not
                       expand THROUGH a vertex owned by the opponent - matching
                       the engine's road rule (``catanBoard.get_potential_roads``
                       refuses to extend out of an opponent-occupied vertex).
                       Distance 1 is never legal (it is adjacent to v).
                                                                       [high=more room]
  exp_d2/3/4_pip       the same sets, weighted by each site's own pip_sum.
  centrality_dist      Euclidean distance from the board centroid to the vertex,
                       in units of one hex edge length (the engine's 80 px). The
                       centroid is the layout origin (500, 400), which is also
                       the exact mean of all 54 vertex pixel positions.
                                                                       [high=further OUT]

SETTLEMENT-PAIR METRICS (well-defined only at decision index 2, where the first
settlement is already fixed, so each candidate for the second settlement
determines a complete pair; also reported for the pair the policy realised)

  Let ``E_R`` = expected cards of resource R produced by the pair per DICE ROLL
  = sum over both settlements s, over h in adj(s) with resource(h)==R, of
  dots(h)/36. (A settlement yields 1 card per hit; no cities exist at setup.)

  pair_pip_sum             pip_sum(s1) + pip_sum(s2).
  pair_ore_pips            ore_pips(s1) + ore_pips(s2).
  pair_wheat_pips          wheat_pips(s1) + wheat_pips(s2).
  pair_city_self_sufficient
                           1 iff E_ORE > 0 AND E_WHEAT > 0, i.e. the pair's own
                           production can eventually reach a city's 3 ore + 2
                           wheat with no trade of any kind. 0 otherwise.
  pair_exp_rolls_to_city   max(3 / E_ORE, 2 / E_WHEAT), in DICE ROLLS (both
                           seats roll, so ~2 per game round). This is the
                           deterministic-rate approximation "time = requirement
                           / rate" per resource, then the slower of the two - it
                           is NOT the exact expectation of the max of two
                           hitting times, and it ignores 7s, the robber,
                           discards and any spending. Infinite (recorded as
                           null in the JSON, excluded from means) when E_ORE or
                           E_WHEAT is 0.                               [LOW is faster]
  pair_max_ore_lump        max over number tokens t of the count of pair
                           settlements adjacent to an ORE hex bearing t: the
                           most ore cards a single dice number pays the pair at
                           once.                                       [high=lumpier]
  pair_robber_robustness   pair_pip_sum minus the largest single-hex loss,
                           where a hex h costs dots(h) x (number of the pair's
                           settlements adjacent to h).                 [high=more robust]
  pair_spread              Euclidean distance between the two settlements, in
                           hex edge lengths.                           [high=further apart]

ROAD-CANDIDATE METRICS. Every legal setup road runs from the settlement just
placed (``v1``) to a neighbour ``v2`` (``catanBoard.get_setup_roads``), so there
are at most 3 candidates. Write ``shared(r) = adj(v1) & adj(v2)`` and
``nonshared(r) = adj(v2) - adj(v1)`` - the hexes the road newly reaches.

  road_2hop_pip_max    max pip_sum(u) over u in N(v2)\\{v1} that are legal
                       future settlement sites. Distance 2 from v1 is the
                       NEAREST legal settlement site along this road (distance 1
                       is barred by the distance rule), so this is the best site
                       the road unlocks. 0 if there is none.           [high=better payoff]
  road_2hop_pip_sum    the same set, summed instead of maxed.
  road_breadth         count of legal future settlement sites within road-
                       distance 2 of v2, BFS'd on the graph with v1 deleted so
                       every path runs outward through this road (equivalently:
                       distance <= 3 from v1 via this road).           [high=opens more]
  road_breadth_pip     the same set weighted by each site's pip_sum.
  nonshared_pip_sum    sum of dots(h) over h in nonshared(r).
  nonshared_ore_pips   sum of dots(h) over ORE hexes in nonshared(r).
  shared_pip_sum       sum of dots(h) over h in shared(r).
  shared_ore_pips      sum of dots(h) over ORE hexes in shared(r).

  Road argmax rates are reported on ``road_2hop_pip_max`` and on
  ``road_breadth``; a chosen road that ties the maximum counts as argmax.

  SHARED / NON-SHARED PARTITION. For each road decision let ``c`` be the chosen
  road and ``b`` the best alternative by ``road_2hop_pip_max`` (excluding ``c``;
  skipped when only one road is legal). The question is where ``b``'s extra pip
  mass sits relative to ``c``: on the NON-SHARED hex (a genuinely new direction
  the road opens) or on the SHARED hexes (already inside the settlement's own
  1-hop neighbourhood). Both blocks are recorded per candidate, so the class is
  the sign pair ``(nonshared_pip_sum[b] - nonshared_pip_sum[c],
  shared_pip_sum[b] - shared_pip_sum[c])``; hexes adjacent to both payoff
  vertices cancel. Four mutually-exclusive classes: richer on NON-SHARED only /
  SHARED only / both / not richer. The ORE subset repeats the partition on
  ``nonshared_ore_pips`` and ``shared_ore_pips``.
  NOTE: "the discriminating resource of the best alternative" is ambiguous as
  stated - "best" depends on the metric - so ``road_2hop_pip_max`` is the
  operationalisation used, and it is the only one reported.

ORE-SUBSTITUTION RATE. Over all settlement decisions: the fraction where the
chosen candidate has ore_pips <= 1 AND some legal alternative has ore_pips >= 3
with ``abs(pip_sum(alt) - pip_sum(chosen)) <= 2``. Reported with a Wilson 95%
CI. A conditional variant (denominator restricted to decisions where such an
alternative existed) is reported alongside.

PERCENTILE RANK. For a metric ``m`` at a decision with legal candidate set
``S`` (the chosen candidate included) the chosen candidate's percentile is the
mid-rank

    pct = 100 * ( #{s in S : m(s) < m(chosen)} + 0.5 * #{s in S : m(s) == m(chosen)} ) / |S|

so ties are split symmetrically and a uniformly-random chooser averages 50.
Higher percentile always means a higher RAW metric value; each metric's
direction is stated above and repeated in the report.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    nice -n 19 python scripts/opening_sweep.py \\
        --checkpoint runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt \\
        --seeds 200 --device cpu

Writes ``runs/analysis/opening_sweep.json`` (raw per-decision records) and
``docs/plans/opening_sweep_results.md`` (the numbers-only report).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.wilson import wilson_interval
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.heads import masked_log_softmax
from catan_rl.policy.network import CatanPolicy
from catan_rl.policy.obs_encoder import DOTS_BY_TOKEN
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt"
DEFAULT_JSON = REPO_ROOT / "runs/analysis/opening_sweep.json"
DEFAULT_REPORT = REPO_ROOT / "docs/plans/opening_sweep_results.md"

BUILD_SETTLEMENT = 0
BUILD_ROAD = 2
TOP_K_DIVERSE = 8

# Per-candidate metric names, split by decision kind. Order fixes the report.
SETTLEMENT_METRICS = (
    "pip_sum",
    "ore_pips",
    "has_ore",
    "wheat_pips",
    "robber_robustness",
    "exp_d2",
    "exp_d3",
    "exp_d4",
    "exp_d2_pip",
    "exp_d3_pip",
    "exp_d4_pip",
    "centrality_dist",
)
PAIR_METRICS = (
    "pair_pip_sum",
    "pair_ore_pips",
    "pair_wheat_pips",
    "pair_city_self_sufficient",
    "pair_exp_rolls_to_city",
    "pair_max_ore_lump",
    "pair_robber_robustness",
    "pair_spread",
)
ROAD_METRICS = (
    "road_2hop_pip_max",
    "road_2hop_pip_sum",
    "road_breadth",
    "road_breadth_pip",
    "nonshared_pip_sum",
    "nonshared_ore_pips",
    "shared_pip_sum",
    "shared_ore_pips",
)

# "high = <text>" direction of each metric, for the report's direction column.
METRIC_DIRECTION: dict[str, str] = {
    "pip_sum": "more total production",
    "ore_pips": "more ore production",
    "has_ore": "touches ore",
    "wheat_pips": "more wheat production",
    "robber_robustness": "more production survives a robber on the best hex",
    "exp_d2": "more legal sites 2 roads away",
    "exp_d3": "more legal sites 3 roads away",
    "exp_d4": "more legal sites 4 roads away",
    "exp_d2_pip": "richer sites 2 roads away",
    "exp_d3_pip": "richer sites 3 roads away",
    "exp_d4_pip": "richer sites 4 roads away",
    "centrality_dist": "FURTHER from the board centre",
    "pair_pip_sum": "more total production",
    "pair_ore_pips": "more ore production",
    "pair_wheat_pips": "more wheat production",
    "pair_city_self_sufficient": "pair can reach 3 ore + 2 wheat unaided",
    "pair_exp_rolls_to_city": "SLOWER to the first city",
    "pair_max_ore_lump": "bigger single-roll ore payout",
    "pair_robber_robustness": "more production survives a robber on the best hex",
    "pair_spread": "settlements FURTHER apart",
    "road_2hop_pip_max": "richer best site unlocked",
    "road_2hop_pip_sum": "richer sites unlocked",
    "road_breadth": "opens more legal sites",
    "road_breadth_pip": "opens richer sites",
    "nonshared_pip_sum": "road reaches toward richer new hexes",
    "nonshared_ore_pips": "road reaches toward more ore",
    "shared_pip_sum": "road's shared hexes are richer",
    "shared_ore_pips": "road's shared hexes hold more ore",
}


# ---------------------------------------------------------------------------
# Board metric primitives
# ---------------------------------------------------------------------------


# Geometric distances are quantised before ranking. The engine rounds vertex
# PIXEL coordinates to 2 dp (``HexCoordinates.get_corners``), so two vertices
# that are geometrically equidistant from the centroid land ~1e-4 edge-lengths
# apart in float. Left raw, that noise imposes an arbitrary STRICT order on what
# should be an exact tie, and the mid-rank percentile then mis-ranks by up to
# ~9 points: the 54 vertices form only 6 distinct radius clusters (sizes 6/6/12/
# 12/6/12 under the board's hexagonal symmetry), within-cluster spread 8.1e-5,
# minimum between-cluster gap 0.359. 3 dp sits ~12x above the noise floor and
# ~350x below the smallest real separation, so it merges only true ties.
_DISTANCE_DP = 3


def _quantize(x: float) -> float:
    return round(x, _DISTANCE_DP)


def _dots(board: Any, hex_idx: int) -> int:
    return int(DOTS_BY_TOKEN.get(board.hexTileDict[hex_idx].number_token, 0))


def _resource(board: Any, hex_idx: int) -> str:
    return str(board.hexTileDict[hex_idx].resource_type)


class BoardScorer:
    """Per-board caches for the vertex metrics that never depend on occupancy."""

    def __init__(self, board: Any) -> None:
        self.board = board
        self.centroid = (board.flat.origin.x, board.flat.origin.y)
        self.edge_len = float(board.edgeLength)
        self.pip_sum: dict[Any, int] = {}
        self.res_pips: dict[Any, dict[str, int]] = {}
        self.max_hex_dots: dict[Any, int] = {}
        self.centrality: dict[Any, float] = {}
        for v_px, v_obj in board.boardGraph.items():
            adj = v_obj.adjacent_hex_indices
            per_res: dict[str, int] = {}
            total = 0
            best = 0
            for h in adj:
                d = _dots(board, h)
                total += d
                best = max(best, d)
                per_res[_resource(board, h)] = per_res.get(_resource(board, h), 0) + d
            self.pip_sum[v_px] = total
            self.res_pips[v_px] = per_res
            self.max_hex_dots[v_px] = best
            dx = v_px[0] - self.centroid[0]
            dy = v_px[1] - self.centroid[1]
            self.centrality[v_px] = _quantize(math.hypot(dx, dy) / self.edge_len)

    # -- occupancy-dependent helpers ------------------------------------

    def occupied(self, extra: Iterable[Any] = ()) -> set[Any]:
        """Vertices carrying a building, plus any hypothetical extras."""
        occ = {v for v, o in self.board.boardGraph.items() if o.owner is not None}
        occ.update(extra)
        return occ

    def opponent_blocked(self, own_player: Any) -> set[Any]:
        """Vertices a road network may not be extended THROUGH (engine rule)."""
        return {
            v
            for v, o in self.board.boardGraph.items()
            if o.owner is not None and o.owner is not own_player
        }

    def legal_sites(self, occupied: set[Any]) -> set[Any]:
        """Vertices where a settlement could legally be built later on."""
        out: set[Any] = set()
        for v_px, v_obj in self.board.boardGraph.items():
            if v_px in occupied:
                continue
            if any(nb in occupied for nb in v_obj.neighbors):
                continue
            out.add(v_px)
        return out

    def bfs(self, source: Any, blocked: set[Any], max_depth: int) -> dict[Any, int]:
        """BFS distances from ``source``; ``blocked`` vertices are reachable but
        are not expanded through (they terminate a path)."""
        dist = {source: 0}
        frontier = [source]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for v in frontier:
                if v != source and v in blocked:
                    continue
                for nb in self.board.boardGraph[v].neighbors:
                    if nb not in dist:
                        dist[nb] = depth
                        nxt.append(nb)
            frontier = nxt
        return dist

    # -- candidate scorers ----------------------------------------------

    def settlement_metrics(
        self, v_px: Any, base_occupied: set[Any], blocked: set[Any]
    ) -> dict[str, float]:
        res = self.res_pips[v_px]
        pip = self.pip_sum[v_px]
        occ = set(base_occupied)
        occ.add(v_px)
        legal = self.legal_sites(occ)
        dist = self.bfs(v_px, blocked, max_depth=4)
        counts = {2: 0, 3: 0, 4: 0}
        pips = {2: 0, 3: 0, 4: 0}
        for u, d in dist.items():
            if d in counts and u in legal:
                counts[d] += 1
                pips[d] += self.pip_sum[u]
        ore = float(res.get("ORE", 0))
        return {
            "pip_sum": float(pip),
            "ore_pips": ore,
            "has_ore": 1.0 if ore > 0 else 0.0,
            "wheat_pips": float(res.get("WHEAT", 0)),
            "robber_robustness": float(pip - self.max_hex_dots[v_px]),
            "exp_d2": float(counts[2]),
            "exp_d3": float(counts[3]),
            "exp_d4": float(counts[4]),
            "exp_d2_pip": float(pips[2]),
            "exp_d3_pip": float(pips[3]),
            "exp_d4_pip": float(pips[4]),
            "centrality_dist": self.centrality[v_px],
        }

    def pair_metrics(self, v1: Any, v2: Any) -> dict[str, float | None]:
        board = self.board
        e_ore = 0.0
        e_wheat = 0.0
        hex_hits: Counter[int] = Counter()
        ore_by_token: Counter[int] = Counter()
        for s in (v1, v2):
            for h in board.boardGraph[s].adjacent_hex_indices:
                hex_hits[h] += 1
                d = _dots(board, h)
                r = _resource(board, h)
                if r == "ORE":
                    e_ore += d / 36.0
                    tok = board.hexTileDict[h].number_token
                    if tok is not None:
                        ore_by_token[int(tok)] += 1
                elif r == "WHEAT":
                    e_wheat += d / 36.0
        pair_pip = float(self.pip_sum[v1] + self.pip_sum[v2])
        worst = max((_dots(board, h) * n for h, n in hex_hits.items()), default=0)
        self_suff = 1.0 if (e_ore > 0.0 and e_wheat > 0.0) else 0.0
        rolls: float | None
        # 4 dp matches the JSON serialisation, so the stored values are exactly
        # the ones the percentiles were computed from.
        rolls = round(max(3.0 / e_ore, 2.0 / e_wheat), 4) if self_suff > 0 else None
        spread = _quantize(math.hypot(v1[0] - v2[0], v1[1] - v2[1]) / self.edge_len)
        return {
            "pair_pip_sum": pair_pip,
            "pair_ore_pips": float(
                self.res_pips[v1].get("ORE", 0) + self.res_pips[v2].get("ORE", 0)
            ),
            "pair_wheat_pips": float(
                self.res_pips[v1].get("WHEAT", 0) + self.res_pips[v2].get("WHEAT", 0)
            ),
            "pair_city_self_sufficient": self_suff,
            "pair_exp_rolls_to_city": rolls,
            "pair_max_ore_lump": float(max(ore_by_token.values(), default=0)),
            "pair_robber_robustness": pair_pip - float(worst),
            "pair_spread": spread,
        }

    def road_metrics(
        self, v1: Any, v2: Any, base_occupied: set[Any], blocked: set[Any]
    ) -> dict[str, float]:
        board = self.board
        legal = self.legal_sites(base_occupied)
        two_hop = [u for u in board.boardGraph[v2].neighbors if u != v1 and u in legal]
        # Breadth: BFS from v2 with v1 deleted, so all paths run outward.
        dist = self.bfs(v2, blocked | {v1}, max_depth=2)
        breadth_sites = [u for u, d in dist.items() if 0 < d <= 2 and u in legal and u != v1]
        adj1 = set(board.boardGraph[v1].adjacent_hex_indices)
        adj2 = set(board.boardGraph[v2].adjacent_hex_indices)
        nonshared = adj2 - adj1
        shared = adj2 & adj1
        return {
            "road_2hop_pip_max": float(max((self.pip_sum[u] for u in two_hop), default=0)),
            "road_2hop_pip_sum": float(sum(self.pip_sum[u] for u in two_hop)),
            "road_breadth": float(len(breadth_sites)),
            "road_breadth_pip": float(sum(self.pip_sum[u] for u in breadth_sites)),
            "nonshared_pip_sum": float(sum(_dots(board, h) for h in nonshared)),
            "nonshared_ore_pips": float(
                sum(_dots(board, h) for h in nonshared if _resource(board, h) == "ORE")
            ),
            "shared_pip_sum": float(sum(_dots(board, h) for h in shared)),
            "shared_ore_pips": float(
                sum(_dots(board, h) for h in shared if _resource(board, h) == "ORE")
            ),
        }


def percentile_rank(values: Sequence[float], chosen: float) -> float:
    """Mid-rank percentile of ``chosen`` within ``values`` (chosen included)."""
    n = len(values)
    if n == 0:
        return float("nan")
    below = sum(1 for v in values if v < chosen)
    equal = sum(1 for v in values if v == chosen)
    return 100.0 * (below + 0.5 * equal) / n


# ---------------------------------------------------------------------------
# Policy plumbing
# ---------------------------------------------------------------------------


def load_policy(ckpt: Path, device: torch.device) -> CatanPolicy:
    """Construct + strict-load the champion, mirroring the eval harness order."""
    from catan_rl.checkpoint import load_checkpoint

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy = policy.to(device)
    load_checkpoint(ckpt, map_location=device).apply_to_policy(policy, strict=True)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy


@torch.no_grad()
def setup_logp(
    policy: CatanPolicy,
    obs_t: dict[str, torch.Tensor],
    masks_t: dict[str, torch.Tensor],
    settlement: bool,
) -> torch.Tensor:
    """Masked log-probs over the single open head of a setup decision.

    At setup the type mask admits exactly one action type, so the joint
    distribution collapses onto the corner head (settlement) or the edge head
    (road). This mirrors ``CatanActionHeads.sample`` for that head exactly.
    """
    out = policy.forward(obs_t)
    heads = policy.action_heads
    if settlement:
        type_idx = torch.full(
            (out["trunk"].shape[0],), BUILD_SETTLEMENT, dtype=torch.long, device=out["trunk"].device
        )
        ctx = heads._corner_context(type_idx, out.get("_is_setup"))
        logits = heads.corner_head(out["trunk"], out["_node_v"], ctx)
        return masked_log_softmax(logits, masks_t["corner_settlement"])
    logits = heads.edge_head(out["trunk"], out["_node_e"])
    return masked_log_softmax(logits, masks_t["edge"])


@torch.no_grad()
def greedy_action(
    policy: CatanPolicy, obs_t: dict[str, torch.Tensor], masks_t: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Full autoregressive argmax — ``CatanActionHeads.sample`` with the
    categorical draws replaced by ``argmax``. Batched (B, 6) int64 out."""
    out = policy.forward(obs_t)
    heads = policy.action_heads
    trunk = out["trunk"]

    type_idx = masked_log_softmax(heads.type_head(trunk), masks_t["type"]).argmax(-1)

    corner_ctx = heads._corner_context(type_idx, out.get("_is_setup"))
    corner_logits = heads.corner_head(trunk, out["_node_v"], corner_ctx)
    corner_idx = masked_log_softmax(corner_logits, heads._corner_mask(type_idx, masks_t)).argmax(-1)

    edge_idx = masked_log_softmax(heads.edge_head(trunk, out["_node_e"]), masks_t["edge"]).argmax(
        -1
    )
    tile_idx = masked_log_softmax(heads.tile_head(trunk, out["_node_h"]), masks_t["tile"]).argmax(
        -1
    )

    res1_logits = heads.resource1_head(trunk, heads._resource1_context(type_idx))
    res1_idx = masked_log_softmax(res1_logits, heads._resource1_mask(type_idx, masks_t)).argmax(-1)

    res2_logits = heads.resource2_head(trunk, heads._resource2_context(type_idx, res1_idx))
    res2_idx = masked_log_softmax(
        res2_logits, heads._resource2_mask(type_idx, res1_idx, masks_t)
    ).argmax(-1)

    return torch.stack([type_idx, corner_idx, edge_idx, tile_idx, res1_idx, res2_idx], dim=-1)


class SweepOpponent:
    """Opponent seat driven by the champion policy.

    ``diverse=False`` -> argmax on every head. ``diverse=True`` -> identical
    except that a setup SETTLEMENT decision draws uniformly from the top-8
    legal vertices by corner-head logit, using this object's own numpy
    Generator (never torch's global RNG, so it cannot perturb anything else).
    """

    def __init__(
        self, policy: CatanPolicy, device: torch.device, *, diverse: bool, seed: int
    ) -> None:
        self._policy = policy
        self._device = device
        self._diverse = diverse
        self._rng = np.random.default_rng(seed)

    @property
    def device(self) -> torch.device:
        return self._device

    def reset_rng(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

    @torch.no_grad()
    def sample(self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        is_settle_setup = (
            self._diverse
            and bool(masks["type"][0, BUILD_SETTLEMENT].item())
            and int(masks["type"][0].sum().item()) == 1
            and float(obs["is_setup"].reshape(-1)[0].item()) > 0.5
        )
        action = greedy_action(self._policy, obs, masks)
        if is_settle_setup:
            logp = setup_logp(self._policy, obs, masks, settlement=True)[0]
            legal = torch.nonzero(masks["corner_settlement"][0], as_tuple=True)[0]
            k = min(TOP_K_DIVERSE, int(legal.numel()))
            top = legal[torch.argsort(logp[legal], descending=True)[:k]]
            action[0, 1] = int(top[int(self._rng.integers(k))].item())
        return action


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One scored setup decision."""

    seed: int
    seat: int
    condition: str
    decision_index: int
    kind: str
    chosen_id: int
    candidate_ids: list[int]
    metrics: dict[str, list[float | None]]
    chosen_pos: int
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "seat": self.seat,
            "condition": self.condition,
            "decision_index": self.decision_index,
            "kind": self.kind,
            "chosen_id": self.chosen_id,
            "candidate_ids": self.candidate_ids,
            "metrics": {
                k: [None if v is None else round(v, 4) for v in vals]
                for k, vals in self.metrics.items()
            },
            "chosen_pos": self.chosen_pos,
            "percentiles": {k: round(v, 3) for k, v in self.percentiles.items()},
        }


def _score_decision(
    env: CatanEnv,
    scorer: BoardScorer,
    *,
    settlement: bool,
    first_settlement: Any | None,
) -> tuple[list[int], dict[str, list[float | None]], list[Any]]:
    """Enumerate every legal candidate and score it. Returns ids, per-metric
    value lists (parallel to ids) and the candidate board objects."""
    assert env.game is not None and env.agent_player is not None
    masks = env.get_action_masks()
    board = env.game.board
    agent = env.agent_player
    base_occupied = scorer.occupied()
    blocked = scorer.opponent_blocked(agent)
    metrics: dict[str, list[float | None]] = {}
    ids: list[int] = []
    objs: list[Any] = []

    if settlement:
        names: tuple[str, ...] = SETTLEMENT_METRICS + (
            PAIR_METRICS if first_settlement is not None else ()
        )
        for name in names:
            metrics[name] = []
        for idx in np.nonzero(masks["corner_settlement"])[0]:
            v_px = env._idx_to_vertex[int(idx)]
            vals: dict[str, float | None] = dict(
                scorer.settlement_metrics(v_px, base_occupied, blocked)
            )
            if first_settlement is not None:
                vals.update(scorer.pair_metrics(first_settlement, v_px))
            ids.append(int(idx))
            objs.append(v_px)
            for name in names:
                metrics[name].append(vals[name])
    else:
        for name in ROAD_METRICS:
            metrics[name] = []
        settle_v = agent.buildGraph["SETTLEMENTS"][-1]
        for idx in np.nonzero(masks["edge"])[0]:
            a, b = env._idx_to_edge[int(idx)]
            v1, v2 = (a, b) if a == settle_v else (b, a)
            vals_r = scorer.road_metrics(v1, v2, base_occupied, blocked)
            ids.append(int(idx))
            objs.append((v1, v2))
            for name in ROAD_METRICS:
                metrics[name].append(vals_r[name])
    return ids, metrics, objs


def run_sweep(
    policy: CatanPolicy,
    device: torch.device,
    *,
    n_seeds: int,
    conditions: Sequence[str],
    seats: Sequence[int],
) -> tuple[list[DecisionRecord], list[dict[str, Any]]]:
    records: list[DecisionRecord] = []
    games: list[dict[str, Any]] = []

    for condition in conditions:
        opponent = SweepOpponent(policy, device, diverse=(condition == "diverse"), seed=0)
        env = CatanEnv(opponent_type="snapshot")
        env.set_snapshot_opponent(opponent)
        for seed in range(n_seeds):
            for seat in seats:
                # Reproducible, condition/seat-distinct opponent stream.
                opponent.reset_rng(seed * 1000 + seat * 7 + (1 if condition == "diverse" else 0))
                obs, _ = env.reset(seed=seed, options={"agent_seat": seat})
                assert env.game is not None
                scorer = BoardScorer(env.game.board)
                first_settlement: Any | None = None
                chosen_pair: list[Any] = []

                for d_idx in range(4):
                    settlement = d_idx in (0, 2)
                    masks = env.get_action_masks()
                    ids, metrics, objs = _score_decision(
                        env,
                        scorer,
                        settlement=settlement,
                        first_settlement=first_settlement if d_idx == 2 else None,
                    )
                    obs_t = obs_to_torch(obs, device, add_batch=True)
                    masks_t = masks_to_torch(masks, device, add_batch=True)
                    logp = setup_logp(policy, obs_t, masks_t, settlement)[0]
                    chosen_id = int(logp.argmax().item())
                    if chosen_id not in ids:  # defensive: argmax must be legal
                        raise RuntimeError(
                            f"argmax {chosen_id} not in legal set "
                            f"(seed={seed} seat={seat} d={d_idx})"
                        )
                    pos = ids.index(chosen_id)

                    pcts: dict[str, float] = {}
                    for name, vals in metrics.items():
                        chosen_val = vals[pos]
                        if chosen_val is None:
                            continue
                        present = [v for v in vals if v is not None]
                        pcts[name] = percentile_rank(present, chosen_val)

                    records.append(
                        DecisionRecord(
                            seed=seed,
                            seat=seat,
                            condition=condition,
                            decision_index=d_idx,
                            kind="settlement" if settlement else "road",
                            chosen_id=chosen_id,
                            candidate_ids=ids,
                            metrics=metrics,
                            chosen_pos=pos,
                            percentiles=pcts,
                        )
                    )
                    if settlement:
                        chosen_pair.append(objs[pos])
                        if d_idx == 0:
                            first_settlement = objs[pos]
                    if d_idx == 3:
                        break  # scored from the mask; never applied (see docstring)
                    act = np.zeros(6, dtype=np.int64)
                    if settlement:
                        act[0], act[1] = BUILD_SETTLEMENT, chosen_id
                    else:
                        act[0], act[2] = BUILD_ROAD, chosen_id
                    obs, _, _, _, _ = env.step(act)

                realised = scorer.pair_metrics(chosen_pair[0], chosen_pair[1])
                games.append(
                    {
                        "seed": seed,
                        "seat": seat,
                        "condition": condition,
                        "realised_pair": {
                            k: (None if v is None else round(v, 4)) for k, v in realised.items()
                        },
                    }
                )
    return records, games


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _quantiles(xs: Sequence[float]) -> tuple[float, float, float]:
    """(q1, median, q3) — inclusive method; degenerate inputs fall back to the
    median for all three."""
    if not xs:
        return (float("nan"),) * 3
    s = sorted(xs)
    med = statistics.median(s)
    if len(s) < 4:
        return med, med, med
    q1, _, q3 = statistics.quantiles(s, n=4, method="inclusive")
    return q1, med, q3


@dataclass
class PctSummary:
    metric: str
    n: int
    q1: float
    median: float
    q3: float
    frac_above_90: float
    frac_below_50: float


def summarise_percentiles(records: Iterable[DecisionRecord], metric: str) -> PctSummary | None:
    vals = [r.percentiles[metric] for r in records if metric in r.percentiles]
    if not vals:
        return None
    q1, med, q3 = _quantiles(vals)
    return PctSummary(
        metric=metric,
        n=len(vals),
        q1=q1,
        median=med,
        q3=q3,
        frac_above_90=sum(1 for v in vals if v > 90.0) / len(vals),
        frac_below_50=sum(1 for v in vals if v < 50.0) / len(vals),
    )


def ore_substitution(records: Sequence[DecisionRecord]) -> dict[str, float | int]:
    """Chosen has <=1 ore pip while some alternative has >=3 ore pips within 2
    pip_sum of the chosen candidate."""
    n = 0
    hits = 0
    avail = 0
    for r in records:
        if r.kind != "settlement":
            continue
        n += 1
        ore = r.metrics["ore_pips"]
        pip = r.metrics["pip_sum"]
        c_ore, c_pip = ore[r.chosen_pos], pip[r.chosen_pos]
        assert c_ore is not None and c_pip is not None
        exists = any(
            o is not None and p is not None and o >= 3.0 and abs(p - c_pip) <= 2.0
            for i, (o, p) in enumerate(zip(ore, pip, strict=True))
            if i != r.chosen_pos
        )
        if exists:
            avail += 1
            if c_ore <= 1.0:
                hits += 1
    # ``hits`` is by construction a subset of ``avail``, so the conditional rate
    # is hits / avail and the unconditional rate is hits / n.
    return {
        "n_settlement_decisions": n,
        "n_substitutions": hits,
        "n_alternative_available": avail,
    }


def road_argmax_rate(records: Sequence[DecisionRecord], metric: str) -> tuple[int, int, int]:
    """(n_decisions, n_chosen_is_argmax, n_with_a_unique_max)."""
    n = 0
    hits = 0
    unique = 0
    for r in records:
        if r.kind != "road":
            continue
        vals = [v for v in r.metrics[metric] if v is not None]
        if not vals:
            continue
        n += 1
        top = max(vals)
        if vals.count(top) == 1:
            unique += 1
        chosen = r.metrics[metric][r.chosen_pos]
        if chosen is not None and chosen >= top:
            hits += 1
    return n, hits, unique


def _delta(r: DecisionRecord, metric: str, best: int) -> float:
    hi = r.metrics[metric][best]
    lo = r.metrics[metric][r.chosen_pos]
    return float(hi or 0.0) - float(lo or 0.0)


def shared_partition(records: Sequence[DecisionRecord]) -> dict[str, int]:
    """Discriminating-hex partition for the best alternative road.

    For each road decision with >=2 legal candidates, ``b`` is the best
    alternative to the chosen road ``c`` by ``road_2hop_pip_max``. The question
    is where ``b``'s extra pip mass sits: on the NON-SHARED hex
    (``adj(v2) - adj(v1)`` — the hex only the road reaches, i.e. a genuinely
    new direction) or on the SHARED hexes (``adj(v1) & adj(v2)`` — hexes the
    settlement already touches). Both blocks are recorded per candidate, so the
    partition is the sign of each block's ``b``-minus-``c`` difference; a hex
    adjacent to both payoff vertices cancels out.

    The four classes are mutually exclusive and sum to ``n``.
    """
    out = dict.fromkeys(
        (
            "n_road_decisions_with_alternative",
            "alt_richer_on_nonshared_only",
            "alt_richer_on_shared_only",
            "alt_richer_on_both",
            "alt_not_richer",
            "alt_gains_ore_on_nonshared_only",
            "alt_gains_ore_on_shared_only",
            "alt_gains_ore_on_both",
            "alt_gains_no_ore",
        ),
        0,
    )
    for r in records:
        if r.kind != "road" or len(r.candidate_ids) < 2:
            continue
        vals = r.metrics["road_2hop_pip_max"]
        alt = [i for i in range(len(r.candidate_ids)) if i != r.chosen_pos and vals[i] is not None]
        if not alt:
            continue
        best = max(alt, key=lambda i: vals[i] or 0.0)
        out["n_road_decisions_with_alternative"] += 1

        d_non = _delta(r, "nonshared_pip_sum", best)
        d_sh = _delta(r, "shared_pip_sum", best)
        if d_non > 0 and d_sh > 0:
            out["alt_richer_on_both"] += 1
        elif d_non > 0:
            out["alt_richer_on_nonshared_only"] += 1
        elif d_sh > 0:
            out["alt_richer_on_shared_only"] += 1
        else:
            out["alt_not_richer"] += 1

        o_non = _delta(r, "nonshared_ore_pips", best)
        o_sh = _delta(r, "shared_ore_pips", best)
        if o_non > 0 and o_sh > 0:
            out["alt_gains_ore_on_both"] += 1
        elif o_non > 0:
            out["alt_gains_ore_on_nonshared_only"] += 1
        elif o_sh > 0:
            out["alt_gains_ore_on_shared_only"] += 1
        else:
            out["alt_gains_no_ore"] += 1
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x != x:  # nan
        return "n/a"
    return f"{x:.1f}"


def _pct_table(records: Sequence[DecisionRecord], metrics: Sequence[str]) -> list[str]:
    lines = [
        "| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        s = summarise_percentiles(records, m)
        if s is None:
            continue
        lines.append(
            f"| `{m}` | {s.n} | {METRIC_DIRECTION.get(m, '?')} | {_fmt(s.q1)} | "
            f"{_fmt(s.median)} | {_fmt(s.q3)} | {s.frac_above_90:.3f} | {s.frac_below_50:.3f} |"
        )
    return lines


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def write_report(
    path: Path,
    *,
    records: Sequence[DecisionRecord],
    games: Sequence[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    settle = [r for r in records if r.kind == "settlement"]
    roads = [r for r in records if r.kind == "road"]
    d0 = [r for r in settle if r.decision_index == 0]
    d2 = [r for r in settle if r.decision_index == 2]

    out: list[str] = []
    add = out.append
    add("# Opening-quality sweep — champion policy, setup phase")
    add("")
    add("**Numbers only.** No conclusions, diagnoses or recommendations appear in this")
    add("document by design; it is an input to a separate review process.")
    add("")
    add("## Run metadata")
    add("")
    add("| field | value |")
    add("|---|---|")
    for k, v in meta.items():
        add(f"| {k} | `{v}` |")
    add("")

    add("## Metric definitions")
    add("")
    add("Reproduced verbatim from the script docstring")
    add("(`scripts/opening_sweep.py`), which is the authoritative definition.")
    add("")
    add("`dots(h)` is the standard Catan dot count of hex `h` (2/12=1, 3/11=2, 4/10=3,")
    add("5/9=4, 6/8=5; desert = 0), imported from")
    add("`catan_rl.policy.obs_encoder.DOTS_BY_TOKEN`. `adj(v)` are the hexes touching")
    add("vertex `v`; `N(v)` the vertices one road away. *Road-distance* is the number of")
    add("board edges on a shortest path.")
    add("")
    add("| metric | definition |")
    add("|---|---|")
    add("| `pip_sum` | Σ `dots(h)` over `h ∈ adj(v)`. |")
    add("| `ore_pips` / `wheat_pips` | Σ `dots(h)` over ORE / WHEAT hexes in `adj(v)`. |")
    add("| `has_ore` | 1 iff `ore_pips > 0`. |")
    add(
        "| `robber_robustness` | `pip_sum - max(dots(h) for h ∈ adj(v))` — production "
        "surviving a robber on the single best hex. |"
    )
    add(
        "| `exp_d2/d3/d4` | count of legal future settlement sites at road-distance "
        "exactly 2 / 3 / 4. A site `u` is legal iff `u` and every vertex in `N(u)` are "
        "unoccupied, evaluated on the live board **plus** the candidate placed at `v`. "
        "BFS does not expand *through* an opponent-owned vertex (the engine's road "
        "rule). Distance 1 is never legal. |"
    )
    add("| `exp_d2/3/4_pip` | the same sets weighted by each site's own `pip_sum`. |")
    add(
        "| `centrality_dist` | Euclidean distance from the board centroid (the layout "
        "origin `(500,400)`, which is exactly the mean of all 54 vertex positions) in "
        "units of one hex edge length (80 px). |"
    )
    add(
        "| `pair_*` | pair-level; defined only once both settlements exist, so scored "
        "per-candidate at decision 2 and for the realised pair. `E_R` = expected cards "
        "of resource `R` per **dice roll** = Σ `dots(h)/36` over both settlements' "
        "adjacent `R` hexes. |"
    )
    add(
        "| `pair_city_self_sufficient` | 1 iff `E_ORE > 0` **and** `E_WHEAT > 0`, i.e. "
        "the pair's own production can eventually cover a city's 3 ore + 2 wheat with "
        "no trade. |"
    )
    add(
        "| `pair_exp_rolls_to_city` | `max(3/E_ORE, 2/E_WHEAT)` **dice rolls** (both "
        "seats roll, ~2 per game round). Deterministic-rate approximation "
        "(requirement ÷ rate, then the slower resource); **not** the exact expectation "
        "of the max of two hitting times. Ignores 7s, the robber, discards and "
        "spending. Undefined (excluded) when `E_ORE` or `E_WHEAT` is 0. |"
    )
    add(
        "| `pair_max_ore_lump` | max over number tokens `t` of the count of the pair's "
        "settlements adjacent to an ORE hex bearing `t` — the most ore a single dice "
        "number pays at once. |"
    )
    add(
        "| `pair_robber_robustness` | `pair_pip_sum` minus the largest single-hex loss, "
        "where hex `h` costs `dots(h) * (#pair settlements adjacent to h)`. |"
    )
    add("| `pair_spread` | distance between the two settlements, in hex edge lengths. |")
    add(
        "| `road_2hop_pip_max` / `_sum` | over `u ∈ N(v2)\\{v1}` that are legal future "
        "sites (`v1` = the settlement just placed, `v2` = the road's far end): max / sum "
        "of `pip_sum(u)`. Distance 2 is the nearest legal site along the road. |"
    )
    add(
        "| `road_breadth` / `_pip` | legal future sites within road-distance 2 of `v2` "
        "with `v1` deleted from the graph (all paths run outward through this road); "
        "count / `pip_sum`-weighted. |"
    )
    add(
        "| `nonshared_pip_sum` / `nonshared_ore_pips` | pip / ore-pip mass on "
        "`adj(v2) - adj(v1)` — the hexes the road newly reaches. |"
    )
    add("| `shared_ore_pips` | ore-pip mass on `adj(v1) ∩ adj(v2)`. |")
    add("")
    add("**Percentile rank** of the chosen candidate among the legal set `S`")
    add("(chosen included), mid-rank so ties split symmetrically:")
    add("")
    add("```")
    add("pct = 100 * ( #{s ∈ S : m(s) < m(chosen)} + 0.5*#{s ∈ S : m(s) = m(chosen)} ) / |S|")
    add("```")
    add("")
    add("A uniformly-random chooser averages 50. **Higher percentile always means a")
    add("higher RAW value**; the `high percentile =` column in every table below states")
    add("what that means for each metric (note `centrality_dist` and")
    add("`pair_exp_rolls_to_city` are metrics where high = further out / slower).")
    add("")
    add(
        "Geometric distances (`centrality_dist`, `pair_spread`) are rounded to 3 decimal "
        "places before ranking. The engine rounds vertex pixel coordinates to 2 dp, so "
        "vertices that are geometrically equidistant from the centroid differ by ~1e-4 "
        "edge-lengths in float; unrounded, that noise imposes an arbitrary strict order "
        "on an exact tie. The 54 vertices form only 6 distinct radius clusters (sizes "
        "6/6/12/12/6/12), within-cluster spread 8.1e-5, minimum between-cluster gap "
        "0.359 — so 3 dp merges only true ties and separates every real one."
    )
    add("")

    add("## n per cell")
    add("")
    add("| cell | games | settlement decisions | road decisions |")
    add("|---|---:|---:|---:|")
    for cond in ("greedy", "diverse"):
        for seat in (0, 1):
            g = [x for x in games if x["condition"] == cond and x["seat"] == seat]
            s = [r for r in settle if r.condition == cond and r.seat == seat]
            rd = [r for r in roads if r.condition == cond and r.seat == seat]
            add(f"| {cond}, seat {seat} | {len(g)} | {len(s)} | {len(rd)} |")
    add(f"| **total** | **{len(games)}** | **{len(settle)}** | **{len(roads)}** |")
    add("")
    add("Legal-candidate-set sizes (the percentile denominators):")
    add("")
    add("| decision | n | mean candidates | min | max |")
    add("|---|---:|---:|---:|---:|")
    size_groups = (
        ("settlement #1 (d0)", d0),
        ("settlement #2 (d2)", d2),
        ("road (d1,d3)", list(roads)),
    )
    for label, grp in size_groups:
        sizes = [len(r.candidate_ids) for r in grp]
        if sizes:
            add(
                f"| {label} | {len(sizes)} | {_mean([float(x) for x in sizes]):.1f} | "
                f"{min(sizes)} | {max(sizes)} |"
            )
    add("")
    add(
        "> Note on design overlap: at `seat 0`, decision 0 is taken on an empty board, "
        "so it is *identical* between the greedy and diverse conditions for a given "
        "seed (the opponent has not acted yet). The two conditions diverge from "
        "decision 2 onward at seat 0, and from decision 0 onward at seat 1."
    )
    add("")

    add("## Percentile ranks — settlement decisions (pooled)")
    add("")
    out.extend(_pct_table(settle, SETTLEMENT_METRICS))
    add("")
    add("### Settlement #1 (decision 0)")
    add("")
    out.extend(_pct_table(d0, SETTLEMENT_METRICS))
    add("")
    add("### Settlement #2 (decision 2) — includes pair-level metrics")
    add("")
    add(
        "Note: at decision 2 the first settlement is already fixed, so "
        "`pair_pip_sum`, `pair_ore_pips` and `pair_wheat_pips` are each the "
        "corresponding vertex metric plus a per-decision constant. A constant "
        "shift is a monotone transform, so their percentile rows are *identical* "
        "to `pip_sum` / `ore_pips` / `wheat_pips` by construction — not a "
        "duplication bug. `pair_robber_robustness`, `pair_city_self_sufficient`, "
        "`pair_exp_rolls_to_city`, `pair_max_ore_lump` and `pair_spread` are not "
        "constant shifts and do differ."
    )
    add("")
    out.extend(_pct_table(d2, SETTLEMENT_METRICS + PAIR_METRICS))
    add("")

    add("## Percentile ranks — road decisions (pooled)")
    add("")
    add(
        "Setup roads emanate from the settlement just placed, so there are at most 3 "
        "legal candidates; percentiles are correspondingly coarse (with 3 candidates "
        "the only attainable mid-ranks are 16.7 / 50.0 / 83.3, before ties). The argmax "
        "rates below are the primary road statistic."
    )
    add("")
    out.extend(_pct_table(roads, ROAD_METRICS))
    add("")

    add("## Per-condition and per-seat breakdown")
    add("")
    for cond in ("greedy", "diverse"):
        for seat in (0, 1):
            sub = [r for r in settle if r.condition == cond and r.seat == seat]
            add(f"### Settlement decisions — {cond}, seat {seat}")
            add("")
            out.extend(_pct_table(sub, SETTLEMENT_METRICS))
            add("")
    for cond in ("greedy", "diverse"):
        for seat in (0, 1):
            sub = [r for r in roads if r.condition == cond and r.seat == seat]
            add(f"### Road decisions — {cond}, seat {seat}")
            add("")
            out.extend(_pct_table(sub, ROAD_METRICS))
            add("")

    add("### Pair metrics at decision 2, by condition and seat")
    add("")
    for cond in ("greedy", "diverse"):
        for seat in (0, 1):
            sub = [r for r in d2 if r.condition == cond and r.seat == seat]
            add(f"**{cond}, seat {seat}**")
            add("")
            out.extend(_pct_table(sub, PAIR_METRICS))
            add("")

    add("## Ore-substitution rate")
    add("")
    add(
        "Fraction of **settlement decisions** where the chosen candidate has "
        "`ore_pips ≤ 1` while some legal alternative has `ore_pips ≥ 3` and "
        "`|pip_sum(alt) - pip_sum(chosen)| ≤ 2`."
    )
    add("")
    add("| slice | n | substitutions | rate | Wilson 95% CI |")
    add("|---|---:|---:|---:|---|")
    slices: list[tuple[str, list[DecisionRecord]]] = [("all settlement decisions", settle)]
    slices += [("decision 0 (settlement #1)", d0), ("decision 2 (settlement #2)", d2)]
    for cond in ("greedy", "diverse"):
        slices.append((f"condition = {cond}", [r for r in settle if r.condition == cond]))
    for seat in (0, 1):
        slices.append((f"seat {seat}", [r for r in settle if r.seat == seat]))
    for label, grp in slices:
        st = ore_substitution(grp)
        n_dec = int(st["n_settlement_decisions"])
        n_sub = int(st["n_substitutions"])
        if n_dec == 0:
            continue
        ci = wilson_interval(wins=n_sub, n=n_dec, alpha=0.05)
        add(
            f"| {label} | {n_dec} | {n_sub} | {n_sub / n_dec:.4f} | "
            f"[{ci.lower:.4f}, {ci.upper:.4f}] |"
        )
    add("")
    st_all = ore_substitution(settle)
    n_av = int(st_all["n_alternative_available"])
    k_av = int(st_all["n_substitutions"])
    add("Conditional variant — denominator restricted to decisions where a qualifying")
    add("ore-rich alternative actually existed:")
    add("")
    add("| slice | n (alternative existed) | substitutions | rate | Wilson 95% CI |")
    add("|---|---:|---:|---:|---|")
    if n_av:
        ci = wilson_interval(wins=k_av, n=n_av, alpha=0.05)
        add(
            f"| all settlement decisions | {n_av} | {k_av} | {k_av / n_av:.4f} | "
            f"[{ci.lower:.4f}, {ci.upper:.4f}] |"
        )
    add("")
    add(
        f"A qualifying ore-rich alternative existed at {n_av} / "
        f"{int(st_all['n_settlement_decisions'])} = "
        f"{n_av / max(1, int(st_all['n_settlement_decisions'])):.4f} of settlement decisions."
    )
    add("")

    add("## Road argmax rates")
    add("")
    add("A chosen road that **ties** the maximum counts as argmax.")
    add("")
    add("| slice | metric | n | chosen = argmax | rate | Wilson 95% CI | n with unique max |")
    add("|---|---|---:|---:|---:|---|---:|")
    road_slices: list[tuple[str, list[DecisionRecord]]] = [("all road decisions", list(roads))]
    road_slices += [
        ("road #1 (d1)", [r for r in roads if r.decision_index == 1]),
        ("road #2 (d3)", [r for r in roads if r.decision_index == 3]),
    ]
    for cond in ("greedy", "diverse"):
        road_slices.append((f"condition = {cond}", [r for r in roads if r.condition == cond]))
    for seat in (0, 1):
        road_slices.append((f"seat {seat}", [r for r in roads if r.seat == seat]))
    for label, grp in road_slices:
        for metric in ("road_2hop_pip_max", "road_breadth"):
            n, hits, uniq = road_argmax_rate(grp, metric)
            if n == 0:
                continue
            ci = wilson_interval(wins=hits, n=n, alpha=0.05)
            add(
                f"| {label} | `{metric}` | {n} | {hits} | {hits / n:.4f} | "
                f"[{ci.lower:.4f}, {ci.upper:.4f}] | {uniq} |"
            )
    add("")

    add("## Shared / non-shared hex partition (road decisions)")
    add("")
    add(
        "For each road decision with ≥2 legal candidates, `b` = the best alternative to "
        "the chosen road `c` by `road_2hop_pip_max`. The table asks where `b`'s extra "
        "pip mass sits: on the **non-shared** hex (`adj(v2) - adj(v1)`, the hex only the "
        "road reaches) or on the **shared** hexes (`adj(v1) ∩ adj(v2)`, already touched "
        "by the settlement)."
    )
    add("")
    add("The four classes are mutually exclusive and sum to `n`.")
    add("")
    add(
        "| slice | n | richer on NON-SHARED only | richer on SHARED only | richer on both | "
        "not richer |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    for label, grp in road_slices:
        sp = shared_partition(grp)
        n = sp["n_road_decisions_with_alternative"]
        if n == 0:
            continue
        add(
            f"| {label} | {n} | {sp['alt_richer_on_nonshared_only']} | "
            f"{sp['alt_richer_on_shared_only']} | {sp['alt_richer_on_both']} | "
            f"{sp['alt_not_richer']} |"
        )
    add("")
    add("Ore subset — same partition, asking where the best alternative's EXTRA ORE")
    add("sits (`nonshared_ore_pips` / `shared_ore_pips`, `b` minus `c`):")
    add("")
    add(
        "| slice | n | ore gain on NON-SHARED only | ore gain on SHARED only | "
        "ore gain on both | no ore gain |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    for label, grp in road_slices:
        sp = shared_partition(grp)
        n = sp["n_road_decisions_with_alternative"]
        if n == 0:
            continue
        add(
            f"| {label} | {n} | {sp['alt_gains_ore_on_nonshared_only']} | "
            f"{sp['alt_gains_ore_on_shared_only']} | {sp['alt_gains_ore_on_both']} | "
            f"{sp['alt_gains_no_ore']} |"
        )
    add("")

    add("## Realised opening pairs (the settlement pair the policy actually built)")
    add("")
    add(
        "| slice | n | mean pair_pip_sum | mean pair_ore_pips | frac has ore | "
        "frac city-self-sufficient | median exp. rolls to city (defined only) | "
        "mean pair_max_ore_lump | mean pair_spread |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    game_slices: list[tuple[str, list[dict[str, Any]]]] = [("all games", list(games))]
    for cond in ("greedy", "diverse"):
        game_slices.append((f"condition = {cond}", [g for g in games if g["condition"] == cond]))
    for seat in (0, 1):
        game_slices.append((f"seat {seat}", [g for g in games if g["seat"] == seat]))
    for label, g_grp in game_slices:
        if not g_grp:
            continue
        rp = [g["realised_pair"] for g in g_grp]
        rolls_def = [p["pair_exp_rolls_to_city"] for p in rp if p["pair_exp_rolls_to_city"]]
        add(
            f"| {label} | {len(rp)} | {_mean([p['pair_pip_sum'] for p in rp]):.2f} | "
            f"{_mean([p['pair_ore_pips'] for p in rp]):.2f} | "
            f"{_mean([1.0 if p['pair_ore_pips'] > 0 else 0.0 for p in rp]):.4f} | "
            f"{_mean([p['pair_city_self_sufficient'] for p in rp]):.4f} | "
            f"{(statistics.median(rolls_def) if rolls_def else float('nan')):.2f} "
            f"(n={len(rolls_def)}) | "
            f"{_mean([p['pair_max_ore_lump'] for p in rp]):.2f} | "
            f"{_mean([p['pair_spread'] for p in rp]):.2f} |"
        )
    add("")
    all_rp = [g["realised_pair"] for g in games]
    ci_ore = wilson_interval(
        wins=sum(1 for p in all_rp if p["pair_ore_pips"] > 0), n=len(all_rp), alpha=0.05
    )
    ci_ss = wilson_interval(
        wins=sum(1 for p in all_rp if p["pair_city_self_sufficient"] > 0),
        n=len(all_rp),
        alpha=0.05,
    )
    add(
        f"Wilson 95% CIs over all {len(all_rp)} games — pair touches ore: "
        f"[{ci_ore.lower:.4f}, {ci_ore.upper:.4f}]; pair is city-self-sufficient: "
        f"[{ci_ss.lower:.4f}, {ci_ss.upper:.4f}]."
    )
    add("")
    add("Distribution of `pair_ore_pips` over the realised pairs:")
    add("")
    add("| pair_ore_pips | count | fraction |")
    add("|---:|---:|---:|")
    hist = Counter(int(p["pair_ore_pips"]) for p in all_rp)
    for k_ in sorted(hist):
        add(f"| {k_} | {hist[k_]} | {hist[k_] / len(all_rp):.4f} |")
    add("")
    add("Distribution of `pair_max_ore_lump` (largest single-dice-number ore payout):")
    add("")
    add("| pair_max_ore_lump | count | fraction |")
    add("|---:|---:|---:|")
    hist2 = Counter(int(p["pair_max_ore_lump"]) for p in all_rp)
    for k_ in sorted(hist2):
        add(f"| {k_} | {hist2[k_]} | {hist2[k_] / len(all_rp):.4f} |")
    add("")

    add("## LIMITATIONS — what this sweep does NOT measure")
    add("")
    add(
        "1. **No outcome, no counterfactual win rate.** Every number here scores the "
        "opening against hand-defined board metrics. Not one game was played past the "
        "setup phase. Nothing here establishes that a candidate with a better "
        "percentile would have won more; the metrics are assumptions about what a good "
        "opening is, not measurements of value."
    )
    add(
        "2. **No value-head or policy-confidence signal.** Only the argmax identity is "
        "recorded; the margin between the chosen and runner-up candidate, the head's "
        "entropy and the value head's estimate are not."
    )
    add(
        "3. **Ports are not scored.** No metric references `BoardVertex.port`, so a "
        "candidate on a 2:1 ore port and one on open coast are treated identically. "
        "Port access changes the real cost of the 4:1 trades referenced in the "
        "motivating observation."
    )
    add(
        "4. **Dev cards, robber play, largest army and longest road are out of scope.** "
        "The metrics cover production geometry and expansion room only."
    )
    add(
        "5. **`pair_exp_rolls_to_city` is a rate approximation**, not an expectation "
        "(see the definition table): it ignores 7s, the robber, the 9-card discard, "
        "spending, the finite 19-per-resource bank and the variance of hitting times. "
        "It is undefined whenever the pair produces no ore or no wheat, and those "
        "cases are excluded from its statistics rather than imputed — so its median is "
        "conditioned on a self-sufficient pair and is NOT comparable across slices "
        "with different self-sufficiency rates."
    )
    add(
        "6. **Expansion metrics ignore resource costs and tempo.** `exp_d2/3/4` and "
        "`road_breadth` count reachable legal sites; they do not model whether the "
        "player can afford the roads, nor who arrives first."
    )
    add(
        "7. **The opponent-cut dimension is only partially captured.** BFS refuses to "
        "expand through opponent-owned vertices, so present blocking is reflected, but "
        "no metric simulates *future* opponent road-building or robber placement."
    )
    add(
        "8. **Territory / contested-middle is proxied by `centrality_dist` and "
        "`pair_spread` alone.** Neither measures control of the middle relative to the "
        "opponent's settlements."
    )
    add(
        "9. **Percentile ranks are relative to the legal set, not to an absolute "
        "standard.** On a board where every legal vertex is poor, a 99th-percentile "
        "choice is still poor; the raw per-candidate values in the JSON are needed to "
        "separate the two."
    )
    add(
        "10. **Road decisions have ≤3 candidates**, so their percentile distributions "
        "are coarse and heavily tied; argmax rates carry the signal."
    )
    add(
        "11. **Single checkpoint, single architecture.** One champion "
        "(`ckpt_000000500.pt`) under argmax decoding. No comparison against another "
        "checkpoint, a human corpus, a heuristic baseline or a search-augmented agent "
        "is made here, so no number in this report is a *relative* strength claim."
    )
    add(
        "12. **The two conditions are not independent samples of the same population.** "
        "The diverse condition perturbs only the opponent's setup settlement; at seat 0 "
        "decision 0 the two conditions are identical by construction (see the n-per-cell "
        "note), so pooled statistics double-count those decisions."
    )
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop trailing blank lines so the file ends in exactly one newline and
    # pre-commit's end-of-file-fixer leaves regenerated reports alone.
    while out and not out[-1]:
        out.pop()
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--seeds", type=int, default=200, help="boards, seeds 0..N-1")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device(args.device)

    t_start = time.time()
    policy = load_policy(args.checkpoint, device)
    records, games = run_sweep(
        policy,
        device,
        n_seeds=int(args.seeds),
        conditions=("greedy", "diverse"),
        seats=(0, 1),
    )
    wall = time.time() - t_start

    meta = {
        "checkpoint": str(args.checkpoint),
        "n_boards": int(args.seeds),
        "seats": "0 (drafts first), 1 (drafts second)",
        "conditions": (
            "greedy (opponent argmax); diverse (opponent setup settlement ~ Uniform(top-8))"
        ),
        "decoding": "deterministic argmax under the legal mask",
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "games": len(games),
        "decisions": len(records),
        "wall_clock_seconds": round(wall, 1),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "meta": meta,
                "metric_direction": METRIC_DIRECTION,
                "records": [r.to_json() for r in records],
                "games": list(games),
            },
            fh,
        )

    write_report(args.report_out, records=records, games=games, meta=meta)
    print(f"wrote {args.json_out} ({args.json_out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {args.report_out}")
    print(f"wall clock: {wall:.1f}s over {len(games)} games / {len(records)} decisions")


if __name__ == "__main__":
    main()
