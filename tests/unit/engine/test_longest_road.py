"""Longest-road = longest TRAIL (each road segment used once; vertices may
repeat, so a closed loop counts in full), broken at an opponent settlement/city.

Pins the engine against a brute-force trail reference + hand-checked fixtures,
including the ringed-hex + spur = 7 case and the opponent-break split. Replaces
the prior DFS, which tracked visited VERTICES (simple-path semantics) and shared
one mutable set across recursive branches (undercounting branchy networks).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from catan_rl.engine.game import catanGame


class _Vtx:
    def __init__(self, owner: object = None) -> None:
        self.owner = owner


class _Board:
    def __init__(self, owners: dict[object, _Vtx]) -> None:
        self.boardGraph = owners


def _brute_trail(edges: list[tuple[int, int]], opp_vertices: tuple[int, ...] = ()) -> int:
    """Reference: longest trail (no repeated edge), stopping when it reaches a
    vertex carrying an opponent building (the segment in still counts)."""
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))
    best = 0

    def dfs(v: int, used: frozenset[int], length: int) -> None:
        nonlocal best
        best = max(best, length)
        if v in opp_vertices:
            return
        for nb, i in adj[v]:
            if i not in used:
                dfs(nb, used | {i}, length + 1)

    for s in adj:
        dfs(s, frozenset(), 0)
    return best


def _engine_len(edges: list[tuple[int, int]], opp_vertices: tuple[int, ...] = ()) -> int:
    game = catanGame(render_mode=None)
    players = list(game.playerQueue.queue)
    me, opp = players[0], players[1]
    me.buildGraph["ROADS"] = list(edges)
    verts = {v for e in edges for v in e}
    owners = {v: _Vtx(opp if v in opp_vertices else None) for v in verts}
    return me.get_road_length(_Board(owners))


_HEXLOOP = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


@pytest.mark.parametrize(
    "edges,opp,expected",
    [
        ([], (), 0),
        ([(0, 1)], (), 1),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], (), 4),  # straight line
        ([(0, 1), (1, 2), (2, 0)], (), 3),  # triangle loop counts fully (trail, not simple-path=2)
        (_HEXLOOP, (), 6),  # ringed hex
        ([*_HEXLOOP, (0, 6)], (), 7),  # ringed hex + spur = 7 (the canonical example)
        ([(0, 1), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4)], (), 6),  # branchy w/ cycle
    ],
)
def test_longest_road_is_trail(edges, opp, expected) -> None:
    assert _engine_len(edges, opp) == expected == _brute_trail(edges, opp)


def test_opponent_settlement_breaks_road() -> None:
    # A-B-C with an opponent building at B: the road is split, longest = 1.
    assert _engine_len([(0, 1), (1, 2)], opp_vertices=(1,)) == 1
    # Opponent at an endpoint doesn't shorten an otherwise-continuous road.
    assert _engine_len([(0, 1), (1, 2)], opp_vertices=(0,)) == 2


def test_own_settlement_does_not_break_road() -> None:
    # The player's own building does NOT break the road (only opponents do).
    game = catanGame(render_mode=None)
    me = next(iter(game.playerQueue.queue))
    me.buildGraph["ROADS"] = [(0, 1), (1, 2)]
    owners = {0: _Vtx(None), 1: _Vtx(me), 2: _Vtx(None)}  # own building at the junction
    assert me.get_road_length(_Board(owners)) == 2
