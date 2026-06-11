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


# ---------------------------------------------------------------------------
# check_longest_road: global recompute + acquire / transfer / tie / REVOKE
# ---------------------------------------------------------------------------


def _two_player_game(roads0, roads1, owners):
    game = catanGame(render_mode=None)
    p0, p1 = list(game.playerQueue.queue)[:2]
    p0.buildGraph["ROADS"], p1.buildGraph["ROADS"] = list(roads0), list(roads1)
    p0.longestRoadFlag = p1.longestRoadFlag = False
    p0.victoryPoints = p1.victoryPoints = 0
    game.board = _Board(owners)  # check_longest_road uses self.board
    return game, p0, p1


_LINE6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  # 5 segments -> length 5
_EMPTY_OWNERS = {v: _Vtx(None) for v in range(7)}


def test_acquire_longest_road() -> None:
    game, p0, p1 = _two_player_game(_LINE6, [], dict(_EMPTY_OWNERS))
    game.check_longest_road(p0)
    assert p0.longestRoadFlag and p0.victoryPoints == 2
    assert not p1.longestRoadFlag and p0.maxRoadLength == 5  # recomputed


def test_below_five_never_awarded() -> None:
    game, p0, _p1 = _two_player_game([(0, 1), (1, 2), (2, 3)], [], dict(_EMPTY_OWNERS))
    game.check_longest_road(p0)
    assert not p0.longestRoadFlag and p0.victoryPoints == 0


def test_transfer_to_strictly_longer() -> None:
    game, p0, p1 = _two_player_game(_LINE6, [], dict(_EMPTY_OWNERS))
    game.check_longest_road(p0)  # p0 holds at 5
    p1.buildGraph["ROADS"] = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]  # 6
    game.board = _Board({v: _Vtx(None) for v in range(7)} | {v: _Vtx(None) for v in range(10, 17)})
    game.check_longest_road(p1)
    assert p1.longestRoadFlag and p1.victoryPoints == 2
    assert not p0.longestRoadFlag and p0.victoryPoints == 0  # lost the 2 VP


def test_tie_does_not_transfer_from_holder() -> None:
    game, p0, p1 = _two_player_game(_LINE6, [], dict(_EMPTY_OWNERS))
    game.check_longest_road(p0)  # p0 holds at 5
    p1.buildGraph["ROADS"] = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]  # ties at 5
    game.board = _Board({v: _Vtx(None) for v in range(7)} | {v: _Vtx(None) for v in range(10, 16)})
    game.check_longest_road(p1)
    assert p0.longestRoadFlag and not p1.longestRoadFlag  # holder keeps it on a tie


def test_revoke_when_opponent_settlement_splits_below_five() -> None:
    # p0 holds LR at length 5; p1 settles at vertex 3, splitting p0's road into
    # 3 + 2 -> p0 drops to 3 -> LR revoked, going to nobody (p1 has no road).
    game, p0, p1 = _two_player_game(_LINE6, [], dict(_EMPTY_OWNERS))
    game.check_longest_road(p0)
    assert p0.longestRoadFlag and p0.victoryPoints == 2
    game.board = _Board(dict(_EMPTY_OWNERS) | {3: _Vtx(p1)})  # opponent splits at vertex 3
    game.check_longest_road(p1)
    assert not p0.longestRoadFlag and p0.victoryPoints == 0  # revoked
    assert p0.maxRoadLength == 3  # recomputed (no longer stale at 5)


def test_split_transfers_to_opponent_with_five() -> None:
    # p0 holds at 6; a split drops p0 to 4 while p1 has 5 -> transfers to p1.
    line7 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # length 6
    p1roads = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]  # length 5
    owners = {v: _Vtx(None) for v in range(7)} | {v: _Vtx(None) for v in range(10, 16)}
    game, p0, p1 = _two_player_game(line7, p1roads, owners)
    game.check_longest_road(p0)  # p0 holds at 6
    assert p0.longestRoadFlag
    # opponent settles at vertex 5 -> p0 splits into 5 (0..5) + 1 (5..6)... actually
    # breaks AT 5: 0-1-2-3-4-5 stops at 5 (len 5) but 5 is opp -> can't pass; recompute.
    game.board = _Board(dict(owners) | {2: _Vtx(p1)})  # split p0 at vertex 2 -> 2 + 4 = max 4
    game.check_longest_road(p1)
    assert p1.longestRoadFlag and p1.victoryPoints == 2
    assert not p0.longestRoadFlag and p0.victoryPoints == 0
