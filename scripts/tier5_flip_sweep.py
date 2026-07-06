#!/usr/bin/env python3
"""TIER-5 joint-D6 flip sweep + consensus-supply audit over a harvested corpus.

Measures the **joint-flip leakage residual** the human-corpus plan's §joint-flip
question needs. The glyph anchor (:func:`assert_glyph_anchor`) is the sole defence
against a jointly-flipped board+openings (the provenance desert-binding cannot see
a joint flip — both stages flip together, so the desert stamps still agree). Its
discriminating power is the granted-card multiset: under the correct orientation
the 2nd (resource-granting) settlement's 3-hex adjacency resource multiset equals
the log-read granted cards; a D6 relabel of the openings moves the settlement onto
a different vertex whose adjacency multiset *usually* diverges — but the committed
board has only **28 distinct 3-hex resource multisets across 54 vertices**
(38/54 vertices share a multiset with ≥1 other), so a relabel that lands on a
collision-partner vertex slips through (a false-accept = leakage).

The sweep, per ACCEPTED + order-established game, relabels the openings by each of
the 11 non-identity D6 elements (holding the true board read and the log-grant
ground truth fixed — the presentation a jointly-flipped candidate read makes to the
anchor) and counts how many :func:`assert_glyph_anchor` calls REJECT, under:

  (i)  the current EITHER-settlement matching (production ``assert_glyph_anchor``),
  (ii) 2nd-SETTLEMENT-ONLY matching (the plan's fix — the record now carries log
       placement order, so ``settlements[1]`` is known and only it is checked).

Higher reject count = better firewall; leakage = ``11·n_games − rejects``. Mode (ii)
should reject at least as many as (i) (matching only the granting settlement halves
the collision surface). Numbers are reported as ``rejects/(11·n)`` per mode.

The true grant per player is reconstructed from the accepted record as the adjacency
multiset of ``settlements[1]`` (the 2nd/granting settlement). This is exact: the
record was ACCEPTED, so the anchor already matched the log-read grant against a
settlement, and for an order-established record that settlement IS ``settlements[1]``
by the placement-order contract — so the reconstruction equals the original log read.

CPU-only; no ``gui/`` import (the D6 permutation tables come from
``catan_rl.augmentation.symmetry_tables``, which lazily builds from the pure engine
board geometry — no pixel/GUI surface).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    corner_perm,
    edge_perm,
)
from catan_rl.human_data.orientation import (
    assert_glyph_anchor,
    granted_resources_under_orientation,
)
from catan_rl.human_data.record import (
    PROVENANCE_PLACEMENT_ORDER_ESTABLISHED,
    GameRecord,
    PlayerOpening,
)
from catan_rl.human_data.topology import Topology, load_topology

#: The 11 non-identity D6 elements applied jointly to the openings.
NON_IDENTITY: tuple[int, ...] = tuple(g for g in range(D6_GROUP_SIZE) if g != D6_IDENTITY)


def board_resource_by_hex(record: GameRecord) -> dict[int, str]:
    """``{hex_id: resource}`` for the record's board (the true read, held fixed)."""
    return {int(h["hex_id"]): str(h["resource"]) for h in record.hexes}


def reconstruct_true_grants(record: GameRecord, topology: Topology) -> dict[str, Counter[str]]:
    """The log-read granted multiset per player, reconstructed from ``settlements[1]``.

    Exact for an order-established accepted record: the placement-order contract
    pins ``settlements[1]`` as the 2nd/granting settlement, and acceptance means the
    log grant matched it, so its adjacency multiset equals the original glyph read.
    """
    board = board_resource_by_hex(record)
    grants: dict[str, Counter[str]] = {}
    for player, opening in record.openings.items():
        granting_vertex = opening.settlements[1]
        grants[player] = granted_resources_under_orientation(granting_vertex, board, topology)
    return grants


def flip_openings(openings: dict[str, PlayerOpening], g: int) -> dict[str, PlayerOpening]:
    """Relabel every opening settlement/road by D6 element ``g`` (a joint flip)."""
    cperm = corner_perm(g)
    eperm = edge_perm(g)
    return {
        player: PlayerOpening(
            settlements=tuple(int(cperm[v]) for v in opening.settlements),
            roads=tuple(int(eperm[e]) for e in opening.roads),
        )
        for player, opening in openings.items()
    }


def _matches_either(
    settlements: tuple[int, ...], grant: Counter[str], board: dict[int, str], topology: Topology
) -> bool:
    """True iff the grant equals EITHER settlement's adjacency (mode i semantics)."""
    return any(
        granted_resources_under_orientation(v, board, topology) == grant for v in settlements
    )


def _matches_second_only(
    settlements: tuple[int, ...], grant: Counter[str], board: dict[int, str], topology: Topology
) -> bool:
    """True iff the grant equals the 2nd settlement's adjacency (mode ii semantics)."""
    return granted_resources_under_orientation(settlements[1], board, topology) == grant


def anchor_rejects_either(
    flipped: GameRecord, grants: dict[str, Counter[str]], topology: Topology
) -> bool:
    """Whether production ``assert_glyph_anchor`` (either-settlement) rejects the flip."""
    try:
        assert_glyph_anchor(flipped, grants, topology)
    except ValueError:
        return True
    return False


def anchor_rejects_second_only(
    flipped: GameRecord, grants: dict[str, Counter[str]], topology: Topology
) -> bool:
    """Whether a 2nd-settlement-only anchor rejects the flip (the plan's fix).

    Mirrors ``assert_glyph_anchor`` but matches each player's grant against ONLY
    ``settlements[1]`` (the known granting settlement) instead of either settlement.
    Rejects (returns True) iff any player's grant matches its 2nd settlement's
    adjacency under the flipped orientation for NO player — i.e. the anchor would
    raise.
    """
    board = board_resource_by_hex(flipped)
    for player, grant in grants.items():
        if not _matches_second_only(flipped.openings[player].settlements, grant, board, topology):
            return True
    return False


@dataclass(frozen=True)
class SweepResult:
    """Flip-sweep totals over the accepted, order-established corpus."""

    n_games: int
    n_flips_per_game: int
    reject_either: int
    reject_second_only: int
    per_game: tuple[dict[str, object], ...]

    @property
    def total_flips(self) -> int:
        return self.n_games * self.n_flips_per_game

    @property
    def leak_either(self) -> int:
        return self.total_flips - self.reject_either

    @property
    def leak_second_only(self) -> int:
        return self.total_flips - self.reject_second_only

    def to_dict(self) -> dict[str, object]:
        return {
            "n_games": self.n_games,
            "n_flips_per_game": self.n_flips_per_game,
            "total_flips": self.total_flips,
            "reject_either_over_total": f"{self.reject_either}/{self.total_flips}",
            "reject_second_only_over_total": f"{self.reject_second_only}/{self.total_flips}",
            "leak_either": self.leak_either,
            "leak_second_only": self.leak_second_only,
            "reject_either": self.reject_either,
            "reject_second_only": self.reject_second_only,
            "per_game": list(self.per_game),
        }


def is_order_established(record: GameRecord) -> bool:
    """Whether the record's placement order was established (``settlements[1]`` valid)."""
    return record.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is True


def flip_sweep(records: list[GameRecord], topology: Topology) -> SweepResult:
    """Run the joint-D6 flip sweep over accepted, order-established records."""
    per_game: list[dict[str, object]] = []
    reject_either = 0
    reject_second_only = 0
    n_games = 0
    for record in records:
        if not record.passed_crosscheck or not is_order_established(record):
            continue
        n_games += 1
        grants = reconstruct_true_grants(record, topology)
        g_either = 0
        g_second = 0
        for g in NON_IDENTITY:
            flipped = replace(record, openings=flip_openings(record.openings, g))
            if anchor_rejects_either(flipped, grants, topology):
                g_either += 1
            if anchor_rejects_second_only(flipped, grants, topology):
                g_second += 1
        reject_either += g_either
        reject_second_only += g_second
        per_game.append(
            {
                "video_id": record.video_id,
                "game_index": record.game_index,
                "reject_either": g_either,
                "reject_second_only": g_second,
                "n_flips": len(NON_IDENTITY),
            }
        )
    return SweepResult(
        n_games=n_games,
        n_flips_per_game=len(NON_IDENTITY),
        reject_either=reject_either,
        reject_second_only=reject_second_only,
        per_game=tuple(per_game),
    )


@dataclass(frozen=True)
class BoardLeakage:
    """Per-settlement joint-flip leakage surface of ONE board (openings-independent).

    A game-level flip sweep needs the accepted openings; but the *residual* leakage a
    joint flip leaves is fundamentally a board property: the glyph anchor's only
    discriminator is the 3-hex resource multiset, so a D6 relabel of a settlement onto
    a vertex that SHARES its multiset is undetectable. This characterises that surface
    over ALL 54 vertices × 11 non-identity flips, with no openings required — so it can
    be computed on the real boards the harvest read even when their openings failed.

    - ``leak_pairs`` — (vertex, g) pairs where the settlement MOVES to a different
      vertex (``corner_perm(g)[v] != v``) yet the resource multiset is preserved
      (``multiset(corner_perm(g)[v]) == multiset(v)`` against the fixed board read):
      a single-settlement false accept a 2nd-settlement-only anchor would leak.
      Fixed points (``corner_perm(g)[v] == v`` — the settlement does not move) are
      NOT leaks and are excluded.
    - ``moved_pairs`` — (vertex, g) pairs where the settlement moves (the denominator).
    - ``distinct_multisets`` / ``colliding_vertices`` — the board's collision structure
      (the leakage-bounding property pinned in ``test_glyph_anchor_multiset_collision_rate``).
    """

    desert_hex: int
    leak_pairs: int
    moved_pairs: int
    distinct_multisets: int
    colliding_vertices: int
    n_vertices: int

    @property
    def single_settlement_leak_rate(self) -> float:
        return self.leak_pairs / self.moved_pairs if self.moved_pairs else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "desert_hex": self.desert_hex,
            "leak_pairs": self.leak_pairs,
            "moved_pairs": self.moved_pairs,
            "single_settlement_leak_rate": round(self.single_settlement_leak_rate, 4),
            "distinct_multisets": self.distinct_multisets,
            "colliding_vertices": self.colliding_vertices,
            "n_vertices": self.n_vertices,
        }


def board_leakage_surface(hexes: tuple[dict[str, Any], ...], topology: Topology) -> BoardLeakage:
    """The openings-independent joint-flip leakage surface of one board.

    See :class:`BoardLeakage` for the fields.
    """
    board = {int(h["hex_id"]): str(h["resource"]) for h in hexes}
    desert = next(int(h["hex_id"]) for h in hexes if str(h["resource"]) == "DESERT")
    n = len(topology.vertex_adjacent_hexes)
    multisets = [
        frozenset(granted_resources_under_orientation(v, board, topology).items()) for v in range(n)
    ]
    counts = Counter(multisets)
    colliding = sum(1 for m in multisets if counts[m] > 1)
    leak_pairs = 0
    moved_pairs = 0
    for g in NON_IDENTITY:
        cperm = corner_perm(g)
        for v in range(n):
            moved_to = int(cperm[v])
            if moved_to == v:
                continue  # fixed point — the settlement does not move; not a leak
            moved_pairs += 1
            if multisets[moved_to] == multisets[v]:
                leak_pairs += 1
    return BoardLeakage(
        desert_hex=desert,
        leak_pairs=leak_pairs,
        moved_pairs=moved_pairs,
        distinct_multisets=len(counts),
        colliding_vertices=colliding,
        n_vertices=n,
    )


def load_corpus(path: Path) -> list[GameRecord]:
    """Load accepted ``GameRecord`` rows from a JSONL corpus (tolerant of torn tails)."""
    out: list[GameRecord] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(GameRecord.from_json_line(line))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="path to corpus.jsonl (accepted records)")
    parser.add_argument("--out", default=None, help="write the sweep JSON here")
    args = parser.parse_args(argv)

    topology = load_topology()
    records = load_corpus(Path(args.corpus))
    result = flip_sweep(records, topology)
    payload = result.to_dict()
    # board-level leakage surface over every distinct board in the corpus (openings-
    # independent — usable even when a board's openings failed and it was rejected).
    seen: set[int] = set()
    boards: list[dict[str, object]] = []
    for record in records:
        surface = board_leakage_surface(record.hexes, topology)
        if surface.desert_hex in seen:
            continue
        seen.add(surface.desert_hex)
        boards.append(surface.to_dict())
    payload["board_leakage_surface"] = boards
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
