"""Opening-archetype featurizer tests (step6 §2.1, frozen v5.2 spec).

Pins the frozen contract: the 5-vector pip-share by resource (Charlesworth order),
total pips, port-slot adjacency, and the 5 buckets in their fixed precedence
(ORE_ENGINE ≥0.45 / WOOD_BRICK ≥0.45 / PORT_LED / BALANCED_HIGH ≥26 pips /
BALANCED_LOW), plus the histogram + Shannon-entropy helpers. Boundary tests bracket
0.45 and 26; the pair-share tests pin that "share" is the named PAIR-SUM only (a
lone non-pair resource never names a bucket); port adjacency is checked against the
committed :mod:`topology`. Measurement-only per §2.1 — nothing here touches
training or seed selection.
"""

from __future__ import annotations

import math
from typing import Any

from catan_rl.human_data.opening_archetypes import (
    BALANCED_HIGH_MIN_PIPS,
    PAIR_SHARE_THRESHOLD,
    RESOURCE_ORDER_CW,
    OpeningArchetype,
    archetype_entropy,
    archetype_histogram,
    classify_archetype,
    featurize_opening,
    opening_pips_by_resource,
)
from catan_rl.human_data.topology import load_topology

# The re-snapped game-1 golden board (desert=11), mirrored from test_placement_order.
_GAME1_HEXES: tuple[dict[str, Any], ...] = (
    {"hex_id": 0, "resource": "SHEEP", "number": 11},
    {"hex_id": 1, "resource": "BRICK", "number": 9},
    {"hex_id": 2, "resource": "WHEAT", "number": 10},
    {"hex_id": 3, "resource": "WHEAT", "number": 3},
    {"hex_id": 4, "resource": "BRICK", "number": 6},
    {"hex_id": 5, "resource": "SHEEP", "number": 5},
    {"hex_id": 6, "resource": "ORE", "number": 4},
    {"hex_id": 7, "resource": "WOOD", "number": 6},
    {"hex_id": 8, "resource": "SHEEP", "number": 2},
    {"hex_id": 9, "resource": "WOOD", "number": 5},
    {"hex_id": 10, "resource": "BRICK", "number": 8},
    {"hex_id": 11, "resource": "DESERT", "number": None},
    {"hex_id": 12, "resource": "SHEEP", "number": 4},
    {"hex_id": 13, "resource": "WOOD", "number": 11},
    {"hex_id": 14, "resource": "WHEAT", "number": 12},
    {"hex_id": 15, "resource": "ORE", "number": 9},
    {"hex_id": 16, "resource": "ORE", "number": 10},
    {"hex_id": 17, "resource": "WOOD", "number": 8},
    {"hex_id": 18, "resource": "WHEAT", "number": 3},
)


def _pips(**kw: int) -> dict[str, int]:
    """Build a per-resource pip mapping over :data:`RESOURCE_ORDER_CW` (default 0)."""
    return {r: kw.get(r, 0) for r in RESOURCE_ORDER_CW}


# --------------------------------------------------------------------------- #
# classify_archetype — precedence + boundaries (the frozen decision function)  #
# --------------------------------------------------------------------------- #


def test_precedence_ore_engine_wins_when_both_pairs_clear() -> None:
    # WOOD+BRICK = 10/20 = 0.5 and ORE+WHEAT = 10/20 = 0.5 both clear 0.45;
    # fixed precedence assigns ORE_ENGINE (checked first).
    pips = _pips(WOOD=10, ORE=10)
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.ORE_ENGINE


def test_wood_brick_when_only_wood_brick_pair_clears() -> None:
    # ORE+WHEAT = 0; WOOD+BRICK = 13/21 ≈ 0.619 clears.
    pips = _pips(WOOD=4, BRICK=9, WHEAT=3, ORE=3, SHEEP=2)
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.WOOD_BRICK


def test_ore_engine_boundary_exactly_0_45_is_inclusive() -> None:
    # ORE+WHEAT = 9/20 = 0.45 exactly → ORE_ENGINE (>= is inclusive).
    pips = _pips(ORE=9, SHEEP=11)
    ore_wheat = (pips["ORE"] + pips["WHEAT"]) / sum(pips.values())
    assert ore_wheat == PAIR_SHARE_THRESHOLD
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.ORE_ENGINE


def test_ore_engine_boundary_just_below_0_45_does_not_fire() -> None:
    # ORE+WHEAT = 11/25 = 0.44 < 0.45 → not ORE_ENGINE.
    pips = _pips(ORE=11, SHEEP=14)
    assert (pips["ORE"] + pips["WHEAT"]) / sum(pips.values()) < PAIR_SHARE_THRESHOLD
    assert classify_archetype(pips, port_adjacent=False) is not OpeningArchetype.ORE_ENGINE


def test_ore_engine_boundary_just_above_0_45_fires() -> None:
    # ORE+WHEAT = 12/25 = 0.48 ≥ 0.45 → ORE_ENGINE.
    pips = _pips(ORE=12, SHEEP=13)
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.ORE_ENGINE


def test_wood_brick_boundary_exactly_0_45_is_inclusive() -> None:
    pips = _pips(WOOD=9, SHEEP=11)  # WOOD+BRICK = 9/20 = 0.45
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.WOOD_BRICK


# --------------------------------------------------------------------------- #
# pair-share semantics — "share" is the named PAIR-SUM only                    #
# --------------------------------------------------------------------------- #


def test_lone_non_pair_resource_at_half_does_not_trigger_a_pair_bucket() -> None:
    # SHEEP is in NEITHER named pair. A SHEEP share of 0.5 clears no pair sum, so
    # no pair bucket fires — it falls through to PORT_LED / BALANCED.
    pips = _pips(SHEEP=10, WHEAT=5, WOOD=5)  # total 20, SHEEP=0.5
    assert (pips["ORE"] + pips["WHEAT"]) / sum(pips.values()) < PAIR_SHARE_THRESHOLD
    assert (pips["WOOD"] + pips["BRICK"]) / sum(pips.values()) < PAIR_SHARE_THRESHOLD
    result = classify_archetype(pips, port_adjacent=False)
    assert result is OpeningArchetype.BALANCED_LOW


def test_pair_member_single_resource_at_half_does_trigger_its_pair() -> None:
    # Contrast: WHEAT IS in the ore pair, so a WHEAT-only 0.5 makes the PAIR sum
    # ORE+WHEAT = 0.5 ≥ 0.45 → ORE_ENGINE fires.
    pips = _pips(WHEAT=10, SHEEP=10)  # total 20, ORE+WHEAT = 0.5
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.ORE_ENGINE


# --------------------------------------------------------------------------- #
# PORT_LED / BALANCED_HIGH / BALANCED_LOW + the 26-pip boundary                #
# --------------------------------------------------------------------------- #


def test_port_led_when_port_adjacent_and_no_pair_clears() -> None:
    pips = _pips(WOOD=4, BRICK=4, WHEAT=4, ORE=4, SHEEP=4)  # total 20, pairs 0.4
    assert classify_archetype(pips, port_adjacent=True) is OpeningArchetype.PORT_LED
    # same pips, not port-adjacent, total < 26 → BALANCED_LOW.
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.BALANCED_LOW


def test_port_led_takes_precedence_over_balanced_high() -> None:
    pips = _pips(WOOD=6, BRICK=6, WHEAT=6, ORE=6, SHEEP=6)  # total 30, pairs 0.4
    assert classify_archetype(pips, port_adjacent=True) is OpeningArchetype.PORT_LED
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.BALANCED_HIGH


def test_balanced_high_boundary_exactly_26_is_inclusive() -> None:
    pips = _pips(WOOD=5, BRICK=5, WHEAT=5, ORE=5, SHEEP=6)  # total 26, pairs < 0.45
    assert sum(pips.values()) == BALANCED_HIGH_MIN_PIPS
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.BALANCED_HIGH


def test_balanced_low_just_below_26() -> None:
    pips = _pips(WOOD=5, BRICK=5, WHEAT=5, ORE=5, SHEEP=5)  # total 25, pairs 0.4
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.BALANCED_LOW


def test_zero_pip_is_balanced_low_directly_even_when_port_adjacent() -> None:
    pips = _pips()  # all zero
    # Port adjacency cannot rescue a zero-pip opening (§2.1 shortcut).
    assert classify_archetype(pips, port_adjacent=True) is OpeningArchetype.BALANCED_LOW
    assert classify_archetype(pips, port_adjacent=False) is OpeningArchetype.BALANCED_LOW


# --------------------------------------------------------------------------- #
# featurize_opening — geometry against the committed topology                  #
# --------------------------------------------------------------------------- #


def test_featurize_golden_thephantom_opening_ore_engine() -> None:
    topology = load_topology()
    # ThePhantom's game-1 opening (settlements 1, 19), hand-verified golden board.
    feats = featurize_opening((1, 19), _GAME1_HEXES, topology)
    # pips: WHEAT 5 (hex2=3,hex3=2), ORE 6 (hex6=3,hex16=3), SHEEP 6 (hex0=2,hex5=4).
    assert feats.total_pips == 17
    idx = {r: i for i, r in enumerate(RESOURCE_ORDER_CW)}
    assert math.isclose(feats.pip_share[idx["ORE"]], 6 / 17)
    assert math.isclose(feats.pip_share[idx["WHEAT"]], 5 / 17)
    assert math.isclose(feats.pip_share[idx["SHEEP"]], 6 / 17)
    assert feats.pip_share[idx["WOOD"]] == 0.0
    assert feats.pip_share[idx["BRICK"]] == 0.0
    assert math.isclose(sum(feats.pip_share), 1.0)
    assert feats.port_adjacent is False
    assert feats.archetype is OpeningArchetype.ORE_ENGINE


def test_featurize_golden_rayman_opening_wood_brick() -> None:
    topology = load_topology()
    # rayman147's game-1 opening (settlements 3, 11 — 11 is desert, 0 pips).
    feats = featurize_opening((3, 11), _GAME1_HEXES, topology)
    # pips: WOOD 4 (hex9), BRICK 9 (hex1=4,hex10=5), WHEAT 3 (hex2), ORE 3 (hex6),
    # SHEEP 2 (hex0). total 21; WOOD+BRICK = 13/21 ≈ 0.619.
    assert feats.total_pips == 21
    assert feats.archetype is OpeningArchetype.WOOD_BRICK
    assert feats.port_adjacent is False


def test_featurize_symmetric_in_settlement_order() -> None:
    topology = load_topology()
    assert featurize_opening((1, 19), _GAME1_HEXES, topology) == featurize_opening(
        (19, 1), _GAME1_HEXES, topology
    )


def test_opening_pips_double_counts_a_shared_hex_per_settlement() -> None:
    topology = load_topology()
    # Vertices 0 and 2 both border hex 0 (SHEEP, 11 → 2 pips). Production is
    # per-settlement, so the shared hex contributes to BOTH.
    assert 0 in topology.vertex_adjacent_hexes[0]
    assert 0 in topology.vertex_adjacent_hexes[2]
    pips = opening_pips_by_resource((0, 2), _GAME1_HEXES, topology)
    # hex 0 SHEEP counted twice = 4; verify > single-settlement contribution.
    single = opening_pips_by_resource((0,), _GAME1_HEXES, topology)
    assert pips["SHEEP"] == single["SHEEP"] + 2


def test_featurize_port_slot_adjacency_against_committed_topology() -> None:
    topology = load_topology()
    # Slot 0 sits on vertices {25, 26}; a settlement there is port-adjacent.
    port_vertex = topology.port_slots[0]["vertices"][0]
    assert port_vertex == 25
    feats = featurize_opening((25, 1), _GAME1_HEXES, topology)
    assert feats.port_adjacent is True
    # Neither game-1 golden settlement sits on a port slot.
    assert featurize_opening((1, 19), _GAME1_HEXES, topology).port_adjacent is False


def test_featurize_is_deterministic() -> None:
    topology = load_topology()
    a = featurize_opening((1, 19), _GAME1_HEXES, topology)
    b = featurize_opening((1, 19), _GAME1_HEXES, topology)
    assert a == b
    assert a.pip_share == b.pip_share


# --------------------------------------------------------------------------- #
# histogram + Shannon entropy                                                  #
# --------------------------------------------------------------------------- #


def test_histogram_always_carries_all_five_buckets() -> None:
    hist = archetype_histogram([OpeningArchetype.ORE_ENGINE, OpeningArchetype.ORE_ENGINE])
    assert set(hist) == set(OpeningArchetype)
    assert hist[OpeningArchetype.ORE_ENGINE] == 2
    assert hist[OpeningArchetype.WOOD_BRICK] == 0


def test_entropy_degenerate_histogram_is_zero() -> None:
    hist = archetype_histogram([OpeningArchetype.ORE_ENGINE] * 10)
    assert archetype_entropy(hist) == 0.0


def test_entropy_empty_histogram_is_zero() -> None:
    assert archetype_entropy(archetype_histogram([])) == 0.0


def test_entropy_uniform_histogram_is_log2_five() -> None:
    hist = archetype_histogram([b for b in OpeningArchetype for _ in range(2)])
    assert math.isclose(archetype_entropy(hist), math.log2(5))


def test_entropy_is_between_zero_and_ceiling() -> None:
    hist = archetype_histogram(
        [OpeningArchetype.ORE_ENGINE] * 8
        + [OpeningArchetype.WOOD_BRICK] * 1
        + [OpeningArchetype.PORT_LED] * 1
    )
    ent = archetype_entropy(hist)
    assert 0.0 < ent < math.log2(5)
