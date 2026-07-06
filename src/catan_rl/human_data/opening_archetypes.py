"""Opening-archetype featurizer (step6 §2.1, frozen v5.2 spec).

**Scope restriction (user decision, v5.2): the buckets are MEASUREMENT-ONLY.**
They are consumed by exactly two gates — PRE-GATE-0's collapse verdict and
GATE-B3(3)'s diversity criterion — plus descriptive reporting / TensorBoard
dashboards. They **never** enter training, seed selection, or any other gate. The
model learns from the raw opening positions; nothing is ever steered, filtered,
or weighted by bucket. Seed draws are uniform over distinct openings (§5); this
module does not touch that path.

Named ``opening_archetypes`` to avoid colliding with the training-side
``labeling/archetypes.py`` (council rename).

The frozen spec (v4 §2.1, carried verbatim into v5.2):

Features per ``(seat, opening)`` — the seat's two setup settlements:

- **5-vector pip-share by resource** over the two settlements. For each
  settlement, every *adjacent hex* (via the committed :mod:`topology`) contributes
  its number-token **pips** (``2/12→1 … 6/8→5``; desert → 0) to that hex's
  resource. A hex bordered by both settlements is counted once per settlement
  (production is per-settlement). ``pip_share[r] = pips[r] / total_pips``,
  normalized over the five resources. ``total_pips == 0`` (e.g. two all-desert
  settlements) ⟹ **BALANCED_LOW directly** (zero vector, no division).
- **total pips** — the un-normalized sum.
- **port-slot adjacency** — ``True`` iff either settlement sits on one of the 9
  fixed port-slot vertex pairs (``topology.port_slots[*]['vertices']``).

**"Share" is the named PAIR-SUM only** — a single resource never names a bucket.

Buckets, in this **fixed precedence** (first match wins):

1. ``ORE_ENGINE``     — ``share[ORE] + share[WHEAT] >= 0.45``
2. ``WOOD_BRICK``     — ``share[WOOD] + share[BRICK] >= 0.45``
3. ``PORT_LED``       — port-adjacent AND neither pair-share ``>= 0.45``
4. ``BALANCED_HIGH``  — neither pair-share ``>= 0.45`` AND ``total_pips >= 26``
5. ``BALANCED_LOW``   — else (also the ``total_pips == 0`` shortcut)

Plus histogram + Shannon-entropy (base-2 / bits) helpers over the 5 buckets, for
the PRE-GATE-0 collapse verdict and the GATE-B3(3)/dashboards diversity readout.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from catan_rl.human_data.topology import Topology

#: Resource order of the exported 5-vector: Charlesworth-canonical
#: (``RESOURCES_CW`` at the RL boundary, distinct from the engine's alphabetical
#: ``RESOURCES`` — CLAUDE.md rule 6). Re-declared locally (as ``env/hand_tracker``
#: does) so this measurement-only parsing module never imports the policy/torch
#: stack. ``DESERT`` is intentionally absent — a desert hex contributes 0 pips to
#: no resource.
RESOURCE_ORDER_CW: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Number-token → pip-dot count (the standard Catan probability weighting; desert
#: carries no token and contributes 0). Identical to ``board_cv._PIP_FOR`` but
#: re-declared here to keep the featurizer self-contained.
PIP_BY_NUMBER: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

#: Frozen bucket thresholds (v5.2). ``PAIR_SHARE_THRESHOLD`` is inclusive (``>=``);
#: the 26-pip ``BALANCED_HIGH`` floor is inclusive (``>=``). The 26 boundary was
#: sanity-checked against PRE-GATE-0 per §2.1 (any amendment lands pre-corpus).
PAIR_SHARE_THRESHOLD: float = 0.45
BALANCED_HIGH_MIN_PIPS: int = 26


class OpeningArchetype(StrEnum):
    """The 5 measurement-only opening buckets (v5.2 §2.1). ``StrEnum`` so a bucket
    serializes as its name and sorts/keys cleanly in histograms/JSON."""

    ORE_ENGINE = "ORE_ENGINE"
    WOOD_BRICK = "WOOD_BRICK"
    PORT_LED = "PORT_LED"
    BALANCED_HIGH = "BALANCED_HIGH"
    BALANCED_LOW = "BALANCED_LOW"


@dataclass(frozen=True, slots=True)
class OpeningFeatures:
    """Featurization of one ``(seat, opening)`` — the frozen §2.1 feature set."""

    #: pip-share per resource in :data:`RESOURCE_ORDER_CW` (sums to 1.0, or all
    #: zeros when ``total_pips == 0``).
    pip_share: tuple[float, float, float, float, float]
    #: un-normalized total adjacent-hex pips over the two settlements.
    total_pips: int
    #: either settlement on a fixed port-slot vertex pair.
    port_adjacent: bool
    #: the assigned bucket (fixed precedence).
    archetype: OpeningArchetype


def _port_slot_vertices(topology: Topology) -> frozenset[int]:
    """The 18 fixed port-slot vertices (9 slots x 2) as engine vertex IDs."""
    verts: set[int] = set()
    for slot in topology.port_slots:
        for v in slot["vertices"]:
            verts.add(int(v))
    return frozenset(verts)


def opening_pips_by_resource(
    settlements: Sequence[int],
    hexes: Sequence[Mapping[str, Any]],
    topology: Topology,
) -> dict[str, int]:
    """Sum adjacent-hex pips per resource over the opening's settlements.

    Production is per-settlement: a hex bordered by both settlements contributes
    its pips once for each. Desert hexes (``number is None``) contribute 0. Keys
    are exactly :data:`RESOURCE_ORDER_CW` (a resource with no adjacent production
    is present with value 0).
    """
    resource_by_hex: dict[int, str] = {int(h["hex_id"]): str(h["resource"]) for h in hexes}
    number_by_hex: dict[int, int | None] = {
        int(h["hex_id"]): (None if h["number"] is None else int(h["number"])) for h in hexes
    }
    pips: dict[str, int] = {r: 0 for r in RESOURCE_ORDER_CW}
    for vertex in settlements:
        for hex_id in topology.vertex_adjacent_hexes[int(vertex)]:
            resource = resource_by_hex[hex_id]
            if resource not in pips:  # DESERT (or any non-producing hex)
                continue
            number = number_by_hex[hex_id]
            if number is None:
                continue
            pips[resource] += PIP_BY_NUMBER[number]
    return pips


def classify_archetype(
    pips_by_resource: Mapping[str, int],
    *,
    port_adjacent: bool,
) -> OpeningArchetype:
    """Assign the bucket from per-resource pip totals + port adjacency.

    Pure classifier over the frozen precedence (§2.1). ``pips_by_resource`` need
    only carry the :data:`RESOURCE_ORDER_CW` keys (missing keys read as 0). This
    is the testable seam that pins the 0.45 / 26 boundaries without fabricating
    board geometry.
    """
    total = sum(int(pips_by_resource.get(r, 0)) for r in RESOURCE_ORDER_CW)
    if total == 0:
        # No production at all ⇒ BALANCED_LOW directly (no division, and port
        # adjacency cannot rescue a zero-pip opening — §2.1).
        return OpeningArchetype.BALANCED_LOW

    ore_wheat = (
        int(pips_by_resource.get("ORE", 0)) + int(pips_by_resource.get("WHEAT", 0))
    ) / total
    wood_brick = (
        int(pips_by_resource.get("WOOD", 0)) + int(pips_by_resource.get("BRICK", 0))
    ) / total

    if ore_wheat >= PAIR_SHARE_THRESHOLD:
        return OpeningArchetype.ORE_ENGINE
    if wood_brick >= PAIR_SHARE_THRESHOLD:
        return OpeningArchetype.WOOD_BRICK
    # Reaching here ⇒ neither pair-share cleared the threshold.
    if port_adjacent:
        return OpeningArchetype.PORT_LED
    if total >= BALANCED_HIGH_MIN_PIPS:
        return OpeningArchetype.BALANCED_HIGH
    return OpeningArchetype.BALANCED_LOW


def featurize_opening(
    settlements: Sequence[int],
    hexes: Sequence[Mapping[str, Any]],
    topology: Topology,
) -> OpeningFeatures:
    """Featurize one ``(seat, opening)`` — the frozen §2.1 feature set.

    ``settlements`` are the seat's two setup settlement vertex IDs (order
    irrelevant — the featurizer is symmetric in the two settlements). ``hexes`` is
    the board's ``hex_id / resource / number`` records (``GameRecord.hexes`` form).
    """
    pips = opening_pips_by_resource(settlements, hexes, topology)
    total = sum(pips[r] for r in RESOURCE_ORDER_CW)
    port_adjacent = any(int(v) in _port_slot_vertices(topology) for v in settlements)
    if total == 0:
        share = (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        share = tuple(pips[r] / total for r in RESOURCE_ORDER_CW)  # type: ignore[assignment]
    archetype = classify_archetype(pips, port_adjacent=port_adjacent)
    return OpeningFeatures(
        pip_share=share,
        total_pips=total,
        port_adjacent=port_adjacent,
        archetype=archetype,
    )


def archetype_histogram(
    archetypes: Iterable[OpeningArchetype],
) -> dict[OpeningArchetype, int]:
    """Count openings per bucket. Always returns all 5 keys (absent → 0) so the
    histogram shape is stable across corpora / dashboards."""
    counts: Counter[OpeningArchetype] = Counter(archetypes)
    return {bucket: int(counts.get(bucket, 0)) for bucket in OpeningArchetype}


def archetype_entropy(histogram: Mapping[OpeningArchetype, int]) -> float:
    """Shannon entropy (base-2 / bits) of the bucket distribution.

    ``0.0`` for an empty or single-bucket (degenerate) histogram; ``log2(5) ≈
    2.3219`` for a uniform spread over the 5 buckets (the diversity ceiling). This
    is the PRE-GATE-0 collapse readout + the GATE-B3(3)/dashboards diversity
    statistic.
    """
    total = sum(int(v) for v in histogram.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        c = int(count)
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy
