"""Strategy archetype enum recorded per labeled scenario.

The user declares an archetype before each setup pick so the labeling
data can be sliced by strategic intent later (e.g., "give me only OWS
openings"). String values are part of the on-disk JSONL schema —
renaming a value retroactively breaks every label that referenced it.
"""

from __future__ import annotations

from enum import StrEnum


class Archetype(StrEnum):
    """Strategy archetype declared by the labeler.

    Values are exact strings serialised to JSONL. Single shortcut keys
    for keyboard-driven labeling (see plan §C carry-forward) map to
    these enum members via the first letter of the value.
    """

    BALANCED = "balanced"
    OWS = "OWS"
    OWS_HYBRID = "OWS_hybrid"
    ROAD_BUILDER = "road_builder"
    OTHER = "other"
