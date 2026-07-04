"""Human-data video-parsing pipeline (ThePhantom Colonist.io 1v1 openings).

Parses ThePhantom's YouTube 1v1 Colonist.io games into a dataset of human
**openings + outcomes** for two uses (in priority order):

- **(A)** an external opening *scoreboard* to measure champion ``v8``'s opening
  play (internal metrics can't — v8 grades its own homework), and
- **(B)** diverse human opening *seeds* to break v8's self-play
  opening-diversity collapse.

This package is **measurement / seeding only**: it changes no engine rule and is
never a deploy policy. It is CPU-only and must never import ``gui/`` or the
training path. All board coordinates in :class:`GameRecord` are engine integer
IDs (19 hex / 54 vertex / 72 edge) from the committed :mod:`topology` fixture, so
records drop straight into the RL stack. Resources are string literals
(``"WOOD"``, ``"BRICK"``, ``"WHEAT"``, ``"ORE"``, ``"SHEEP"``, ``"DESERT"``) — see
the build brief ``docs/plans/human_data_pipeline.md`` §5.8 (there is no
``RESOURCES`` enum in ``engine/``).
"""

from catan_rl.human_data.ffmpeg import FFmpegNotFoundError, resolve_ffmpeg
from catan_rl.human_data.orientation import (
    MAX_AFFINE_RESIDUAL_PX,
    MIN_RESOLUTION,
    GlyphClassifierNotValidated,
    assert_glyph_anchor,
    assert_scale_up_orientation_gates,
    granted_multiset_matches_a_settlement,
    granted_resources_under_orientation,
)
from catan_rl.human_data.record import (
    RESOURCE_LITERALS,
    SCHEMA_VERSION,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
    check_road_incidence,
    derive_opponent_strength,
)
from catan_rl.human_data.topology import Topology, load_topology

__all__ = [
    "MAX_AFFINE_RESIDUAL_PX",
    "MIN_RESOLUTION",
    "RESOURCE_LITERALS",
    "SCHEMA_VERSION",
    "FFmpegNotFoundError",
    "GameRecord",
    "GlyphClassifierNotValidated",
    "OpponentStrength",
    "PlayerOpening",
    "Topology",
    "assert_glyph_anchor",
    "assert_scale_up_orientation_gates",
    "check_road_incidence",
    "derive_opponent_strength",
    "granted_multiset_matches_a_settlement",
    "granted_resources_under_orientation",
    "load_topology",
    "resolve_ffmpeg",
]
