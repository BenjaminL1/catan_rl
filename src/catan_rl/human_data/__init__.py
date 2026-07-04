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

from catan_rl.human_data.ffmpeg import (
    FFmpegNotFoundError,
    resolve_ffmpeg,
    resolve_ffprobe,
)
from catan_rl.human_data.ingest import (
    DEFAULT_DENSE_INTERVAL_S,
    DEFAULT_SPARSE_INTERVAL_S,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    OCR_SECONDS_PER_CROP,
    DecodedFrame,
    ScheduledFrame,
    SubResolutionError,
    TimeWindow,
    VideoDownloadError,
    build_sampling_schedule,
    decode_frames_at,
    download_video,
    estimate_ocr_wall_clock_s,
    ingest_video,
    probe_resolution,
    schedule_ocr_eta_s,
)
from catan_rl.human_data.logparse import (
    LOG_CROP_FRAC,
    LogEvent,
    ParsedLog,
    crop_log,
    ocr_log_crop,
    parse_log,
)
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
from catan_rl.human_data.segment import (
    GameSegment,
    load_strength_manifest,
    manifest_entry,
    ruleset_ok,
    segment_games,
    segment_opponent_strength,
)
from catan_rl.human_data.topology import Topology, load_topology

__all__ = [
    "DEFAULT_DENSE_INTERVAL_S",
    "DEFAULT_SPARSE_INTERVAL_S",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "LOG_CROP_FRAC",
    "MAX_AFFINE_RESIDUAL_PX",
    "MIN_RESOLUTION",
    "OCR_SECONDS_PER_CROP",
    "RESOURCE_LITERALS",
    "SCHEMA_VERSION",
    "DecodedFrame",
    "FFmpegNotFoundError",
    "GameRecord",
    "GameSegment",
    "GlyphClassifierNotValidated",
    "LogEvent",
    "OpponentStrength",
    "ParsedLog",
    "PlayerOpening",
    "ScheduledFrame",
    "SubResolutionError",
    "TimeWindow",
    "Topology",
    "VideoDownloadError",
    "assert_glyph_anchor",
    "assert_scale_up_orientation_gates",
    "build_sampling_schedule",
    "check_road_incidence",
    "crop_log",
    "decode_frames_at",
    "derive_opponent_strength",
    "download_video",
    "estimate_ocr_wall_clock_s",
    "granted_multiset_matches_a_settlement",
    "granted_resources_under_orientation",
    "ingest_video",
    "load_strength_manifest",
    "load_topology",
    "manifest_entry",
    "ocr_log_crop",
    "parse_log",
    "probe_resolution",
    "resolve_ffmpeg",
    "resolve_ffprobe",
    "ruleset_ok",
    "schedule_ocr_eta_s",
    "segment_games",
    "segment_opponent_strength",
]
