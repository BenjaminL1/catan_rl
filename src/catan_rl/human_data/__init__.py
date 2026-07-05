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

from catan_rl.human_data.batch import (
    HARVEST_STRENGTHS,
    BatchResult,
    HarvestPlan,
    LedgerEntry,
    VideoParseError,
    harvest_plan,
    load_ledger,
    run_batch,
)
from catan_rl.human_data.board_cv import (
    BoardRead,
    ContentAnchor,
    EngineTemplate,
    classify_resources,
    derive_screen_anchors,
    hue_cluster_margin,
    load_engine_template,
    read_board,
    read_board_stable,
)
from catan_rl.human_data.ffmpeg import (
    FFmpegNotFoundError,
    resolve_ffmpeg,
    resolve_ffprobe,
)
from catan_rl.human_data.glyph_anchor import (
    FALLBACK_PALETTE,
    GRANTABLE_RESOURCES,
    HUE_RESOURCES_BY_RANK,
    MIN_GLYPH_HUE_MARGIN,
    MIN_GRANT_CONSENSUS_FRAMES,
    MIN_VALIDATION_ACCURACY,
    MIN_VALIDATION_FRAMES,
    RESOURCE_CARD_HUES,
    GlyphPalette,
    GlyphValidation,
    LabeledGrantFrame,
    calibrate_glyph_palette,
    classify_glyph,
    classify_granted_glyphs,
    consensus_granted_glyphs,
    glyph_classifier_fingerprint,
    glyph_classifier_is_validated,
    validate_glyph_classifier,
)
from catan_rl.human_data.ingest import (
    BOARD_OCR_CALLS_PER_ACCEPTED_FRAME,
    BOARD_OCR_SECONDS_PER_DIGIT,
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
from catan_rl.human_data.openings import (
    PALETTE,
    ColorProfile,
    OpeningResult,
    detect_openings,
    detect_openings_result,
    read_hud_seat_colors,
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
from catan_rl.human_data.validate import (
    CrossCheckResult,
    cross_check,
    road_incidence_offenders,
)

__all__ = [
    "BOARD_OCR_CALLS_PER_ACCEPTED_FRAME",
    "BOARD_OCR_SECONDS_PER_DIGIT",
    "DEFAULT_DENSE_INTERVAL_S",
    "DEFAULT_SPARSE_INTERVAL_S",
    "FALLBACK_PALETTE",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "GRANTABLE_RESOURCES",
    "HARVEST_STRENGTHS",
    "HUE_RESOURCES_BY_RANK",
    "LOG_CROP_FRAC",
    "MAX_AFFINE_RESIDUAL_PX",
    "MIN_GLYPH_HUE_MARGIN",
    "MIN_GRANT_CONSENSUS_FRAMES",
    "MIN_RESOLUTION",
    "MIN_VALIDATION_ACCURACY",
    "MIN_VALIDATION_FRAMES",
    "OCR_SECONDS_PER_CROP",
    "PALETTE",
    "RESOURCE_CARD_HUES",
    "RESOURCE_LITERALS",
    "SCHEMA_VERSION",
    "BatchResult",
    "BoardRead",
    "ColorProfile",
    "ContentAnchor",
    "CrossCheckResult",
    "DecodedFrame",
    "EngineTemplate",
    "FFmpegNotFoundError",
    "GameRecord",
    "GameSegment",
    "GlyphClassifierNotValidated",
    "GlyphPalette",
    "GlyphValidation",
    "HarvestPlan",
    "LabeledGrantFrame",
    "LedgerEntry",
    "LogEvent",
    "OpeningResult",
    "OpponentStrength",
    "ParsedLog",
    "PlayerOpening",
    "ScheduledFrame",
    "SubResolutionError",
    "TimeWindow",
    "Topology",
    "VideoDownloadError",
    "VideoParseError",
    "assert_glyph_anchor",
    "assert_scale_up_orientation_gates",
    "build_sampling_schedule",
    "calibrate_glyph_palette",
    "check_road_incidence",
    "classify_glyph",
    "classify_granted_glyphs",
    "classify_resources",
    "consensus_granted_glyphs",
    "crop_log",
    "cross_check",
    "decode_frames_at",
    "derive_opponent_strength",
    "derive_screen_anchors",
    "detect_openings",
    "detect_openings_result",
    "download_video",
    "estimate_ocr_wall_clock_s",
    "glyph_classifier_fingerprint",
    "glyph_classifier_is_validated",
    "granted_multiset_matches_a_settlement",
    "granted_resources_under_orientation",
    "harvest_plan",
    "hue_cluster_margin",
    "ingest_video",
    "load_engine_template",
    "load_ledger",
    "load_strength_manifest",
    "load_topology",
    "manifest_entry",
    "ocr_log_crop",
    "parse_log",
    "probe_resolution",
    "read_board",
    "read_board_stable",
    "read_hud_seat_colors",
    "resolve_ffmpeg",
    "resolve_ffprobe",
    "road_incidence_offenders",
    "ruleset_ok",
    "run_batch",
    "schedule_ocr_eta_s",
    "segment_games",
    "segment_opponent_strength",
    "validate_glyph_classifier",
]
