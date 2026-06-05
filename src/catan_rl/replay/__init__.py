"""1v1 Catan game replay system.

Two scripts share this package as their only contract:

* ``scripts/record_game.py`` simulates a single game between two
  configurable bots and writes the JSON replay via
  :func:`save_replay`. Only the recorder imports the heavyweight
  stuff (engine, env, agents, optionally torch).
* ``scripts/replay_viewer.py`` loads the JSON via :func:`load_replay`
  and renders an interactive pygame UI. The viewer imports ONLY
  pygame + this package — no torch, no env, no engine.

The schema is versioned (:data:`REPLAY_SCHEMA_VERSION`) and migrated
on read through :mod:`catan_rl.replay.migrations`, mirroring the
checkpoint module's lineage discipline.
"""

from catan_rl.replay.io import load_replay, save_replay
from catan_rl.replay.migrations import (
    MigrationError,
    apply_migrations,
    register_migration,
    registered_versions,
    unregister_migration,
)
from catan_rl.replay.player_factory import (
    Actor,
    PlayerKind,
    build_actor,
)
from catan_rl.replay.player_factory import (
    PlayerSpec as RecorderPlayerSpec,
)
from catan_rl.replay.recorder import (
    EventCollector,
    classify_step_events,
    extract_sub_actions,
    snapshot_step_state,
    split_burst_one_placement,
    split_burst_two_placements,
    synthesize_intermediate_setup_snapshot,
)
from catan_rl.replay.schema import (
    EVENT_REGISTRY,
    REPLAY_SCHEMA_VERSION,
    STATE_DEV_CARD_ORDER,
    STATE_RESOURCE_ORDER,
    BoardStatic,
    EdgeStatic,
    GameEnd,
    HexStatic,
    LargestArmyChange,
    LongestRoadChange,
    Metadata,
    Monopoly,
    PlayerSpec,
    PlayerStateSnapshot,
    PortStatic,
    Replay,
    ReplaySchemaError,
    ReplayStep,
    Robber,
    Steal,
    StepEvent,
    StepStateSnapshot,
    SubAction,
    UnknownEvent,
    VertexStatic,
    event_from_dict,
    event_to_dict,
)

__all__ = [
    "EVENT_REGISTRY",
    "REPLAY_SCHEMA_VERSION",
    "STATE_DEV_CARD_ORDER",
    "STATE_RESOURCE_ORDER",
    "Actor",
    "BoardStatic",
    "EdgeStatic",
    "EventCollector",
    "GameEnd",
    "HexStatic",
    "LargestArmyChange",
    "LongestRoadChange",
    "Metadata",
    "MigrationError",
    "Monopoly",
    "PlayerKind",
    "PlayerSpec",
    "PlayerStateSnapshot",
    "PortStatic",
    "RecorderPlayerSpec",
    "Replay",
    "ReplaySchemaError",
    "ReplayStep",
    "Robber",
    "Steal",
    "StepEvent",
    "StepStateSnapshot",
    "SubAction",
    "UnknownEvent",
    "VertexStatic",
    "apply_migrations",
    "build_actor",
    "classify_step_events",
    "event_from_dict",
    "event_to_dict",
    "extract_sub_actions",
    "load_replay",
    "register_migration",
    "registered_versions",
    "save_replay",
    "snapshot_step_state",
    "split_burst_one_placement",
    "split_burst_two_placements",
    "synthesize_intermediate_setup_snapshot",
    "unregister_migration",
]
