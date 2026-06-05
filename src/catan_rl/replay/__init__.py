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

**Import contract**: this ``__init__`` re-exports public names but
keeps the engine-touching modules (``recorder``, ``recorder_loop``,
``player_factory``) behind a PEP 562 lazy ``__getattr__`` so the
viewer can ``import catan_rl.replay.schema`` (or
``catan_rl.replay.io``) without dragging the engine + torch chain
into ``sys.modules``. The lazy gate is asserted in
:mod:`tests.unit.replay.test_viewer_import_isolation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Lightweight always-imported modules. These have NO engine / torch
# transitive imports — safe for the viewer.
from catan_rl.replay.io import load_replay, save_replay
from catan_rl.replay.migrations import (
    MigrationError,
    apply_migrations,
    register_migration,
    registered_versions,
    unregister_migration,
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

# Recorder + player factory symbols are surfaced via lazy
# ``__getattr__`` below. The names appear in ``__all__`` so static
# tools still see them, but the actual module import is deferred.
if TYPE_CHECKING:
    from catan_rl.replay.player_factory import Actor, PlayerKind, build_actor
    from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec
    from catan_rl.replay.recorder import (
        EventCollector,
        classify_step_events,
        extract_sub_actions,
        snapshot_step_state,
        split_burst_one_placement,
        split_burst_two_placements,
        synthesize_intermediate_setup_snapshot,
    )
    from catan_rl.replay.recorder_loop import record_game

# Map of lazy attribute name → (module path, attribute name in that module).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Actor": ("catan_rl.replay.player_factory", "Actor"),
    "PlayerKind": ("catan_rl.replay.player_factory", "PlayerKind"),
    "RecorderPlayerSpec": ("catan_rl.replay.player_factory", "PlayerSpec"),
    "build_actor": ("catan_rl.replay.player_factory", "build_actor"),
    "EventCollector": ("catan_rl.replay.recorder", "EventCollector"),
    "classify_step_events": ("catan_rl.replay.recorder", "classify_step_events"),
    "extract_sub_actions": ("catan_rl.replay.recorder", "extract_sub_actions"),
    "snapshot_step_state": ("catan_rl.replay.recorder", "snapshot_step_state"),
    "split_burst_one_placement": (
        "catan_rl.replay.recorder",
        "split_burst_one_placement",
    ),
    "split_burst_two_placements": (
        "catan_rl.replay.recorder",
        "split_burst_two_placements",
    ),
    "synthesize_intermediate_setup_snapshot": (
        "catan_rl.replay.recorder",
        "synthesize_intermediate_setup_snapshot",
    ),
    "record_game": ("catan_rl.replay.recorder_loop", "record_game"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 module-level lazy attribute resolver.

    Imports the recorder / player_factory submodules only when a
    consumer references one of their re-exported symbols. The viewer
    never touches these, so its import chain stays clean.
    """
    if name in _LAZY_ATTRS:
        import importlib

        mod_path, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_path)
        value = getattr(mod, attr)
        # Cache on the package so subsequent lookups bypass __getattr__.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'catan_rl.replay' has no attribute {name!r}")


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
    "record_game",
    "register_migration",
    "registered_versions",
    "save_replay",
    "snapshot_step_state",
    "split_burst_one_placement",
    "split_burst_two_placements",
    "synthesize_intermediate_setup_snapshot",
    "unregister_migration",
]
