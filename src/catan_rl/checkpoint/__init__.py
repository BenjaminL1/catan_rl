"""Checkpoint persistence for the v2 PPO training loop.

Phase 8 of the v2 training-infra build-out. Three pieces:

* :mod:`catan_rl.checkpoint.manager` — atomic save/load, pruning,
  cross-process determinism (RNG state for numpy / stdlib random /
  torch is captured + restored). Versioned schema so future migrations
  have a clean route.
* :mod:`catan_rl.checkpoint.migrations` — registry of v(n)→v(n+1)
  upgraders. v1 is the current schema; older formats raise on load
  unless a migration exists.
* ``scripts/migrate_checkpoint.py`` — CLI that runs all applicable
  migrations against an old checkpoint and writes the upgraded form.

Scope:

* Resume-from-checkpoint is the primary use case — the resume must
  produce bit-identical rollouts to a never-interrupted run (modulo
  CPU/GPU non-determinism). RNG capture covers both PPO update order
  and rollout sampling.
* League snapshots ship in the same checkpoint. The snapshot pool's
  monotonic ``_next_snapshot_id`` is preserved so consumers holding
  ids across a resume see the same handles.
* Pre-Phase-8 checkpoints don't exist — Phase 1 shipped before this
  module, so the migration registry starts empty. The plumbing is in
  place for the first real migration when one is needed.
"""

from catan_rl.checkpoint.manager import (
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
    CheckpointPayload,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from catan_rl.checkpoint.migrations import (
    MigrationError,
    apply_migrations,
    register_migration,
)

__all__ = [
    "SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointPayload",
    "MigrationError",
    "apply_migrations",
    "list_checkpoints",
    "load_checkpoint",
    "register_migration",
    "save_checkpoint",
]
