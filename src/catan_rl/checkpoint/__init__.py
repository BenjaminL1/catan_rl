"""Checkpoint persistence for the v2 PPO training loop.

Phase 8 of the v2 training-infra build-out. Three pieces:

* :mod:`catan_rl.checkpoint.manager` — atomic save/load, pruning,
  cross-process determinism (RNG state for numpy / stdlib random /
  torch is captured + restored). Versioned schema so future migrations
  have a clean route.
* :mod:`catan_rl.checkpoint.league_store` — content-addressed sidecar
  store for the league snapshot pool (schema v2). Dedups the ~99/100
  byte-identical snapshots across consecutive saves.
* :mod:`catan_rl.checkpoint.migrations` — registry of v(n)→v(n+1)
  upgraders. v2 is the current schema; the registered ``v1 -> v2``
  migration (league-sidecar cutover) is shape-preserving on load.
* ``scripts/migrate_checkpoint.py`` — CLI that runs all applicable
  migrations against an old checkpoint and writes the upgraded form.
* ``scripts/reclaim_disk.py`` — report + reclaim disk from a checkpoint
  tree (dedup byte-identical files, slim fat anchors); dry-run default.

Scope:

* Resume-from-checkpoint is the primary use case — the resume must
  produce bit-identical rollouts to a never-interrupted run (modulo
  CPU/GPU non-determinism). RNG capture covers both PPO update order
  and rollout sampling.
* League snapshots are persisted to the content-addressed sidecar store
  (NOT embedded), with the checkpoint holding a small ``ref`` per
  snapshot. The snapshot pool's monotonic ``_next_snapshot_id`` is
  preserved so consumers holding ids across a resume see the same
  handles. Fat (schema v1, embedded) checkpoints still load unchanged.
* :func:`save_policy_only` / :meth:`CheckpointManager.save_slim` write a
  ~5.6 MB policy-only checkpoint — the free-disk-guard fallback and the
  basis of :func:`bank_anchor` (slim ``runs/anchors/*``).
"""

from catan_rl.checkpoint.league_store import LeagueStore, LeagueStoreError
from catan_rl.checkpoint.manager import (
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
    CheckpointPayload,
    bank_anchor,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
    save_policy_only,
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
    "LeagueStore",
    "LeagueStoreError",
    "MigrationError",
    "apply_migrations",
    "bank_anchor",
    "list_checkpoints",
    "load_checkpoint",
    "register_migration",
    "save_checkpoint",
    "save_policy_only",
]
