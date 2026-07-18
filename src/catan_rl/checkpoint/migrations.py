"""Versioned migration registry for checkpoint schemas.

Each migration is a function ``payload: dict -> dict`` that upgrades
from schema version ``N`` to schema version ``N+1``. The registry is
ordered; ``apply_migrations(payload, target_version)`` walks the chain
until it reaches the target.

Migrations are stored in a flat dict (no decorators, no class
hierarchy) so they're trivially serialisable + testable. Registering a
migration that overwrites an existing key raises immediately — schema
bumps should never silently shadow each other.

The current schema is ``SCHEMA_VERSION = 2`` (see
:mod:`catan_rl.checkpoint.manager`). The one registered migration is
``v1 -> v2`` (:func:`_migrate_v1_to_v2`): the league-pool sidecar cutover.
It is intentionally *shape-preserving* — a v1 "fat" checkpoint keeps its
inline ``league.snapshots[i].state_dict`` entries and only its
``schema_version`` is bumped; :func:`~catan_rl.checkpoint.manager.load_checkpoint`
then loads those inline snapshots directly (never touching the sidecar store),
which is exactly the backward-compat path that keeps ``v11_cand_u724.pt`` and
all ``runs/anchors/*`` loadable. The migration to the *slim* on-disk layout
(refs instead of inline state-dicts) happens one-way on the next SAVE, not on
load.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# from_version -> upgrader (payload v_from -> payload v_from+1)
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


class MigrationError(RuntimeError):
    """Raised when a checkpoint cannot be upgraded to the target version
    (missing migration, malformed payload, etc.)."""


def register_migration(
    from_version: int,
    upgrader: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Add a v(from_version) → v(from_version+1) upgrader to the registry.

    Raises :class:`MigrationError` if a migration is already registered
    for the same ``from_version`` — overwriting silently would let two
    PRs land conflicting schema bumps without a merge conflict.
    """
    if from_version in _MIGRATIONS:
        raise MigrationError(
            f"migration from v{from_version} is already registered; refusing to overwrite"
        )
    _MIGRATIONS[from_version] = upgrader


def unregister_migration(from_version: int) -> None:
    """Remove a migration. Used only by the test suite to clean up
    registrations made by individual tests; production code should
    never need this."""
    _MIGRATIONS.pop(from_version, None)


def registered_versions() -> tuple[int, ...]:
    """Return the sorted tuple of registered ``from_version`` keys."""
    return tuple(sorted(_MIGRATIONS.keys()))


def _migrate_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2: league-pool sidecar cutover (shape-preserving on load).

    A v1 checkpoint embeds each league snapshot's ``state_dict`` inline. v2
    checkpoints written by :func:`~catan_rl.checkpoint.manager.save_checkpoint`
    instead carry a content-addressed ``ref`` per snapshot. On LOAD we do NOT
    rewrite the inline snapshots into refs — the loader handles inline
    state-dicts directly — so this migration only bumps the version tag. The
    physical slim layout is adopted the next time the checkpoint is saved."""
    out = dict(payload)
    out["schema_version"] = 2
    return out


register_migration(1, _migrate_v1_to_v2)


def apply_migrations(payload: dict[str, Any], *, target_version: int) -> dict[str, Any]:
    """Walk the registered migrations from ``payload["schema_version"]``
    up to ``target_version`` and return the upgraded payload.

    Returns the payload unchanged if it is already at ``target_version``.
    Raises :class:`MigrationError` if the payload is at a higher version
    than the target (we can't downgrade) or if a step in the chain is
    missing.
    """
    if "schema_version" not in payload:
        raise MigrationError("payload missing 'schema_version' key — refusing to guess")
    current = int(payload["schema_version"])
    if current > target_version:
        raise MigrationError(
            f"checkpoint schema v{current} is newer than the target "
            f"v{target_version}; downgrading is not supported"
        )
    out = dict(payload)
    while current < target_version:
        upgrader = _MIGRATIONS.get(current)
        if upgrader is None:
            raise MigrationError(
                f"no migration registered from v{current} to "
                f"v{current + 1} (target v{target_version})"
            )
        out = upgrader(out)
        next_version = int(out.get("schema_version", -1))
        if next_version != current + 1:
            raise MigrationError(
                f"migration from v{current} produced schema_version="
                f"{next_version}; expected v{current + 1}"
            )
        current = next_version
    return out
