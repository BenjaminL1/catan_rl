"""Versioned migration registry for replay JSON schemas.

Mirrors :mod:`catan_rl.checkpoint.migrations` so the on-disk
artifact lineage story is consistent across the two file formats
the project produces.

Each migration is a function ``payload: dict -> dict`` that upgrades
from schema version ``N`` to schema version ``N+1``. The registry is
ordered; :func:`apply_migrations` walks the chain until it reaches
the target.

The current schema is :data:`catan_rl.replay.schema.REPLAY_SCHEMA_VERSION`
== 2. Exactly one migration is registered at import time:
:func:`_v1_to_v2`, a no-op version bump. v2 only ADDED fields that all
carry defaults (``ReplayStep.policy_internals`` and the ``Metadata``
provenance flags ``mode`` / ``sims`` / ``clairvoyant`` / ``reveal_bot``),
so nothing inside a v1 payload needs rewriting — but the bump still
needs a registered step, or :func:`apply_migrations` would raise on
every replay written before it.

No pre-v1 replays exist in the wild (the replay system shipped *after*
this module), so there is no v0 → v1 entry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# from_version -> upgrader (payload v_from -> payload v_from+1)
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


class MigrationError(RuntimeError):
    """Raised when a replay cannot be upgraded to the target version
    (missing migration, malformed payload, downgrade attempt, etc.)."""


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
    """Remove a migration. Used by the test suite to clean up
    registrations made by individual tests; production code should
    never need this."""
    _MIGRATIONS.pop(from_version, None)


def registered_versions() -> tuple[int, ...]:
    """Return the sorted tuple of registered ``from_version`` keys."""
    return tuple(sorted(_MIGRATIONS.keys()))


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
            f"replay schema v{current} is newer than the target "
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


# ---------------------------------------------------------------------------
# Registered migrations
# ---------------------------------------------------------------------------


def _v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2: a pure version bump.

    v2 added ``ReplayStep.policy_internals`` and the ``Metadata``
    provenance fields (``mode``/``sims``/``clairvoyant``/``reveal_bot``
    and the ``git_sha``/``ckpt_sha256`` attribution pair), all of which
    default to empty/``False``/``None`` in
    :mod:`catan_rl.replay.schema` and are read with ``.get`` defaults in
    :mod:`catan_rl.replay.io`. A v1 payload is therefore already a valid
    v2 payload — the only thing to change is the stamped version."""
    return {**payload, "schema_version": 2}


register_migration(1, _v1_to_v2)
