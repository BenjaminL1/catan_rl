"""Tests for `catan_rl.replay.migrations`."""

from __future__ import annotations

import pytest

from catan_rl.replay.migrations import (
    MigrationError,
    apply_migrations,
    register_migration,
    registered_versions,
    unregister_migration,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot + restore the registry around each test so registrations
    don't leak across tests."""
    before = registered_versions()
    yield
    after = registered_versions()
    for v in after:
        if v not in before:
            unregister_migration(v)


class TestNoOp:
    def test_already_at_target_returns_unchanged(self) -> None:
        payload = {"schema_version": 1, "metadata": {"seed": 7}}
        out = apply_migrations(payload, target_version=1)
        assert out["schema_version"] == 1
        assert out["metadata"] == {"seed": 7}

    def test_missing_schema_version_raises(self) -> None:
        with pytest.raises(MigrationError, match="missing 'schema_version'"):
            apply_migrations({"foo": "bar"}, target_version=1)


class TestRegistry:
    def test_register_and_apply(self) -> None:
        def v0_to_v1(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1, "extra_field": "added_by_v0_to_v1"}

        register_migration(0, v0_to_v1)
        out = apply_migrations({"schema_version": 0}, target_version=1)
        assert out["schema_version"] == 1
        assert out["extra_field"] == "added_by_v0_to_v1"

    def test_chain_two_migrations(self) -> None:
        def v0_to_v1(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1, "a": "first"}

        def v1_to_v2(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 2, "b": "second"}

        register_migration(0, v0_to_v1)
        register_migration(1, v1_to_v2)
        out = apply_migrations({"schema_version": 0}, target_version=2)
        assert out["schema_version"] == 2
        assert out["a"] == "first"
        assert out["b"] == "second"

    def test_missing_step_raises(self) -> None:
        def v0_to_v1(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1}

        register_migration(0, v0_to_v1)
        with pytest.raises(MigrationError, match="no migration registered from v1"):
            apply_migrations({"schema_version": 0}, target_version=3)

    def test_duplicate_registration_raises(self) -> None:
        def v0_to_v1_a(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1}

        def v0_to_v1_b(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 1, "different": True}

        register_migration(0, v0_to_v1_a)
        with pytest.raises(MigrationError, match="already registered"):
            register_migration(0, v0_to_v1_b)

    def test_migration_must_advance_one_step(self) -> None:
        # A buggy migration that bumps to v3 instead of v1 must be
        # caught — silent skipping of intermediate versions would
        # corrupt the migration contract.
        def bad(p):  # type: ignore[no-untyped-def]
            return {**p, "schema_version": 3}

        register_migration(0, bad)
        with pytest.raises(MigrationError, match="expected v1"):
            apply_migrations({"schema_version": 0}, target_version=2)

    def test_downgrade_attempt_raises(self) -> None:
        with pytest.raises(MigrationError, match="newer than the target"):
            apply_migrations({"schema_version": 5}, target_version=1)
