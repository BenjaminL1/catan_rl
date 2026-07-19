"""Tests for `checkpoint/migrations.py`.

Pins:
1. Empty registry → only no-op upgrades succeed.
2. Registered upgraders form a chain.
3. Missing step in the chain raises MigrationError.
4. Re-registering the same from_version raises (no silent shadow).
5. Downgrade attempts raise.
6. Payload missing schema_version raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from catan_rl.checkpoint.migrations import (
    MigrationError,
    apply_migrations,
    register_migration,
    registered_versions,
    unregister_migration,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    # Snapshot + restore the global registry around each test so the
    # tests can register / unregister freely without leaking state.
    before = registered_versions()
    yield
    after = registered_versions()
    for v in after:
        if v not in before:
            unregister_migration(v)


class TestNoOp:
    def test_already_at_target_returns_unchanged(self) -> None:
        payload = {"schema_version": 1, "config": {"a": 1}}
        out = apply_migrations(payload, target_version=1)
        assert out["schema_version"] == 1
        assert out["config"] == {"a": 1}

    def test_missing_schema_version_raises(self) -> None:
        with pytest.raises(MigrationError, match="missing 'schema_version'"):
            apply_migrations({"foo": "bar"}, target_version=1)


class TestRegistry:
    # These tests exercise the registry mechanics with FAKE migrations at
    # versions >= 10 so they never collide with the real registered
    # v1 -> v2 (league-sidecar) migration in migrations.py.
    def test_register_and_apply(self) -> None:
        def v10_to_v11(p: dict[str, Any]) -> dict[str, Any]:
            out = dict(p)
            out["schema_version"] = 11
            out["new_field"] = "added"
            return out

        register_migration(10, v10_to_v11)
        out = apply_migrations({"schema_version": 10}, target_version=11)
        assert out["schema_version"] == 11
        assert out["new_field"] == "added"

    def test_chain_two_migrations(self) -> None:
        def v10_to_v11(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 11, "a": "v11"}

        def v11_to_v12(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 12, "b": "v12"}

        register_migration(10, v10_to_v11)
        register_migration(11, v11_to_v12)
        out = apply_migrations({"schema_version": 10}, target_version=12)
        assert out["schema_version"] == 12
        assert out["a"] == "v11"
        assert out["b"] == "v12"

    def test_missing_step_raises(self) -> None:
        # v10 -> v11 registered; jump to v13 requested → fail.
        def v10_to_v11(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 11}

        register_migration(10, v10_to_v11)
        with pytest.raises(MigrationError, match="no migration registered from v11"):
            apply_migrations({"schema_version": 10}, target_version=13)

    def test_duplicate_registration_raises(self) -> None:
        def v0_to_v1_a(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 1, "src": "a"}

        def v0_to_v1_b(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 1, "src": "b"}

        register_migration(0, v0_to_v1_a)
        with pytest.raises(MigrationError, match="already registered"):
            register_migration(0, v0_to_v1_b)

    def test_migration_must_advance_one_step(self) -> None:
        # A buggy migration that bumps to v13 instead of v11 is caught.
        def bad(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": 13}

        register_migration(10, bad)
        with pytest.raises(MigrationError, match="expected v11"):
            apply_migrations({"schema_version": 10}, target_version=12)

    def test_downgrade_attempt_raises(self) -> None:
        with pytest.raises(MigrationError, match="newer than the target"):
            apply_migrations({"schema_version": 5}, target_version=1)
