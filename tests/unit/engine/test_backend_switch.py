"""Tests for the CATAN_ENGINE_BACKEND switch added in R10.

The switch itself is dead code in the production rollout loop as of
2026-06-06 (see ``docs/plans/rust_engine_actual_state.md``); these
tests verify the selector's *resolution* logic (env-var aliases,
explicit-arg override, fallback) without asserting anything about
which engine the training loop actually instantiates. Phase 4 of
the remediation plan ships the actual dispatch.
"""

from __future__ import annotations

import pytest

from catan_rl.engine.backend import (
    DEFAULT_BACKEND,
    current_backend,
    is_rust_available,
    resolve_backend,
)


class TestCurrentBackend:
    def test_defaults_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Phase 0 forensic audit: default reverted from "rust" to
        # "python" because no code under ``src/catan_rl/`` actually
        # dispatches on this value, and a default of "rust" was
        # misrepresenting which engine the training loop used.
        monkeypatch.delenv("CATAN_ENGINE_BACKEND", raising=False)
        assert current_backend() == DEFAULT_BACKEND == "python"

    @pytest.mark.parametrize("val", ["rust", "RUST", "r", "ru"])
    def test_rust_aliases(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", val)
        assert current_backend() == "rust"

    @pytest.mark.parametrize("val", ["python", "PY", "p", "py"])
    def test_python_aliases(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", val)
        assert current_backend() == "python"

    def test_unknown_value_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", "xyz")
        assert current_backend() == DEFAULT_BACKEND


class TestIsRustAvailable:
    def test_returns_true_when_extension_built(self) -> None:
        # The test suite assumes ``make rust-build`` has been run.
        assert is_rust_available()


class TestResolveBackend:
    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", "python")
        assert resolve_backend("rust") == "rust"
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", "rust")
        assert resolve_backend("python") == "python"

    def test_no_arg_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CATAN_ENGINE_BACKEND", "python")
        assert resolve_backend() == "python"
