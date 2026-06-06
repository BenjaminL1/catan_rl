"""Smoke tests for the ``engine_backend`` parametric fixture.

The fixture lives in ``tests/conftest.py`` and is the mechanism
through which future tests run the same logic against both the
Python and the Rust engines. Phase 2 of the Rust migration
remediation plan. The original migration plan promised this
fixture at ``docs/plans/rust_engine_migration.md`` but never
landed it — this module is its acceptance gate.

Pins:

1. The fixture yields the backend tag the parametrisation labels it
   with — no accidental drift between the ID and the value.
2. The fixture ``importorskip``s ``catan_engine`` when the rust
   parametrisation is selected so machines without the ``.so``
   skip cleanly rather than fail.
3. The fixture honors the ``CATAN_TEST_ENGINE_BACKEND`` env-var
   override (strong-rec from the architect review).
4. The fixture's resolution agrees with
   ``catan_rl.engine.backend.resolve_backend(<tag>)`` so callers
   that pass the fixture value through the production resolver
   get the same answer.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from catan_rl.engine.backend import resolve_backend

#: pytest collect-only line for a parametrised test of the form
#: ``...test_value_matches_param_tag[py_engine]``. Captures the
#: parametrisation id (the part between brackets).
#:
#: NOTE: this regex hardcodes the test name
#: ``test_value_matches_param_tag``. If that test is renamed, the
#: env-var override tests below silently match nothing and fail
#: loudly with ``assert set() == {"py_engine", ...}``. The failure
#: is loud (not silent) so the brittleness is acceptable; if you
#: rename the test, update the regex.
_PARAM_ID_RE = re.compile(r"test_value_matches_param_tag\[(?P<param_id>[^\]]+)\]")

#: Regex matching any parametrised test id that ends with ``[py_engine]``
#: or ``[rust_engine]``. Used by the "no silent doubling" guard
#: that asserts the only tests consuming ``engine_backend`` are
#: under ``test_backend_fixture.py``.
_ANY_ENGINE_PARAM_RE = re.compile(r"\[(?P<param_id>py_engine|rust_engine)\]")


class TestEngineBackendFixture:
    def test_value_matches_param_tag(self, engine_backend: str) -> None:
        """The fixture's value is one of the two backend tags."""
        assert engine_backend in ("python", "rust")

    def test_value_round_trips_through_production_resolver(self, engine_backend: str) -> None:
        """A test that passes the fixture value into
        ``resolve_backend`` must get back the same tag (modulo the
        rust→python fallback that fires when the extension isn't
        importable, which the fixture itself guards against via
        ``importorskip`` — so by the time we land here the answer
        is deterministic)."""
        assert resolve_backend(engine_backend) == engine_backend  # type: ignore[arg-type]


class TestEnvVarOverride:
    def _run_pytest_subprocess(
        self,
        env_value: str | None,
        repo_root: Path,
    ) -> list[str]:
        """Run pytest as a subprocess collecting only the ids of the
        backend-fixture tests under this module. Returns the list
        of test ids reported by ``-q --collect-only``."""
        env = os.environ.copy()
        env["CATAN_TEST_ENGINE_BACKEND"] = env_value if env_value is not None else ""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/engine/test_backend_fixture.py::TestEngineBackendFixture::test_value_matches_param_tag",
                "--collect-only",
                "-q",
                "--no-header",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                f"subprocess pytest collect failed (rc={result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        suffixes: set[str] = set()
        for line in result.stdout.splitlines():
            m = _PARAM_ID_RE.search(line)
            if m:
                suffixes.add(m.group("param_id"))
        return list(suffixes)

    def test_env_unset_collects_both_backends(self, repo_root: Path) -> None:
        """With the env var unset, the fixture must produce both
        ``py_engine`` and ``rust_engine`` parametrisations."""
        suffixes = set(self._run_pytest_subprocess(env_value=None, repo_root=repo_root))
        assert suffixes == {"py_engine", "rust_engine"}

    def test_env_python_restricts_to_python_backend(self, repo_root: Path) -> None:
        """Setting the env var to ``python`` collects only the
        ``py_engine`` parametrisation."""
        suffixes = set(self._run_pytest_subprocess(env_value="python", repo_root=repo_root))
        assert suffixes == {"py_engine"}

    def test_env_rust_restricts_to_rust_backend(self, repo_root: Path) -> None:
        """Setting the env var to ``rust`` collects only the
        ``rust_engine`` parametrisation."""
        suffixes = set(self._run_pytest_subprocess(env_value="rust", repo_root=repo_root))
        assert suffixes == {"rust_engine"}

    def test_env_invalid_value_falls_back_to_both(self, repo_root: Path) -> None:
        """An unrecognised env-var value falls back to the full
        sweep — same behaviour as unset. Avoids a typo silently
        skipping half the suite."""
        suffixes = set(self._run_pytest_subprocess(env_value="xyz", repo_root=repo_root))
        assert suffixes == {"py_engine", "rust_engine"}


class TestImportOrSkipPath:
    """Architect MUST-FIX #1: prove the ``importorskip("catan_engine")``
    branch actually fires when the extension is unavailable.

    Without this, every dev box that has the ``.so`` built leaves
    the rust-skip path as dead code. The test simulates a missing
    extension by running pytest in a subprocess with a fake
    ``catan_engine`` module that raises ``ImportError`` on
    ``importorskip``.
    """

    def test_rust_parametrisation_skips_when_extension_unavailable(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """Spawn a subprocess pytest whose ``sys.path`` is
        engineered to make ``import catan_engine`` raise. Verify
        the ``rust_engine`` parametrisation is reported as
        ``SKIPPED`` rather than collected or errored.

        Implementation: write a sitecustomize.py that shadows
        ``catan_engine`` with a dummy module whose ``__spec__`` is
        None (causing ``importorskip`` to treat it as unavailable
        and skip). The subprocess inherits the parent's
        environment so the production install is reachable —
        sitecustomize injects the shadow first.
        """
        # Subprocess workspace with sitecustomize that nukes catan_engine.
        site_dir = tmp_path / "site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(
            "import sys\n"
            "# Inject a sentinel that ``importorskip`` will detect as\n"
            "# unavailable. ``pytest.importorskip`` does an actual\n"
            "# ``importlib.import_module`` followed by a version check\n"
            "# (no version arg → just the import). Setting the\n"
            "# entry in ``sys.modules`` to ``None`` makes the import\n"
            "# raise ``ImportError`` per PEP 328 / CPython behaviour.\n"
            "sys.modules['catan_engine'] = None\n"
        )
        env = os.environ.copy()
        # Prepend the tmp site dir so sitecustomize loads first.
        env["PYTHONPATH"] = str(site_dir) + os.pathsep + env.get("PYTHONPATH", "")
        # Make sure the env-var override doesn't force python-only.
        env.pop("CATAN_TEST_ENGINE_BACKEND", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/engine/test_backend_fixture.py::TestEngineBackendFixture::test_value_matches_param_tag",
                "-v",
                "--no-header",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        stdout = result.stdout
        # py_engine parametrisation passes.
        assert "[py_engine] PASSED" in stdout, (
            f"py_engine should pass when catan_engine is unavailable; "
            f"stdout:\n{stdout}\nstderr:\n{result.stderr}"
        )
        # rust_engine parametrisation skips (NOT errors, NOT passes).
        assert "[rust_engine] SKIPPED" in stdout, (
            f"rust_engine must be SKIPPED via importorskip when "
            f"catan_engine is unavailable; stdout:\n{stdout}\n"
            f"stderr:\n{result.stderr}"
        )


class TestNoSilentSuiteDoubling:
    """Architect MUST-FIX #2: assert the acceptance gate at the
    whole-tree level, not just the new module.

    Without this guard, a future test file that consumes
    ``engine_backend`` (intentionally or by name collision) would
    silently double its parametrisation across the suite. The
    fixture's acceptance gate explicitly forbids that.
    """

    def test_only_backend_fixture_module_consumes_engine_backend(self, repo_root: Path) -> None:
        """Collect every test under ``tests/`` and assert every
        ``[py_engine]`` / ``[rust_engine]`` parametrisation comes
        from ``test_backend_fixture.py``. When Phase 3+ files start
        consuming the fixture, this test should be updated to
        widen its allow-list explicitly — never blanket-removed."""
        env = os.environ.copy()
        env.pop("CATAN_TEST_ENGINE_BACKEND", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--collect-only",
                "-q",
                "--no-header",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                f"whole-tree collect failed (rc={result.returncode}):\n"
                f"stdout (tail):\n...{result.stdout[-2000:]}\n"
                f"stderr (tail):\n...{result.stderr[-2000:]}"
            )
        consumers: set[str] = set()
        for line in result.stdout.splitlines():
            if not _ANY_ENGINE_PARAM_RE.search(line):
                continue
            stripped = line.strip()
            # Two pytest collect formats observed:
            #   1) Plain id:  ``tests/unit/.../foo.py::TestX::test_y[py_engine]``
            #   2) Wrapped:   ``<Function test_y[py_engine]>`` (no file path)
            # We only care about (1), so skip lines that don't
            # start with ``tests/``. Wrapped lines are pytest's
            # node-repr noise from older versions and would yield
            # a non-file token like ``<Function`` if parsed.
            if not stripped.startswith("tests/"):
                continue
            node_id = stripped.split()[0]
            test_file = node_id.split("::", 1)[0]
            consumers.add(test_file)
        # Phase 2 allow-list: only the new fixture module.
        allowed = {"tests/unit/engine/test_backend_fixture.py"}
        unexpected = consumers - allowed
        assert not unexpected, (
            f"unexpected consumers of engine_backend (Phase 2 allow-list "
            f"is {allowed}): {unexpected}. Add the new file to the "
            f"allow-list intentionally if this is a Phase 3+ wiring step."
        )
