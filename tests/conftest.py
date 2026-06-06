"""Shared pytest fixtures for catan_rl tests."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "train"
FROZEN_CHAMPION = CHECKPOINT_DIR / "checkpoint_07390040.pt"

#: Backend tags exposed to parametric tests. Must stay in sync
#: with :data:`catan_rl.engine.backend.Backend`. Keeping the tuple
#: here (not importing) avoids a circular dependency when a test
#: file imports ``conftest`` before the backend module is ready.
_ENGINE_BACKEND_PARAMS: tuple[str, ...] = ("python", "rust")

#: Display ids used by pytest's parametrisation. Matches the
#: original migration plan's ``ids=["py_engine", "rust_engine"]``
#: convention (the long-form ``python_engine`` was rejected by the
#: architect review for being chatty in test ids).
_ENGINE_BACKEND_IDS: dict[str, str] = {"python": "py_engine", "rust": "rust_engine"}

#: Env-var name that overrides the parametric backend sweep. Set
#: it to ``python`` or ``rust`` to restrict which backend any
#: ``engine_backend``-consuming test runs against (CI flips it per
#: matrix job; humans use it for "just run the rust path"). Unset
#: → both backends are exercised.
_BACKEND_OVERRIDE_ENV = "CATAN_TEST_ENGINE_BACKEND"


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Reset all RNGs before each test for reproducibility."""
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except ImportError:
        pass


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def frozen_champion_path() -> Path:
    """Path to the frozen champion checkpoint, or skip if missing."""
    if not FROZEN_CHAMPION.exists():
        pytest.skip(f"Frozen champion not found at {FROZEN_CHAMPION}")
    return FROZEN_CHAMPION


def _engine_backend_params_for_run() -> tuple[str, ...]:
    """Compute the parametric set for ``engine_backend`` honoring
    the :data:`_BACKEND_OVERRIDE_ENV` env-var override.

    CI matrix jobs set the env var to one of the literal backend
    names so the suite runs against only that backend per job;
    local runs leave it unset and exercise both.
    """
    override = os.environ.get(_BACKEND_OVERRIDE_ENV, "").strip().lower()
    if override in _ENGINE_BACKEND_PARAMS:
        return (override,)
    return _ENGINE_BACKEND_PARAMS


@pytest.fixture(
    params=_engine_backend_params_for_run(),
    ids=lambda b: _ENGINE_BACKEND_IDS[b],
)
def engine_backend(request: pytest.FixtureRequest) -> str:
    """Yield ``"python"`` or ``"rust"`` so the test runs against
    both engines unless restricted via :data:`_BACKEND_OVERRIDE_ENV`.

    Phase 2 of the Rust migration remediation plan. Tests that
    consume this fixture should branch on the value to construct
    the appropriate backend (or pass it through to a helper that
    does). When the rust path is selected, this fixture
    ``importorskip``s the ``catan_engine`` extension so CI hosts
    without the ``.so`` cleanly skip the rust parametrisation
    instead of erroring.

    Acceptance gate (per the remediation plan):
        * ``pytest tests/unit/engine/test_backend_fixture.py -v``
          collects both ``py_engine`` and ``rust_engine`` parametrisations.
        * The overall suite count grows by exactly the number of
          parametrised tests, not silently doubled.
    """
    if request.param == "rust":
        pytest.importorskip("catan_engine")
    return request.param
