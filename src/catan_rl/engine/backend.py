"""Engine backend selector — declarative only.

**Status (2026-06-06): this switch is dead code in the production
rollout / SGD loop.** The 2026-06-06 forensic audit confirmed that
``resolve_backend`` is read only by ``analysis/diag_phase_timing.py``
and ``tests/unit/engine/test_backend_switch.py``. No code under
``src/catan_rl/env/``, ``policy/``, ``ppo/``, ``selfplay/``, or
``eval/`` branches on it. ``CatanEnv.reset`` (`env/catan_env.py:280`)
unconditionally constructs ``catanGame()`` regardless of the
selector's value. See
``docs/plans/rust_engine_actual_state.md`` for the audit and the
remediation plan (Phase 4 wires the actual dispatch).

Selection is driven by the ``CATAN_ENGINE_BACKEND`` environment
variable:

* ``CATAN_ENGINE_BACKEND=rust`` — DECLARES that the operator wants
  the Rust engine. Currently honoured only by the diagnostic above;
  the production training loop ignores it.
* ``CATAN_ENGINE_BACKEND=py`` (default) — Python engine. This is
  the actual production engine.

Once Phase 4 of the remediation plan lands, the env-var-driven
dispatch will take effect; until then this module exists so the
diagnostic + parametric pytest fixture (Phase 2) can probe an
operator-declared intent without imposing it on `CatanEnv`.

The R12 phase (Python engine deletion) remains gated on the
end-to-end ≥ 3× speedup criterion at Phase 9; until then both
backends must keep their independent test coverage.
"""

from __future__ import annotations

import os
from typing import Literal

Backend = Literal["python", "rust"]

#: Default backend. Reverted from ``"rust"`` → ``"python"`` on
#: 2026-06-06 because no production code branches on the selector
#: and the previous default lied about which engine the training
#: loop actually used. Phase 4 of the remediation plan flips this
#: back to ``"rust"`` once `CatanEnv` actually dispatches.
DEFAULT_BACKEND: Backend = "python"


def current_backend() -> Backend:
    """Read ``CATAN_ENGINE_BACKEND`` from the environment and
    resolve it to a normalized backend tag.

    Aliases:
        - ``rust`` / ``ru`` / ``r`` → ``"rust"``
        - ``python`` / ``py`` / ``p`` → ``"python"``
        - unset or unknown → :data:`DEFAULT_BACKEND`
    """
    raw = os.environ.get("CATAN_ENGINE_BACKEND", "").strip().lower()
    if raw in ("rust", "ru", "r"):
        return "rust"
    if raw in ("python", "py", "p"):
        return "python"
    return DEFAULT_BACKEND


def is_rust_available() -> bool:
    """``True`` if the ``catan_engine`` PyO3 extension is importable."""
    try:
        import catan_engine  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(requested: Backend | None = None) -> Backend:
    """Pick a backend honoring (in order): explicit ``requested`` arg,
    env var, then ``DEFAULT_BACKEND``. Falls back to ``"python"`` if
    Rust is requested but unavailable, with a stderr warning."""
    import sys

    chosen: Backend = requested if requested is not None else current_backend()
    if chosen == "rust" and not is_rust_available():
        print(
            "warning: CATAN_ENGINE_BACKEND=rust requested but "
            "catan_engine extension not built (run `make rust-build`); "
            "falling back to Python engine.",
            file=sys.stderr,
        )
        return "python"
    return chosen
