"""Engine backend selector — chooses Python (legacy) vs Rust
(``catan_engine.RustCatanEnv``) at runtime.

Selection is driven by the ``CATAN_ENGINE_BACKEND`` environment
variable per the Q7 decision in
``docs/plans/rust_engine_migration.md``:

* ``CATAN_ENGINE_BACKEND=rust`` (the post-R10 default) — use the
  Rust engine + obs encoder + mask builder via PyO3.
* ``CATAN_ENGINE_BACKEND=py`` — use the legacy Python engine
  (``catanGame`` + ``catanBoard`` + ``BroadcastHandTracker``).
  Kept for A/B validation during the soak period.

The R12 phase removes the switch and deletes ``backend=py`` support
after the 1-2 week soak window is clean.
"""

from __future__ import annotations

import os
from typing import Literal

Backend = Literal["python", "rust"]

#: Default backend post-R10 cutover. Per Q7 the Rust path is the new
#: production default; the Python path is kept available behind the
#: env var until R12.
DEFAULT_BACKEND: Backend = "rust"


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
