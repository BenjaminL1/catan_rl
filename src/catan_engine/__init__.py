"""``catan_engine`` — native Rust extension for the 1v1 Catan game
engine. The compiled ``.so`` is built by maturin from
``crates/catan_engine/`` and installed adjacent to this file. The
star-import below re-exports every symbol the Rust ``#[pymodule]``
exposes so callers can ``import catan_engine`` and use
``catan_engine.StackedDice``, ``catan_engine.BoardStatic``, etc.
without an explicit submodule path.

This Python file is the package marker maturin requires when
``python-source = "src"`` is set in ``pyproject.toml``. The
companion ``catan_engine.<abi>.so`` shipped in this directory
provides every runtime symbol.
"""

from catan_engine.catan_engine import *  # noqa: F403
