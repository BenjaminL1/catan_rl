"""Native (Rust) extension landing zone.

The ``catan_engine`` extension module is built by ``maturin develop``
(or via the CI ``rust-build`` job) and installed into the active
Python interpreter's site-packages. This empty ``__init__`` exists
only so the package directory ships in the source tree — the actual
compiled artifact (``catan_engine.<abi>.so``) is git-ignored.

If ``import catan_engine`` fails, the wheel hasn't been built. Run::

    make rust-build

See ``docs/plans/rust_engine_migration.md`` for the build contract.
"""
