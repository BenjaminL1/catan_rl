"""User-facing CLI entry-point bodies.

Each module here exposes ``main(argv: list[str] | None = None) -> int``
that ``pyproject.toml [project.scripts]`` wires onto PATH:

* ``catan-rl-train`` → :func:`catan_rl.cli.train.main`
* ``catan-rl-record`` → :func:`catan_rl.cli.record_game.main`
* ``catan-rl-replay`` → :func:`catan_rl.replay.viewer.event_loop.main`
  (already lived in the viewer package; no shim here)
* ``catan-rl-migrate-ckpt`` → :func:`catan_rl.cli.migrate_checkpoint.main`
* ``catan-rl-bc-train`` → :func:`catan_rl.cli.train_bc.main`
* ``catan-rl-bc-generate`` → :func:`catan_rl.cli.generate_bc_dataset.main`
* ``catan-rl-label-setup`` → :func:`catan_rl.cli.label_setup.main`

Migrated from ``scripts/<name>.py`` in the maturin sole-backend
cutover. The ``sys.path.insert(0, REPO_ROOT/'src')`` shims that the
script-tree versions carried are gone — with maturin handling the
editable install, ``catan_rl`` is always importable from the
package's normal location.

``scripts/<name>.py`` retains a 3-line shim for back-compat with
``python scripts/<name>.py`` invocations; the canonical entry point
is the console script.
"""
