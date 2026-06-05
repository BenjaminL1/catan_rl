.PHONY: install install-gui install-all test test-unit test-integration test-all lint format typecheck train eval bench clean smoke-train smoke-eval rust-setup rust-build rust-test rust-bench rust-clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install || true

install-gui:
	$(PYTHON) -m pip install -e ".[dev,gui]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,gui,ratings,notebooks]"

test: test-unit

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v -m integration

test-all:
	pytest tests -v

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

train:
	$(PYTHON) scripts/train.py --verbose

eval:
	$(PYTHON) scripts/evaluate.py checkpoints/train/checkpoint_07390040.pt --opponent heuristic --n-games 100

smoke-train:
	$(PYTHON) scripts/train.py --total-timesteps 200 --verbose

smoke-eval:
	$(PYTHON) scripts/evaluate.py checkpoints/train/checkpoint_07390040.pt --opponent random --n-games 5

bench:
	@echo "Benchmarks not yet implemented. See benchmarks/ once added in Phase 1."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# Rust extension build targets (see docs/plans/rust_engine_migration.md).
#
# ``rust-setup`` templates ``.cargo/config.toml`` from the
# checked-in ``.cargo/config.toml.example``, substituting the
# current Python interpreter's LIBDIR for the rpath placeholder.
# Run this once on first clone; the generated file is gitignored.
rust-setup:
	@test -f .cargo/config.toml.example || (echo "missing .cargo/config.toml.example"; exit 1)
	@LIBDIR=$$($(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") ; \
	    sed "s|__PYTHON_LIBDIR__|$$LIBDIR|g" .cargo/config.toml.example > .cargo/config.toml ; \
	    echo "wrote .cargo/config.toml with LIBDIR=$$LIBDIR"

# ``rust-build`` runs maturin in --release mode and installs the
# compiled ``catan_engine`` .so into the active virtualenv's
# site-packages. NB: maturin uninstalls the hatchling-installed
# ``catan-rl`` wheel as a side effect — for the ``catan_rl`` Python
# package, run development scripts with ``PYTHONPATH=src`` until
# the maturin/hatchling editable-install conflict is resolved
# (see ``docs/plans/file_layout_restructure.md``).
rust-build:
	maturin develop --release

# ``cargo test --release`` runs the pure-Rust unit tests under
# crates/catan_engine/src/**. Use this for fast iteration on game
# rules without crossing the Python boundary.
rust-test:
	cargo test --workspace --release

# Criterion benchmarks land from R3 onward.
rust-bench:
	cargo bench --workspace

# Wipe the cargo target dir; forces a full rebuild on next ``make rust-build``.
rust-clean:
	cargo clean
