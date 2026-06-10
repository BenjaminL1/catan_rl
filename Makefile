.PHONY: install install-gui install-all test test-unit test-integration test-all lint format typecheck train bench clean smoke-train rust-setup rust-build rust-test rust-bench rust-clean

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
	$(PYTHON) scripts/train.py --config configs/ppo_default.yaml --verbose

# Evaluation runs inside the training loop (catan_rl.eval.harness, periodic
# WR-vs-heuristic with Wilson CIs); there is no standalone eval CLI in v2.

smoke-train:
	$(PYTHON) scripts/train.py --dry-run

# ``bench`` runs the engine throughput harness (Rust migration plan
# Phase 1) and writes CSV + JSON to ``benchmarks/results/``. The
# harness must run a frozen ``CatanPolicy.forward()`` inside the
# timing loop so the numbers reflect actual training-loop conditions,
# not env-step throughput in isolation — env stepping is not the
# bottleneck per ``analysis/diag_*.log`` (SGD = ~80% of wall-time).
# See ``docs/plans/rust_engine.md``.
bench:
	@if [ ! -f scripts/bench_engine.py ]; then \
	    echo "scripts/bench_engine.py not yet implemented (Phase 1 of the Rust migration remediation plan). See docs/plans/rust_engine_actual_state.md."; \
	    exit 1; \
	fi
	$(PYTHON) scripts/bench_engine.py --all --n-steps 1024

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
# site-packages. With maturin as the sole build backend (see
# ``pyproject.toml`` + ``docs/plans/rust_engine_migration.md``),
# ``pip install -e .`` produces an equivalent editable install
# AND wires the ``[project.scripts]`` console-script entries
# (``catan-rl-train`` etc.). Use ``make install`` as the default
# editable-install path; reach for ``rust-build`` only when you
# want a fast inner loop on the Rust ``.so`` without re-running
# the Python wheel build.
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
