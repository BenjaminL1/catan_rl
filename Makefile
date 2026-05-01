.PHONY: install install-gui install-all test test-unit test-integration test-all lint format typecheck train eval bench clean smoke-train smoke-eval

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
