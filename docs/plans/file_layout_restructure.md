# File Layout Restructure — Industry-Standard Reorganization

**Document version:** 1.0
**Created:** 2026-04-30
**Goal:** Restructure `catan_rl/` to a modern Python ML project layout (PEP 517/518 src-layout, mirrored tests, YAML configs, CI/lint/precommit, ADRs) so the codebase is easier to navigate, test, and modify — and so Claude Code can operate on it efficiently.

This is a prerequisite for the [superhuman roadmap](./superhuman_roadmap.md). Do this restructure first; reconcile the roadmap's file paths against the new layout before Phase 0.

---

## 1. Current Layout (problems noted)

```
catan_rl/
├── catan/                       ← flat package; not installable as src-layout
│   ├── __init__.py              ← empty; no public API
│   ├── agents/
│   ├── engine/
│   │   └── debug_board.py       ← debug code mixed with engine
│   ├── gui/
│   └── rl/
│       ├── env.py               ← 50KB single file
│       ├── distributions.py     ← floats at top of rl/
│       ├── debug_wrapper.py     ← floats; debug mixed with prod
│       ├── hand_tracker.py      ← floats
│       ├── models/
│       ├── ppo/                 ← mixes PPO + League + GameManager + eval
│       └── setup/               ← name collides with Python's setup
├── scripts/
│   └── *.py                     ← every script does sys.path.insert(...)
├── checkpoints/                 ← in repo (gitignored), but no separation
├── runs/                        ← same
├── docs/
│   ├── BROADCAST_HAND_TRACKING_PLAN.md          ← stale, partly implemented
│   ├── OPPONENT_AND_CHECKPOINT_ALIGNMENT.md     ← stale
│   ├── POLICY_OPPONENT_IMPLEMENTATION_PLAN.md   ← stale
│   └── plans/superhuman_roadmap.md              ← active
├── requirements.txt             ← only dependency manifest; no pyproject.toml
├── README.md
├── .gitignore
├── .claude/
└── (no tests/, no Makefile, no pyproject.toml, no CI, no precommit, no configs/)
```

**Concrete problems:**
1. **No `src/` layout** — package importable from working dir, so scripts must `sys.path.insert(0, os.path.dirname(...))`. Hides bugs where code only works because of CWD coincidence.
2. **No `pyproject.toml`** — `requirements.txt` is the only manifest; no build config, no tool config (ruff, mypy, pytest), no installable distribution.
3. **No `tests/` directory at all** — cannot mirror tests to source. The eval harness in Phase 0 of the roadmap will create the first tests; doing it in a flat `tests/` mirroring the new `src/` is much cleaner.
4. **`catan/rl/ppo/` is overloaded** — contains the PPO trainer (`ppo.py`), the rollout buffer (`rollout_buffer.py`), the league (`league.py`), the game manager (`game_manager.py`), the evaluation manager (`evaluation_manager.py`), and config (`arguments.py`). These are five separate responsibilities forced into one directory.
5. **`catan/rl/setup/` is ambiguous** — collides with Python's `setup.py` / setuptools concept; non-obvious that it's the setup-phase trainer.
6. **Floating files in `catan/rl/`** (`env.py`, `distributions.py`, `debug_wrapper.py`, `hand_tracker.py`) — should be grouped into purpose-named subdirs (`env/`, `models/`, `eval/`).
7. **No `configs/`** — `arguments.py` is a 100-line Python dict; phase-specific overrides require editing it. The roadmap's Phase 1.5 alone needs 5 ablation configs.
8. **No CI / pre-commit / lint config** — the GAE bug found in Phase 0 of the roadmap could have been caught by an integration smoke test in CI.
9. **Stale docs in `docs/`** — three planning docs that are partly implemented and partly superseded; mixing these with the active roadmap pollutes search.
10. **No ADRs** — design decisions like "1v1 ruleset is invariant" or "perfect hand tracking is OK only because no P2P trade" live in CLAUDE.md but should be permanent records in `docs/decisions/`.
11. **`debug_board.py` and `debug_wrapper.py`** mix with production code; should be in `tools/` or `viz/`.

---

## 2. Proposed Layout (industry-standard)

```
catan_rl/
├── pyproject.toml                # PEP 518 build + tool config (ruff, mypy, pytest)
├── README.md                     # short: how to install, train, evaluate. Links to docs/
├── LICENSE                       # add (MIT recommended; user is academic)
├── CHANGELOG.md                  # rolling per-phase change log
├── Makefile                      # canonical task runner
├── .python-version               # pyenv pin (e.g. 3.11)
├── .editorconfig                 # cross-editor formatting
├── .gitignore
├── .pre-commit-config.yaml       # ruff, black/ruff-format, mypy, end-of-file-fixer
├── .github/
│   └── workflows/
│       ├── ci.yml                # lint + typecheck + tests on PR
│       └── nightly-eval.yml      # optional: champion-bench on schedule
├── .claude/
│   ├── settings.json
│   ├── settings.local.json
│   ├── commands/
│   └── agents/                   # optional project-specific subagents
├── CLAUDE.md                     # repo-root project memory (already exists)
│
├── src/                          # PEP 517 src-layout
│   └── catan_rl/                 # importable as `import catan_rl`
│       ├── __init__.py           # exports public API: CatanEnv, CatanPolicy, CatanPPO
│       ├── py.typed              # marks package as typed
│       │
│       ├── engine/               # pure game logic, no RL deps
│       │   ├── __init__.py
│       │   ├── game.py           # was catan/engine/game.py
│       │   ├── board.py
│       │   ├── player.py
│       │   ├── dice.py
│       │   ├── geometry.py
│       │   ├── broadcast.py
│       │   └── tracker.py        # belief-state ResourceTracker
│       │   # Coords table extracted to data/board_coords.py
│       │
│       ├── agents/               # rule-based opponents
│       │   ├── __init__.py
│       │   ├── heuristic.py
│       │   └── random_ai.py
│       │
│       ├── env/                  # was catan/rl/env.py + neighbors
│       │   ├── __init__.py
│       │   ├── catan_env.py      # split from 50KB monolith into:
│       │   ├── observation.py    #   - observation building
│       │   ├── masks.py          #   - action mask computation
│       │   ├── opponent.py       #   - opponent dispatch (random/heuristic/policy)
│       │   ├── hand_tracker.py
│       │   └── debug_wrapper.py  # was catan/rl/debug_wrapper.py
│       │
│       ├── models/               # neural net components
│       │   ├── __init__.py
│       │   ├── policy.py         # CatanPolicy
│       │   ├── builders.py       # was build_agent_model.py
│       │   ├── distributions.py  # MaskedCategorical
│       │   ├── action_heads.py   # MultiActionHeads
│       │   ├── observation/      # observation encoder sub-modules
│       │   │   ├── __init__.py
│       │   │   ├── observation_module.py
│       │   │   ├── tile_encoder.py
│       │   │   ├── player_modules.py
│       │   │   └── multi_head_attention.py
│       │   ├── graph_encoder.py  # Phase 2 (new)
│       │   ├── belief_head.py    # Phase 2.5b (new)
│       │   ├── opponent_action_head.py  # Phase 2.5c (new)
│       │   ├── recurrent_value.py  # Phase 4 (new)
│       │   └── utils.py          # init_weights, ValueFunctionNormalizer
│       │
│       ├── algorithms/           # RL algorithms (was catan/rl/ppo/)
│       │   ├── __init__.py
│       │   ├── common/
│       │   │   ├── __init__.py
│       │   │   ├── gae.py        # was ppo/utils.py compute_gae*
│       │   │   ├── rollout_buffer.py
│       │   │   └── schedules.py  # entropy/LR annealing helpers
│       │   ├── ppo/
│       │   │   ├── __init__.py
│       │   │   └── trainer.py    # was ppo.py
│       │   ├── ppg/              # Phase 2 (new)
│       │   │   ├── __init__.py
│       │   │   └── aux_phase.py
│       │   └── search/           # Phase 4 (new)
│       │       ├── __init__.py
│       │       └── ismcts.py
│       │
│       ├── selfplay/             # was scattered in ppo/
│       │   ├── __init__.py
│       │   ├── league.py
│       │   ├── game_manager.py
│       │   ├── ratings.py        # Phase 0/3 (new)
│       │   └── exploiters.py     # Phase 3 (new)
│       │
│       ├── eval/                 # evaluation harness
│       │   ├── __init__.py
│       │   ├── evaluation_manager.py
│       │   ├── champion_bench.py # Phase 0 (new)
│       │   ├── exploitability.py # Phase 0 (new)
│       │   └── rules_invariants.py  # Phase 0 (new)
│       │
│       ├── augmentation/         # Phase 1 (new)
│       │   ├── __init__.py
│       │   ├── symmetry_tables.py
│       │   ├── dihedral.py       # D6 group ops
│       │   └── player_swap.py    # 1v1 Z_2 swap
│       │
│       ├── setup_phase/          # was catan/rl/setup/, RENAMED to avoid collision
│       │   ├── __init__.py
│       │   ├── env.py            # was setup_env.py
│       │   ├── policy.py         # was setup_policy.py
│       │   └── trainer.py        # was setup_trainer.py
│       │
│       ├── data/                 # static lookup tables
│       │   ├── __init__.py
│       │   ├── board_coords.py   # 19-hex axial coords (was hardcoded in board.py)
│       │   ├── port_layout.py    # 9 port (hex, corner, corner) tuples
│       │   └── hex_neighbors.py  # adjacency matrix for board validation
│       │
│       ├── viz/                  # debug visualization (gitignored runtime, code committed)
│       │   ├── __init__.py
│       │   ├── debug_board.py    # was catan/engine/debug_board.py
│       │   └── tensorboard.py    # custom TB writers if needed
│       │
│       └── gui/                  # human-play pygame GUI (optional dep)
│           ├── __init__.py
│           └── view.py
│
├── scripts/                      # thin CLI wrappers; no sys.path hacks
│   ├── train.py                  # `python -m catan_rl.cli.train` alternative also fine
│   ├── train_setup.py
│   ├── evaluate.py
│   ├── play_vs_model.py
│   ├── eval_harness.py           # Phase 0 (new)
│   └── migrate_checkpoint.py     # Phase 0 (new)
│
├── tests/                        # NEW: mirrors src/catan_rl/
│   ├── conftest.py               # pytest fixtures (game, env, policy)
│   ├── unit/
│   │   ├── engine/
│   │   │   ├── test_game.py
│   │   │   ├── test_board.py
│   │   │   ├── test_dice.py
│   │   │   ├── test_player.py
│   │   │   └── test_broadcast.py
│   │   ├── env/
│   │   │   ├── test_catan_env.py
│   │   │   ├── test_observation.py
│   │   │   ├── test_masks.py
│   │   │   └── test_hand_tracker.py
│   │   ├── models/
│   │   │   ├── test_policy.py
│   │   │   ├── test_action_heads.py
│   │   │   └── test_distributions.py
│   │   ├── algorithms/
│   │   │   └── test_gae.py       # the truncation fix has a unit test here
│   │   ├── selfplay/
│   │   │   ├── test_league.py
│   │   │   └── test_ratings.py
│   │   ├── eval/
│   │   │   └── test_rules_invariants.py
│   │   └── augmentation/
│   │       ├── test_symmetry_tables.py
│   │       └── test_player_swap.py
│   ├── integration/
│   │   ├── test_smoke_train.py   # 100-step training loop runs without crash
│   │   ├── test_smoke_eval.py    # 5-game eval runs without crash
│   │   └── test_resume.py        # save → load → resume produces same step
│   └── fixtures/
│       └── tiny_checkpoint.pt    # 100-step checkpoint for fast tests
│
├── benchmarks/                   # NEW: perf regression checks (CPU FPS, etc.)
│   ├── bench_rollout.py
│   ├── bench_inference.py
│   └── bench_obs_build.py
│
├── configs/                      # NEW: phase-specific YAML configs
│   ├── _base.yaml                # shared defaults
│   ├── phase0_baseline.yaml
│   ├── phase0_fixed.yaml
│   ├── phase1_full.yaml
│   ├── phase1_no_value_clip.yaml
│   ├── phase1_no_symmetry_aug.yaml
│   ├── phase2_full.yaml
│   ├── phase2_ppg.yaml
│   ├── phase3_full.yaml
│   └── eval_harness.yaml
│
├── docs/
│   ├── README.md                 # docs index
│   ├── architecture.md           # one-pager: obs / action / training loop diagram
│   ├── obs_schema.md             # canonical obs keys, dims, ranges
│   ├── action_schema.md          # canonical 6 heads, mask keys
│   ├── 1v1_rules.md              # rules invariant table (already in CLAUDE.md, here as canonical)
│   ├── plans/
│   │   ├── superhuman_roadmap.md
│   │   ├── file_layout_restructure.md  # this document
│   │   └── archive/              # historical, partly implemented plans
│   │       ├── BROADCAST_HAND_TRACKING_PLAN.md
│   │       ├── OPPONENT_AND_CHECKPOINT_ALIGNMENT.md
│   │       └── POLICY_OPPONENT_IMPLEMENTATION_PLAN.md
│   └── decisions/                # ADRs (Architecture Decision Records)
│       ├── 0001-1v1-rules-invariant.md
│       ├── 0002-perfect-hand-tracking.md
│       ├── 0003-charlesworth-style-obs.md
│       ├── 0004-six-head-autoregressive-actions.md
│       ├── 0005-cpu-only-baseline.md
│       └── 0006-src-layout-restructure.md   # records this restructure
│
├── notebooks/                    # NEW: optional Jupyter exploration
│   ├── README.md                 # notebook hygiene rules
│   └── analysis/
│       ├── checkpoint_inspect.ipynb
│       └── obs_distribution.ipynb
│
├── examples/                     # NEW: small runnable demos
│   ├── play_one_game.py          # programmatic single game w/ random policy
│   ├── load_and_evaluate.py
│   └── README.md
│
├── checkpoints/                  # gitignored; training output
│   └── train/
└── runs/                         # gitignored; TensorBoard
    └── train/
```

---

## 3. Rationale (why each change helps)

### 3.1 Wins for code quality

| Change | Why |
|---|---|
| `src/catan_rl/` layout | Forces installable package. No more `sys.path.insert(...)` in scripts. Every test imports from the installed package, catching CWD-coupling bugs. |
| `pyproject.toml` | Single source of truth: dependencies, build, ruff, mypy, pytest config. Replaces `requirements.txt`. |
| `Makefile` | Canonical task runner: `make test`, `make train`, `make eval`, `make lint`, `make typecheck`. Eliminates "what flags do I pass?" confusion. |
| `tests/` mirroring `src/` | Find any test by `s|src|tests/unit|`. New modules get tests as part of merge. |
| `configs/` YAML | Phase-specific configs are diffable and orthogonal. Phase 1 ablations need 5 configs; YAML beats editing `arguments.py` 5 times. |
| `.pre-commit-config.yaml` | Lint/format/typecheck on every commit. Catches drift before review. |
| `.github/workflows/ci.yml` | The GAE bug Phase 0 fixes could have been caught by a 100-step integration test in CI. Investing in CI now pays off across all phases. |
| `docs/decisions/` ADRs | "Why is hand tracking perfect?" "Why 1v1 only?" become permanent records. Reduces re-litigating settled choices. |
| `docs/plans/archive/` | Stale planning docs move out of search results. Active plans stay in `docs/plans/`. |

### 3.2 Wins for splitting overloaded directories

**Current `catan/rl/ppo/` mixes 5 responsibilities** — break it apart:
- `algorithms/ppo/trainer.py` — the PPO update loop
- `algorithms/common/` — GAE, rollout buffer, schedules (reused by future PPG)
- `selfplay/league.py` + `selfplay/game_manager.py` + `selfplay/ratings.py` — opponent diversity
- `eval/evaluation_manager.py` + `eval/champion_bench.py` + `eval/exploitability.py` — evaluation harness
- `configs/*.yaml` — what was in `arguments.py`

**Current `catan/rl/env.py` is 50KB / 1200 lines** — split into:
- `env/catan_env.py` — Gymnasium wrapper, state machine
- `env/observation.py` — `_build_tile_features`, `_build_current_player_main`, `_build_next_player_main`, `_build_dev_sequences`
- `env/masks.py` — `_compute_masks`, `get_action_masks`
- `env/opponent.py` — `_run_opponent_turn`, `_run_policy_opponent_turn`, `_execute_action_for_player`
- `env/hand_tracker.py` — moved from `catan/rl/`
- `env/debug_wrapper.py` — moved from `catan/rl/`

**Current `catan/rl/setup/` collides with `setup.py`** — rename to `setup_phase/`.

### 3.3 Wins for Claude Code specifically

| Change | How it helps Claude |
|---|---|
| `src/` layout enforces package install | Claude doesn't need to guess at `sys.path` configurations; `import catan_rl.engine.game` just works. |
| `tests/` mirrors `src/` | When asked to add a test for a module, Claude finds the right file by mirror path. No ambiguity. |
| `Makefile` is canonical | Claude runs `make test` not `pytest tests/ -v --cov=catan_rl ...`. Reduces cognitive load and avoids stale flag invocations. |
| Configs in YAML | When the user asks for an ablation, Claude copies a YAML, changes one key, commits. No editing of Python config dicts (which are harder to diff). |
| ADRs in `docs/decisions/` | When considering a change, Claude can grep for ADRs that constrain it (e.g. "1v1 only"). Permanent records reduce drift. |
| Stale plans archived | When searching docs, Claude doesn't surface implemented/superseded plans (BROADCAST_HAND_TRACKING_PLAN.md is partly stale). |
| Per-dir `CLAUDE.md` (optional) | Module-specific guidance for engine/, models/, selfplay/. Loaded only when relevant directories are touched. |
| `data/` for static tables | Hardcoded coord dicts (`board.py:88-90`, `checkHexNeighbors:119-124`, ports `190-203`) move to declarative data files. Easier to read, validate, swap. |
| Splitting overloaded dirs | When asked to "fix the league sampler," Claude opens `selfplay/league.py` directly instead of grepping `catan/rl/ppo/`. |
| `examples/` | Smoke-test scripts Claude can run after refactors to verify no regression. |

---

## 4. New Files to Add (top of mind)

### 4.1 `pyproject.toml` (replaces `requirements.txt`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "catan-rl"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=0.29",
    "numpy>=1.24",
    "torch>=2.0",
    "tensorboard>=2.14",
    "tqdm>=4.65",
    "trueskill>=0.4",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
gui = ["pygame>=2.6"]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.0",
]
notebooks = ["jupyter>=1.0", "matplotlib>=3.8"]

[project.scripts]
catan-rl-train = "catan_rl.scripts.train:main"
catan-rl-eval = "catan_rl.scripts.evaluate:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true  # for torch + gymnasium until stubs land

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
markers = [
    "slow: tests that take >5s",
    "integration: smoke training tests",
]
```

### 4.2 `Makefile`

```make
.PHONY: install test lint typecheck format train eval bench clean

install:
	pip install -e ".[dev,gui]"
	pre-commit install

test:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v -m integration

test-all: test test-integration

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src

train:
	python scripts/train.py --config configs/phase0_fixed.yaml --verbose

eval:
	python scripts/eval_harness.py --mode all --champion checkpoints/train/checkpoint_07390040.pt

bench:
	python benchmarks/bench_rollout.py
	python benchmarks/bench_inference.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
```

### 4.3 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        files: ^src/
        additional_dependencies: [types-PyYAML]
```

### 4.4 `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: make lint
      - run: make typecheck
      - run: make test
      - run: make test-integration
```

### 4.5 ADR template (`docs/decisions/0001-1v1-rules-invariant.md`)

```markdown
# ADR 0001: 1v1 Rules Are Invariant

## Status
Accepted (2026-04-30)

## Context
Catan has 4-player and 1v1 (Colonist.io) variants. The 1v1 ruleset differs in
win VP (15 vs 10), discard threshold (9 vs 7), Friendly Robber, StackedDice,
disabled P2P trade, and snake-draft setup specifics.

## Decision
This project implements **1v1 only**. The rules are invariants enforced by
`catan_rl.eval.rules_invariants.run()` and validated by every eval harness run.

## Consequences
- Action space has no propose/accept/counter trade actions.
- Observation models exactly one opponent.
- Hand tracking is deterministic-perfect (relies on no P2P trade).
- Self-play machinery is 2-player symmetric zero-sum.
- D6 board symmetry × Z_2 player swap = 24-fold data augmentation.

## Alternatives Considered
- Generalize to N-player: requires belief-state hand tracking, multi-opponent
  obs encoder, propose/accept trade actions. Out of scope.
```

---

## 5. Migration Plan (5 stages, each its own PR)

Total wall-clock estimate: **2–3 days** (mostly mechanical moves; one day of fixup).

### Stage A — Tooling foundation (no moves yet)
**~3 hours**
- Add `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.editorconfig`, `.python-version`.
- Add `.github/workflows/ci.yml` running lint + typecheck + test (tests are empty for now; CI passes trivially).
- Add empty `tests/` skeleton.
- `pip install -e ".[dev]"` works.
- **PR title:** `chore: add pyproject.toml, Makefile, pre-commit, CI scaffolding`
- **Verify:** `make install && make lint && make test` passes.

### Stage B — Move `catan/` → `src/catan_rl/`
**~1 day**
- `git mv catan src/catan_rl`
- Update every `from catan.X` → `from catan_rl.X` (~150 imports — use a script or sed).
- Update `scripts/*.py` to remove `sys.path.insert`.
- Add `src/catan_rl/__init__.py` exporting public API: `CatanEnv`, `CatanPolicy`, `CatanPPO`, `EvaluationManager`.
- Add `src/catan_rl/py.typed`.
- Update `.gitignore` for `src/catan_rl/__pycache__/`.
- **PR title:** `refactor: adopt src-layout, rename package to catan_rl`
- **Verify:** `make test` (still empty), `python scripts/train.py --total-timesteps 100 --verbose` smoke-runs.
- **Checkpoint compatibility:** `CatanPPO.load('checkpoints/train/checkpoint_07390040.pt')` may need a one-shot upgrade for any pickled module references; bake into `migrate_checkpoint.py` before this PR if so.

### Stage C — Split overloaded dirs
**~1 day**
- Split `catan_rl/rl/ppo/` → `catan_rl/algorithms/`, `catan_rl/selfplay/`, `catan_rl/eval/`.
- Split `catan_rl/rl/env.py` → `catan_rl/env/{catan_env,observation,masks,opponent,hand_tracker,debug_wrapper}.py`.
- Move `catan_rl/rl/distributions.py` → `catan_rl/models/distributions.py`.
- Move `catan_rl/rl/models/` → `catan_rl/models/` (drop the redundant `rl/` layer).
- Rename `catan_rl/rl/setup/` → `catan_rl/setup_phase/`.
- Move `catan_rl/engine/debug_board.py` → `catan_rl/viz/debug_board.py`.
- Extract hardcoded coord/port/neighbor tables from `engine/board.py` into `catan_rl/data/{board_coords,port_layout,hex_neighbors}.py`.
- **PR title:** `refactor: split catan_rl/rl into algorithms, selfplay, eval, env, models`
- **Verify:** `make test`, `make lint`, smoke-train still passes.

### Stage D — Configs + initial tests
**~half day**
- Convert `arguments.py` `TRAIN_CONFIG` and `MODEL_CONFIG` to `configs/_base.yaml`.
- Add `configs/phase0_baseline.yaml` and `configs/phase0_fixed.yaml` skeletons.
- Update `scripts/train.py` to load YAML config via `--config <path>`.
- Add minimal unit tests: `tests/unit/engine/test_dice.py`, `tests/unit/algorithms/test_gae.py`, `tests/unit/eval/test_rules_invariants.py`. These three tests **directly support Phase 0** of the roadmap.
- **PR title:** `feat: yaml configs and initial test suite`
- **Verify:** `make test` runs ≥ 5 tests; `python scripts/train.py --config configs/phase0_baseline.yaml --total-timesteps 100` works.

### Stage E — Docs and ADRs
**~3 hours**
- Move stale docs into `docs/plans/archive/`.
- Write `docs/architecture.md`, `docs/obs_schema.md`, `docs/action_schema.md`, `docs/1v1_rules.md` (extract from `CLAUDE.md`).
- Write ADRs 0001–0006 (templates above).
- Update root `README.md` to be short (install + train + eval) and link to `docs/`.
- Add `docs/README.md` index.
- **PR title:** `docs: ADRs, architecture overview, archive stale plans`
- **Verify:** all internal links resolve.

---

## 6. Reconciliation with the Superhuman Roadmap

Once the restructure lands, edit `docs/plans/superhuman_roadmap.md` Section 13 (File List) so every path matches the new layout. Specifically:

| Old path (in roadmap) | New path |
|---|---|
| `catan/rl/env.py` | `src/catan_rl/env/catan_env.py` (and split children) |
| `catan/rl/ppo/utils.py` | `src/catan_rl/algorithms/common/gae.py` |
| `catan/rl/ppo/rollout_buffer.py` | `src/catan_rl/algorithms/common/rollout_buffer.py` |
| `catan/rl/ppo/ppo.py` | `src/catan_rl/algorithms/ppo/trainer.py` |
| `catan/rl/ppo/league.py` | `src/catan_rl/selfplay/league.py` |
| `catan/rl/ppo/game_manager.py` | `src/catan_rl/selfplay/game_manager.py` |
| `catan/rl/ppo/evaluation_manager.py` | `src/catan_rl/eval/evaluation_manager.py` |
| `catan/rl/ppo/arguments.py` | `configs/*.yaml` (no longer Python) |
| `catan/rl/eval/rules_invariants.py` | `src/catan_rl/eval/rules_invariants.py` |
| `catan/rl/hand_tracker.py` | `src/catan_rl/env/hand_tracker.py` |
| `catan/rl/distributions.py` | `src/catan_rl/models/distributions.py` |
| `catan/rl/models/*` | `src/catan_rl/models/*` (drop the `rl/` layer) |
| `catan/rl/setup/*` | `src/catan_rl/setup_phase/*` |
| `catan/rl/augmentation.py` | `src/catan_rl/augmentation/dihedral.py` (split out) |
| `catan/rl/ppo/symmetry_tables.py` | `src/catan_rl/augmentation/symmetry_tables.py` |
| `catan/rl/ppo/player_swap.py` | `src/catan_rl/augmentation/player_swap.py` |
| `catan/rl/ppo/ratings.py` | `src/catan_rl/selfplay/ratings.py` |
| `catan/rl/models/graph_encoder.py` | `src/catan_rl/models/graph_encoder.py` |
| `catan/rl/models/belief_head.py` | `src/catan_rl/models/belief_head.py` |
| `catan/rl/models/opponent_action_head.py` | `src/catan_rl/models/opponent_action_head.py` |
| `catan/rl/ppo/ppg.py` | `src/catan_rl/algorithms/ppg/aux_phase.py` |
| `catan/rl/search/ismcts.py` | `src/catan_rl/algorithms/search/ismcts.py` |
| `catan/rl/models/recurrent_value.py` | `src/catan_rl/models/recurrent_value.py` |

Update `CLAUDE.md` "Layout" section similarly.

---

## 7. What this restructure does NOT do

To keep scope contained, this restructure does **not**:

- Change any algorithm or hyperparameter (that's the roadmap's job).
- Change checkpoint format (one-shot migration handles `checkpoint_07390040.pt`).
- Change TensorBoard log directory layout (`runs/train/` stays).
- Add CUDA-specific code paths.
- Touch the engine's game logic (it stays in `engine/` unchanged).
- Modify the action space, observation schema, or rules (those are 1v1 invariants).
- Add new training capability (Phase 0+ does that; this is purely structural).
- Reorganize TensorBoard scalar names.

---

## 8. Risks

| Severity | Risk | Mitigation |
|---|---|---|
| HIGH | Stage B mass rename breaks something silently | Run smoke-train after each stage; CI catches lint/typecheck regressions |
| HIGH | Pickled checkpoints reference old module paths (`catan.rl.models.policy`) | One-shot migration script that rewrites pickled module references; validated against `checkpoint_07390040.pt` before merging Stage B |
| MEDIUM | Existing imports outside the package (notebooks, user scripts) break | Document the rename clearly in the Stage B PR; search-and-replace template |
| MEDIUM | `arguments.py` → YAML loses Python features (e.g. tuple defaults) | YAML loader has a custom `!tuple` tag, or convert tuples to lists where they round-trip safely |
| LOW | `pre-commit` slows down commits | Use `--no-verify` for emergency commits; document the escape hatch |
| LOW | `src/` layout requires editable install (`pip install -e .`) | Document in README and Makefile; standard practice |

---

## 9. Quick Decision Summary

**Do this restructure if:**
- You plan to implement the superhuman roadmap (it benefits from a clean structure).
- You want CI / tests / lint / typecheck enforced from now on.
- You want Claude Code to navigate the codebase faster.

**Skip this restructure if:**
- You plan to implement only one or two phases of the roadmap.
- You're comfortable with current layout and don't want a 2–3 day mechanical detour.
- You have unmerged work-in-progress that would conflict catastrophically with mass renames.

**If skipping:** at minimum, do **Stage A** (pyproject.toml, Makefile, pre-commit, CI scaffolding) and **Stage D** (a few unit tests). Those two are nearly free and unlock the most value for Claude.

---

**End of restructure plan.** Reconcile the roadmap (`docs/plans/superhuman_roadmap.md` Section 13) with this layout before beginning Phase 0.
