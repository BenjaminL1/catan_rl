# ADR 0006: Adopt PEP 517/518 src-Layout

**Status:** Accepted
**Date:** 2026-04-30

## Context

The original layout was `catan/{engine,agents,gui,rl/{models,ppo,setup}}` with no `src/`, no `pyproject.toml`, no `tests/`, and `sys.path.insert(...)` hacks in every script. This made the project hard to import as a library, harder to test, and easy to drift (the package was importable from CWD only when the user happened to be in the right directory).

The PR re-organized the project into the standard Python ML layout. See `docs/plans/file_layout_restructure.md` for the full plan.

## Decision

- Top-level package: `src/catan_rl/` (PEP 517/518 src-layout).
- Build system: `pyproject.toml` with hatchling backend.
- Test suite: `tests/{unit,integration}/` mirroring `src/`.
- Configs: `configs/*.yaml` (phase-specific YAML, with `_base` inheritance).
- CI: GitHub Actions running lint + typecheck + tests.
- Pre-commit: ruff (lint + format), mypy, basic hygiene hooks.
- Dev tooling pinned via `pyproject.toml [project.optional-dependencies] dev`.

The internal layout dropped the redundant `catan/rl/` layer and split the overloaded `ppo/` directory:

| Old | New |
|---|---|
| `catan/engine/` | `src/catan_rl/engine/` |
| `catan/agents/` | `src/catan_rl/agents/` |
| `catan/gui/` | `src/catan_rl/gui/` |
| `catan/engine/debug_board.py` | `src/catan_rl/viz/debug_board.py` |
| `catan/rl/env.py` | `src/catan_rl/env/catan_env.py` |
| `catan/rl/distributions.py` | `src/catan_rl/models/distributions.py` |
| `catan/rl/hand_tracker.py` | `src/catan_rl/env/hand_tracker.py` |
| `catan/rl/debug_wrapper.py` | `src/catan_rl/env/debug_wrapper.py` |
| `catan/rl/models/*` | `src/catan_rl/models/*` |
| `catan/rl/ppo/ppo.py` | `src/catan_rl/algorithms/ppo/trainer.py` |
| `catan/rl/ppo/utils.py` | `src/catan_rl/algorithms/common/gae.py` |
| `catan/rl/ppo/rollout_buffer.py` | `src/catan_rl/algorithms/common/rollout_buffer.py` |
| `catan/rl/ppo/arguments.py` | `src/catan_rl/algorithms/ppo/arguments.py` |
| `catan/rl/ppo/league.py` | `src/catan_rl/selfplay/league.py` |
| `catan/rl/ppo/game_manager.py` | `src/catan_rl/selfplay/game_manager.py` |
| `catan/rl/ppo/evaluation_manager.py` | `src/catan_rl/eval/evaluation_manager.py` |
| `catan/rl/setup/` | `src/catan_rl/setup_phase/` |

## Consequences

- All imports rewritten from `from catan.X` to `from catan_rl.X` (~150 imports).
- `pip install -e ".[dev]"` is the canonical install command.
- Scripts (`scripts/*.py`) now contain a tiny `sys.path.insert(0, "src")` shim that lets them run without `pip install -e .`. Once the user has installed the package, the shim is a no-op.
- `python -m pytest tests/unit` runs the suite; `make test` is the canonical command.
- Old plans in `docs/plans/archive/` reference the old paths and are not corrected (they are archived, not active).
- The active roadmap (`docs/plans/superhuman_roadmap.md`) and `CLAUDE.md` are updated to reference the new paths.

## Alternatives Considered

- **Keep flat `catan/` package, add `pyproject.toml` only.** Rejected: doesn't fix the `sys.path.insert` smell and makes the `catan` import path ambiguous (the engine is in `catan.engine`, but the project name is also `catan`).
- **Skip the rename and keep `catan` as the package name.** Rejected: the project repo is `catan_rl` and the package name should match for installability and discoverability.

## Related

- `docs/plans/file_layout_restructure.md` (full plan)
- `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`
