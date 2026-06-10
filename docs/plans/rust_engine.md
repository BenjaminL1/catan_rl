# Rust Engine — Status & Migration Notes

**Last updated:** 2026-06-09
**Status:** Scaffolding only. **NOT** the default backend
(`engine_backend: python`). The production training/eval/BC path is the
pure-Python engine under `src/catan_rl/engine/`.

This document is the single source of truth for the Rust crate. It replaces
the earlier `rust_engine_migration.md` (aspirational R0–R13 plan) and
`rust_engine_actual_state.md` (forensic audit), both deleted; the
substance of both is condensed here.

## TL;DR

A Rust engine crate (`crates/catan_engine/`, exposed via PyO3 + maturin)
was scaffolded to let `n_envs` scale so the policy forward batch lands
where MPS/CUDA wins. **The migration was halted after a measurement showed
the engine is not the bottleneck.** Only `catan_engine.StackedDice` ships
to the production loop; everything else compiles and passes isolated unit
tests but has **zero callers** under `src/catan_rl/`.

## Why it was halted (the measurement)

A full PPO update at `n_envs=128` is **~80% PyTorch SGD, ~20% rollout**
(`analysis/diag_py.log`). The measured Rust rollout speedup was `1.78×` at
`n_envs=128` (`benchmarks/results/bench_phase1.csv`, M1 Pro CPU, policy
forward in the loop), which Amdahl-projects to **~1.09× end-to-end** — far
below the `≥1.15–2.5×` gate. Even a *free* rollout caps end-to-end at
`1/0.80 = 1.25×`. Reaching more requires re-engineering the SGD path
(mixed precision, fused kernels, smaller model), which is out of scope for
an engine migration. The earlier plan's headline "43.9× engine / 6–12×
end-to-end" numbers were Amdahl-blind and **could not be reproduced** from
any artifact in the repo.

Decision (2026-06-06, user PIVOT at remediation Phase 4): terminate the
migration; keep the Rust path *available* for future inference /
deterministic eval / MCTS rollouts; training stays on Python. The Python
engine is canonical and **stays** (no deletion).

## What ships to production

| Artifact | Where | Caller |
|---|---|---|
| `catan_engine.StackedDice` | `crates/catan_engine/src/dice.rs` | `engine/dice.py` → `engine/game.py` → `CatanEnv.reset` |

That is the entire Rust contact surface of `make train`.

## What is built but unwired (zero callers under `src/catan_rl/`)

| Artifact | Where | Note |
|---|---|---|
| `PyRustEnv` (`RustCatanEnv`) | `src/env.rs` | No opponent hook; `truncated` hardcoded `false`. |
| `PyRustVecEnv` (`RustVectorizedEnv`) | `src/vec_env.rs` | `truncs` hardcoded `false`. |
| Native obs encoder | `src/obs.rs` | Slots `[19..79]` zero-filled (vertex/edge/port). Slots `[0..19]` byte-parity with Python (resource/token order, robber bit, dots/5) per `test_obs_cross_impl_byte_parity.py`. |
| Native mask builder | `src/masks.rs` | Reached only via `PyRustEnv`. |
| `PyBoardStatic` | `src/board_static.rs` | Only test/diag callers. |
| `chacha8_keystream` | `src/rng.rs` | Byte-parity test only. |
| `Event` enum + `drain_events()` | `src/{events,env}.rs` | Reached only via `PyRustEnv`. |

The `CATAN_ENGINE_BACKEND` / `engine_backend` switch
(`engine/backend.py:resolve_backend`) is read only by
`analysis/diag_phase_timing.py` and `test_backend_switch.py`; **no**
production code in `env/`, `policy/`, `ppo/`, `selfplay/`, `eval/` branches
on it.

## Still-relevant migration notes (if work ever resumes)

- **Build:** maturin is the **sole** build backend (`pyproject.toml`
  `[tool.maturin] python-source = "src"`); it builds the `.so` into
  `src/catan_engine/` and ships the `catan_rl` Python package in the same
  wheel. `pip install -e .` and `maturin develop --release` are equivalent.
  (A brief hatchling+maturin dual-backend layout caused editable-install
  conflicts and was abandoned.)
- **Cargo release profile:** `lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`, `opt-level = 3`.
- **RNG:** ChaCha8 (`rand_chacha`); byte-parity test vectors in
  `tests/refs/chacha8.py`. Heuristic/BC parity was scoped as *statistical
  equivalence + one-shot BC regen*, not exact `np.random` parity.
- **Known content gaps to fix before any wiring:** obs slots `[19..79]`
  (vertex/edge ownership, port adjacency); `truncated` wiring for GAE
  bootstrap; opponent-injection hook in `env.rs`; register
  `hand_tracker` as a `#[pyclass]`.
- **Go/no-go gate (if resumed):** measure end-to-end against the
  production loop at `n_envs=128` and write the number here before wiring
  the vec env. `< 1.15×` → archive the Rust path; the engine is not the
  bottleneck.

## Maintenance rules

- Any PR adding a bench number must cite the CSV row
  (`benchmarks/results/<timestamp>.csv`) and `platform.machine()`.
- Any PR changing `DEFAULT_BACKEND` (`engine/backend.py`) must update this
  doc and `configs/ppo_default.yaml`.
- Any PR adding/removing/renaming a Rust `#[pyclass]` must update the
  "built but unwired" table above.
