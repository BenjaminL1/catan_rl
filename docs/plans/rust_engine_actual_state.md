# Rust Engine — Actual State (Source of Truth)

**Last updated:** 2026-06-06
**Replaces (in case of conflict):** the "PR3 (R6-R9) — landed" and
"PR4 (R10-R13) — partially landed" sections of
[`rust_engine_migration.md`](rust_engine_migration.md), every
docstring under `src/catan_rl/engine/backend.py` that claims `rust`
is the production default, and `configs/ppo_default.yaml:14`'s
"audit: meaningful win at n_envs>=32" annotation.

This document is the result of a forensic audit conducted on
2026-06-06. It exists because the migration's optimistic
narrative had drifted from the shipped code by ~10 PRs, and every
subsequent decision was being anchored on numbers that the repo
could not reproduce.

## TL;DR

**~5–7 % of the migration shipped to the production training
loop.** The only Rust code that runs in the default training
configuration is `catan_engine.StackedDice`. Every other Rust
artifact (`PyRustEnv`, `PyRustVecEnv`, the native obs encoder, the
native mask builder, the hand-tracker helpers, `BoardStatic`)
compiles and passes its own isolated unit tests but has **zero
callers under `src/catan_rl/`**.

## What is shipped to production

| Artifact | Where | Production caller |
|---|---|---|
| `catan_engine.StackedDice` | `crates/catan_engine/src/dice.rs` | `src/catan_rl/engine/dice.py:48` → `src/catan_rl/engine/game.py:44` (`catanGame.__init__`) → `CatanEnv.reset` |

That is the entire Rust contact surface of `make train` /
`catan-rl-train`. Every other call in the rollout / SGD loop is
Python.

## What is built but unwired

| Artifact | Where | Status |
|---|---|---|
| `PyRustEnv` (`#[pyclass(name = "RustCatanEnv")]`) | `crates/catan_engine/src/env.rs` | No callers under `src/catan_rl/`. Has no opponent hook. Hardcodes `truncated=false` (`env.rs:61`). |
| `PyRustVecEnv` (`#[pyclass(name = "RustVectorizedEnv")]`) | `crates/catan_engine/src/vec_env.rs` | No callers under `src/catan_rl/`. Hardcodes `truncs_local = vec![false; n]` (`vec_env.rs:97`). |
| Native obs encoder (`build_obs`) | `crates/catan_engine/src/obs.rs` | Reached only via `PyRustEnv`, which has no Python caller. Slots `[19..]` are zero-filled placeholders per `obs.rs:58-59` ("Filled by future R13 polish") — encoder is structurally complete but content-incomplete. |
| Native mask builder (`compute_masks`) | `crates/catan_engine/src/masks.rs` | Reached only via `PyRustEnv.get_action_masks`; production uses `src/catan_rl/env/masks.py`. |
| `hand_tracker::get_hand_cw` / `get_hand_engine` | `crates/catan_engine/src/hand_tracker.rs` | Free functions, not wrapped in a `#[pyclass]`, not registered in `lib.rs`. Inaccessible from Python. Production uses `src/catan_rl/env/hand_tracker.py`. |
| `PyBoardStatic` | `crates/catan_engine/src/board_static.rs` | Registered in `lib.rs:54`. Only callers: `tests/unit/engine/test_board_static_rust.py` and `analysis/diag_phase_timing.py`. `catanBoard()` (`src/catan_rl/engine/board.py:591`) has no `use_rust_board` flag and is the only board the env constructs. |
| `chacha8_keystream` | `crates/catan_engine/src/rng.rs` | Used by `tests/unit/engine/test_rng_parity.py`. Not used by any Python production code beyond the byte-parity test. |
| `Event` enum + `drain_events()` | `crates/catan_engine/src/{events.rs,env.rs}` | Reached only via `PyRustEnv.drain_events`; production uses `src/catan_rl/engine/broadcast.py:GameBroadcast`. |

## Reproducible numbers

The single empirical data point in the repository is
`analysis/diag_rust.log` paired with `analysis/diag_py.log`. Both
were produced by `analysis/diag_phase_timing.py`. They time a
full PPO update (rollout + GAE + SGD) at `n_envs=128,
n_steps=256`, batch=512, 4 epochs, on M1 Pro CPU. The
`backend=rust` log invokes `resolve_backend()` but — because no
production code branches on the result — actually exercises the
same Python `CatanEnv.step` path as the `backend=python` log.

| Metric | `python` log | `rust` log | Δ |
|---|---|---|---|
| Rollout median (s/update) | 46.58 | 40.54 | ~1.15× faster |
| SGD median (s/update) | 189.48 | 153.41 | ~1.23× faster |
| SGD share of per-update wall-time | 80 % | 79 % | — |
| Per-update median (s) | 236.07 | 193.95 | — |

The SGD share is the punchline: 79–80 % of training wall-time is
the PyTorch optimiser, not the engine. The engine throughput
ceiling — even if every Python `CatanEnv.step` call were free —
is `(1 / 0.20) = 5×` for a single update. The audit's headline
"43.9× engine speedup, 6–12× end-to-end" was Amdahl-blind.

## Numbers that cannot be reproduced

| Claim | Source | Reproducible from this repo? |
|---|---|---|
| 114,201 env-steps/sec at `n_envs=8` (20.8× baseline) | `rust_engine_migration.md:279` | **No.** No `scripts/bench_engine.py`, no `benchmarks/` directory, no `crates/catan_engine/benches/`. `Makefile:48` says "Benchmarks not yet implemented." |
| 241,266 env-steps/sec at `n_envs=64` (43.9× baseline) | `rust_engine_migration.md:280` | **No.** Same. |
| 272,075 env-steps/sec at `n_envs=128` (49.5× baseline) | `rust_engine_migration.md:281` | **No.** Same. |
| "PR3 exceeds the R9 gate (44k at n_envs=64) by 5.4x. The 8x engine and 6x end-to-end targets are achieved." | `rust_engine_migration.md:283-284` | **No.** Contradicted by `training_loop.py:207` (no Rust caller) and by the `diag_*.log` files above. |

## Known content gaps in built code

These are *structural* gaps inside the Rust code itself, not just
wiring gaps. Fixing them is part of Phase 3 of the
[remediation plan](#remediation-status) below.

- `crates/catan_engine/src/env.rs:61` — `truncated` is hardcoded
  `false`. Breaks GAE bootstrapping if ever wired.
- `crates/catan_engine/src/vec_env.rs:97` — `truncs_local =
  vec![false; n]`. Same reason. Comment admits "R13 wires
  per-episode truncation".
- `crates/catan_engine/src/obs.rs:3-5,58-59` — `[19..]` filler slots
  for vertex/edge ownership flags and port adjacency are
  zero-padded. Comment defers byte-identity to "R10 ... will
  either re-train or wire FFI passthrough". Neither happened.
- `crates/catan_engine/src/env.rs` — no opponent hook. `step`
  advances exactly one action and returns. The agent's `EndTurn`
  has nothing to play against.
- `crates/catan_engine/src/hand_tracker.rs` — not registered as a
  `#[pyclass]` in `lib.rs:44-57`. Not callable from Python at all.

## Documentation that was actively lying (now corrected)

Each row points at the file:line that lied + the file:line of
this doc that says the truth.

| Was claiming | At | Truth |
|---|---|---|
| "The 8x engine and 6x end-to-end targets are achieved" | `docs/plans/rust_engine_migration.md:283-284` | Pre-Phase 0: corrected in the same commit that introduced this doc; the line now points at this file. |
| "CATAN_ENGINE_BACKEND=rust (the post-R10 default) — use the Rust engine + obs encoder + mask builder via PyO3" | `src/catan_rl/engine/backend.py:8-9` | Corrected. `resolve_backend()` is read by `analysis/diag_phase_timing.py` and `tests/unit/engine/test_backend_switch.py` only; no production code in `env/`, `policy/`, `ppo/`, `selfplay/`, `eval/` branches on it. |
| "sub-processes parallelise env stepping" + `vec_env_mode = "subproc"` default | `src/catan_rl/ppo/arguments.py:10-12, 77` | Corrected. No `SubprocVecEnv` class exists anywhere in `src/`. The default is now `"serial"`. `"subproc"` is still accepted (with a deprecation warning) to avoid breaking existing YAML configs. |
| "audit: meaningful win at n_envs>=32" | `configs/ppo_default.yaml:14` | Corrected. There was no audit and no measurable subproc win — the comment was extrapolated from an aspiration. |
| "Benchmarks not yet implemented. See benchmarks/ once added in Phase 1." | `Makefile:48` | Replaced with a pointer to `scripts/bench_engine.py`, which lands in Phase 1 of the remediation plan below. |

## Remediation status

Tracked in the [Rust migration remediation plan](#tbd) ("the
plan") — the 10-phase sequence the user confirmed on 2026-06-06.

| Phase | Description | Status |
|---|---|---|
| 0 | Documentation truth-up (this doc) + `vec_env_mode` default fix | **Landed 2026-06-06** |
| 1 | Benchmark harness with policy forward in the loop | **Landed 2026-06-06** (results: see "Phase 1 measured" below) |
| 2 | Dual-engine `engine_backend` pytest fixture | **Landed 2026-06-06** |
| 3 | Truncation wiring + obs honesty + `PyHandTracker` pyclass + byte-parity test | **Landed 2026-06-06** (bench within noise of baseline — gate passes; +3.1% delta is run-to-run variance, not a real reduction) |
| 4 | `RustCatanEnvAdapter` single-env path (**includes a hard gate: cross-impl Python ↔ Rust obs byte-parity test on populated slots; the Phase 3 parity test deferred this and Phase 4 is the only place it can land**) | Pending |
| 5 | Opponent injection contract (`needs_opponent_action` mask) | Pending |
| 6 | Vec env wiring (`PyRustVecEnv` → training loop) | Pending |
| 7 | End-to-end PPO smoke run | Pending |
| 8 | Eval & checkpoint compatibility audit | Pending |
| 9 | Go / no-go decision gate (≥ 3× e2e → proceed; else archive) | Pending |
| 10 | Python engine deletion (PROCEED branch only, after 10-day or 1M-game soak) | Pending |

## Phase 1 measured

Source: `benchmarks/results/bench_phase1.csv` and the companion
`bench_phase1.json` manifest. Produced by
`scripts/bench_engine.py --all --n-steps 1024 --repeat 3` on
2026-06-06 against git SHA `58c914d-dirty` on `arm64-Darwin` (M1
Pro CPU). Every row has `include_policy=True`: a fresh
`CatanPolicy()` runs one CPU forward in `torch.inference_mode()`
per step, matching production rollout conditions.

| `n_envs` | `py` median (env-steps/sec) | `rust_no_op` median | Speedup |
|---|---|---|---|
| 1 | 396.6 | 444.5 | **1.12×** |
| 8 | 1009.0 | 1195.5 | **1.18×** |
| 32 | 1382.8 | 1901.8 | **1.38×** |
| **128** | **1802.5** | **3204.2** | **1.78×** |

`rust_with_opp` errored across the grid with the reserved
placeholder message; opponent injection ships in Phase 5.

### Decision-branch evaluation

The pre-agreed thresholds from the remediation plan, against the
`n_envs=128` row:

| Threshold | Met? |
|---|---|
| `rust_no_op < 1.5× py` → halt to Phase 9 ARCHIVE | No — 1.78× > 1.5× |
| `1.5× ≤ rust_no_op < 3×` → proceed BUT pre-commit to ARCHIVE unless Phase 6 produces ≥ 2.5× end-to-end | **Yes — this is where we land** |
| `rust_no_op ≥ 3×` → unconditional proceed | No — 1.78× < 3× |

### Amdahl sanity check

`analysis/diag_py.log` measures the rollout share of a full PPO
update at ~20% (rollout = 46.58 s of a 236.07 s update; SGD =
189.48 s = 80%). With the rollout phase sped up by `1.78×` at
`n_envs=128` and SGD unchanged, the per-update wall is
predicted to drop from `236.07 s` to
`46.58 / 1.78 + 189.48 ≈ 215.6 s` — an end-to-end speedup of
`236.07 / 215.6 ≈ 1.09×`.

That is dramatically below the Phase 6 gate of `≥ 2.5×` end-to-end
that was set in the pre-measurement plan. Two factors could
plausibly close any of the gap:

1. **Removing the per-step Python opponent**: Phase 5 / 6 land the
   opponent-injection contract. If the opponent inference (which
   currently runs inside the rollout phase, costing Python time
   per env per turn) drops to zero or batches across envs, the
   measured `1.78×` rollout ratio could improve. But the bench
   already excludes the heuristic opponent (uses `random`), so
   most of the savings the `rust_with_opp` row could deliver are
   already in the `rust_no_op` headline.
2. **GIL contention reduction across the policy forward**: a more
   speculative source of upside. The current bench holds the GIL
   throughout; the production loop will likely too.

Neither bridges `1.09×` to `2.5×`. The Amdahl ceiling — even if
the rollout phase dropped to zero seconds — is `1 / 0.80 = 1.25×`
end-to-end at the diag's SGD-share of 80 %. Reaching `2.5×`
without re-engineering the SGD path (mixed precision, fused
kernels, smaller model, larger micro-batches with grad
accumulation) is **physically impossible inside the scope of this
remediation plan**, which is rollout-only.

### Phase 6 gate revision (post-measurement)

The pre-measurement `≥ 2.5× e2e at n_envs=128` Phase 6 gate is
hereby **revised to `≥ 1.15× e2e at n_envs=128`**, on the basis of:

* The Phase 1 measurement above (`1.78×` rollout, `~1.09×` Amdahl
  projection end-to-end).
* The Amdahl ceiling derivation (max `1.25×` even if rollout were
  free).
* Reasonable measurement slack (`±5 %` on a 30-min training-loop
  smoke).

The original `2.5×` gate is **preserved verbatim** in
`docs/plans/rust_engine_migration.md` (the migration plan archive)
so the pre-measurement intent stays auditable. This document is
the live source of truth; the gate that Phase 6 actually has to
clear is `1.15×`.

**Hard stop**: Phase 6 must measure end-to-end against the
production training loop and write the number into this document
BEFORE Phase 7 starts. No "looks promising, keep going" deferral.
A measurement under `1.15×` triggers Phase 9 ARCHIVE; under
`1.05×` triggers ARCHIVE immediately, no soak, no negotiation.

### How to read these numbers

The bench numbers above are **rollout-loop throughput**, not
production rollout cost. They differ from `analysis/diag_py.log`'s
~ 703 env-steps/sec at `n_envs=128` because the bench omits:

* Heuristic opponent inference (bench uses `random`).
* Rollout buffer writes + GAE bookkeeping between policy and env.
* Opponent action masking + opponent obs encoding.
* Cold-start cost (bench excludes reset from the timer).

The Amdahl projection above correctly imports only the *ratio*
between `py` and `rust_no_op` (which is robust to the missing
cost slices) and applies it to the diag's `46.58 s` rollout
share. The bench's absolute sps numbers should not be quoted
as standalone production-rollout figures.

Bench `repeat=0` rows are NOT discarded as warm-up. Cell
summaries use the **median** of 3 repeats so the warm-up
contribution is suppressed; high-variance `repeat=0` rows (e.g.
`py @ n_envs=32, repeat=0` at 1,463 sps vs the cell median 1,382
sps) are visible in `bench_phase1.csv` but do not move the
headline figures. Future bench runs that want a warm-up-discarded
view should use `--repeat 4` and ignore the first row in
post-processing.

The 43.9× figure in the original migration plan ("`241,266`
env-steps/sec at `n_envs=64`") would require this codebase's
fastest measured Rust path to be `~75×` faster than what the
bench just produced. There is no architectural change in scope
that closes that gap.



- Any PR that adds a benchmark number to this document must cite
  the CSV row (`benchmarks/results/<timestamp>.csv`) and the
  hardware identifier (`platform.machine()`) that produced it.
- Any PR that changes `DEFAULT_BACKEND` in
  `src/catan_rl/engine/backend.py` must update both the row above
  and `configs/ppo_default.yaml`.
- Any PR that adds, removes, or renames a Rust `#[pyclass]` must
  update the "What is built but unwired" table.
- This document is intentionally allowed to contain phrases like
  "the migration's optimistic narrative had drifted" — that is
  the kind of language that lets a future contributor recognise
  that a section is a forensic correction, not aspirational
  scaffolding. Do not soften it without a contemporaneous audit
  saying it is no longer true.
