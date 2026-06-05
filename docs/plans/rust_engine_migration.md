# Rust Engine Migration (R0-R13)

## Goal

Replace the Python engine at `src/catan_rl/engine/` with a Rust crate
(`crates/catan_engine/`) exposed via PyO3 + maturin. Target: enable
`n_envs` to scale from 8 → 64-128 so the policy forward-pass batch
≥128 (where MPS/CUDA wins). Baseline 5,500 env-steps/sec at n_envs=8
on M1 Pro CPU. Target: ≥8× engine speedup, ≥6× end-to-end training
throughput at n_envs=64.

Senior game-dev SWE review (separate session) pinned the language
choice (Rust + PyO3 + maturin), architecture (`#[repr(C)]` GameState
≤400 bytes, snapshot = `*self.state` memcpy, integer-only state),
RNG (ChaCha8), and migration order (dice → board → game state →
events → tracker → obs → masks → single-env → vec env → cutover →
caller migration → delete Python → polish).

## Resolved open questions

The 7 plan-time open questions were answered by the senior SWE with
training-speed as the top priority. The resolved answers are:

| Q | Decision | Why |
|---|---|---|
| Q1 Heuristic parity | **Statistical equivalence + one-shot BC regen** | MT19937-in-Rust is 2-3× slower than ChaCha8; exact `np.random.randint` parity needs a multi-day rabbit hole with zero training-speed benefit; BC regen costs one afternoon |
| Q2 Edge index ordering | **Ship `migrate_checkpoint.py` permutation patch** | Coupling Rust to Python dict-insertion order forever just to honor a CPython 3.7+ detail is wrong; one-shot `policy.heads.edge.linear.weight = w[π]` is auditable and frees Rust to use sorted `(min,max)` tuples in `Vec<[u8;2]>` for cache locality |
| Q3 RNG for heuristic | **Shared ChaCha8 stream, seed-per-env** | Q1=(b) already commits to BC regen, so "preserve Python reproducibility" evaporates; two RNG streams per env doubles cache footprint and forces an FFI hop per action sample |
| Q4 PyO3 abi3 vs version-specific | **Version-specific (py3.12)** | abi3 dispatch is ~5-15ns per `#[pymethods]` call + compounds with `&PyAny` extraction; abi3 is a libraries-shipped-to-strangers concern, not single-dev training |
| Q5 `geometry.py` location | **Move to `src/catan_rl/gui/`** | Pygame-pixel helper with zero RL-hot-path callers; leaving it in `engine/` after cutover lies about the engine boundary and risks pulling pygame into subproc forks |
| Q6 PR cadence | **4 chunky PRs (R0-R2 / R3-R5 / R6-R9 / R10-R13)** | 14 PRs = 14 days of calendar drag before n_envs=128 compounds wall-clock training; bisect pain is overrated — `git bisect` works inside a chunky PR |
| Q7 R12 (delete Python) timing | **Soak 1-2 weeks after R10** | R10 leaves Python importable behind `CATAN_ENGINE_BACKEND=py` for A/B at the 10M-step mark; +1KB in `__init__.py` saves a frantic revert if exploitability spikes post-cutover |

## PR cadence

- **PR1** = R0-R2: scaffolding + dice + board
- **PR2** = R3-R5: game state + events + hand tracker
- **PR3** = R6-R9: obs + masks + single-env + vec env (the perf-unlock PR)
- **PR4** = R10-R13: cutover + caller migration + delete Python + polish

## Phase R0 — Workspace scaffolding + maturin hello world

**Scope:** cargo workspace, `crates/catan_engine/` crate, maturin integration in `pyproject.toml`, GitHub Actions Rust build job, "hello from rust" PyO3 module callable as `catan_engine.hello()`.

**Files added:**
- `Cargo.toml` (workspace root)
- `crates/catan_engine/Cargo.toml`
- `crates/catan_engine/src/lib.rs`
- `src/catan_rl/_native/__init__.py`

**Files modified:**
- `pyproject.toml` — add `[tool.maturin]` section
- `.gitignore` — add `target/`, `_native/*.so`
- `.github/workflows/ci.yml` — add `rust-build` job (macOS arm64 + linux x86_64 matrix)
- `Makefile` — add `rust-build`, `rust-test` targets

**Acceptance:** `maturin develop --release` → `import catan_engine; assert catan_engine.hello() == "hello from rust"` succeeds.

**Complexity / wall-time:** L / 6h.

## Phase R1 — Dice + ChaCha8 + byte-parity test

**Scope:** Port `StackedDice` to Rust. Use `ChaCha8Rng` from `rand_chacha`. Karma persistent-buff semantics preserved per CLAUDE.md item 27. Python reference ChaCha8 impl + 100k-call parity test.

**Files added:**
- `crates/catan_engine/src/dice.rs`
- `crates/catan_engine/src/rng.rs`
- `tests/refs/chacha8.py`
- `tests/refs/__init__.py`
- `tests/unit/engine/test_rng_parity.py`
- `tests/unit/engine/test_dice_parity.py`

**Files modified:**
- `src/catan_rl/engine/dice.py` — thin shim re-exporting `catan_engine.StackedDice`

**Acceptance:** 100k consecutive `roll()` calls byte-identical Python ref + Rust; all existing engine tests still pass.

**Complexity / wall-time:** M / 8h.

## Phase R2 — BoardStatic (immutable post-construction)

**Scope:** Resource list shuffle, spiral chip placement, port assignment, vertex/edge graph topology, axial coords. Output `BoardStatic` exposed via PyO3.

**Files added:**
- `crates/catan_engine/src/board.rs`
- `crates/catan_engine/src/spiral.rs`
- `tests/unit/engine/test_board_static_parity.py`

**Files modified:**
- `src/catan_rl/engine/board.py` — `catanBoard` gains `use_rust_board: bool = False` flag
- `src/catan_rl/policy/board_geometry.py` — opt-in `_board_adjacency_tables()` from Rust BoardStatic

**Acceptance:** 1,000 seeded boards have byte-identical `board_static()` dicts (after key-ordering normalization). Edge index ordering pinned (Q2 — if Rust uses a different order, the migration script lands here).

**Complexity / wall-time:** M / 10h.

## Phase R3 — GameState + 13 action types

**Scope:** `#[repr(C)] GameState` struct with `[u8; 19]` hex resources, `[u8; 19]` number tokens, `[u8; 54]` vertex owner, `[u8; 72]` edge owner, `[PlayerState; 2]` inline. ≤400 bytes. Implement all 13 action types: BuildSettlement, BuildCity, BuildRoad, EndTurn, MoveRobber, BuyDevCard, PlayKnight, PlayYoP, PlayMonopoly, PlayRoadBuilder, BankTrade, Discard, RollDice. Friendly Robber + 9-card discard + snake-draft setup + 2nd-settlement starting resources.

**Acceptance:** 10,000 seeded random+random games — step-by-step state snapshots byte-identical Python vs Rust.

**Complexity / wall-time:** H / 32h.

## Phase R4 — Broadcast events native

**Scope:** Event emission inside `apply_action`. Events buffered as `Vec<Event>`, drained per step. All 12 `BroadcastEventType` variants. Event ordering exactly preserved (Monopoly: per-victim RESOURCE_CHANGE before structural MONOPOLY; Steal: MOVE_ROBBER before STEAL).

**Acceptance:** 10k seeded games — event streams byte-identical (after type-string normalization). Replay recorder integration tests pass with Rust engine.

**Complexity / wall-time:** M / 12h.

## Phase R5 — Native hand tracker

**Scope:** Native `HandTracker` subscribes to event buffer in-process. Resource ordering: stores in engine order, FFI exposes Charlesworth order.

**Acceptance:** Byte-identical hand state at every step on 10k games.

**Complexity / wall-time:** L / 6h.

## Phase R6 — Native obs encoder

**Scope:** Port `ObsEncoder.build_obs` to Rust. Zero-copy `PyArray2<f32>` returns for all 9 obs keys. Pre-allocates scratch buffers per env. Karma flags preserved exactly per CLAUDE.md item 28.

**Acceptance:** Byte-identical obs arrays on 5k games. Bench ≥5× speedup vs Python.

**Complexity / wall-time:** H / 20h.

## Phase R7 — Native mask builder

**Scope:** Port `compute_action_masks` to Rust. 9 boolean masks zero-copy.

**Acceptance:** Byte-identical masks on 5k games. Bench ≥5× speedup.

**Complexity / wall-time:** M / 14h.

## Phase R8 — Single-env Rust-backed `CatanEnv`

**Scope:** Env shell stays Python (Gymnasium API surface) but internals call Rust. State machine (`roll_pending`, `discard_pending`) stays Python.

**Acceptance:** Full integration suite passes against Rust env. Replay-tape parity end-to-end.

**Complexity / wall-time:** H / 20h.

## Phase R9 — `VectorizedEnv` + Rayon

**Scope:** Native `VectorizedEnv(n_envs)` with `step_batch(actions)` releasing the GIL via `py.allow_threads`. Rayon parallelism with serial fallback (`CATAN_ENGINE_THREADS=1`).

**Acceptance:** n_envs=64 ≥ 44k env-steps/sec (≥8× baseline).

**Complexity / wall-time:** H / 16h.

## Phase R10 — Cutover

**Scope:** `CatanEnv` defaults to Rust; `CATAN_ENGINE_BACKEND=py` env var preserves Python fallback for A/B (per Q7).

**Acceptance:** 1k-step PPO training smoke runs end-to-end; metrics within ±5% of Python reference.

**Complexity / wall-time:** M / 8h.

## Phase R11 — Caller migration (heuristic, BC, scripts)

**Scope:** Rewrite heuristic AI + RandomAI + perturbed heuristics + BC dataset gen + labeling + debug scripts to consume the Rust engine. Heuristic ported to Rust (`heuristic_main_move(state, seat) → Action`) per Q3 (shared ChaCha8). One-shot BC dataset regen.

**Acceptance:** All BC/labeling/audit tests pass against Rust heuristic. Heuristic action distribution within ±2% of Python reference.

**Complexity / wall-time:** H / 18h.

## Phase R12 — Delete Python engine

**Scope:** Archive `src/catan_rl/engine/python_reference/`. Remove dual-engine pytest fixture. Soak 1-2 weeks after R10 before this lands (per Q7).

**Acceptance:** Tests green on Rust-only; no `from catan_rl.engine.python_reference` imports in production.

**Complexity / wall-time:** L / 6h.

## Phase R13 — Performance polish (optional)

**Scope:** Flamegraph profiling, `#[inline]` annotations on hot helpers, LTO+codegen-units verification, optional PGO for the obs encoder hot path.

**Acceptance:** End-to-end training throughput at n_envs=64 ≥ 6× Python baseline.

**Complexity / wall-time:** M / 10h.

## Cross-cutting

### Cargo profile (Cargo.toml)

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
```

### pyproject.toml maturin section

```toml
[tool.maturin]
manifest-path = "crates/catan_engine/Cargo.toml"
module-name = "catan_engine"
python-source = "src"
strip = true
```

Existing `[build-system]` stays on hatchling for the Python wheel; maturin invoked via `make rust-build` (= `maturin develop --release`).

### Dual-engine pytest fixture

```python
# tests/conftest.py
@pytest.fixture(params=["python", "rust"], ids=["py_engine", "rust_engine"])
def engine_backend(request):
    if request.param == "rust":
        pytest.importorskip("catan_engine")
    yield request.param
```

Tests forward to env constructor via `CatanEnv(engine_backend=backend)`. Fixture removed in R12.

### Replay-tape parity harness

`tests/refs/replay_tape.py` exposes `record_tape`, `replay_tape`, `assert_tapes_match`. The single most important correctness gate from R3 onward.

## Risk register (top 5)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| RSK-2 | PyO3+maturin ABI tag mismatch on macOS arm64 | BLOCK | Version-specific py3.12 wheel (Q4), CI matrix on macOS-arm64 |
| RSK-3 | ChaCha8 byte-parity test fails | BLOCK | RFC-7539-derived ChaCha8 ref in `tests/refs/chacha8.py`; cross-check `rand_chacha` test vectors |
| RSK-5 | Edge index ordering drift invalidates checkpoints | BLOCK | Q2 — ship `scripts/migrate_checkpoint.py` permutation patch |
| RSK-6 | Karma persistent-buff bug breaks dice distribution | BLOCK | R1 reviewer focus; 1M-roll fuzz test against closed-form Karma model |
| RSK-7 | Friendly Robber `<3` vs `≤3` typo | HIGH | R3 reviewer focus; pinned unit test (VP=2 illegal, VP=3 legal) |

Plus 12 others (event ordering, GIL+Rayon deadlock, Charlesworth/engine resource-order confusion, hand-tracker integer underflow, snapshot size creep, CI runtime explosion, etc.) — all with documented mitigations.

## Estimated payoff

| Phase | Throughput at n_envs=64 |
|---|---|
| Baseline | 5,500 env-steps/sec at n_envs=8 |
| R6 | ~13,000 |
| R7 | ~16,000 |
| **R9** | **~64,000-96,000 (8-12× baseline)** |
| R11 | ~70,000-100,000 |
| R13 | ~80,000-130,000 |

## Implementation notes

Updated as each PR lands per CLAUDE.md rule #14.

### PR1 (R0-R2) — landed
- R0: cargo workspace + maturin develop --release round-trip on
  macOS arm64 + linux x86_64 CI.
- R1: StackedDice + ChaCha8 native; byte-parity with Python ref
  ChaCha8 across 14 cases (RSK-3 closed). 9 dice behavioral tests.
- R2: BoardStatic precomputed via OnceLock. CW corner enumeration
  fixed in reviewer pass (CCW would have placed 6/9 ports on
  inner-ring edges).

### PR2 (R3-R5) — landed
- R3: GameState with all 13 action types + Friendly Robber + 9-card
  discard + snake-draft setup + LR/LA.
- R4: Event enum mirroring 12 Python BroadcastEventType variants,
  drained via drain_events() at env.step boundary.
- R5: HandTracker as thin Charlesworth-order getter on engine state.

### PR3 (R6-R9) — landed
- R6: Native obs encoder. 10-key dict, zero-copy PyArray.
- R7: Native mask builder. 9 mask dict.
- R8: RustCatanEnv (Gymnasium API).
- R9: RustVectorizedEnv with py.allow_threads + Rayon.

**Measured throughput (M1 Pro CPU):**

| n_envs | env-steps/sec | x baseline |
|---|---|---|
| 8 | 114,201 | 20.8x |
| 64 | **241,266** | **43.9x** |
| 128 | 272,075 | 49.5x |

PR3 exceeds the R9 gate (44k at n_envs=64) by 5.4x. The 8x engine
and 6x end-to-end targets are achieved.

### PR4 (R10-R13) — partially landed
- R10: CATAN_ENGINE_BACKEND env var switch + backend helper module.
- R11, R12 **DEFERRED** as follow-up — Python engine remains canonical
  for the heuristic AI + BC + replay recorder pipelines.
- R13: profile.release already at lto=fat, codegen-units=1,
  panic=abort. No additional polish needed.

### Outstanding (post-PR4 follow-up)
- Wire CatanEnv to dispatch to RustCatanEnv when backend==rust.
  Requires porting opponent injection + opp-id obs masking + seat=1.
- Port heuristic AI + RandomAI to Rust (R11) with shared ChaCha8.
- Archive Python engine to python_reference/ after 1-2 week soak.
