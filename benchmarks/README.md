# Engine throughput benchmarks

This directory holds the harness for measuring the Catan engine's
throughput under conditions that approximate a real PPO training
loop. It exists because the headline "43.9× engine speedup" claim
in the migration plan was unfalsifiable until Phase 1 of the
remediation plan landed — there were no scripts that produced the
numbers. See
[`../docs/plans/rust_engine_actual_state.md`](../docs/plans/rust_engine_actual_state.md).

## Why a "policy in the loop" bench

A bench that measures only env-step throughput will report numbers
that vanish the moment a real PyTorch policy is plugged in — the
v2 audit explicitly documents that env stepping is NOT the rollout
bottleneck on this codebase (see
`memory/project_training_bottleneck_measured.md`). The architect
review of the remediation plan called this out as the single most
important adjustment: every backend mode in this harness must run
a frozen `CatanPolicy.forward()` (CPU, `batch=n_envs`, `no_grad`)
inside the timing loop, so the resulting CSV reflects production
conditions, not a misleading microbench.

## Backends

| Mode | What it times |
|---|---|
| `py` | Real `SerialVecEnv` over `CatanEnv` with `opponent_type='random'` (cheapest opponent so the harness isolates engine cost from heuristic-AI cost). One frozen policy forward per step. |
| `rust_no_op` | `catan_engine.RustVectorizedEnv.step_batch` driven with a constant `EndTurn` action (action type 3 — always legal, advances the state machine). One frozen policy forward per step. No opponent — the Rust env has no opponent hook. |
| `rust_with_opp` | Reserved row. Errors until Phase 5 of the remediation plan lands the opponent-injection contract; the schema is stable from day one so the CSV grows monotonically. |

The policy is `CatanPolicy()` with default architecture flags
(~1.4M params, matches production) on CPU. Board geometry is left
at the zero placeholder because we measure wall-time, not policy
quality — the geometry only affects what the GNN encoder computes,
not how fast it computes it.

## How to run

```bash
make bench                                            # all backends, all n_envs, default n_steps
python scripts/bench_engine.py --backend py --n-envs 8 --n-steps 1024 --repeat 3
python scripts/bench_engine.py --backend rust_no_op --n-envs 128 --n-steps 1024 --repeat 3
python scripts/bench_engine.py --all --n-steps 1024   # exercises every (backend, n_envs) cell
```

## Output

Every run appends a row to `benchmarks/results/bench_<UTC>.csv` and
writes a sibling `bench_<UTC>.json` with the run manifest:

* git SHA, dirty flag, branch
* `platform.machine()`, `platform.system()`, `platform.release()`
* `catan_engine.version()` and `torch.__version__`
* CLI invocation
* per-row median / min / max env-steps/sec across `--repeat`

CSV columns:

```
timestamp_utc, git_sha, hardware, backend, n_envs, n_steps,
include_policy, env_steps_per_sec, wall_s, repeat_idx
```

Numbers in this directory are the ONLY benchmark figures any
remediation-plan phase is allowed to quote. Adding a number to
`docs/plans/rust_engine_actual_state.md` or any other doc without
a CSV-row citation is forbidden by the guard-rail at the bottom of
that document.

## Phase 1 acceptance gate (recapped here so the numbers can be
graded in-context)

The harness must produce, on this codebase's M1 Pro:

1. A `py` row at `n_envs=8, n_steps=1024` whose median sps lands
   in a reasonable range vs the historical `analysis/diag_py.log`
   anchor — the diag rollout was 32,768 env-steps in 46.58 s
   (≈ 703 sps total / ≈ 5.5 sps per env) WITH heuristic opponent
   and policy forward. With random opponent and policy forward we
   expect the same ballpark per env.
2. A `rust_no_op` row at `n_envs=128, n_steps=1024` that completes
   in under 30 s (i.e. the Rust env doesn't deadlock or crawl).
3. A `rust_with_opp` row that errors out with a clear "Phase 5
   pending" message so the CSV schema is exercised end-to-end.

Decision branch downstream of this phase (per the agreed plan):

* `rust_no_op` median sps with policy forward `< 1.5×` of `py` at
  `n_envs=128` → halt the remediation plan and proceed to Phase 9
  ARCHIVE.
* `rust_no_op` median sps `≥ 1.5×` but `< 3×` at `n_envs=128` →
  proceed but pre-commit to ARCHIVE unless Phase 6 produces
  `≥ 2.5×` end-to-end.
* `rust_no_op` median sps `≥ 3×` at `n_envs=128` → unconditional
  proceed.
