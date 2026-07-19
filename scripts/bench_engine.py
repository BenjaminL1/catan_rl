#!/usr/bin/env python3
"""Engine throughput bench — Phase 1 of the Rust migration remediation plan.

Produces the only reproducible env-step throughput numbers in this
codebase. See ``benchmarks/README.md`` for what this measures and
why "policy in the loop" is non-negotiable (the architect review of
the remediation plan flagged a no-policy bench as misleading).

**Backends.** Three modes are exercised by a single CLI:

* ``py``        — real ``SerialVecEnv`` over ``CatanEnv`` with
                  ``opponent_type='random'``; one frozen
                  ``CatanPolicy.forward()`` per step.
* ``rust_no_op`` — ``catan_engine.RustVectorizedEnv.step_batch``
                  driven with a constant ``EndTurn`` action; one
                  frozen ``CatanPolicy.forward()`` per step. No
                  opponent — the Rust env has no opponent hook.
* ``rust_with_opp`` — reserved row; errors with a "Phase 5 pending"
                  message so the CSV schema is exercised end-to-end
                  from day one.

**Acceptance gate (Phase 1):**

1. ``py @ n_envs=8, n_steps=1024`` median sps lands in the
   ballpark of the historical ``analysis/diag_py.log`` per-env
   rate (≈ 5–10 sps per env with policy + opponent + obs encoding
   in the loop).
2. ``rust_no_op @ n_envs=128, n_steps=1024`` median wall < 30 s.
3. ``rust_with_opp`` errors with a clear "Phase 5 pending" message.

**Decision branch downstream** (per the agreed plan):

* ``rust_no_op`` median sps `< 1.5×` of ``py`` at ``n_envs=128``
  → halt the remediation plan and proceed to Phase 9 ARCHIVE.
* ``rust_no_op`` median sps `≥ 1.5×` but `< 3×` at ``n_envs=128``
  → proceed but pre-commit to ARCHIVE unless Phase 6 produces
  `≥ 2.5×` end-to-end.
* ``rust_no_op`` median sps `≥ 3×` at ``n_envs=128`` → unconditional
  proceed.

Usage::

    python scripts/bench_engine.py --backend py --n-envs 8 --n-steps 1024
    python scripts/bench_engine.py --all --n-steps 1024
    python scripts/bench_engine.py --backend rust_no_op --n-envs 128 --n-steps 1024 --repeat 3
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default n_envs grid for ``--all``. Matches the architect's
#: adjustment over the planner's grid (added n_envs=1 for the
#: single-env cost floor).
DEFAULT_N_ENVS_GRID: tuple[int, ...] = (1, 8, 32, 128)

#: Reserved EndTurn action for ``rust_no_op``. ``EndTurn`` (action
#: type 3) is the only universally-legal action in any
#: non-terminal Catan state under the Rust engine's state machine,
#: so a constant action vector drives the loop without needing the
#: per-env mask query that ``rust_no_op`` is specifically designed
#: to avoid.
END_TURN_ACTION = np.array([3, 0, 0, 0, 0, 0], dtype=np.uint8)

Backend = Literal["py", "rust_no_op", "rust_with_opp"]
BACKENDS: tuple[Backend, ...] = ("py", "rust_no_op", "rust_with_opp")


@dataclass(frozen=True)
class BenchRow:
    """One row in the CSV. Schema must stay stable across remediation
    plan phases — Phase 6 will append more rows of the same shape."""

    timestamp_utc: str
    git_sha: str
    hardware: str
    backend: str
    n_envs: int
    n_steps: int
    include_policy: bool
    env_steps_per_sec: float
    wall_s: float
    repeat_idx: int


# ---------------------------------------------------------------------------
# Manifest / output helpers
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        sha = out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        if dirty.stdout.strip():
            sha += "-dirty"
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass
    return sha


def _hardware_tag() -> str:
    """Short ``machine + system`` tag for the manifest."""
    return f"{platform.machine()}-{platform.system()}"


def _catan_engine_version() -> str:
    try:
        import catan_engine

        return str(catan_engine.version())
    except (ImportError, AttributeError):
        return "unavailable"


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _write_csv_row(csv_path: Path, row: BenchRow) -> None:
    """Append a row. Creates the file with header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(row))


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the JSON manifest. Overwrites — one manifest per run."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Frozen policy
# ---------------------------------------------------------------------------


def _build_frozen_policy() -> torch.nn.Module:
    """Build a CatanPolicy with default flags, freeze it for inference.

    Board geometry is left at the zero placeholder. The
    `set_board_geometry` call is NOT required for `forward()` to
    run — verified at Phase 1 review by exercising the full
    forward path against a zero-init policy with placeholder
    geometry; trunk (B, 512), value (B,), belief (B, 5) all emit
    cleanly. The GNN encoder + tile encoder still execute their
    matmuls against placeholder indices, which is the wall-time
    that matters for this bench. Production differs by a single
    per-process `set_board_geometry` call (microseconds) and
    different scalar adjacency values fed into the same matmuls.

    Returns the policy on CPU in `eval()` mode. Callers wrap the
    forward call in `torch.inference_mode()` for the extra few
    percent saved on autograd bookkeeping.
    """
    from catan_rl.policy import CatanPolicy

    policy = CatanPolicy()
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Obs converters
# ---------------------------------------------------------------------------


#: Known-scalar Rust obs keys that arrive as ``(1,)`` arrays and
#: must be squeezed to scalar before stacking into ``(N,)``. The
#: rest of the obs is batched along axis 0 directly. Narrower than
#: a blanket "all (1,) arrays are scalars" rule so a future Rust
#: change that adds a different ``(1,)`` field can't get silently
#: squeezed. (Architect-flagged 2026-06-06.)
_RUST_SCALAR_KEYS = frozenset({"opponent_kind", "opponent_policy_id"})


def _stack_rust_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack Rust's per-env obs dict list into a batched dict.

    Rust returns ``opponent_kind`` and ``opponent_policy_id`` as
    ``(1,)`` int64 arrays; ``SerialVecEnv`` produces them as
    ``(N,)`` int64 after its own stack. Squeeze the (1,) here so
    the batched output has matching shape — only for the known
    scalar keys; other (1,) shaped fields stack as-is.
    """
    if not obs_list:
        raise ValueError("empty obs_list")
    keys = obs_list[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        arrs = [d[k] for d in obs_list]
        if k in _RUST_SCALAR_KEYS and arrs[0].shape == (1,):
            # Squeeze (1,) -> scalar before stacking into (N,).
            out[k] = np.stack([a[0] for a in arrs])
        else:
            out[k] = np.stack(arrs)
    return out


def _reconcile_rust_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Zero-pad the legacy Rust obs up to the current Python schema.

    The Rust engine (scaffolding, not the default backend) still emits the
    pre-pointer-arch obs: a 54-dim ``current_player_main``, a 61-dim
    ``next_player_main`` and no ``global_features`` / ``is_setup``. The frozen
    policy now expects the appended-tail schema, so we synthesize the missing
    columns/keys as strict zeros — the same zero-pad an old-checkpoint migration
    applies to the fusion's new input columns. This keeps the throughput bench
    runnable against the Rust backend without touching the Rust crate.
    """
    from catan_rl.policy.obs_schema import (
        CURR_PLAYER_DIM,
        GLOBAL_DIM,
        NEXT_PLAYER_DIM,
    )

    out = dict(obs)

    def _pad_tail(key: str, target: int) -> None:
        arr = out.get(key)
        if arr is None or arr.shape[-1] >= target:
            return
        pad = np.zeros((*arr.shape[:-1], target - arr.shape[-1]), dtype=arr.dtype)
        out[key] = np.concatenate([arr, pad], axis=-1)

    _pad_tail("current_player_main", CURR_PLAYER_DIM)
    _pad_tail("next_player_main", NEXT_PLAYER_DIM)

    n = out["current_player_main"].shape[0]
    if "global_features" not in out:
        out["global_features"] = np.zeros((n, GLOBAL_DIM), dtype=np.float32)
    if "is_setup" not in out:
        out["is_setup"] = np.zeros((n, 1), dtype=np.float32)
    return out


def _obs_to_torch(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Move a batched obs dict onto CPU torch tensors. The frozen
    policy lives on CPU; cross-device tensor copies are out of scope
    for Phase 1."""
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Backend drivers
# ---------------------------------------------------------------------------


def _run_py_backend(
    *, n_envs: int, n_steps: int, seed: int, policy: torch.nn.Module, include_policy: bool
) -> float:
    """Drive ``SerialVecEnv`` for ``n_steps`` with a constant
    ``EndTurn`` action per env per step.

    Returns wall-time in seconds for the rollout loop only (reset
    is excluded so the bench measures steady-state throughput, not
    cold-start cost — same convention as ``rust_no_op``).
    """
    from catan_rl.ppo.vec_env import SerialVecEnv

    env_kwargs_list = [{"opponent_type": "random"} for _ in range(n_envs)]
    vec_env = SerialVecEnv(env_kwargs_list=env_kwargs_list, seed=seed)
    try:
        seeds = [seed + i for i in range(n_envs)]
        obs, _masks = vec_env.reset_all(seeds=seeds)
        actions = np.tile(END_TURN_ACTION.astype(np.int64), (n_envs, 1))

        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_steps):
                if include_policy:
                    obs_t = _obs_to_torch(obs)
                    _ = policy(obs_t)
                obs, _masks, _r, _term, _trunc, _final, _final_info = vec_env.step_all(actions)
        return time.perf_counter() - t0
    finally:
        vec_env.close()


def _run_rust_no_op_backend(
    *, n_envs: int, n_steps: int, seed: int, policy: torch.nn.Module, include_policy: bool
) -> float:
    """Drive ``catan_engine.RustVectorizedEnv.step_batch`` for
    ``n_steps`` with a constant ``EndTurn`` action per env per step.

    No opponent. No action masks (the Rust env has no batched mask
    API; per-env ``get_action_masks`` would defeat the no-op
    premise). Wall-time excludes the reset / construction step.
    """
    import catan_engine

    vec_env = catan_engine.RustVectorizedEnv(n_envs, seed)
    actions = np.tile(END_TURN_ACTION, (n_envs, 1))

    obs_list, _r, _t, _tr = vec_env.step_batch(actions)
    obs = _stack_rust_obs(list(obs_list))

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_steps):
            if include_policy:
                obs_t = _obs_to_torch(_reconcile_rust_obs(obs))
                _ = policy(obs_t)
            obs_list, _r, _t, _tr = vec_env.step_batch(actions)
            obs = _stack_rust_obs(list(obs_list))
    return time.perf_counter() - t0


def _run_rust_with_opp_backend(
    *,
    n_envs: int,
    n_steps: int,
    seed: int,
    policy: torch.nn.Module,
    include_policy: bool,
) -> float:
    """Reserved row. Errors with a clear pointer at the phase that
    lands the opponent injection contract (Phase 5 of the
    remediation plan)."""
    raise NotImplementedError(
        "backend 'rust_with_opp' is reserved; opponent injection ships in "
        "Phase 5 of the Rust migration remediation plan. The CSV row is "
        "kept available so the schema is stable from day one. See "
        "docs/plans/rust_engine_actual_state.md."
    )


_BACKEND_DRIVERS = {
    "py": _run_py_backend,
    "rust_no_op": _run_rust_no_op_backend,
    "rust_with_opp": _run_rust_with_opp_backend,
}


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def _run_one_cell(
    *,
    backend: Backend,
    n_envs: int,
    n_steps: int,
    repeat: int,
    seed: int,
    include_policy: bool,
    policy: torch.nn.Module,
    csv_path: Path,
    git_sha: str,
    hardware: str,
) -> dict[str, float] | dict[str, str]:
    """Run ``repeat`` measurements of one (backend, n_envs) cell and
    append a row per repeat to the CSV. Returns the median sps or
    an error dict if the backend errored."""
    driver = _BACKEND_DRIVERS[backend]
    sps_samples: list[float] = []
    wall_samples: list[float] = []
    for repeat_idx in range(repeat):
        try:
            wall_s = driver(
                n_envs=n_envs,
                n_steps=n_steps,
                seed=seed + repeat_idx,
                policy=policy,
                include_policy=include_policy,
            )
        except NotImplementedError as e:
            # Reserved-row backend. Single-shot, no retries.
            return {"error": str(e)}
        sps = (n_envs * n_steps) / wall_s if wall_s > 0 else float("inf")
        sps_samples.append(sps)
        wall_samples.append(wall_s)
        row = BenchRow(
            timestamp_utc=_utc_stamp(),
            git_sha=git_sha,
            hardware=hardware,
            backend=backend,
            n_envs=n_envs,
            n_steps=n_steps,
            include_policy=include_policy,
            env_steps_per_sec=sps,
            wall_s=wall_s,
            repeat_idx=repeat_idx,
        )
        _write_csv_row(csv_path, row)
        print(
            f"  {backend:>14s}  n_envs={n_envs:>3d}  n_steps={n_steps}  "
            f"rep={repeat_idx}  wall={wall_s:7.3f}s  sps={sps:>9.1f}"
        )
    return {
        "median_sps": statistics.median(sps_samples),
        "min_sps": min(sps_samples),
        "max_sps": max(sps_samples),
        "median_wall_s": statistics.median(wall_samples),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_engine.py",
        description=(
            "Engine throughput bench with frozen CatanPolicy.forward() in "
            "the loop. Phase 1 of the Rust migration remediation plan."
        ),
    )
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="One of {py, rust_no_op, rust_with_opp}. Mutually exclusive with --all.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Sweep over the standard n_envs grid (1, 8, 32, 128) for "
            "every backend. Equivalent to running this script once per "
            "(backend, n_envs) pair."
        ),
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override n_envs. Ignored when --all is set.",
    )
    p.add_argument("--n-steps", type=int, default=1024, help="Steps per repeat (default 1024).")
    p.add_argument("--repeat", type=int, default=3, help="Repeats per cell (default 3).")
    p.add_argument("--seed", type=int, default=42, help="Base seed (default 42).")
    p.add_argument(
        "--exclude-policy",
        action="store_true",
        help=(
            "Skip the per-step policy.forward() inside the timing loop. "
            "Useful for env-only sanity checks; the CSV's "
            "include_policy column captures this so the row is not "
            "comparable with the policy-in-loop rows. The plan's "
            "decision gates use policy-in-loop numbers."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="Where to write the CSV + JSON manifest.",
    )
    p.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help=(
            "Override the CSV basename. By default a timestamped name "
            "is used so each invocation gets its own file."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.all and args.backend is None:
        parser.error("either --backend or --all must be supplied")
    if not args.all and args.n_envs is None:
        parser.error("--n-envs is required unless --all is set")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    git_sha = _git_sha(REPO_ROOT)
    hardware = _hardware_tag()
    csv_basename = args.csv_name or f"bench_{_utc_stamp()}.csv"
    csv_path = args.out_dir / csv_basename
    manifest_path = csv_path.with_suffix(".json")
    include_policy = not args.exclude_policy

    print(
        f"bench: git={git_sha}  hw={hardware}  catan_engine={_catan_engine_version()}  "
        f"include_policy={include_policy}  out={csv_path}"
    )

    policy = _build_frozen_policy()

    if args.all:
        backend_list: list[Backend] = list(BACKENDS)
        n_envs_grid: tuple[int, ...] = DEFAULT_N_ENVS_GRID
    else:
        backend_list = [args.backend]  # type: ignore[list-item]
        n_envs_grid = (args.n_envs,)

    cell_results: dict[str, dict[str, Any]] = {}
    for backend in backend_list:
        for n_envs in n_envs_grid:
            cell_key = f"{backend}@n_envs={n_envs}"
            print(f"\n--- {cell_key} ---")
            result = _run_one_cell(
                backend=backend,
                n_envs=n_envs,
                n_steps=args.n_steps,
                repeat=args.repeat,
                seed=args.seed,
                include_policy=include_policy,
                policy=policy,
                csv_path=csv_path,
                git_sha=git_sha,
                hardware=hardware,
            )
            cell_results[cell_key] = result

    manifest = {
        "timestamp_utc": _utc_stamp(),
        "git_sha": git_sha,
        "hardware": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "catan_engine_version": _catan_engine_version(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "invocation": ["bench_engine.py", *sys.argv[1:]],
        "include_policy": include_policy,
        "n_steps": args.n_steps,
        "repeat": args.repeat,
        "seed": args.seed,
        "cells": cell_results,
    }
    _write_manifest(manifest_path, manifest)
    print(f"\nmanifest -> {manifest_path}")
    print(f"csv      -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
