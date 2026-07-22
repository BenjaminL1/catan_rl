"""PPO training entry point (Phase 2 of the v2 training infra build-out).

This script is the only intended way to launch a PPO training run. It:

  1. Resolves the configuration via the documented override chain:
     dataclass defaults → ``--config <yaml>`` → ``CATAN_PPO__*`` env vars
     → CLI flag overrides (highest precedence).
  2. Validates the device target and seeds all three global PRNGs:
     ``numpy``, ``torch``, and stdlib ``random`` (stdlib seeding matters
     because :mod:`catan_rl.engine.dice` still draws from it; see commit
     ``f31927d``).
  3. Creates the run output directory and writes a verbatim YAML snapshot
     of the resolved config so the run can be reproduced bit-identically
     from ``runs/train/<run_name>/config.yaml``.
  4. (Future, Phase 3-10) Wires up the policy, vec env, rollout buffer,
     and trainer, then calls ``trainer.learn()``.

For now, Phase 2 ships the entry point's outer scaffolding only. The
trainer construction step raises ``NotImplementedError("Phase 4
pending")`` with a clear pointer to where the wiring will happen.

Examples::

  # Default 50M-step run on M1 with audit defaults
  PYTHONPATH=src python scripts/train.py --config configs/ppo_default.yaml

  # Short smoke run to validate config + run-dir + seeding without
  # actually starting training
  PYTHONPATH=src python scripts/train.py --dry-run

  # Override total_steps via CLI flag (highest precedence)
  PYTHONPATH=src python scripts/train.py --total-steps 32768 --dry-run

  # Layer env-var overrides on top of YAML
  CATAN_PPO__ROLLOUT__N_ENVS=64 PYTHONPATH=src python scripts/train.py \
      --config configs/ppo_default.yaml --dry-run

  # Resumable long run: a STABLE --run-dir (no timestamp) so a mid-run death
  # (thermal / OOM / reboot) can be relaunched and RESUME (league pool +
  # optimizer + plateau clock + reanchor promotions restored) instead of
  # re-warm-starting from the seed into a new dir. Relaunch with the same
  # --run-dir; add --resume to fail fast if there is nothing to resume.
  PYTHONPATH=src python scripts/train.py --config configs/selfplay_pointer_arch.yaml \
      --run-dir runs/train/selfplay_pointer_arch
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# REPO_ROOT is used as the subprocess cwd around line ~273. Computed
# from this file's location: ``src/catan_rl/cli/train.py``, so
# ``parents[3]`` = repo root. The previous ``sys.path.insert`` shim
# was dropped in the maturin sole-backend cutover — ``catan_rl`` is
# importable from the install path now.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Local imports.
from catan_rl.ppo.arguments import (
    TrainConfig,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Kept as a free function so tests can introspect it without invoking
    ``main``.
    """
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Launch a v2 PPO training run.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (loaded on top of dataclass defaults). "
        "Falls back to dataclass defaults if omitted.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the run name (sub-directory under output_dir).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory root.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use this EXACT directory as the run dir (STABLE — no timestamp "
        "suffix), instead of the default '<output_dir>/<run_name>_<timestamp>'. "
        "Enables crash recovery for long runs: relaunching with the same "
        "--run-dir reuses the directory so the resume path picks up "
        "'checkpoints/' (restoring policy + optimizer + league + RNG + "
        "plateau clock). First launch into an empty dir starts fresh "
        "(warm-start); a later relaunch resumes automatically. Required for a "
        "launchd/KeepAlive-supervised run to resume rather than restart from "
        "the seed.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Guard flag for a deliberate resume: requires --run-dir and fails "
        "fast unless that directory already holds a resumable checkpoint. "
        "Without it, --run-dir silently starts fresh when empty (the intended "
        "first-launch behaviour); pass --resume when a relaunch MUST continue "
        "an existing run so a typo'd path can't silently restart from the seed.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the RNG seed.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Override the device (auto / cpu / mps / cuda).",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total_steps (must remain a multiple of rollout.n_envs * rollout.n_steps).",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override rollout.n_envs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve + validate the config and write the snapshot, but "
        "exit before constructing the trainer. Smoke-tests the wire-up "
        "without actually training.",
    )
    p.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Hard cap on the number of PPO updates to execute (independent "
        "of total_steps). Used by the smoke tests; production runs leave "
        "this unset so the loop runs until total_steps is exhausted.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def _normalize_path(p: Path | None) -> Path | None:
    """Apply tilde expansion + absolute resolution to a CLI-supplied path.

    Cloud-rental wrappers (Modal, SkyPilot) routinely launch with a
    ``WORKDIR`` that differs from the operator's invocation cwd, so an
    unexpanded ``~`` or relative ``configs/foo.yaml`` silently writes
    into the wrong place. Always normalise.
    """
    if p is None:
        return None
    return p.expanduser().resolve()


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    """Resolve the final config from defaults + YAML + env + CLI overrides.

    CLI flags have higher precedence than env vars and YAML, so they're
    applied last via ``dataclasses.replace``. Paths passed via CLI are
    normalised with ``expanduser().resolve()`` so a literal ``~`` never
    ends up as a directory name (real cloud-rental foot-gun caught in
    Phase 2 review).
    """
    # Normalise CLI paths up-front so ``~/foo`` and ``./bar`` both expand.
    config_path = _normalize_path(args.config)
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(
            f"--config path does not exist: {config_path} (was {args.config!r} before resolution)"
        )
    output_dir = _normalize_path(args.output_dir)

    base = TrainConfig.load(yaml_path=config_path)

    # CLI override stack: only apply fields the user explicitly passed.
    cli_top_level: dict[str, Any] = {}
    if args.run_name is not None:
        cli_top_level["run_name"] = args.run_name
    if output_dir is not None:
        cli_top_level["output_dir"] = str(output_dir)
    if args.seed is not None:
        cli_top_level["seed"] = args.seed
    if args.device is not None:
        cli_top_level["device"] = args.device
    if args.total_steps is not None:
        cli_top_level["total_steps"] = args.total_steps

    # Rollout-section overrides.
    cli_rollout: dict[str, Any] = {}
    if args.n_envs is not None:
        cli_rollout["n_envs"] = args.n_envs

    if cli_rollout:
        base = replace(base, rollout=replace(base.rollout, **cli_rollout))
    if cli_top_level:
        base = replace(base, **cli_top_level)

    return base


def setup_seeding(seed: int) -> None:
    """Seed numpy, torch, and stdlib random.

    All three matter for reproducibility:

    * ``numpy``: env / encoder / sampling.
    * ``torch``: policy weights at init + sampling at rollout.
    * stdlib ``random``: :mod:`catan_rl.engine.dice` (``StackedDice``).
      Missing this seed is the root of the v1 reproducibility incident
      fixed in commit ``f31927d``.
    """
    import torch  # local import — keeps script-import cheap

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS / CUDA also seeded by torch.manual_seed under the hood, but the
    # explicit calls below are defensive against future torch changes
    # (and free). The ``hasattr`` guard handles older torch builds (< 1.12)
    # where ``torch.mps`` may be absent even though the build runs.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def build_run_directory(cfg: TrainConfig, *, now: float | None = None) -> Path:
    """Create ``<output_dir>/<run_name>_<YYYYmmdd_HHMMSS>[_N]/`` and return it.

    The timestamp suffix exists so the same ``run_name`` can be launched
    repeatedly without colliding. Two launches at the same wall-clock
    second (CI fan-out, Modal parallel cells, clock-skewed cloud hosts)
    would otherwise merge into the same directory and corrupt TB +
    checkpoints. To prevent that we create the directory with
    ``exist_ok=False``; on collision we append ``_2``, ``_3``, ... until
    we find a free name.
    """
    when = time.localtime(now) if now is not None else time.localtime()
    stamp = time.strftime("%Y%m%d_%H%M%S", when)
    base = Path(cfg.output_dir) / f"{cfg.run_name}_{stamp}"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base.with_name(f"{base.name}_{suffix}")
            if suffix > 999:
                raise RuntimeError(
                    f"could not allocate a unique run dir at {base} "
                    f"after 999 collision retries — wall clock or filesystem bug?"
                ) from None


def _run_dir_has_resumable_checkpoint(run_dir: Path) -> bool:
    """True iff ``run_dir/checkpoints`` holds at least one *resumable* (rolling)
    checkpoint.

    Mirrors :meth:`CheckpointManager.latest` exactly — it enumerates via
    ``list_checkpoints``, which matches only the canonical ``ckpt_NNNNNNNNN.pt``
    rolling files (promotion + slim checkpoints are deliberately excluded there
    too, since a slim/policy-only file must never be silently resumed as a full
    run). So this predicate reflects precisely what
    :func:`catan_rl.ppo.training_loop.maybe_resume_from_checkpoint` would pick up.
    """
    from catan_rl.checkpoint.manager import list_checkpoints

    return bool(list_checkpoints(run_dir / "checkpoints"))


def prepare_run_directory(run_dir: Path, *, resume: bool = False) -> Path:
    """Resolve + create a STABLE (non-timestamped) run directory for a resumable
    run (the ``--run-dir`` path), and return it.

    Unlike :func:`build_run_directory`, this does NOT append a timestamp: the
    directory name is exactly ``run_dir`` (tilde/relative expanded), created with
    ``exist_ok=True`` so a relaunch reuses it. That stability is the crash-recovery
    contract for a long run — because the run dir is stable, the checkpoint
    manager's directory is stable, and
    :func:`~catan_rl.ppo.training_loop.maybe_resume_from_checkpoint` finds
    ``checkpoints/`` on relaunch and restores policy + optimizer + league pool +
    RNG + the per-lineage plateau clock instead of re-warm-starting from the seed
    into a fresh timestamped dir (which would silently discard the league pool,
    optimizer, and every reanchor promotion).

    Idempotent: the FIRST launch into an empty dir starts fresh (the warm-start
    from ``init_policy_checkpoint`` fires; no checkpoint to resume), and any later
    relaunch into the same dir resumes. This is exactly what a launchd/KeepAlive
    supervisor needs — but see the ordering note: KeepAlive is only safe to add
    AFTER resume is confirmed, else a fresh dir with no ckpt would restart from
    the seed on every crash.

    ``resume=True`` (the ``--resume`` guard) additionally REQUIRES an existing
    resumable checkpoint and raises ``FileNotFoundError`` otherwise — a fail-fast
    against the "relaunch silently restarts from the seed" foot-gun when the
    operator *intended* to continue a crashed run (e.g., a typo'd path).
    """
    run_dir = run_dir.expanduser().resolve()
    if resume and not _run_dir_has_resumable_checkpoint(run_dir):
        raise FileNotFoundError(
            f"--resume was set but no resumable checkpoint exists under "
            f"{run_dir / 'checkpoints'} (a rolling 'ckpt_NNNNNNNNN.pt'). Drop "
            f"--resume to start a fresh run in this directory, or point --run-dir "
            f"at the crashed run's directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(cfg: TrainConfig, run_dir: Path) -> Path:
    """Write the resolved config to ``run_dir/config.yaml`` and return the
    path. Reloading the snapshot reproduces the run."""
    path = run_dir / "config.yaml"
    cfg.to_yaml(path)
    return path


def _git_metadata() -> dict[str, str]:
    """Return ``{sha, dirty, branch}`` from git, or sentinel values on
    non-git checkouts. Cloud-rental ops use this to identify the lineage
    of code that produced a run when re-reading a checkpoint weeks later."""
    import subprocess

    def _run(cmd: list[str]) -> str:
        try:
            out = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode != 0:
                return "unknown"
            return out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"

    sha = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    diff_check = _run(["git", "status", "--porcelain"])
    dirty = "true" if diff_check and diff_check != "unknown" else "false"
    return {"sha": sha, "branch": branch, "dirty": dirty}


def write_run_metadata(
    run_dir: Path,
    *,
    resolved_device: str,
    launch_argv: list[str],
    start_utc: float | None = None,
) -> Path:
    """Write ``run_dir/run_metadata.yaml`` capturing the run's identity.

    Separate from ``config.yaml`` because the config is supposed to
    round-trip exactly back into a ``TrainConfig``; this file captures
    everything ELSE that's needed to reproduce a run from a
    checkpoint — git SHA, the actual resolved device (which the
    snapshotted config may have as ``"auto"``), the launch command,
    hostname, and UTC start time.
    """
    import socket

    git = _git_metadata()
    metadata = {
        "git": git,
        "resolved_device": resolved_device,
        "launch_cmd": list(launch_argv),
        "hostname": socket.gethostname(),
        "start_utc": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_utc))
            if start_utc is not None
            else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
    }
    path = run_dir / "run_metadata.yaml"
    path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return path


_OWNED_HANDLER_TAG = "catan_rl.train.entry"


def setup_logging(run_dir: Path, *, verbose: bool) -> logging.Logger:
    """Attach a file handler at ``run_dir/train.log`` plus stdout.

    Only removes handlers WE installed (tagged via ``handler.name``) so
    cloud supervisors (Modal, wandb, SkyPilot) that pre-install their own
    handlers on the same logger aren't silently nuked. Closes file
    handlers from prior ``main()`` calls in the same process to avoid
    FD leaks on Modal-style respawn-on-crash supervisors.
    """
    logger = logging.getLogger("catan_rl.train")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(logger.handlers):
        if h.name == _OWNED_HANDLER_TAG:
            h.close()
            logger.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.name = _OWNED_HANDLER_TAG
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(fmt)
    file_handler.name = _OWNED_HANDLER_TAG
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def construct_trainer(
    cfg: TrainConfig,
    run_dir: Path,
    *,
    device_label: str,
    logger: logging.Logger,
    max_updates: int | None = None,
) -> str | None:
    """Drive the Phase 10 end-to-end training loop.

    Delegates to
    :func:`catan_rl.ppo.training_loop.run_training` which composes
    Phases 0-8 into a single resumable loop. piKL (Phase 9) is
    parked at the primitive layer and is not consumed here yet —
    the trainer doesn't accept an anchor argument and ``PiKLConfig``
    is not a field on ``TrainConfig``.

    Returns the loop's ``stop_reason`` (``None`` = budget/max_updates
    exhausted; ``"hard"``/``"soft"`` = plateau auto_stop; ``"disk"`` =
    free-disk-guard trip / disk-abort) so :func:`main` can pick an exit code.
    """
    from catan_rl.ppo.training_loop import run_training

    state = run_training(
        cfg,
        run_dir=run_dir,
        device_label=device_label,
        logger=logger,
        max_updates=max_updates,
    )
    return state.stop_reason


#: Retained for back-compat with any prior caller that ran a Phase 2
#: dry-run and expected the placeholder exit code. The new loop never
#: emits it (success → 0, failure → raised exception).
EXIT_TRAINER_NOT_WIRED = 64

#: Process exit code when the free-disk guard tripped a save and the run
#: aborted (after writing a policy-only slim fallback + ``disk_abort.json``).
#: Nonzero so a supervisor / the operator can tell a stranded run apart from a
#: clean plateau auto-stop (which still exits 0). See the disk-guard path in
#: ``training_loop.run_training_loop``.
EXIT_DISK_ABORT = 65


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the desired process exit code."""
    launch_argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

    # Resolve the device target up-front so a missing CUDA/MPS backend
    # fails immediately rather than midway through trainer construction.
    resolved_device = resolve_device(cfg.device)

    # Run directory: a stable ``--run-dir`` (resumable, no timestamp — the
    # crash-recovery path for long runs) OR the default timestamped dir. The
    # ``--resume`` guard is meaningless without a ``--run-dir`` to resume into.
    if args.run_dir is not None:
        run_dir = prepare_run_directory(args.run_dir, resume=args.resume)
    else:
        if args.resume:
            raise ValueError(
                "--resume requires --run-dir (which directory should it resume?). "
                "Pass --run-dir <stable path> for a resumable run."
            )
        run_dir = build_run_directory(cfg)
    logger = setup_logging(run_dir, verbose=args.verbose)

    setup_seeding(cfg.seed)

    snapshot_path = snapshot_config(cfg, run_dir)
    metadata_path = write_run_metadata(
        run_dir, resolved_device=resolved_device, launch_argv=launch_argv
    )
    logger.info("run_dir=%s", run_dir)
    if args.run_dir is not None:
        will_resume = _run_dir_has_resumable_checkpoint(run_dir)
        logger.info(
            "stable run-dir (resumable): %s at launch — will %s",
            "checkpoint present" if will_resume else "no checkpoint",
            "RESUME (restore policy+optimizer+league+RNG+plateau clock)"
            if will_resume
            else "start FRESH (warm-start from init_policy_checkpoint if set)",
        )
    logger.info("config snapshot at %s", snapshot_path)
    logger.info("run metadata at %s", metadata_path)
    logger.info("resolved device: %s (requested: %s)", resolved_device, cfg.device)
    if resolved_device == "cpu":
        import torch  # local import — keeps script-import cheap

        if torch.backends.mps.is_available():
            logger.warning(
                "running on CPU while MPS is available. The batched SGD update "
                "is ~80%% of each PPO update and runs ~2.6-3.3x faster on MPS "
                "at batch=512 (see the device docstring in arguments.py). Drop "
                "any explicit '--device cpu' / keep device:auto to use MPS. "
                "Eval is pinned to CPU regardless (batch=1 is faster on CPU)."
            )
    logger.info("seeded numpy + torch + stdlib random with seed=%d", cfg.seed)

    transitions_per_rollout = cfg.rollout.n_envs * cfg.rollout.n_steps
    n_updates = cfg.total_steps // transitions_per_rollout
    logger.info(
        "rollout shape: n_envs=%d, n_steps=%d, transitions_per_rollout=%d, "
        "n_updates=%d, total_steps=%d",
        cfg.rollout.n_envs,
        cfg.rollout.n_steps,
        transitions_per_rollout,
        n_updates,
        cfg.total_steps,
    )

    if args.dry_run:
        logger.info("--dry-run set; exiting before trainer construction.")
        return 0

    try:
        stop_reason = construct_trainer(
            cfg,
            run_dir,
            device_label=resolved_device,
            logger=logger,
            max_updates=args.max_updates,
        )
    except NotImplementedError as e:
        logger.error("trainer not yet wired: %s", e)
        return EXIT_TRAINER_NOT_WIRED

    if stop_reason == "disk":
        logger.critical(
            "run aborted by the free-disk guard (disk_abort.json written under %s); "
            "exiting %d — reclaim disk (scripts/reclaim_disk.py) before resuming",
            run_dir,
            EXIT_DISK_ABORT,
        )
        return EXIT_DISK_ABORT

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Internal helper re-exports for tests (so they can avoid spawning the
# script as a subprocess for unit tests).
# ---------------------------------------------------------------------------

__all__ = [
    "EXIT_DISK_ABORT",
    "EXIT_TRAINER_NOT_WIRED",
    "build_parser",
    "build_run_directory",
    "construct_trainer",
    "main",
    "prepare_run_directory",
    "resolve_config",
    "setup_logging",
    "setup_seeding",
    "snapshot_config",
    "write_run_metadata",
]


# Sanity: when imported as a module, expose the yaml dependency so static
# checkers can see it's intentionally pulled in (the snapshot path uses it
# transitively via TrainConfig.to_yaml).
_ = yaml
