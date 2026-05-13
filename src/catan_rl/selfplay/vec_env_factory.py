"""Factory for vec env managers — keeps the trainer mode-agnostic.

The factory dispatches on a single ``mode`` string. New modes (e.g. shared-
memory IPC, async pipelined eval) can be added here without touching the
trainer.

Usage:
    gm = make_vec_env("subproc", n_envs=8, opponent_type="random", ...)
"""

from __future__ import annotations

from typing import Any

from catan_rl.selfplay.game_manager import BaseGameManager, GameManager


def make_vec_env(mode: str, **kwargs: Any) -> BaseGameManager:
    """Construct a vec env manager of the requested mode.

    Args:
        mode: ``"serial"`` (default) for in-process N envs, ``"subproc"``
            for one subprocess per env. Any other value raises a clear
            ``ValueError`` at trainer-construction time so a typo in the
            YAML doesn't silently fall through to a default.
        **kwargs: Forwarded verbatim to the chosen manager. Both managers
            accept the same constructor arguments by design.

    Returns:
        A ``BaseGameManager`` subclass. Trainer code uses it polymorphically
        through the methods declared on ``BaseGameManager`` and the per-env
        operations declared on each subclass (which share signatures).
    """
    if mode == "serial":
        return GameManager(**kwargs)
    if mode == "subproc":
        # Imported lazily so test environments without ``multiprocessing``
        # workers running don't pay the import cost when they don't use it.
        from catan_rl.selfplay.subproc_vec_env import SubprocGameManager

        return SubprocGameManager(**kwargs)
    raise ValueError(
        f"unknown vec_env_mode={mode!r}; expected 'serial' or 'subproc'. "
        f"Check your YAML / arguments.py vec_env_mode key."
    )
