"""Round-trip tests for the SubprocVecEnv worker loop.

Phase 1 deliverable: confirm a single worker can be forked, take RESET /
STEP / GET_ACTION_MASKS / CLOSE commands, and return the same shapes/dtypes
the in-process env would. No SubprocGameManager yet — this tests the worker
in isolation against a raw ``mp.Pipe``.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from catan_rl.selfplay.subproc_vec_env import _worker
from catan_rl.selfplay.vec_env_protocol import Cmd


def _start_worker(env_kwargs: dict, seed: int = 0, env_idx: int = 0):
    """Spawn one worker over a ``fork`` context. Returns ``(process, parent_pipe)``."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=True)
    proc = ctx.Process(
        target=_worker,
        args=(child, env_kwargs, seed, env_idx),
        daemon=True,
    )
    proc.start()
    # The parent never uses the child end after fork.
    child.close()
    return proc, parent


def _close_worker(proc, parent):
    parent.send((Cmd.CLOSE, None))
    parent.recv()
    parent.close()
    proc.join(timeout=2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)


def test_worker_reset_step_close_round_trip() -> None:
    """Smoke test: fork worker, reset, step a no-op-ish action, close cleanly."""
    proc, parent = _start_worker(
        env_kwargs={"opponent_type": "random", "max_turns": 50},
        seed=42,
    )
    try:
        # RESET — initial obs + info dict
        parent.send((Cmd.RESET, {}))
        obs, info = parent.recv()
        assert isinstance(obs, dict), "obs must be a dict"
        # The phase4 obs schema includes these keys (most fundamental — no
        # phase flags needed). If any disappear we want to know loudly.
        for k in ("tile_representations", "current_player_main", "next_player_main"):
            assert k in obs, f"missing obs key: {k}"
        assert isinstance(info, dict)

        # GET_ACTION_MASKS — must come back as a dict of ndarrays
        parent.send((Cmd.GET_ACTION_MASKS, None))
        masks = parent.recv()
        assert isinstance(masks, dict)
        assert "type" in masks  # the action-type head is always present

        # STEP with action 12 (RollDice) — game starts in the setup phase
        # before the dice phase, but we just want to verify the round-trip.
        # Using EndTurn (3) is safer because it always has *some* response;
        # we don't care if the env returns negative reward, just that the
        # tuple shape is right.
        action = np.array([3, 0, 0, 0, 0, 0], dtype=np.int64)
        parent.send((Cmd.STEP, action))
        obs2, reward, terminated, truncated, info2 = parent.recv()
        assert isinstance(obs2, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info2, dict)
    finally:
        _close_worker(proc, parent)


def test_worker_unknown_cmd_kills_pipe() -> None:
    """Sending an invalid cmd ID should crash the worker; the parent sees EOF.

    This is the failure mode crash-recovery (Phase 4) hooks into. We just
    confirm the pipe breaks cleanly so the recv() raises rather than hanging.
    """
    proc, parent = _start_worker(env_kwargs={"opponent_type": "random"})
    try:
        # 999 is intentionally not a valid Cmd value
        parent.send((999, None))
        try:  # noqa: SIM105
            parent.recv()
        except (EOFError, BrokenPipeError):
            pass  # expected — worker raised, pipe closed
        # The worker process should also have exited.
        proc.join(timeout=2.0)
        assert not proc.is_alive(), "worker should have died on unknown cmd"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        parent.close()
