"""IPC protocol for the SubprocVecEnv worker loop.

The protocol is intentionally minimal: each command is an `(int_cmd, payload)`
tuple sent over a `multiprocessing.Pipe`, and each response is whatever the
underlying ``CatanEnv`` method returned. Workers run a single ``CatanEnv``
each; the main process owns league sampling, opponent NN inference, and
match-result reporting.

The IntEnum command IDs are stable across the codebase — adding a new command
appends a new value, never reorders. Workers built against an older protocol
on a newer trainer (or vice versa) will fail loudly on the unknown command
match in ``_worker`` rather than silently corrupting state.
"""

from __future__ import annotations

from enum import IntEnum


class Cmd(IntEnum):
    """Worker IPC commands. Each maps to a ``CatanEnv`` method or admin op."""

    STEP = 1
    RESET = 2
    GET_ACTION_MASKS = 3
    GET_OPP_OBS_MASKS = 4
    APPLY_OPP_ACTION = 5
    SET_OPPONENT_TYPE = 6
    CLOSE = 7
