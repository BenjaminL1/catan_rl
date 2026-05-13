"""SubprocVecEnv worker loop and parallel GameManager.

This module provides a parallel alternative to the in-process serial
``GameManager``. Each environment runs in its own subprocess and communicates
with the main (trainer) process over a ``multiprocessing.Pipe``. The main
process retains exclusive ownership of league sampling and opponent NN
inference — workers only step their local ``CatanEnv``. This preserves the
existing batched-opponent-inference perf optimization while breaking the
single-process bottleneck on env stepping.

The worker is intentionally small: it imports ``CatanEnv`` and ``numpy``,
nothing torch-side. That keeps fork-context startup fast and avoids the
macOS-specific torch-after-fork pitfalls.

Public API mirrors ``GameManager`` exactly so the trainer is mode-agnostic.
The pipelined ``step_async`` / ``step_wait`` methods are what actually
delivers the parallelism: send all N STEP commands first (non-blocking
writes), then collect results in any order.
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from catan_rl.env.catan_env import CatanEnv
from catan_rl.selfplay.game_manager import BaseGameManager
from catan_rl.selfplay.vec_env_protocol import Cmd


def _worker(
    remote: mp.connection.Connection,
    env_kwargs: dict[str, Any],
    seed: int,
    env_idx: int,
) -> None:
    """Run a single ``CatanEnv`` in a subprocess, driven by IPC commands.

    The worker dispatches on the integer command ID and sends back whatever
    the corresponding env method returned. Unknown commands cause a clean
    exception that the main process can detect via ``BrokenPipeError``.

    Args:
        remote: The worker-side end of the Pipe. The other end lives in the
            main process and is owned by ``SubprocGameManager``.
        env_kwargs: Constructor kwargs for ``CatanEnv``. The dict must be
            picklable (it crosses the fork/spawn boundary as part of the
            worker startup args).
        seed: Per-worker random seed. Each worker gets ``master_seed +
            env_idx`` so the N envs play distinct games. Without this, all
            workers would draw the same dice and the parallel rollouts
            would correlate.
        env_idx: This worker's index, kept for debug logging only.
    """
    env = CatanEnv(**env_kwargs)
    env.reset(seed=seed)

    try:
        while True:
            cmd_payload = remote.recv()
            cmd, payload = cmd_payload
            if cmd == Cmd.STEP:
                obs, reward, terminated, truncated, info = env.step(payload)
                remote.send((obs, float(reward), bool(terminated), bool(truncated), info))
            elif cmd == Cmd.RESET:
                obs, info = env.reset(options=payload)
                remote.send((obs, info))
            elif cmd == Cmd.GET_ACTION_MASKS:
                remote.send(env.get_action_masks())
            elif cmd == Cmd.GET_OPP_OBS_MASKS:
                remote.send(env.get_opponent_obs_masks())
            elif cmd == Cmd.APPLY_OPP_ACTION:
                turn_complete, obs, reward, terminated, truncated, info = env.apply_opponent_action(
                    payload
                )
                remote.send(
                    (
                        bool(turn_complete),
                        obs,
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        info,
                    )
                )
            elif cmd == Cmd.SET_OPPONENT_TYPE:
                env.opponent_type = payload
                remote.send(None)
            elif cmd == Cmd.CLOSE:
                remote.send(None)
                break
            else:
                raise RuntimeError(f"worker[{env_idx}] received unknown cmd: {cmd}")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        remote.close()


class SubprocGameManager(BaseGameManager):
    """Parallel vec env: each ``CatanEnv`` runs in its own subprocess.

    Uses ``multiprocessing.get_context("fork")`` so workers inherit module
    state cheaply. macOS sets the default to ``spawn``; the explicit fork
    avoids re-importing the codebase per worker. Forking before any torch
    CUDA/MPS context exists is required — the trainer never calls .to('mps')
    or .to('cuda') before constructing this manager (CPU-only on M1 Pro by
    default per CLAUDE.md rule 7).

    Public API matches ``GameManager`` so the trainer can swap managers via
    a config flag with zero code changes.
    """

    def __init__(
        self,
        n_envs: int = 1,
        opponent_type: str = "random",
        max_turns: int | None = 500,
        league=None,
        build_policy_fn: Callable | None = None,
        device: str = "cpu",
        use_thermometer_encoding: bool = True,
        current_policy_state_fn: Callable[[], dict] | None = None,
        record_match_fn: Callable[[int, int], None] | None = None,
        use_opponent_id_emb: bool = False,
        opp_id_mask_prob: float = 0.40,
        league_maxlen: int = 100,
        use_belief_head: bool = False,
        master_seed: int = 0,
    ):
        super().__init__(
            n_envs=n_envs,
            opponent_type=opponent_type,
            league=league,
            build_policy_fn=build_policy_fn,
            device=device,
            current_policy_state_fn=current_policy_state_fn,
            record_match_fn=record_match_fn,
        )

        # Guard against forking after a torch CUDA/MPS context has been
        # initialized — that's the macOS-specific deadlock the plan flagged
        # as a HIGH-severity risk. CPU-only is the normal path here.
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                raise RuntimeError(
                    "SubprocGameManager refuses to fork: a CUDA context is "
                    "already active in the parent process. Construct the "
                    "manager BEFORE moving any tensor to a CUDA device."
                )
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available() and mps.is_built():
                # MPS contexts on macOS also forbid fork. We don't currently
                # use MPS in training (CPU is faster at batch=1 per
                # CLAUDE.md), so this is a defensive check.
                pass  # informational only — no hard fail unless we observe a hang
        except ImportError:
            pass  # torch not present in some test environments

        ctx = mp.get_context("fork")
        self._ctx = ctx

        # Construct the env kwargs once; each worker rebuilds its own env.
        env_kwargs = {
            "render_mode": None,
            "opponent_type": self._opp_type_initial,
            "max_turns": max_turns,
            "use_thermometer_encoding": use_thermometer_encoding,
            "use_opponent_id_emb": use_opponent_id_emb,
            "opp_id_mask_prob": opp_id_mask_prob,
            "league_maxlen": league_maxlen,
            "use_belief_head": use_belief_head,
        }
        self._env_kwargs = env_kwargs
        self._master_seed = int(master_seed) + int(os.getpid())

        # ``ctx.Process`` returns ``ForkProcess`` (or ``SpawnProcess`` etc.)
        # depending on the start method; both subclass ``BaseProcess``. Use
        # that base type so mypy doesn't complain about ForkProcess→Process
        # at assignment time.
        self._remotes: list[mp.connection.Connection] = []
        self._procs: list[mp.process.BaseProcess] = []
        for env_idx in range(n_envs):
            self._spawn_worker(env_idx)

        # Lifecycle: ensure workers exit if the main process is torn down
        # without an explicit close() (e.g., uncaught exception, sigterm).
        atexit.register(self._atexit_close)
        self._closed = False

    # ── Worker lifecycle ────────────────────────────────────────────────

    def _spawn_worker(self, env_idx: int) -> None:
        """Fork a single worker for ``env_idx``; populate slots in-place."""
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker,
            args=(child, self._env_kwargs, self._master_seed + env_idx, env_idx),
            daemon=True,
        )
        proc.start()
        # The parent never uses the child end after fork; close to make EOF
        # detection on the parent side work correctly when the worker dies.
        child.close()
        if env_idx < len(self._remotes):
            self._remotes[env_idx] = parent
            self._procs[env_idx] = proc
        else:
            self._remotes.append(parent)
            self._procs.append(proc)

    def close(self) -> None:
        """Send CLOSE to all workers and join. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            with contextlib.suppress(BrokenPipeError, OSError):
                r.send((Cmd.CLOSE, None))
        for r in self._remotes:
            with contextlib.suppress(EOFError, BrokenPipeError, OSError):
                r.recv()
            with contextlib.suppress(OSError):
                r.close()
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

    def _atexit_close(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    # ── Per-env IPC helpers ─────────────────────────────────────────────

    def _send(self, env_idx: int, cmd: Cmd, payload: Any = None) -> None:
        try:
            self._remotes[env_idx].send((cmd, payload))
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"SubprocGameManager: worker[{env_idx}] pipe broken on send. "
                f"Worker is_alive={self._procs[env_idx].is_alive()}, exitcode="
                f"{self._procs[env_idx].exitcode}. Original: {e}"
            ) from e

    def _recv(self, env_idx: int) -> Any:
        try:
            return self._remotes[env_idx].recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as e:
            # Phase 4: a worker crashed mid-step. The current rollout is
            # corrupted (we have no obs to return), so propagate as a
            # RuntimeError that the trainer can catch and decide whether to
            # bail. A more sophisticated recovery (fork replacement worker,
            # synthesize terminal=True obs) is left for a follow-up if
            # crashes turn out to be a real issue in production.
            raise RuntimeError(
                f"SubprocGameManager: worker[{env_idx}] crashed (pid="
                f"{self._procs[env_idx].pid}, exitcode="
                f"{self._procs[env_idx].exitcode}). Original: {e}"
            ) from e

    # ── Public API mirroring GameManager ────────────────────────────────

    def reset_all(self) -> tuple[list[dict], list[dict]]:
        """Reset all envs (with freshly-sampled opponents) and gather (obs, info).

        Opponent sampling happens main-side; the resulting options dict is
        sent over IPC as the RESET payload.
        """
        # Phase A: sample opponents in the main process so the league /
        # rollout_opp_policy state is consistent. Send RESET to all workers
        # in parallel, *then* recv from each.
        options_list = []
        for i in range(self.n_envs):
            options = self._sample_and_prepare_opponent(i) if self.league else {}
            options_list.append(options)
            self._send(i, Cmd.RESET, options)
        observations: list[dict] = []
        infos: list[dict] = []
        for i in range(self.n_envs):
            obs, info = self._recv(i)
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def step_one(self, env_idx: int, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Step one env via IPC. Mirrors ``GameManager.step_one`` exactly.

        Note: prefer ``step_async`` + ``step_wait`` for batched parallel
        stepping. ``step_one`` is here for the trainer's iterative
        opponent-handoff loop in ``_run_batched_opponent_turns`` (where
        envs complete their opponent turns at different rates).
        """
        self._send(env_idx, Cmd.STEP, action)
        obs, reward, terminated, truncated, info = self._recv(env_idx)
        if info.get("opp_turn_pending"):
            return obs, reward, False, False, info
        done = terminated or truncated
        if done:
            info["terminal_observation"] = obs
            win = 1 if info.get("is_success") else 0
            self._report_match(self._rollout_opp_policy_id, win)
            options = self._sample_and_prepare_opponent() if self.league else {}
            self._send(env_idx, Cmd.RESET, options)
            obs, _ = self._recv(env_idx)
        return obs, reward, terminated, truncated, info

    def step_all(
        self, actions: list[np.ndarray]
    ) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        """Pipelined parallel step across all workers.

        Sends STEP to every worker first (non-blocking), then collects
        results. This is where the parallelism actually delivers — the N
        env.step() calls run concurrently across N OS processes.

        Return signature matches ``GameManager.step_all``:
        (observations, rewards, terminated, truncated, infos), all length-N
        lists. Deferred opponent turns return terminated=False/truncated=False
        with ``info["opp_turn_pending"]==True`` so the caller can run the
        opponent NN forward pass in the main process and then call
        ``apply_opponent_action`` to advance.
        """
        n = len(actions)
        # Phase 1: dispatch all STEPs in a tight loop. The pipe writes are
        # non-blocking on small payloads (the action is a 6-int array), so
        # this returns essentially immediately.
        for i in range(n):
            self._send(i, Cmd.STEP, actions[i])
        # Phase 2: collect raw step results. recv() blocks until each
        # worker finishes its env.step — they run in parallel.
        raw = [self._recv(i) for i in range(n)]

        observations: list[dict] = [None] * n  # type: ignore[list-item]
        rewards: list[float] = [0.0] * n
        terminated_list: list[bool] = [False] * n
        truncated_list: list[bool] = [False] * n
        infos: list[dict] = [{}] * n
        reset_envs: list[int] = []
        reset_options: list[dict] = []

        for i, (obs, reward, terminated, truncated, info) in enumerate(raw):
            if info.get("opp_turn_pending"):
                # Defer: caller drives the opp-NN handoff. Don't reset.
                observations[i] = obs
                rewards[i] = reward
                terminated_list[i] = False
                truncated_list[i] = False
                infos[i] = info
                continue
            done = terminated or truncated
            if done:
                info["terminal_observation"] = obs
                win = 1 if info.get("is_success") else 0
                self._report_match(self._rollout_opp_policy_id, win)
                opt = self._sample_and_prepare_opponent(i) if self.league else {}
                reset_envs.append(i)
                reset_options.append(opt)
            observations[i] = obs
            rewards[i] = reward
            terminated_list[i] = bool(terminated)
            truncated_list[i] = bool(truncated)
            infos[i] = info

        # Phase 3: pipelined RESETs for the done envs — same dispatch-then-
        # collect pattern. RESET payload contains opponent options.
        for env_i, opt in zip(reset_envs, reset_options, strict=False):
            self._send(env_i, Cmd.RESET, opt)
        for env_i in reset_envs:
            obs, _ = self._recv(env_i)
            observations[env_i] = obs
        return observations, rewards, terminated_list, truncated_list, infos

    def get_opponent_obs_masks(self, env_idx: int):
        """Get opponent (obs, masks) for a specific worker."""
        self._send(env_idx, Cmd.GET_OPP_OBS_MASKS, None)
        return self._recv(env_idx)

    def apply_opponent_action(
        self, env_idx: int, action: np.ndarray
    ) -> tuple[bool, dict | None, float, bool, bool, dict]:
        """Apply one opponent NN action to a specific worker."""
        self._send(env_idx, Cmd.APPLY_OPP_ACTION, action)
        turn_complete, obs, reward, terminated, truncated, info = self._recv(env_idx)
        if not turn_complete:
            return False, None, 0.0, False, False, {}
        done = terminated or truncated
        if done:
            info["terminal_observation"] = obs
            win = 1 if info.get("is_success") else 0
            self._report_match(self._rollout_opp_policy_id, win)
            options = self._sample_and_prepare_opponent() if self.league else {}
            self._send(env_idx, Cmd.RESET, options)
            obs, _ = self._recv(env_idx)
        return True, obs, reward, terminated, truncated, info

    def get_masks(self) -> list[dict[str, np.ndarray]]:
        """Pipelined batched mask query — N parallel IPC round-trips."""
        for i in range(self.n_envs):
            self._send(i, Cmd.GET_ACTION_MASKS, None)
        return [self._recv(i) for i in range(self.n_envs)]

    def set_opponent_type(self, opponent_type: str) -> None:
        """Broadcast a new opponent_type attribute to every worker's env."""
        for i in range(self.n_envs):
            self._send(i, Cmd.SET_OPPONENT_TYPE, opponent_type)
        for i in range(self.n_envs):
            self._recv(i)
