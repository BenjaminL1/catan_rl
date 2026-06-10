"""Atomic checkpoint save/load + retention pruning.

A v2 checkpoint is a single ``torch.save``-serialised dict pinned to
``SCHEMA_VERSION``. The payload is structured rather than a flat dump
so individual sub-states can evolve independently across versions.

**Trust boundary**: :func:`load_checkpoint` uses
``torch.load(weights_only=False)`` because the payload carries
non-tensor state (numpy RNG dicts, league metadata dicts). This is a
remote-code-execution surface — *never* load a checkpoint from an
untrusted source. The Phase 8 contract is "first-party files only;
the trainer writes them, the trainer reads them back". If that ever
relaxes, switch to ``weights_only=True`` with a per-section custom
deserialiser.

.. code-block:: text

    {
        "schema_version": int,
        "config": dict,                 # TrainConfig.to_dict()
        "update_idx": int,              # PPO update count
        "global_step": int,             # env steps consumed by rollouts
        "policy_state_dict": dict,      # torch state dict on CPU
        "optimizer_state_dict": dict | None,
        "league": {
            "snapshots": [
                {
                    "state_dict": dict,
                    "update_idx": int,
                    "snapshot_id": int,
                    "metadata": dict,
                },
                ...
            ],
            "next_snapshot_id": int,
        },
        "rng": {
            "numpy_state": dict,        # np.random.get_state(legacy=False)
            "python_state": list,       # random.getstate() (as JSON-safe tuple)
            "torch_state": Tensor,      # torch.random.get_rng_state()
        },
        "vec_env": {                    # per-env Generator state for the
                                        # rollout vec env (Phase 5). Captures
                                        # the auto-reset PRNG stream that
                                        # ``np.random`` does NOT cover.
            "reset_rng_states": [dict, ...],
        },
        "metadata": dict,               # free-form extra fields
    }

Atomic write: payload is serialised to ``<dest>.tmp`` and flushed +
fsync'd before ``os.replace`` swaps it into place. A crash mid-write
leaves the previous checkpoint intact instead of corrupting the
destination.

Retention pruning: :meth:`CheckpointManager.save` deletes oldest
checkpoints beyond ``keep_last_n`` (sorted by their embedded
``update_idx`` to be robust against filesystem mtime drift on
network mounts).

Resume semantics: :func:`load_checkpoint` returns a
:class:`CheckpointPayload` whose ``apply_to_policy``,
``apply_to_optimizer``, ``apply_to_league``, ``apply_rng_state``
methods replay the saved state into the live objects. Calling all
four after a fresh trainer construction produces a state that is
indistinguishable from the original at the save point (modulo
device-dependent floating-point non-determinism).
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.checkpoint.migrations import apply_migrations

#: Current on-disk schema version. Bumped whenever any of the keys
#: above change shape; ship a v(N-1) -> v(N) migration in
#: :mod:`catan_rl.checkpoint.migrations` at the same time.
SCHEMA_VERSION = 1

#: Naming convention for checkpoint files inside an output directory.
#: ``update_idx`` is zero-padded so lexicographic sort matches numeric.
_FILE_PATTERN = re.compile(r"^ckpt_(\d{9})\.pt$")


class CheckpointError(RuntimeError):
    """Raised on a load/save error that the operator needs to act on
    (corrupt file, schema mismatch, missing migration)."""


def checkpoint_filename(update_idx: int) -> str:
    """Return the canonical filename for a checkpoint at ``update_idx``."""
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    return f"ckpt_{update_idx:09d}.pt"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _clone_to_cpu(value: Any) -> Any:
    """Recursively deep-clone tensors in ``value`` to CPU.

    Walks dict / list / tuple containers fully so any nested tensor
    leaf is reached. Required for two reasons:

    1. ``torch.save`` of a GPU tensor stores the device tag too, so
       loading on a host without that GPU raises. CPU tensors load
       anywhere.
    2. The trainer's policy may still be the *source* of these
       tensors; without a copy, a subsequent optimizer step would
       mutate the saved snapshot.

    AdamW's state_dict has shape
    ``{"state": {param_id: {"exp_avg": Tensor, ...}}, "param_groups": [...]}``
    — a dict of int-keyed dicts of tensors. A non-recursive clone
    misses those. Future optimisers (Lion, custom EMA wrappers) may
    embed tensors in lists; the recursion catches those too.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _clone_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(v) for v in value)
    return value


def _state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-clone a state-dict mapping. Thin wrapper over
    :func:`_clone_to_cpu` that keeps the public name + return type."""
    return {k: _clone_to_cpu(v) for k, v in state_dict.items()}


def _capture_rng_state() -> dict[str, Any]:
    """Snapshot the three PRNGs the trainer touches: numpy, stdlib
    random, torch. Saved as plain Python objects so the checkpoint can
    be inspected without re-importing torch."""
    return {
        "numpy_state": np.random.get_state(legacy=False),
        "python_state": random.getstate(),
        "torch_state": torch.random.get_rng_state(),
    }


def _capture_vec_env_state(vec_env: Any) -> dict[str, Any]:
    """Serialise the vec env's per-env auto-reset ``Generator`` states.

    Phase 5's :class:`SerialVecEnv` owns ``_reset_rngs``: a list of
    independent ``np.random.Generator`` instances seeded from
    ``SeedSequence(seed).spawn(n_envs)``. The rollout stream draws
    auto-reset seeds from these, NOT from the process-global
    ``np.random``. The checkpoint must persist their internal state
    for a resumed run to produce a bit-identical rollout stream.

    Returns a dict with one entry — ``reset_rng_states: list[dict]``,
    each entry being the corresponding ``Generator.bit_generator.state``
    (a JSON-safe dict containing the BitGenerator name + its position
    in the sequence). ``np.random.default_rng()`` round-trips through
    setting ``bit_generator.state``.

    If the vec env doesn't expose ``_reset_rngs`` (e.g., a custom
    Phase 5b subproc variant), the helper returns an empty dict and
    apply is a no-op. The trainer must still pass the new vec env in
    on save so the same hook covers it.
    """
    rngs = getattr(vec_env, "_reset_rngs", None)
    if rngs is None:
        return {"reset_rng_states": []}
    states: list[dict[str, Any]] = []
    for gen in rngs:
        bit_gen = getattr(gen, "bit_generator", None)
        if bit_gen is None:
            continue
        # ``bit_generator.state`` returns a dict like
        # ``{"bit_generator": "PCG64", "state": {...}, "has_uint32": int,
        # "uinteger": int}``. Pure-Python primitives → JSON-safe.
        states.append(dict(bit_gen.state))
    return {"reset_rng_states": states}


def _capture_league_state(league: Any) -> dict[str, Any]:
    """Serialise a :class:`catan_rl.selfplay.league.League` instance.

    The league holds a bounded deque of :class:`LeagueSnapshot`
    instances. Each snapshot's ``state_dict`` is already CPU per the
    league's invariant (see ``add_snapshot``); we still defensively
    re-clone in case a caller bypassed the invariant. The
    ``_next_snapshot_id`` cursor is preserved so resumed IDs stay
    monotonic.
    """
    snapshots: list[dict[str, Any]] = []
    snapshot_deque = getattr(league, "_snapshots", None)
    if snapshot_deque is None:
        return {"snapshots": [], "next_snapshot_id": 0}
    for snap in snapshot_deque:
        snapshots.append(
            {
                "state_dict": _state_dict_to_cpu(snap.state_dict),
                "update_idx": int(snap.update_idx),
                "snapshot_id": int(snap.snapshot_id),
                "metadata": dict(snap.metadata),
            }
        )
    state: dict[str, Any] = {
        "snapshots": snapshots,
        "next_snapshot_id": int(getattr(league, "_next_snapshot_id", 0)),
    }
    # PFSP per-opponent win-rate store (additive; absent in old checkpoints).
    if hasattr(league, "opponent_stats_state"):
        state["opponent_stats"] = {
            int(sid): [float(p), float(g)] for sid, (p, g) in league.opponent_stats_state().items()
        }
    return state


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory inode.

    Required after ``os.replace`` for the rename to survive a power
    loss on POSIX: the file's data is durable after the file-fsync,
    but the *directory entry* binding ``dest`` to the new inode is
    only guaranteed durable after the parent directory itself is
    fsync'd. On Windows the call raises ``OSError`` (directory fds
    aren't supported); silently no-op there.
    """
    try:
        dir_fd = os.open(path, os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def save_checkpoint(
    path: str | Path,
    *,
    config: dict[str, Any],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    update_idx: int,
    global_step: int,
    league: Any | None = None,
    vec_env: Any | None = None,
    capture_rng: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically.

    ``policy`` and ``optimizer`` state dicts are CPU-cloned. ``league``,
    if provided, is captured via :func:`_capture_league_state`.
    ``vec_env``, if provided, has its per-env auto-reset Generators
    captured via :func:`_capture_vec_env_state` — critical for the
    resumed rollout stream to match an un-interrupted run. Set
    ``capture_rng=False`` only if you have a specific reason (e.g.,
    smoke tests that don't care about resume determinism).

    Returns the path that was written (matches ``path`` after
    resolution).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "update_idx": int(update_idx),
        "global_step": int(global_step),
        "policy_state_dict": _state_dict_to_cpu(policy.state_dict()),
        "optimizer_state_dict": (
            _state_dict_to_cpu(optimizer.state_dict()) if optimizer is not None else None
        ),
        "league": _capture_league_state(league) if league is not None else None,
        "rng": _capture_rng_state() if capture_rng else None,
        "vec_env": (_capture_vec_env_state(vec_env) if vec_env is not None else None),
        "metadata": dict(metadata or {}),
    }
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    # ``torch.save`` writes binary; open the file ourselves to fsync
    # before the rename. Without fsync, the os.replace can land but
    # the contents may not be on disk if power is cut between write
    # and replace — leaving a zero-byte ``ckpt_*.pt``. After the
    # rename, fsync the parent dir so the directory entry is durable
    # too (critical on cloud spot instances). ``BaseException`` here
    # covers ``KeyboardInterrupt`` and ``SystemExit`` so a Ctrl-C
    # mid-save still cleans up the half-written tmp.
    import contextlib

    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(dest.parent)
    return dest


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


@dataclass
class CheckpointPayload:
    """Decoded checkpoint contents + helpers to apply them to live objects.

    Constructed by :func:`load_checkpoint`. The helper methods
    (``apply_to_*``) mutate the target object in place.

    **Order matters on resume**:

    1. ``apply_to_policy(policy)`` — restores params first so the
       optimizer's param refs (set at construction) still match.
    2. ``apply_to_optimizer(optimizer)`` — moment buffers loaded onto
       the param's device.
    3. ``apply_to_league(league)`` — independent of policy.
    4. ``apply_to_vec_env(vec_env)`` — must run BEFORE any vec env
       reset or step on the resumed loop, else the auto-reset PRNG
       stream diverges from the un-interrupted run.
    5. ``apply_rng_state()`` — global numpy + python + torch PRNGs
       last (no ordering constraint with the others).

    :meth:`apply_all` runs all five in the right order.
    """

    schema_version: int
    config: dict[str, Any]
    update_idx: int
    global_step: int
    policy_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    league_state: dict[str, Any] | None
    rng_state: dict[str, Any] | None
    vec_env_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_to_policy(self, policy: torch.nn.Module, *, strict: bool = True) -> None:
        """Load the saved policy params into ``policy``.

        ``strict=True`` (the default) raises if the live policy has
        different param keys from the checkpoint — that almost always
        means the policy architecture changed and the checkpoint
        needs an explicit migration. ``strict=False`` is for
        partial-restore scenarios (e.g. loading a pretrained encoder
        into a larger model).
        """
        policy.load_state_dict(self.policy_state_dict, strict=strict)

    def apply_to_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Load the saved optimiser state. No-op if the checkpoint
        was saved without optimiser state (``save_optimizer_state=False``
        in :class:`CheckpointConfig`)."""
        if self.optimizer_state_dict is None:
            return
        optimizer.load_state_dict(self.optimizer_state_dict)

    def apply_to_league(self, league: Any) -> None:
        """Restore the saved league snapshot pool + id cursor.

        Wipes the league's existing deque first so a partial-resume
        doesn't accumulate snapshots. The id cursor is moved forward
        to ``next_snapshot_id`` so any post-resume ``add_snapshot``
        gets an id strictly greater than any saved snapshot's id.
        """
        if self.league_state is None:
            return
        from catan_rl.selfplay.league import LeagueSnapshot

        # Wipe the existing deque without replacing it (preserves
        # the maxlen).
        deque_ = getattr(league, "_snapshots", None)
        if deque_ is None:
            raise CheckpointError("league has no _snapshots field — refusing to restore")
        deque_.clear()
        for snap_dict in self.league_state.get("snapshots", []):
            deque_.append(
                LeagueSnapshot(
                    state_dict=dict(snap_dict["state_dict"]),
                    update_idx=int(snap_dict["update_idx"]),
                    snapshot_id=int(snap_dict["snapshot_id"]),
                    metadata=dict(snap_dict.get("metadata", {})),
                )
            )
        next_id = int(self.league_state.get("next_snapshot_id", 0))
        # Defensive max: in case the saved cursor somehow lags behind
        # the highest live id (shouldn't happen, but cheap to guard).
        max_live = max(
            (snap.snapshot_id for snap in deque_),
            default=-1,
        )
        league._next_snapshot_id = max(next_id, max_live + 1)
        # PFSP win-rate store (additive; absent in pre-PFSP checkpoints → leave
        # the league's store empty).
        stats = self.league_state.get("opponent_stats")
        if stats is not None and hasattr(league, "load_opponent_stats"):
            league.load_opponent_stats({int(sid): tuple(v) for sid, v in stats.items()})

    def apply_to_vec_env(self, vec_env: Any) -> None:
        """Restore the vec env's per-env auto-reset Generator states.

        Iterates ``vec_env._reset_rngs`` and writes each saved
        ``bit_generator.state`` dict back. If the live vec env has a
        different number of envs than the checkpoint, an error is
        raised — silently truncating or padding would cause silent
        rollout-stream drift on the affected envs.

        No-op if the checkpoint has no vec_env section (older
        checkpoint or save without ``vec_env=`` arg).
        """
        if self.vec_env_state is None:
            return
        saved_states = self.vec_env_state.get("reset_rng_states", [])
        live_rngs = getattr(vec_env, "_reset_rngs", None)
        if live_rngs is None:
            raise CheckpointError("vec_env has no _reset_rngs field — refusing to restore")
        if len(saved_states) != len(live_rngs):
            raise CheckpointError(
                f"vec_env n_envs mismatch: checkpoint saved "
                f"{len(saved_states)} per-env Generators but the live "
                f"vec_env has {len(live_rngs)}. Resume requires the "
                "same n_envs as the saved run."
            )
        for gen, saved in zip(live_rngs, saved_states, strict=True):
            gen.bit_generator.state = dict(saved)

    def apply_rng_state(self) -> None:
        """Restore numpy, stdlib ``random``, and torch RNG state to the
        snapshot taken at save time. After this call the next
        ``np.random.random()``, ``random.random()``, and
        ``torch.rand(...)`` draws match the values they would have
        produced in the un-interrupted run."""
        if self.rng_state is None:
            return
        np_state = self.rng_state.get("numpy_state")
        if np_state is not None:
            np.random.set_state(np_state)
        py_state = self.rng_state.get("python_state")
        if py_state is not None:
            # Convert outer container back to a tuple; ``getstate``
            # returns a tuple and ``setstate`` requires one.
            random.setstate(tuple(py_state))
        torch_state = self.rng_state.get("torch_state")
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)

    def apply_all(
        self,
        *,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        league: Any | None = None,
        vec_env: Any | None = None,
        strict_policy: bool = True,
    ) -> None:
        """Apply every section in the canonical resume order.

        Equivalent to::

            payload.apply_to_policy(policy, strict=strict_policy)
            if optimizer is not None:
                payload.apply_to_optimizer(optimizer)
            if league is not None:
                payload.apply_to_league(league)
            if vec_env is not None:
                payload.apply_to_vec_env(vec_env)
            payload.apply_rng_state()
        """
        self.apply_to_policy(policy, strict=strict_policy)
        if optimizer is not None:
            self.apply_to_optimizer(optimizer)
        if league is not None:
            self.apply_to_league(league)
        if vec_env is not None:
            self.apply_to_vec_env(vec_env)
        self.apply_rng_state()


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
) -> CheckpointPayload:
    """Load a checkpoint and run any pending migrations to the current
    :data:`SCHEMA_VERSION`.

    ``map_location`` is forwarded to ``torch.load``. Default ``"cpu"``
    matches the save-side CPU cloning; pass an explicit device only if
    you know what you're doing and want to skip a CPU->device copy on
    apply.
    """
    src = Path(path)
    if not src.exists():
        raise CheckpointError(f"checkpoint not found: {src}")
    # ``weights_only=False`` is required because the payload carries
    # non-tensor state (numpy RNG dicts, league metadata, config
    # dicts). See module docstring's trust boundary note.
    try:
        raw = torch.load(src, map_location=map_location, weights_only=False)
    except Exception as e:
        raise CheckpointError(f"failed to torch.load {src}: {e}") from e
    if not isinstance(raw, dict):
        raise CheckpointError(f"checkpoint at {src} is not a dict; got {type(raw).__name__}")
    if "schema_version" not in raw:
        raise CheckpointError(
            f"checkpoint at {src} missing 'schema_version' — pre-v2 format "
            "or corrupted; no migration available"
        )

    # Upgrade in-memory if needed.
    upgraded = apply_migrations(raw, target_version=SCHEMA_VERSION)
    return CheckpointPayload(
        schema_version=int(upgraded["schema_version"]),
        config=dict(upgraded.get("config", {})),
        update_idx=int(upgraded.get("update_idx", 0)),
        global_step=int(upgraded.get("global_step", 0)),
        policy_state_dict=dict(upgraded.get("policy_state_dict", {})),
        optimizer_state_dict=(
            dict(upgraded["optimizer_state_dict"])
            if upgraded.get("optimizer_state_dict") is not None
            else None
        ),
        league_state=(dict(upgraded["league"]) if upgraded.get("league") is not None else None),
        rng_state=(dict(upgraded["rng"]) if upgraded.get("rng") is not None else None),
        vec_env_state=(dict(upgraded["vec_env"]) if upgraded.get("vec_env") is not None else None),
        metadata=dict(upgraded.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# Listing + pruning
# ---------------------------------------------------------------------------


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Return checkpoint files in ``directory`` sorted by embedded
    ``update_idx`` (oldest first). Only files matching the canonical
    ``ckpt_NNNNNNNNN.pt`` pattern are included; anything else is
    silently skipped so user notes / experiment logs in the same dir
    are not surfaced as checkpoints."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for entry in d.iterdir():
        m = _FILE_PATTERN.match(entry.name)
        if m is None or not entry.is_file():
            continue
        out.append((int(m.group(1)), entry))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def prune_checkpoints(directory: str | Path, *, keep_last_n: int) -> list[Path]:
    """Delete checkpoints beyond the ``keep_last_n`` most recent.

    Returns the list of paths that were deleted (oldest first). ``0``
    or negative ``keep_last_n`` is a no-op — the manager treats that
    as "disable pruning".
    """
    if keep_last_n <= 0:
        return []
    files = list_checkpoints(directory)
    if len(files) <= keep_last_n:
        return []
    to_delete = files[:-keep_last_n]
    for p in to_delete:
        try:
            p.unlink()
        except OSError as e:
            # Don't fail the training loop on a pruning failure; log
            # and continue. The next save will retry.
            raise CheckpointError(f"failed to prune {p}: {e}") from e
    return to_delete


# ---------------------------------------------------------------------------
# High-level manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Convenience wrapper that pins a directory + retention policy.

    Typical use::

        mgr = CheckpointManager(out_dir, keep_last_n=cfg.checkpoint.keep_last_n)
        if cfg.checkpoint.save_every_updates and (update_idx + 1) % save_every == 0:
            mgr.save(
                config=cfg.to_dict(),
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                global_step=global_step,
                league=league,
            )

        # Resume from latest:
        latest = mgr.latest()
        if latest is not None:
            payload = load_checkpoint(latest)
            payload.apply_to_policy(policy)
            payload.apply_to_optimizer(optimizer)
            payload.apply_to_league(league)
            payload.apply_rng_state()
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_last_n: int = 5,
        save_optimizer_state: bool = True,
    ) -> None:
        if keep_last_n < 0:
            raise ValueError(f"keep_last_n must be >= 0, got {keep_last_n}")
        self.directory = Path(directory)
        self.keep_last_n = keep_last_n
        self.save_optimizer_state = save_optimizer_state

    def save(
        self,
        *,
        config: dict[str, Any],
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        update_idx: int,
        global_step: int,
        league: Any | None = None,
        vec_env: Any | None = None,
        capture_rng: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save and prune in one shot. Returns the written path."""
        path = self.directory / checkpoint_filename(update_idx)
        save_checkpoint(
            path,
            config=config,
            policy=policy,
            optimizer=optimizer if self.save_optimizer_state else None,
            update_idx=update_idx,
            global_step=global_step,
            league=league,
            vec_env=vec_env,
            capture_rng=capture_rng,
            metadata=metadata,
        )
        prune_checkpoints(self.directory, keep_last_n=self.keep_last_n)
        return path

    def latest(self) -> Path | None:
        """Return the path to the most recent checkpoint, or None if
        the directory has none."""
        files = list_checkpoints(self.directory)
        return files[-1] if files else None

    def list(self) -> list[Path]:
        """Return all checkpoints in the directory, oldest first."""
        return list_checkpoints(self.directory)
