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
                    # schema v2: content-addressed sidecar reference. The
                    # snapshot's state_dict lives in <ckpt_dir>/league_store/
                    # <ref>.pt (deduped across saves). Schema v1 (fat) files —
                    # and the backward-compat load path — carry an inline
                    # "state_dict" instead; load_checkpoint resolves either.
                    "ref": str,             # sha256 of the state-dict tensors
                    "update_idx": int,
                    "snapshot_id": int,
                    "metadata": dict,
                },
                ...
            ],
            "next_snapshot_id": int,
            "opponent_stats": {snapshot_id: [p_hat, games]},  # optional (PFSP)
            "anchor": {                     # optional (frozen anchor); same shape
                "ref": str,                 # as a snapshot, kept with its
                "update_idx": int,          # ORIGINAL id so it round-trips
                "snapshot_id": int,         # without orphaning opponent_stats
                "metadata": dict,
            },
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

League sidecar (schema v2): the ~560 MB league pool is NO LONGER embedded
in the checkpoint. Each snapshot's ``state_dict`` is written once to a
content-addressed sidecar store (``<ckpt_dir>/league_store/<hash>.pt``, see
:mod:`catan_rl.checkpoint.league_store`) and the checkpoint holds only a
small ``ref`` per snapshot. Because ~99/100 snapshots are byte-identical
across consecutive saves, the store dedups them and a rolling checkpoint
drops from ~577 MB to < 25 MB. Each checkpoint also writes a sibling
``<ckpt>.refs.json`` manifest listing the store hashes it references;
:class:`CheckpointManager` garbage-collects store entries no surviving
manifest references (skipping GC entirely if any ``.pt`` in the dir lacks a
manifest, e.g. a legacy fat file). Loading a v1 fat checkpoint (inline
``state_dict``) still works unchanged — migration to v2 is one-way, on save.

Slim / policy-only saves: :func:`save_policy_only` writes a ~5.6 MB
policy-only checkpoint (``metadata["kind"]="policy_only"``, no optimizer /
league / RNG). Used both as the free-disk-guard fallback (so a run whose
save tripped the guard still writes its final weights) and by
:func:`bank_anchor` to re-save ``runs/anchors/*`` slim.

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

import contextlib
import json
import os
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.checkpoint.league_store import STORE_DIRNAME, LeagueStore
from catan_rl.checkpoint.migrations import apply_migrations

#: Current on-disk schema version. Bumped whenever any of the keys
#: above change shape; ship a v(N-1) -> v(N) migration in
#: :mod:`catan_rl.checkpoint.migrations` at the same time.
#: v2 (2026-07): league pool moved to the content-addressed sidecar store
#: (snapshots carry a ``ref`` instead of an inline ``state_dict``).
SCHEMA_VERSION = 2

#: Naming convention for checkpoint files inside an output directory.
#: ``update_idx`` is zero-padded so lexicographic sort matches numeric.
_FILE_PATTERN = re.compile(r"^ckpt_(\d{9})\.pt$")

#: Slim / policy-only checkpoint filename pattern. Deliberately OUTSIDE
#: ``_FILE_PATTERN`` so slim files are never pruned by the rolling window AND
#: never returned by ``latest()`` (a policy-only file must not be silently
#: resumed as if it carried optimizer / league / RNG state).
_SLIM_FILE_PATTERN = re.compile(r"^slim_ckpt_(\d{9})\.pt$")

#: Suffix appended to a checkpoint filename to form its league-store ref
#: manifest (a tiny JSON list of the store hashes the checkpoint references).
_REFS_MANIFEST_SUFFIX = ".refs.json"


class CheckpointError(RuntimeError):
    """Raised on a load/save error that the operator needs to act on
    (corrupt file, schema mismatch, missing migration)."""


def checkpoint_filename(update_idx: int) -> str:
    """Return the canonical filename for a checkpoint at ``update_idx``."""
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    return f"ckpt_{update_idx:09d}.pt"


def promotion_checkpoint_filename(update_idx: int) -> str:
    """Return the filename for a PERMANENT promotion-era checkpoint.

    Uses the ``promo_ckpt_`` prefix so it does NOT match :data:`_FILE_PATTERN`
    (``^ckpt_...``) — :func:`list_checkpoints` and :func:`prune_checkpoints`
    therefore never see it, making it exempt from the rolling ``keep_last_n``
    window by construction (no separate exemption bookkeeping needed)."""
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    return f"promo_ckpt_{update_idx:09d}.pt"


def slim_checkpoint_filename(update_idx: int) -> str:
    """Return the filename for a policy-only slim checkpoint.

    Uses the ``slim_ckpt_`` prefix (matched by :data:`_SLIM_FILE_PATTERN`, not
    :data:`_FILE_PATTERN`) so it is exempt from rolling pruning AND is never
    returned by :meth:`CheckpointManager.latest` — a slim file lacks optimizer
    / league / RNG state and must never be silently resumed as a full run."""
    if update_idx < 0:
        raise ValueError(f"update_idx must be >= 0, got {update_idx}")
    return f"slim_ckpt_{update_idx:09d}.pt"


def refs_manifest_path(ckpt_path: str | Path) -> Path:
    """Return the league-store ref-manifest path for a checkpoint file."""
    p = Path(ckpt_path)
    return p.with_name(p.name + _REFS_MANIFEST_SUFFIX)


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
    state: dict[str, Any] = {
        "numpy_state": np.random.get_state(legacy=False),
        "python_state": random.getstate(),
        "torch_state": torch.random.get_rng_state(),
    }
    # Rollout action sampling runs on the TRAINING device (MPS), whose per-device
    # generator the CPU torch_state above does NOT cover — additively capture it
    # (+ CUDA) so resume is bit-identical on-device. Mirrors snapshot_opponent.
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps_state"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_state"] = torch.cuda.get_rng_state_all()
    return state


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


def _capture_league_state(league: Any, store: LeagueStore) -> tuple[dict[str, Any], list[str]]:
    """Serialise a :class:`catan_rl.selfplay.league.League` instance (schema v2).

    The league holds a bounded deque of :class:`LeagueSnapshot`
    instances. Rather than embed each snapshot's ~5.6 MB ``state_dict`` in the
    checkpoint (the ~560 MB bloat this feature removes), we write it to the
    content-addressed sidecar ``store`` — deduped across saves — and keep only a
    small ``ref`` (its content hash) per snapshot. Each snapshot's ``state_dict``
    is already CPU per the league's invariant (see ``add_snapshot``); the store
    re-derives CPU bytes defensively.

    Returns ``(state, refs)`` where ``refs`` is the list of every store hash
    referenced (snapshots + anchor) so the caller can write a GC manifest.
    """
    refs: list[str] = []
    snapshots: list[dict[str, Any]] = []
    snapshot_deque = getattr(league, "_snapshots", None)
    if snapshot_deque is None:
        return {"snapshots": [], "next_snapshot_id": 0}, refs
    for snap in snapshot_deque:
        ref = store.put(snap.state_dict)
        refs.append(ref)
        snapshots.append(
            {
                "ref": ref,
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
    # Frozen anchor (additive; absent in pre-anchor checkpoints). Persisted with
    # its ORIGINAL stable id so its opponent_stats key + obs embedding slot stay
    # put on resume — the drift guard must survive a restart, not vanish.
    anchor = getattr(league, "_anchor", None)
    if anchor is not None:
        anchor_ref = store.put(anchor.state_dict)
        refs.append(anchor_ref)
        state["anchor"] = {
            "ref": anchor_ref,
            "update_idx": int(anchor.update_idx),
            "snapshot_id": int(anchor.snapshot_id),
            "metadata": dict(anchor.metadata),
            # Auto-re-anchor bookkeeping (additive; absent in pre-feature
            # checkpoints -> restored to the cold-start zero/never state).
            "reanchor_streak": int(getattr(league, "_reanchor_streak", 0)),
            "last_promote_update": int(getattr(league, "_last_promote_update", -1)),
            "n_promotions": int(getattr(league, "_n_promotions", 0)),
        }
    return state, refs


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
    league_store: LeagueStore | None = None,
) -> Path:
    """Write a checkpoint atomically.

    ``policy`` and ``optimizer`` state dicts are CPU-cloned. ``league``,
    if provided, is captured via :func:`_capture_league_state` — its snapshot
    pool is written to the content-addressed sidecar ``league_store`` (defaults
    to ``<dest.parent>/league_store``) rather than embedded, and a sibling
    ``<dest>.refs.json`` manifest records the store hashes referenced (for GC).
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
    league_state: dict[str, Any] | None = None
    refs: list[str] = []
    if league is not None:
        store = league_store or LeagueStore(dest.parent / STORE_DIRNAME)
        league_state, refs = _capture_league_state(league, store)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "update_idx": int(update_idx),
        "global_step": int(global_step),
        "policy_state_dict": _state_dict_to_cpu(policy.state_dict()),
        "optimizer_state_dict": (
            _state_dict_to_cpu(optimizer.state_dict()) if optimizer is not None else None
        ),
        "league": league_state,
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
    # League-store GC manifest: written only when a league was sidecar'd. Records
    # exactly the store hashes this checkpoint depends on so CheckpointManager can
    # later delete store entries referenced by no surviving manifest. Absence of a
    # manifest for a .pt (e.g. a legacy fat file, or a league-less save) makes GC
    # conservatively skip — see CheckpointManager._gc_league_store.
    if league is not None:
        _write_refs_manifest(dest, refs)
    return dest


def _write_refs_manifest(ckpt_path: Path, refs: list[str]) -> None:
    """Atomically write the ``<ckpt>.refs.json`` league-store GC manifest."""
    manifest = refs_manifest_path(ckpt_path)
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    import contextlib

    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"refs": list(refs)}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, manifest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(manifest.parent)


def _read_refs_manifest(ckpt_path: Path) -> list[str] | None:
    """Return the referenced store hashes for a checkpoint, or ``None`` if the
    checkpoint has no manifest (legacy fat file / league-less save)."""
    manifest = refs_manifest_path(ckpt_path)
    if not manifest.is_file():
        return None
    try:
        with open(manifest, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    refs = data.get("refs")
    if not isinstance(refs, list):
        return None
    return [str(r) for r in refs]


def save_policy_only(
    path: str | Path,
    *,
    config: dict[str, Any],
    policy: torch.nn.Module,
    update_idx: int,
    global_step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a ~5.6 MB policy-only checkpoint (no optimizer / league / RNG).

    Same atomic tmp + fsync + ``os.replace`` machinery as
    :func:`save_checkpoint`, but every heavy section is ``None`` and
    ``metadata["kind"]`` is stamped ``"policy_only"``. Two uses:

    * the free-disk-guard fallback — a ~5.6 MB write fits under a GB-scale
      threshold, so a run whose full save tripped the guard still persists its
      final weights before exiting;
    * :func:`bank_anchor`, which re-saves ``runs/anchors/*`` slim.

    Writes NO league-store manifest (there is no league to reference)."""
    meta = dict(metadata or {})
    meta["kind"] = "policy_only"
    return save_checkpoint(
        path,
        config=config,
        policy=policy,
        optimizer=None,
        update_idx=update_idx,
        global_step=global_step,
        league=None,
        vec_env=None,
        capture_rng=False,
        metadata=meta,
    )


def bank_anchor(src: str | Path, dest: str | Path | None = None) -> Path:
    """Re-save a checkpoint as a policy-only slim file (for ``runs/anchors/*``).

    Loads ``src`` (fat or slim, any schema), then writes just its policy weights
    via :func:`save_policy_only`. ``dest`` defaults to ``src`` (in-place slim
    rewrite). Optimizer / league / RNG are dropped — an anchor is only ever used
    as a frozen reference opponent / eval baseline, never resumed. The original
    ``config`` + ``update_idx`` + ``global_step`` are preserved so lineage
    bookkeeping survives."""
    src_path = Path(src)
    dest_path = Path(dest) if dest is not None else src_path
    payload = load_checkpoint(src_path)

    # Reconstruct a bare nn.Module-like holder is unnecessary: save_policy_only
    # calls policy.state_dict(), so wrap the loaded state dict in a shim.
    class _StateDictShim(torch.nn.Module):
        def __init__(self, sd: dict[str, Any]) -> None:
            super().__init__()
            self._sd = sd

        def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override, unused-ignore]
            return self._sd

    shim = _StateDictShim(payload.policy_state_dict)
    return save_policy_only(
        dest_path,
        config=payload.config,
        policy=shim,
        update_idx=payload.update_idx,
        global_step=payload.global_step,
        metadata={"banked_from": str(src_path)},
    )


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
                    # Force CPU: load_checkpoint(map_location=<train device>) moved
                    # these onto e.g. MPS, but the league's invariant is CPU-resident
                    # snapshots (else maxlen=100 pool = ~470MB on the train device).
                    state_dict=_state_dict_to_cpu(snap_dict["state_dict"]),
                    update_idx=int(snap_dict["update_idx"]),
                    snapshot_id=int(snap_dict["snapshot_id"]),
                    metadata=dict(snap_dict.get("metadata", {})),
                )
            )
        next_id = int(self.league_state.get("next_snapshot_id", 0))
        # Defensive max: in case the saved cursor somehow lags behind
        # the highest live id (shouldn't happen, but cheap to guard).
        # Frozen anchor (additive; absent in pre-anchor checkpoints). Restore it
        # with its ORIGINAL stable id — NOT via set_anchor, which would mint a
        # fresh id and orphan the anchor's opponent_stats + shift its obs
        # embedding slot. The checkpoint's anchor (with accumulated EMA) wins
        # over any path-installed one.
        anchor_state = self.league_state.get("anchor")
        if anchor_state is not None:
            league._anchor = LeagueSnapshot(
                state_dict=_state_dict_to_cpu(anchor_state["state_dict"]),  # CPU invariant
                update_idx=int(anchor_state["update_idx"]),
                snapshot_id=int(anchor_state["snapshot_id"]),
                metadata=dict(anchor_state.get("metadata", {})),
            )
            # Auto-re-anchor counters (additive; .get defaults make pre-feature
            # checkpoints restore to the cold-start state without error).
            league._reanchor_streak = int(anchor_state.get("reanchor_streak", 0))
            league._last_promote_update = int(anchor_state.get("last_promote_update", -1))
            league._n_promotions = int(anchor_state.get("n_promotions", 0))
            # The promotion window tracks outcomes vs the CURRENT anchor; the
            # checkpoint's anchor just replaced whatever this league had, so any
            # accrued window outcomes are against the wrong opponent. Clear it
            # (the window is deliberately not checkpointed — the gate waits for
            # a fresh window after restore; see League.__init__).
            league._anchor_window.clear()
        live_ids = [snap.snapshot_id for snap in deque_]
        if getattr(league, "_anchor", None) is not None:
            live_ids.append(league._anchor.snapshot_id)
        max_live = max(live_ids, default=-1)
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
            # torch.set_rng_state requires a CPU uint8 ByteTensor. load_checkpoint
            # passes map_location=<train device>, which moves EVERY payload tensor
            # (including this one) onto e.g. MPS — set_rng_state then rejects it
            # ("RNG state must be a torch.ByteTensor"). Force it back to CPU uint8.
            # (MPS-only; CPU resume kept it on-host so this was dormant in CI.)
            torch.random.set_rng_state(torch_state.to(device="cpu", dtype=torch.uint8))
        # Device generators (additive; absent in pre-device-rng checkpoints). Each
        # ByteTensor must be CPU uint8 (load_checkpoint's map_location moved them).
        mps_state = self.rng_state.get("mps_state")
        if mps_state is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_state.to(device="cpu", dtype=torch.uint8))
        cuda_state = self.rng_state.get("cuda_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.to(device="cpu", dtype=torch.uint8) for s in cuda_state]
            )

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


def _resolve_league_refs(
    league_dict: dict[str, Any],
    store: LeagueStore,
    *,
    map_location: torch.device | str | None,
) -> None:
    """Materialise sidecar ``ref`` snapshots into inline ``state_dict`` entries.

    Mutates ``league_dict`` in place: any snapshot / anchor entry that carries a
    ``ref`` but no ``state_dict`` (a schema-v2 slim save) has its state-dict
    loaded from ``store`` and inlined, so downstream :meth:`apply_to_league`
    sees a uniform inline shape. Entries that already carry ``state_dict`` (a
    migrated v1 fat file — the backward-compat path) are left untouched. A
    missing store file raises :class:`CheckpointError`."""

    def _inline(entry: dict[str, Any]) -> None:
        if "state_dict" in entry and entry["state_dict"] is not None:
            return
        ref = entry.get("ref")
        if ref is None:
            raise CheckpointError(
                "league snapshot entry has neither 'state_dict' nor 'ref' — "
                "corrupt or truncated checkpoint"
            )
        from catan_rl.checkpoint.league_store import LeagueStoreError

        try:
            entry["state_dict"] = store.get(str(ref), map_location=map_location)
        except LeagueStoreError as e:
            raise CheckpointError(str(e)) from e

    for snap in league_dict.get("snapshots", []):
        _inline(snap)
    anchor = league_dict.get("anchor")
    if anchor is not None:
        _inline(anchor)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
    league_store_dir: str | Path | None = None,
) -> CheckpointPayload:
    """Load a checkpoint and run any pending migrations to the current
    :data:`SCHEMA_VERSION`.

    ``map_location`` is forwarded to ``torch.load``. Default ``"cpu"``
    matches the save-side CPU cloning; pass an explicit device only if
    you know what you're doing and want to skip a CPU->device copy on
    apply.

    ``league_store_dir`` overrides where sidecar snapshot refs are resolved
    from (defaults to ``<path.parent>/league_store``). Fat v1 checkpoints carry
    inline snapshots and never touch the store.
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
    # Resolve sidecar league refs (schema v2 slim saves) into inline state_dicts
    # BEFORE constructing the payload, so apply_to_league is layout-agnostic.
    league_dict = upgraded.get("league")
    if league_dict is not None:
        store = LeagueStore(
            league_store_dir if league_store_dir is not None else src.parent / STORE_DIRNAME
        )
        # Mutates the nested snapshot/anchor dicts in place; the payload's
        # shallow dict(upgraded["league"]) shares those same nested references.
        _resolve_league_refs(league_dict, store, map_location=map_location)
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
        # Delete the checkpoint's league-store ref manifest alongside it so a
        # pruned checkpoint stops pinning its store entries (GC then reclaims
        # any snapshot no surviving checkpoint references).
        manifest = refs_manifest_path(p)
        with contextlib.suppress(OSError):
            manifest.unlink(missing_ok=True)
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
        self._gc_league_store()
        return path

    def _gc_league_store(self) -> None:
        """Delete league-store entries referenced by no surviving checkpoint.

        Reads every ``<ckpt>.refs.json`` manifest in the directory, unions their
        referenced hashes, and deletes any ``<hash>.pt`` in the sidecar store not
        in that set. CONSERVATIVE: if ANY ``*.pt`` checkpoint in the directory
        (rolling / promo) lacks a manifest — e.g. a legacy fat file, or a
        checkpoint written before this feature — GC is skipped entirely, because
        an un-manifested checkpoint may reference store entries we can't see.
        Slim (disk-trip) checkpoints are exempt from this bail: they are
        policy-only and reference nothing in the store, so their (expected)
        missing manifest is ignored rather than disabling GC.
        Promotion checkpoints keep permanent manifests, so their (deduped)
        snapshots stay pinned for the life of the run."""
        store_dir = self.directory / STORE_DIRNAME
        if not store_dir.is_dir():
            return
        referenced: set[str] = set()
        for entry in self.directory.iterdir():
            if not entry.is_file() or entry.suffix != ".pt":
                continue
            if _SLIM_FILE_PATTERN.match(entry.name):
                # Slim (disk-trip) checkpoints are policy-only (league=None) and
                # by design reference NOTHING in the sidecar store, so they carry
                # no manifest. Skip them rather than treating the absent manifest
                # as unknown provenance — otherwise a single leftover slim file
                # (e.g. from a prior disk-abort) would permanently disable GC and
                # let the store grow unbounded.
                continue
            refs = _read_refs_manifest(entry)
            if refs is None:
                # An un-manifested checkpoint of unknown provenance — bail out
                # rather than risk deleting a store entry it depends on.
                return
            referenced.update(refs)
        from catan_rl.checkpoint.league_store import is_snapshot_hash

        for entry in store_dir.iterdir():
            if not entry.is_file() or not is_snapshot_hash(entry.name):
                continue
            if entry.name[:-3] not in referenced:  # strip ".pt"
                with contextlib.suppress(OSError):
                    entry.unlink()

    def save_slim(
        self,
        *,
        config: dict[str, Any],
        policy: torch.nn.Module,
        update_idx: int,
        global_step: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a policy-only ``slim_ckpt_NNNNNNNNN.pt`` (the disk-trip fallback).

        Uses the ``slim_ckpt_`` prefix so the file is never pruned by the rolling
        window and never returned by :meth:`latest` — it lacks optimizer / league
        / RNG state and must not be silently resumed as a full run. Returns the
        written path."""
        path = self.directory / slim_checkpoint_filename(update_idx)
        return save_policy_only(
            path,
            config=config,
            policy=policy,
            update_idx=update_idx,
            global_step=global_step,
            metadata=metadata,
        )

    def save_promotion(
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
        """Save a PERMANENT promotion-era checkpoint (``promo_ckpt_NNNNNNNNN.pt``).

        Same payload + atomic-write machinery as :meth:`save`, but written under
        the ``promo_ckpt_`` prefix and deliberately NOT pruned: these are the
        candidate-selection insurance for a run whose length nobody knows in
        advance (v9's crowned candidate was a promotion-era ckpt, not the final
        one). The rolling pruner never touches them because they don't match the
        ``ckpt_`` glob, so no exemption call is needed here."""
        path = self.directory / promotion_checkpoint_filename(update_idx)
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
        return path

    def latest(self) -> Path | None:
        """Return the path to the most recent checkpoint, or None if
        the directory has none."""
        files = list_checkpoints(self.directory)
        return files[-1] if files else None

    def list(self) -> list[Path]:
        """Return all checkpoints in the directory, oldest first."""
        return list_checkpoints(self.directory)
