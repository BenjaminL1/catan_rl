"""Content-addressed sidecar store for league snapshot state-dicts.

**Why this exists.** A v2 training checkpoint embeds the full 100-snapshot
league pool. At ~1.4M params * fp32 that pool is ~560 MB — 97% of a ~577 MB
checkpoint — and across two *consecutive* rolling saves ~99/100 snapshots are
byte-identical (only the newest snapshot changed). Embedding the pool in every
file therefore rewrites ~560 MB of near-duplicate bytes each save. That bloat
tripped the free-disk guard mid-run (v11 died at u749) and, worse, the guard
skipped the terminal save so the final weights were never written.

**The fix.** Persist each snapshot's ``state_dict`` ONCE to a content-addressed
sidecar file keyed by a hash of its tensor bytes; the checkpoint then stores only
a small ``ref`` (the hash) per snapshot. Identical snapshots across saves collapse
to a single store file (the dedup), so a rolling checkpoint drops from ~577 MB to
well under 25 MB and consecutive saves add essentially nothing.

**Content address.** The hash is computed over a deterministic serialisation of
the tensor bytes (sorted keys; per key: name, dtype, shape, contiguous CPU
``numpy().tobytes()``) — NOT over ``torch.save`` bytes, whose pickle framing is
not byte-stable across processes/versions. So two state-dicts with equal tensors
always hash equally regardless of how they were serialised.

**Durability.** Store files are written with the same tmp + fsync + ``os.replace``
+ dir-fsync discipline as :func:`catan_rl.checkpoint.manager.save_checkpoint`; a
write is skipped entirely when the target hash already exists (the dedup path).

**Trust boundary.** Store files carry pickled tensors and are loaded with
``torch.load(weights_only=False)`` — first-party files only, same contract as the
checkpoint payload.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

#: Default sidecar directory name, created next to the checkpoint files.
STORE_DIRNAME = "league_store"

_HASH_PATTERN_LEN = 64  # sha256 hex digest length


class LeagueStoreError(RuntimeError):
    """Raised when a referenced sidecar snapshot cannot be resolved
    (missing store file, corrupt store file)."""


def _tensor_bytes(value: Any) -> bytes:
    """Return the contiguous CPU byte-representation of a tensor leaf.

    Uses ``numpy().tobytes()`` on a contiguous CPU copy so the digest depends
    only on the numeric contents, not on storage strides or device tags.
    """
    t = value.detach().to("cpu").contiguous()
    return t.numpy().tobytes()


def snapshot_hash(state_dict: Mapping[str, Any]) -> str:
    """Return the sha256 content address for a snapshot ``state_dict``.

    Deterministic across processes: keys are visited in sorted order and each
    contributes its name, dtype, shape, and raw tensor bytes to the digest.
    Non-tensor leaves (rare in a policy state-dict) contribute their ``repr``.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode("utf-8"))
            h.update(b"\x00")
            h.update(str(tuple(value.shape)).encode("utf-8"))
            h.update(b"\x00")
            h.update(_tensor_bytes(value))
        else:
            # Defensive: policy state-dicts are all tensors, but a caller could
            # smuggle a scalar. repr keeps the address deterministic.
            h.update(repr(value).encode("utf-8"))
        h.update(b"\xff")
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory inode (see manager._fsync_dir)."""
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


class LeagueStore:
    """A content-addressed directory of snapshot ``state_dict`` files.

    Layout: ``<root>/<sha256>.pt``. Each file is a ``torch.save``-serialised
    plain state-dict (no wrapping payload) written atomically. The store is
    stateless beyond the directory itself — an instance is a thin handle.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, snap_hash: str) -> Path:
        """Return the on-disk path for a given content hash."""
        return self.root / f"{snap_hash}.pt"

    def exists(self, snap_hash: str) -> bool:
        """True if a store file for ``snap_hash`` is already present."""
        return self.path_for(snap_hash).is_file()

    def put(self, state_dict: Mapping[str, Any]) -> str:
        """Write ``state_dict`` to the store and return its content hash.

        Idempotent + deduplicating: if a file for the computed hash already
        exists the write is skipped entirely (the whole point — ~99/100
        snapshots are byte-identical across consecutive saves). The write uses
        tmp + fsync + ``os.replace`` + dir-fsync so a crash mid-write never
        leaves a truncated store file bound to a valid hash.
        """
        snap_hash = snapshot_hash(state_dict)
        dest = self.path_for(snap_hash)
        if dest.is_file():
            return snap_hash
        self.root.mkdir(parents=True, exist_ok=True)
        # Per-hash tmp suffix so concurrent puts of different snapshots don't
        # collide on the same tmp name.
        tmp = dest.with_suffix(f".pt.{os.getpid()}.tmp")
        import contextlib

        try:
            with open(tmp, "wb") as fh:
                torch.save({k: v for k, v in state_dict.items()}, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        _fsync_dir(self.root)
        return snap_hash

    def get(
        self, snap_hash: str, *, map_location: torch.device | str | None = "cpu"
    ) -> dict[str, Any]:
        """Load and return the ``state_dict`` for ``snap_hash``.

        Raises :class:`LeagueStoreError` if the store file is missing (a
        checkpoint referencing a snapshot whose sidecar was pruned/lost) or
        cannot be deserialised.
        """
        src = self.path_for(snap_hash)
        if not src.is_file():
            raise LeagueStoreError(
                f"league store entry not found: {src} — the checkpoint references a "
                "snapshot whose sidecar file is missing (deleted or never written)"
            )
        try:
            raw = torch.load(src, map_location=map_location, weights_only=False)
        except Exception as e:  # surface as a typed store error
            raise LeagueStoreError(f"failed to torch.load store entry {src}: {e}") from e
        if not isinstance(raw, dict):
            raise LeagueStoreError(
                f"league store entry {src} is not a dict; got {type(raw).__name__}"
            )
        return raw


def is_snapshot_hash(name: str) -> bool:
    """True if ``name`` looks like a ``<sha256>.pt`` store filename."""
    if not name.endswith(".pt"):
        return False
    stem = name[:-3]
    return len(stem) == _HASH_PATTERN_LEN and all(c in "0123456789abcdef" for c in stem)
