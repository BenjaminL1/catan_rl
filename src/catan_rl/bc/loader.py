"""PyTorch Dataset for BC training data.

Reads sharded NPZ files produced by :func:`catan_rl.bc.dataset.generate_dataset`,
indexes them into a flat global pair ID, decompresses **bounded row-ranges
(chunks)** on demand, converts numpy arrays to torch tensors at
``__getitem__`` time, applies D6 augmentation at configured probability,
and supports a stratified train/val split by ``game_id``.

Memory model (why this is not a naive ``np.load``)
--------------------------------------------------
The shards are ``np.savez_compressed`` ZIP archives. ``np.load(...,
mmap_mode="r")`` **silently ignores** ``mmap_mode`` for ``.npz`` and returns
an eager :class:`numpy.lib.npyio.NpzFile`: indexing any member (e.g.
``npz["obs/tile_representations"]``) decompresses the **entire** member into
RAM. At this dataset's scale one shard is ~15 GB uncompressed (the
``tile_representations`` member alone is ~5.5 GB), so a per-row
``npz[key][row]`` access would decompress ~5.5 GB to read one row — under a
shuffled multi-worker ``DataLoader`` that multiplies into an OOM SIGKILL, and
a single training step could never run.

The fix: never hold a whole shard. Each shard is divided into **chunks** of
``chunk_rows`` rows. A chunk is materialised by **streaming** each ``.npy``
member out of the ZIP and keeping only the chunk's row-range (the DEFLATE
stream is read sequentially and the pre-chunk prefix is discarded in bounded
pieces, so the transient never exceeds one chunk-slice, not one whole member).
Chunks live in a small **LRU** (``max_cached_chunks``, default 1). Paired with
:class:`ShardChunkBatchSampler` — which groups every batch inside a single
chunk and shuffles chunk order + rows within a chunk each epoch — the loader
touches one chunk at a time and reuses it for many rows before eviction. This
keeps peak RSS at a few GB (≈ one chunk + torch) while preserving stochasticity
(chunk-level + within-chunk shuffle).

``num_workers`` note: each worker process holds its **own** chunk LRU, so peak
RSS scales with the worker count. ``num_workers=0`` (single process) is the
safe default on a 16 GB machine; see ``configs/bc.yaml``.

Shard layout (from :mod:`catan_rl.bc.dataset`):
  * top-level keys: ``action``, ``belief_target``, ``z_disc``,
    ``game_id``, ``step_idx``, ``player_seat``, ``phase``, ``forced``;
  * ``obs/<key>`` for each obs schema entry;
  * ``mask/<key>`` for each of the 9 mask entries.

Each item from ``__getitem__`` is a dict::

    {
        "obs":   dict[str, Tensor],   # v2 obs schema
        "action": Tensor (6,) int64,
        "mask":   dict[str, Tensor],  # bool, 9 keys
        "belief_target": Tensor (5,) float32,
        "z_disc":        Tensor scalar float32,
    }

The default ``BcDataset`` constructs from a directory containing
``manifest.json`` + one or more ``shard_*.npz`` files. ``train_val_split``
returns two non-overlapping views split by ``game_id``.

D6 augmentation: when ``aug_prob > 0``, with that probability each
``__getitem__`` call samples a non-identity D6 element and applies
``apply_symmetry`` to the (obs, action, mask) triple. The state and
action are transformed together (the only correct way; state-only
augmentation under deterministic tiebreakers produces inconsistent
labels — faculty review correction).
"""

from __future__ import annotations

import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib import format as _npformat
from torch.utils.data import Dataset, Sampler

from catan_rl.augmentation import apply_symmetry, sample_d6_element

# ---------------------------------------------------------------------------
# Obs / mask member keys
# ---------------------------------------------------------------------------

_OBS_KEYS_FLOAT: tuple[str, ...] = (
    "tile_representations",
    "current_player_main",
    "next_player_main",
    "current_dev_counts",
    "next_played_dev_counts",
    "hex_features",
    "vertex_features",
    "edge_features",
    "global_features",
    "is_setup",
)
_OBS_KEYS_INT: tuple[str, ...] = ("opponent_kind", "opponent_policy_id")
_MASK_KEYS: tuple[str, ...] = (
    "type",
    "corner_settlement",
    "corner_city",
    "edge",
    "tile",
    "resource1_trade",
    "resource1_discard",
    "resource1_default",
    "resource2_default",
)

# Members materialised per chunk (everything ``__getitem__`` reads). ``game_id``
# is read separately at construction time; ``phase`` / ``step_idx`` /
# ``player_seat`` / ``forced`` are never consumed by the loader.
_CHUNK_MEMBERS: tuple[str, ...] = (
    *(f"obs/{k}" for k in _OBS_KEYS_FLOAT),
    *(f"obs/{k}" for k in _OBS_KEYS_INT),
    "action",
    "belief_target",
    "z_disc",
    *(f"mask/{k}" for k in _MASK_KEYS),
)

# Default rows per chunk. One chunk of all members is ~16.5 KB/row, so 200k
# rows ≈ 3.3 GB — decompressed via streaming (peak ≈ one chunk + torch, a few
# GB) and reused across all its batches by ``ShardChunkBatchSampler``.
DEFAULT_CHUNK_ROWS = 200_000

# ---------------------------------------------------------------------------
# Streaming .npy slice reader
# ---------------------------------------------------------------------------


def _read_member_row_slice(zf: zipfile.ZipFile, member: str, start: int, stop: int) -> np.ndarray:
    """Read rows ``[start:stop)`` of an ``.npy`` member with bounded memory.

    A ``savez_compressed`` member is a DEFLATE stream that cannot be seeked
    into, so the pre-``start`` prefix is decompressed and discarded in bounded
    pieces; only the requested row-range is materialised. The returned array is
    an owned (writable, C-contiguous) copy — it does not alias the ZIP buffer,
    so evicting the chunk actually frees memory.
    """
    with zf.open(member + ".npy") as fp:
        major, minor = _npformat.read_magic(fp)
        if (major, minor) == (1, 0):
            shape, fortran, dtype = _npformat.read_array_header_1_0(fp)
        elif (major, minor) == (2, 0):
            shape, fortran, dtype = _npformat.read_array_header_2_0(fp)
        else:  # pragma: no cover - datasets are written with format 1.0
            raise ValueError(f"unsupported .npy version {(major, minor)} in {member}")
        if fortran:  # pragma: no cover - savez writes C-order
            raise ValueError(f"fortran-order member not supported: {member}")

        row_shape = tuple(int(d) for d in shape[1:])
        row_nbytes = int(dtype.itemsize) * int(np.prod(row_shape, dtype=np.int64))

        # Discard the [0:start) prefix in bounded pieces.
        skip = int(start) * row_nbytes
        piece = 1 << 20
        while skip > 0:
            got = fp.read(min(skip, piece))
            if not got:  # pragma: no cover - defensive
                raise EOFError(f"unexpected EOF skipping prefix of {member}")
            skip -= len(got)

        n = int(stop - start) * row_nbytes
        buf = fp.read(n)
        if len(buf) != n:  # pragma: no cover - defensive
            raise EOFError(f"short read for {member}: got {len(buf)} want {n}")
        arr = np.frombuffer(buf, dtype=dtype).reshape((int(stop - start), *row_shape))
        return arr.copy()


# ---------------------------------------------------------------------------
# Shard registry (chunked, LRU-cached, streaming)
# ---------------------------------------------------------------------------


class _ShardRegistry:
    """Flat global pair ID → (shard, chunk, row); serves chunks from an LRU.

    Chunks are decompressed on demand via :func:`_read_member_row_slice` and
    held in an ``OrderedDict`` LRU bounded to ``max_cached_chunks`` entries.
    """

    def __init__(
        self,
        paths: list[Path],
        n_pairs_per_shard: list[int],
        *,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        max_cached_chunks: int = 1,
    ) -> None:
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
        if max_cached_chunks <= 0:
            raise ValueError(f"max_cached_chunks must be positive, got {max_cached_chunks}")
        self.paths = paths
        self.n_pairs = n_pairs_per_shard
        self.cum = np.cumsum([0] + n_pairs_per_shard)  # noqa: RUF005
        self.chunk_rows = int(chunk_rows)
        self.max_cached_chunks = int(max_cached_chunks)
        # Chunks per shard = ceil(n_pairs / chunk_rows).
        self.n_chunks = [(n + self.chunk_rows - 1) // self.chunk_rows for n in n_pairs_per_shard]
        self._chunk_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()

    def total(self) -> int:
        return int(self.cum[-1])

    def locate(self, global_idx: int) -> tuple[int, int]:
        """Map a flat index to (shard_idx, row_in_shard)."""
        if not 0 <= global_idx < self.total():
            raise IndexError(global_idx)
        shard_idx = int(np.searchsorted(self.cum[1:], global_idx, side="right"))
        row = global_idx - int(self.cum[shard_idx])
        return shard_idx, row

    def locate_chunk(self, global_idx: int) -> tuple[int, int, int]:
        """Map a flat index to (shard_idx, chunk_idx, row_within_chunk)."""
        shard_idx, row = self.locate(global_idx)
        chunk_idx = row // self.chunk_rows
        return shard_idx, chunk_idx, row - chunk_idx * self.chunk_rows

    def read_game_ids(self, shard_idx: int) -> np.ndarray:
        """Stream the full ``game_id`` member of a shard (small: int64 per pair)."""
        n = self.n_pairs[shard_idx]
        with zipfile.ZipFile(self.paths[shard_idx]) as zf:
            return _read_member_row_slice(zf, "game_id", 0, n)

    def get_chunk(self, shard_idx: int, chunk_idx: int) -> dict[str, np.ndarray]:
        """Return the (LRU-cached) dict of member arrays for one chunk."""
        key = (shard_idx, chunk_idx)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self._chunk_cache.move_to_end(key)
            return cached

        start = chunk_idx * self.chunk_rows
        stop = min(start + self.chunk_rows, self.n_pairs[shard_idx])
        if not 0 <= start < self.n_pairs[shard_idx]:
            raise IndexError(f"chunk {chunk_idx} out of range for shard {shard_idx}")

        chunk: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(self.paths[shard_idx]) as zf:
            for member in _CHUNK_MEMBERS:
                chunk[member] = _read_member_row_slice(zf, member, start, stop)

        # Evict oldest until we're within budget, then insert.
        while len(self._chunk_cache) >= self.max_cached_chunks:
            self._chunk_cache.popitem(last=False)
        self._chunk_cache[key] = chunk
        return chunk


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BcDataset(Dataset):
    """BC training Dataset over sharded NPZ shards (chunk-streamed).

    Args:
        data_dir: directory containing ``manifest.json`` + ``shard_*.npz``.
        aug_prob: D6 augmentation probability per item (default 0.5 per
            preflight E0.4). Set 0.0 for canonical-only.
        seed: per-dataset RNG seed for augmentation sampling.
        indices: optional subset of pair indices (used by
            ``train_val_split``); when ``None``, expose all pairs.
        chunk_rows: rows per decompressed chunk (memory/throughput knob;
            default :data:`DEFAULT_CHUNK_ROWS`). Smaller ⇒ lower peak RSS +
            more re-decompression.
        max_cached_chunks: LRU size in chunks (default 1). Peak RSS ≈
            ``max_cached_chunks`` chunks + torch; keep at 1 with a
            :class:`ShardChunkBatchSampler` (batches never span a chunk).
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        aug_prob: float = 0.5,
        seed: int = 0,
        indices: np.ndarray | None = None,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        max_cached_chunks: int = 1,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir)
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
        manifest = json.loads(manifest_path.read_text())

        paths: list[Path] = []
        n_pairs: list[int] = []
        for shard in manifest["shards"]:
            shard_path = data_dir / shard["shard"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Manifest references missing shard {shard_path}")
            paths.append(shard_path)
            n_pairs.append(int(shard["n_pairs"]))
        self._registry = _ShardRegistry(
            paths, n_pairs, chunk_rows=chunk_rows, max_cached_chunks=max_cached_chunks
        )
        self._manifest = manifest
        self._aug_prob = float(aug_prob)
        self._rng = np.random.default_rng(seed)

        # Pre-cache the flat game_id array for stratified splits. Streams only
        # the (small) game_id member per shard — never the heavy obs members.
        gids: list[np.ndarray] = []
        for s_idx in range(len(self._registry.paths)):
            gids.append(self._registry.read_game_ids(s_idx))
        self._game_ids = np.concatenate(gids) if gids else np.zeros(0, dtype=np.int64)

        if indices is None:
            self._indices = np.arange(self._registry.total(), dtype=np.int64)
        else:
            self._indices = np.asarray(indices, dtype=np.int64)
        # Restrict ``_game_ids`` to the view's pairs so consumers
        # (e.g. ``train_val_split``) see only this subset's game ids.
        self._game_ids = self._game_ids[self._indices]

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        flat = int(self._indices[idx])
        shard_idx, chunk_idx, row = self._registry.locate_chunk(flat)
        chunk = self._registry.get_chunk(shard_idx, chunk_idx)

        obs: dict[str, torch.Tensor] = {}
        for key in _OBS_KEYS_FLOAT:
            arr = np.asarray(chunk[f"obs/{key}"][row], dtype=np.float32)
            obs[key] = torch.from_numpy(arr)
        for key in _OBS_KEYS_INT:
            v = int(chunk[f"obs/{key}"][row])
            obs[key] = torch.tensor(v, dtype=torch.int64)

        action = torch.from_numpy(np.asarray(chunk["action"][row], dtype=np.int64))
        mask: dict[str, torch.Tensor] = {
            key: torch.from_numpy(np.asarray(chunk[f"mask/{key}"][row], dtype=bool))
            for key in _MASK_KEYS
        }
        belief_target = torch.from_numpy(np.asarray(chunk["belief_target"][row], dtype=np.float32))
        z_disc = torch.tensor(float(chunk["z_disc"][row]), dtype=torch.float32)

        # D6 augmentation. apply_symmetry expects batched inputs (B, ...);
        # we add a fake batch dim, apply, then squeeze.
        if self._aug_prob > 0.0 and self._rng.random() < self._aug_prob:
            g = sample_d6_element(self._rng, exclude_identity=True)
            obs_b = {k: v.unsqueeze(0) for k, v in obs.items()}
            mask_b = {k: v.unsqueeze(0) for k, v in mask.items()}
            action_b = action.unsqueeze(0)
            obs_b, action_b, mask_b = apply_symmetry(obs_b, action_b, mask_b, g)
            obs = {k: v.squeeze(0) for k, v in obs_b.items()}
            mask = {k: v.squeeze(0) for k, v in mask_b.items()}
            action = action_b.squeeze(0)

        return {
            "obs": obs,
            "action": action,
            "mask": mask,
            "belief_target": belief_target,
            "z_disc": z_disc,
        }

    # ------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------

    @classmethod
    def train_val_split(
        cls,
        data_dir: Path,
        *,
        val_pct: float = 0.10,
        seed: int = 0,
        train_aug_prob: float = 0.5,
        val_aug_prob: float = 0.0,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        max_cached_chunks: int = 1,
    ) -> tuple[BcDataset, BcDataset]:
        """Stratified split by ``game_id`` — no within-game leakage.

        ``val_aug_prob`` defaults to 0.0 because the val set is for
        NLL / WR evaluation; augmentation would inflate variance.
        """
        if not 0.0 < val_pct < 1.0:
            raise ValueError(f"val_pct must be in (0, 1), got {val_pct}")
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        all_game_ids: list[int] = []
        for s in manifest["shards"]:
            all_game_ids.extend(int(g) for g in s["game_ids"])
        unique_ids = sorted(set(all_game_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_ids)
        n_val_games = max(1, int(round(len(unique_ids) * val_pct)))  # noqa: RUF046
        val_set = set(unique_ids[:n_val_games])

        # Build the boolean masks over the flat pair index. This helper view
        # only reads game_id, so chunk_rows/cache are irrelevant here.
        tmp = cls(data_dir, aug_prob=0.0)
        gids = tmp._game_ids
        val_mask = np.isin(gids, list(val_set))
        train_idx = np.where(~val_mask)[0].astype(np.int64)
        val_idx = np.where(val_mask)[0].astype(np.int64)

        train = cls(
            data_dir,
            aug_prob=train_aug_prob,
            seed=seed,
            indices=train_idx,
            chunk_rows=chunk_rows,
            max_cached_chunks=max_cached_chunks,
        )
        val = cls(
            data_dir,
            aug_prob=val_aug_prob,
            seed=seed + 1,
            indices=val_idx,
            chunk_rows=chunk_rows,
            max_cached_chunks=max_cached_chunks,
        )
        return train, val


# ---------------------------------------------------------------------------
# Chunk-grouped batch sampler
# ---------------------------------------------------------------------------


class ShardChunkBatchSampler(Sampler[list[int]]):
    """Yield batches whose items all live in one shard-chunk.

    Grouping every batch inside a single chunk is what makes the ``max_cached``
    -bounded LRU effective: consecutive ``__getitem__`` calls hit the same
    chunk, so a chunk is decompressed once and reused for all its batches
    before eviction. Shuffling (chunk order across the epoch + rows within a
    chunk) preserves training stochasticity while keeping memory bounded.

    Args:
        dataset: the :class:`BcDataset` view to iterate (respects its
            ``indices`` subset — train/val).
        batch_size: rows per yielded batch.
        shuffle: reshuffle chunk order and within-chunk rows each epoch (train);
            ``False`` gives a deterministic chunk-ordered pass (val).
        seed: base RNG seed; combined with the epoch counter.
        drop_last: drop a trailing partial batch within each chunk.
    """

    def __init__(
        self,
        dataset: BcDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        reg = dataset._registry
        globals_ = dataset._indices  # global pair id per dataset position
        # Vectorised (shard, chunk) key per position.
        shard = np.searchsorted(reg.cum[1:], globals_, side="right")
        row = globals_ - reg.cum[shard]
        chunk = row // reg.chunk_rows
        max_chunks = max(reg.n_chunks) if reg.n_chunks else 1
        group_key = shard.astype(np.int64) * max_chunks + chunk

        # Group dataset positions (0..len-1) by (shard, chunk).
        positions = np.arange(globals_.shape[0], dtype=np.int64)
        order = np.argsort(group_key, kind="stable")
        sorted_keys = group_key[order]
        sorted_pos = positions[order]
        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        self._groups: list[np.ndarray] = [g for g in np.split(sorted_pos, boundaries) if g.size > 0]

        # Number of batches (fixed across epochs; depends only on group sizes).
        n = 0
        for grp in self._groups:
            if self.drop_last:
                n += grp.size // self.batch_size
            else:
                n += (grp.size + self.batch_size - 1) // self.batch_size
        self._length = n

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so ``shuffle`` produces a fresh permutation."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Any:
        rng = np.random.default_rng([self.seed, self.epoch])
        group_order = np.arange(len(self._groups))
        if self.shuffle:
            rng.shuffle(group_order)
        for gi in group_order:
            pos = self._groups[gi]
            if self.shuffle:
                pos = pos.copy()
                rng.shuffle(pos)
            for start in range(0, pos.size, self.batch_size):
                batch = pos[start : start + self.batch_size]
                if self.drop_last and batch.size < self.batch_size:
                    continue
                yield [int(i) for i in batch]


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def bc_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of per-item dicts into a batched dict.

    Mirrors what ``torch.utils.data.default_collate`` would do but
    handles the nested ``obs`` / ``mask`` sub-dicts explicitly.
    """
    obs_keys = list(batch[0]["obs"].keys())
    mask_keys = list(batch[0]["mask"].keys())
    return {
        "obs": {k: torch.stack([b["obs"][k] for b in batch]) for k in obs_keys},
        "action": torch.stack([b["action"] for b in batch]),
        "mask": {k: torch.stack([b["mask"][k] for b in batch]) for k in mask_keys},
        "belief_target": torch.stack([b["belief_target"] for b in batch]),
        "z_disc": torch.stack([b["z_disc"] for b in batch]),
    }
