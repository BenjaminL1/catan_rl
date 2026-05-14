"""PyTorch Dataset for BC training data.

Reads sharded NPZ files produced by :func:`catan_rl.bc.dataset.generate_dataset`,
indexes them lazily (mmap), converts numpy arrays to torch tensors at
``__getitem__`` time, applies D6 augmentation at configured probability,
and supports a stratified train/val split by ``game_id``.

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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from catan_rl.augmentation import apply_symmetry, sample_d6_element

# ---------------------------------------------------------------------------
# Shard registry
# ---------------------------------------------------------------------------


class _ShardRegistry:
    """Holds shard file paths and indexes a flat global pair ID → (shard, row).

    Loads shards lazily on first access via ``np.load`` with ``mmap_mode="r"``.
    """

    def __init__(self, paths: list[Path], n_pairs_per_shard: list[int]) -> None:
        self.paths = paths
        self.n_pairs = n_pairs_per_shard
        self.cum = np.cumsum([0] + n_pairs_per_shard)  # noqa: RUF005
        self._cache: dict[int, Any] = {}

    def total(self) -> int:
        return int(self.cum[-1])

    def locate(self, global_idx: int) -> tuple[int, int]:
        """Map a flat index to (shard_idx, row_in_shard)."""
        if not 0 <= global_idx < self.total():
            raise IndexError(global_idx)
        shard_idx = int(np.searchsorted(self.cum[1:], global_idx, side="right"))
        row = global_idx - int(self.cum[shard_idx])
        return shard_idx, row

    def get_shard(self, shard_idx: int) -> Any:
        if shard_idx not in self._cache:
            # ``np.load`` with mmap returns a lazy reader; per-key arrays
            # are mmap-backed and copies happen only at .copy() / index time.
            self._cache[shard_idx] = np.load(self.paths[shard_idx], mmap_mode="r")
        return self._cache[shard_idx]


# ---------------------------------------------------------------------------
# Dataset
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


class BcDataset(Dataset):
    """BC training Dataset over sharded NPZ shards.

    Args:
        data_dir: directory containing ``manifest.json`` + ``shard_*.npz``.
        aug_prob: D6 augmentation probability per item (default 0.5 per
            preflight E0.4). Set 0.0 for canonical-only.
        seed: per-dataset RNG seed for augmentation sampling.
        indices: optional subset of pair indices (used by
            ``train_val_split``); when ``None``, expose all pairs.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        aug_prob: float = 0.5,
        seed: int = 0,
        indices: np.ndarray | None = None,
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
        self._registry = _ShardRegistry(paths, n_pairs)
        self._manifest = manifest
        self._aug_prob = float(aug_prob)
        self._rng = np.random.default_rng(seed)

        # Pre-cache the flat game_id array for stratified splits.
        gids: list[np.ndarray] = []
        for s_idx in range(len(self._registry.paths)):
            gids.append(np.asarray(self._registry.get_shard(s_idx)["game_id"]))
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
        shard_idx, row = self._registry.locate(flat)
        shard = self._registry.get_shard(shard_idx)

        obs: dict[str, torch.Tensor] = {}
        for key in _OBS_KEYS_FLOAT:
            arr = np.asarray(shard[f"obs/{key}"][row], dtype=np.float32)
            obs[key] = torch.from_numpy(arr)
        for key in _OBS_KEYS_INT:
            v = int(shard[f"obs/{key}"][row])
            obs[key] = torch.tensor(v, dtype=torch.int64)

        action = torch.from_numpy(np.asarray(shard["action"][row], dtype=np.int64))
        mask: dict[str, torch.Tensor] = {
            key: torch.from_numpy(np.asarray(shard[f"mask/{key}"][row], dtype=bool))
            for key in _MASK_KEYS
        }
        belief_target = torch.from_numpy(np.asarray(shard["belief_target"][row], dtype=np.float32))
        z_disc = torch.tensor(float(shard["z_disc"][row]), dtype=torch.float32)

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

        # Build the boolean masks over the flat pair index.
        tmp = cls(data_dir, aug_prob=0.0)
        gids = tmp._game_ids
        val_mask = np.isin(gids, list(val_set))
        train_idx = np.where(~val_mask)[0].astype(np.int64)
        val_idx = np.where(val_mask)[0].astype(np.int64)

        train = cls(data_dir, aug_prob=train_aug_prob, seed=seed, indices=train_idx)
        val = cls(data_dir, aug_prob=val_aug_prob, seed=seed + 1, indices=val_idx)
        return train, val


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
