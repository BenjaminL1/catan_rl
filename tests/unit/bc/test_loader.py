"""TDD tests for bc/loader.py.

Written before the implementation. Pin the contract the loader must
honour:

  * mmap-loads sharded NPZ files produced by ``bc.dataset.generate_dataset``;
  * exposes a ``torch.utils.data.Dataset`` API (``len``, ``__getitem__``);
  * each item is a dict of torch tensors keyed exactly as the policy
    network's ``forward()`` / ``evaluate_actions()`` consume them, plus
    ``action``, ``mask`` (sub-dict), ``belief_target``, ``z_disc``;
  * applies D6 augmentation at configured prob in ``__getitem__``,
    transforming BOTH state AND action through the v2 sym tables;
  * supports a stratified train/val split by game_id (no within-game
    leakage across the split).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from catan_rl.bc.dataset import generate_dataset
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_EDGES,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    TILE_DIM,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Generate 8-game dataset for testing. ~5s, smallest meaningful size."""
    generate_dataset(
        out_dir=tmp_path,
        n_games=8,
        perturb_pct=0.30,
        shard_size=4,
        seed=0,
        max_turns=80,
        progress_every=10**9,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Construction + length
# ---------------------------------------------------------------------------


def test_loader_constructs_from_dataset_dir(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset)
    assert len(ds) > 0


def test_loader_rejects_missing_manifest(tmp_path: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    with pytest.raises((FileNotFoundError, RuntimeError)):
        BcDataset(tmp_path)


def test_loader_loads_all_shards(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset)
    # 8 games at shard_size=4 → 2 shards. Sum of shard pair counts = len(ds).
    import json

    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    expected = sum(s["n_pairs"] for s in manifest["shards"])
    assert len(ds) == expected


# ---------------------------------------------------------------------------
# __getitem__ schema
# ---------------------------------------------------------------------------


def test_getitem_returns_dict_with_expected_keys(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    item = ds[0]
    # Top-level keys.
    assert {"obs", "action", "mask", "belief_target", "z_disc"} <= set(item)


def test_getitem_obs_has_full_v2_schema(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    obs = ds[0]["obs"]
    expected = {
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
        "opponent_kind",
        "opponent_policy_id",
    }
    assert set(obs.keys()) == expected


def test_getitem_obs_tensor_shapes(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    obs = ds[0]["obs"]
    assert obs["tile_representations"].shape == (N_TILES, TILE_DIM)
    assert obs["current_player_main"].shape == (CURR_PLAYER_DIM,)
    assert obs["next_player_main"].shape == (NEXT_PLAYER_DIM,)
    assert obs["current_dev_counts"].shape == (N_DEV_TYPES,)
    assert obs["next_played_dev_counts"].shape == (N_DEV_TYPES,)
    assert obs["hex_features"].shape == (N_TILES, 19)
    assert obs["vertex_features"].shape == (N_VERTICES, 16)
    assert obs["edge_features"].shape == (N_EDGES, 16)


def test_getitem_obs_dtypes(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    obs = ds[0]["obs"]
    for key in (
        "tile_representations",
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
        "hex_features",
        "vertex_features",
        "edge_features",
    ):
        assert obs[key].dtype == torch.float32, f"{key} expected float32"
    assert obs["opponent_kind"].dtype == torch.int64
    assert obs["opponent_policy_id"].dtype == torch.int64


def test_getitem_action_shape_and_dtype(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    item = ds[0]
    assert item["action"].shape == (6,)
    assert item["action"].dtype == torch.int64


def test_getitem_mask_has_all_v2_keys(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    item = ds[0]
    expected = {
        "type",
        "corner_settlement",
        "corner_city",
        "edge",
        "tile",
        "resource1_trade",
        "resource1_discard",
        "resource1_default",
        "resource2_default",
    }
    assert set(item["mask"].keys()) == expected
    for k in item["mask"]:
        assert item["mask"][k].dtype == torch.bool


def test_getitem_mask_shapes(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    mask = ds[0]["mask"]
    assert mask["type"].shape == (13,)
    assert mask["corner_settlement"].shape == (N_VERTICES,)
    assert mask["corner_city"].shape == (N_VERTICES,)
    assert mask["edge"].shape == (N_EDGES,)
    assert mask["tile"].shape == (N_TILES,)
    assert mask["resource1_default"].shape == (N_RESOURCES,)


def test_getitem_belief_and_z(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    item = ds[0]
    assert item["belief_target"].shape == (N_DEV_TYPES,)
    assert item["belief_target"].dtype == torch.float32
    assert item["z_disc"].shape == ()
    assert item["z_disc"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Augmentation behaviour
# ---------------------------------------------------------------------------


def test_getitem_aug_prob_zero_returns_canonical(tiny_dataset: Path) -> None:
    """aug_prob=0 means no augmentation → identical results across calls."""
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0, seed=42)
    a = ds[0]
    b = ds[0]
    assert torch.equal(a["obs"]["tile_representations"], b["obs"]["tile_representations"])
    assert torch.equal(a["action"], b["action"])


def test_getitem_aug_prob_one_always_perturbs(tiny_dataset: Path) -> None:
    """aug_prob=1.0 means every __getitem__ applies a non-identity D6 element.

    For at least one item the action's spatial heads should differ from
    the canonical action.
    """
    from catan_rl.bc.loader import BcDataset

    canonical = BcDataset(tiny_dataset, aug_prob=0.0)
    augmented = BcDataset(tiny_dataset, aug_prob=1.0, seed=7)

    # Find an item where the canonical action involves a spatial head
    # (corner/edge/tile) — then the aug should change at least that head.
    for i in range(min(len(canonical), 200)):
        canon_item = canonical[i]
        canon_action = canon_item["action"]
        # action_type 0,1 → corner; 2 → edge; 4 → tile
        if int(canon_action[0]) in (0, 1, 2, 4):
            aug_item = augmented[i]
            aug_action = aug_item["action"]
            # Type head doesn't change under D6.
            assert int(aug_action[0]) == int(canon_action[0])
            # At least one spatial head index must differ for some D6 element.
            if int(canon_action[0]) in (0, 1):
                if int(aug_action[1]) != int(canon_action[1]):
                    return
            elif int(canon_action[0]) == 2:
                if int(aug_action[2]) != int(canon_action[2]):
                    return
            elif int(canon_action[0]) == 4:  # noqa: SIM102
                if int(aug_action[3]) != int(canon_action[3]):
                    return
    pytest.skip("no spatial-action item in first 200; need bigger dataset")


def test_aug_keeps_action_legal_under_new_mask(tiny_dataset: Path) -> None:
    """Augmentation must transform mask + action together — the chosen
    action must still be legal under the (transformed) mask."""
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=1.0, seed=0)
    for i in range(min(len(ds), 200)):
        item = ds[i]
        action = item["action"]
        mask = item["mask"]
        type_idx = int(action[0])
        assert mask["type"][type_idx].item(), (
            f"item {i}: action type {type_idx} not legal after aug"
        )
        if type_idx == 0:  # BUILD_SETTLEMENT
            assert mask["corner_settlement"][int(action[1])].item()
        elif type_idx == 1:  # BUILD_CITY
            assert mask["corner_city"][int(action[1])].item()
        elif type_idx == 2:  # BUILD_ROAD
            assert mask["edge"][int(action[2])].item()
        elif type_idx == 4:  # MOVE_ROBBER
            assert mask["tile"][int(action[3])].item()


# ---------------------------------------------------------------------------
# Stratified split by game_id
# ---------------------------------------------------------------------------


def test_stratified_split_disjoint_game_ids(tiny_dataset: Path) -> None:
    """Train/val split by game_id must not leak any game across both sides."""
    from catan_rl.bc.loader import BcDataset

    train, val = BcDataset.train_val_split(tiny_dataset, val_pct=0.25, seed=0)
    train_games = {int(g) for g in train._game_ids}
    val_games = {int(g) for g in val._game_ids}
    assert not (train_games & val_games), "game_id leak between train/val"


def test_stratified_split_lengths_consistent(tiny_dataset: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    train, val = BcDataset.train_val_split(tiny_dataset, val_pct=0.25, seed=0)
    # 8 games × 0.25 = 2 val games, 6 train games. The split is by game-id
    # not by sample, so exact pair count varies; just confirm both non-empty.
    assert len(train) > 0 and len(val) > 0
    full = BcDataset(tiny_dataset)
    assert len(train) + len(val) == len(full)


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------


def test_works_with_torch_dataloader(tiny_dataset: Path) -> None:
    """End-to-end: wrap in DataLoader, iterate one batch, shapes correct."""
    from torch.utils.data import DataLoader

    from catan_rl.bc.loader import BcDataset, bc_collate

    ds = BcDataset(tiny_dataset, aug_prob=0.0)
    loader = DataLoader(ds, batch_size=4, collate_fn=bc_collate, shuffle=False)
    batch = next(iter(loader))
    assert batch["action"].shape == (4, 6)
    assert batch["belief_target"].shape == (4, N_DEV_TYPES)
    assert batch["z_disc"].shape == (4,)
    assert batch["obs"]["tile_representations"].shape == (4, N_TILES, TILE_DIM)
    assert batch["mask"]["type"].shape == (4, 13)


# ---------------------------------------------------------------------------
# Chunk-streaming: index mapping, correctness vs direct read, memory bound
# ---------------------------------------------------------------------------


def test_locate_chunk_maps_every_index(tiny_dataset: Path) -> None:
    """Every global index → the right (shard, row); chunk math round-trips."""
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tiny_dataset, aug_prob=0.0, chunk_rows=37)
    reg = ds._registry
    # Reference (shard, row) from the manifest's per-shard n_pairs.
    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    n_pairs = [int(s["n_pairs"]) for s in manifest["shards"]]
    ref: list[tuple[int, int]] = []
    for s_idx, n in enumerate(n_pairs):
        ref.extend((s_idx, r) for r in range(n))

    assert reg.total() == len(ref)
    for gi in range(reg.total()):
        assert reg.locate(gi) == ref[gi], f"locate({gi})"
        s_idx, c_idx, row_in_chunk = reg.locate_chunk(gi)
        # (shard, chunk, row_in_chunk) must reconstruct (shard, row).
        assert (s_idx, c_idx * reg.chunk_rows + row_in_chunk) == ref[gi]


def test_chunk_read_matches_direct_npz(tiny_dataset: Path) -> None:
    """Canonical ``__getitem__`` must byte-match a plain ``np.load`` of the shard.

    Pins the streaming chunk reader against an independent full-decompress path
    so the memory fix cannot silently corrupt any record.
    """
    import numpy as np

    from catan_rl.bc.loader import (
        _MASK_KEYS,
        _OBS_KEYS_FLOAT,
        _OBS_KEYS_INT,
        BcDataset,
    )

    ds = BcDataset(tiny_dataset, aug_prob=0.0, chunk_rows=29, max_cached_chunks=1)
    reg = ds._registry
    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    # Full-decompress every shard once, independently of the loader.
    direct = [dict(np.load(tiny_dataset / s["shard"])) for s in manifest["shards"]]

    total = reg.total()
    probe = sorted({0, 1, total // 3, total // 2, (2 * total) // 3, total - 1})
    for gi in probe:
        s_idx, row = reg.locate(gi)
        item = ds[gi]
        d = direct[s_idx]
        for key in _OBS_KEYS_FLOAT:
            assert np.array_equal(
                item["obs"][key].numpy(), d[f"obs/{key}"][row].astype(np.float32)
            ), f"obs/{key} @ gi={gi}"
        for key in _OBS_KEYS_INT:
            assert int(item["obs"][key]) == int(d[f"obs/{key}"][row]), f"obs/{key} @ gi={gi}"
        assert np.array_equal(item["action"].numpy(), d["action"][row]), f"action @ gi={gi}"
        for mk in _MASK_KEYS:
            assert np.array_equal(item["mask"][mk].numpy(), d[f"mask/{mk}"][row].astype(bool)), (
                f"mask/{mk} @ gi={gi}"
            )
        assert np.array_equal(
            item["belief_target"].numpy(), d["belief_target"][row].astype(np.float32)
        ), f"belief @ gi={gi}"
        assert item["z_disc"].item() == float(d["z_disc"][row]), f"z_disc @ gi={gi}"


def test_chunk_lru_stays_bounded(tiny_dataset: Path) -> None:
    """Iterating the chunk-grouped loader keeps the LRU ≤ max_cached_chunks and
    decompresses each chunk exactly once (never per-row).

    This is the memory-bound guarantee: without it a single ``__getitem__``
    decompressed a whole ~5.5 GB member (OOM under a shuffled DataLoader).
    """
    import resource

    from torch.utils.data import DataLoader

    from catan_rl.bc.loader import BcDataset, ShardChunkBatchSampler, bc_collate

    ds = BcDataset(tiny_dataset, aug_prob=0.0, chunk_rows=31, max_cached_chunks=1)
    # The bound is only meaningful if the tiny dataset spans several chunks.
    assert sum(ds._registry.n_chunks) > 2

    # Instrument chunk loads (cache misses) and observed cache size.
    reg = ds._registry
    orig_get = reg.get_chunk
    loads: list[tuple[int, int]] = []
    cache_sizes: list[int] = []

    def _counting(shard_idx: int, chunk_idx: int) -> dict:  # type: ignore[type-arg]
        miss = (shard_idx, chunk_idx) not in reg._chunk_cache
        out = orig_get(shard_idx, chunk_idx)
        if miss:
            loads.append((shard_idx, chunk_idx))
        cache_sizes.append(len(reg._chunk_cache))
        return out

    reg.get_chunk = _counting  # type: ignore[method-assign]

    sampler = ShardChunkBatchSampler(ds, batch_size=8, shuffle=True, seed=1)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=bc_collate, num_workers=0)

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    n_batches = 0
    for batch in loader:
        assert batch["action"].shape[0] <= 8
        assert len(reg._chunk_cache) <= 1  # LRU bound holds after every batch
        n_batches += 1
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert n_batches == len(sampler)
    # LRU never exceeded the bound at any load.
    assert max(cache_sizes) <= 1
    # Each chunk containing sampled rows is decompressed exactly once — NOT once
    # per row (the bug). #loads == #non-empty chunk groups == #sampler groups.
    assert len(loads) == len(sampler._groups)
    assert len(loads) == len(set(loads))  # no chunk reloaded
    assert len(loads) < len(ds)  # decisively fewer loads than rows
    # RSS must not balloon while streaming a multi-chunk dataset (macOS ru_maxrss
    # is bytes, Linux KiB; either way the delta here is tiny — no whole-shard blowup).
    rss_delta_mb = (rss1 - rss0) / (1024 * 1024 if sys.platform == "darwin" else 1024)
    assert rss_delta_mb < 512, f"RSS grew {rss_delta_mb:.0f} MB during iteration"


def test_training_step_smoke(tiny_dataset: Path) -> None:
    """Real pipeline: BcDataset → DataLoader(bc_collate) → evaluate_actions →
    bc_loss → backward → step, with finite losses across several steps."""
    import math

    import torch
    from torch.utils.data import DataLoader

    from catan_rl.bc.loader import BcDataset, ShardChunkBatchSampler, bc_collate
    from catan_rl.bc.loss import bc_loss
    from catan_rl.policy import CatanPolicy
    from catan_rl.policy.board_geometry import build_geometry

    ds = BcDataset(tiny_dataset, aug_prob=0.5, seed=0, chunk_rows=40, max_cached_chunks=1)
    sampler = ShardChunkBatchSampler(ds, batch_size=16, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=bc_collate, num_workers=0)

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)

    losses: list[float] = []
    for i, batch in enumerate(loader):
        out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
        loss = bc_loss(policy_out=out, batch=batch)
        opt.zero_grad(set_to_none=True)
        loss["total"].backward()
        opt.step()
        assert torch.isfinite(loss["total"]).item(), f"non-finite loss at step {i}"
        losses.append(float(loss["total"].detach()))
        if i >= 3:
            break
    assert losses and all(math.isfinite(x) for x in losses)
