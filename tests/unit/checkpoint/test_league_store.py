"""Tests for `checkpoint/league_store.py` (content-addressed snapshot store).

Pins:
1. snapshot_hash is deterministic + content-sensitive (equal tensors → equal
   hash; a single changed value → different hash).
2. put() dedups byte-identical state-dicts to one store file.
3. put()/get() round-trips a state-dict bit-exactly.
4. get() of a missing hash raises LeagueStoreError.
5. is_snapshot_hash recognises only <sha256>.pt names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from catan_rl.checkpoint.league_store import (
    LeagueStore,
    LeagueStoreError,
    is_snapshot_hash,
    snapshot_hash,
)


def _sd(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"w": torch.randn(4, 4), "b": torch.randn(4)}


def test_hash_deterministic_and_content_sensitive() -> None:
    a = _sd(0)
    a_copy = {k: v.clone() for k, v in a.items()}
    b = _sd(1)
    assert snapshot_hash(a) == snapshot_hash(a_copy)
    assert snapshot_hash(a) != snapshot_hash(b)
    # A single perturbed element flips the hash.
    a_perturbed = {k: v.clone() for k, v in a.items()}
    a_perturbed["b"][0] += 1.0
    assert snapshot_hash(a_perturbed) != snapshot_hash(a)


def test_hash_ignores_key_order() -> None:
    a = _sd(3)
    reordered = {"b": a["b"], "w": a["w"]}
    assert snapshot_hash(reordered) == snapshot_hash(a)


def test_put_dedups_identical(tmp_path: Path) -> None:
    store = LeagueStore(tmp_path / "store")
    sd = _sd(7)
    h1 = store.put(sd)
    h2 = store.put({k: v.clone() for k, v in sd.items()})
    assert h1 == h2
    files = list((tmp_path / "store").glob("*.pt"))
    assert len(files) == 1  # dedup: one file, not two


def test_put_get_round_trip(tmp_path: Path) -> None:
    store = LeagueStore(tmp_path / "store")
    sd = _sd(11)
    h = store.put(sd)
    loaded = store.get(h)
    assert set(loaded.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(loaded[k], sd[k])


def test_get_missing_raises(tmp_path: Path) -> None:
    store = LeagueStore(tmp_path / "store")
    with pytest.raises(LeagueStoreError, match="not found"):
        store.get("0" * 64)


def test_is_snapshot_hash() -> None:
    assert is_snapshot_hash("a" * 64 + ".pt")
    assert not is_snapshot_hash("ckpt_000000001.pt")
    assert not is_snapshot_hash(("a" * 64) + ".json")
    assert not is_snapshot_hash("shortname.pt")
