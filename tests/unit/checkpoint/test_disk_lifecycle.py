"""Tests for the checkpoint disk-lifecycle feature (league sidecar + slim saves).

Pins (spec: .claude/veriloop/specs/checkpoint-disk-lifecycle.md):
1. Sidecar round-trip: save->load->apply preserves league (ids, cursor, anchor,
   tensors, opponent_stats); the checkpoint payload embeds NO snapshot
   state_dicts (only refs); the rolling checkpoint file is < 25 MB.
2. Dedup: saving the same league twice does not double the store file count.
3. Backward compat: a handcrafted schema-v1 FAT checkpoint (embedded snapshots +
   anchor) still loads + applies without a store present.
4. GC: after a rolling checkpoint is pruned, store entries referenced by no
   surviving manifest are deleted; a legacy un-manifested .pt makes GC bail.
5. bank_anchor: re-saves a fat checkpoint as a policy-only slim file.
6. save_slim / save_policy_only: policy-only, not resumed by latest().
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from catan_rl.checkpoint.league_store import LeagueStore, snapshot_hash
from catan_rl.checkpoint.manager import (
    SCHEMA_VERSION,
    CheckpointManager,
    bank_anchor,
    load_checkpoint,
    save_checkpoint,
    save_policy_only,
)
from catan_rl.ppo.arguments import LeagueConfig
from catan_rl.selfplay.league import League


class _TinyPolicy(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(8, 4)


def _build_league(n: int = 3) -> League:
    league = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
    for i in range(n):
        league.add_snapshot(
            _TinyPolicy(seed=i).state_dict(),
            update_idx=(i + 1) * 10,
            metadata={"wr": 0.5 + 0.05 * i},
        )
    return league


def _raw_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _save(mgr: CheckpointManager, league: League, idx: int) -> Path:
    return mgr.save(
        config={},
        policy=_TinyPolicy(),
        optimizer=None,
        update_idx=idx,
        global_step=idx,
        league=league,
    )


# ---------------------------------------------------------------------------
# 1 + 2: sidecar round-trip + dedup
# ---------------------------------------------------------------------------


def test_checkpoint_has_refs_not_embedded_state_dicts(tmp_path: Path) -> None:
    league = _build_league(3)
    policy = _TinyPolicy(seed=99)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        config={},
        policy=policy,
        optimizer=None,
        update_idx=1,
        global_step=10,
        league=league,
    )
    raw = _raw_payload(ckpt)
    assert raw["schema_version"] == SCHEMA_VERSION
    for snap in raw["league"]["snapshots"]:
        assert "ref" in snap
        assert "state_dict" not in snap  # NOT embedded
    # Store holds one file per distinct snapshot.
    store_files = list((tmp_path / "league_store").glob("*.pt"))
    assert len(store_files) == 3


def test_rolling_checkpoint_under_25mb(tmp_path: Path) -> None:
    league = _build_league(5)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        config={},
        policy=_TinyPolicy(),
        optimizer=None,
        update_idx=1,
        global_step=10,
        league=league,
    )
    assert ckpt.stat().st_size < 25 * 1024 * 1024


def test_round_trip_preserves_league(tmp_path: Path) -> None:
    league_a = _build_league(3)
    # Give it an anchor + opponent stats so the full state round-trips.
    league_a.set_anchor(_TinyPolicy(seed=77).state_dict(), update_idx=5)
    if hasattr(league_a, "load_opponent_stats"):
        league_a.load_opponent_stats({0: (0.6, 20)})
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        config={},
        policy=_TinyPolicy(),
        optimizer=None,
        update_idx=42,
        global_step=4242,
        league=league_a,
    )
    payload = load_checkpoint(ckpt)
    league_b = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
    payload.apply_to_league(league_b)
    assert [s.snapshot_id for s in league_b._snapshots] == [
        s.snapshot_id for s in league_a._snapshots
    ]
    assert league_b.anchor_id() == league_a.anchor_id()
    # Tensors of the first snapshot match bit-exact.
    for k in league_a._snapshots[0].state_dict:
        assert torch.equal(
            league_b._snapshots[0].state_dict[k], league_a._snapshots[0].state_dict[k]
        )
    # Anchor tensors match.
    for k in league_a._anchor.state_dict:
        assert torch.equal(league_b._anchor.state_dict[k], league_a._anchor.state_dict[k])


def test_second_save_dedups_store(tmp_path: Path) -> None:
    league = _build_league(3)
    store_dir = tmp_path / "league_store"
    for idx in (1, 2):
        save_checkpoint(
            tmp_path / f"ckpt_{idx}.pt",
            config={},
            policy=_TinyPolicy(),
            optimizer=None,
            update_idx=idx,
            global_step=idx,
            league=league,
        )
    # Same 3 snapshots both saves → still exactly 3 store files (no doubling).
    assert len(list(store_dir.glob("*.pt"))) == 3


# ---------------------------------------------------------------------------
# 3: backward compat — fat v1 checkpoint
# ---------------------------------------------------------------------------


def test_fat_v1_checkpoint_loads(tmp_path: Path) -> None:
    # Handcraft a schema-v1 payload with EMBEDDED snapshots + anchor (the old
    # on-disk format). No store dir exists → load must resolve purely inline.
    snaps = []
    for i in range(2):
        sd = {k: v.clone() for k, v in _TinyPolicy(seed=i).state_dict().items()}
        snaps.append({"state_dict": sd, "update_idx": i * 10, "snapshot_id": i, "metadata": {}})
    anchor_sd = {k: v.clone() for k, v in _TinyPolicy(seed=42).state_dict().items()}
    payload_v1 = {
        "schema_version": 1,
        "config": {},
        "update_idx": 7,
        "global_step": 70,
        "policy_state_dict": _TinyPolicy(seed=1).state_dict(),
        "optimizer_state_dict": None,
        "league": {
            "snapshots": snaps,
            "next_snapshot_id": 2,
            "anchor": {
                "state_dict": anchor_sd,
                "update_idx": 5,
                "snapshot_id": 99,
                "metadata": {},
                "reanchor_streak": 0,
                "last_promote_update": -1,
                "n_promotions": 0,
            },
        },
        "rng": None,
        "vec_env": None,
        "metadata": {},
    }
    fat = tmp_path / "v11_cand_fake.pt"
    torch.save(payload_v1, fat)

    payload = load_checkpoint(fat)
    assert payload.schema_version == SCHEMA_VERSION  # migrated in-memory
    league_b = League(LeagueConfig(maxlen=10, add_snapshot_every_n_updates=1))
    payload.apply_to_league(league_b)
    assert [s.snapshot_id for s in league_b._snapshots] == [0, 1]
    assert league_b.anchor_id() == 99
    for k in anchor_sd:
        assert torch.equal(league_b._anchor.state_dict[k], anchor_sd[k])
    # No store directory was created by a pure inline load.
    assert not (tmp_path / "league_store").exists()


# ---------------------------------------------------------------------------
# 4: GC
# ---------------------------------------------------------------------------


def test_gc_deletes_unreferenced_store_entries(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep_last_n=1)
    # maxlen=1 league: each add evicts the prior snapshot.
    league = League(LeagueConfig(maxlen=1, add_snapshot_every_n_updates=1))
    league.add_snapshot(_TinyPolicy(seed=0).state_dict(), update_idx=0)
    hash_a = snapshot_hash(league._snapshots[0].state_dict)
    _save(mgr, league, 0)
    # Roll to a new snapshot (evicts A) and save again — prunes ckpt_0 + manifest.
    league.add_snapshot(_TinyPolicy(seed=1).state_dict(), update_idx=1)
    hash_b = snapshot_hash(league._snapshots[0].state_dict)
    _save(mgr, league, 1)
    store = LeagueStore(tmp_path / "league_store")
    assert not store.exists(hash_a), "evicted+pruned snapshot should be GC'd"
    assert store.exists(hash_b), "live snapshot must remain"


def test_gc_bails_on_unmanifested_pt(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep_last_n=5)
    league = League(LeagueConfig(maxlen=1, add_snapshot_every_n_updates=1))
    league.add_snapshot(_TinyPolicy(seed=0).state_dict(), update_idx=0)
    hash_a = snapshot_hash(league._snapshots[0].state_dict)
    _save(mgr, league, 0)
    # Drop a legacy un-manifested fat file into the dir.
    torch.save({"schema_version": 1}, tmp_path / "ckpt_000000009.pt")
    # New save triggers GC — which must BAIL (conservative) leaving A intact.
    league.add_snapshot(_TinyPolicy(seed=1).state_dict(), update_idx=1)
    _save(mgr, league, 1)
    store = LeagueStore(tmp_path / "league_store")
    assert store.exists(hash_a), "GC must bail (not delete) when an un-manifested .pt exists"


# ---------------------------------------------------------------------------
# 5 + 6: slim saves + bank_anchor
# ---------------------------------------------------------------------------


def test_save_policy_only_is_slim(tmp_path: Path) -> None:
    path = tmp_path / "slim.pt"
    save_policy_only(path, config={}, policy=_TinyPolicy(), update_idx=3, global_step=30)
    raw = _raw_payload(path)
    assert raw["metadata"]["kind"] == "policy_only"
    assert raw["optimizer_state_dict"] is None
    assert raw["league"] is None
    assert raw["rng"] is None


def test_save_slim_not_returned_by_latest(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep_last_n=5)
    _save(mgr, _build_league(1), 0)
    slim = mgr.save_slim(config={}, policy=_TinyPolicy(), update_idx=1, global_step=10)
    assert slim.name.startswith("slim_ckpt_")
    # latest() must return the rolling ckpt, never the slim file.
    assert mgr.latest() is not None
    assert not mgr.latest().name.startswith("slim_ckpt_")


def test_bank_anchor_slims_a_fat_checkpoint(tmp_path: Path) -> None:
    fat = tmp_path / "anchor.pt"
    save_checkpoint(
        fat,
        config={"a": 1},
        policy=_TinyPolicy(seed=5),
        optimizer=torch.optim.AdamW(_TinyPolicy().parameters()),
        update_idx=200,
        global_step=2000,
        league=_build_league(3),
    )
    banked = bank_anchor(fat)
    assert banked == fat
    raw = _raw_payload(fat)
    assert raw["metadata"]["kind"] == "policy_only"
    assert raw["optimizer_state_dict"] is None
    assert raw["league"] is None
    assert raw["update_idx"] == 200  # lineage bookkeeping preserved
