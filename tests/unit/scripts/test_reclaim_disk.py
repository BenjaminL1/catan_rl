"""Tests for `catan_rl.cli.reclaim_disk` (scripts/reclaim_disk.py shim).

Runs ONLY against synthetic tmp trees of tiny torch payloads — never the real
runs/ tree. Pins:
1. scan() finds byte-identical duplicate groups + fat anchors.
2. Dry-run (default) mutates nothing (hashes/mtimes unchanged).
3. --execute dedups duplicates (via hardlink) + slims fat anchors.
4. execute() never touches files outside the scanned root.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from catan_rl.cli.reclaim_disk import execute, main, scan


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fat_ckpt(path: Path, seed: int = 0) -> None:
    """Write a fat checkpoint with a small policy but a LARGE embedded league +
    optimizer, so slimming to policy-only frees a measurable amount of bytes."""
    torch.manual_seed(seed)
    policy_sd = {"w": torch.randn(8, 8), "b": torch.randn(8)}
    big = {"w": torch.randn(256, 256), "b": torch.randn(256)}  # ~256 KB tensor
    payload = {
        "schema_version": 2,
        "config": {},
        "update_idx": 1,
        "global_step": 1,
        "policy_state_dict": policy_sd,
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.randn(256, 256)}},
            "param_groups": [],
        },
        "league": {
            "snapshots": [{"state_dict": big, "update_idx": 0, "snapshot_id": 0, "metadata": {}}],
            "next_snapshot_id": 1,
        },
        "rng": None,
        "vec_env": None,
        "metadata": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _make_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Two byte-identical duplicates under a run dir.
    (root / "run").mkdir()
    torch.save({"schema_version": 2, "x": [1, 2, 3]}, root / "run" / "a.pt")
    (root / "run" / "b.pt").write_bytes((root / "run" / "a.pt").read_bytes())
    # A fat anchor.
    _fat_ckpt(root / "anchors" / "v10.pt", seed=1)


def test_scan_finds_dups_and_fat_anchors(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    report = scan(tmp_path)
    assert report.total_files == 3
    assert len(report.duplicate_groups) == 1
    assert {p.name for p in report.duplicate_groups[0]} == {"a.pt", "b.pt"}
    assert len(report.fat_anchors) == 1
    assert report.fat_anchors[0].name == "v10.pt"
    assert report.estimated_reclaim_bytes() > 0


def test_dry_run_mutates_nothing(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    before = {p: (_digest(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*.pt")}
    rc = main([str(tmp_path)])  # no --execute
    assert rc == 0
    after = {p: (_digest(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*.pt")}
    assert before == after


def test_execute_dedups_and_slims(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    fat = tmp_path / "anchors" / "v10.pt"
    fat_size_before = fat.stat().st_size
    report = scan(tmp_path)
    dup_freed, anchor_freed = execute(report)
    assert dup_freed > 0
    assert anchor_freed > 0
    # Dedup: the two copies are now the same inode (hardlinked).
    a = tmp_path / "run" / "a.pt"
    b = tmp_path / "run" / "b.pt"
    assert a.stat().st_ino == b.stat().st_ino
    # Slim: anchor shrank + lost its league/optimizer.
    assert fat.stat().st_size < fat_size_before
    raw = torch.load(fat, map_location="cpu", weights_only=False)
    assert raw["optimizer_state_dict"] is None
    assert raw["league"] is None
    assert raw["metadata"]["kind"] == "policy_only"


def test_execute_confined_to_root(tmp_path: Path) -> None:
    # A file OUTSIDE the root that is byte-identical to one inside must not be
    # dragged into a dedup group (scan only walks the root).
    root = tmp_path / "inside"
    _make_tree(root)
    outside = tmp_path / "outside.pt"
    outside.write_bytes((root / "run" / "a.pt").read_bytes())
    outside_ino_before = outside.stat().st_ino
    report = scan(root)
    execute(report)
    assert outside.stat().st_ino == outside_ino_before  # untouched
