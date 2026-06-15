"""Search-label generation (T005): BcDataset-compatible, legal, forced-skipped."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from catan_rl.bc.loader import BcDataset
from catan_rl.expert_iteration.config import SearchLabelConfig
from catan_rl.expert_iteration.labeler import generate_search_labels


def test_shards_are_bcdataset_loadable_and_legal(tiny_ckpt: Path, tmp_path: Path) -> None:
    out = tmp_path / "labels"
    cfg = SearchLabelConfig(
        out_dir=str(out),
        base_ckpt=str(tiny_ckpt),
        sims_per_move=2,
        n_positions=5,
        opponent="heuristic",
        seed=0,
        max_turns=60,
    )
    manifest = generate_search_labels(cfg)
    assert manifest["n_pairs_total"] >= 5
    assert (out / "manifest.json").exists()

    z = np.load(out / "shard_0000.npz")
    n = int(z["action"].shape[0])
    assert n == manifest["n_pairs_total"]
    assert z["action"].shape[1] == 6
    # Every recorded action's TYPE is legal under its mask, and no forced
    # (single-legal-type) position survives the write filter.
    for i in range(n):
        action_type = int(z["action"][i, 0])
        assert bool(z["mask/type"][i][action_type]), f"row {i}: illegal action type"
        assert int(z["mask/type"][i].sum()) > 1, f"row {i}: forced position not filtered"
    # z_disc is a discounted outcome in [-1, 1].
    assert np.all(np.abs(z["z_disc"]) <= 1.0 + 1e-6)

    # The existing BcDataset loads the labeler's output verbatim (split by game_id).
    train, val = BcDataset.train_val_split(out, val_pct=0.5, seed=0, train_aug_prob=0.0)
    assert len(train) + len(val) == n


def test_labeler_is_reproducible(tiny_ckpt: Path, tmp_path: Path) -> None:
    def gen(sub: str) -> np.ndarray:
        out = tmp_path / sub
        cfg = SearchLabelConfig(
            out_dir=str(out),
            base_ckpt=str(tiny_ckpt),
            sims_per_move=2,
            n_positions=5,
            opponent="heuristic",
            seed=7,
            max_turns=60,
        )
        generate_search_labels(cfg)
        return np.load(out / "shard_0000.npz")["action"]

    a = gen("a")
    b = gen("b")
    assert a.shape == b.shape
    assert np.array_equal(a, b)
