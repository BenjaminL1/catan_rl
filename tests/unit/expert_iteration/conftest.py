"""Fixtures for expert-iteration tests.

A random-init v2 checkpoint is enough for STRUCTURAL tests (label format, warm-start
load, distillation v2-lineage round-trip) — none depend on the weights being trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A schema'd v2-lineage checkpoint of a random-init CatanPolicy."""
    from catan_rl.checkpoint.manager import save_checkpoint
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    path = tmp_path_factory.mktemp("ckpt") / "tiny_v2.pt"
    save_checkpoint(
        path,
        config={"test": True},
        policy=policy,
        optimizer=None,
        update_idx=0,
        global_step=0,
        capture_rng=False,
    )
    return path


@pytest.fixture(scope="module")
def tiny_labels(tmp_path_factory: pytest.TempPathFactory, tiny_ckpt: Path) -> Path:
    """A tiny search-labeled dataset dir (generated once per module)."""
    from catan_rl.expert_iteration.config import SearchLabelConfig
    from catan_rl.expert_iteration.labeler import generate_search_labels

    out = tmp_path_factory.mktemp("labels")
    cfg = SearchLabelConfig(
        out_dir=str(out),
        base_ckpt=str(tiny_ckpt),
        sims_per_move=2,
        n_positions=20,
        opponent="heuristic",
        seed=0,
        max_turns=60,
    )
    generate_search_labels(cfg)
    return out
