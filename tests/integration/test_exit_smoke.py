"""Integration smoke (T012): a tiny ExIt round runs label -> distill -> gate
end-to-end and produces a GateResult with zero ruleset violations.

Random-init policies + tiny budgets keep it fast; the point is that the full
pipeline drives the engine end-to-end (search labeling -> warm-started distill ->
search-free gate), not that the distilled policy wins.
"""

from __future__ import annotations

from pathlib import Path


def _tiny_ckpt(path: Path) -> Path:
    from catan_rl.checkpoint.manager import save_checkpoint
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    save_checkpoint(
        path,
        config={"smoke": True},
        policy=policy,
        optimizer=None,
        update_idx=0,
        global_step=0,
        capture_rng=False,
    )
    return path


def test_exit_round_label_distill_gate_end_to_end(tmp_path: Path) -> None:
    from catan_rl.expert_iteration.config import DistillConfig, SearchLabelConfig
    from catan_rl.expert_iteration.distill import distill
    from catan_rl.expert_iteration.gate import run_gate
    from catan_rl.expert_iteration.labeler import generate_search_labels

    base = _tiny_ckpt(tmp_path / "base.pt")

    labels = tmp_path / "labels"
    manifest = generate_search_labels(
        SearchLabelConfig(
            out_dir=str(labels),
            base_ckpt=str(base),
            sims_per_move=2,
            n_positions=8,
            opponent="heuristic",
            seed=0,
            max_turns=60,
        )
    )
    assert manifest["n_pairs_total"] >= 1

    distilled = distill(
        DistillConfig(
            data_dir=str(labels),
            out_dir=str(tmp_path / "distill"),
            init_ckpt=str(base),
            peak_lr=1e-4,
            max_epochs=1,
            batch_size=8,
            seed=0,
            device="cpu",
        )
    )
    assert distilled.exists()

    # Gate vs the (random) base — pipeline check; verdict is a valid GateResult.
    verdict = run_gate(str(distilled), str(base), n_quick=4, n_confirm=4, seed=0, device="cpu")
    assert "passed" in verdict
    assert "wr_quick_vs_v6" in verdict
    assert verdict["rules_violations_quick"] == 0
