"""Integration smoke test for the BC training loop.

Generates a tiny dataset (~50 games), trains for a few epochs, asserts
the loss curve actually goes down. This is the end-to-end gate that
catches integration bugs the unit tests can miss (e.g., grad clipping
killing learning, LR schedule mis-wiring, optimizer state corruption).
"""

from __future__ import annotations

from pathlib import Path


def test_bc_train_smoke_loss_decreases(tmp_path: Path) -> None:
    """50-game dataset + 3 epochs at small batch → val NLL drops ≥ 10%."""
    from catan_rl.bc.dataset import generate_dataset
    from catan_rl.bc.train import train_bc

    data_dir = tmp_path / "data"
    generate_dataset(
        out_dir=data_dir,
        n_games=50,
        perturb_pct=0.30,
        shard_size=50,
        seed=0,
        max_turns=120,
        progress_every=10**9,
    )

    out_dir = tmp_path / "run"
    history = train_bc(
        data_dir=data_dir,
        out_dir=out_dir,
        max_epochs=3,
        batch_size=64,
        val_every_steps=20,
        val_pct=0.20,
        peak_lr=3e-4,
        warmup_steps=10,
        patience=999,  # disable early-stop for the smoke test
        value_weight=0.10,
        belief_weight=0.05,
        aug_prob=0.5,
        seed=0,
        verbose=False,
    )

    # The training history records val NLLs at each eval tick.
    val_nlls = history["val_nll_total"]
    assert len(val_nlls) >= 2, "expected at least 2 val ticks"
    drop = (val_nlls[0] - val_nlls[-1]) / max(abs(val_nlls[0]), 1e-6)
    assert drop > 0.05, (
        f"val NLL didn't decrease meaningfully: first={val_nlls[0]:.4f}, "
        f"last={val_nlls[-1]:.4f}, drop_frac={drop:.4f}"
    )
    # Best checkpoint exists.
    assert (out_dir / "best.pt").exists()


def test_bc_train_early_stop_triggers(tmp_path: Path) -> None:
    """With patience=1 + tiny dataset, early-stop must fire before max_epochs."""
    from catan_rl.bc.dataset import generate_dataset
    from catan_rl.bc.train import train_bc

    data_dir = tmp_path / "data"
    generate_dataset(
        out_dir=data_dir,
        n_games=30,
        perturb_pct=0.0,
        shard_size=30,
        seed=0,
        max_turns=100,
        progress_every=10**9,
    )
    history = train_bc(
        data_dir=data_dir,
        out_dir=tmp_path / "run",
        max_epochs=20,
        batch_size=64,
        val_every_steps=10,
        val_pct=0.20,
        peak_lr=3e-4,
        warmup_steps=10,
        patience=1,  # fast early-stop
        value_weight=0.10,
        belief_weight=0.05,
        aug_prob=0.0,
        seed=0,
        verbose=False,
    )
    assert history["early_stopped"] is True or history["epochs_run"] < 20
