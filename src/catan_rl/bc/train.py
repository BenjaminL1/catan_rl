"""BC training loop.

Wires :class:`BcDataset` + :func:`bc_loss` + ``CatanPolicy`` into the
outer train/val/early-stop loop per ``v2_step3_bc.md`` §4-§5.

Behaviour:

  * AdamW, weight_decay 1e-4, β=(0.9, 0.999), eps=1e-5.
  * 500-step linear warmup → constant LR (3-2 panel vote; cosine on a
    short BC run is theater).
  * Val every ``val_every_steps`` minibatches; full ``BcDataset``
    train/val split by game_id (no within-game leakage).
  * Early-stop on val NLL plateau with configurable patience.
  * Best checkpoint saved to ``out_dir/best.pt``; final state saved to
    ``out_dir/last.pt``; history (per-eval NLLs, per-epoch elapsed,
    early-stop flag) written to ``out_dir/history.json``.
  * Board geometry plumbed into the policy once before training so the
    GNN encoder + axial pos emb have real adjacency tables (not the
    placeholder zeros from Step 2).

WR-vs-heuristic evaluation is intentionally NOT run during training —
it costs ~50ms/step x ~250 steps/game x N games per call, which would
dominate wall-clock with no real signal beyond what val NLL already
gives. The final gate evaluation (separate script) runs the 600-game
WR check once, after the training loop terminates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from catan_rl.bc.loader import BcDataset, bc_collate
from catan_rl.bc.loss import bc_loss
from catan_rl.policy import CatanPolicy
from catan_rl.policy.board_geometry import build_geometry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {k2: v2.to(device) for k2, v2 in v.items()}
        else:
            out[k] = v.to(device)
    return out


def _eval_val(
    policy: CatanPolicy,
    val_loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    value_weight: float,
    belief_weight: float,
) -> dict[str, float]:
    """Compute mean val losses over the whole val loader."""
    policy.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
            loss = bc_loss(
                policy_out=out,
                batch=batch,
                value_weight=value_weight,
                belief_weight=belief_weight,
            )
            for k, v in loss.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().item())
            n_batches += 1
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def train_bc(
    *,
    data_dir: Path,
    out_dir: Path,
    max_epochs: int = 10,
    batch_size: int = 1024,
    val_every_steps: int = 500,
    val_pct: float = 0.10,
    peak_lr: float = 3e-4,
    warmup_steps: int = 500,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 3,
    value_weight: float = 0.10,
    belief_weight: float = 0.05,
    aug_prob: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
    num_workers: int = 0,
    verbose: bool = True,
    init_ckpt: Path | None = None,
) -> dict[str, Any]:
    """Run the BC training loop and return the per-eval history.

    ``init_ckpt`` (default None) warm-starts the policy from an existing
    v2-lineage checkpoint before training — used for expert-iteration
    distillation (fine-tune v6 on search-derived targets). ``None`` keeps the
    original from-scratch behavior, byte-identical.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    train_ds, val_ds = BcDataset.train_val_split(
        data_dir, val_pct=val_pct, seed=seed, train_aug_prob=aug_prob, val_aug_prob=0.0
    )

    def _worker_init(worker_id: int) -> None:
        # Reseed each worker's private augmentation RNG from (seed, worker_id).
        # BcDataset._rng is a np.random.default_rng that torch's default
        # per-worker reseed does NOT touch, so on fork every worker would
        # otherwise share one stream and draw correlated D6 augmentations.
        # Keying on worker_id (not fork divergence) is also correct under spawn.
        info = torch.utils.data.get_worker_info()
        if info is not None:
            ds = info.dataset
            assert isinstance(ds, BcDataset)
            ds._rng = np.random.default_rng([seed, worker_id])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bc_collate,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bc_collate,
        num_workers=num_workers,
        drop_last=False,
    )

    # The D4 auxiliary value head is a PPO representation-shaping term folded by
    # PPOTrainer via ``aux_value_coef``; the BC objective (bc_loss) does not
    # train it. We still build it (the CatanPolicy default) so BC checkpoints
    # carry the same state-dict keys as PPO / distill policies: those load
    # BC-lineage weights under ``strict=True`` (init_policy_checkpoint warm-start,
    # distill.py), which would raise on a missing/extra aux head. The head stays
    # at its init until a downstream PPO stage trains it.
    policy = CatanPolicy().to(dev)
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    if init_ckpt is not None:
        # Warm-start (expert-iteration distillation): load v2-lineage weights into
        # the policy before fine-tuning. set_board_geometry ran first so the
        # strict load overwrites the geometry buffers with the checkpoint's.
        from catan_rl.checkpoint import load_checkpoint

        load_checkpoint(Path(init_ckpt), map_location=dev).apply_to_policy(policy, strict=True)

    optimizer = AdamW(
        policy.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=weight_decay,
    )

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, _lr_lambda)

    history: dict[str, list[float]] = {
        "train_step": [],
        "train_loss_total": [],
        "train_loss_policy": [],
        "train_loss_value": [],
        "train_loss_belief": [],
        "val_step": [],
        "val_nll_total": [],
        "val_nll_policy": [],
        "val_nll_value": [],
        "val_nll_belief": [],
        "epoch_elapsed_seconds": [],
    }

    best_val_nll = float("inf")
    best_step = 0
    patience_counter = 0
    early_stopped = False
    global_step = 0
    epochs_run = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        epoch_t0 = time.time()
        policy.train()
        for batch in train_loader:
            batch = _move_batch_to_device(batch, dev)
            out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
            loss = bc_loss(
                policy_out=out,
                batch=batch,
                value_weight=value_weight,
                belief_weight=belief_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            history["train_step"].append(global_step)
            history["train_loss_total"].append(float(loss["total"].detach().item()))
            history["train_loss_policy"].append(float(loss["policy"].detach().item()))
            history["train_loss_value"].append(float(loss["value"].detach().item()))
            history["train_loss_belief"].append(float(loss["belief"].detach().item()))

            if global_step % val_every_steps == 0:
                val_metrics = _eval_val(
                    policy,
                    val_loader,
                    device=dev,
                    value_weight=value_weight,
                    belief_weight=belief_weight,
                )
                if not val_metrics:
                    continue
                history["val_step"].append(global_step)
                history["val_nll_total"].append(val_metrics.get("total", 0.0))
                history["val_nll_policy"].append(val_metrics.get("policy", 0.0))
                history["val_nll_value"].append(val_metrics.get("value", 0.0))
                history["val_nll_belief"].append(val_metrics.get("belief", 0.0))

                cur = val_metrics.get("total", float("inf"))
                if cur < best_val_nll - 1e-6:
                    best_val_nll = cur
                    best_step = global_step
                    patience_counter = 0
                    _save_checkpoint(policy, out_dir / "best.pt", global_step, cur)
                else:
                    patience_counter += 1

                if verbose:
                    print(
                        f"[bc.train] step={global_step:>7,} "
                        f"val_total={cur:.4f} "
                        f"best={best_val_nll:.4f}@{best_step:,} "
                        f"patience={patience_counter}/{patience}",
                        flush=True,
                    )

                if patience_counter >= patience:
                    early_stopped = True
                    break
                policy.train()

        epoch_elapsed = time.time() - epoch_t0
        history["epoch_elapsed_seconds"].append(epoch_elapsed)
        epochs_run = epoch + 1
        if verbose:
            print(
                f"[bc.train] epoch {epoch + 1}/{max_epochs} done in {epoch_elapsed:.1f}s",
                flush=True,
            )
        if early_stopped:
            break

    # Always save the last checkpoint too.
    _save_checkpoint(policy, out_dir / "last.pt", global_step, val_nll=None)

    summary = {
        **history,
        "best_val_nll": best_val_nll,
        "best_step": best_step,
        "early_stopped": early_stopped,
        "epochs_run": epochs_run,
        "global_steps": global_step,
        "wall_clock_seconds": time.time() - t0,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    return summary


def _save_checkpoint(policy: CatanPolicy, path: Path, step: int, val_nll: float | None) -> None:
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "step": int(step),
            "val_nll": float(val_nll) if val_nll is not None else None,
        },
        path,
    )
