"""PPO update orchestrator.

Thin coordinator between the rollout buffer, the policy, and the
optimiser. The numerical work lives in :mod:`catan_rl.ppo.losses` and
the scheduling in :mod:`catan_rl.ppo.schedules` so the trainer itself
is testable as glue: per-update loop, KL early-stop, schedule
application, gradient clip.

Design pins:

* The trainer does NOT collect rollouts — that lives in Phase 5's
  vec env + GameManager. Phase 4 ships :meth:`PPOTrainer.update` which
  consumes a *finalised* :class:`CompositeRolloutBuffer` for one PPO
  update (``n_epochs`` passes over the buffer, ``n_minibatches`` per
  epoch).
* The trainer does NOT load checkpoints — Phase 8 will own that.
* Belief / opp-action losses are gated: ``belief_logits`` and
  ``opp_action_target`` flow only when the policy / buffer expose them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.buffer import CompositeRolloutBuffer
from catan_rl.ppo.losses import (
    compute_belief_loss,
    compute_entropy_bonus,
    compute_kl_approximation,
    compute_policy_loss,
    compute_value_loss,
)
from catan_rl.ppo.schedules import (
    linear_entropy_coef_schedule,
    linear_lr_schedule,
)


@dataclass(frozen=True)
class UpdateMetrics:
    """One PPO update's worth of TB-bound scalars.

    Each field is the mean across all SGD steps within the update
    (computed in :meth:`PPOTrainer.update`). The trainer caller dumps
    these into TensorBoard via :func:`logging.Logger.info` (Phase 4 has
    no TB writer; Phase 7 wires that).
    """

    policy_loss: float
    value_loss: float
    entropy_bonus: float
    setup_head_entropy: float
    belief_loss: float
    total_loss: float
    approx_kl: float
    clip_frac: float
    ratio_mean: float
    grad_norm: float
    lr: float
    entropy_coef: float
    n_epochs_run: int  # may be < n_epochs if KL early-stop fired
    n_sgd_steps: int


class PPOTrainer:
    """Orchestrates one or more PPO updates on a policy + optimiser pair.

    Construction is parameterised by:

    * ``cfg``: :class:`TrainConfig` (Phase 1). Read-only by the trainer;
      pass a modified config to take a different update path.
    * ``policy``: the :class:`CatanPolicy` instance to update. Must
      expose ``evaluate_actions(obs, action, masks) -> dict`` (see
      :mod:`catan_rl.policy.network`).
    * ``optimizer``: a torch optimiser. The trainer mutates its LR per
      update via :func:`linear_lr_schedule` against the configured
      total update count (Phase 4 derives this from
      ``cfg.total_steps // (n_envs * n_steps)`` if
      ``cfg.optimizer.lr_anneal_total_updates == 0``).
    * ``device``: where the buffer minibatches and the policy live.

    Usage::

        trainer = PPOTrainer(cfg=cfg, policy=policy, optimizer=opt, device="mps")
        rng = np.random.default_rng(cfg.seed)
        for update_idx in range(n_updates):
            buffer = collect_rollout(...)  # Phase 5
            metrics = trainer.update(buffer, update_idx=update_idx, rng=rng)
    """

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device

        # Total updates over which the LR anneals. ``0`` = infer from
        # the rollout schedule.
        self._total_updates_for_lr = (
            cfg.optimizer.lr_anneal_total_updates
            if cfg.optimizer.lr_anneal_total_updates > 0
            else cfg.total_steps // (cfg.rollout.n_envs * cfg.rollout.n_steps)
        )

    # ------------------------------------------------------------------
    # Schedules (exposed for tests + tracing)
    # ------------------------------------------------------------------

    def lr_at(self, update_idx: int) -> float:
        return linear_lr_schedule(
            update_idx=update_idx,
            lr_start=self.cfg.optimizer.lr_start,
            lr_end=self.cfg.optimizer.lr_end,
            total_updates=max(1, self._total_updates_for_lr),
        )

    def entropy_coef_at(self, update_idx: int) -> float:
        return linear_entropy_coef_schedule(
            update_idx=update_idx,
            coef_start=self.cfg.loss.entropy_coef_start,
            coef_end=self.cfg.loss.entropy_coef_end,
            start_update=self.cfg.loss.entropy_anneal_start_update,
            end_update=self.cfg.loss.entropy_anneal_end_update,
        )

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update(
        self,
        buffer: CompositeRolloutBuffer,
        *,
        update_idx: int,
        rng: np.random.Generator,
    ) -> UpdateMetrics:
        """Run one PPO update over the buffer's contents.

        Loops over ``cfg.ppo.n_epochs`` passes; within each epoch,
        samples minibatches of ``cfg.ppo.batch_size`` via the buffer's
        :meth:`CompositeRolloutBuffer.minibatch_indices`. After each
        epoch checks whether the approximate KL (per the configured
        estimator) exceeds ``cfg.ppo.target_kl`` — if so, the remaining
        epochs are skipped.

        Mutates the policy parameters + optimiser state in place.
        Returns :class:`UpdateMetrics` summarising what happened so the
        caller can log it.
        """
        if not buffer.is_finalised:
            raise RuntimeError(
                "PPOTrainer.update requires a buffer with "
                "compute_returns_and_advantages already called."
            )

        # Apply scheduled LR.
        lr = self.lr_at(update_idx)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        entropy_coef = self.entropy_coef_at(update_idx)

        # Accumulators (will be averaged across SGD steps).
        sums: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_bonus": 0.0,
            "setup_head_entropy": 0.0,
            "belief_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "ratio_mean": 0.0,
            "grad_norm": 0.0,
        }
        n_sgd_steps = 0
        n_epochs_run = 0

        for _epoch_idx in range(self.cfg.ppo.n_epochs):
            n_epochs_run += 1
            epoch_kl_sum = 0.0
            epoch_kl_count = 0
            batches = buffer.minibatch_indices(batch_size=self.cfg.ppo.batch_size, rng=rng)

            for batch_indices in batches:
                batch = buffer.get_batch(batch_indices, device=self.device)
                step_stats = self._sgd_step(batch=batch, entropy_coef=entropy_coef)
                for k, v in step_stats.items():
                    sums[k] += float(v)
                epoch_kl_sum += float(step_stats["approx_kl"])
                epoch_kl_count += 1
                n_sgd_steps += 1

            # KL early stop: if the EPOCH-mean KL exceeds the target,
            # skip subsequent epochs in this update.
            if (
                self.cfg.ppo.target_kl > 0.0
                and epoch_kl_count > 0
                and (epoch_kl_sum / epoch_kl_count) > self.cfg.ppo.target_kl
            ):
                break

        if n_sgd_steps == 0:
            raise RuntimeError("No SGD steps executed (empty buffer or zero epochs?).")
        means = {k: v / n_sgd_steps for k, v in sums.items()}
        return UpdateMetrics(
            policy_loss=means["policy_loss"],
            value_loss=means["value_loss"],
            entropy_bonus=means["entropy_bonus"],
            setup_head_entropy=means["setup_head_entropy"],
            belief_loss=means["belief_loss"],
            total_loss=means["total_loss"],
            approx_kl=means["approx_kl"],
            clip_frac=means["clip_frac"],
            ratio_mean=means["ratio_mean"],
            grad_norm=means["grad_norm"],
            lr=lr,
            entropy_coef=entropy_coef,
            n_epochs_run=n_epochs_run,
            n_sgd_steps=n_sgd_steps,
        )

    # ------------------------------------------------------------------
    # Single SGD step
    # ------------------------------------------------------------------

    def _sgd_step(
        self,
        *,
        batch: dict[str, object],
        entropy_coef: float,
    ) -> dict[str, torch.Tensor]:
        """One forward + backward + optimiser step on a single minibatch.

        Returns per-batch scalar tensors for accumulation by the caller.
        """
        # The buffer's get_batch returns a heterogeneous dict (tensors at
        # the leaves, dicts in obs / masks). Cast at use site so the
        # downstream loss helpers see ``Tensor`` directly.
        from typing import cast

        obs = cast(dict[str, torch.Tensor], batch["obs"])
        masks = cast(dict[str, torch.Tensor], batch["masks"])
        actions = cast(torch.Tensor, batch["actions"])
        old_log_prob = cast(torch.Tensor, batch["log_probs"])
        old_value = cast(torch.Tensor, batch["values"])
        advantages = cast(torch.Tensor, batch["advantages"])
        returns = cast(torch.Tensor, batch["returns"])

        # Per-batch advantage normalisation. Under ``advantage_norm=
        # "rollout"`` the buffer already standardised over the full
        # rollout, so ``batch["advantages"]`` arrives zero-mean / unit-
        # std and this branch is a no-op (mean ≈ 0, std ≈ 1). Under
        # "batch" the buffer left the raw advantages and we standardise
        # per minibatch here — otherwise the trainer would silently
        # train against unnormalised advantages, scaling policy
        # gradients by O(advantage) instead of O(1). Reviewer-caught
        # HIGH foot-gun.
        if self.cfg.ppo.advantage_norm == "batch":
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # ``policy`` is typed ``nn.Module`` for flexibility (CatanPolicy
        # or a stub); the trainer requires the ``evaluate_actions``
        # method but does not constrain the class.
        # Cast through Any so static checkers don't insist on a
        # ``evaluate_actions`` method on the bare ``nn.Module`` type.
        from typing import Any as _Any

        head_out: dict[str, torch.Tensor] = cast(_Any, self.policy).evaluate_actions(
            obs, actions, masks
        )
        new_log_prob = head_out["log_prob"]
        new_value = head_out["value"]
        joint_entropy = head_out["entropy"]

        policy_loss, p_stats = compute_policy_loss(
            new_log_prob=new_log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            clip_range=self.cfg.ppo.clip_range,
        )
        value_loss, _v_stats = compute_value_loss(
            new_value=new_value,
            old_value=old_value,
            returns=returns,
            clip_range_vf=self.cfg.ppo.clip_range_vf,
            use_value_clipping=self.cfg.ppo.use_value_clipping,
        )
        entropy_term = compute_entropy_bonus(joint_entropy=joint_entropy)

        # Setup-phase-only entropy: masked mean of H(π) over the transitions
        # taken during the initial settlement-placement phase (step6 §2.2).
        # Computed for the TB diagnostic regardless of the coef; only folded
        # into the loss when the run opts in (coef != 0) so a coef-0 run is
        # byte-identical to before this term existed.
        setup_coef = self.cfg.loss.setup_entropy_coef
        is_setup = batch.get("is_setup")
        if is_setup is not None:
            setup_mask = cast(torch.Tensor, is_setup).to(joint_entropy.dtype)
            n_setup = setup_mask.sum().clamp(min=1.0)
            setup_entropy_mean = (joint_entropy * setup_mask).sum() / n_setup
        else:
            setup_entropy_mean = torch.zeros((), device=self.device)

        belief_loss = torch.zeros((), device=self.device)
        if "belief_logits" in head_out and "belief_target" in batch:
            belief_loss = compute_belief_loss(
                belief_logits=head_out["belief_logits"],
                belief_target=cast(torch.Tensor, batch["belief_target"]),
            )

        total = (
            policy_loss
            + self.cfg.loss.value_coef * value_loss
            - entropy_coef * entropy_term
            + self.cfg.loss.belief_coef * belief_loss
        )
        # Additional setup-phase-only entropy bonus. Guarded on ``!= 0`` so the
        # coef-0 default leaves ``total`` bit-for-bit identical to the four-term
        # objective above (no dependence on ``is_setup`` at all).
        if setup_coef != 0.0:
            total = total - setup_coef * setup_entropy_mean

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.cfg.optimizer.grad_clip_max_norm,
        )
        # On torch < 2.4 clip_grad_norm_ returns a scalar tensor; on
        # newer it's an alias for the same. Coerce defensively.
        grad_norm = torch.as_tensor(grad_norm_value)
        self.optimizer.step()

        # ``new_log_prob`` was captured BEFORE optimizer.step(); detaching
        # it preserves the pre-step ratio for the KL gate. The KL fired
        # here measures "how far did this epoch try to push the policy",
        # not "how far did the new (already-stepped) policy land".
        approx_kl = compute_kl_approximation(
            new_log_prob=new_log_prob.detach(),
            old_log_prob=old_log_prob,
            estimator=self.cfg.ppo.kl_approx,
        )

        return {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_bonus": entropy_term.detach(),
            "setup_head_entropy": setup_entropy_mean.detach(),
            "belief_loss": belief_loss.detach(),
            "total_loss": total.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": p_stats["clip_frac"],
            "ratio_mean": p_stats["ratio_mean"],
            "grad_norm": grad_norm.detach(),
        }
