"""
Custom PPO implementation for 1v1 Catan.

This is a synchronous, single-process PPO loop optimized for M1 Mac:
  1. Collect n_steps of experience by playing games
  2. Compute GAE advantages
  3. Run n_epochs of minibatch updates on the collected data
  4. Repeat

The key PPO idea: we have an "old" policy (used to collect experience)
and a "new" policy (being updated). The ratio new/old is clipped to
prevent the policy from changing too much in one update (catastrophic
forgetting). See Schulman et al. 2017.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from catan_rl.algorithms.common.gae import explained_variance, safe_mean
from catan_rl.algorithms.common.rollout_buffer import CompositeRolloutBuffer
from catan_rl.eval.evaluation_manager import EvaluationManager
from catan_rl.models.build_agent_model import build_agent_model
from catan_rl.models.utils import ValueFunctionNormalizer
from catan_rl.selfplay.game_manager import GameManager
from catan_rl.selfplay.league import League


class CatanPPO:
    """Proximal Policy Optimization trainer for Catan.

    Typical usage:
        ppo = CatanPPO(config)
        ppo.train()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: merged dict from ppo/arguments.py get_config().
                    Contains both model architecture and training hyperparams.
        """
        self.config = config

        # CPU is fastest for this model at batch_size=1 rollouts (MPS kernel
        # launch overhead makes it 9x slower). CUDA is only useful if available
        # and explicitly requested. Override with config["device"] if needed.
        if config.get("device"):
            self.device = config["device"]
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Build policy network (config keys match build_agent_model.DEFAULT_MODEL_CONFIG)
        model_kwargs = {
            k: config[k]
            for k in (
                "obs_output_dim",
                "value_hidden_dims",
                "action_head_hidden_dim",
                "tile_in_dim",
                "tile_model_dim",
                "curr_player_main_in_dim",
                "other_player_main_in_dim",
                "dev_card_embed_dim",
                "dev_card_model_dim",
                "tile_model_num_heads",
                "proj_dev_card_dim",
                "dev_card_model_num_heads",
                "tile_encoder_num_layers",
                "proj_tile_dim",
                "dropout",
            )
            if k in config
        }
        self.policy = build_agent_model(device=self.device, **model_kwargs)

        # torch.compile: fuses kernels, reduces overhead. Opt-in since dev-card
        # lists trigger dynamic-shape recompiles. Enable after confirming stable.
        if config.get("torch_compile", False):
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            print("torch.compile() enabled for policy (mode=reduce-overhead).")

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Hyperparameters
        self.n_steps = config["n_steps"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_range = config["clip_range"]
        # Entropy: use start value if annealing, else fixed
        self._entropy_coef_start = config.get("entropy_coef_start", config["entropy_coef"])
        self._entropy_coef_final = config.get("entropy_coef_final", config["entropy_coef"])
        self._entropy_anneal_start = config.get("entropy_coef_anneal_start", 0)
        self._entropy_anneal_end = config.get("entropy_coef_anneal_end", 0)
        self.use_entropy_annealing = self._entropy_anneal_end > self._entropy_anneal_start
        self.entropy_coef = self._entropy_coef_start
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.total_timesteps = config["total_timesteps"]
        self.checkpoint_freq = config["checkpoint_freq"]
        # Linear LR decay (Charlesworth-style)
        self.use_linear_lr_decay = config.get("use_linear_lr_decay", False)
        self.lr_final = config.get("lr_final", config["learning_rate"] * 0.1)

        # Value normalization
        self.value_normalizer = ValueFunctionNormalizer()
        self.normalize_values = config.get("normalize_values", True)

        # Multi-env: in-process parallel games (no subprocess overhead on M1)
        self.n_envs = config.get("n_envs", 1)

        # Rollout buffer (Charlesworth-style dict observations)
        self.buffer = CompositeRolloutBuffer(
            n_steps=self.n_steps,
            device=self.device,
            n_envs=self.n_envs,
        )

        # KL early stopping: break out of update epochs if policy drifts too far
        self.target_kl = config.get("target_kl")

        # Entropy floor: prevent policy collapse by boosting entropy_coef if needed
        self.entropy_floor = config.get("entropy_floor", 0.003)
        self.entropy_floor_coef = config.get("entropy_floor_coef", 0.01)
        self._last_entropy = float("inf")  # updated after each update() call

        # Stagnation detection
        self.stagnation_window = config.get("stagnation_window", 10)
        self.stagnation_threshold = config.get("stagnation_threshold", 0.03)
        self._eval_win_rate_history: list = []

        # League (Charlesworth-style): past policies for self-play
        self.league = League(
            maxlen=config.get("league_maxlen", 100),
            add_every=config.get("add_policy_every", 4),
            random_weight=config.get("league_random_weight", 0.0),
            heuristic_weight=config.get("heuristic_opponent_weight", 0.0),
        )

        # Add initial policy so league always has at least one (required for policy opponents)
        self.league.add(copy.deepcopy(self._raw_policy_state_dict()))

        # Build policy factory for league/GameManager (creates policy instances for inference)
        def build_policy_fn(device: str = "cpu"):
            return build_agent_model(device=device, **model_kwargs)

        # Environment manager: use league for policy opponents when league has policies
        self.game_manager = GameManager(
            n_envs=self.n_envs,
            opponent_type="random",
            max_turns=config.get("max_turns", 500),
            league=self.league,
            build_policy_fn=build_policy_fn,
            device=self.device,
        )

        # Evaluation opponent: starts as configured, auto-upgrades to heuristic
        # once the model crosses eval_upgrade_threshold win rate vs the current opponent.
        self._eval_opponent = config.get("opponent_type", "random")
        self._eval_upgrade_threshold = config.get("eval_upgrade_threshold", 0.95)
        self.eval_manager = EvaluationManager(
            n_games=config.get("eval_games", 40),
            opponent_type=self._eval_opponent,
            max_turns=config.get("max_turns", 500),
        )

        # Logging
        self.log_dir = config.get("log_dir", "runs/train")
        self.writer = SummaryWriter(self.log_dir)

        # Checkpoint directory
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints/train")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []

    def _raw_policy_state_dict(self) -> dict:
        """Return policy state dict without torch.compile's _orig_mod. prefix.

        torch.compile wraps the policy in OptimizedModule, which prefixes all
        keys with '_orig_mod.'. League policies and checkpoints are always stored
        in bare format so they can be loaded into uncompiled CatanPolicy instances.
        """
        sd = self.policy.state_dict()
        if next(iter(sd)).startswith("_orig_mod."):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
        return sd

    def _obs_to_tensor_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        """Convert a single env obs dict (numpy/ints) to a batched tensor dict (B=1)."""
        return self._batch_obs_to_tensor_dict([obs])

    def _batch_obs_to_tensor_dict(self, obs_list: list[dict]) -> dict[str, torch.Tensor]:
        """Convert a list of B env obs dicts to a single batched tensor dict.

        Uses np.stack + from_numpy for a single memcopy per array (no intermediate
        per-element tensors). Dev cards are (B, 16) int64; player_modules.py tensor
        branch handles this directly without pad_sequence.
        """
        device = self.device
        tile = np.stack([o["tile_representations"] for o in obs_list])  # (B,19,79)
        curr = np.stack([o["current_player_main"] for o in obs_list])  # (B,166)
        nxt = np.stack([o["next_player_main"] for o in obs_list])  # (B,173)
        hid = np.stack([o["current_player_hidden_dev"] for o in obs_list])  # (B,16) int32
        cpd = np.stack([o["current_player_played_dev"] for o in obs_list])  # (B,16) int32
        npd = np.stack([o["next_player_played_dev"] for o in obs_list])  # (B,16) int32
        return {
            "tile_representations": torch.from_numpy(tile).to(device),
            "current_player_main": torch.from_numpy(curr).to(device),
            "next_player_main": torch.from_numpy(nxt).to(device),
            "current_player_hidden_dev": torch.from_numpy(hid).to(device).long(),
            "current_player_played_dev": torch.from_numpy(cpd).to(device).long(),
            "next_player_played_dev": torch.from_numpy(npd).to(device).long(),
        }

    def _batch_masks_to_tensor(self, masks_list: list[dict]) -> dict[str, torch.Tensor]:
        """Stack a list of B per-env mask dicts into batched bool tensors."""
        device = self.device
        return {
            k: torch.from_numpy(np.stack([m[k] for m in masks_list])).to(device)
            for k in masks_list[0]
        }

    def _run_batched_opponent_turns(
        self, pending: list[int], partial_rewards: list[float]
    ) -> dict[int, tuple]:
        """Run batched opponent NN inference until all pending envs finish their turn.

        Args:
            pending: list of env indices with _opp_turn_in_progress == True
            partial_rewards: reward accumulated before opponent acts, aligned with pending

        Returns:
            dict mapping env_idx → (final_obs, total_reward, done, info)
        """
        gm = self.game_manager
        opp_policy = gm.rollout_opp_policy
        results = {}
        # Use dict so slot alignment survives as pending shrinks each loop
        accum = {env_i: partial_rewards[slot] for slot, env_i in enumerate(pending)}

        while pending:
            # Batch opponent obs/masks across all still-pending envs
            opp_pairs = [gm.get_opponent_obs_masks(i) for i in pending]
            opp_obs_t = self._batch_obs_to_tensor_dict([p[0] for p in opp_pairs])
            opp_mask_t = self._batch_masks_to_tensor([p[1] for p in opp_pairs])

            with torch.inference_mode():
                opp_actions, _, _ = opp_policy.act(opp_obs_t, opp_mask_t, deterministic=True)
            opp_actions_np = opp_actions.cpu().numpy()  # (len(pending), 6)

            still_pending = []
            for slot, env_i in enumerate(pending):
                turn_done, obs, rew_delta, done, info = gm.apply_opponent_action(
                    env_i, opp_actions_np[slot]
                )
                if turn_done:
                    results[env_i] = (obs, accum[env_i] + rew_delta, done, info)
                else:
                    still_pending.append(env_i)
            pending = still_pending

        return results

    def collect_rollouts(self) -> dict[str, float]:
        """Play n_steps of games and store transitions in the buffer.

        Main agent inference is batched (batch=n_envs). Opponent NN inference
        is also batched via _run_batched_opponent_turns — all envs whose
        opponent turn was triggered in the same outer loop share one forward
        pass through the shared rollout opponent policy.

        Returns:
            Dict with rollout statistics (mean reward, mean length, etc.).
        """
        self.policy.eval()
        self.buffer.reset()

        observations, _ = self.game_manager.reset_all()
        ep_rewards = [0.0] * self.n_envs
        ep_lengths = [0] * self.n_envs
        steps_collected = 0

        while steps_collected < self.n_steps:
            # Clamp batch to remaining steps (handles n_steps % n_envs != 0)
            batch_n = min(self.n_envs, self.n_steps - steps_collected)

            obs_batch = [observations[i] for i in range(batch_n)]
            masks_list = self.game_manager.get_masks()[:batch_n]

            obs_t = self._batch_obs_to_tensor_dict(obs_batch)
            masks_t = self._batch_masks_to_tensor(masks_list)

            with torch.inference_mode():
                actions, values, log_probs = self.policy.act(obs_t, masks_t)

            actions_np = actions.cpu().numpy()  # (batch_n, 6)
            values_squeezed = values.squeeze(-1)
            if self.normalize_values:
                values_np = self.value_normalizer.denormalize(values_squeezed).cpu().numpy()
            else:
                values_np = values_squeezed.cpu().numpy()  # (batch_n,)
            log_probs_np = log_probs.cpu().numpy()  # (batch_n,)

            # Phase 1: step all main agents; collect envs that need opponent NN turns
            pending_opp = []  # env indices whose opponent turn is deferred
            pending_prewd = []  # partial reward accumulated before opponent acts
            immediate = []  # env indices with complete transitions

            for i in range(batch_n):
                new_obs, reward, done, info = self.game_manager.step_one(i, actions_np[i])
                if info.get("opp_turn_pending"):
                    pending_opp.append(i)
                    pending_prewd.append(reward)
                else:
                    observations[i] = new_obs
                    immediate.append((i, new_obs, reward, done, info))

            # Phase 2: batch opponent inference for deferred turns
            if pending_opp and self.game_manager.rollout_opp_policy is not None:
                opp_results = self._run_batched_opponent_turns(pending_opp, pending_prewd)
                for env_i, (fin_obs, fin_rew, fin_done, fin_info) in opp_results.items():
                    observations[env_i] = fin_obs
                    immediate.append((env_i, fin_obs, fin_rew, fin_done, fin_info))

            # Phase 3: store all completed transitions to buffer
            for env_i, new_obs, reward, done, info in immediate:
                self.buffer.add(
                    obs_batch[env_i],
                    actions_np[env_i],
                    reward,
                    done,
                    float(values_np[env_i]),
                    float(log_probs_np[env_i]),
                    masks_list[env_i],
                )
                ep_rewards[env_i] += reward
                ep_lengths[env_i] += 1
                self.global_step += 1
                steps_collected += 1

                if done:
                    self.episode_rewards.append(ep_rewards[env_i])
                    self.episode_lengths.append(ep_lengths[env_i])
                    self.episode_wins.append(1.0 if info.get("is_success") else 0.0)
                    ep_rewards[env_i] = 0.0
                    ep_lengths[env_i] = 0

        # Bootstrapped last values — one batched forward pass for all envs
        with torch.inference_mode():
            obs_t = self._batch_obs_to_tensor_dict(observations)
            raw_last = self.policy.get_value(obs_t).squeeze(-1)
            if self.normalize_values:
                last_values = self.value_normalizer.denormalize(raw_last).cpu().numpy()
            else:
                last_values = raw_last.cpu().numpy()  # (n_envs,)

        self.buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)
        self.buffer.finalize(self.device)

        if self.normalize_values:
            self.value_normalizer.update(
                torch.tensor(self.buffer.returns[: self.n_steps], dtype=torch.float32)
            )

        stats = {}
        if self.episode_rewards:
            stats["mean_reward"] = safe_mean(self.episode_rewards[-100:])
            stats["mean_length"] = safe_mean(self.episode_lengths[-100:])
            stats["mean_win_rate"] = safe_mean(self.episode_wins[-100:])
        return stats

    def update(self) -> dict[str, float]:
        """Run up to n_epochs of PPO updates with KL early stopping.

        Epochs are terminated early if the mean KL divergence between the
        old and new policy exceeds target_kl. This prevents over-updating
        stale data while allowing the full n_epochs when updates are safe.

        Returns:
            Dict with training loss statistics including approx_kl and
            epochs_completed (useful for diagnosing under/over-updating).
        """
        self.policy.train()

        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0
        n_updates = 0
        epochs_completed = 0

        for epoch in range(self.n_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_prob = batch["old_log_prob"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                masks = batch["masks"]

                # Normalize advantages (per-batch, not globally)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass: re-evaluate old actions under current policy
                values, new_log_prob, entropy = self.policy.evaluate_actions(obs, masks, actions)
                values = values.squeeze(-1)  # (B,)

                # ── Policy loss (clipped surrogate objective) ─────────
                ratio = torch.exp(new_log_prob - old_log_prob)
                pg_loss_1 = -advantages * ratio
                pg_loss_2 = -advantages * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # ── Value loss ────────────────────────────────────────
                if self.normalize_values:
                    norm_returns = self.value_normalizer.normalize(returns)
                else:
                    norm_returns = returns
                value_loss = nn.functional.mse_loss(values, norm_returns)

                # ── Entropy bonus ─────────────────────────────────────
                entropy_loss = -entropy

                # ── Total loss ────────────────────────────────────────
                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    batch_kl = (old_log_prob - new_log_prob).mean().item()

                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction
                total_kl += batch_kl
                epoch_kl += batch_kl
                n_updates += 1
                epoch_batches += 1

            epochs_completed += 1

            # KL early stopping: if mean KL this epoch exceeds target, the
            # data is too stale to update safely — stop before damaging the policy.
            if self.target_kl is not None and epoch_batches > 0:
                if (epoch_kl / epoch_batches) > self.target_kl:
                    break

        n_updates = max(n_updates, 1)
        ev = explained_variance(
            self.buffer.values[: self.n_steps], self.buffer.returns[: self.n_steps]
        )

        mean_entropy = total_entropy / n_updates
        self._last_entropy = mean_entropy  # used by entropy floor in _update_annealing

        return {
            "policy_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": mean_entropy,
            "clip_fraction": total_clip_fraction / n_updates,
            "explained_variance": ev,
            "approx_kl": total_kl / n_updates,
            "epochs_completed": float(epochs_completed),
        }

    def _update_annealing(self, update_num: int) -> None:
        """Apply entropy annealing and linear LR decay (Charlesworth-style)."""
        # Entropy annealing
        if self.use_entropy_annealing:
            start_u = self._entropy_anneal_start
            end_u = self._entropy_anneal_end
            if start_u <= update_num <= end_u:
                frac = (update_num - start_u) / max(end_u - start_u, 1)
                self.entropy_coef = self._entropy_coef_start + frac * (
                    self._entropy_coef_final - self._entropy_coef_start
                )
            elif update_num > end_u:
                self.entropy_coef = self._entropy_coef_final

        # Entropy floor: if the policy has become nearly deterministic, temporarily
        # raise entropy_coef to restore exploration. This fires after annealing has
        # set entropy_coef, so it only overrides when truly needed.
        if self._last_entropy < self.entropy_floor:
            self.entropy_coef = max(self.entropy_coef, self.entropy_floor_coef)

        # Linear LR decay
        if self.use_linear_lr_decay:
            num_updates = self.total_timesteps // self.n_steps
            if update_num < num_updates:
                frac = 1.0 - update_num / num_updates
                lr = self.lr_final + frac * (self.config["learning_rate"] - self.lr_final)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

    def train(self, verbose: bool = True) -> None:
        """Main training loop.

        Alternates between:
          1. Collecting experience (playing games)
          2. Updating the policy (learning from experience)

        Charlesworth-style: entropy annealing, linear LR decay, update-based eval.
        """
        print(
            f"Starting training for {self.total_timesteps:,} timesteps "
            f"({self.n_envs} envs, {self.n_steps} steps/update)..."
        )
        start_time = time.time()
        last_eval_step = 0
        last_checkpoint_step = 0
        eval_interval = self.config.get("eval_freq", self.checkpoint_freq)
        # Update count for annealing (resume-safe: derived from global_step)
        update_num = self.global_step // self.n_steps

        try:
            while self.global_step < self.total_timesteps:
                # ── Annealing (before collect, so update uses correct LR/entropy) ─
                self._update_annealing(update_num)

                # ── Collect experience ────────────────────────────────────
                rollout_stats = self.collect_rollouts()

                # ── Update policy ─────────────────────────────────────────
                update_stats = self.update()
                update_num += 1

                # ── League: add policy every N updates (Charlesworth-style) ─
                self.league.maybe_add(update_num, self._raw_policy_state_dict())

                # ── Logging ───────────────────────────────────────────────
                elapsed = time.time() - start_time
                fps = self.global_step / max(elapsed, 1e-8)

                if verbose:
                    wr = rollout_stats.get("mean_win_rate", 0)
                    mr = rollout_stats.get("mean_reward", 0)
                    ev = update_stats.get("explained_variance", 0)
                    ent = update_stats.get("entropy", 0)
                    kl = update_stats.get("approx_kl", 0)
                    ep = int(update_stats.get("epochs_completed", self.n_epochs))
                    print(
                        f"Step {self.global_step:>8,} | "
                        f"WR {wr:.2f} | "
                        f"R {mr:+.3f} | "
                        f"EV {ev:.2f} | "
                        f"Ent {ent:.3f} | "
                        f"KL {kl:.4f} | "
                        f"Ep {ep}/{self.n_epochs} | "
                        f"FPS {fps:.0f}"
                    )

                # TensorBoard logging
                for key, val in rollout_stats.items():
                    self.writer.add_scalar(f"rollout/{key}", val, self.global_step)
                for key, val in update_stats.items():
                    self.writer.add_scalar(f"train/{key}", val, self.global_step)
                self.writer.add_scalar("train/fps", fps, self.global_step)
                self.writer.add_scalar("train/entropy_coef", self.entropy_coef, self.global_step)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

                # ── Periodic evaluation ───────────────────────────────────
                if self.global_step - last_eval_step >= eval_interval:
                    self.policy.eval()
                    eval_stats = self.eval_manager.evaluate(self.policy, self.device)
                    self.policy.train()

                    wr_now = eval_stats["win_rate"]
                    print(
                        f"  [EVAL vs {self._eval_opponent}] Win rate: {wr_now:.2f}, "
                        f"Avg VP: {eval_stats['avg_vp']:.1f}, "
                        f"Avg Length: {eval_stats['avg_game_length']:.0f}"
                    )

                    for key, val in eval_stats.items():
                        self.writer.add_scalar(f"eval/{key}", val, self.global_step)
                    self.writer.add_scalar(
                        "eval/opponent_is_heuristic",
                        1.0 if self._eval_opponent == "heuristic" else 0.0,
                        self.global_step,
                    )

                    # Upgrade eval opponent from random → heuristic once threshold is crossed
                    if self._eval_opponent == "random" and wr_now >= self._eval_upgrade_threshold:
                        self._eval_opponent = "heuristic"
                        self.eval_manager = EvaluationManager(
                            n_games=self.config.get("eval_games", 40),
                            opponent_type="heuristic",
                            max_turns=self.config.get("max_turns", 500),
                        )
                        # Reset win-rate history — old random-opponent values are not comparable
                        self._eval_win_rate_history = []
                        print(
                            f"  [EVAL] Upgraded eval opponent to HEURISTIC "
                            f"(WR {wr_now:.2f} >= {self._eval_upgrade_threshold:.2f})"
                        )
                        self.writer.add_scalar("eval/opponent_upgraded", 1.0, self.global_step)

                    # Stagnation detection: warn if win rate hasn't improved
                    self._eval_win_rate_history.append(wr_now)
                    window = self._eval_win_rate_history[-self.stagnation_window :]
                    if len(window) >= self.stagnation_window:
                        improvement = max(window) - min(window)
                        if improvement < self.stagnation_threshold:
                            print(
                                f"  [WARN] STAGNATION detected: win rate range "
                                f"{min(window):.2f}–{max(window):.2f} over last "
                                f"{self.stagnation_window} evals (< {self.stagnation_threshold} "
                                f"improvement). Check EV and entropy in TensorBoard."
                            )
                            self.writer.add_scalar("train/stagnation_flag", 1.0, self.global_step)

                    last_eval_step = self.global_step

                    # Early stopping check
                    target = self.config.get("win_rate_target")
                    if target and wr_now >= target:
                        print(f"  Win rate target {target} reached! Saving final model.")
                        self.save(os.path.join(self.checkpoint_dir, "best_model.pt"))
                        break

                # ── Periodic checkpoint ───────────────────────────────────
                if self.global_step - last_checkpoint_step >= self.checkpoint_freq:
                    self.save(
                        os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step:08d}.pt")
                    )
                    last_checkpoint_step = self.global_step
        except KeyboardInterrupt:
            # Graceful interrupt: always save a checkpoint you can resume from.
            interrupt_path = os.path.join(
                self.checkpoint_dir,
                f"interrupt_{self.global_step:08d}.pt",
            )
            print("\nKeyboardInterrupt received. Saving interrupt checkpoint...")
            self.save(interrupt_path)
            self.writer.close()
            print(
                f"Training interrupted at step {self.global_step:,}. "
                f"Checkpoint saved to {interrupt_path}."
            )
            return

        # Final save
        self.save(os.path.join(self.checkpoint_dir, "final_model.pt"))
        self.writer.close()
        print(f"Training complete. Total steps: {self.global_step:,}")

    def save(self, path: str) -> None:
        """Save everything needed to resume training."""
        torch.save(
            {
                "policy_state_dict": self._raw_policy_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "value_normalizer": self.value_normalizer.state_dict(),
                "config": self.config,
                "eval_win_rate_history": self._eval_win_rate_history,
                "eval_opponent": self._eval_opponent,
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    @classmethod
    def load(cls, path: str, config_override: dict | None = None) -> "CatanPPO":
        """Load a saved trainer, optionally overriding config values."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config_override:
            config.update(config_override)

        trainer = cls(config)
        # torch.compile wraps policy in OptimizedModule (_orig_mod prefix in state_dict keys).
        # Load into the underlying model to handle checkpoints saved before compile was enabled.
        policy_target = (
            trainer.policy._orig_mod if hasattr(trainer.policy, "_orig_mod") else trainer.policy
        )
        policy_target.load_state_dict(checkpoint["policy_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Force LR from config — overrides the stale LR stored in optimizer state.
        # This is needed when resuming with a deliberately reset learning rate.
        for g in trainer.optimizer.param_groups:
            g["lr"] = config["learning_rate"]
        trainer.global_step = checkpoint["global_step"]
        if "value_normalizer" in checkpoint:
            trainer.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        if "eval_win_rate_history" in checkpoint:
            trainer._eval_win_rate_history = checkpoint["eval_win_rate_history"]
        if "eval_opponent" in checkpoint:
            trainer._eval_opponent = checkpoint["eval_opponent"]
            trainer.eval_manager = EvaluationManager(
                n_games=config.get("eval_games", 40),
                opponent_type=trainer._eval_opponent,
                max_turns=config.get("max_turns", 500),
            )

        print(
            f"Loaded checkpoint from {path} (step {trainer.global_step:,}, "
            f"eval opponent: {trainer._eval_opponent})"
        )
        return trainer
