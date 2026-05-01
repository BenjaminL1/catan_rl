"""
PPO trainer for the decoupled setup model.

Episode structure:
  - 4 steps per episode (settle1, road1, settle2, road2)
  - Rewards: 0 for steps 0-2; margin-of-victory at step 3
  - γ = 1.0 (no temporal discounting for 4-step episodes)

Training loop (Charlesworth-style):
  1. Collect n_episodes_per_update × 4 steps of rollouts
  2. Compute returns (with γ=1.0, all steps in an episode get the same return)
  3. Run n_epochs of PPO updates

Checkpointing: saves both the setup policy and the best margin achieved.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from catan_rl.setup_phase.setup_env import N_EDGES, N_VERTICES, OBS_DIM, SetupEnv

N_CORNERS = N_VERTICES  # alias: corner = vertex placement head size
from catan_rl.models.utils import ValueFunctionNormalizer
from catan_rl.setup_phase.setup_policy import SetupPolicy

# Number of vector action dimensions stored per transition
_ACTION_DIM = 2  # [corner_idx, edge_idx]


class SetupTrainer:
    """PPO trainer for the Catan setup phase.

    Usage:
        trainer = SetupTrainer(config)
        trainer.train()

    Configuration keys (with defaults):
        n_envs           : 4      — parallel environments
        n_rollouts       : 20     — game rollouts per episode to estimate value
        max_game_turns   : 500    — safety cap per rollout game
        episodes_per_update: 128  — episodes collected before each PPO update
        n_epochs         : 10     — PPO update epochs
        batch_size       : 64     — minibatch size (in transitions)
        learning_rate    : 3e-4
        lr_final         : 1e-5
        use_linear_lr_decay: True
        gamma            : 1.0    — no time-discounting (4-step episodes)
        gae_lambda       : 1.0
        clip_range       : 0.2
        entropy_coef     : 0.01
        value_coef       : 0.5
        max_grad_norm    : 0.5
        total_episodes   : 100_000
        checkpoint_freq  : 1_000  — episodes between checkpoints
        log_dir          : "runs/setup"
        checkpoint_dir   : "checkpoints/setup"
        obs_dim          : 1417
        hidden_dim       : 256
        head_hidden      : 128
        device           : "cpu"
    """

    DEFAULTS: dict = {
        "n_envs": 4,
        "n_rollouts": 20,
        "max_game_turns": 500,
        "episodes_per_update": 128,
        "n_epochs": 10,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "lr_final": 1e-5,
        "use_linear_lr_decay": True,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_episodes": 100_000,
        "checkpoint_freq": 1_000,
        "log_dir": "runs/setup",
        "checkpoint_dir": "checkpoints/setup",
        "obs_dim": OBS_DIM,
        "hidden_dim": 256,
        "head_hidden": 128,
        "device": "cpu",
    }

    def __init__(self, config: dict | None = None):
        cfg = dict(self.DEFAULTS)
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.device = cfg["device"]

        self.policy = SetupPolicy(
            obs_dim=cfg["obs_dim"],
            hidden_dim=cfg["hidden_dim"],
            head_hidden=cfg["head_hidden"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"SetupPolicy: {n_params:,} trainable parameters")

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=1e-4,
        )

        self.n_envs = cfg["n_envs"]
        self.n_rollouts = cfg["n_rollouts"]
        self.episodes_per_update = cfg["episodes_per_update"]
        self.n_epochs = cfg["n_epochs"]
        self.batch_size = cfg["batch_size"]
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_range = cfg["clip_range"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_coef = cfg["value_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.total_episodes = cfg["total_episodes"]
        self.checkpoint_freq = cfg["checkpoint_freq"]
        self.use_lr_decay = cfg["use_linear_lr_decay"]
        self.lr_final = cfg["lr_final"]

        self.value_normalizer = ValueFunctionNormalizer()

        # Environments
        self.envs = [
            SetupEnv(
                n_rollouts=cfg["n_rollouts"],
                max_game_turns=cfg["max_game_turns"],
            )
            for _ in range(self.n_envs)
        ]

        # Logging
        os.makedirs(cfg["log_dir"], exist_ok=True)
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        self.writer = SummaryWriter(cfg["log_dir"])

        # State
        self.global_episodes = 0
        self._margin_history: list[float] = []

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> None:
        """Run the training loop until total_episodes is reached."""
        print(
            f"SetupTrainer: training for {self.total_episodes:,} episodes "
            f"({self.n_envs} envs, {self.n_rollouts} rollouts/episode)..."
        )
        start_time = time.time()
        update_num = 0

        try:
            while self.global_episodes < self.total_episodes:
                # ── Anneal LR ────────────────────────────────────────────────
                if self.use_lr_decay:
                    n_updates = self.total_episodes // self.episodes_per_update
                    frac = 1.0 - update_num / max(n_updates, 1)
                    lr = self.lr_final + frac * (self.cfg["learning_rate"] - self.lr_final)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

                # ── Collect rollouts ─────────────────────────────────────────
                buffer = self._collect_rollouts()
                update_num += 1

                # ── PPO update ───────────────────────────────────────────────
                update_stats = self._update(buffer)

                self.global_episodes += buffer["n_episodes"]

                # ── Logging ──────────────────────────────────────────────────
                elapsed = time.time() - start_time
                eps_per_sec = self.global_episodes / max(elapsed, 1e-8)
                mean_margin = float(np.mean(buffer["margins"])) if buffer["margins"] else 0.0
                self._margin_history.append(mean_margin)

                if verbose:
                    print(
                        f"Episode {self.global_episodes:>7,} | "
                        f"Margin {mean_margin:+.3f} | "
                        f"PG {update_stats['pg_loss']:.4f} | "
                        f"V {update_stats['value_loss']:.4f} | "
                        f"Ent {update_stats['entropy']:.3f} | "
                        f"KL {update_stats['approx_kl']:.4f} | "
                        f"EP/s {eps_per_sec:.1f}"
                    )

                self.writer.add_scalar("setup/mean_margin", mean_margin, self.global_episodes)
                self.writer.add_scalar(
                    "setup/pg_loss", update_stats["pg_loss"], self.global_episodes
                )
                self.writer.add_scalar(
                    "setup/value_loss", update_stats["value_loss"], self.global_episodes
                )
                self.writer.add_scalar(
                    "setup/entropy", update_stats["entropy"], self.global_episodes
                )
                self.writer.add_scalar(
                    "setup/approx_kl", update_stats["approx_kl"], self.global_episodes
                )
                self.writer.add_scalar("setup/episodes_per_s", eps_per_sec, self.global_episodes)

                # ── Checkpoint ───────────────────────────────────────────────
                if self.global_episodes % self.checkpoint_freq < self.episodes_per_update:
                    path = os.path.join(
                        self.cfg["checkpoint_dir"],
                        f"setup_{self.global_episodes:08d}.pt",
                    )
                    self.save(path)

        except KeyboardInterrupt:
            interrupt_path = os.path.join(
                self.cfg["checkpoint_dir"],
                f"setup_interrupt_{self.global_episodes:08d}.pt",
            )
            print(f"\nInterrupted. Saving to {interrupt_path}")
            self.save(interrupt_path)
            self.writer.close()
            return

        self.save(os.path.join(self.cfg["checkpoint_dir"], "setup_final.pt"))
        self.writer.close()
        print(f"Setup training complete. {self.global_episodes:,} episodes.")

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollouts(self) -> dict:
        """Collect episodes_per_update episodes across n_envs environments.

        Returns a dict of numpy arrays ready for PPO update.
        """
        self.policy.eval()

        n_eps_per_env = max(1, self.episodes_per_update // self.n_envs)
        n_total_eps = n_eps_per_env * self.n_envs
        steps_per_ep = 4  # always exactly 4 setup decisions
        n_steps = n_total_eps * steps_per_ep

        # Pre-allocate storage
        obs_buf = np.zeros((n_steps, self.cfg["obs_dim"]), dtype=np.float32)
        corner_mask_buf = np.zeros((n_steps, N_CORNERS), dtype=bool)
        edge_mask_buf = np.zeros((n_steps, N_EDGES), dtype=bool)
        is_corner_buf = np.zeros(n_steps, dtype=bool)
        actions_buf = np.zeros((n_steps, _ACTION_DIM), dtype=np.int64)
        rewards_buf = np.zeros(n_steps, dtype=np.float32)
        dones_buf = np.zeros(n_steps, dtype=bool)
        values_buf = np.zeros(n_steps, dtype=np.float32)
        log_probs_buf = np.zeros(n_steps, dtype=np.float32)

        margins = []
        step_ptr = 0

        for env in self.envs:
            obs, _ = env.reset()

            for _ in range(n_eps_per_env):
                ep_obs = []
                ep_corner_masks = []
                ep_edge_masks = []
                ep_is_corner = []
                ep_actions = []
                ep_rewards = []
                ep_dones = []
                ep_values = []
                ep_log_probs = []

                done = False
                while not done:
                    raw_mask = env.get_action_masks()  # (126,) bool
                    setup_step = env._setup_step
                    is_corner = setup_step in (0, 2)

                    # Split flat mask into corner/edge masks
                    corner_mask_np = raw_mask[:N_VERTICES]  # 54
                    edge_mask_np = raw_mask[N_VERTICES:]  # 72

                    # To tensor (B=1)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    cm_t = torch.tensor(
                        corner_mask_np, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)
                    em_t = torch.tensor(
                        edge_mask_np, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)
                    ic_t = torch.tensor([is_corner], dtype=torch.bool, device=self.device)

                    with torch.no_grad():
                        actions_t, value_t, lp_t = self.policy.act(obs_t, cm_t, em_t, ic_t)

                    corner_act = int(actions_t[0, 0].item())
                    edge_act = int(actions_t[0, 1].item())

                    # Convert to flat action for env.step
                    flat_action = corner_act if is_corner else (N_VERTICES + edge_act)

                    ep_obs.append(obs.copy())
                    ep_corner_masks.append(corner_mask_np)
                    ep_edge_masks.append(edge_mask_np)
                    ep_is_corner.append(is_corner)
                    ep_actions.append([corner_act, edge_act])
                    ep_values.append(float(value_t.item()))
                    ep_log_probs.append(float(lp_t.item()))

                    obs, reward, done, truncated, info = env.step(flat_action)
                    ep_rewards.append(reward)
                    ep_dones.append(done or truncated)

                if info.get("margin") is not None:
                    margins.append(info["margin"])

                # Compute returns for this episode (γ=1.0 → all get the final reward)
                ep_returns = _compute_returns(
                    ep_rewards,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    values=ep_values,
                    last_value=0.0,
                )

                # Write to buffer
                for t_local, (ob, cm, em, ic, ac, rw, dn, vl, lp, ret) in enumerate(
                    zip(
                        ep_obs,
                        ep_corner_masks,
                        ep_edge_masks,
                        ep_is_corner,
                        ep_actions,
                        ep_rewards,
                        ep_dones,
                        ep_values,
                        ep_log_probs,
                        ep_returns,
                    )
                ):
                    if step_ptr >= n_steps:
                        break
                    obs_buf[step_ptr] = ob
                    corner_mask_buf[step_ptr] = cm
                    edge_mask_buf[step_ptr] = em
                    is_corner_buf[step_ptr] = ic
                    actions_buf[step_ptr] = ac
                    rewards_buf[step_ptr] = rw
                    dones_buf[step_ptr] = dn
                    values_buf[step_ptr] = vl
                    log_probs_buf[step_ptr] = lp
                    step_ptr += 1

                obs, _ = env.reset()

        n_valid = min(step_ptr, n_steps)

        # Compute advantages (reuse returns stored as ep_returns approximation)
        # For simplicity, advantages = returns - values (no separate GAE pass needed
        # since γ=1.0 and episodes are only 4 steps long).
        returns_buf = np.zeros(n_valid, dtype=np.float32)
        adv_buf = np.zeros(n_valid, dtype=np.float32)

        # Group transitions into episodes (every 4 steps) to compute returns
        ptr = 0
        for ep_start in range(0, n_valid, steps_per_ep):
            ep_end = min(ep_start + steps_per_ep, n_valid)
            ep_rews = rewards_buf[ep_start:ep_end]
            ep_vals = values_buf[ep_start:ep_end]
            ep_rets = _compute_returns(
                list(ep_rews),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                values=list(ep_vals),
                last_value=0.0,
            )
            returns_buf[ep_start:ep_end] = ep_rets
            adv_buf[ep_start:ep_end] = ep_rets - ep_vals

        # Update value normalizer
        self.value_normalizer.update(torch.tensor(returns_buf, dtype=torch.float32))

        return {
            "obs": obs_buf[:n_valid],
            "corner_masks": corner_mask_buf[:n_valid],
            "edge_masks": edge_mask_buf[:n_valid],
            "is_corner": is_corner_buf[:n_valid],
            "actions": actions_buf[:n_valid],
            "returns": returns_buf,
            "advantages": adv_buf,
            "old_log_probs": log_probs_buf[:n_valid],
            "n_steps": n_valid,
            "n_episodes": n_total_eps,
            "margins": margins,
        }

    # ── PPO update ────────────────────────────────────────────────────────────

    def _update(self, buffer: dict) -> dict:
        """Run n_epochs of PPO updates on the collected buffer."""
        self.policy.train()

        n = buffer["n_steps"]
        obs = torch.tensor(buffer["obs"], dtype=torch.float32, device=self.device)
        corner_masks = torch.tensor(buffer["corner_masks"], dtype=torch.bool, device=self.device)
        edge_masks = torch.tensor(buffer["edge_masks"], dtype=torch.bool, device=self.device)
        is_corner = torch.tensor(buffer["is_corner"], dtype=torch.bool, device=self.device)
        actions = torch.tensor(buffer["actions"], dtype=torch.int64, device=self.device)
        returns_t = torch.tensor(buffer["returns"], dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(buffer["advantages"], dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(buffer["old_log_probs"], dtype=torch.float32, device=self.device)

        norm_returns = self.value_normalizer.normalize(returns_t)

        total_pg = 0.0
        total_val = 0.0
        total_ent = 0.0
        total_kl = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                bi = indices[start : start + self.batch_size]

                adv_b = advantages_t[bi]
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                values, new_lp, entropy = self.policy.evaluate_actions(
                    obs[bi], corner_masks[bi], edge_masks[bi], is_corner[bi], actions[bi]
                )
                values = values.squeeze(-1)

                ratio = torch.exp(new_lp - old_lp[bi])
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                val_loss = nn.functional.mse_loss(values, norm_returns[bi])
                ent_loss = -entropy

                loss = pg_loss + self.value_coef * val_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = (old_lp[bi] - new_lp).mean().item()

                total_pg += pg_loss.item()
                total_val += val_loss.item()
                total_ent += entropy.item()
                total_kl += kl
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "pg_loss": total_pg / n_updates,
            "value_loss": total_val / n_updates,
            "entropy": total_ent / n_updates,
            "approx_kl": total_kl / n_updates,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_episodes": self.global_episodes,
                "value_normalizer": self.value_normalizer.state_dict(),
                "config": self.cfg,
                "margin_history": self._margin_history,
            },
            path,
        )
        print(f"  Saved: {path}")

    @classmethod
    def load(cls, path: str, config_override: dict | None = None) -> "SetupTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if config_override:
            cfg.update(config_override)
        trainer = cls(cfg)
        trainer.policy.load_state_dict(ckpt["policy_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.global_episodes = ckpt["global_episodes"]
        if "value_normalizer" in ckpt:
            trainer.value_normalizer.load_state_dict(ckpt["value_normalizer"])
        if "margin_history" in ckpt:
            trainer._margin_history = ckpt["margin_history"]
        print(f"Loaded setup checkpoint from {path} (ep {trainer.global_episodes:,})")
        return trainer


# ── GAE return computation ────────────────────────────────────────────────────


def _compute_returns(
    rewards: list[float],
    gamma: float,
    gae_lambda: float,
    values: list[float],
    last_value: float,
) -> list[float]:
    """Compute GAE returns for a single episode.

    With γ=λ=1.0 (default for 4-step setup episodes), all steps
    receive the same return = sum of rewards (which equals the final
    reward since intermediate rewards are 0).
    """
    T = len(rewards)
    adv = 0.0
    gae_returns = [0.0] * T
    next_val = last_value

    for t in reversed(range(T)):
        next_non_terminal = 0.0 if (t == T - 1) else 1.0
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        adv = delta + gamma * gae_lambda * next_non_terminal * adv
        gae_returns[t] = adv + values[t]
        next_val = values[t]

    return gae_returns
