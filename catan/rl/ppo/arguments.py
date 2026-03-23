"""
Hyperparameter configuration for PPO training.

Charlesworth-style: single self-play phase from start (no curriculum).
Adapted for M1 Pro (8-core CPU): CPU-only, in-process multi-env,
entropy annealing, linear LR decay, league of past policies.
"""
from typing import Dict, Any


# Single phase: Charlesworth-style self-play (League = in-memory pool of past policies)
TRAIN_CONFIG: Dict[str, Any] = {
    "learning_rate": 1e-4,        # reset from decayed floor; will decay again to lr_final
    "lr_final": 1e-5,
    "use_linear_lr_decay": True,
    "entropy_coef": 0.01,
    "entropy_coef_start": 0.04,
    "entropy_coef_final": 0.005,
    "entropy_coef_anneal_start": 500,   # ~8.2M steps; keep exploration until league matures
    "entropy_coef_anneal_end": 3000,    # ~49M steps
    "clip_range": 0.2,
    "target_kl": 0.025,          # was 0.015; raised to allow ~6-8 epochs instead of 3
    "n_steps": 4096,             # longer rollout for long-horizon credit assignment
    "n_envs": 8,
    "batch_size": 512,
    "n_epochs": 6,                # reduced from 15; PPO update was 67% of wall time
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "weight_decay": 1e-4,
    "recompute_returns": True,
    "normalize_values": True,
    "total_timesteps": 200_000_000,  # extended from 100M; LR decays from 1e-4 over remaining steps
    "checkpoint_freq": 500_000,
    "eval_games": 40,
    "opponent_type": "random",         # start vs random; auto-upgrades to heuristic
    "eval_upgrade_threshold": 0.95,    # switch eval opponent to heuristic at this WR
    "max_turns": 500,
    "win_rate_target": None,  # no early stopping (Charlesworth style)
    # League (Charlesworth-style): add policy every N updates
    "league_maxlen": 100,
    "add_policy_every": 4,
    "league_random_weight": 0.05,   # 5% chaos injection; avoids early self-play echo chamber
    "heuristic_opponent_weight": 0.25,  # 25% heuristic as stable anchor until WR >90%
    # Entropy floor: if policy entropy drops below this, coef is temporarily raised
    # to prevent premature convergence to a deterministic policy.
    "entropy_floor": 0.003,
    "entropy_floor_coef": 0.01,
    # Stagnation detection: warn if win rate doesn't improve over this many evals.
    "stagnation_window": 20,       # was 10; longer window avoids false positives at high WR
    "stagnation_threshold": 0.01,  # was 0.03; smaller delta acceptable at 95%+ win rate
    # torch.compile: disabled — reduce-overhead mode uses CUDA graphs (GPU-only);
    # on CPU it adds per-call overhead that outweighs any kernel fusion gains.
    "torch_compile": False,
    "eval_freq": 100_000,       # evaluate every 100k steps
    # checkpoint_freq controls save interval (500k); eval_freq controls eval interval (100k)
}

# Shared across all phases — defines the neural network shape.
# Keys must match build_agent_model.DEFAULT_MODEL_CONFIG.
MODEL_CONFIG: Dict[str, Any] = {
    "obs_output_dim": 512,
    "tile_in_dim": 79,
    "tile_model_dim": 128,
    "curr_player_main_in_dim": 166,
    "other_player_main_in_dim": 173,
    "dev_card_embed_dim": 64,
    "dev_card_model_dim": 64,
    "tile_model_num_heads": 4,
    "proj_dev_card_dim": 25,
    "dev_card_model_num_heads": 4,
    "tile_encoder_num_layers": 2,
    "proj_tile_dim": 25,
    "action_head_hidden_dim": 128,
    "value_hidden_dims": (256, 128),
    "dropout": 0.0,
}


def get_config(phase: str = "train") -> Dict[str, Any]:
    """Merge training config with model architecture config.

    Single phase (Charlesworth-style): no curriculum, self-play from start.

    Args:
        phase: Ignored for compatibility; always returns TRAIN_CONFIG.

    Returns:
        Single dict containing everything needed to create CatanPPO + CatanPolicy.
    """
    config = {}
    config.update(MODEL_CONFIG)
    config.update(TRAIN_CONFIG)
    return config
