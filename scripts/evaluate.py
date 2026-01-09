import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import sys

# DISABLE STRICT DISTRIBUTION VALIDATION
# This prevents "Simplex" errors when masking produces probabilities that don't sum to exactly 1.0000000...
torch.distributions.Distribution.set_default_validate_args(False)

import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from catan.rl.env import CatanEnv
from catan.rl.debug_wrapper import CatanCrashDebugWrapper


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_wrapper_attr("get_action_mask")()


def make_env():
    env = CatanEnv(render_mode=None)
    env = CatanCrashDebugWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def evaluate():
    model_path = "models/catan_ppo_final.zip"
    print(f"Loading model from {model_path}...")

    env = make_env()
    model = MaskablePPO.load(model_path, env=env, device='cpu')

    n_episodes = 100
    wins = 0
    total_rewards = []

    print(f"Starting evaluation over {n_episodes} episodes against Heuristic AI...")

    for i in tqdm(range(n_episodes)):
        # FORCE Heuristic Opponent
        obs, _ = env.reset(options={'opponent_type': 'heuristic'})
        done = False
        episode_reward = 0

        while not done:
            action_masks = env.get_wrapper_attr("get_action_mask")()
            action, _ = model.predict(obs, action_masks=action_masks)

            # Cast to int to avoid numpy issues in env
            if isinstance(action, np.ndarray):
                action = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if done:
                total_rewards.append(episode_reward)
                if info.get('is_success', False):
                    wins += 1

    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)
    print(f"Opponent: Heuristic AI (Hard)")
    print(f"Episodes: {n_episodes}")
    print(f"Win Rate: {wins/n_episodes*100:.2f}%")
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print("=" * 30)


if __name__ == "__main__":
    evaluate()
