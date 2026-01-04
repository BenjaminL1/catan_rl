import torch
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
import glob
from catan.rl.env import CatanEnv
from stable_baselines3.common.env_checker import check_env
import sys
import os

# Ensure we can import from the catan directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_wrapper_attr("get_action_mask")()


def make_env():
    env = CatanEnv(render_mode=None)
    env = Monitor(env)  # For logging
    env = ActionMasker(env, mask_fn)  # For action masking
    return env


def main():
    # Define paths - NEW DIRECTORY FOR LARGER MODEL
    models_dir = "models/PPO_Large_v4"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize environment
    # env = make_env() # Old way

    # New way with VecNormalize
    env = DummyVecEnv([make_env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Check environment
    print("Checking environment...")
    # check_env expects a Gym env, ActionMasker is a Wrapper which is an Env.
    # However, check_env might not like ActionMasker if it doesn't expose everything.
    # But let's try.
    # check_env(env, warn=True)
    print("Environment check skipped (ActionMasker wrapper).")

    # Callback for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="catan_ppo"
    )

    # Check for existing models
    existing_models = glob.glob(f"{models_dir}/*.zip")

    if existing_models:
        # Find the latest model based on modification time
        latest_model = max(existing_models, key=os.path.getmtime)
        print(f"Loading latest model: {latest_model}")

        try:
            # Load the model
            model = MaskablePPO.load(latest_model, env=env, tensorboard_log=log_dir)
            print(f"Resuming training from {latest_model}...")

            # FORCE the Learning Rate (Both Schedule and Optimizer)
            new_lr = 5e-5  # 0.00005

            # A. Update the internal scheduler function
            model.lr_schedule = lambda _: new_lr

            # B. Update the current value in the model
            model.learning_rate = new_lr

            # C. Update the active optimizer
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr

            # Ensure Safety Settings
            model.clip_range = lambda _: 0.1
            model.target_kl = 0.05  # Relax slightly to 0.05
            print(f"Resuming with FORCED LR: {new_lr}")

            reset_timesteps = False
        except Exception as e:
            print(f"Failed to load model {latest_model}: {e}")
            print("Starting fresh training instead.")
            # Define larger network architecture
            policy_kwargs = dict(net_arch=[256, 256])
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=3e-4,       # Fast learning rate
                n_steps=2048,
                batch_size=64,
                gamma=0.995,              # Focus on long-term rewards
                ent_coef=0.01,            # 1% Randomness for exploration
                clip_range=0.2,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs  # Apply larger network
            )
            reset_timesteps = True
    else:
        print("No existing model found. Creating new LARGE model...")
        # Define larger network architecture
        policy_kwargs = dict(net_arch=[256, 256])

        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,       # Fast learning rate
            n_steps=2048,
            batch_size=64,
            gamma=0.995,              # Focus on long-term rewards
            ent_coef=0.01,            # 1% Randomness for exploration
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs  # Apply larger network
        )
        reset_timesteps = True

    print("Starting training...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback,
                    reset_num_timesteps=reset_timesteps)
        model.save(f"{models_dir}/catan_ppo_final")
        print("Training complete. Model saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save(f"{models_dir}/catan_ppo_interrupted")
        print("Model saved.")


if __name__ == "__main__":
    main()
