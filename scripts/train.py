from stable_baselines3.common.env_checker import check_env
import glob
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import sys
import os
from pathlib import Path

# FIX: Disable strict probability checks to prevent Simplex errors on small FP deviations
torch.distributions.Distribution.set_default_validate_args(False)

# Get the absolute path of the 'CATAN_RL' root directory
# Path(__file__) is scripts/train.py -> .parent is scripts/ -> .parent is CATAN_RL/
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Now this should work
from catan.rl.env import CatanEnv
from catan.rl.debug_wrapper import CatanCrashDebugWrapper


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_wrapper_attr("get_action_mask")()


def make_env():
    env = CatanEnv(render_mode=None)
    env = CatanCrashDebugWrapper(env)
    env = Monitor(env)  # For logging
    env = ActionMasker(env, mask_fn)  # For action masking
    return env


class HallOfFameUpdateCallback(BaseCallback):
    def __init__(self, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            try:
                # Trigger scan in env (DummyVecEnv wrapper)
                self.training_env.env_method("_scan_model_pool")
                if self.verbose > 0:
                    print("Hall of Fame Pool Updated.")
            except Exception as e:
                print(f"HoF Update Failed: {e}")
        return True


class LeagueUpdateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if np.any(dones):
            try:
                # Get access to the league object from the environment
                leagues = self.training_env.get_attr("league")
                if not leagues:
                    return True

                # Use the first one (assuming DummyVecEnv)
                league = leagues[0]

                for i, done in enumerate(dones):
                    if done and i < len(infos):
                        info = infos[i]
                        if 'league_update' in info:
                            update = info['league_update']
                            # Update stats
                            league.update_match_result(update['opponent'], update['win'])
                            league.save_data()

                            # if self.verbose > 0:
                            #     res = "WIN" if update['win'] else "LOSS"
                            #     print(f"League Match: Agent vs {update['opponent']} -> {res}")

                # Log Average League Difficulty
                try:
                    avg_difficulty = league.get_avg_difficulty()
                    self.logger.record("league/avg_difficulty", avg_difficulty)
                except Exception as e:
                    pass

            except Exception as e:
                # pass silently or print
                if self.verbose > 0:
                    print(f"League match update error: {e}")

        return True


def main():
    # Define paths - NEW DIRECTORY FOR LARGER MODEL
    models_dir = "models"
    hof_dir = "models/hof"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(hof_dir):
        os.makedirs(hof_dir)

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
    SAVE_FREQ = 1000000

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=hof_dir,  # Save to HoF
        name_prefix="catan_ppo"
    )

    hof_callback = HallOfFameUpdateCallback(save_freq=SAVE_FREQ, verbose=1)
    league_callback = LeagueUpdateCallback(verbose=1)

    callback_list = CallbackList([checkpoint_callback, hof_callback, league_callback])

    # Priority Loading: 1. catan_ppo_final.zip  2. Latest Checkpoint
    final_path = os.path.join(models_dir, "catan_ppo_final.zip")
    latest_model = None

    if os.path.exists(final_path):
        latest_model = final_path
        print(f"Loading primary model: {latest_model}")
    else:
        root_models = glob.glob(f"{models_dir}/*.zip")
        hof_models = glob.glob(f"{hof_dir}/*.zip")
        candidates = root_models + hof_models
        if candidates:
            latest_model = max(candidates, key=os.path.getmtime)
            print(f"Primary model missing. Loading latest checkpoint: {latest_model}")

    if latest_model:

        try:
            # Load the model - Force CPU to avoid MPS/Simplex errors
            model = MaskablePPO.load(latest_model, env=env, tensorboard_log=log_dir, device='cpu')
            print(f"Resuming training from {latest_model}...")

            # FORCE the Learning Rate (Both Schedule and Optimizer)
            new_lr = 5e-6  # Decayed to 0.000005 for Hall of Fame Self-Play Fine-Tuning

            # A. Update the internal scheduler function
            model.lr_schedule = lambda _: new_lr

            # B. Update the current value in the model
            model.learning_rate = new_lr

            # C. Update the active optimizer
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr

            # Adjust Entropy Coefficient for Self-Play Exploration
            # REDUCED from 0.015 to 0.005 to encourage more "serious" play and recover win rate
            model.ent_coef = 0.005

            # Update Batch Size and N_Steps for Stability
            model.n_steps = 4096
            model.batch_size = 256
            model.rollout_buffer.buffer_size = 4096
            model.rollout_buffer.n_steps = 4096

            # Reset buffer to handle new size
            model.rollout_buffer.reset()

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
                verbose=0,
                tensorboard_log=log_dir,
                learning_rate=1e-5,       # Slower learning rate for self-play
                n_steps=2048,
                batch_size=64,
                gamma=0.995,              # Focus on long-term rewards
                ent_coef=0.015,           # 1.5% Randomness for exploration
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
            verbose=0,
            tensorboard_log=log_dir,
            learning_rate=1e-5,       # Slower learning rate for self-play
            n_steps=2048,
            batch_size=64,
            gamma=0.995,              # Focus on long-term rewards
            ent_coef=0.015,           # 1.5% Randomness for exploration
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs  # Apply larger network
        )
        reset_timesteps = True

    print("Starting training...")
    try:
        # Extended to 3M steps to allow full curriculum saturation
        model.learn(total_timesteps=3000000, callback=callback_list,
                    reset_num_timesteps=reset_timesteps)
        model.save(f"{models_dir}/catan_ppo_final")
        print("Training complete. Model saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save(f"{models_dir}/catan_ppo_interrupted")
        print("Model saved.")


if __name__ == "__main__":
    main()
