"""
Training entry point for the decoupled setup model.

The setup model is trained independently from the main-game model.
It learns settlement and road placement via Monte Carlo rollouts:
  1. The setup model places 2 settlements + 2 roads.
  2. K full-game heuristic rollouts estimate the position's value.
  3. PPO updates the setup model based on average margin of victory.

Usage:
    # Start from scratch
    python scripts/train_setup.py --verbose

    # Resume from checkpoint
    python scripts/train_setup.py --resume checkpoints/setup/setup_final.pt --verbose

    # Quick smoke test (20 episodes, 2 rollouts/game)
    python scripts/train_setup.py --total-episodes 20 --n-rollouts 2 --verbose

Prerequisites:
    Train the main model first with scripts/train.py.
    Once the main model achieves ≥60% win rate vs heuristic, the setup
    model can be trained. (For now, heuristic rollouts are used as a proxy
    for the main model — see docs/SETUP_MODEL_PLAN.md for the full roadmap.)
"""
import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor.*norm_first")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.rl.setup.setup_trainer import SetupTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Catan RL setup model (decoupled Monte Carlo rollouts)"
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to setup checkpoint .pt file to resume from")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress to console")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Override total training episodes (default: 100,000)")
    parser.add_argument("--n-rollouts", type=int, default=None,
                        help="Override rollouts per episode (default: 20; start with 20, increase to 50 if noisy)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override number of parallel environments (default: 4)")
    parser.add_argument("--episodes-per-update", type=int, default=None,
                        help="Override episodes collected per PPO update (default: 128)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint save directory")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Override checkpoint frequency in episodes")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Override compute device")
    args = parser.parse_args()

    # Build config from CLI args
    config = {}
    if args.total_episodes is not None:
        config["total_episodes"]       = args.total_episodes
    if args.n_rollouts is not None:
        config["n_rollouts"]           = args.n_rollouts
    if args.n_envs is not None:
        config["n_envs"]               = args.n_envs
    if args.episodes_per_update is not None:
        config["episodes_per_update"]  = args.episodes_per_update
    if args.log_dir is not None:
        config["log_dir"]              = args.log_dir
    else:
        config["log_dir"]              = "runs/setup"
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"]       = args.checkpoint_dir
    else:
        config["checkpoint_dir"]       = "checkpoints/setup"
    if args.checkpoint_freq is not None:
        config["checkpoint_freq"]      = args.checkpoint_freq
    if args.device is not None:
        config["device"]               = args.device

    # Create or resume trainer
    if args.resume:
        print(f"Resuming setup training from {args.resume}")
        trainer = SetupTrainer.load(args.resume, config_override=config)
    else:
        trainer = SetupTrainer(config)

    trainer.train(verbose=args.verbose)


if __name__ == "__main__":
    main()
