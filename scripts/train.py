"""
Training entry point for Catan RL.

Charlesworth-style: single self-play phase from start (no curriculum).
Trains against random initially; league of past policies populated for
future policy-opponent support.

Usage:
    python scripts/train.py --verbose
    python scripts/train.py --resume checkpoints/train/final_model.pt --verbose
"""

import argparse
import os
import sys
import warnings

# Suppress known harmless PyTorch transformer warning (norm_first + nested tensor)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*norm_first was True")

# Allow `python scripts/train.py` without `pip install -e .` by adding src/ to path.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from catan_rl.algorithms.ppo.arguments import resolve_config
from catan_rl.algorithms.ppo.trainer import CatanPPO


def main():
    parser = argparse.ArgumentParser(description="Train Catan RL agent (Charlesworth-style)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config under configs/ (e.g. configs/phase0_baseline.yaml)",
    )
    parser.add_argument(
        "--phase", type=str, default="train", help=argparse.SUPPRESS
    )  # Ignored; kept for compatibility
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from"
    )
    parser.add_argument("--verbose", action="store_true", help="Print training progress to console")
    parser.add_argument(
        "--total-timesteps", type=int, default=None, help="Override total timesteps"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Override TensorBoard log directory"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Override checkpoint save directory"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max (agent+opponent) turns per game; 0 = no limit",
    )
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Override eval/checkpoint frequency in steps",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=None,
        help="Override number of evaluation games per checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Override device"
    )
    args = parser.parse_args()

    # Resolve config: legacy hard-coded defaults, or YAML if --config given.
    config = resolve_config(args.config)

    # CLI overrides
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.log_dir:
        config["log_dir"] = args.log_dir
    else:
        config["log_dir"] = "runs/train"
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    else:
        config["checkpoint_dir"] = "checkpoints/train"
    if args.max_turns is not None:
        config["max_turns"] = None if args.max_turns == 0 else args.max_turns
    if args.n_envs is not None:
        config["n_envs"] = args.n_envs
    if args.checkpoint_freq is not None:
        config["checkpoint_freq"] = args.checkpoint_freq
    if args.eval_games is not None:
        config["eval_games"] = args.eval_games
    if args.device is not None:
        config["device"] = args.device

    # Create or resume trainer
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer = CatanPPO.load(args.resume, config_override=config)
    else:
        trainer = CatanPPO(config)

    # Train
    trainer.train(verbose=args.verbose)


if __name__ == "__main__":
    main()
