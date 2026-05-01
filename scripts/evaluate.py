"""
Evaluate a trained Catan RL agent.

Usage:
    python scripts/evaluate.py checkpoints/train/final_model.pt --n-games 100
    python scripts/evaluate.py checkpoints/train/final_model.pt --opponent heuristic
"""

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: F401

from catan_rl.algorithms.ppo.trainer import CatanPPO
from catan_rl.eval.evaluation_manager import EvaluationManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate Catan RL agent")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--n-games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "heuristic"],
        help="Opponent type",
    )
    args = parser.parse_args()

    # Load trainer (just for the policy)
    trainer = CatanPPO.load(args.checkpoint)
    trainer.policy.eval()

    evaluator = EvaluationManager(
        n_games=args.n_games,
        opponent_type=args.opponent,
    )

    print(f"Evaluating against {args.opponent} over {args.n_games} games...")
    stats = evaluator.evaluate(trainer.policy, trainer.device)

    print("\nResults:")
    print(f"  Win Rate:        {stats['win_rate']:.1%}")
    print(f"  Average VP:      {stats['avg_vp']:.1f}")
    print(f"  Average Length:  {stats['avg_game_length']:.0f} steps")


if __name__ == "__main__":
    main()
