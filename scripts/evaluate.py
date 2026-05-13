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
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write evaluator stats as JSON to this path. Used by the trainer's async eval path.",  # noqa: E501
    )
    args = parser.parse_args()

    # Load trainer (just for the policy)
    trainer = CatanPPO.load(args.checkpoint)
    trainer.policy.eval()

    # Mirror the trainer's env-side flags so the EvaluationManager's
    # CatanEnv emits the same obs schema the loaded policy expects.
    # Without this, the env emits the legacy 166-dim current_player_main
    # but the phase 1.3+ checkpoint expects the compact 54-dim — and
    # the model crashes with a Linear-shape mismatch on the first step.
    cfg = trainer.config
    evaluator = EvaluationManager(
        n_games=args.n_games,
        opponent_type=args.opponent,
        use_thermometer_encoding=bool(cfg.get("use_thermometer_encoding", True)),
        use_opponent_id_emb=bool(cfg.get("use_opponent_id_emb", False)),
        opp_id_mask_prob=float(cfg.get("opp_id_mask_prob", 0.40)),
        league_maxlen=int(cfg.get("league_maxlen", 100)),
        use_belief_head=bool(cfg.get("use_belief_head", False)),
    )

    print(f"Evaluating against {args.opponent} over {args.n_games} games...")
    stats = evaluator.evaluate(trainer.policy, trainer.device)

    print("\nResults:")
    print(f"  Win Rate:        {stats['win_rate']:.1%}")
    print(f"  Average VP:      {stats['avg_vp']:.1f}")
    print(f"  Average Length:  {stats['avg_game_length']:.0f} steps")

    if args.output_json:
        import json

        # Coerce numpy scalars to plain Python floats so json.dump works.
        out = {k: (float(v) if hasattr(v, "item") else v) for k, v in stats.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
