"""
Evaluation manager: play N games and report win rate.

Used between training rollouts to track progress.
"""
import numpy as np
from typing import Dict, Optional
from catan.rl.env import CatanEnv
import torch


class EvaluationManager:
    """Run deterministic evaluation games and report statistics."""

    def __init__(self, n_games: int = 20, opponent_type: str = "random",
                 max_turns: Optional[int] = 500):
        self.n_games = n_games
        self.opponent_type = opponent_type
        self.max_turns = max_turns

    def evaluate(self, policy, device: str = "cpu") -> Dict[str, float]:
        """Play n_games with the given policy (deterministic) and return stats.

        Args:
            policy: CatanPolicy instance (set to eval mode externally).
            device: torch device string.

        Returns:
            Dict with 'win_rate', 'avg_vp', 'avg_game_length'.
        """
        env = CatanEnv(render_mode=None, opponent_type=self.opponent_type,
                       max_turns=self.max_turns)
        wins = 0
        total_vp = 0
        total_length = 0

        for _ in range(self.n_games):
            obs, _ = env.reset()
            done = False

            while not done:
                masks = env.get_action_masks()

                # Convert dict obs to tensors (B=1)
                def _dev_seq(key: str):
                    seq = obs.get(key, None)
                    if seq is None:
                        return [torch.zeros(1, dtype=torch.long, device=device)]
                    arr = np.asarray(seq, dtype=np.int64)
                    if arr.size == 0:
                        arr = np.zeros(1, dtype=np.int64)
                    return [torch.tensor(arr, dtype=torch.long, device=device)]

                obs_t = {
                    "tile_representations": torch.tensor(
                        obs["tile_representations"], dtype=torch.float32, device=device
                    ).unsqueeze(0),
                    "current_player_main": torch.tensor(
                        obs["current_player_main"], dtype=torch.float32, device=device
                    ).unsqueeze(0),
                    "next_player_main": torch.tensor(
                        obs["next_player_main"], dtype=torch.float32, device=device
                    ).unsqueeze(0),
                    "current_player_hidden_dev": _dev_seq("current_player_hidden_dev"),
                    "current_player_played_dev": _dev_seq("current_player_played_dev"),
                    "next_player_played_dev": _dev_seq("next_player_played_dev"),
                }

                masks_t = {k: torch.tensor(v, dtype=torch.bool, device=device).unsqueeze(0)
                           for k, v in masks.items()}

                with torch.no_grad():
                    actions, _, _ = policy.act(obs_t, masks_t, deterministic=True)

                action_np = actions.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

            stats = info.get('terminal_stats', {})
            if info.get('is_success'):
                wins += 1
            total_vp += stats.get('agent_vp', 0)
            total_length += stats.get('game_length', 0)

        return {
            'win_rate': wins / max(self.n_games, 1),
            'avg_vp': total_vp / max(self.n_games, 1),
            'avg_game_length': total_length / max(self.n_games, 1),
        }
