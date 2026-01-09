import gymnasium as gym
import traceback
import numpy as np
import json


class CatanCrashDebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)

            # UPGRADE: Check for NaNs in Observation
            if not np.isfinite(obs).all():
                self._report_critical_failure("NaN/Inf detected in Observation", action)

            self.last_obs = obs
            return obs, reward, terminated, truncated, info
        except Exception as e:
            self._report_critical_failure(f"Exception in step(): {str(e)}", action)
            raise e

    def get_action_mask(self):
        mask = self.env.get_action_mask()

        # UPGRADE: Pre-emptive Simplex Check
        if not np.any(mask):
            self._report_critical_failure("ALL-ZERO ACTION MASK (Simplex Trigger)", None)
            raise ValueError("Environment returned a mask with no legal actions.")

        return mask

    def _report_critical_failure(self, reason, action):
        inner = self.unwrapped

        # Helper to convert numpy types for JSON serialization
        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        report = {
            "Reason": reason,
            "Action_Attempted": serialize(action),
            "Step": getattr(inner, 'current_step', 'Unknown'),
            "Phase": {
                "Robber_Pending": getattr(inner, 'robber_placement_pending', False),
                "Road_Building_Left": getattr(inner, 'road_building_roads_left', 0),
                "Dev_Card_Played": getattr(inner, 'played_dev_card_this_turn', False)
            },
            "Agent_Stats": {
                "VP": inner.agent_player.victoryPoints,
                "Resources": inner.agent_player.resources,
                "Roads_Left": inner.agent_player.roadsLeft
            },
            "Opponent_Type": type(inner.opponent_player).__name__
        }

        print("\n" + "!" * 60)
        print(f"CRITICAL ERROR: {reason}")
        print(json.dumps(report, indent=4))
        print("!" * 60 + "\n", flush=True)

        # Save to a permanent file for long-term training runs
        with open("last_crash_report.json", "w") as f:
            json.dump(report, f, indent=4)
