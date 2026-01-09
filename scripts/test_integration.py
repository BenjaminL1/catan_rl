import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from catan.rl.env import CatanEnv
from catan.engine.dice import StackedDice


def test_integration():
    print("Initializing CatanEnv...")
    env = CatanEnv()

    # 1. Check Observation Space
    obs_shape = env.observation_space.shape
    print(f"Observation Space Shape: {obs_shape}")
    assert obs_shape == (1258,), f"Expected (1258,), got {obs_shape}"
    print("PASS: Observation Space Shape")

    # 2. Check Action Space
    action_shape = env.action_space.n
    print(f"Action Space Size: {action_shape}")
    assert action_shape == 246, f"Expected 246, got {action_shape}"
    print("PASS: Action Space Size")

    # 3. Test Reset
    obs, info = env.reset()
    assert obs.shape == (1258,), "Reset observation shape mismatch"
    print("PASS: Environment Reset")

    # 4. Test Random Actions
    print("Testing random actions...")
    for _ in range(10):
        action = env.action_space.sample()
        # Use mask to pick valid action
        mask = env.get_action_mask()
        if not mask[action]:
            # Pick a valid one
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                print("No valid actions!")
                break
        
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (1258,), "Step observation shape mismatch"
        if terminated or truncated:
            env.reset()
    
    print("PASS: Random Actions Execution")
    print("ALL INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_integration()
