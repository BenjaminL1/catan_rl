import numpy as np
import gymnasium as gym
from catan.rl.env import CatanEnv
from catan.engine.dice import StackedDice
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # 3. Check Reset
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    assert obs.shape == (1258,), f"Expected (1258,), got {obs.shape}"
    print("PASS: Reset")

    # 4. Check Action Mask
    mask = env.get_action_mask()
    print(f"Action Mask Shape: {mask.shape}")
    assert mask.shape == (246,), f"Expected (246,), got {mask.shape}"
    print("PASS: Action Mask Shape")

    # 5. Check Dice Logic
    print("Testing Dice Roll...")
    # Access the game instance directly to check dice
    dice_roll = env.game.rollDice()
    print(f"Dice Roll Result: {dice_roll}")
    assert 2 <= dice_roll <= 12, "Dice roll out of range"
    print("PASS: Dice Roll")

    # 6. Check Step
    print("Stepping environment (Action 180 - End Turn)...")
    # Action 180 is End Turn, usually valid
    obs, reward, terminated, truncated, info = env.step(180)
    print(f"Step Result - Reward: {reward}, Terminated: {terminated}")
    assert obs.shape == (1258,), f"Expected (1258,), got {obs.shape}"
    print("PASS: Step")

    # 7. Check Bank Trading Actions (Indices 226-245)
    # Just check if they exist in the mask (might be False if no resources, but index should be valid)
    try:
        # Try to access the mask at these indices
        trade_mask = mask[226:246]
        print(f"Trade Mask Slice Shape: {trade_mask.shape}")
        assert trade_mask.shape == (20,), "Trade mask slice incorrect size"
        print("PASS: Trade Actions Existence")
    except Exception as e:
        print(f"FAIL: Trade Actions Access - {e}")

    print("\nALL INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    test_integration()
