"""Decoupled setup-phase training pipeline for 1v1 Catan.

Design: Monte Carlo rollout evaluation.
  1. Setup model places settlements/roads (4 decisions per game).
  2. After setup, K full-game rollouts are simulated with heuristic bots.
  3. Reward = average normalized margin of victory across K rollouts.

See `setup_env.py` (SetupEnv), `setup_policy.py` (SetupPolicy),
`setup_trainer.py` (SetupTrainer).
"""
