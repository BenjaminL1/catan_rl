Catan RL: Mastering 1v1 Settlers of Catan with PPO
This project implements a high-performance Reinforcement Learning (RL) agent trained to play one-vs-one Settlers of Catan. Built upon a refactored version of the classic Catan-AI engine, this environment leverages Maskable Proximal Policy Optimization (PPO) to handle the complex, multi-phase action space of the game.

📜 Credits & Citations
The core game logic and hex-grid geometry in this project are adapted and refactored from the Catan-AI repository by Karan Vombatkere.

Original Source: > Vombatkere, K. (2018). Catan-AI: A Python Implementation of Settlers of Catan with Heuristic Agents [Computer software]. GitHub. https://github.com/kvombatkere/Catan-AI

Major Modifications
Geometry Consolidation: Refactored hexTile and hexLib into a unified geometry.py module for improved performance.

Gymnasium Integration: Developed CatanEnv, a custom Gymnasium-compliant environment.

Action Masking: Implemented strict action masking to handle invalid moves in a 246-dimension discrete action space.

Stacked Dice & Karma: Added StackedDice logic to ensure a standard distribution of rolls while introducing a "Karma" system to penalize consecutive 7s.

🤖 Reinforcement Learning Architecture
Action Space (246 Discrete Actions)
The agent navigates a comprehensive action space representing all possible Catan moves:

0–53: Build Settlement

54–107: Build City

108–179: Build Road

180: End Turn

181–199: Move Robber

200: Buy Development Card

201–222: Play Development Cards (Knight, Year of Plenty, Monopoly, Road Building)

226–245: Bank Trading (Give X, Get Y)

Observation Space (1,258 Features)
The agent receives a detailed "Belief State" of the board, including:

Global Board State: Hex resource types, number tokens, and robber location.

Spatial Connectivity: Vertex ownership, building types, and edge road occupancy.

Expert Features: Expected income dots per resource, trade rates based on port ownership, and tactical lookaheads for road potential.

Public Information: Opponent hand size and "Karma" status.

📈 Training Curriculum
Training follows a three-stage curriculum to transition the agent from basic mechanics to high-level strategy:

Phase 1: The Bootcamp (0–200k steps)

Trained against RandomAIPlayer.

Focus: Learning basic building rules and winning simple games.

Phase 2: Strategy Transition (200k–500k steps)

Gradual introduction of the heuristicAIPlayer.

Difficulty ramps linearly from 0% to 100% strategic play.

Phase 3: Self-Play (500k+ steps)

Agent competes against "Ghost" versions of its previous checkpoints.

Ensures robustness against evolving human-like tactics.

🛠 Project Structure
catan/engine/: Core game logic including board.py, player.py, and geometry.py.

catan/agents/: Heuristic and Random AI implementations.

scripts/train.py: Main training loop using Stable Baselines3 MaskablePPO.

env.py: The Gymnasium RL wrapper.

debug_wrapper.py: Specialized diagnostic tool for trapping environment logic errors.

🚀 Getting Started
Prerequisites
Bash

pip install gymnasium stable-baselines3 sb3-contrib pygame numpy
Training the Agent
To start the training process:

Bash

python scripts/train.py
Visualizing Results
You can monitor the agent's progress, including success rates and episode lengths, via Tensorboard:

Bash

tensorboard --logdir ./logs/
