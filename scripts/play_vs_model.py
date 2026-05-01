"""
Play a game against a trained model using the GUI.

Human plays as the agent; the model plays as the opponent.
Uses pygame for board display and mouse input.

Usage:
    python scripts/play_vs_model.py checkpoints/train/checkpoint_00102400.pt
"""

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pygame
import torch

from catan_rl.algorithms.ppo.trainer import CatanPPO
from catan_rl.env.catan_env import RESOURCES, CatanEnv
from catan_rl.gui.view import catanGameView


def _refresh_display(env, view):
    """Sync display state and redraw the game screen."""
    if env.game.currentPlayer is None:
        env.game.currentPlayer = env.agent_player
    view.diceRoll = env.last_dice_roll
    view.displayGameScreen()
    pygame.display.update()


def _get_automated_action(env, view):
    """Get a valid action programmatically (for smoke test)."""
    board = env.game.board
    agent = env.agent_player
    masks = env.get_action_masks()
    action = np.zeros(6, dtype=np.int64)

    def vertex_to_index(vertex_coord):
        if vertex_coord in board.boardGraph:
            return board.boardGraph[vertex_coord].vertex_index
        return None

    def road_to_edge_idx(road):
        v1_pix, v2_pix = road
        v1_idx = board.boardGraph[v1_pix].vertex_index
        v2_idx = board.boardGraph[v2_pix].vertex_index
        edge = tuple(sorted((v1_idx, v2_idx)))
        return env._edge_to_idx.get(edge, 0)

    # Setup phase
    if env.initial_placement_phase and env._setup_pending:
        _, placement_type = env._setup_pending
        if placement_type == "settlement":
            possible = list(board.get_setup_settlements(agent).keys())
            if possible:
                v_idx = vertex_to_index(possible[0])
                if v_idx is not None:
                    action[0] = 0
                    action[1] = v_idx
                    return action
        else:
            possible = list(board.get_setup_roads(agent).keys())
            if possible:
                road = possible[0]
                action[0] = 2
                action[2] = road_to_edge_idx(road)
                return action
        return None

    if env.roll_pending:
        action[0] = 12
        return action

    if env.discard_pending:
        for i in range(5):
            if masks["resource1_discard"][i]:
                action[0] = 11
                action[4] = i
                return action
        return None

    if env.road_building_roads_left > 0:
        possible = list(board.get_potential_roads(agent).keys())
        if possible:
            action[0] = 2
            action[2] = road_to_edge_idx(possible[0])
            return action
        return None

    if env.robber_placement_pending:
        possible = list(board.get_robber_spots().keys())
        if possible:
            action[0] = 4
            action[3] = possible[0]
            return action
        return None

    # Normal phase: prefer End Turn or Roll
    if masks["type"][3]:
        action[0] = 3
        return action
    if masks["type"][12]:
        action[0] = 12
        return action
    return None


def _run_smoke_test(env, view, clock, max_steps=80):
    """Run automated smoke test: complete setup + a few turns."""
    print("Running smoke test (automated)...")
    step_count = 0
    done = False
    while not done and step_count < max_steps:
        _refresh_display(env, view)
        action = _get_automated_action(env, view)
        if action is None:
            print("Smoke test: no valid action (may be stuck)")
            break
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        clock.tick(30)
    print(f"Smoke test completed: {step_count} steps, done={done}")
    if done:
        print(f"  Result: {info.get('terminal_stats', {})}")
    pygame.quit()


def _obs_to_tensor(obs, device):
    """Convert obs dict to tensor dict for policy."""

    def _dev_seq(key):
        seq = obs.get(key, None)
        if seq is None:
            return [torch.zeros(1, dtype=torch.long, device=device)]
        arr = np.asarray(seq, dtype=np.int64)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.int64)
        return [torch.tensor(arr, dtype=torch.long, device=device)]

    return {
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


def get_human_action(env, view):
    """Get action from human via GUI. Returns (6,) action array."""
    board = env.game.board
    agent = env.agent_player
    _refresh_display(env, view)

    def vertex_to_index(vertex_coord):
        """vertex_coord is the key from boardGraph (pixel/Point)."""
        if vertex_coord in board.boardGraph:
            return board.boardGraph[vertex_coord].vertex_index
        return None

    def road_to_edge_idx(road):
        """road is (v1_pixel, v2_pixel). Return edge index."""
        v1_pix, v2_pix = road
        v1_idx = board.boardGraph[v1_pix].vertex_index
        v2_idx = board.boardGraph[v2_pix].vertex_index
        edge = tuple(sorted((v1_idx, v2_idx)))
        return env._edge_to_idx.get(edge, 0)

    masks = env.get_action_masks()
    action = np.zeros(6, dtype=np.int64)

    # Setup phase
    if env.initial_placement_phase and env._setup_pending:
        _, placement_type = env._setup_pending
        if placement_type == "settlement":
            possible = board.get_setup_settlements(agent)
            if possible:
                _refresh_display(env, view)
                vertex = view.buildSettlement_display(agent, possible.copy())
                if vertex is not None:
                    v_idx = vertex_to_index(vertex)
                    if v_idx is not None:
                        action[0] = 0
                        action[1] = v_idx
                        return action
        else:  # road
            possible = board.get_setup_roads(agent)
            if possible:
                _refresh_display(env, view)
                road = view.buildRoad_display(agent, possible.copy())
                if road is not None:
                    action[0] = 2
                    action[2] = road_to_edge_idx(road)
                    return action
        return None

    # Roll pending
    if env.roll_pending:
        clock = pygame.time.Clock()
        while True:
            _refresh_display(env, view)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if hasattr(view, "rollDice_button") and view.rollDice_button.collidepoint(
                        e.pos
                    ):
                        action[0] = 12
                        return action
            clock.tick(30)
        return None

    # Discard phase (one resource per action)
    if env.discard_pending:
        result = view.get_resource_selection(agent, "DISCARD", 1)
        if result and len(result) > 0:
            res_name = result[0]
            # View modifies player.resources; revert so env.step can do it
            agent.resources[res_name] += 1
            res_idx = RESOURCES.index(res_name) if res_name in RESOURCES else 0
            action[0] = 11
            action[4] = res_idx
            return action
        return None

    # Road building phase
    if env.road_building_roads_left > 0:
        possible = board.get_potential_roads(agent)
        if possible:
            _refresh_display(env, view)
            road = view.buildRoad_display(agent, possible.copy())
            if road is not None:
                action[0] = 2
                action[2] = road_to_edge_idx(road)
                return action
        return None

    # Robber phase
    if env.robber_placement_pending:
        possible = board.get_robber_spots()
        _refresh_display(env, view)
        hex_i, player_robbed = view.moveRobber_display(agent, dict(possible))
        if hex_i is not None:
            action[0] = 4
            action[3] = hex_i
            return action
        return None

    # Normal phase: poll GUI
    clock = pygame.time.Clock()
    while True:
        _refresh_display(env, view)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            if e.type != pygame.MOUSEBUTTONDOWN:
                continue

            # End Turn
            if view.endTurn_button.collidepoint(e.pos) and masks["type"][3]:
                action[0] = 3
                return action

            # Roll
            if view.rollDice_button.collidepoint(e.pos) and masks["type"][12]:
                action[0] = 12
                return action

            # Build Settlement
            if view.buildSettlement_button.collidepoint(e.pos) and masks["type"][0]:
                possible = board.get_potential_settlements(agent)
                if possible:
                    vertex = view.buildSettlement_display(agent, possible.copy())
                    if vertex is not None:
                        v_idx = vertex_to_index(vertex)
                        if v_idx is not None and masks["corner_settlement"][v_idx]:
                            action[0] = 0
                            action[1] = v_idx
                            return action

            # Build City
            if view.buildCity_button.collidepoint(e.pos) and masks["type"][1]:
                possible = board.get_potential_cities(agent)
                if possible:
                    vertex = view.buildCity_display(agent, possible.copy())
                    if vertex is not None:
                        v_idx = vertex_to_index(vertex)
                        if v_idx is not None and masks["corner_city"][v_idx]:
                            action[0] = 1
                            action[1] = v_idx
                            return action

            # Build Road
            if view.buildRoad_button.collidepoint(e.pos) and masks["type"][2]:
                possible = board.get_potential_roads(agent)
                if possible:
                    road = view.buildRoad_display(agent, possible.copy())
                    if road is not None:
                        action[0] = 2
                        action[2] = road_to_edge_idx(road)
                        return action

            # Buy Dev Card
            if view.devCard_button.collidepoint(e.pos) and masks["type"][5]:
                action[0] = 5
                return action

            # Bank Trade
            if view.tradeBank_button.collidepoint(e.pos) and masks["type"][10]:
                result = view.get_resource_selection(agent, "BANK", 1)
                if result and len(result) == 2:
                    give_idx = RESOURCES.index(result[0]) if result[0] in RESOURCES else 0
                    get_idx = RESOURCES.index(result[1]) if result[1] in RESOURCES else 0
                    action[0] = 10
                    action[4] = give_idx
                    action[5] = get_idx
                    return action

            # Play Dev Card
            if view.playDevCard_button.collidepoint(e.pos):
                if masks["type"][6]:  # Knight
                    action[0] = 6
                    return action
                if masks["type"][7]:  # YoP
                    result = view.get_resource_selection(agent, "YOP", 2)
                    if result and len(result) == 2:
                        # View adds resources on click; revert so env.step applies them
                        for r in result:
                            agent.resources[r] -= 1
                        action[0] = 7
                        action[4] = RESOURCES.index(result[0]) if result[0] in RESOURCES else 0
                        action[5] = RESOURCES.index(result[1]) if result[1] in RESOURCES else 0
                        return action
                if masks["type"][8]:  # Monopoly
                    result = view.get_resource_selection(agent, "MONOPOLY", 1)
                    if result:
                        res_name = result if isinstance(result, str) else result[0]
                        action[0] = 8
                        action[4] = RESOURCES.index(res_name) if res_name in RESOURCES else 0
                        return action
                if masks["type"][9]:  # Road Builder
                    action[0] = 9
                    return action

        clock.tick(30)
    return None


def main():
    parser = argparse.ArgumentParser(description="Play vs trained model (GUI)")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "--max-turns", type=int, default=500, help="Max turns per game; 0 = no limit"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run automated smoke test (no human input)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    trainer = CatanPPO.load(args.checkpoint)
    policy = trainer.policy
    policy.eval()
    device = trainer.device

    max_turns = None if args.max_turns == 0 else args.max_turns
    env = CatanEnv(render_mode=None, opponent_type="policy", max_turns=max_turns)

    obs, info = env.reset(
        options={
            "opponent_type": "policy",
            "opponent_policy": policy,
            "human_first": True,  # You (human) always go first
        }
    )

    # Clarify roles: agent_player = You (human), opponent_player = Model (AI)
    env.agent_player.name = "You"
    env.opponent_player.name = "Model"
    env.opponent_player.isAI = True  # Model stats hidden; only your turn is shown

    # Create GUI view and attach to game
    view = catanGameView(env.game.board, env.game)
    env.game.boardView = view

    pygame.display.set_caption("Catan — You vs Model")
    clock = pygame.time.Clock()
    done = False

    # Show board immediately (fixes black screen on open)
    _refresh_display(env, view)

    if args.smoke_test:
        _run_smoke_test(env, view, clock)
        return

    try:
        while not done:
            _refresh_display(env, view)

            # Human's turn
            if env.game.currentPlayer is env.agent_player:
                action = get_human_action(env, view)
                if action is None:
                    clock.tick(30)
                    continue
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                # Should not reach: opponent runs inside step on End Turn
                obs, reward, terminated, truncated, info = env.step(
                    np.array([3, 0, 0, 0, 0, 0], dtype=np.int64)
                )

            _refresh_display(env, view)
            clock.tick(30)
            done = terminated or truncated
    except Exception as e:
        print(f"Error during game: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        pygame.quit()
        raise

    # Show result (agent=You/human, opponent=Model)
    your_vp = info.get("terminal_stats", {}).get("agent_vp", "?")
    model_vp = info.get("terminal_stats", {}).get("opponent_vp", "?")
    outcome = "WIN" if info.get("is_success") else "LOSS"
    print(f"\n{outcome}! You: {your_vp} VP | Model: {model_vp} VP")
    print("Close the window to exit.")

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        view.displayGameScreen()
        pygame.display.update()
        clock.tick(30)


if __name__ == "__main__":
    main()
