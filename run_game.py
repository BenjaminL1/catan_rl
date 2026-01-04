from catan.game_engine.catanGame import catanGame
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    newGame = catanGame()
    newGame.playCatan()
