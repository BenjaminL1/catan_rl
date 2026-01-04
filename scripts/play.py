import sys
import os

# Add the project root to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.engine.game import catanGame

if __name__ == "__main__":
    newGame = catanGame()
    newGame.playCatan()
