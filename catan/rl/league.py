import os
import json
import numpy as np
import glob
import shutil
from sb3_contrib import MaskablePPO


class LeagueManager:
    def __init__(self, root_dir="models"):
        self.root_dir = root_dir
        self.dirs = {
            "main": os.path.join(root_dir, "main_leaguers"),
            "exploiters": os.path.join(root_dir, "exploiters"),
            "archive": os.path.join(root_dir, "archive"),
            "hof": os.path.join(root_dir, "hof")  # Keep legacy support
        }
        self.data_path = os.path.join(root_dir, "league_stats.json")
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "models": {},  # {name: {path, category, elo}}
                "matrix": {},  # {model_vs_opponent: win_rate}
                "active_exploiters": [],
                "current_agent_stats": {}  # {opponent_name: win_rate_vs_current}
            }
        self.scan_models()

    def save_data(self):
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def scan_models(self):
        # Scan all directories and register unknown models
        full_pool = []
        for cat, path in self.dirs.items():
            if not os.path.exists(path):
                continue

            for f in glob.glob(os.path.join(path, "*.zip")):
                name = os.path.basename(f)
                full_pool.append(name)
                if name not in self.data["models"]:
                    self.data["models"][name] = {
                        "path": f,
                        "category": cat,
                        "elo": 1200  # Default ELO
                    }
        # Also cleanup deleted
        to_del = [k for k in self.data["models"] if k not in full_pool]
        for k in to_del:
            del self.data["models"][k]

    def get_opponent_distribution(self):
        """
        Returns a list of (model_path, probability) tuples based on Sweet Spot and Exploiters.
        """
        candidates = list(self.data["models"].keys())
        if not candidates:
            return []

        avg_difficulty = self.get_avg_difficulty()

        weights = []
        for name in candidates:
            # Base weight

            # --- OLD LOGIC REMOVED ---
            # if name in self.data.get("active_exploiters", []):
            #    w += 10.0  # Huge weight to force confrontation
            # -------------------------

            # 2. Sweet Spot (Win Rate ~ 50%)
            # We check the recorded win rate of 'Current' vs 'Name'
            # If (current vs name) is missing, assume 0.5 (perfect learning opportunity)
            wr = self.data["current_agent_stats"].get(name, 0.5)

            # Distance from 0.5
            dist = abs(wr - 0.5)

            # Sweet spot formula: Gaussian centered at 0.5
            # W = exp(-k * |WR - 0.5|^2) with k=5
            sweet_factor = np.exp(-5.0 * (dist ** 2))

            # Base probability is sweet_factor.
            # We add a small epsilon to ensure even 0% or 100% WR ghosts get picked occasionally (e.g. Anchors/Bullies)
            w = sweet_factor + 0.1

            # --- DYNAMIC BALANCING ---
            # If the league is too hard (avg_dif < 0.4), prioritize "Teachers" (WR > 0.55)
            if avg_difficulty < 0.4:
                if wr > 0.55:
                    w += 2.0  # Boost easy opponents to restore confidence

            # If the league is too easy (avg_dif > 0.6), prioritize "Bullies" (WR < 0.45)
            elif avg_difficulty > 0.6:
                if wr < 0.45:
                    w += 2.0  # Boost hard opponents

            # 1. Exploiter Boost (Standard)
            if name in self.data.get("active_exploiters", []):
                w += 1.0  # Moderate boost

            weights.append(w)

        # Softmax-ish normalization
        total = sum(weights)
        probs = [w / total for w in weights]

        return list(zip([self.data["models"][c]["path"] for c in candidates], probs))

    def get_opponent(self):
        """Samples an opponent based on the calculated distribution."""
        dist = self.get_opponent_distribution()
        if not dist:
            return None

        paths, probs = zip(*dist)
        # normalize probs just in case floating point issues
        probs = np.array(probs)
        probs /= probs.sum()

        chosen_path = np.random.choice(paths, p=probs)
        return chosen_path

    def get_avg_difficulty(self):
        """Returns the mean win rate of the current agent against the league."""
        if not self.data["current_agent_stats"]:
            return 0.5
        return float(np.mean(list(self.data["current_agent_stats"].values())))

    def update_match_result(self, opponent_name, win):
        """
        Updates the win rate of Current Agent vs Opponent Name.
        Called after tournament evaluation.
        """
        current_wr = self.data["current_agent_stats"].get(opponent_name, 0.5)
        # Moving average update
        alpha = 0.2
        new_wr = (1 - alpha) * current_wr + alpha * (1.0 if win else 0.0)
        self.data["current_agent_stats"][opponent_name] = new_wr

        # Check for Exploiter Status (Opponent beating Current > 60% => Opponent WR > 0.6 => Current WR < 0.4)
        # RELAXED: Only mark as exploiter if we are truly getting crushed (< 30%).
        if new_wr < 0.3:
            if opponent_name not in self.data["active_exploiters"]:
                # Added 'hof' to allow promoting recent models that are crushing us
                cat = self.data["models"][opponent_name].get("category", "hof")
                if cat in ["archive", "hof"]:
                    self.data["active_exploiters"].append(opponent_name)
        elif new_wr > 0.4:
            # If we recover to 40%+, remove strict exploiter status to let Sweet Spot handle it
            if opponent_name in self.data["active_exploiters"]:
                self.data["active_exploiters"].remove(opponent_name)

    def calculate_elo(self):
        # Placeholder for full ELO implementation
        # For now, we rely on win-rates
        pass

    def promote_model(self, model_path, name, new_category):
        """Moves a model to a new category directory."""
        dest_dir = self.dirs[new_category]
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_path = os.path.join(dest_dir, name)
        shutil.copy(model_path, dest_path)

        self.data["models"][name]["category"] = new_category
        self.data["models"][name]["path"] = dest_path
        self.save_data()
