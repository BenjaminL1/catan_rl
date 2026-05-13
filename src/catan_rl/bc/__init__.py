"""Step 3 behaviour-cloning package — heuristic-vs-heuristic data
generation, masked-action dataset, per-head CE loss, and the
paired-bootstrap / TOST gates per ``v2_step3_bc.md``.
"""

from catan_rl.bc.perturbed_heuristic import (
    EpsilonGreedyHeuristicAIPlayer,
    WeightNoisedHeuristicAIPlayer,
)

__all__ = [
    "EpsilonGreedyHeuristicAIPlayer",
    "WeightNoisedHeuristicAIPlayer",
]
