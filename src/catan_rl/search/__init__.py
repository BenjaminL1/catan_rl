"""Inference-time determinized search (offline) for the v2 1v1 Catan policy.

A new, self-contained module (Spec Kit feature ``003-inference-search``). It only
*reads* the engine/policy/env and *clones* the game to simulate — nothing in
``engine/``, ``policy/``, ``ppo/``, ``env/``, or ``checkpoint/`` is modified, so
training/eval behaviour is byte-identical when search is unused (FR-009).

Architecture note (why this is a max-only tree): ``CatanEnv.step`` folds the
opponent's ENTIRE turn (and the dice) into the agent's ``EndTurn`` transition, so
every search node is an *agent* decision point and the opponent + dice are the
(determinized) environment. Hence the value head is always queried from the
agent's POV and backup needs no per-ply sign flip. See ``mcts.py``.
"""

from __future__ import annotations

from catan_rl.search.agent import SearchAgent
from catan_rl.search.bakeoff import run_bakeoff
from catan_rl.search.config import SearchConfig
from catan_rl.search.eval_search import evaluate_search_vs_policy, run_search_matchup
from catan_rl.search.mcts import MCTS
from catan_rl.search.priors import action_priors
from catan_rl.search.value import leaf_value, squash_value, value_from_obs

__all__ = [
    "MCTS",
    "SearchAgent",
    "SearchConfig",
    "action_priors",
    "evaluate_search_vs_policy",
    "leaf_value",
    "run_bakeoff",
    "run_search_matchup",
    "squash_value",
    "value_from_obs",
]
