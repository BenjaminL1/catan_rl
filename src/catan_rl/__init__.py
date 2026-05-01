"""catan_rl — 1v1 Settlers of Catan reinforcement learning agent.

Public API: env, policy, trainer, evaluation. Internal modules are not exported.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Lazy-imported to avoid heavy torch/pygame load at package import time.
__all__ = [
    "CatanEnv",
    "CatanPPO",
    "CatanPolicy",
    "EvaluationManager",
    "__version__",
    "build_agent_model",
]


def __getattr__(name: str):
    if name == "CatanEnv":
        from catan_rl.env.catan_env import CatanEnv

        return CatanEnv
    if name == "CatanPolicy":
        from catan_rl.models.policy import CatanPolicy

        return CatanPolicy
    if name == "CatanPPO":
        from catan_rl.algorithms.ppo.trainer import CatanPPO

        return CatanPPO
    if name == "EvaluationManager":
        from catan_rl.eval.evaluation_manager import EvaluationManager

        return EvaluationManager
    if name == "build_agent_model":
        from catan_rl.models.build_agent_model import build_agent_model

        return build_agent_model
    raise AttributeError(f"module 'catan_rl' has no attribute {name!r}")
