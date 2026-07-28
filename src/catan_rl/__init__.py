"""catan_rl — 1v1 Settlers of Catan reinforcement learning agent.

Public API: env, policy, trainer, evaluation. Internal modules are not exported.
"""

from __future__ import annotations

__version__ = "0.2.0"


def _warn_if_foreign_tree() -> None:
    """Warn when the imported ``catan_rl`` is NOT the checkout you are standing in.

    The dev-loop runs ``pip install -e .`` inside its own git worktree; against a
    shared interpreter that rewrites the global ``catan_rl.pth`` to point at the
    worktree. Two consequences, both observed: a later ``python scripts/foo.py``
    from a different checkout silently imports the OTHER tree's code, and deleting
    that worktree breaks ``import catan_rl`` outright.

    pytest is immune (``pythonpath = ["src"]`` in ``pyproject.toml`` inserts at
    ``sys.path[0]``, beating the ``.pth``) — which is exactly why this belongs
    here and not in a test fixture: the vulnerable paths are plain scripts and
    bare subprocesses, and 31 of 41 ``scripts/*.py`` resolve purely through the
    ``.pth``.

    Deliberately a WARNING, not an error: importing one tree's package while
    standing in another is legitimate (that is how a worktree runs its own gate).
    Only stdlib, no I/O beyond two ``exists()`` calls, so the lazy-import contract
    above is preserved.
    """
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.getcwd(), "src", "catan_rl")
    if not os.path.isdir(candidate):
        return
    if os.path.realpath(candidate) == os.path.realpath(here):
        return
    print(
        f"WARNING: `catan_rl` imported from {here}, but the checkout you are in "
        f"({os.getcwd()}) has its own copy at {candidate}. A `pip install -e .` "
        "from another tree has redirected the shared environment; results may "
        "describe the WRONG code. Fix with `pip install -e . --no-deps` here.",
        file=sys.stderr,
    )


_warn_if_foreign_tree()

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
