"""Expert iteration (search-as-teacher distillation) for the v2 1v1 Catan policy.

Spec Kit feature ``004-expert-iteration``. Banks the offline 003 inference-search
lever into the base policy: play games where the agent is the 003 determinized
``SearchAgent``, record each agent decision as a training target, and DISTILL those
targets into a warm-started fine-tune of v6 — reusing the behavior-cloning pipeline
(``catan_rl.bc``) wholesale. A new, isolated module: it only *reads/reuses* the
search, BC pipeline, env, and eval; nothing in engine/policy/ppo/env/checkpoint
changes (the one existing-code edit is an additive ``train_bc(init_ckpt=...)``
warm-start param). See specs/004-expert-iteration/.

The pilot gate (one round, distilled-vs-raw-v6 SEARCH-FREE, Wilson LB>0.50 at
n>=200->500) is the go/no-go, mirroring the 003 bake-off.
"""

from __future__ import annotations

from catan_rl.expert_iteration.config import DistillConfig, SearchLabelConfig

__all__ = ["DistillConfig", "SearchLabelConfig"]
