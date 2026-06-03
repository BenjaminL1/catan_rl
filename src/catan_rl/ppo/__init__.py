"""Custom PPO trainer for 1v1 Settlers of Catan.

Phase 1+ of the v2 training infrastructure
(`docs/templates/phase_review_prompt.md`). Built phase by phase:

  * Phase 1: config + arguments (this file's neighbor :mod:`arguments`)
  * Phase 2: train script entry point
  * Phase 3: rollout buffer
  * Phase 4: trainer core (loss + update loop)
  * Phase 5: vec env + GameManager
  * Phase 6: league (deferred to selfplay/)
  * Phase 7: eval harness (deferred to eval/)
  * Phase 8: checkpoint + migration
  * Phase 9: piKL anchor (optional)
  * Phase 10: first sanity training run

All defaults baked in are derived from the hardware audit measured
against this exact v2 code on M1 Pro (see commit history for details).
"""
