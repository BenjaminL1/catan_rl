"""Evaluation harness for the v2 PPO training pipeline.

Phase 7 of the v2 training-infra build-out. Provides:

* :class:`catan_rl.eval.harness.EvalHarness` — symmetrised
  champion-vs-opponent matches with Wilson-CI win-rate reporting.
* :mod:`catan_rl.eval.wilson` — Wilson score interval (the right CI
  for small-sample binomial proportions; raw normal approximation
  under-covers at extreme p and at low N).
* :mod:`catan_rl.eval.rules_invariants` — post-game audit of the 1v1
  Colonist.io rule pins (15 VP, no P2P trade, StackedDice, etc.).
  Any drift surfaces as a test failure, not a silent training regression.

The harness is intentionally read-only on the policy: it consumes
``policy.sample(obs, masks)`` under ``torch.no_grad()``. Phase 8's
checkpoint loader uses the harness to gate promotions.
"""
