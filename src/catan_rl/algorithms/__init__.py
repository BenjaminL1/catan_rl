"""v2 algorithmic primitives.

Sub-packages here implement training-time regularisers and search
that operate on top of the core PPO loop. They are deliberately
stubs at Phase 9; the trainer integration happens in Phase 10+.

* :mod:`catan_rl.algorithms.pikl` — piKL anchor KL regularisation
  (Bakhtin et al., 2022). Anchors a self-play policy to a frozen
  BC policy via a KL penalty so the policy doesn't drift into
  self-play artefacts off the human distribution.
"""
