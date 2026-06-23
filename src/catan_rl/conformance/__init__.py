"""Cross-engine conformance harness (Torevan Phase 7).

This package records full reference games played by the pure-Python
``catanGame`` engine (``catan_rl.engine.game``) under a random-legal
policy and emits a deterministic *replay-log* JSON that the
all-TypeScript Torevan engine (``@torevan/engine``) replays
step-for-step in ``packages/engine/src/conformance/conformance.test.ts``.

The replay-log is the only contract between the two engines. Its schema
is documented in :mod:`catan_rl.conformance.recorder` (``record_game``)
and mirrored by the TS-side loader/normaliser in the conformance test.

Unlike the RL replay system (:mod:`catan_rl.replay`) — which records a
game for human *viewing* and is driven by the env/broadcast layer — this
recorder records a game for cross-engine *differential testing*: every
RNG-derived outcome (dice roll, steal victim + stolen resource, drawn
dev-card type) is captured alongside the action so the TS engine can
reproduce the exact line of play through its production ``applyAction``
path via the typed ``ReplayOutcome`` seam.
"""

from catan_rl.conformance.recorder import record_game, save_log

__all__ = ["record_game", "save_log"]
