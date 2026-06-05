"""``StackedDice`` shim — delegates to the Rust
``catan_engine.StackedDice`` impl.

The shim preserves the existing Python class so callers like
``catan_rl.engine.game.catanGame`` and
``catan_rl.eval.rules_invariants`` (which does
``isinstance(game.dice, StackedDice)``) keep working unchanged.

The Rust impl uses ChaCha8 + Lemire bounded sampling; per the Q1
decision in ``docs/plans/rust_engine_migration.md``, we don't pursue
byte-parity with the legacy Python ``random``-module impl —
statistical equivalence is the gate.
"""

from __future__ import annotations

import random as _random

# Importing ``catan_engine`` is allowed because R1 is shipping a hard
# dependency on the Rust extension. If the extension isn't built,
# importing this module will fail with a clear error directing the
# user to run ``make rust-build``. This is the intended cutover
# behavior for the Rust path.
import catan_engine

__all__ = ["StackedDice"]


class StackedDice:
    """Python wrapper around ``catan_engine.StackedDice``.

    Preserves the legacy constructor signature ``StackedDice()`` and
    the legacy ``roll(current_player_obj, last_7_roller_obj)`` method
    so existing engine callers don't need to change.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        # Default seed: drawn from the stdlib ``random`` global state.
        # The legacy Python ``StackedDice`` consumed
        # ``random.choice`` / ``random.shuffle`` directly, so
        # ``CatanEnv.reset(seed=X)`` which calls ``random.seed(X)``
        # made the dice deterministic. Pulling 64 bits via
        # ``getrandbits`` here preserves that contract for the
        # transition: callers that seed ``random`` get a determined
        # Rust dice, and ad-hoc callers get OS-entropy.
        if seed is None:
            seed = _random.getrandbits(64)
        self._inner = catan_engine.StackedDice(seed)

    @property
    def bag(self) -> list[int]:
        """Read-only view of the current bag. Legacy callers (eval
        diagnostics, debug scripts) iterate it; the Rust impl owns
        the canonical state so we materialize on each read."""
        return list(self._inner.bag_view())

    def roll(self, current_player_obj, last_7_roller_obj):  # type: ignore[no-untyped-def]
        """Roll the dice. Args are kept as Python ``player`` objects
        for API compat with the legacy impl; we map to integer IDs
        based on object identity since the Rust Karma rule only
        cares about ``buff_player != current_player``."""
        if last_7_roller_obj is None:
            return self._inner.roll(0, None)
        # Identity comparison. The Karma rule fires iff
        # ``last_7_roller_obj is not current_player_obj``.
        if last_7_roller_obj is not current_player_obj:
            # Different identity → buffed-other-player case.
            return self._inner.roll(1, 0)
        # Same identity → buff is on the current player → no Karma.
        return self._inner.roll(0, 0)
