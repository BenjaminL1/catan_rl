"""``RustCatanEnvAdapter`` — Python-side shim around
``catan_engine.RustCatanEnv``.

Phase 4 of the Rust migration remediation plan + the Phase 4
PIVOT user decision (2026-06-06). The migration is **FROZEN** at
Phase 4-pivot: the Rust path is available for inference /
deterministic eval / future MCTS rollouts, but is NOT wired into
the production training loop. Phases 5-10 of the remediation plan
are explicitly not pursued. See
``docs/plans/rust_engine_actual_state.md``.

The shim's original purpose was to expose the minimum
``catanGame``-style attribute surface that
:class:`catan_rl.env.catan_env.CatanEnv` reads, so the env could
dispatch on ``engine_backend="rust"`` without rewriting its own
body. After the freeze, the shim's purpose is to provide a stable,
loud-on-access boundary so anyone who tries to enable
``engine_backend="rust"`` for training is stopped with a clear
pointer to the freeze decision.

**What ships.** This module's constructor wraps
``catan_engine.RustCatanEnv``; ``__getattr__`` is a safety net
that raises ``NotImplementedError`` with a pointer to the freeze
decision for every ``catanGame``-style attribute the env reads.
The Phase 4 audit found 13 distinct attribute reads on
``CatanEnv.game.*``:

* ``board`` — the ``catanBoard`` object (with ``boardGraph``,
  ``hexTileDict``, etc.).
* ``broadcast`` — the ``GameBroadcast`` event bus.
* ``check_largest_army(player)`` — recompute LA.
* ``check_longest_road(player)`` — recompute LR.
* ``currentPlayer`` — settable.
* ``gameSetup`` — bool flag, settable.
* ``log_discard(player, list)`` — broadcasts a discard.
* ``log_yop(player, list)`` — broadcasts YoP.
* ``maxPoints`` — constant 15.
* ``playerQueue`` — ``queue.Queue([2 players])``.
* ``resource_tracker`` — ``ResourceTracker`` instance.
* ``rollDice()`` — rolls + applies + returns int.
* ``update_playerResources(dice, player)`` — settle distribution.

**Phases 5/6 were FROZEN on 2026-06-06.** The Phase 1 bench +
architect review concluded the end-to-end ceiling from rolling
out is `~1.09×` (per Amdahl on the SGD-dominated update), below
the revised Phase 6 gate of `≥ 1.15×`. The user accepted the
architect's HALT recommendation for the rollout-loop wiring AND
chose PIVOT for the obs encoder (Rust now byte-parities with
Python on the populated `[0..19]` slots so
``checkpoint_07390040.pt`` can theoretically load against the
Rust path for inference / eval / MCTS). The catanGame attribute
proxies in this module are **not coming**; any code that needs
them must either (a) accept they will not land, (b) wire what it
needs ad-hoc against ``self._engine: catan_engine.RustCatanEnv``,
or (c) re-open the freeze decision with the user.
"""

from __future__ import annotations

from typing import Any


class RustCatanEnvAdapter:
    """``catanGame``-shaped shim around ``catan_engine.RustCatanEnv``.

    Phase 4 of the Rust migration remediation plan ships the
    constructor and a clear ``NotImplementedError`` boundary for
    every unimplemented attribute. Phase 5 / 6 wire the proxies.

    Constructor signature mirrors ``catanGame(render_mode=None)``
    so :meth:`catan_rl.env.catan_env.CatanEnv.reset` can swap
    impls with a single dispatch line.
    """

    #: Implemented attribute names. Listed explicitly (not derived)
    #: so a future contributor adding a real proxy must intentionally
    #: update this set. Anything not in the set raises
    #: :class:`NotImplementedError` from :meth:`__getattr__`. The
    #: set is empty post-freeze (2026-06-06); adding entries
    #: re-opens the migration scope and must be justified against
    #: the freeze decision.
    _IMPLEMENTED_ATTRS: frozenset[str] = frozenset()

    def __init__(self, *, render_mode: Any = None, seed: int | None = None) -> None:
        """Build the adapter. ``render_mode`` is accepted for
        signature parity with ``catanGame`` but ignored (the Rust
        env has no pygame surface).

        ``seed`` is forwarded to the underlying
        ``catan_engine.RustCatanEnv``. When ``None``, the
        adapter constructs with a deterministic placeholder seed
        (0) — callers that need reproducibility must pass an
        explicit seed.
        """
        # Defer the import so this module is importable even when
        # the Rust extension is unavailable (CI smoke tests don't
        # all build the ``.so``). The constructor still raises
        # immediately if the extension is missing.
        try:
            import catan_engine
        except ImportError as e:
            raise NotImplementedError(
                "RustCatanEnvAdapter requires the catan_engine extension; "
                "run `make rust-build` or `pip install -e .` to build it. "
                "See docs/plans/rust_engine_actual_state.md."
            ) from e
        self._engine = catan_engine.RustCatanEnv(seed=seed if seed is not None else 0)
        self._render_mode = render_mode

    def __getattr__(self, name: str) -> Any:
        """Safety net (post-Phase-4-pivot freeze).

        Any ``catanGame``-style attribute read raises with a
        pointer at the freeze decision. Phases 5/6 were FROZEN on
        2026-06-06; the catanGame proxies are not coming. Code
        that needs the Rust path for inference / eval / MCTS must
        access ``self._engine: catan_engine.RustCatanEnv`` directly
        rather than going through these proxies.
        """
        if name in self._IMPLEMENTED_ATTRS:
            raise AttributeError(
                f"RustCatanEnvAdapter._IMPLEMENTED_ATTRS lists {name!r} "
                f"but no real proxy is defined. This is a programming error."
            )
        raise NotImplementedError(
            f"RustCatanEnvAdapter.{name} is not wired. The Rust migration "
            f"FROZE at Phase 4-pivot on 2026-06-06 (HALT verdict on the "
            f"rollout-loop wiring; PIVOT decision to fix the obs encoder "
            f"so checkpoints can theoretically load against the Rust path). "
            f"The catanGame proxies are not coming. For inference / eval / "
            f"MCTS use cases, access self._engine: catan_engine.RustCatanEnv "
            f"directly. For training, use engine_backend='python'. "
            f"See docs/plans/rust_engine_actual_state.md."
        )
