"""``RustCatanEnvAdapter`` — Python-side shim around
``catan_engine.RustCatanEnv``.

Phase 4 of the Rust migration remediation plan. The shim's job is
to expose the minimum ``catanGame``-style attribute surface that
:class:`catan_rl.env.catan_env.CatanEnv` reads, so the env can
dispatch on ``engine_backend="rust"`` without rewriting its own
body.

**Scope of Phase 4 (the scaffolding pass).** This module ships
the constructor, the documented dispatch contract, and a
``NotImplementedError`` safety net for every ``catanGame`` attribute
the env reads. The Phase 4 audit found 13 distinct attribute reads
on ``CatanEnv.game.*``:

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

**Phase 5 / 6 (the wiring pass)** implements each proxy on top of
the corresponding Rust state-machine call, OR moves the relevant
state-machine ownership into Python with the adapter forwarding
only the legal-action application to ``catan_engine.RustCatanEnv``.
The right split depends on the opponent injection contract
landing in Phase 5; until then this module raises clearly when
any uninstrumented attribute is accessed.

**Why scaffold today instead of wiring fully?** The Phase 1 bench
+ architect review concluded that the end-to-end ceiling from
rolling out is `~1.09×` (per Amdahl on the SGD-dominated update).
The Phase 6 gate is `≥ 1.15× e2e`. The expected payoff of wiring
the full adapter is bounded by that gate. The scaffolding pass
makes the dispatch flag real and pins the *interface contract*
(what the adapter must satisfy) so that if Phase 5 measures
opponent-injection cost as small, Phase 6's full implementation
is a mechanical fill-in. If the measurement shows the cost is
prohibitive, the scaffolding is the maximally-honest record of
why Phase 4 stopped here.
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

    #: Implemented-by-Phase-4 attribute names. Listed explicitly
    #: (not derived) so a future contributor adding a real proxy
    #: must intentionally update this set. Anything not in the
    #: set raises :class:`NotImplementedError` from
    #: :meth:`__getattr__`.
    _IMPLEMENTED_ATTRS: frozenset[str] = frozenset(
        {
            # Phase 4 deliberately implements zero proxies — the
            # adapter exists as a typed, importable scaffold so
            # the engine_backend dispatch line in CatanEnv has a
            # real target. Phase 5 will populate this set.
        }
    )

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
        """Phase 4 safety net.

        Any ``catanGame``-style attribute read that hasn't been
        wired through to a real Rust proxy raises with a clear
        pointer at the remediation plan. The Phase 5 / 6 work is
        what closes this gap; until then the failure mode is
        loud-and-early (at attribute access) rather than
        silent-and-late (mysterious downstream behaviour).
        """
        if name in self._IMPLEMENTED_ATTRS:
            # Should never reach here — implemented attrs are
            # defined as normal methods/properties on the class.
            # Hitting this means an _IMPLEMENTED_ATTRS entry
            # lacks a real impl, which is a programming error.
            raise AttributeError(
                f"RustCatanEnvAdapter._IMPLEMENTED_ATTRS lists {name!r} "
                f"but no real proxy is defined. This is a programming error."
            )
        raise NotImplementedError(
            f"RustCatanEnvAdapter.{name} is not wired yet. "
            f"Phase 5 / 6 of the Rust migration remediation plan will "
            f"implement the catanGame attribute proxies; until then the "
            f"production training loop must use engine_backend='python'. "
            f"See docs/plans/rust_engine_actual_state.md."
        )
