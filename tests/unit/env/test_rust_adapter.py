"""Tests for the Phase 4 ``RustCatanEnvAdapter`` scaffolding.

The adapter ships intentionally minimal: a constructor that
wraps ``catan_engine.RustCatanEnv`` and a clear
``NotImplementedError`` boundary for every ``catanGame``-style
attribute the env reads but that Phase 4 does not wire.

Pins:

1. Importable independent of the Rust extension build state;
   constructor raises ``NotImplementedError`` (NOT ``ImportError``)
   when the extension is missing.
2. With the extension built, ``RustCatanEnvAdapter()`` constructs
   without errors.
3. The Phase-4 attribute safety net: any read of a known
   ``catanGame`` attribute raises ``NotImplementedError`` with a
   pointer at the remediation plan, NOT a silent ``None`` or
   ``AttributeError``.
4. ``_IMPLEMENTED_ATTRS`` is the source of truth — anything in
   the set must have a real impl; anything not in it routes
   through ``__getattr__`` to the safety net.
5. ``CatanEnv(engine_backend="rust")`` itself raises with a clear
   pointer at the Phase-4 scope (not "scaffolding works, now you
   silently train against an unwired engine").
"""

from __future__ import annotations

import pytest


class TestImport:
    def test_module_imports_without_catan_engine_extension(self) -> None:
        """The module must be importable even when the .so is
        unavailable — the safety-net failure mode is at
        construction time, not import time."""
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        assert RustCatanEnvAdapter is not None


class TestConstructor:
    def test_constructs_when_extension_available(self) -> None:
        pytest.importorskip("catan_engine")
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        adapter = RustCatanEnvAdapter()
        assert adapter is not None

    def test_constructor_accepts_seed(self) -> None:
        pytest.importorskip("catan_engine")
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        adapter = RustCatanEnvAdapter(seed=42)
        assert adapter is not None

    def test_constructor_accepts_render_mode_for_signature_parity(self) -> None:
        """``catanGame(render_mode=None)`` is what the env passes;
        the adapter must accept the kwarg (it ignores the value).
        """
        pytest.importorskip("catan_engine")
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        adapter = RustCatanEnvAdapter(render_mode=None)
        assert adapter is not None


class TestPhase4SafetyNet:
    """Every ``catanGame`` attribute that ``CatanEnv`` reads must
    raise ``NotImplementedError`` with a pointer at the
    remediation plan when accessed on the adapter. Phase 5 / 6
    replace these with real proxies."""

    @pytest.mark.parametrize(
        "attr_name",
        [
            "board",
            "broadcast",
            "check_largest_army",
            "check_longest_road",
            "currentPlayer",
            "gameSetup",
            "log_discard",
            "log_yop",
            "maxPoints",
            "playerQueue",
            "resource_tracker",
            "rollDice",
            "update_playerResources",
        ],
    )
    def test_unimplemented_attribute_raises_with_pointer(self, attr_name: str) -> None:
        pytest.importorskip("catan_engine")
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        adapter = RustCatanEnvAdapter()
        with pytest.raises(NotImplementedError, match="Phase 5 / 6"):
            getattr(adapter, attr_name)

    def test_safety_net_points_at_actual_state_doc(self) -> None:
        pytest.importorskip("catan_engine")
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        adapter = RustCatanEnvAdapter()
        with pytest.raises(NotImplementedError, match="rust_engine_actual_state"):
            _ = adapter.board


class TestImplementedAttrsContract:
    """``_IMPLEMENTED_ATTRS`` is the source of truth. Phase 4
    intentionally ships zero proxies — the constant must be
    empty so the safety net catches everything. Phase 5 / 6
    populate it as proxies land."""

    def test_implemented_attrs_is_empty_in_phase_4(self) -> None:
        from catan_rl.env.rust_adapter import RustCatanEnvAdapter

        assert frozenset() == RustCatanEnvAdapter._IMPLEMENTED_ATTRS


class TestCatanEnvDispatch:
    """The ``engine_backend`` flag on ``CatanEnv`` must dispatch
    visibly. ``"python"`` is the production default. ``"rust"``
    raises ``NotImplementedError`` with the Phase 5 / 6 pointer
    so a typo in YAML doesn't silently train against the
    Python engine."""

    def test_python_backend_default_constructs_cleanly(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random")
        assert env.engine_backend == "python"

    def test_python_backend_explicit_constructs_cleanly(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random", engine_backend="python")
        assert env.engine_backend == "python"

    def test_rust_backend_raises_not_implemented_with_pointer(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        with pytest.raises(NotImplementedError, match="RustCatanEnvAdapter"):
            CatanEnv(opponent_type="random", engine_backend="rust")

    def test_rust_backend_error_points_at_remediation_plan(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        with pytest.raises(NotImplementedError, match="rust_engine_actual_state"):
            CatanEnv(opponent_type="random", engine_backend="rust")

    def test_unknown_backend_rejected(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        with pytest.raises(ValueError, match="engine_backend"):
            CatanEnv(opponent_type="random", engine_backend="cuda_engine_lol")
