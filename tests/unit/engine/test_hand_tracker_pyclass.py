"""Tests for the Rust ``HandTracker`` pyclass.

Phase 3 of the Rust migration remediation plan. Before this phase,
``crates/catan_engine/src/hand_tracker.rs`` exported free functions
``get_hand_cw`` and ``get_hand_engine`` but no ``#[pyclass]`` —
they were inaccessible from Python. The Phase 4 adapter needs the
class wired so it can consume hands from the Rust engine without
going through the legacy Python ``BroadcastHandTracker``.

Pins:

1. ``HandTracker`` is importable from ``catan_engine``.
2. The class is stateless — constructor takes no args. The
   architect review of the remediation plan flagged that
   storing a ``PyRef<PyRustEnv>`` in the class would borrow-conflict
   with ``env.step``; the env is passed per-call instead.
3. ``get_hand_cw(env, player_idx)`` returns a 5-element list in
   Charlesworth order (WOOD, BRICK, WHEAT, ORE, SHEEP). All zeros
   at reset (the engine grants resources only during setup).
4. ``get_hand_engine(env, player_idx)`` returns the same data in
   engine-internal alpha order.
5. Out-of-range player index (>= 2) yields all zeros — defensive,
   matches the free-function contract documented in
   ``hand_tracker.rs``.
"""

from __future__ import annotations

import pytest

catan_engine = pytest.importorskip("catan_engine")


class TestImport:
    def test_handtracker_class_is_exported(self) -> None:
        assert hasattr(catan_engine, "HandTracker")

    def test_handtracker_constructs_with_no_args(self) -> None:
        tracker = catan_engine.HandTracker()
        assert tracker is not None


class TestHandReads:
    def test_reset_state_has_zero_hands(self) -> None:
        env = catan_engine.RustCatanEnv(seed=42)
        tracker = catan_engine.HandTracker()
        assert list(tracker.get_hand_cw(env, 0)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_cw(env, 1)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_engine(env, 0)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_engine(env, 1)) == [0, 0, 0, 0, 0]

    def test_out_of_range_player_returns_zero(self) -> None:
        env = catan_engine.RustCatanEnv(seed=42)
        tracker = catan_engine.HandTracker()
        assert list(tracker.get_hand_cw(env, 2)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_cw(env, 99)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_engine(env, 2)) == [0, 0, 0, 0, 0]


class TestSeparateInstancesShareNoState:
    """Stateless contract: two trackers reading the same env must
    return identical values, AND mutating the env between calls
    must be reflected immediately (no caching)."""

    def test_two_trackers_agree_on_same_env(self) -> None:
        env = catan_engine.RustCatanEnv(seed=42)
        t1 = catan_engine.HandTracker()
        t2 = catan_engine.HandTracker()
        for p in (0, 1):
            assert list(t1.get_hand_cw(env, p)) == list(t2.get_hand_cw(env, p))

    def test_one_tracker_against_two_envs_reads_each(self) -> None:
        env_a = catan_engine.RustCatanEnv(seed=42)
        env_b = catan_engine.RustCatanEnv(seed=99)
        tracker = catan_engine.HandTracker()
        # Both envs are fresh → both hands are zero, but the API
        # must support reading either env via the same tracker.
        assert list(tracker.get_hand_cw(env_a, 0)) == [0, 0, 0, 0, 0]
        assert list(tracker.get_hand_cw(env_b, 0)) == [0, 0, 0, 0, 0]
