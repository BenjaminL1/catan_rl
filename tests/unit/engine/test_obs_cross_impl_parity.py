"""Cross-impl Python ↔ Rust obs byte-parity tests.

Phase 4 of the Rust migration remediation plan. The Phase 3
review deferred this work explicitly: the Phase 3 tests covered
*structural* parity (resource one-hot sums to 1, exactly 1 desert,
robber bits boolean, etc.) but did NOT cross-compare the Rust and
Python encoders for byte-equivalence on a known-equivalent board
layout. Phase 4 closes that gap — or proves it cannot be closed
without a wider encoder change.

**Phase-4 finding** (`2026-06-06`): the Python and Rust encoders
use **different one-hot orderings** for both the resource and
number-token slots. The mismatch is structural, not a per-byte
drift:

| Slot range            | Python order        | Rust order          |
|-----------------------|---------------------|---------------------|
| resource one-hot      | BRICK, ORE, SHEEP,  | DESERT, WOOD, BRICK,|
|                       | WHEAT, WOOD, DESERT | WHEAT, ORE, SHEEP   |
| number-token slots    | None, 2..6, 8..12   | 2..12 (inc. dead 7) |

The Python order is documented at
``src/catan_rl/policy/obs_encoder.py:76-100``; the Rust order is
documented at ``crates/catan_engine/src/obs.rs:60-65`` (resource
index = Rust enum value; token slot = ``6 + (token - 2)``).

Consequence: ``checkpoint_07390040.pt`` (which was trained against
the Python encoder) sees a **permuted** input distribution from
the Rust path. The "fresh-train-required" badge from Phase 3 is
*structural*, not optional.

This module ships:

1. **Honest tests** that *prove* the mismatch exists — these are
   the regression guard against a future contributor "fixing" the
   parity by aligning one encoder to the other without
   intentional scope (which would silently break ``ckpt_07390040.pt``).
2. A SKIPPED-with-clear-reason ``test_cross_impl_byte_parity_*``
   test that documents what would have to hold for cross-impl
   parity, and *why it does not hold today*.

Cross-impl parity is **not a Phase 4 deliverable as originally
specified**; the structural mismatch is the blocker. A future
phase that elects either (a) to rewrite the Rust encoder to
match the Python ordering, or (b) to retrain a fresh policy
against the Rust ordering and supersede the legacy checkpoint,
must remove or replace this module's skipped tests.
"""

from __future__ import annotations

import numpy as np
import pytest

catan_engine = pytest.importorskip("catan_engine")

# Match the encoder constants exactly. Any change in either
# encoder must update one of these tuples AND surface in the
# remediation plan.
_PY_RESOURCE_ONEHOT_ORDER: tuple[str, ...] = (
    "BRICK",
    "ORE",
    "SHEEP",
    "WHEAT",
    "WOOD",
    "DESERT",
)
_RUST_RESOURCE_ONEHOT_ORDER: tuple[str, ...] = (
    "DESERT",
    "WOOD",
    "BRICK",
    "WHEAT",
    "ORE",
    "SHEEP",
)
_PY_TOKEN_ONEHOT_ORDER: tuple[int | None, ...] = (
    None,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
)
_RUST_TOKEN_ONEHOT_ORDER: tuple[int | None, ...] = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
)


class TestEncoderOrderingsDiffer:
    """Pin the encoder ordering mismatch. These tests would
    *fail* if a future contributor accidentally aligns the two
    encoders without updating ``checkpoint_07390040.pt``'s
    compatibility doc — exactly the silent regression the
    Phase 3 review feared."""

    def test_resource_orderings_differ_between_encoders(self) -> None:
        assert _PY_RESOURCE_ONEHOT_ORDER != _RUST_RESOURCE_ONEHOT_ORDER, (
            "Python and Rust resource one-hot orderings have been aligned. "
            "If this was intentional, update ``ckpt_07390040.pt``'s "
            "compatibility doc + retrain or migrate the checkpoint, then "
            "REPLACE this test with a real cross-impl byte-parity gate. "
            "See docs/plans/rust_engine_actual_state.md."
        )

    def test_token_orderings_differ_between_encoders(self) -> None:
        assert _PY_TOKEN_ONEHOT_ORDER != _RUST_TOKEN_ONEHOT_ORDER, (
            "Python and Rust number-token one-hot orderings have been "
            "aligned. Same compat path as above must be followed."
        )

    def test_both_orderings_cover_the_same_resource_set(self) -> None:
        """Sanity: while the orderings differ, the *underlying
        resource set* must be identical. If a future Rust change
        adds or drops a resource, the encoders disagree about
        *what the game contains*, not just the order — that's a
        much deeper bug than ordering."""
        assert set(_PY_RESOURCE_ONEHOT_ORDER) == set(_RUST_RESOURCE_ONEHOT_ORDER)


class TestRustEncoderMatchesItsDocumentedOrdering:
    """Verify the live Rust encoder uses the ordering pinned in
    ``_RUST_RESOURCE_ONEHOT_ORDER``. If the Rust enum is renumbered,
    this test fails loudly so the constant + the mismatch doc both
    get updated together."""

    @pytest.mark.parametrize("seed", [42, 99, 1337])
    def test_rust_resource_slot_distribution_matches_documented_order(self, seed: int) -> None:
        """Standard 1v1 board: 1 desert + 4 wood + 3 brick + 4 wheat
        + 3 ore + 4 sheep. Permuted into the Rust ordering this is
        ``[1, 4, 3, 4, 3, 4]`` (DESERT, WOOD, BRICK, WHEAT, ORE,
        SHEEP). Phase 3's structural parity test already asserts
        this; Phase 4 re-asserts here to double-pin the ordering
        if the Rust enum is ever renumbered."""
        env = catan_engine.RustCatanEnv(seed=seed)
        obs = env.reset(seed)
        tile = np.asarray(obs["tile_representations"])
        counts = tile[:, 0:6].sum(axis=0).astype(int).tolist()
        expected = [1, 4, 3, 4, 3, 4]
        assert counts == expected, (
            f"Rust resource distribution {counts} does not match "
            f"expected {expected} under the documented ordering "
            f"{_RUST_RESOURCE_ONEHOT_ORDER} at seed={seed}. If the "
            f"Rust enum was renumbered, update _RUST_RESOURCE_ONEHOT_ORDER "
            f"AND the mismatch doc in this module."
        )

    def test_rust_token_slot_distribution_matches_documented_order(self) -> None:
        """Standard 1v1 board: each non-2/12 token appears 2 times,
        tokens 2 and 12 appear once each. Token 7 is reserved for
        dice rolls and never appears on a hex; its slot stays at 0.

        Under the Rust ordering [2,3,4,5,6,7,8,9,10,11,12] the
        expected per-slot counts are [1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1].
        """
        env = catan_engine.RustCatanEnv(seed=42)
        obs = env.reset(42)
        tile = np.asarray(obs["tile_representations"])
        counts = tile[:, 6:17].sum(axis=0).astype(int).tolist()
        expected = [1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1]
        assert counts == expected, (
            f"Rust token distribution {counts} does not match expected "
            f"{expected} under ordering {_RUST_TOKEN_ONEHOT_ORDER}. "
            f"Token 7 slot must stay 0 (never appears on a hex)."
        )


class TestPythonEncoderMatchesItsDocumentedOrdering:
    """Symmetric pin for the Python encoder. If
    ``_RESOURCE_TYPES_FOR_ONEHOT`` in obs_encoder.py changes order,
    this test fails and forces an intentional update of the doc +
    the legacy checkpoint compat note."""

    def test_python_resource_onehot_constant_matches(self) -> None:
        from catan_rl.policy.obs_encoder import _RESOURCE_TYPES_FOR_ONEHOT

        assert _RESOURCE_TYPES_FOR_ONEHOT == _PY_RESOURCE_ONEHOT_ORDER, (
            "Python encoder _RESOURCE_TYPES_FOR_ONEHOT diverged from "
            "the pinned _PY_RESOURCE_ONEHOT_ORDER. Update both AND "
            "the mismatch doc in this module."
        )

    def test_python_token_onehot_constant_matches(self) -> None:
        from catan_rl.policy.obs_encoder import _NUMBER_TOKEN_ORDER

        assert _NUMBER_TOKEN_ORDER == _PY_TOKEN_ONEHOT_ORDER, (
            "Python encoder _NUMBER_TOKEN_ORDER diverged from the "
            "pinned _PY_TOKEN_ONEHOT_ORDER. Same compat update path."
        )


@pytest.mark.skip(
    reason=(
        "Cross-impl Python ↔ Rust byte-parity on the populated obs "
        "slots is BLOCKED by encoder ordering mismatch. See module "
        "docstring. To unblock: either rewrite one encoder to match "
        "the other (and retrain or migrate ckpt_07390040.pt), or "
        "supersede the legacy checkpoint with a fresh-trained one "
        "against whichever ordering the Rust path uses. Phase 4 "
        "ships the structural mismatch tests above as the regression "
        "guard until that decision lands."
    )
)
def test_cross_impl_byte_parity_on_populated_slots() -> None:
    """Placeholder for the gate the Phase 3 review carved into
    Phase 4. Marked ``skip`` with a clear reason rather than
    silently absent so future contributors find it via
    ``pytest -rs`` and understand the blocker."""
    raise AssertionError("see skip reason")
