"""Cross-impl Python ↔ Rust obs ordering pins.

Phase 4 pivot (2026-06-06): after the user chose to fix the Rust
encoder rather than archive the migration, the Rust encoder was
rewritten to match the Python encoder ordering for the populated
``tile_representations[:, 0..19]`` and ``hex_features[:, 0..19]``
slots. The pre-pivot mismatch is documented in this file's git
history (commit ``b1904c2``) and in
``docs/plans/rust_engine_actual_state.md``.

This module pins the alignment so a future contributor who
accidentally re-introduces the mismatch trips a loud failure.
The byte-parity assertions on actual obs values live in
``test_obs_cross_impl_byte_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

catan_engine = pytest.importorskip("catan_engine")

# Canonical ordering (Phase 4 pivot). BOTH encoders must produce
# this layout. Any divergence on either side is a regression and
# fails one of the tests below.
_CANONICAL_RESOURCE_ORDER: tuple[str, ...] = (
    "BRICK",
    "ORE",
    "SHEEP",
    "WHEAT",
    "WOOD",
    "DESERT",
)
_CANONICAL_TOKEN_ORDER: tuple[int | None, ...] = (
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


class TestEncoderOrderingsAreAligned:
    """Phase 4 pivot regression guard: both encoders must use
    ``_CANONICAL_RESOURCE_ORDER`` for the resource one-hot AND
    ``_CANONICAL_TOKEN_ORDER`` for the number-token one-hot. If
    either diverges, ``checkpoint_07390040.pt`` becomes
    unloadable against the Rust path and the pivot work is
    silently undone."""

    def test_python_resource_onehot_matches_canonical(self) -> None:
        from catan_rl.policy.obs_encoder import _RESOURCE_TYPES_FOR_ONEHOT

        assert _RESOURCE_TYPES_FOR_ONEHOT == _CANONICAL_RESOURCE_ORDER, (
            "Python encoder _RESOURCE_TYPES_FOR_ONEHOT diverged from the "
            "canonical Phase 4 pivot ordering. If this is intentional, "
            "update BOTH the constant here AND the corresponding "
            "`resource_to_python_slot` helper in "
            "crates/catan_engine/src/obs.rs to keep parity."
        )

    def test_python_token_onehot_matches_canonical(self) -> None:
        from catan_rl.policy.obs_encoder import _NUMBER_TOKEN_ORDER

        assert _NUMBER_TOKEN_ORDER == _CANONICAL_TOKEN_ORDER, (
            "Python encoder _NUMBER_TOKEN_ORDER diverged from the "
            "canonical Phase 4 pivot ordering. Same dual-update path "
            "as above."
        )

    @pytest.mark.parametrize("seed", [42, 99, 1337])
    def test_rust_resource_slot_distribution_matches_canonical(self, seed: int) -> None:
        """The Rust resource distribution under the canonical
        ordering is ``[3, 3, 4, 4, 4, 1]`` (BRICK, ORE, SHEEP,
        WHEAT, WOOD, DESERT)."""
        env = catan_engine.RustCatanEnv(seed=seed)
        obs = env.reset(seed)
        tile = np.asarray(obs["tile_representations"])
        counts = tile[:, 0:6].sum(axis=0).astype(int).tolist()
        expected = [3, 3, 4, 4, 4, 1]
        assert counts == expected, (
            f"Rust resource distribution {counts} does not match "
            f"{expected} under canonical ordering "
            f"{_CANONICAL_RESOURCE_ORDER} at seed={seed}. If the Rust "
            f"`resource_to_python_slot` helper was changed, this test "
            f"must be updated AND the docstring in obs.rs."
        )

    def test_rust_token_slot_distribution_matches_canonical(self) -> None:
        """The Rust token distribution under the canonical
        ordering is ``[1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1]`` —
        slot 0 (None) for the desert, slot 1 (token 2) once,
        through slot 10 (token 12) once."""
        env = catan_engine.RustCatanEnv(seed=42)
        obs = env.reset(42)
        tile = np.asarray(obs["tile_representations"])
        counts = tile[:, 6:17].sum(axis=0).astype(int).tolist()
        expected = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        assert counts == expected, (
            f"Rust token distribution {counts} does not match expected "
            f"{expected} under canonical ordering {_CANONICAL_TOKEN_ORDER}."
        )
