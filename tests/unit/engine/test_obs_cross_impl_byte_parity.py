"""Cross-impl Python ↔ Rust obs **byte-parity** tests.

Phase 4 pivot (2026-06-06). The Phase 3 review carved a cross-impl
byte-parity gate into Phase 4; Phase 4's audit found it was
blocked by encoder ordering mismatch; the user chose PIVOT, the
Rust encoder was rewritten to match the Python ordering, and
this module finally lands the gate.

Scope (matches the pivot's scope, no wider):

* ``tile_representations[:, 0..19]`` — populated slots
  (resource one-hot, token one-hot, current-robber bit, dot
  count). The pivot rewrote the Rust encoder so these match the
  Python encoder byte-for-byte for the same board layout.
* ``hex_features[:, 0..19]`` — same population, same parity.

Out of scope (still fresh-train-required):

* ``tile_representations[:, 19..79]`` — vertex/edge ownership
  blocks. Zero-filled in Rust per the Phase 3 placeholder badge.
* ``vertex_features`` and ``edge_features`` — Rust ships
  placeholder content (Phase 3 audit).
* ``current_player_main`` / ``next_player_main`` / dev counts —
  encoder semantics differ; not addressed by the pivot.

The strategy is: build a fresh ``RustCatanEnv``, read its board
layout via ``BoardStatic.board_static()``, then construct a
Python ``catanGame`` and mutate its ``hexTileDict`` so the
resource / number-token / robber state matches the Rust state
exactly. Then build both obs and compare the populated slots.

NOTE on board geometry: only the per-hex *content* (resource,
token, robber) flows into the [0..19] block of the Python
encoder. The Python encoder DOES use board geometry to fill the
[19..79] vertex/edge blocks, but those are outside scope (still
fresh-train-required in the Rust output). The test does NOT need
to translate the Rust board's spiral geometry into the Python
board's spiral — they can disagree on geometry as long as the
per-hex content matches.
"""

from __future__ import annotations

import numpy as np
import pytest

catan_engine = pytest.importorskip("catan_engine")


def _build_python_obs_matching_rust_board(seed: int) -> dict:
    """Build a Python obs whose per-hex content matches the Rust
    board at ``seed``. Returns the Python ``tile_representations``
    + ``hex_features`` arrays.
    """
    from catan_rl.engine.game import catanGame
    from catan_rl.engine.player import player as PlainPlayer
    from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder

    # Pull the Rust board layout.
    bs = catan_engine.BoardStatic(seed=seed).board_static()
    rust_hexes = bs["hexes"]

    # Construct a Python game (its own board layout is irrelevant —
    # we'll overwrite the per-hex content).
    game = catanGame(render_mode=None)
    board = game.board

    # Translate Rust resource enum strings → Python obs-encoder
    # resource_type strings. Both use uppercase names matching
    # ``_RESOURCE_TYPES_FOR_ONEHOT``, so the mapping is identity.
    for rust_hex in rust_hexes:
        h_idx = rust_hex["hex_idx"]
        py_tile = board.hexTileDict[h_idx]
        py_tile.resource_type = rust_hex["resource"]
        py_tile.number_token = rust_hex["number_token"]
        # Rust's ``has_robber_initial`` is True iff the hex is
        # desert; Python's ``has_robber`` is the CURRENT robber.
        # At reset both are equivalent (robber starts on desert).
        py_tile.has_robber = bool(rust_hex["has_robber_initial"])

    # Build a single agent player; the obs encoder needs one but
    # only consumes it for the vertex/edge sections (out of scope).
    agent = PlainPlayer("Agent", "black")
    agent.game = game

    encoder = ObsEncoder(board)
    env_state = EnvObsState(initial_placement_phase=True)
    py_obs = encoder.build_obs(game, agent, agent, env_state)
    return py_obs


def _build_rust_obs(seed: int) -> dict:
    env = catan_engine.RustCatanEnv(seed=seed)
    return env.reset(seed)


class TestTileRepresentationsByteParity:
    """``tile_representations[:, 0..19]`` byte-for-byte between
    encoders for the same board layout."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 1337, 12345])
    def test_tile_representations_populated_slots_match_byte_for_byte(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust_tile = np.asarray(rust_obs["tile_representations"])
        py_tile = np.asarray(py_obs["tile_representations"])
        # Compare the populated [0..19] block only — the
        # [19..79] block is still zero-fill in Rust by design.
        diff = py_tile[:, :19] - rust_tile[:, :19]
        max_abs = float(np.max(np.abs(diff)))
        assert max_abs == 0.0, (
            f"tile_representations[:, 0..19] differs between Python and "
            f"Rust at seed={seed}; max abs diff = {max_abs}. The Phase 4 "
            f"pivot promised byte-parity on this range; a non-zero diff "
            f"is a regression. Diff sample (first 3 rows):\n{diff[:3, :19]}"
        )


class TestHexFeaturesByteParity:
    """``hex_features[:, 0..19]`` byte-for-byte. Same recipe as
    tile_representations' [0..19] block per the Python encoder
    at ``obs_encoder.py:382-387``."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 1337, 12345])
    def test_hex_features_populated_slots_match_byte_for_byte(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust_hex = np.asarray(rust_obs["hex_features"])
        py_hex = np.asarray(py_obs["hex_features"])
        assert rust_hex.shape == py_hex.shape == (19, 19)
        diff = py_hex - rust_hex
        max_abs = float(np.max(np.abs(diff)))
        assert max_abs == 0.0, (
            f"hex_features differs between Python and Rust at "
            f"seed={seed}; max abs diff = {max_abs}. Diff sample:\n"
            f"{diff[:3]}"
        )


class TestPerSlotParity:
    """Granular per-slot parity assertions so a slot-specific
    regression points at the right place quickly."""

    @pytest.mark.parametrize("seed", [42, 99])
    def test_resource_onehot_byte_parity(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust = np.asarray(rust_obs["tile_representations"])[:, 0:6]
        py = np.asarray(py_obs["tile_representations"])[:, 0:6]
        assert np.array_equal(rust, py), (
            f"resource one-hot mismatch at seed={seed}\nrust:\n{rust}\npy:\n{py}"
        )

    @pytest.mark.parametrize("seed", [42, 99])
    def test_token_onehot_byte_parity(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust = np.asarray(rust_obs["tile_representations"])[:, 6:17]
        py = np.asarray(py_obs["tile_representations"])[:, 6:17]
        assert np.array_equal(rust, py), (
            f"token one-hot mismatch at seed={seed}\nrust:\n{rust}\npy:\n{py}"
        )

    @pytest.mark.parametrize("seed", [42, 99])
    def test_current_robber_bit_byte_parity(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust = np.asarray(rust_obs["tile_representations"])[:, 17]
        py = np.asarray(py_obs["tile_representations"])[:, 17]
        assert np.array_equal(rust, py), (
            f"current-robber bit mismatch at seed={seed}\nrust:{rust.tolist()}\npy:{py.tolist()}"
        )

    @pytest.mark.parametrize("seed", [42, 99])
    def test_dot_count_byte_parity(self, seed: int) -> None:
        rust_obs = _build_rust_obs(seed)
        py_obs = _build_python_obs_matching_rust_board(seed)
        rust = np.asarray(rust_obs["tile_representations"])[:, 18]
        py = np.asarray(py_obs["tile_representations"])[:, 18]
        assert np.allclose(rust, py, atol=0.0), (
            f"dot count mismatch at seed={seed}\nrust:{rust.tolist()}\npy:{py.tolist()}"
        )
