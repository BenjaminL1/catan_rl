"""Byte-parity tests: the Python ChaCha8 reference impl in
``tests/refs/chacha8.py`` must produce the same keystream as the
Rust ``rand_chacha::ChaCha8Rng`` exposed via
``catan_engine.chacha8_keystream``.

These tests guard the migration's RNG-determinism gate (RSK-3 in the
plan). If they fail, every downstream RNG-using game decision drifts.
"""

from __future__ import annotations

import pytest

catan_engine = pytest.importorskip("catan_engine")

from tests.fixtures.python_refs.chacha8 import ChaCha8  # noqa: E402  importorskip gates above


@pytest.mark.parametrize(
    "seed",
    [
        bytes(32),  # all zeros
        b"\xff" * 32,  # all ones
        bytes(range(32)),  # 0..31
        b"\x42" * 32,
        b"\x01\x02\x03\x04" * 8,
    ],
)
def test_chacha8_first_64_bytes_match_reference(seed: bytes) -> None:
    """The first block (64 bytes) of the Rust keystream must equal
    the Python reference impl's first block. This catches sigma
    constant errors, quarter-round bugs, key-loading endianness
    mistakes, and counter-init bugs in one shot."""
    rust_bytes = bytes(catan_engine.chacha8_keystream(seed, 64))
    ref = ChaCha8.from_seed(seed)
    py_bytes = ref.fill_bytes(64)
    assert rust_bytes == py_bytes


@pytest.mark.parametrize("seed_int", [0, 1, 7, 42, 0xDEADBEEF, 0xFFFFFFFFFFFFFFFF])
def test_chacha8_long_stream_matches_reference(seed_int: int) -> None:
    """1024 bytes (16 blocks worth) — exercises the counter increment
    boundary at every block transition."""
    seed = seed_int.to_bytes(8, "little") + bytes(24)  # pad to 32
    n = 1024
    rust_bytes = bytes(catan_engine.chacha8_keystream(seed, n))
    ref = ChaCha8.from_seed(seed)
    py_bytes = ref.fill_bytes(n)
    assert rust_bytes == py_bytes


def test_chacha8_seed_length_validation() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        catan_engine.chacha8_keystream(b"\x00" * 16, 64)
    with pytest.raises(ValueError, match="32 bytes"):
        catan_engine.chacha8_keystream(b"", 64)


def test_chacha8_zero_length_request() -> None:
    assert bytes(catan_engine.chacha8_keystream(b"\x00" * 32, 0)) == b""


def test_chacha8_unaligned_lengths() -> None:
    """Non-multiple-of-4 byte requests exercise the partial-word
    tail in ``chacha8_keystream``. Lengths 1, 3, 5, 63, 65 hit
    every boundary case."""
    seed = b"\x11" * 32
    for n in (1, 3, 5, 63, 65, 127, 129):
        rust_bytes = bytes(catan_engine.chacha8_keystream(seed, n))
        ref = ChaCha8.from_seed(seed)
        py_bytes = ref.fill_bytes(n)
        assert rust_bytes == py_bytes, f"mismatch at n={n}"
