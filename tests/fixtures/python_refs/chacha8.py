"""Python reference implementation of ChaCha8 — RFC 7539 ChaCha20
truncated to 8 quarter-round iterations (instead of 20).

Used as the byte-parity oracle for ``rand_chacha::ChaCha8Rng`` in
``tests/unit/engine/test_rng_parity.py``. Match contract:

* ``ChaCha8.from_seed(seed_bytes)`` initializes with the 32-byte
  seed as the key, counter=0, nonce=all zeros — matching
  ``rand_chacha::ChaCha8Rng::from_seed(seed)``.
* ``next_u32()`` yields one 32-bit word at a time from the keystream,
  little-endian-decoded from the 64-byte block. Counter increments
  every 16 words (i.e., every block).

This file imports nothing outside the stdlib so a future drop of
the ``rand_chacha`` dependency can re-validate against this ref
without bringing in numpy/torch. See
``docs/plans/rust_engine_migration.md`` for the parity contract.
"""

from __future__ import annotations

import struct

# ChaCha state layout constants. The first 4 words are the
# "expand 32-byte k" sigma constants per RFC 7539.
_SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)

_MASK32 = 0xFFFFFFFF


def _rotl32(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & _MASK32


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    """In-place ChaCha quarter-round on four state words."""
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl32(state[b] ^ state[c], 7)


def _chacha8_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """Run 8 rounds (4 double-rounds — column then diagonal) of the
    ChaCha core on a freshly-initialized state. Returns 64 bytes."""
    assert len(key) == 32
    assert len(nonce) == 12
    state = [
        _SIGMA[0],
        _SIGMA[1],
        _SIGMA[2],
        _SIGMA[3],
        *struct.unpack("<8I", key),
        counter & _MASK32,
        *struct.unpack("<3I", nonce),
    ]
    working = list(state)
    # 8 rounds = 4 column rounds interleaved with 4 diagonal rounds.
    # RFC 7539 specifies 20 rounds for ChaCha20; ChaCha8 takes the
    # first 8 — i.e., 4 (column + diagonal) double-rounds.
    for _ in range(4):
        # Column rounds.
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        # Diagonal rounds.
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)
    # Add the initial state back; little-endian-pack as 16 u32s = 64 bytes.
    output_words = [(working[i] + state[i]) & _MASK32 for i in range(16)]
    return struct.pack("<16I", *output_words)


class ChaCha8:
    """Lightweight ChaCha8 keystream generator with the same seeding
    convention as ``rand_chacha::ChaCha8Rng::from_seed(seed)``: the
    32-byte seed becomes the key, counter starts at 0, nonce is
    all zeros.
    """

    __slots__ = ("_block", "_counter", "_key", "_nonce", "_word_idx")

    def __init__(self, key: bytes, *, nonce: bytes = b"\x00" * 12) -> None:
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes, got {len(key)}")
        if len(nonce) != 12:
            raise ValueError(f"nonce must be 12 bytes, got {len(nonce)}")
        self._key = key
        self._nonce = nonce
        self._counter = 0
        self._block: bytes = b""
        # 16 words per block; ``_word_idx`` ranges over [0, 16). When it
        # hits 16, we generate the next block and reset to 0.
        self._word_idx = 16

    @classmethod
    def from_seed(cls, seed: bytes) -> ChaCha8:
        """Match ``rand_chacha::ChaCha8Rng::from_seed`` — 32-byte seed
        is the key; nonce + counter zero."""
        return cls(seed)

    def _refill_block(self) -> None:
        self._block = _chacha8_block(self._key, self._counter, self._nonce)
        self._counter = (self._counter + 1) & _MASK32
        self._word_idx = 0

    def next_u32(self) -> int:
        """Return the next 32-bit word from the keystream."""
        if self._word_idx >= 16:
            self._refill_block()
        offset = self._word_idx * 4
        word = struct.unpack("<I", self._block[offset : offset + 4])[0]
        self._word_idx += 1
        return word

    def next_u64(self) -> int:
        """Return the next 64-bit word, formed as
        ``low_u32 | (high_u32 << 32)`` matching ``rand_chacha``'s
        ``RngCore::next_u64`` default impl."""
        low = self.next_u32()
        high = self.next_u32()
        return low | (high << 32)

    def fill_bytes(self, n: int) -> bytes:
        """Pull ``n`` bytes of keystream — useful for parity tests."""
        out = bytearray(n)
        i = 0
        while i < n:
            if self._word_idx >= 16:
                self._refill_block()
            word = self._block[self._word_idx * 4 : self._word_idx * 4 + 4]
            self._word_idx += 1
            take = min(4, n - i)
            out[i : i + take] = word[:take]
            i += take
        return bytes(out)
