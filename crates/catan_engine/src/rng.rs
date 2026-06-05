//! ChaCha8 RNG wrapper used by every random surface in the engine
//! (dice bag shuffle, dice noise step, port assignment, hex resource
//! shuffle, dev card draws, heuristic AI action choices). One stream
//! per `GameState`, seeded per env from the trainer's master seed via
//! `master ^ env_idx` (see `docs/plans/rust_engine_migration.md` Q3).
//!
//! The byte-parity contract against `tests/refs/chacha8.py` runs in
//! `tests/unit/engine/test_rng_parity.py`. To make that test work
//! we expose a tiny "draw N bytes from the keystream" surface to
//! Python — exposed via the PyO3 module entry, NOT through the engine
//! API. Game code uses `EngineRng` directly.

// PyO3 0.22's `#[pyfunction]` macro emits an `.into()` on the
// `PyResult<T>` return — clippy flags it as a useless conversion
// against the same `PyErr` type. The warning is on the macro
// expansion, not source code, so we suppress at module level.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// One ChaCha8 stream per env. Wraps `ChaCha8Rng` with the seeding
/// convention pinned by the migration plan.
///
/// `Clone` is derived so `StackedDice` and the future `GameState`
/// can support Python's `copy.deepcopy` via `__deepcopy__`.
/// `rand_chacha::ChaCha8Rng` derives `Clone` and snapshots the
/// internal counter + cached block — clone fidelity matches the
/// snapshot-via-memcpy guarantee that R3 will provide.
#[derive(Clone)]
pub struct EngineRng {
    inner: ChaCha8Rng,
}

impl EngineRng {
    /// Seed via a raw 32-byte seed — matches
    /// `ChaCha8Rng::from_seed(seed)` 1:1. The Python ref impl in
    /// `tests/refs/chacha8.py` uses this same constructor.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }

    /// Seed via a u64 — equivalent to `master ^ env_idx`. Internally
    /// calls `ChaCha8Rng::seed_from_u64` which runs PCG32 to fill the
    /// 32-byte seed buffer. Use this for env-level seeding only;
    /// the parity test uses `from_seed` to avoid pinning the seeder.
    pub fn from_u64(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Pull the next 32-bit word from the keystream.
    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Pull the next 64-bit word from the keystream.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Pull a uniform sample from `[0, n)`. Uses the lemire-style
    /// unbiased rejection sample on a u32 multiplied by n. For our
    /// engine paths `n <= 72` (edge count) so the rejection is rare.
    /// Caller must ensure `n > 0`.
    pub fn gen_range_u32(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        // Lemire's nearly-divisionless unbiased bounded random.
        // Reference: https://lemire.me/blog/2019/06/06/nearly-divisionless-random-integer-generation-on-various-systems/
        let mut x = self.next_u32() as u64 * n as u64;
        let mut lo = x as u32;
        if lo < n {
            let t = (u32::MAX - n + 1) % n;
            while lo < t {
                x = self.next_u32() as u64 * n as u64;
                lo = x as u32;
            }
        }
        (x >> 32) as u32
    }

    /// Fisher-Yates in-place shuffle. Used by the dice bag shuffle
    /// and the resource list permutation at board construction.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.gen_range_u32(i as u32 + 1) as usize;
            slice.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// PyO3 exposure — parity-test surface only
// ---------------------------------------------------------------------------

/// Python-callable helper that pulls `n` bytes of keystream from a
/// fresh ChaCha8 stream seeded with `seed_bytes`. The byte-parity
/// test uses this to assert the Rust + Python ref ChaCha8 outputs
/// match.
#[pyfunction]
pub(crate) fn chacha8_keystream(seed_bytes: &[u8], n: usize) -> PyResult<Vec<u8>> {
    if seed_bytes.len() != 32 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed_bytes must be 32 bytes, got {}",
            seed_bytes.len()
        )));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(seed_bytes);
    let mut rng = EngineRng::from_seed(seed);
    let mut out = vec![0u8; n];
    let mut i = 0;
    while i + 4 <= n {
        let word = rng.next_u32();
        out[i..i + 4].copy_from_slice(&word.to_le_bytes());
        i += 4;
    }
    if i < n {
        let word = rng.next_u32();
        let remainder = n - i;
        out[i..n].copy_from_slice(&word.to_le_bytes()[..remainder]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Internal cargo tests — exercise the rng wrapper without crossing
// the PyO3 boundary.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_seed_is_deterministic() {
        let mut a = EngineRng::from_seed([7; 32]);
        let mut b = EngineRng::from_seed([7; 32]);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = EngineRng::from_seed([1; 32]);
        let mut b = EngineRng::from_seed([2; 32]);
        // First 32 outputs should differ in at least one byte.
        let mut diffs = 0;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                diffs += 1;
            }
        }
        assert!(diffs > 20, "expected most outputs to differ, got {diffs}");
    }

    #[test]
    fn gen_range_distribution_is_uniform() {
        let mut rng = EngineRng::from_seed([42; 32]);
        let mut counts = [0u32; 6];
        let n = 60_000;
        for _ in 0..n {
            counts[rng.gen_range_u32(6) as usize] += 1;
        }
        // Each bucket should be within 5% of n/6.
        let expected = n / 6;
        let tol = expected / 20; // 5% tolerance
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                (c as i64 - expected as i64).abs() < tol as i64,
                "bucket {i} count {c} not within tol of expected {expected}"
            );
        }
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = EngineRng::from_seed([3; 32]);
        let mut v: Vec<u8> = (0..36).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..36).collect::<Vec<u8>>());
        // Statistically, the shuffled order is different — check a
        // weak invariant: at least one element moved.
        assert!(v != (0..36).collect::<Vec<u8>>());
    }
}
