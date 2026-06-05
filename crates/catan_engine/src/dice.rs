//! `StackedDice` — Colonist.io's variant of 2d6 used in 1v1 Catan.
//!
//! Mechanics (per CLAUDE.md):
//!
//! 1. **Pre-shuffled bag of 36 sums** matching the standard 2d6
//!    distribution: `{2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3,
//!    11:2, 12:1}` = 36 total.
//! 2. **Noise swap**: remove one random non-7 from the bag, replace
//!    with a uniform random number in `[2, 12]`.
//! 3. **Shuffle** the bag.
//! 4. **Karma**: on `roll(current_player, last_seven_roller)`, if
//!    `last_seven_roller` is `Some(p)` AND `p != current_player`,
//!    roll a uniform `[0, 99]` — if < 20, return 7 immediately
//!    WITHOUT consuming the bag.
//! 5. **Refill**: if the bag is empty, refill (regenerates the
//!    distribution + noise swap + shuffle).
//! 6. **Standard roll**: pop one value from the back of the bag.
//!
//! Note: per the Q1 decision (statistical equivalence, not bit
//! parity with the Python `StackedDice`), this impl uses ChaCha8 +
//! Lemire rejection sampling — the bag distribution is statistically
//! identical to Python but the per-call byte stream differs.

use crate::rng::EngineRng;
use pyo3::prelude::*;

/// Number of dice outcomes pulled from the bag before a refill.
const BAG_SIZE: usize = 36;

/// The Karma forced-7 probability — 20% per the 1v1 Colonist ruleset.
const KARMA_PROBABILITY_PERCENT: u32 = 20;

/// Standard 2d6 sum distribution. Sum 0 and 1 are placeholders so
/// indexing by `sum` is direct.
const SUM_DISTRIBUTION: [u8; 13] = [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1];

#[derive(Clone)]
pub struct StackedDice {
    bag: Vec<u8>,
    rng: EngineRng,
}

impl StackedDice {
    /// Build a fresh dice rolling state seeded with the given 32-byte
    /// ChaCha8 seed. Bag is filled + shuffled at construction.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let mut dice = Self {
            bag: Vec::with_capacity(BAG_SIZE),
            rng: EngineRng::from_seed(seed),
        };
        dice.refill_bag();
        dice
    }

    /// Convenience constructor — same as `from_seed` but via u64.
    pub fn from_u64(seed: u64) -> Self {
        let mut dice = Self {
            bag: Vec::with_capacity(BAG_SIZE),
            rng: EngineRng::from_u64(seed),
        };
        dice.refill_bag();
        dice
    }

    /// Refill the bag with the standard distribution, apply the
    /// noise swap, and shuffle. Mirrors `dice.py:_refill_bag`.
    fn refill_bag(&mut self) {
        self.bag.clear();
        for (sum, &count) in SUM_DISTRIBUTION.iter().enumerate() {
            for _ in 0..count {
                self.bag.push(sum as u8);
            }
        }
        debug_assert_eq!(self.bag.len(), BAG_SIZE);

        // Noise swap: remove one random non-7, replace with uniform [2, 12].
        let non_seven_indices: Vec<usize> = self
            .bag
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 7 { Some(i) } else { None })
            .collect();
        if !non_seven_indices.is_empty() {
            let pick = self.rng.gen_range_u32(non_seven_indices.len() as u32) as usize;
            let remove_idx = non_seven_indices[pick];
            self.bag.swap_remove(remove_idx);
            // Replace with a uniform random sum in [2, 12].
            let replacement = (self.rng.gen_range_u32(11) + 2) as u8;
            self.bag.push(replacement);
        }

        self.rng.shuffle(&mut self.bag);
    }

    /// Roll the dice for `current_player`. Returns the sum (2-12,
    /// or specifically 7 on Karma fire).
    ///
    /// `last_seven_roller`: `None` if no 7 has ever been rolled in
    /// this game, else the player id (0 or 1) of whoever rolled the
    /// most recent 7.
    pub fn roll(&mut self, current_player: u8, last_seven_roller: Option<u8>) -> u8 {
        // Karma check (Friendly-7 mechanic).
        if let Some(roller) = last_seven_roller {
            if roller != current_player {
                let d100 = self.rng.gen_range_u32(100);
                if d100 < KARMA_PROBABILITY_PERCENT {
                    return 7;
                }
            }
        }

        // Refill check.
        if self.bag.is_empty() {
            self.refill_bag();
        }

        // Standard pop.
        self.bag.pop().expect("bag is non-empty after refill check")
    }

    /// Read-only view of the current bag — useful for tests + the
    /// future `bag_remaining` obs feature.
    pub fn bag_view(&self) -> &[u8] {
        &self.bag
    }

    /// Bag size for parity with `len(dice.bag)`.
    pub fn bag_remaining(&self) -> usize {
        self.bag.len()
    }
}

// ---------------------------------------------------------------------------
// PyO3 exposure
// ---------------------------------------------------------------------------

/// Python-facing wrapper. Pinning the API surface that the existing
/// Python `engine/dice.py` consumes — the Python shim will instantiate
/// this and call `roll()`.
#[pyclass(name = "StackedDice", module = "catan_engine")]
pub(crate) struct PyStackedDice {
    inner: StackedDice,
}

#[pymethods]
impl PyStackedDice {
    /// Construct from a u64 seed. If `seed=None`, defaults to a
    /// random-but-stable seed derived from system entropy via
    /// `getrandom`. The Python shim normally passes an explicit seed
    /// so games are reproducible.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn py_new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            // Best-effort entropy from the system clock. The Python
            // ``engine/dice.py`` used the global ``random`` module
            // which seeds itself from ``os.urandom``; for the engine
            // we expect explicit seeds from the env at every reset,
            // so this fallback is only hit by ad-hoc Python callers.
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });
        Self {
            inner: StackedDice::from_u64(seed),
        }
    }

    /// Roll the dice. Matches the Python signature
    /// ``roll(current_player_obj, last_7_roller_obj)`` — except
    /// the Rust impl takes player IDs (0 or 1) rather than player
    /// objects.
    #[pyo3(name = "roll", signature = (current_player, last_seven_roller=None))]
    fn py_roll(&mut self, current_player: u8, last_seven_roller: Option<u8>) -> u8 {
        self.inner.roll(current_player, last_seven_roller)
    }

    /// Read-only access to the bag for the future
    /// ``bag_remaining`` obs feature (A2 in the obs-schema plan).
    #[pyo3(name = "bag_view")]
    fn py_bag_view(&self) -> Vec<u8> {
        self.inner.bag_view().to_vec()
    }

    #[pyo3(name = "bag_remaining")]
    fn py_bag_remaining(&self) -> usize {
        self.inner.bag_remaining()
    }

    /// Support for ``copy.deepcopy``. The Python ``catanGame.copy()``
    /// path runs ``deepcopy(self.dice)``; the Rust impl clones the
    /// bag + RNG state in one ``derive(Clone)`` shot, preserving the
    /// internal ChaCha8 counter so the cloned dice continues the
    /// same keystream the original was about to produce. The
    /// ``memo`` arg is ignored — there are no shared substructures
    /// inside the dice to deduplicate.
    fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyDict>) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// ``copy.copy`` shorthand — the dice has no shared state, so
    /// shallow copy is the same as deep.
    fn __copy__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal cargo tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bag_size_is_36() {
        let dice = StackedDice::from_seed([1; 32]);
        assert_eq!(dice.bag_remaining(), BAG_SIZE);
    }

    #[test]
    fn all_bag_sums_are_in_range() {
        let dice = StackedDice::from_seed([1; 32]);
        for &v in dice.bag_view() {
            assert!((2..=12).contains(&v), "bag sum {v} out of range");
        }
    }

    #[test]
    fn distribution_over_many_games_is_close_to_2d6() {
        // Roll 36 * 1000 = 36k rolls; expect each face's frequency
        // to be within a few % of the standard 2d6 distribution.
        // The noise swap perturbs the per-game distribution but
        // averages to ~uniform-weighted-by-pmf over many games.
        let mut dice = StackedDice::from_seed([99; 32]);
        let mut counts = [0u32; 13];
        for _ in 0..36_000 {
            let v = dice.roll(0, None);
            counts[v as usize] += 1;
        }
        // Compare against 2d6 expected fractions.
        let total = 36_000.0;
        let expected_fractions = [
            (2, 1.0 / 36.0),
            (3, 2.0 / 36.0),
            (4, 3.0 / 36.0),
            (5, 4.0 / 36.0),
            (6, 5.0 / 36.0),
            (7, 6.0 / 36.0),
            (8, 5.0 / 36.0),
            (9, 4.0 / 36.0),
            (10, 3.0 / 36.0),
            (11, 2.0 / 36.0),
            (12, 1.0 / 36.0),
        ];
        for &(sum, expected) in &expected_fractions {
            let observed = counts[sum] as f64 / total;
            let err = (observed - expected).abs();
            // Generous tolerance (4%) — the noise swap inflates the
            // variance of less-common sums and one of the per-game
            // bags can carry an extra 2 or 12.
            assert!(
                err < 0.04,
                "sum={sum} observed={observed} expected={expected} err={err}"
            );
        }
    }

    #[test]
    fn karma_returns_7_when_buffed_player_rolls() {
        // Set up the buff: player 0 rolled the previous 7.
        // Now player 1 rolls — they should hit 7 sometimes via Karma
        // (NOT from the bag). We can't deterministically assert the
        // Karma fired, but we can assert the 7-rate is well above
        // the bag's 6/36 = 16.7%.
        let mut dice = StackedDice::from_seed([42; 32]);
        let mut sevens = 0;
        let n = 1000;
        for _ in 0..n {
            let v = dice.roll(1, Some(0));
            if v == 7 {
                sevens += 1;
            }
        }
        let rate = sevens as f64 / n as f64;
        // With Karma: P(7) = 0.20 + 0.80 * (6/36 + noise effect).
        // Expected rate ~0.33; we accept anything > 0.25 as evidence
        // the Karma path fired.
        assert!(rate > 0.25, "Karma seven rate {rate} too low");
    }

    #[test]
    fn karma_inactive_when_same_player_rolls_again() {
        // Buff is on player 0. Player 0 rolls again — no Karma.
        // The 7-rate should match the bag's natural rate.
        let mut dice = StackedDice::from_seed([7; 32]);
        let mut sevens = 0;
        let n = 1000;
        for _ in 0..n {
            let v = dice.roll(0, Some(0));
            if v == 7 {
                sevens += 1;
            }
        }
        let rate = sevens as f64 / n as f64;
        // Natural bag rate ≈ 6/36 ≈ 0.167; noise swap can push it
        // a bit. Accept [0.10, 0.25].
        assert!(
            (0.10..0.25).contains(&rate),
            "no-Karma seven rate {rate} out of expected range"
        );
    }

    #[test]
    fn karma_inactive_when_no_prior_seven() {
        let mut dice = StackedDice::from_seed([100; 32]);
        let mut sevens = 0;
        let n = 1000;
        for _ in 0..n {
            // last_seven_roller=None → Karma path skipped.
            let v = dice.roll(0, None);
            if v == 7 {
                sevens += 1;
            }
        }
        let rate = sevens as f64 / n as f64;
        assert!(
            (0.10..0.25).contains(&rate),
            "no-prior-7 seven rate {rate} out of expected range"
        );
    }

    #[test]
    fn refills_when_bag_empties() {
        let mut dice = StackedDice::from_seed([1; 32]);
        let initial = dice.bag_remaining();
        for _ in 0..initial {
            let _ = dice.roll(0, None);
        }
        // Bag should have refilled at some point — at least one more
        // roll succeeds.
        let v = dice.roll(0, None);
        assert!((2..=12).contains(&v));
    }
}
