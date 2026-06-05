//! # catan_engine — Rust implementation of the 1v1 Catan game engine
//!
//! Exposed to Python via PyO3 as the `catan_engine` extension module.
//! See `docs/plans/rust_engine_migration.md` for the migration plan.
//!
//! R0: hello-world skeleton + cargo + maturin round-trip.
//! R1 (current): `StackedDice` + ChaCha8 RNG + byte-parity test.
//! R2: `BoardStatic`.
//! ...

use pyo3::prelude::*;

pub mod board;
pub mod dice;
pub mod env;
pub mod events;
pub mod hand_tracker;
pub mod masks;
pub mod obs;
pub mod rng;
pub mod spiral;
pub mod state;
pub mod vec_env;

/// Smoke-test entry point. Returns a fixed string so the R0
/// acceptance gate can assert the wheel was built + loaded
/// correctly from Python.
#[pyfunction]
fn hello() -> &'static str {
    "hello from rust"
}

/// Returns the crate version string for introspection from Python.
/// Useful when debugging which `.so` was actually loaded after a
/// `maturin develop --release` rebuild.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The Python module definition. PyO3 wires this into
/// `import catan_engine`.
#[pymodule]
fn catan_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    // R1 surfaces: dice + RNG keystream helper for parity tests.
    m.add_class::<dice::PyStackedDice>()?;
    m.add_function(wrap_pyfunction!(rng::chacha8_keystream, m)?)?;
    // R2 surfaces: BoardStatic with the JSON-safe board_static() dict.
    m.add_class::<board::PyBoardStatic>()?;
    // R8 surfaces: RustCatanEnv single-env Gymnasium-style wrapper.
    m.add_class::<env::PyRustEnv>()?;
    // R9 surfaces: RustVectorizedEnv batch-step wrapper.
    m.add_class::<vec_env::PyRustVecEnv>()?;
    Ok(())
}
