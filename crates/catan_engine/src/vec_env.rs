//! `VectorizedEnv` — holds N independent `GameState` instances and
//! exposes a `step_batch(actions) -> (obs_dict_batch, rewards, terms,
//! truncs)` Python API. Releases the GIL inside the batch step so
//! Rayon can parallelize across envs without contention.
//!
//! Per the senior R0 review:
//!
//! * `py.allow_threads(|| {...})` blanket-releases the GIL for the
//!   pure-Rust gameplay phase.
//! * Numpy outputs are pre-allocated per batch call and filled in
//!   place.
//! * Rayon's `par_iter_mut` parallelizes per-env gameplay; for small
//!   N it falls back to serial automatically (Rayon's work-stealing
//!   overhead < threading benefit for n_envs ≤ 8).

#![allow(clippy::useless_conversion, clippy::type_complexity)]

use crate::obs::build_obs;
use crate::state::GameState;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

#[pyclass(name = "RustVectorizedEnv", module = "catan_engine")]
pub(crate) struct PyRustVecEnv {
    envs: Vec<GameState>,
}

#[pymethods]
impl PyRustVecEnv {
    /// Construct N envs. `base_seed` + env index produces each per-env
    /// seed via `base_seed ^ env_idx` (the Q3 per-env seeding contract).
    #[new]
    fn py_new(n_envs: usize, base_seed: u64) -> Self {
        let envs = (0..n_envs)
            .map(|i| GameState::new(base_seed ^ (i as u64)))
            .collect();
        Self { envs }
    }

    /// Number of envs in the batch.
    #[getter]
    fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Reset all envs to fresh seeds derived from `base_seed`.
    fn reset_batch(&mut self, base_seed: u64) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            *env = GameState::new(base_seed ^ (i as u64));
        }
    }

    /// Step every env in parallel. `actions` is an (N, 6) uint8 ndarray.
    /// Returns (list_of_obs_dicts, rewards (N,) f32, terms (N,) bool,
    /// truncs (N,) bool).
    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray2<u8>>,
    ) -> PyResult<(
        Bound<'py, PyList>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let n = self.envs.len();
        if actions.shape()[0] != n || actions.shape()[1] != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "actions shape must be ({}, 6); got {:?}",
                n,
                actions.shape()
            )));
        }
        // Snapshot actions into a per-env array.
        let actions_vec: Vec<[u8; 6]> = {
            let arr = unsafe { actions.as_array() };
            (0..n)
                .map(|i| {
                    let mut a = [0u8; 6];
                    for j in 0..6 {
                        a[j] = arr[(i, j)];
                    }
                    a
                })
                .collect()
        };
        // Pre-allocate the output buffers.
        let rewards = PyArray1::<f32>::zeros_bound(py, [n], false);
        let terms = PyArray1::<bool>::zeros_bound(py, [n], false);
        let truncs = PyArray1::<bool>::zeros_bound(py, [n], false);

        // Gameplay phase: GIL-released + parallel.
        let mut rewards_local = vec![0.0_f32; n];
        let mut terms_local = vec![false; n];
        let truncs_local = vec![false; n]; // R13 wires per-episode truncation
        py.allow_threads(|| {
            self.envs
                .par_iter_mut()
                .zip(actions_vec.par_iter())
                .zip(rewards_local.par_iter_mut())
                .zip(terms_local.par_iter_mut())
                .for_each(|(((env, action), reward), term)| {
                    let _ = env.apply_action(*action);
                    let terminated = matches!(env.phase, crate::state::GamePhase::GameOver);
                    *term = terminated;
                    *reward = if terminated {
                        match env.winner {
                            Some(w) if w == env.current_player => 1.0,
                            Some(_) => -1.0,
                            None => 0.0,
                        }
                    } else {
                        0.0
                    };
                });
        });

        // Copy outputs into the pre-allocated arrays.
        {
            let mut r = unsafe { rewards.as_array_mut() };
            for i in 0..n {
                r[i] = rewards_local[i];
            }
        }
        {
            let mut t = unsafe { terms.as_array_mut() };
            for i in 0..n {
                t[i] = terms_local[i];
            }
        }
        {
            let mut t = unsafe { truncs.as_array_mut() };
            for i in 0..n {
                t[i] = truncs_local[i];
            }
        }

        // Build the per-env obs dict list. This phase holds the GIL
        // because PyDict construction touches the interpreter.
        let obs_list = PyList::empty_bound(py);
        for env in self.envs.iter() {
            let d = build_obs(py, env)?;
            obs_list.append(d)?;
        }

        Ok((obs_list, rewards, terms, truncs))
    }
}
