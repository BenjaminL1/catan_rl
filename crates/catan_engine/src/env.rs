//! Single-env Rust-backed `CatanEnv` PyO3 wrapper. Exposes the
//! Gymnasium-style API the Python `CatanEnv` class consumes:
//!
//! * `reset(seed) -> obs`
//! * `step(action) -> (obs, reward, terminated, truncated, info)`
//! * `get_action_masks() -> dict`
//! * `drain_events() -> list[dict]` (replay recorder boundary)
//!
//! All hot-path returns are zero-copy numpy arrays. The R9
//! `VectorizedEnv` consumes this same wrapper internally per-env.

#![allow(clippy::useless_conversion, clippy::type_complexity)]

use crate::events::{Event, ResourceChangeSource};
use crate::masks::compute_masks;
use crate::obs::build_obs;
use crate::state::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass(name = "RustCatanEnv", module = "catan_engine")]
pub(crate) struct PyRustEnv {
    pub(crate) state: GameState,
    /// Optional per-episode turn cap. When `Some(n)`, `step` returns
    /// `truncated=true` as soon as `state.turn_count >= n`. Plumbed
    /// from Python at construction (`RustCatanEnv(seed, max_turns)`).
    /// Phase 3 of the Rust migration remediation plan; previously
    /// hardcoded `false` on every step.
    pub(crate) max_turns: Option<u16>,
}

#[pymethods]
impl PyRustEnv {
    #[new]
    #[pyo3(signature = (seed=0, max_turns=None))]
    fn py_new(seed: u64, max_turns: Option<u16>) -> Self {
        Self {
            state: GameState::new(seed),
            max_turns,
        }
    }

    /// Reset the env with a new seed. Returns the initial obs dict.
    /// Does NOT reset `max_turns` — it's an env-level config, not a
    /// per-episode parameter.
    fn reset<'py>(&mut self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyDict>> {
        self.state = GameState::new(seed);
        build_obs(py, &self.state)
    }

    /// Read the current per-episode turn cap. ``None`` means no cap.
    #[getter]
    fn max_turns(&self) -> Option<u16> {
        self.max_turns
    }

    /// Apply an action vector and return (obs, reward, terminated,
    /// truncated, info). Action is a 6-element tuple of u8.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: [u8; 6],
    ) -> PyResult<(Bound<'py, PyDict>, f32, bool, bool, Bound<'py, PyDict>)> {
        // Apply the action; any EngineError becomes a Python exception.
        match self.state.apply_action(action) {
            Ok(_) => {}
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "engine action error: {}",
                    e
                )));
            }
        }
        let obs = build_obs(py, &self.state)?;
        let terminated = matches!(self.state.phase, GamePhase::GameOver);
        // Truncation: fires when ``state.turn_count`` reaches the
        // configured cap. GAE in PPO consumes ``terminated`` and
        // ``truncated`` separately; the previous hardcoded ``false``
        // was a structural bug that broke bootstrapping (see
        // ``docs/plans/rust_engine_actual_state.md``).
        let truncated = !terminated
            && self
                .max_turns
                .is_some_and(|cap| self.state.turn_count >= cap);
        let reward = if terminated {
            match self.state.winner {
                Some(w) if w == self.state.current_player => 1.0_f32,
                Some(_) => -1.0_f32,
                None => 0.0_f32,
            }
        } else {
            0.0_f32
        };
        let info = PyDict::new_bound(py);
        info.set_item("phase", self.state.phase as u8)?;
        info.set_item("current_player", self.state.current_player)?;
        info.set_item("turn_count", self.state.turn_count)?;
        info.set_item("last_dice_roll", self.state.last_dice_roll)?;
        Ok((obs, reward, terminated, truncated, info))
    }

    /// Compute the 9-key action mask dict for the current state.
    fn get_action_masks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        compute_masks(py, &self.state)
    }

    /// Drain buffered events. Returns a list of dicts mirroring the
    /// Python `BroadcastEventType` payload shape so the replay
    /// recorder's `classify_step_events` can consume them.
    fn drain_events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let events = self.state.drain_events();
        let out = PyList::empty_bound(py);
        for ev in events {
            let d = PyDict::new_bound(py);
            d.set_item("type", ev.type_str())?;
            match ev {
                Event::DiceRoll { player, value } => {
                    d.set_item("player", player)?;
                    d.set_item("value", value)?;
                }
                Event::Discard { player, resources } => {
                    d.set_item("player", player)?;
                    d.set_item("resources", resources)?;
                }
                Event::YearOfPlenty { player, resources } => {
                    d.set_item("player", player)?;
                    d.set_item("resources", resources.to_vec())?;
                }
                Event::ResourceChange {
                    player,
                    delta_alpha,
                    source,
                } => {
                    d.set_item("player", player)?;
                    d.set_item("delta_alpha", delta_alpha.to_vec())?;
                    d.set_item("source", source.as_str())?;
                }
                Event::Monopoly {
                    player,
                    resource_cw,
                    count,
                } => {
                    d.set_item("player", player)?;
                    d.set_item("resource", resource_cw)?;
                    d.set_item("count", count)?;
                }
                Event::MoveRobber { player, hex_idx } => {
                    d.set_item("player", player)?;
                    d.set_item("hex_idx", hex_idx)?;
                }
                Event::Steal {
                    robber,
                    victim,
                    resource_cw,
                } => {
                    d.set_item("robber", robber)?;
                    d.set_item("victim", victim)?;
                    d.set_item("resource", resource_cw)?;
                }
                Event::Build {
                    player,
                    kind,
                    location,
                } => {
                    d.set_item("player", player)?;
                    d.set_item("kind", kind.as_str())?;
                    d.set_item("location", location)?;
                }
                Event::LongestRoadChange {
                    prev_owner,
                    new_owner,
                    length,
                } => {
                    d.set_item("prev_owner", prev_owner)?;
                    d.set_item("new_owner", new_owner)?;
                    d.set_item("length", length)?;
                }
                Event::LargestArmyChange {
                    prev_owner,
                    new_owner,
                    knights,
                } => {
                    d.set_item("prev_owner", prev_owner)?;
                    d.set_item("new_owner", new_owner)?;
                    d.set_item("knights", knights)?;
                }
                Event::GameEnd {
                    winner,
                    vp_p0,
                    vp_p1,
                } => {
                    d.set_item("winner", winner)?;
                    d.set_item("vp_p0", vp_p0)?;
                    d.set_item("vp_p1", vp_p1)?;
                }
                Event::SetupComplete => {}
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// Current player getter (for the Python env adapter).
    #[getter]
    fn current_player(&self) -> u8 {
        self.state.current_player
    }

    /// Phase as a u8 enum value (for the Python env adapter).
    #[getter]
    fn phase(&self) -> u8 {
        self.state.phase as u8
    }

    /// Game-over flag.
    #[getter]
    fn game_over(&self) -> bool {
        matches!(self.state.phase, GamePhase::GameOver)
    }

    /// Winner (0 or 1), or None if game not over.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.state.winner
    }
}

// Suppress unused-import warnings — `ResourceChangeSource::as_str`
// is the indirect consumer.
#[allow(dead_code)]
fn _force_use_change_source(_: ResourceChangeSource) {}
