//! Native obs encoder — builds the 9-key obs dict from `GameState`
//! and returns zero-copy `PyArray*` views. The Python encoder
//! (`policy/obs_encoder.py`) is replicated structurally — same
//! dims and dtype.
//!
//! ## Cross-impl byte-parity (Phase 4 pivot, 2026-06-06)
//!
//! The user's PIVOT decision on the Phase 4 review: rewrite the
//! Rust encoder so the populated slots of `tile_representations`
//! and `hex_features` are **byte-identical** with the Python
//! encoder output for the same board layout. This preserves the
//! option to load `checkpoint_07390040.pt` against the Rust path
//! (e.g. for inference benchmarks, deterministic eval, or future
//! MCTS rollouts) without requiring a full retrain.
//!
//! Byte-parity slots (asserted in
//! `tests/unit/engine/test_obs_cross_impl_byte_parity.py`):
//!
//! * `tile_representations[h, 0..6]` — resource one-hot in the
//!   Python order (BRICK, ORE, SHEEP, WHEAT, WOOD, DESERT).
//! * `tile_representations[h, 6..17]` — number-token one-hot in
//!   the Python order (None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12).
//! * `tile_representations[h, 17]` — current-robber bit
//!   (`hex.hex_idx == state.robber_hex`).
//! * `tile_representations[h, 18]` — dot count / 5.0.
//! * `hex_features[h, 0..19]` — same as tile representations
//!   `[0..19]` (Python's `_tile_static` is reused for the GNN's
//!   per-hex input).
//!
//! ## Fresh-train-required slots (Phase 3 badge, still in effect)
//!
//! The vertex / edge content slots in `tile_representations[19..79]`
//! remain zero-filled placeholders. Phase 3's "fresh-train-required"
//! badge covers these. The pivot intentionally does not fill them
//! because (a) they require board-graph translation which the
//! adapter does not yet plumb, and (b) the Python checkpoint was
//! trained against those slots being populated, so loading the
//! ckpt against zero-filled vertex slots will still lose
//! performance even with the [0..19] byte-parity fix. The pivot
//! is a "ckpt-can-load" gate, not a "ckpt-plays-as-well" gate.
//! See `docs/plans/rust_engine_actual_state.md` and the Phase 4
//! reviewer notes.
//!
//! Tensors returned:
//! * `tile_representations`: (19, 79) f32
//! * `current_player_main`: (54,) f32
//! * `next_player_main`: (61,) f32
//! * `current_dev_counts`: (5,) f32
//! * `next_played_dev_counts`: (5,) f32
//! * `hex_features`: (19, 19) f32
//! * `vertex_features`: (54, 16) f32
//! * `edge_features`: (72, 16) f32
//! * `opponent_kind`: () i64
//! * `opponent_policy_id`: () i64
//!
//! Per the senior R0 review, all arrays are owned by numpy
//! (`PyArray*::zeros`) and filled in place via `as_array_mut()` to
//! avoid per-step allocator pressure.

#![allow(clippy::useless_conversion, clippy::manual_range_contains)]

use crate::state::{GameState, N_DEV_TYPES};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const TILE_FEATURE_DIM: usize = 79;
pub const N_TILES: usize = 19;
pub const N_VERTICES: usize = 54;
pub const N_EDGES: usize = 72;
pub const HEX_FEATURE_DIM: usize = 19;
pub const VERTEX_FEATURE_DIM: usize = 16;
pub const EDGE_FEATURE_DIM: usize = 16;
pub const CURR_PLAYER_DIM: usize = 54;
pub const NEXT_PLAYER_DIM: usize = 61;

/// `tile_representations` slot range that the Rust encoder leaves
/// at zero. Phase 3 of the remediation plan opted to surface this
/// honestly via the "fresh-train-required" badge rather than fill
/// the slots without byte-parity against the Python encoder. The
/// parity test at `tests/unit/engine/test_obs_byte_parity.py`
/// asserts these slots are always zero across 1000 random states
/// — any non-zero write here is a Phase 3 regression. The
/// populated slots `[0..19]` (one-hot resource, number token,
/// robber bits) ARE byte-parity-tested against the Python obs.
pub const TILE_PLACEHOLDER_SLOTS: std::ops::Range<usize> = 19..TILE_FEATURE_DIM;

/// Resource one-hot slot in the **Python-encoder** ordering:
/// BRICK=0, ORE=1, SHEEP=2, WHEAT=3, WOOD=4, DESERT=5. The Rust
/// `Resource` enum uses a different internal numbering for
/// gameplay convenience (DESERT=0 so default-zero matches "no
/// resource"); the obs encoder permutes through this helper to
/// preserve byte-parity with the Python encoder and keep
/// `checkpoint_07390040.pt` loadable. Phase 4 pivot
/// (`docs/plans/rust_engine_actual_state.md`).
#[inline]
fn resource_to_python_slot(r: crate::board::Resource) -> usize {
    use crate::board::Resource;
    match r {
        Resource::Brick => 0,
        Resource::Ore => 1,
        Resource::Sheep => 2,
        Resource::Wheat => 3,
        Resource::Wood => 4,
        Resource::Desert => 5,
    }
}

/// Number-token one-hot slot in the **Python-encoder** ordering:
/// the slot relative to the start of the token block. Python orders
/// `(None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)` — desert (None) at
/// slot 0, no slot for the impossible-on-a-hex token 7.
///
/// Returns the relative slot index, NOT the absolute index in the
/// per-tile feature vector. Caller adds 6 (the resource block
/// width) to get the absolute slot.
#[inline]
fn token_to_python_slot(token: Option<u8>) -> Option<usize> {
    match token {
        None => Some(0),
        Some(2) => Some(1),
        Some(3) => Some(2),
        Some(4) => Some(3),
        Some(5) => Some(4),
        Some(6) => Some(5),
        // Token 7 is reserved for dice rolls and never appears on
        // a hex; the Python encoder skips it. The Rust encoder
        // matches: None here means "do not set a one-hot bit."
        Some(7) => None,
        Some(8) => Some(6),
        Some(9) => Some(7),
        Some(10) => Some(8),
        Some(11) => Some(9),
        Some(12) => Some(10),
        _ => None,
    }
}

/// Number of standard 2d6 outcomes that roll each token. Used to
/// produce the normalised dot-count feature at slot 18 of each
/// per-hex block: ``arr[h, 18] = dots(token) / MAX_DOTS``.
/// Matches Python's ``_DOTS_BY_TOKEN`` at
/// `src/catan_rl/policy/obs_encoder.py:106-118`.
#[inline]
fn dots_by_token(token: Option<u8>) -> u8 {
    match token {
        Some(2) | Some(12) => 1,
        Some(3) | Some(11) => 2,
        Some(4) | Some(10) => 3,
        Some(5) | Some(9) => 4,
        Some(6) | Some(8) => 5,
        // None (desert) and 7 (never on a hex) both have 0 dots.
        _ => 0,
    }
}

/// Maximum dot count, used as the denominator in the per-hex
/// dot-count feature at slot 18. Matches Python's `_MAX_DOTS`.
const MAX_DOTS: f32 = 5.0;

/// Build the obs dict. Allocates 8 numpy arrays per call (one per
/// obs key). Future R13 polish: preallocate scratch buffers per env
/// and write in place.
pub fn build_obs<'py>(py: Python<'py>, state: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);

    // -------------------------------------------------------------
    // tile_representations: (19, 79) f32
    //
    // Slot layout (Phase 4 pivot — byte-parity with the Python
    // encoder on the populated `[0..19]` range so
    // `checkpoint_07390040.pt` can load):
    //
    // [0..6]  one-hot resource (Python order: BRICK, ORE, SHEEP,
    //         WHEAT, WOOD, DESERT)
    // [6..17] one-hot number-token (Python order: None, 2, 3, 4,
    //         5, 6, 8, 9, 10, 11, 12). Desert lights up slot 6
    //         (None). Token 7 has no slot because it never
    //         appears on a hex.
    // [17]    has_robber (CURRENT robber bit, not initial).
    //         Matches Python `board.hexTileDict[h].has_robber`.
    // [18]    dot count / 5.0 (normalised). Matches Python
    //         `_DOTS_BY_TOKEN[token] / _MAX_DOTS`. Desert is 0.
    // [19..]  zero-fill placeholders. Fresh-train-required slot
    //         range — see `TILE_PLACEHOLDER_SLOTS` and the Phase
    //         3 badge in the module docstring.
    // -------------------------------------------------------------
    let tiles = PyArray2::<f32>::zeros_bound(py, [N_TILES, TILE_FEATURE_DIM], false);
    {
        let mut arr = unsafe { tiles.as_array_mut() };
        for (h_idx, hex) in state.board.hexes.iter().enumerate() {
            arr[(h_idx, resource_to_python_slot(hex.resource))] = 1.0;
            if let Some(slot) = token_to_python_slot(hex.number_token) {
                arr[(h_idx, 6 + slot)] = 1.0;
            }
            // Slot 17: current-robber flag.
            arr[(h_idx, 17)] = if hex.hex_idx == state.robber_hex {
                1.0
            } else {
                0.0
            };
            // Slot 18: normalised dot count.
            arr[(h_idx, 18)] = (dots_by_token(hex.number_token) as f32) / MAX_DOTS;
        }
    }
    out.set_item("tile_representations", tiles)?;

    // -------------------------------------------------------------
    // current_player_main: (54,) f32  — current player's full feature vector
    // -------------------------------------------------------------
    let curr = PyArray1::<f32>::zeros_bound(py, [CURR_PLAYER_DIM], false);
    {
        let mut arr = unsafe { curr.as_array_mut() };
        fill_player_features(&mut arr.view_mut(), state, state.current_player);
    }
    out.set_item("current_player_main", curr)?;

    // -------------------------------------------------------------
    // next_player_main: (61,) f32  — opponent's feature vector + extras
    // -------------------------------------------------------------
    let next = PyArray1::<f32>::zeros_bound(py, [NEXT_PLAYER_DIM], false);
    {
        let mut arr = unsafe { next.as_array_mut() };
        let opp = state.opponent();
        let mut view = arr.view_mut();
        fill_player_features(&mut view, state, opp);
        // Extra opp-specific slots beyond CURR_PLAYER_DIM=54:
        // [54..59]  opp dev cards played counts (5 entries)
        // [59]      opp has_longest_road flag
        // [60]      opp has_largest_army flag
        for (i, &c) in state.players[opp as usize]
            .dev_cards_played
            .iter()
            .enumerate()
        {
            view[54 + i] = c as f32;
        }
        view[59] = if state.players[opp as usize].has_longest_road {
            1.0
        } else {
            0.0
        };
        view[60] = if state.players[opp as usize].has_largest_army {
            1.0
        } else {
            0.0
        };
    }
    out.set_item("next_player_main", next)?;

    // -------------------------------------------------------------
    // current_dev_counts: (5,) f32 — current player's hidden hand
    // -------------------------------------------------------------
    let cdev = PyArray1::<f32>::zeros_bound(py, [N_DEV_TYPES], false);
    {
        let mut arr = unsafe { cdev.as_array_mut() };
        for (i, &c) in state.players[state.current_player as usize]
            .dev_cards_hand
            .iter()
            .enumerate()
        {
            arr[i] = c as f32;
        }
    }
    out.set_item("current_dev_counts", cdev)?;

    // -------------------------------------------------------------
    // next_played_dev_counts: (5,) f32 — opp's played dev cards
    // -------------------------------------------------------------
    let ndev = PyArray1::<f32>::zeros_bound(py, [N_DEV_TYPES], false);
    {
        let mut arr = unsafe { ndev.as_array_mut() };
        for (i, &c) in state.players[state.opponent() as usize]
            .dev_cards_played
            .iter()
            .enumerate()
        {
            arr[i] = c as f32;
        }
    }
    out.set_item("next_played_dev_counts", ndev)?;

    // -------------------------------------------------------------
    // hex_features: (19, 19) f32 — GNN per-hex input
    //
    // Phase 4 pivot: byte-identical with the first 19 columns of
    // `tile_representations` above. Python builds `hex_features`
    // as `_tile_static.copy()` then overwrites slot 17 with the
    // dynamic robber bit; the Rust encoder follows the same
    // recipe so `checkpoint_07390040.pt`-lineage policies can
    // load. See `src/catan_rl/policy/obs_encoder.py:382-387`.
    // -------------------------------------------------------------
    let hex_feat = PyArray2::<f32>::zeros_bound(py, [N_TILES, HEX_FEATURE_DIM], false);
    {
        let mut arr = unsafe { hex_feat.as_array_mut() };
        for (h_idx, hex) in state.board.hexes.iter().enumerate() {
            arr[(h_idx, resource_to_python_slot(hex.resource))] = 1.0;
            if let Some(slot) = token_to_python_slot(hex.number_token) {
                arr[(h_idx, 6 + slot)] = 1.0;
            }
            arr[(h_idx, 17)] = if hex.hex_idx == state.robber_hex {
                1.0
            } else {
                0.0
            };
            arr[(h_idx, 18)] = (dots_by_token(hex.number_token) as f32) / MAX_DOTS;
        }
    }
    out.set_item("hex_features", hex_feat)?;

    // -------------------------------------------------------------
    // vertex_features: (54, 16) f32 — GNN per-vertex input
    // -------------------------------------------------------------
    let vfeat = PyArray2::<f32>::zeros_bound(py, [N_VERTICES, VERTEX_FEATURE_DIM], false);
    {
        let mut arr = unsafe { vfeat.as_array_mut() };
        for v_idx in 0..N_VERTICES {
            let owner = state.vertex_owner[v_idx];
            // [0]     empty flag
            // [1..3]  one-hot owner_player (0 or 1) for settlement
            // [3..5]  one-hot owner_player for city
            // [5]     constant 1.0 bias
            // [6..14] reserved port flags (8 slots, sparse)
            // [14..]  reserved
            arr[(v_idx, 0)] = if owner == 0 { 1.0 } else { 0.0 };
            match owner {
                1 => arr[(v_idx, 1)] = 1.0,
                2 => arr[(v_idx, 2)] = 1.0,
                3 => arr[(v_idx, 3)] = 1.0,
                4 => arr[(v_idx, 4)] = 1.0,
                _ => {}
            }
            arr[(v_idx, 5)] = 1.0;
            // Port flags: bit set per port adjacent to this vertex.
            for (port_offset, port) in state.board.ports.iter().enumerate() {
                if port.vertex_idx_pair[0] as usize == v_idx
                    || port.vertex_idx_pair[1] as usize == v_idx
                {
                    arr[(v_idx, 6 + port_offset.min(7))] = 1.0;
                }
            }
        }
    }
    out.set_item("vertex_features", vfeat)?;

    // -------------------------------------------------------------
    // edge_features: (72, 16) f32 — GNN per-edge input
    // -------------------------------------------------------------
    let efeat = PyArray2::<f32>::zeros_bound(py, [N_EDGES, EDGE_FEATURE_DIM], false);
    {
        let mut arr = unsafe { efeat.as_array_mut() };
        for e_idx in 0..N_EDGES {
            let owner = state.edge_owner[e_idx];
            // [0]    empty flag
            // [1..3] one-hot owner_player (0 or 1)
            // [3]    constant 1.0 bias
            arr[(e_idx, 0)] = if owner == 0 { 1.0 } else { 0.0 };
            if owner == 1 {
                arr[(e_idx, 1)] = 1.0;
            } else if owner == 2 {
                arr[(e_idx, 2)] = 1.0;
            }
            arr[(e_idx, 3)] = 1.0;
        }
    }
    out.set_item("edge_features", efeat)?;

    // -------------------------------------------------------------
    // opponent_kind: () i64 — placeholder 0 (UNKNOWN) for now
    // -------------------------------------------------------------
    let opp_kind = PyArray1::<i64>::zeros_bound(py, [1], false);
    {
        let mut arr = unsafe { opp_kind.as_array_mut() };
        arr[0] = 0;
    }
    out.set_item("opponent_kind", opp_kind)?;

    // -------------------------------------------------------------
    // opponent_policy_id: () i64 — placeholder 0
    // -------------------------------------------------------------
    let opp_pid = PyArray1::<i64>::zeros_bound(py, [1], false);
    {
        let mut arr = unsafe { opp_pid.as_array_mut() };
        arr[0] = 0;
    }
    out.set_item("opponent_policy_id", opp_pid)?;

    Ok(out)
}

fn fill_player_features(
    arr: &mut numpy::ndarray::ArrayViewMut1<'_, f32>,
    state: &GameState,
    player: u8,
) {
    let p = &state.players[player as usize];
    // [0..5]   resources (engine alpha order)
    for (i, &c) in p.resources.iter().enumerate() {
        arr[i] = c as f32;
    }
    // [5..10]  dev cards hand
    for (i, &c) in p.dev_cards_hand.iter().enumerate() {
        arr[5 + i] = c as f32;
    }
    // [10..15] dev cards played
    for (i, &c) in p.dev_cards_played.iter().enumerate() {
        arr[10 + i] = c as f32;
    }
    // [15] VP / 15.0 (normalized)
    arr[15] = p.victory_points as f32 / 15.0;
    // [16] knights played
    arr[16] = p.knights_played as f32;
    // [17] max road length
    arr[17] = p.max_road_length as f32;
    // [18] has_longest_road
    arr[18] = if p.has_longest_road { 1.0 } else { 0.0 };
    // [19] has_largest_army
    arr[19] = if p.has_largest_army { 1.0 } else { 0.0 };
    // [20] dev_card_played_this_turn
    arr[20] = if p.dev_card_played_this_turn {
        1.0
    } else {
        0.0
    };
    // [21] settlements_left / 5.0
    arr[21] = p.settlements_left as f32 / 5.0;
    // [22] roads_left / 15.0
    arr[22] = p.roads_left as f32 / 15.0;
    // [23] cities_left / 4.0
    arr[23] = p.cities_left as f32 / 4.0;
    // [24..32] port mask bits (8 slots)
    for i in 0..8 {
        arr[24 + i] = if (p.port_mask >> i) & 1 == 1 {
            1.0
        } else {
            0.0
        };
    }
    // [32] Karma buff active for THIS player as roller
    let karma_buffed = matches!(state.last_seven_roller, Some(r) if r != player);
    arr[32] = if karma_buffed { 1.0 } else { 0.0 };
    // [33] turn_count (normalized by 100)
    arr[33] = state.turn_count as f32 / 100.0;
    // [34..54] reserved for future feature engineering.
}
