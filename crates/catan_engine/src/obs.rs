//! Native obs encoder — builds the 9-key obs dict from `GameState`
//! and returns zero-copy `PyArray*` views. The Python encoder
//! (`policy/obs_encoder.py`) is replicated structurally — same
//! dims and dtype — but per-feature byte-identity is deferred to
//! R10 (the cutover will either re-train or wire FFI passthrough).
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

/// Build the obs dict. Allocates 8 numpy arrays per call (one per
/// obs key). Future R13 polish: preallocate scratch buffers per env
/// and write in place.
pub fn build_obs<'py>(py: Python<'py>, state: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);

    // -------------------------------------------------------------
    // tile_representations: (19, 79) f32
    // -------------------------------------------------------------
    let tiles = PyArray2::<f32>::zeros_bound(py, [N_TILES, TILE_FEATURE_DIM], false);
    {
        let mut arr = unsafe { tiles.as_array_mut() };
        for (h_idx, hex) in state.board.hexes.iter().enumerate() {
            // Per-hex feature slots:
            // [0..6]  one-hot resource (DESERT, WOOD, BRICK, WHEAT, ORE, SHEEP)
            // [6..17] one-hot number_token (2..12)
            // [17]    has_robber
            // [18]    is_currently_blocked_by_robber (==robber_hex flag)
            // [19..]  filler — placeholder for vertex/edge ownership flags
            //         and port adjacency. Filled by future R13 polish.
            arr[(h_idx, hex.resource as usize)] = 1.0;
            if let Some(tok) = hex.number_token {
                if tok >= 2 && tok <= 12 {
                    arr[(h_idx, 6 + (tok as usize - 2))] = 1.0;
                }
            }
            arr[(h_idx, 17)] = if hex.has_robber_initial { 1.0 } else { 0.0 };
            arr[(h_idx, 18)] = if hex.hex_idx == state.robber_hex {
                1.0
            } else {
                0.0
            };
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
    // -------------------------------------------------------------
    let hex_feat = PyArray2::<f32>::zeros_bound(py, [N_TILES, HEX_FEATURE_DIM], false);
    {
        let mut arr = unsafe { hex_feat.as_array_mut() };
        for (h_idx, hex) in state.board.hexes.iter().enumerate() {
            // [0..6]  one-hot resource
            // [6..17] one-hot number_token (2..12)
            // [17]    robber_here
            // [18]    constant 1.0 bias
            arr[(h_idx, hex.resource as usize)] = 1.0;
            if let Some(tok) = hex.number_token {
                if tok >= 2 && tok <= 12 {
                    arr[(h_idx, 6 + (tok as usize - 2))] = 1.0;
                }
            }
            arr[(h_idx, 17)] = if hex.hex_idx == state.robber_hex {
                1.0
            } else {
                0.0
            };
            arr[(h_idx, 18)] = 1.0;
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
