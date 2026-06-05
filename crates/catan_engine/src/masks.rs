//! Native action-mask builder — returns 9 boolean masks matching
//! the Python `compute_action_masks` shape (env/masks.py):
//!
//! * `type`: (13,) bool — which of the 13 action types are legal
//! * `corner_settlement`: (54,) bool — legal settlement vertices
//! * `corner_city`: (54,) bool — legal city upgrade vertices
//! * `edge`: (72,) bool — legal road edges
//! * `tile`: (19,) bool — legal robber-destination hexes (Friendly Robber)
//! * `resource1_trade`: (5,) bool — legal give-resources for BankTrade
//! * `resource1_discard`: (5,) bool — legal discard resources
//! * `resource1_default`: (5,) bool — legal res1 for YoP/Monopoly
//! * `resource2_default`: (5,) bool — legal res2 for YoP/BankTrade
//!
//! Reads from `GameState`; pre-allocates each mask per call.

#![allow(clippy::useless_conversion)]

use crate::state::*;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const N_ACTION_TYPES: usize = 13;

/// Build the action-mask dict.
pub fn compute_masks<'py>(py: Python<'py>, state: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);

    // ---- type mask ----
    let type_mask = PyArray1::<bool>::zeros_bound(py, [N_ACTION_TYPES], false);
    {
        let mut arr = unsafe { type_mask.as_array_mut() };
        let p = state.current_player as usize;
        match state.phase {
            GamePhase::Setup => {
                // Setup-step parity decides settle vs road.
                // setup_step: 0=settle1, 1=road1, 2=settle2, 3=road2 (and
                // 4 = setup done). The current_player's just_placed_vertex
                // tells us whether we still need a settlement or road.
                let needs_settlement = state.just_placed_vertex.is_none()
                    || state.vertex_owner[state.just_placed_vertex.unwrap() as usize]
                        != (p as u8 + 1)
                    || state.players[p].roads_left == 15 - state.setup_step / 2;
                // Simplification: in setup, alternate settle/road; let
                // both be legal and let action_build_* enforce.
                arr[ActionType::BuildSettlement as usize] = true;
                arr[ActionType::BuildRoad as usize] = true;
                arr[ActionType::EndTurn as usize] = true;
                let _ = needs_settlement;
            }
            GamePhase::Roll => {
                arr[ActionType::RollDice as usize] = true;
            }
            GamePhase::Main => {
                let r = &state.players[p].resources;
                // Settlement: brick + wood + sheep + wheat ≥ 1 each.
                if r[IDX_BRICK] >= 1
                    && r[IDX_WOOD] >= 1
                    && r[IDX_SHEEP] >= 1
                    && r[IDX_WHEAT] >= 1
                    && state.players[p].settlements_left > 0
                {
                    arr[ActionType::BuildSettlement as usize] = true;
                }
                // City: ore ≥ 3 + wheat ≥ 2.
                if r[IDX_ORE] >= 3 && r[IDX_WHEAT] >= 2 && state.players[p].cities_left > 0 {
                    arr[ActionType::BuildCity as usize] = true;
                }
                // Road: brick + wood ≥ 1 each.
                if r[IDX_BRICK] >= 1 && r[IDX_WOOD] >= 1 && state.players[p].roads_left > 0 {
                    arr[ActionType::BuildRoad as usize] = true;
                }
                // Buy dev card: wheat + sheep + ore ≥ 1 each + deck not empty.
                let deck_total: u32 = state.dev_deck.iter().map(|&v| v as u32).sum();
                if r[IDX_WHEAT] >= 1 && r[IDX_SHEEP] >= 1 && r[IDX_ORE] >= 1 && deck_total > 0 {
                    arr[ActionType::BuyDevCard as usize] = true;
                }
                // Play dev cards (only if hand has them and not played this turn).
                if !state.players[p].dev_card_played_this_turn {
                    if state.players[p].dev_cards_hand[DEV_KNIGHT] > 0 {
                        arr[ActionType::PlayKnight as usize] = true;
                    }
                    if state.players[p].dev_cards_hand[DEV_YEAROFPLENTY] > 0 {
                        arr[ActionType::PlayYoP as usize] = true;
                    }
                    if state.players[p].dev_cards_hand[DEV_MONOPOLY] > 0 {
                        arr[ActionType::PlayMonopoly as usize] = true;
                    }
                    if state.players[p].dev_cards_hand[DEV_ROADBUILDER] > 0
                        && state.players[p].roads_left >= 2
                    {
                        arr[ActionType::PlayRoadBuilder as usize] = true;
                    }
                }
                // Bank trade: at least 4 of one resource (or 3 with generic
                // port or 2 with specific port). Conservative legality:
                // expose true and let action_bank_trade refuse if insufficient.
                arr[ActionType::BankTrade as usize] = true;
                // EndTurn always legal in Main.
                arr[ActionType::EndTurn as usize] = true;
            }
            GamePhase::Discard => {
                arr[ActionType::Discard as usize] = true;
            }
            GamePhase::Robber => {
                arr[ActionType::MoveRobber as usize] = true;
            }
            GamePhase::RoadBuilder => {
                arr[ActionType::BuildRoad as usize] = true;
                arr[ActionType::EndTurn as usize] = true;
            }
            GamePhase::GameOver => {
                // No legal actions; type mask all false.
            }
        }
    }
    out.set_item("type", type_mask)?;

    // ---- corner_settlement (54 bool) ----
    let corner_set = PyArray1::<bool>::zeros_bound(py, [54], false);
    {
        let mut arr = unsafe { corner_set.as_array_mut() };
        let p = state.current_player;
        let is_setup = matches!(state.phase, GamePhase::Setup);
        for v_idx in 0..54u8 {
            if state.vertex_owner[v_idx as usize] != 0 {
                continue;
            }
            // Distance rule.
            let mut neighbor_taken = false;
            for e in state.board.edges() {
                if e.v1_idx == v_idx && state.vertex_owner[e.v2_idx as usize] != 0 {
                    neighbor_taken = true;
                    break;
                }
                if e.v2_idx == v_idx && state.vertex_owner[e.v1_idx as usize] != 0 {
                    neighbor_taken = true;
                    break;
                }
            }
            if neighbor_taken {
                continue;
            }
            if is_setup {
                arr[v_idx as usize] = true;
                continue;
            }
            // Main phase: must be adjacent to a road we own.
            let owner_marker = p + 1;
            let mut adj_own_road = false;
            for e in state.board.edges() {
                if (e.v1_idx == v_idx || e.v2_idx == v_idx)
                    && state.edge_owner[e.edge_idx as usize] == owner_marker
                {
                    adj_own_road = true;
                    break;
                }
            }
            if adj_own_road {
                arr[v_idx as usize] = true;
            }
        }
    }
    out.set_item("corner_settlement", corner_set)?;

    // ---- corner_city (54 bool) ----
    let corner_city = PyArray1::<bool>::zeros_bound(py, [54], false);
    {
        let mut arr = unsafe { corner_city.as_array_mut() };
        let p = state.current_player;
        let settle_marker = p + 1;
        for v_idx in 0..54u8 {
            if state.vertex_owner[v_idx as usize] == settle_marker {
                arr[v_idx as usize] = true;
            }
        }
    }
    out.set_item("corner_city", corner_city)?;

    // ---- edge (72 bool) ----
    let edge_mask = PyArray1::<bool>::zeros_bound(py, [72], false);
    {
        let mut arr = unsafe { edge_mask.as_array_mut() };
        let p = state.current_player;
        let owner_marker_s = p + 1;
        let owner_marker_c = p + 3;
        let opp_s = state.opponent() + 1;
        let opp_c = state.opponent() + 3;
        for e_idx in 0..72u8 {
            if state.edge_owner[e_idx as usize] != 0 {
                continue;
            }
            let edge = state.board.edges()[e_idx as usize];
            let v1_own = state.vertex_owner[edge.v1_idx as usize] == owner_marker_s
                || state.vertex_owner[edge.v1_idx as usize] == owner_marker_c;
            let v2_own = state.vertex_owner[edge.v2_idx as usize] == owner_marker_s
                || state.vertex_owner[edge.v2_idx as usize] == owner_marker_c;
            let mut connected = v1_own || v2_own;
            if !connected {
                // Connected via adjacent owned road if shared vertex
                // isn't blocked by opponent.
                for other in state.board.edges() {
                    if other.edge_idx == e_idx
                        || state.edge_owner[other.edge_idx as usize] != owner_marker_s
                    {
                        continue;
                    }
                    let shares_v1 = other.v1_idx == edge.v1_idx || other.v2_idx == edge.v1_idx;
                    let shares_v2 = other.v1_idx == edge.v2_idx || other.v2_idx == edge.v2_idx;
                    let opp_v1 = state.vertex_owner[edge.v1_idx as usize] == opp_s
                        || state.vertex_owner[edge.v1_idx as usize] == opp_c;
                    let opp_v2 = state.vertex_owner[edge.v2_idx as usize] == opp_s
                        || state.vertex_owner[edge.v2_idx as usize] == opp_c;
                    if (shares_v1 && !opp_v1) || (shares_v2 && !opp_v2) {
                        connected = true;
                        break;
                    }
                }
            }
            if connected {
                arr[e_idx as usize] = true;
            }
        }
    }
    out.set_item("edge", edge_mask)?;

    // ---- tile (19 bool, Friendly Robber) ----
    let tile_mask = PyArray1::<bool>::zeros_bound(py, [19], false);
    {
        let mut arr = unsafe { tile_mask.as_array_mut() };
        for hex_idx in 0..19u8 {
            if hex_idx == state.robber_hex {
                continue;
            }
            // Friendly Robber: any adjacent vertex owned by a player
            // with visible_VP < 3 makes this hex illegal.
            let mut friendly_blocked = false;
            for v in state.board.vertices() {
                if !v.adjacent_hex_indices[..v.adjacent_count as usize].contains(&hex_idx) {
                    continue;
                }
                let owner = state.vertex_owner[v.vertex_idx as usize];
                if owner == 0 {
                    continue;
                }
                let owner_player = if owner == 1 || owner == 3 { 0 } else { 1 };
                if state.players[owner_player as usize].visible_victory_points() < 3 {
                    friendly_blocked = true;
                    break;
                }
            }
            if !friendly_blocked {
                arr[hex_idx as usize] = true;
            }
        }
    }
    out.set_item("tile", tile_mask)?;

    // ---- resource1_trade (5 bool, give-side) ----
    let res1_trade = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_trade.as_array_mut() };
        // CW order. Legal if the player has at least 2 (best port case).
        let p = &state.players[state.current_player as usize];
        let cw_to_eng = [IDX_WOOD, IDX_BRICK, IDX_WHEAT, IDX_ORE, IDX_SHEEP];
        for (cw, &eng) in cw_to_eng.iter().enumerate() {
            if p.resources[eng] >= 2 {
                arr[cw] = true;
            }
        }
    }
    out.set_item("resource1_trade", res1_trade)?;

    // ---- resource1_discard (5 bool) ----
    let res1_disc = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_disc.as_array_mut() };
        let p = &state.players[state.current_player as usize];
        let cw_to_eng = [IDX_WOOD, IDX_BRICK, IDX_WHEAT, IDX_ORE, IDX_SHEEP];
        for (cw, &eng) in cw_to_eng.iter().enumerate() {
            if p.resources[eng] >= 1 {
                arr[cw] = true;
            }
        }
    }
    out.set_item("resource1_discard", res1_disc)?;

    // ---- resource1_default (5 bool, all true for YoP/Mono picks) ----
    let res1_def = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_def.as_array_mut() };
        for i in 0..5 {
            arr[i] = true;
        }
    }
    out.set_item("resource1_default", res1_def)?;

    // ---- resource2_default (5 bool) ----
    let res2_def = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res2_def.as_array_mut() };
        for i in 0..5 {
            arr[i] = true;
        }
    }
    out.set_item("resource2_default", res2_def)?;

    Ok(out)
}
