//! Hand tracker — exposes per-player resource hands in Charlesworth
//! order (WOOD, BRICK, WHEAT, ORE, SHEEP), reading from the
//! canonical engine state. The Python `BroadcastHandTracker`
//! maintains its own integer counts by subscribing to the broadcast
//! bus; here the engine already owns the canonical state so the
//! tracker is just a thin permutation getter.
//!
//! Resource ordering semantics:
//! * **Engine internal** = alphabetical (BRICK, ORE, SHEEP, WHEAT, WOOD).
//! * **Charlesworth wire** = WOOD, BRICK, WHEAT, ORE, SHEEP.
//!
//! The permutation matrix is fixed at compile time.

use crate::state::{GameState, IDX_BRICK, IDX_ORE, IDX_SHEEP, IDX_WHEAT, IDX_WOOD, N_RESOURCES};

/// Charlesworth-order indexing → engine-alpha indexing.
/// CW[0] = WOOD → engine 4 = IDX_WOOD
/// CW[1] = BRICK → engine 0 = IDX_BRICK
/// CW[2] = WHEAT → engine 3 = IDX_WHEAT
/// CW[3] = ORE → engine 1 = IDX_ORE
/// CW[4] = SHEEP → engine 2 = IDX_SHEEP
pub const CW_PERMUTATION: [usize; N_RESOURCES] =
    [IDX_WOOD, IDX_BRICK, IDX_WHEAT, IDX_ORE, IDX_SHEEP];

/// Return ``player``'s resource hand in Charlesworth order.
///
/// Player index is 0 or 1. Out-of-range yields all zeros (defensive).
pub fn get_hand_cw(state: &GameState, player_idx: u8) -> [u8; N_RESOURCES] {
    if player_idx as usize >= 2 {
        return [0; N_RESOURCES];
    }
    let p = &state.players[player_idx as usize];
    let mut out = [0u8; N_RESOURCES];
    for (cw_idx, &eng_idx) in CW_PERMUTATION.iter().enumerate() {
        out[cw_idx] = p.resources[eng_idx];
    }
    out
}

/// Return the engine-order hand (BRICK, ORE, SHEEP, WHEAT, WOOD).
/// Useful for tests that pin the internal layout.
pub fn get_hand_engine(state: &GameState, player_idx: u8) -> [u8; N_RESOURCES] {
    if player_idx as usize >= 2 {
        return [0; N_RESOURCES];
    }
    state.players[player_idx as usize].resources
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;

    #[test]
    fn cw_permutation_is_a_valid_5_element_permutation() {
        let mut sorted = CW_PERMUTATION;
        sorted.sort();
        assert_eq!(sorted, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn cw_permutation_matches_documented_mapping() {
        // WOOD: CW=0, engine=4
        assert_eq!(CW_PERMUTATION[0], IDX_WOOD);
        // BRICK: CW=1, engine=0
        assert_eq!(CW_PERMUTATION[1], IDX_BRICK);
        // WHEAT: CW=2, engine=3
        assert_eq!(CW_PERMUTATION[2], IDX_WHEAT);
        // ORE: CW=3, engine=1
        assert_eq!(CW_PERMUTATION[3], IDX_ORE);
        // SHEEP: CW=4, engine=2
        assert_eq!(CW_PERMUTATION[4], IDX_SHEEP);
    }

    #[test]
    fn empty_state_has_zero_hands() {
        let state = GameState::new(1);
        assert_eq!(get_hand_cw(&state, 0), [0; N_RESOURCES]);
        assert_eq!(get_hand_cw(&state, 1), [0; N_RESOURCES]);
    }

    #[test]
    fn out_of_range_player_returns_zero() {
        let state = GameState::new(1);
        assert_eq!(get_hand_cw(&state, 2), [0; N_RESOURCES]);
        assert_eq!(get_hand_cw(&state, 99), [0; N_RESOURCES]);
    }

    #[test]
    fn hand_cw_pulls_correct_indices_from_engine_order() {
        // Set up: engine resources = [3, 7, 1, 5, 2] (B, O, S, W, Wd)
        // Expected CW = [Wd=2, B=3, W=5, O=7, S=1]
        let mut state = GameState::new(1);
        state.players[0].resources = [3, 7, 1, 5, 2];
        let cw = get_hand_cw(&state, 0);
        assert_eq!(cw, [2, 3, 5, 7, 1]);
    }

    #[test]
    fn engine_order_getter_returns_internal_array() {
        let mut state = GameState::new(1);
        state.players[1].resources = [9, 8, 7, 6, 5];
        assert_eq!(get_hand_engine(&state, 1), [9, 8, 7, 6, 5]);
    }
}
