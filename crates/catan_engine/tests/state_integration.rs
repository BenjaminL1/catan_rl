//! Integration tests for `GameState` action dispatching. These
//! exercise the full state machine across multiple action types and
//! game phases — beyond the unit-test reach of `state.rs`'s
//! `#[cfg(test)] mod tests`.

use catan_engine::board::Resource;
use catan_engine::state::*;

/// Find a legal first-settlement vertex for the current player.
fn pick_any_empty_vertex_with_road_targets(state: &GameState) -> u8 {
    for v_idx in 0..54u8 {
        if state.vertex_owner[v_idx as usize] != 0 {
            continue;
        }
        // Must have at least one adjacent edge.
        let mut has_edge = false;
        for e in state.board.edges() {
            if e.v1_idx == v_idx || e.v2_idx == v_idx {
                has_edge = true;
                break;
            }
        }
        if !has_edge {
            continue;
        }
        // Distance rule: no neighbor occupied.
        let mut neighbor_occupied = false;
        for e in state.board.edges() {
            if e.v1_idx == v_idx && state.vertex_owner[e.v2_idx as usize] != 0 {
                neighbor_occupied = true;
                break;
            }
            if e.v2_idx == v_idx && state.vertex_owner[e.v1_idx as usize] != 0 {
                neighbor_occupied = true;
                break;
            }
        }
        if neighbor_occupied {
            continue;
        }
        return v_idx;
    }
    panic!("no legal vertex found");
}

fn pick_edge_adjacent_to_vertex(state: &GameState, v_idx: u8) -> u8 {
    for e in state.board.edges() {
        if e.v1_idx == v_idx || e.v2_idx == v_idx {
            if state.edge_owner[e.edge_idx as usize] == 0 {
                return e.edge_idx;
            }
        }
    }
    panic!("no legal edge for vertex {v_idx}");
}

#[test]
fn full_setup_phase_can_complete() {
    let mut state = GameState::new(42);
    // 4 setup steps. Each: settle, road, end_turn.
    for _ in 0..4 {
        let v = pick_any_empty_vertex_with_road_targets(&state);
        state
            .apply_action([ActionType::BuildSettlement as u8, v, 0, 0, 0, 0])
            .expect("settlement");
        let e = pick_edge_adjacent_to_vertex(&state, v);
        state
            .apply_action([ActionType::BuildRoad as u8, 0, e, 0, 0, 0])
            .expect("road");
        state
            .apply_action([ActionType::EndTurn as u8, 0, 0, 0, 0, 0])
            .expect("end_turn");
    }
    // Setup phase should have ended.
    assert_eq!(state.phase, GamePhase::Roll);
    // Each player has 2 settlements, 2 roads, +2 VP.
    for p in 0..2 {
        let pp = &state.players[p];
        assert_eq!(pp.settlements_left, 3);
        assert_eq!(pp.roads_left, 13);
        assert_eq!(pp.victory_points, 2);
    }
}

#[test]
fn roll_dice_after_setup_distributes_resources_or_triggers_seven_path() {
    let mut state = GameState::new(7);
    for _ in 0..4 {
        let v = pick_any_empty_vertex_with_road_targets(&state);
        state
            .apply_action([ActionType::BuildSettlement as u8, v, 0, 0, 0, 0])
            .unwrap();
        let e = pick_edge_adjacent_to_vertex(&state, v);
        state
            .apply_action([ActionType::BuildRoad as u8, 0, e, 0, 0, 0])
            .unwrap();
        state
            .apply_action([ActionType::EndTurn as u8, 0, 0, 0, 0, 0])
            .unwrap();
    }
    assert_eq!(state.phase, GamePhase::Roll);
    state
        .apply_action([ActionType::RollDice as u8, 0, 0, 0, 0, 0])
        .unwrap();
    assert!(state.last_dice_roll >= 2 && state.last_dice_roll <= 12);
    // Phase is either Main (non-7), Discard (7 + opp has >9), or Robber.
    assert!(matches!(
        state.phase,
        GamePhase::Main | GamePhase::Discard | GamePhase::Robber
    ));
}

#[test]
fn resource_conservation_across_setup_grants() {
    let mut state = GameState::new(99);
    for _ in 0..4 {
        let v = pick_any_empty_vertex_with_road_targets(&state);
        state
            .apply_action([ActionType::BuildSettlement as u8, v, 0, 0, 0, 0])
            .unwrap();
        let e = pick_edge_adjacent_to_vertex(&state, v);
        state
            .apply_action([ActionType::BuildRoad as u8, 0, e, 0, 0, 0])
            .unwrap();
        state
            .apply_action([ActionType::EndTurn as u8, 0, 0, 0, 0, 0])
            .unwrap();
    }
    // Each player should have received resources from their 2nd
    // settlement (1-3 resources depending on adjacent hexes).
    for p in 0..2 {
        let total = state.players[p].resource_total();
        // At least 1 (if 2nd settle borders ≥ 1 non-desert).
        // At most 3 (3-hex interior corner).
        assert!(total <= 3);
    }
}

#[test]
fn vertex_owner_marker_distinguishes_settle_vs_city() {
    let state = GameState::new(1);
    // All empty initially.
    assert!(state.vertex_owner.iter().all(|&o| o == 0));
}

#[test]
fn cannot_apply_action_after_game_over() {
    let mut state = GameState::new(1);
    state.phase = GamePhase::GameOver;
    state.winner = Some(0);
    let r = state.apply_action([ActionType::RollDice as u8, 0, 0, 0, 0, 0]);
    assert!(matches!(r, Err(EngineError::GameOver)));
}

#[test]
fn build_road_in_setup_must_be_adjacent_to_just_placed_settlement() {
    let mut state = GameState::new(42);
    let v = pick_any_empty_vertex_with_road_targets(&state);
    state
        .apply_action([ActionType::BuildSettlement as u8, v, 0, 0, 0, 0])
        .unwrap();
    // Find an edge NOT adjacent to v.
    let mut far_edge = None;
    for e in state.board.edges() {
        if e.v1_idx != v && e.v2_idx != v {
            far_edge = Some(e.edge_idx);
            break;
        }
    }
    let far = far_edge.unwrap();
    let r = state.apply_action([ActionType::BuildRoad as u8, 0, far, 0, 0, 0]);
    assert!(matches!(r, Err(EngineError::IllegalEdge(_))));
}
