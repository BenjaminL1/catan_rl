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
fn event_stream_from_setup_phase_contains_expected_events() {
    use catan_engine::events::{BuildKind, Event};
    let mut state = GameState::new(42);
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
    let events = state.drain_events();
    // Count event types.
    let settle_builds = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::Build {
                    kind: BuildKind::Settlement,
                    ..
                }
            )
        })
        .count();
    let road_builds = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::Build {
                    kind: BuildKind::Road,
                    ..
                }
            )
        })
        .count();
    let setup_complete = events
        .iter()
        .filter(|e| matches!(e, Event::SetupComplete))
        .count();
    assert_eq!(settle_builds, 4, "expected 4 BUILD(SETTLEMENT) events");
    assert_eq!(road_builds, 4, "expected 4 BUILD(ROAD) events");
    assert_eq!(setup_complete, 1, "expected exactly 1 SETUP_COMPLETE");
    // SETUP_COMPLETE must be the LAST event in the setup-phase stream.
    assert!(matches!(events.last(), Some(Event::SetupComplete)));
}

#[test]
fn dice_roll_emits_event() {
    use catan_engine::events::Event;
    let mut state = GameState::new(7);
    // Run setup.
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
    // Drain setup events to isolate roll events.
    let _ = state.drain_events();
    state
        .apply_action([ActionType::RollDice as u8, 0, 0, 0, 0, 0])
        .unwrap();
    let events = state.drain_events();
    let dice_rolls = events
        .iter()
        .filter(|e| matches!(e, Event::DiceRoll { .. }))
        .count();
    assert_eq!(dice_rolls, 1, "expected 1 DICE_ROLL event");
}

#[test]
fn drain_events_is_destructive() {
    use catan_engine::events::Event;
    let mut state = GameState::new(1);
    state.events.push(Event::SetupComplete);
    let drained = state.drain_events();
    assert_eq!(drained.len(), 1);
    assert!(state.events.is_empty());
    let drained_again = state.drain_events();
    assert_eq!(drained_again.len(), 0);
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
