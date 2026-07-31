//! Native action-mask builder — returns the same 12 boolean masks, under the
//! same keys and shapes, as the Python `compute_action_masks` (env/masks.py):
//!
//! * `type`: (13,) bool — which of the 13 action types are legal
//! * `corner_settlement`: (54,) bool — legal settlement vertices
//! * `corner_city`: (54,) bool — legal city upgrade vertices
//! * `edge`: (72,) bool — legal road edges
//! * `tile`: (19,) bool — legal robber-destination hexes (Friendly Robber)
//! * `resource1_trade`: (5,) bool — legal give-resources for BankTrade
//! * `resource1_discard`: (5,) bool — legal discard resources
//! * `resource1_default`: (5,) bool — legal res1 for Monopoly (bank-independent)
//! * `resource1_yop`: (5,) bool — legal FIRST YoP pick (spec D4 bank gate)
//! * `resource2_yop`: (5,) bool — legal DIFFERENT second YoP pick (`bank >= 1`)
//! * `resource2_yop_same`: (5,) bool — legal DOUBLED YoP pick (`bank >= 2`)
//! * `resource2_trade`: (5,) bool — legal BankTrade receive (`bank >= 1`)
//!
//! Reads from `GameState`; pre-allocates each mask per call.
//!
//! # What is mirrored, and what is NOT
//!
//! The D4 **type-level** bank gates ARE mirrored: `PlayYoP` is withheld unless
//! some first pick is bank-legal, and `BankTrade` is withheld unless some
//! `(give, receive)` pair is, with the same partner-drop on `resource1_trade`
//! and the same zeroing of `resource2_trade` when no pair survives. The
//! per-resource D4 vectors themselves (`res1_yop` / `supplied` / `doubled` /
//! `res2_trade` in `resource_legality`) use the same predicates as Python.
//!
//! What is NOT mirrored is *when* each mask is populated. Python's
//! `compute_action_masks` branches on phase and `return _pack()`s EARLY, so
//! every mask irrelevant to that phase stays all-False. This function instead
//! builds all 12 masks unconditionally on every call; only the `type` mask is
//! phase-switched. Consumers must therefore read a sub-mask only when the
//! `type` mask admits an action that uses it.
//!
//! The list below records the divergences known at the time of writing. Treat
//! it as a log of deliberate choices, not as a proof of completeness — this
//! module is not a byte-for-byte port:
//!
//! * **Unconditional sub-mask construction** (the structural one, and the
//!   consequence of the paragraph above): `resource1_yop` / `resource2_yop` /
//!   `resource2_yop_same` are emitted from the BANK ALONE, where Python
//!   populates them only when a YoP card is held and some first pick is
//!   bank-legal; `resource1_default` is hardcoded all-True, where Python sets
//!   it only when a Monopoly card is held; and `corner_settlement` /
//!   `corner_city` / `edge` / `tile` / `resource1_trade` / `resource1_discard`
//!   are populated in EVERY phase, including Setup / Roll / Discard / Robber /
//!   RoadBuilder where Python has already returned.
//! * Python's two empty-mask fallbacks — `resource1_discard[:] = True` when
//!   the discarding player holds nothing, and `tile[:] = True` when no robber
//!   spot survives the Friendly-Robber filter — have no analogue here.
//! * In the road-builder phase Python withholds `BuildRoad` and offers
//!   `EndTurn` when no edge is legal; `GamePhase::RoadBuilder` here always
//!   offers both.
//! * `resource1_trade` is **port-blind**: it asks only for `>= 2` of a
//!   resource (the best-case 2:1 port ratio), where Python checks the
//!   player's actual `portList` for 2:1 / 3:1 / 4:1. The Rust give-side is
//!   therefore a superset; `action_bank_trade` refuses the shortfall.
//! * The **Setup** type mask is deliberately permissive — both
//!   `BuildSettlement` and `BuildRoad` are offered and `action_build_*`
//!   enforces the settle/road alternation, where Python derives the legal
//!   type from `setup_step`.
//! * `BuildSettlement` / `BuildCity` / `BuildRoad` are gated on cost and
//!   pieces-left only; Python additionally requires a non-empty
//!   `get_potential_*` set (the per-index corner/edge masks below still
//!   encode it, so no illegal placement is reachable).
//! * `PlayRoadBuilder` additionally requires `roads_left >= 2` here; Python
//!   requires only the card.
//! * Python's "if nothing is legal, offer `EndTurn`" fallback has no analogue
//!   in `GamePhase::GameOver`, which legitimately has no legal action.

#![allow(clippy::useless_conversion)]

use crate::state::*;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const N_ACTION_TYPES: usize = 13;
const N_CW: usize = 5;

/// Charlesworth (RL) resource order -> engine (alphabetical) index.
const CW_TO_ENG: [usize; N_CW] = [IDX_WOOD, IDX_BRICK, IDX_WHEAT, IDX_ORE, IDX_SHEEP];

/// The bank-derived resource vectors, in Charlesworth order, plus the two
/// type-level gates they imply. Pure — no GIL, no numpy — so the rules are
/// unit-testable without `maturin develop`.
pub(crate) struct ResourceLegality {
    /// `bank[r] >= 1` — a receive the bank can honour once.
    pub supplied: [bool; N_CW],
    /// `bank[r] >= 2` — a receive the bank can honour twice (YoP doubled pick).
    pub doubled: [bool; N_CW],
    /// Legal BankTrade give-side, after dropping gives with no legal partner.
    pub res1_trade: [bool; N_CW],
    /// Legal FIRST YoP pick.
    pub res1_yop: [bool; N_CW],
    /// Legal BankTrade receive-side (zeroed when no `(give, receive)` pair exists).
    pub res2_trade: [bool; N_CW],
    /// `type[PlayYoP]` gate: some first pick is bank-legal.
    pub yop_playable: bool,
    /// `type[BankTrade]` gate: some `(give, receive)` pair is legal.
    pub trade_playable: bool,
}

/// Per-resource bank supply, in Charlesworth order.
pub(crate) fn bank_supply(state: &GameState) -> ([bool; N_CW], [bool; N_CW]) {
    let mut supplied = [false; N_CW];
    let mut doubled = [false; N_CW];
    for (cw, &eng) in CW_TO_ENG.iter().enumerate() {
        supplied[cw] = state.bank[eng] >= 1;
        doubled[cw] = state.bank[eng] >= 2;
    }
    (supplied, doubled)
}

/// Raw BankTrade give-side, BEFORE the no-legal-partner drop. Port-blind (see
/// the module divergence list): `>= 2` is the best-case 2:1 ratio.
pub(crate) fn res1_trade_vec(state: &GameState) -> [bool; N_CW] {
    let p = &state.players[state.current_player as usize];
    let mut arr = [false; N_CW];
    for (cw, &eng) in CW_TO_ENG.iter().enumerate() {
        arr[cw] = p.resources[eng] >= 2;
    }
    arr
}

/// Legal FIRST YoP pick: doubled, or supplied with some other supplied partner.
pub(crate) fn res1_yop_vec(supplied: &[bool; N_CW], doubled: &[bool; N_CW]) -> [bool; N_CW] {
    let mut arr = [false; N_CW];
    for (i, slot) in arr.iter_mut().enumerate() {
        let has_other = (0..N_CW).any(|j| j != i && supplied[j]);
        *slot = doubled[i] || (supplied[i] && has_other);
    }
    arr
}

/// Assemble every bank-derived vector + the two type-level gates.
pub(crate) fn resource_legality(state: &GameState) -> ResourceLegality {
    let (supplied, doubled) = bank_supply(state);
    let res1_yop = res1_yop_vec(&supplied, &doubled);
    let mut res1_trade = res1_trade_vec(state);
    // A receive must be supplyable AND distinct from the give, so drop any
    // give left with no legal partner before deciding the type gate.
    for i in 0..N_CW {
        if res1_trade[i] && !(0..N_CW).any(|j| j != i && supplied[j]) {
            res1_trade[i] = false;
        }
    }
    let mut res2_trade = supplied;
    let trade_playable = res1_trade.iter().any(|&b| b) && res2_trade.iter().any(|&b| b);
    if !trade_playable {
        res2_trade = [false; N_CW];
    }
    ResourceLegality {
        supplied,
        doubled,
        res1_trade,
        res1_yop,
        res2_trade,
        yop_playable: res1_yop.iter().any(|&b| b),
        trade_playable,
    }
}

/// The (13,) type mask. Split out of [`compute_masks`] so the D4 gates are
/// testable in pure Rust.
pub(crate) fn legal_type_mask(state: &GameState, res: &ResourceLegality) -> [bool; N_ACTION_TYPES] {
    let mut arr = [false; N_ACTION_TYPES];
    let p = state.current_player as usize;
    match state.phase {
        GamePhase::Setup => {
            // Deliberately permissive: alternate settle/road is enforced by
            // `action_build_*`, not by the mask.
            arr[ActionType::BuildSettlement as usize] = true;
            arr[ActionType::BuildRoad as usize] = true;
            arr[ActionType::EndTurn as usize] = true;
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
                // D4: a YoP whose picks the finite bank cannot supply is not a
                // legal move — holding the card is not enough.
                if state.players[p].dev_cards_hand[DEV_YEAROFPLENTY] > 0 && res.yop_playable {
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
            // D4: BankTrade needs a give the player can afford AND a receive
            // the bank can supply that is distinct from it.
            if res.trade_playable {
                arr[ActionType::BankTrade as usize] = true;
            }
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
    arr
}

/// Build the action-mask dict.
pub fn compute_masks<'py>(py: Python<'py>, state: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);

    // Bank-derived legality is computed ONCE, above the type mask, because the
    // D4 gates on `PlayYoP` / `BankTrade` are decided by it (spec
    // bc-coverage-and-bank-legality).
    let res = resource_legality(state);

    // ---- type mask ----
    let type_mask = PyArray1::<bool>::zeros_bound(py, [N_ACTION_TYPES], false);
    {
        let mut arr = unsafe { type_mask.as_array_mut() };
        for (i, &legal) in legal_type_mask(state, &res).iter().enumerate() {
            arr[i] = legal;
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
        for i in 0..N_CW {
            arr[i] = res.res1_trade[i];
        }
    }
    out.set_item("resource1_trade", res1_trade)?;

    // ---- resource1_discard (5 bool) ----
    let res1_disc = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_disc.as_array_mut() };
        let p = &state.players[state.current_player as usize];
        for (cw, &eng) in CW_TO_ENG.iter().enumerate() {
            if p.resources[eng] >= 1 {
                arr[cw] = true;
            }
        }
    }
    out.set_item("resource1_discard", res1_disc)?;

    // ---- resource1_default (5 bool, all true — Monopoly is bank-independent) ----
    let res1_def = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_def.as_array_mut() };
        for i in 0..5 {
            arr[i] = true;
        }
    }
    out.set_item("resource1_default", res1_def)?;

    // ---- D4 bank-gated resource masks (spec bc-coverage-and-bank-legality) ----
    // A receive the finite bank cannot honour must never be OFFERED. YoP is
    // legal for `(first, second)` iff `bank[first] >= 2` when they are equal,
    // else `bank[first] >= 1 && bank[second] >= 1`; BankTrade needs
    // `bank[r2] >= 1` with `r2 != r1`. Vectors come from `resource_legality`,
    // computed above the type mask because the type gates depend on them.

    // ---- resource1_yop (5 bool) ----
    let res1_yop = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut arr = unsafe { res1_yop.as_array_mut() };
        for i in 0..N_CW {
            arr[i] = res.res1_yop[i];
        }
    }
    out.set_item("resource1_yop", res1_yop)?;

    // ---- resource2_yop / resource2_yop_same / resource2_trade (5 bool each) ----
    let res2_yop = PyArray1::<bool>::zeros_bound(py, [5], false);
    let res2_yop_same = PyArray1::<bool>::zeros_bound(py, [5], false);
    let res2_trade = PyArray1::<bool>::zeros_bound(py, [5], false);
    {
        let mut a_yop = unsafe { res2_yop.as_array_mut() };
        let mut a_same = unsafe { res2_yop_same.as_array_mut() };
        let mut a_trade = unsafe { res2_trade.as_array_mut() };
        for i in 0..N_CW {
            a_yop[i] = res.supplied[i];
            a_same[i] = res.doubled[i];
            a_trade[i] = res.res2_trade[i];
        }
    }
    out.set_item("resource2_yop", res2_yop)?;
    out.set_item("resource2_yop_same", res2_yop_same)?;
    out.set_item("resource2_trade", res2_trade)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Pure-Rust coverage of the D4 type-level bank gates. No GIL, no numpy —
    //! these run under `cargo test -p catan_engine` without `maturin develop`.
    use super::*;

    /// A Main-phase state with an empty hand, a full dev hand slot, and a bank
    /// the caller sets explicitly.
    fn main_phase_state(bank: [u8; N_RESOURCES]) -> GameState {
        let mut s = GameState::new(7);
        s.phase = GamePhase::Main;
        s.setup_step = 255;
        s.current_player = 0;
        s.bank = bank;
        s.players[0].resources = [0; N_RESOURCES];
        s.players[0].dev_cards_hand = [0; N_DEV_TYPES];
        s.players[0].dev_card_played_this_turn = false;
        s
    }

    // ---- Year of Plenty ----------------------------------------------------

    #[test]
    fn yop_is_illegal_when_the_bank_is_empty() {
        let mut s = main_phase_state([0; N_RESOURCES]);
        s.players[0].dev_cards_hand[DEV_YEAROFPLENTY] = 1;
        let res = resource_legality(&s);
        assert!(!res.yop_playable);
        assert!(!legal_type_mask(&s, &res)[ActionType::PlayYoP as usize]);
    }

    #[test]
    fn yop_with_two_of_exactly_one_resource_offers_only_the_doubled_pick() {
        let mut bank = [0u8; N_RESOURCES];
        bank[IDX_WHEAT] = 2;
        let mut s = main_phase_state(bank);
        s.players[0].dev_cards_hand[DEV_YEAROFPLENTY] = 1;
        let res = resource_legality(&s);
        assert!(res.yop_playable);
        assert!(legal_type_mask(&s, &res)[ActionType::PlayYoP as usize]);
        // CW order is (WOOD, BRICK, WHEAT, ORE, SHEEP) — index 2 is WHEAT.
        assert_eq!(res.res1_yop, [false, false, true, false, false]);
        assert_eq!(res.doubled, [false, false, true, false, false]);
    }

    #[test]
    fn yop_with_one_of_exactly_one_resource_has_no_legal_pair() {
        let mut bank = [0u8; N_RESOURCES];
        bank[IDX_ORE] = 1;
        let mut s = main_phase_state(bank);
        s.players[0].dev_cards_hand[DEV_YEAROFPLENTY] = 1;
        let res = resource_legality(&s);
        assert!(!res.yop_playable, "a single card cannot fund two picks");
        assert!(!legal_type_mask(&s, &res)[ActionType::PlayYoP as usize]);
    }

    // ---- Bank trade --------------------------------------------------------

    #[test]
    fn bank_trade_is_illegal_when_the_bank_is_empty() {
        let mut s = main_phase_state([0; N_RESOURCES]);
        s.players[0].resources[IDX_WOOD] = 6;
        let res = resource_legality(&s);
        assert!(!res.trade_playable);
        assert!(!legal_type_mask(&s, &res)[ActionType::BankTrade as usize]);
        assert_eq!(res.res2_trade, [false; N_CW]);
    }

    #[test]
    fn bank_trade_is_legal_when_the_bank_supplies_a_distinct_receive() {
        let mut bank = [0u8; N_RESOURCES];
        bank[IDX_BRICK] = 3;
        let mut s = main_phase_state(bank);
        s.players[0].resources[IDX_WOOD] = 6;
        let res = resource_legality(&s);
        assert!(res.trade_playable);
        assert!(legal_type_mask(&s, &res)[ActionType::BankTrade as usize]);
        assert_eq!(res.res1_trade, [true, false, false, false, false]);
        assert_eq!(res.res2_trade, [false, true, false, false, false]);
    }

    #[test]
    fn bank_trade_drops_a_give_whose_only_partner_is_itself() {
        // The bank holds ONLY the resource the player would give away, so
        // every receive collides with the give and the move is not legal.
        let mut bank = [0u8; N_RESOURCES];
        bank[IDX_WOOD] = 19;
        let mut s = main_phase_state(bank);
        s.players[0].resources[IDX_WOOD] = 6;
        let res = resource_legality(&s);
        assert_eq!(res.res1_trade, [false; N_CW]);
        assert!(!res.trade_playable);
        assert!(!legal_type_mask(&s, &res)[ActionType::BankTrade as usize]);
    }
}
