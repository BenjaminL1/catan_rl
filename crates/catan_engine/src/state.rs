//! `GameState` — full 1v1 Catan game state with all 13 action types.
//!
//! Per the migration plan, state is a `#[repr(C)]` struct ≤ ~400
//! bytes so snapshots are a single memcpy. All resource and dev-card
//! counts use fixed-size `[u8; 5]` arrays indexed by enum so the
//! engine boundary carries no string keys.
//!
//! Resource ordering: **engine** = `BRICK, ORE, SHEEP, WHEAT, WOOD`
//! (alphabetical, matches Python's `engine/player.py`); **wire to
//! Python/RL** = Charlesworth order `WOOD, BRICK, WHEAT, ORE, SHEEP`.
//! Conversion happens at the FFI boundary (see `hand_tracker.rs` in
//! R5). Internally we always use the engine order.

use crate::board::{BoardStatic, Resource};
use crate::dice::StackedDice;
use crate::events::{BuildKind, Event, ResourceChangeSource};
use crate::rng::EngineRng;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Charlesworth ↔ engine-alpha resource ordering helpers. These appear
// in PlayYoP / PlayMonopoly / BankTrade / Discard at the FFI boundary
// and are now centralized.
// ---------------------------------------------------------------------------

/// Convert a Charlesworth-order index (WOOD=0, BRICK=1, WHEAT=2,
/// ORE=3, SHEEP=4) into the engine-alpha index (BRICK=0, ORE=1,
/// SHEEP=2, WHEAT=3, WOOD=4). Used at every action-input boundary.
pub fn cw_to_engine(cw: u8) -> Option<usize> {
    Some(match cw {
        0 => IDX_WOOD,
        1 => IDX_BRICK,
        2 => IDX_WHEAT,
        3 => IDX_ORE,
        4 => IDX_SHEEP,
        _ => return None,
    })
}

/// Reverse map: engine-alpha index → Charlesworth-order index.
pub fn engine_to_cw(eng: usize) -> u8 {
    match eng {
        IDX_WOOD => 0,
        IDX_BRICK => 1,
        IDX_WHEAT => 2,
        IDX_ORE => 3,
        IDX_SHEEP => 4,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Resource indexing (engine = alphabetical: BRICK, ORE, SHEEP, WHEAT, WOOD)
// ---------------------------------------------------------------------------

pub const IDX_BRICK: usize = 0;
pub const IDX_ORE: usize = 1;
pub const IDX_SHEEP: usize = 2;
pub const IDX_WHEAT: usize = 3;
pub const IDX_WOOD: usize = 4;
pub const N_RESOURCES: usize = 5;

/// Map the engine's `Resource` enum to the internal alphabetical index.
pub fn resource_to_idx(r: Resource) -> Option<usize> {
    match r {
        Resource::Brick => Some(IDX_BRICK),
        Resource::Ore => Some(IDX_ORE),
        Resource::Sheep => Some(IDX_SHEEP),
        Resource::Wheat => Some(IDX_WHEAT),
        Resource::Wood => Some(IDX_WOOD),
        Resource::Desert => None,
    }
}

// ---------------------------------------------------------------------------
// Dev card indexing (matches Python's devCards dict)
// ---------------------------------------------------------------------------

pub const DEV_KNIGHT: usize = 0;
pub const DEV_VP: usize = 1;
pub const DEV_MONOPOLY: usize = 2;
pub const DEV_ROADBUILDER: usize = 3;
pub const DEV_YEAROFPLENTY: usize = 4;
pub const N_DEV_TYPES: usize = 5;

// ---------------------------------------------------------------------------
// 13-type action space (matches obs schema)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionType {
    BuildSettlement = 0,
    BuildCity = 1,
    BuildRoad = 2,
    EndTurn = 3,
    MoveRobber = 4,
    BuyDevCard = 5,
    PlayKnight = 6,
    PlayYoP = 7,
    PlayMonopoly = 8,
    PlayRoadBuilder = 9,
    BankTrade = 10,
    Discard = 11,
    RollDice = 12,
}

impl TryFrom<u8> for ActionType {
    type Error = EngineError;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        Ok(match v {
            0 => ActionType::BuildSettlement,
            1 => ActionType::BuildCity,
            2 => ActionType::BuildRoad,
            3 => ActionType::EndTurn,
            4 => ActionType::MoveRobber,
            5 => ActionType::BuyDevCard,
            6 => ActionType::PlayKnight,
            7 => ActionType::PlayYoP,
            8 => ActionType::PlayMonopoly,
            9 => ActionType::PlayRoadBuilder,
            10 => ActionType::BankTrade,
            11 => ActionType::Discard,
            12 => ActionType::RollDice,
            _ => return Err(EngineError::InvalidActionType(v)),
        })
    }
}

// ---------------------------------------------------------------------------
// EngineError — every action failure mode the dispatcher can return.
// Per the senior's review (PR1 perf prep), we use `thiserror`-derived
// enums, not panics — Rust exceptions to Python are ~5-10µs each.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum EngineError {
    #[error("invalid action type byte: {0}")]
    InvalidActionType(u8),
    #[error("action illegal in current game phase")]
    IllegalPhase,
    #[error("insufficient resources")]
    InsufficientResources,
    #[error("vertex {0} already owned or has neighbor settlement")]
    IllegalVertex(u8),
    #[error("edge {0} already owned or not connected to player")]
    IllegalEdge(u8),
    #[error("hex {0} out of range or violates Friendly Robber")]
    IllegalRobberHex(u8),
    #[error("player has no settlement at vertex {0} to upgrade to city")]
    NoSettlementToUpgrade(u8),
    #[error("dev card stack empty")]
    DevCardStackEmpty,
    #[error("no dev card of type {0:?} in hand")]
    NoDevCardInHand(usize),
    #[error("dev card already played this turn")]
    DevCardAlreadyPlayed,
    #[error("game already terminated")]
    GameOver,
    #[error("player has fewer than 9 cards; not required to discard")]
    NoDiscardRequired,
    #[error("not in discard phase")]
    NotDiscarding,
}

// ---------------------------------------------------------------------------
// PlayerState — per-player inline state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PlayerState {
    pub resources: [u8; N_RESOURCES],
    /// Hidden hand (newly drawn + previously drawn but unplayed).
    pub dev_cards_hand: [u8; N_DEV_TYPES],
    /// Dev cards played (Knight/YoP/Mono/RoadBuilder/VP-visible).
    /// VP cards stay hidden until counted at game end, but for
    /// scoring purposes we track them in this array under DEV_VP.
    pub dev_cards_played: [u8; N_DEV_TYPES],
    /// Newly drawn dev cards this turn — not playable until next turn.
    pub new_dev_cards: [u8; N_DEV_TYPES],
    pub victory_points: u8,
    pub knights_played: u8,
    pub max_road_length: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
    pub dev_card_played_this_turn: bool,
    pub settlements_left: u8,
    pub roads_left: u8,
    pub cities_left: u8,
    /// Bitmask of port_idx the player has access to (each bit 0..8).
    pub port_mask: u16,
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            resources: [0; N_RESOURCES],
            dev_cards_hand: [0; N_DEV_TYPES],
            dev_cards_played: [0; N_DEV_TYPES],
            new_dev_cards: [0; N_DEV_TYPES],
            victory_points: 0,
            knights_played: 0,
            max_road_length: 0,
            has_longest_road: false,
            has_largest_army: false,
            dev_card_played_this_turn: false,
            settlements_left: 5,
            roads_left: 15,
            cities_left: 4,
            port_mask: 0,
        }
    }

    /// Total resource count (used for the 9-card discard threshold).
    pub fn resource_total(&self) -> u8 {
        self.resources.iter().sum()
    }

    /// "Visible" VP excludes hidden VP dev cards (per Friendly Robber).
    pub fn visible_victory_points(&self) -> u8 {
        self.victory_points
            .saturating_sub(self.dev_cards_hand[DEV_VP])
    }

    pub fn upgrade_new_dev_cards(&mut self) {
        for i in 0..N_DEV_TYPES {
            self.dev_cards_hand[i] += self.new_dev_cards[i];
            self.new_dev_cards[i] = 0;
        }
    }
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GamePhase — the env-side state machine flags rolled into one field
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamePhase {
    /// Snake-draft setup: 1→2→2→1. Each step places 1 settlement +
    /// 1 road. The current `setup_step ∈ {0,1,2,3}` field on
    /// `GameState` advances; the second settlement of each player
    /// grants starting resources.
    Setup = 0,
    /// Main turn — waiting for the dice roll.
    Roll = 1,
    /// Main turn — agent chose actions after rolling.
    Main = 2,
    /// 7 was rolled; one or both players need to discard.
    Discard = 3,
    /// Player must place the robber + optionally steal.
    Robber = 4,
    /// RoadBuilder dev card consumed; `road_builder_left` roads to place.
    RoadBuilder = 5,
    /// Game has terminated (15 VP).
    GameOver = 6,
}

// ---------------------------------------------------------------------------
// GameState — the full state
// ---------------------------------------------------------------------------

/// Maximum VP for victory (1v1 Colonist). Constant baked in per
/// CLAUDE.md.
pub const MAX_VP: u8 = 15;

/// Discard threshold (1v1 Colonist) — 9 cards. **NOT 7**.
pub const DISCARD_THRESHOLD: u8 = 9;

/// LR threshold — 5 roads.
pub const LONGEST_ROAD_THRESHOLD: u8 = 5;

/// LA threshold — 3 knights.
pub const LARGEST_ARMY_THRESHOLD: u8 = 3;

/// Friendly Robber: players with `visible_VP < 3` are immune.
pub const FRIENDLY_ROBBER_VP_THRESHOLD: u8 = 3;

#[derive(Clone)]
pub struct GameState {
    pub board: BoardStatic,
    pub dice: StackedDice,
    pub rng: EngineRng,
    /// Vertex owner: `0 = empty, 1 = p0_settlement, 2 = p1_settlement,
    /// 3 = p0_city, 4 = p1_city`.
    pub vertex_owner: [u8; 54],
    /// Edge owner: `0 = empty, 1 = p0, 2 = p1`.
    pub edge_owner: [u8; 72],
    /// Current robber hex.
    pub robber_hex: u8,
    pub players: [PlayerState; 2],
    pub current_player: u8,
    pub phase: GamePhase,
    /// `0..4` during setup phase, sentinel `255` otherwise.
    pub setup_step: u8,
    /// Dev card deck — counts remaining (KNIGHT, VP, MONOPOLY, ROADBUILDER, YEAROFPLENTY).
    pub dev_deck: [u8; N_DEV_TYPES],
    /// Last 7 roller (None if no 7 has been rolled yet).
    pub last_seven_roller: Option<u8>,
    /// Cards remaining to discard for the current discarding player.
    pub discard_cards_remaining: u8,
    /// RoadBuilder roads remaining to place.
    pub road_builder_left: u8,
    /// Most recent dice roll (used by tests + obs encoder later).
    pub last_dice_roll: u8,
    pub turn_count: u16,
    pub winner: Option<u8>,
    /// Vertex index of the most-recently-placed settlement (any
    /// player). Used by `grant_starting_resources` during setup so we
    /// grant from the actual 2nd placement, not whichever vertex
    /// happens to be highest-indexed.
    pub just_placed_vertex: Option<u8>,
    /// Broadcast event buffer — drained per env.step boundary by the
    /// Python adapter. Per the senior R0 review, events are batched
    /// per step (not emitted per-event) to avoid the GIL-hop tax.
    pub events: Vec<Event>,
}

impl GameState {
    /// Construct a fresh game from a u64 seed. Sets up the board,
    /// dice, dev deck, robber on desert, phase=Setup, current=0.
    pub fn new(seed: u64) -> Self {
        let mut rng = EngineRng::from_u64(seed);
        let board = BoardStatic::new_random(&mut rng);
        let dice = StackedDice::from_u64(seed.wrapping_add(0xCA7A));
        // Initial robber on the desert hex.
        let robber_hex = board
            .hexes
            .iter()
            .find(|h| h.resource == Resource::Desert)
            .map(|h| h.hex_idx)
            .unwrap_or(0);
        Self {
            board,
            dice,
            rng,
            vertex_owner: [0; 54],
            edge_owner: [0; 72],
            robber_hex,
            players: [PlayerState::new(); 2],
            current_player: 0,
            phase: GamePhase::Setup,
            setup_step: 0,
            // Standard dev deck (board.py:191-198).
            dev_deck: {
                let mut d = [0u8; N_DEV_TYPES];
                d[DEV_KNIGHT] = 14;
                d[DEV_VP] = 5;
                d[DEV_MONOPOLY] = 2;
                d[DEV_ROADBUILDER] = 2;
                d[DEV_YEAROFPLENTY] = 2;
                d
            },
            last_seven_roller: None,
            discard_cards_remaining: 0,
            road_builder_left: 0,
            last_dice_roll: 0,
            turn_count: 0,
            winner: None,
            just_placed_vertex: None,
            events: Vec::with_capacity(16),
        }
    }

    /// Drain the buffered events. Called by the Python adapter once
    /// per `env.step` boundary.
    pub fn drain_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    pub fn opponent(&self) -> u8 {
        1 - self.current_player
    }

    /// Apply an action vector `[type, corner, edge, tile, res1, res2]`.
    /// Returns Ok on success or an `EngineError` on illegal action.
    pub fn apply_action(&mut self, action: [u8; 6]) -> Result<(), EngineError> {
        if matches!(self.phase, GamePhase::GameOver) {
            return Err(EngineError::GameOver);
        }
        let action_type = ActionType::try_from(action[0])?;
        let corner = action[1];
        let edge = action[2];
        let tile = action[3];
        let res1 = action[4];
        let res2 = action[5];

        match action_type {
            ActionType::BuildSettlement => self.action_build_settlement(corner),
            ActionType::BuildCity => self.action_build_city(corner),
            ActionType::BuildRoad => self.action_build_road(edge),
            ActionType::EndTurn => self.action_end_turn(),
            ActionType::MoveRobber => self.action_move_robber(tile, None),
            ActionType::BuyDevCard => self.action_buy_dev_card(),
            ActionType::PlayKnight => self.action_play_knight(),
            ActionType::PlayYoP => self.action_play_yop(res1, res2),
            ActionType::PlayMonopoly => self.action_play_monopoly(res1),
            ActionType::PlayRoadBuilder => self.action_play_road_builder(),
            ActionType::BankTrade => self.action_bank_trade(res1, res2),
            ActionType::Discard => self.action_discard(res1),
            ActionType::RollDice => self.action_roll_dice(),
        }
    }

    // -----------------------------------------------------------------
    // BuildSettlement
    // -----------------------------------------------------------------

    fn action_build_settlement(&mut self, vertex_idx: u8) -> Result<(), EngineError> {
        if vertex_idx as usize >= 54 {
            return Err(EngineError::IllegalVertex(vertex_idx));
        }
        let is_setup = matches!(self.phase, GamePhase::Setup);
        if !is_setup && !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }

        let p = self.current_player as usize;
        // Resource check (skip during setup — free settlement).
        if !is_setup {
            let r = &self.players[p].resources;
            if r[IDX_BRICK] == 0 || r[IDX_WOOD] == 0 || r[IDX_SHEEP] == 0 || r[IDX_WHEAT] == 0 {
                return Err(EngineError::InsufficientResources);
            }
        }
        if self.players[p].settlements_left == 0 {
            return Err(EngineError::IllegalVertex(vertex_idx));
        }

        // Vertex must be empty.
        if self.vertex_owner[vertex_idx as usize] != 0 {
            return Err(EngineError::IllegalVertex(vertex_idx));
        }
        // Distance rule: no neighbor settlement/city.
        for e in self.board.edges() {
            if e.v1_idx == vertex_idx && self.vertex_owner[e.v2_idx as usize] != 0 {
                return Err(EngineError::IllegalVertex(vertex_idx));
            }
            if e.v2_idx == vertex_idx && self.vertex_owner[e.v1_idx as usize] != 0 {
                return Err(EngineError::IllegalVertex(vertex_idx));
            }
        }
        // Setup placement legality: any vertex that passes distance rule is fine.
        // Main-phase placement legality: must be adjacent to a player's road.
        if !is_setup {
            let mut adj_own_road = false;
            for e in self.board.edges() {
                let touches = e.v1_idx == vertex_idx || e.v2_idx == vertex_idx;
                if touches && self.edge_owner[e.edge_idx as usize] == (p as u8 + 1) {
                    adj_own_road = true;
                    break;
                }
            }
            if !adj_own_road {
                return Err(EngineError::IllegalVertex(vertex_idx));
            }
        }

        // Apply.
        self.vertex_owner[vertex_idx as usize] = p as u8 + 1; // 1 or 2
        self.just_placed_vertex = Some(vertex_idx);
        self.players[p].settlements_left -= 1;
        self.players[p].victory_points += 1;
        if !is_setup {
            self.players[p].resources[IDX_BRICK] -= 1;
            self.players[p].resources[IDX_WOOD] -= 1;
            self.players[p].resources[IDX_SHEEP] -= 1;
            self.players[p].resources[IDX_WHEAT] -= 1;
            // Emit the resource debit. Python emits one
            // RESOURCE_CHANGE per build (player.py:133).
            let mut delta = [0i8; N_RESOURCES];
            delta[IDX_BRICK] = -1;
            delta[IDX_WOOD] = -1;
            delta[IDX_SHEEP] = -1;
            delta[IDX_WHEAT] = -1;
            self.events.push(Event::ResourceChange {
                player: p as u8,
                delta_alpha: delta,
                source: ResourceChangeSource::BuildSettlement,
            });
        }
        // Emit structural BUILD event (Phase 0.5 contract).
        self.events.push(Event::Build {
            player: p as u8,
            kind: BuildKind::Settlement,
            location: vertex_idx,
        });
        // Port acquisition.
        for port in &self.board.ports {
            if port.vertex_idx_pair[0] == vertex_idx || port.vertex_idx_pair[1] == vertex_idx {
                self.players[p].port_mask |= 1u16 << port.port_idx;
            }
        }
        // Victory check (main-phase only — setup can't reach 15 VP).
        if !is_setup {
            self.check_terminal();
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // BuildCity
    // -----------------------------------------------------------------

    fn action_build_city(&mut self, vertex_idx: u8) -> Result<(), EngineError> {
        if vertex_idx as usize >= 54 {
            return Err(EngineError::IllegalVertex(vertex_idx));
        }
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        let owner_marker = p as u8 + 1; // 1 or 2 for settlement
        if self.vertex_owner[vertex_idx as usize] != owner_marker {
            return Err(EngineError::NoSettlementToUpgrade(vertex_idx));
        }
        let r = &self.players[p].resources;
        if r[IDX_WHEAT] < 2 || r[IDX_ORE] < 3 {
            return Err(EngineError::InsufficientResources);
        }
        if self.players[p].cities_left == 0 {
            return Err(EngineError::IllegalVertex(vertex_idx));
        }
        // Apply: settlement → city. City marker = p + 3 (i.e., 3 or 4).
        self.vertex_owner[vertex_idx as usize] = p as u8 + 3;
        self.players[p].cities_left -= 1;
        self.players[p].settlements_left += 1;
        self.players[p].resources[IDX_WHEAT] -= 2;
        self.players[p].resources[IDX_ORE] -= 3;
        self.players[p].victory_points += 1;
        let mut delta = [0i8; N_RESOURCES];
        delta[IDX_WHEAT] = -2;
        delta[IDX_ORE] = -3;
        self.events.push(Event::ResourceChange {
            player: p as u8,
            delta_alpha: delta,
            source: ResourceChangeSource::BuildCity,
        });
        self.events.push(Event::Build {
            player: p as u8,
            kind: BuildKind::City,
            location: vertex_idx,
        });
        self.check_terminal();
        Ok(())
    }

    // -----------------------------------------------------------------
    // BuildRoad
    // -----------------------------------------------------------------

    fn action_build_road(&mut self, edge_idx: u8) -> Result<(), EngineError> {
        if edge_idx as usize >= 72 {
            return Err(EngineError::IllegalEdge(edge_idx));
        }
        let is_setup = matches!(self.phase, GamePhase::Setup);
        let is_road_builder = matches!(self.phase, GamePhase::RoadBuilder);
        let is_main = matches!(self.phase, GamePhase::Main);
        if !is_setup && !is_main && !is_road_builder {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        let is_free = is_setup || is_road_builder;
        if !is_free {
            let r = &self.players[p].resources;
            if r[IDX_BRICK] == 0 || r[IDX_WOOD] == 0 {
                return Err(EngineError::InsufficientResources);
            }
        }
        if self.players[p].roads_left == 0 {
            return Err(EngineError::IllegalEdge(edge_idx));
        }
        if self.edge_owner[edge_idx as usize] != 0 {
            return Err(EngineError::IllegalEdge(edge_idx));
        }
        // Connectivity: at least one endpoint vertex is owned by this
        // player OR is adjacent to a road this player owns.
        let edge = self.board.edges()[edge_idx as usize];
        let owner_marker_s = p as u8 + 1; // settlement
        let owner_marker_c = p as u8 + 3; // city
        let v1_owned = self.vertex_owner[edge.v1_idx as usize] == owner_marker_s
            || self.vertex_owner[edge.v1_idx as usize] == owner_marker_c;
        let v2_owned = self.vertex_owner[edge.v2_idx as usize] == owner_marker_s
            || self.vertex_owner[edge.v2_idx as usize] == owner_marker_c;
        let mut connected = v1_owned || v2_owned;
        if !connected {
            // Connected via an adjacent road if the adjacent vertex
            // isn't blocked by an opponent's piece.
            for other_edge in self.board.edges() {
                if other_edge.edge_idx == edge_idx {
                    continue;
                }
                if self.edge_owner[other_edge.edge_idx as usize] != p as u8 + 1 {
                    continue;
                }
                let shares_v1 =
                    other_edge.v1_idx == edge.v1_idx || other_edge.v2_idx == edge.v1_idx;
                let shares_v2 =
                    other_edge.v1_idx == edge.v2_idx || other_edge.v2_idx == edge.v2_idx;
                // The shared vertex must not be owned by the opponent.
                let opp_settle_v1 = self.vertex_owner[edge.v1_idx as usize] == self.opponent() + 1
                    || self.vertex_owner[edge.v1_idx as usize] == self.opponent() + 3;
                let opp_settle_v2 = self.vertex_owner[edge.v2_idx as usize] == self.opponent() + 1
                    || self.vertex_owner[edge.v2_idx as usize] == self.opponent() + 3;
                if (shares_v1 && !opp_settle_v1) || (shares_v2 && !opp_settle_v2) {
                    connected = true;
                    break;
                }
            }
        }
        if !connected {
            return Err(EngineError::IllegalEdge(edge_idx));
        }

        self.edge_owner[edge_idx as usize] = p as u8 + 1;
        self.players[p].roads_left -= 1;
        if !is_free {
            self.players[p].resources[IDX_BRICK] -= 1;
            self.players[p].resources[IDX_WOOD] -= 1;
            let mut delta = [0i8; N_RESOURCES];
            delta[IDX_BRICK] = -1;
            delta[IDX_WOOD] = -1;
            self.events.push(Event::ResourceChange {
                player: p as u8,
                delta_alpha: delta,
                source: ResourceChangeSource::BuildRoad,
            });
        }
        self.events.push(Event::Build {
            player: p as u8,
            kind: BuildKind::Road,
            location: edge_idx,
        });
        if is_road_builder {
            self.road_builder_left = self.road_builder_left.saturating_sub(1);
            if self.road_builder_left == 0 {
                self.phase = GamePhase::Main;
            }
        }
        // Recompute LR + transfer.
        self.recompute_longest_road();
        self.check_terminal();
        Ok(())
    }

    // -----------------------------------------------------------------
    // EndTurn
    // -----------------------------------------------------------------

    fn action_end_turn(&mut self) -> Result<(), EngineError> {
        // Setup: progress to next setup step.
        if matches!(self.phase, GamePhase::Setup) {
            // Each setup step expects one settlement + one road from
            // the current player. The caller is responsible for
            // calling BuildSettlement + BuildRoad before EndTurn.
            self.setup_step += 1;
            // Snake draft: 0:P0, 1:P1, 2:P1, 3:P0.
            match self.setup_step {
                1 => self.current_player = 1,
                2 => {
                    // After P1's first placement, grant starting
                    // resources to P1 (their 2nd settle is next, but
                    // we grant from the LAST placed settlement per
                    // the Python convention).
                    // Actually: per Python convention, starting
                    // resources are granted from the 2ND settlement.
                    // So we wait until setup_step transitions out of
                    // P1's 2nd placement (handled in step 3).
                    self.current_player = 1; // P1 places again
                }
                3 => {
                    // P1 just placed their 2nd settle+road; grant
                    // resources from their 2nd settle.
                    self.grant_starting_resources(1);
                    self.current_player = 0;
                }
                4 => {
                    // P0 just placed their 2nd settle+road; grant
                    // resources from their 2nd settle. Setup done.
                    self.grant_starting_resources(0);
                    self.phase = GamePhase::Roll;
                    self.setup_step = 255;
                    self.current_player = 0;
                    self.turn_count = 1;
                    // Phase 0.5 contract — fire once, before any
                    // main-phase action.
                    self.events.push(Event::SetupComplete);
                }
                _ => {}
            }
            return Ok(());
        }
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        // Upgrade newly-drawn dev cards into the hand.
        let p = self.current_player as usize;
        self.players[p].upgrade_new_dev_cards();
        self.players[p].dev_card_played_this_turn = false;
        // Largest army recompute (in case knights were played this turn).
        self.recompute_largest_army();
        self.check_terminal();
        if matches!(self.phase, GamePhase::GameOver) {
            return Ok(());
        }
        // Pass turn.
        self.current_player = self.opponent();
        self.turn_count += 1;
        self.phase = GamePhase::Roll;
        Ok(())
    }

    fn grant_starting_resources(&mut self, player: u8) {
        // Grant starting resources from the player's 2nd setup
        // settlement — tracked by ``just_placed_vertex`` which is
        // populated by BuildSettlement and consumed here at the
        // setup-step boundary. Matches Python's
        // ``_grant_setup_resources`` which reads
        // ``p.buildGraph['SETTLEMENTS'][-1]``.
        let owner_marker = player + 1;
        let last_settle_v = self.just_placed_vertex;
        // Sanity: the just-placed vertex must belong to this player.
        if let Some(v_idx) = last_settle_v {
            if self.vertex_owner[v_idx as usize] != owner_marker {
                return;
            }
        }
        if let Some(v_idx) = last_settle_v {
            // Find adjacent hexes via the static vertex table.
            for v in self.board.vertices() {
                if v.vertex_idx == v_idx {
                    for &h_idx in &v.adjacent_hex_indices[..v.adjacent_count as usize] {
                        if h_idx == u8::MAX {
                            continue;
                        }
                        let hex = &self.board.hexes[h_idx as usize];
                        if let Some(idx) = resource_to_idx(hex.resource) {
                            self.players[player as usize].resources[idx] =
                                self.players[player as usize].resources[idx].saturating_add(1);
                        }
                    }
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // MoveRobber (called from RollDice when 7 fires, or from PlayKnight)
    // -----------------------------------------------------------------

    fn action_move_robber(
        &mut self,
        hex_idx: u8,
        _victim_hint: Option<u8>,
    ) -> Result<(), EngineError> {
        if hex_idx as usize >= 19 {
            return Err(EngineError::IllegalRobberHex(hex_idx));
        }
        if !matches!(self.phase, GamePhase::Robber) {
            return Err(EngineError::IllegalPhase);
        }
        if hex_idx == self.robber_hex {
            return Err(EngineError::IllegalRobberHex(hex_idx));
        }
        // Friendly Robber: the destination hex must NOT be adjacent
        // to a player with visible_VP < FRIENDLY_ROBBER_VP_THRESHOLD.
        for v in self.board.vertices() {
            if v.adjacent_hex_indices[..v.adjacent_count as usize].contains(&hex_idx) {
                let owner = self.vertex_owner[v.vertex_idx as usize];
                if owner == 0 {
                    continue;
                }
                let owner_player = if owner == 1 || owner == 3 { 0 } else { 1 };
                if self.players[owner_player as usize].visible_victory_points()
                    < FRIENDLY_ROBBER_VP_THRESHOLD
                {
                    return Err(EngineError::IllegalRobberHex(hex_idx));
                }
            }
        }
        // Move the robber.
        self.robber_hex = hex_idx;
        self.events.push(Event::MoveRobber {
            player: self.current_player,
            hex_idx,
        });
        // Steal one random resource from a player adjacent to the new hex
        // (other than the current player). Pick a random VICTIM among
        // the ones with resources > 0.
        let me = self.current_player;
        let mut victim: Option<u8> = None;
        for v in self.board.vertices() {
            if v.adjacent_hex_indices[..v.adjacent_count as usize].contains(&hex_idx) {
                let owner = self.vertex_owner[v.vertex_idx as usize];
                if owner == 0 {
                    continue;
                }
                let owner_player = if owner == 1 || owner == 3 { 0u8 } else { 1u8 };
                if owner_player != me && self.players[owner_player as usize].resource_total() > 0 {
                    victim = Some(owner_player);
                    break;
                }
            }
        }
        if let Some(v) = victim {
            // Pick one random resource type weighted by count.
            let total = self.players[v as usize].resource_total() as u32;
            let pick = self.rng.gen_range_u32(total);
            let mut acc = 0u32;
            for r_idx in 0..N_RESOURCES {
                acc += self.players[v as usize].resources[r_idx] as u32;
                if pick < acc {
                    self.players[v as usize].resources[r_idx] -= 1;
                    self.players[me as usize].resources[r_idx] += 1;
                    // Two RESOURCE_CHANGE events (victim -1, robber +1)
                    // then a STEAL marker — matches Python ordering
                    // (player.py:253-265).
                    let mut victim_delta = [0i8; N_RESOURCES];
                    victim_delta[r_idx] = -1;
                    self.events.push(Event::ResourceChange {
                        player: v,
                        delta_alpha: victim_delta,
                        source: ResourceChangeSource::Steal,
                    });
                    let mut robber_delta = [0i8; N_RESOURCES];
                    robber_delta[r_idx] = 1;
                    self.events.push(Event::ResourceChange {
                        player: me,
                        delta_alpha: robber_delta,
                        source: ResourceChangeSource::Steal,
                    });
                    self.events.push(Event::Steal {
                        robber: me,
                        victim: v,
                        resource_cw: engine_to_cw(r_idx),
                    });
                    break;
                }
            }
        }
        // After moving the robber, return to Main (or Roll if from setup
        // — N/A, robber moves only in Main).
        self.phase = GamePhase::Main;
        Ok(())
    }

    // -----------------------------------------------------------------
    // BuyDevCard
    // -----------------------------------------------------------------

    fn action_buy_dev_card(&mut self) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        let r = &self.players[p].resources;
        if r[IDX_WHEAT] == 0 || r[IDX_SHEEP] == 0 || r[IDX_ORE] == 0 {
            return Err(EngineError::InsufficientResources);
        }
        let total_remaining: u32 = self.dev_deck.iter().map(|&v| v as u32).sum();
        if total_remaining == 0 {
            return Err(EngineError::DevCardStackEmpty);
        }
        self.players[p].resources[IDX_WHEAT] -= 1;
        self.players[p].resources[IDX_SHEEP] -= 1;
        self.players[p].resources[IDX_ORE] -= 1;
        // Draw one uniformly-weighted card from the deck.
        let pick = self.rng.gen_range_u32(total_remaining);
        let mut acc = 0u32;
        for d_idx in 0..N_DEV_TYPES {
            acc += self.dev_deck[d_idx] as u32;
            if pick < acc {
                self.dev_deck[d_idx] -= 1;
                self.players[p].new_dev_cards[d_idx] += 1;
                if d_idx == DEV_VP {
                    // VP cards count toward total VP but stay hidden.
                    self.players[p].dev_cards_hand[DEV_VP] += 1;
                    self.players[p].new_dev_cards[DEV_VP] -= 1;
                    self.players[p].victory_points += 1;
                    self.check_terminal();
                }
                break;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // PlayKnight
    // -----------------------------------------------------------------

    fn action_play_knight(&mut self) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        if self.players[p].dev_cards_hand[DEV_KNIGHT] == 0 {
            return Err(EngineError::NoDevCardInHand(DEV_KNIGHT));
        }
        if self.players[p].dev_card_played_this_turn {
            return Err(EngineError::DevCardAlreadyPlayed);
        }
        self.players[p].dev_cards_hand[DEV_KNIGHT] -= 1;
        self.players[p].dev_cards_played[DEV_KNIGHT] += 1;
        self.players[p].knights_played += 1;
        self.players[p].dev_card_played_this_turn = true;
        // Phase shifts to Robber so the next action is MoveRobber.
        self.phase = GamePhase::Robber;
        self.recompute_largest_army();
        Ok(())
    }

    // -----------------------------------------------------------------
    // PlayYoP (Year of Plenty) — take 2 resources of any types from bank
    // -----------------------------------------------------------------

    fn action_play_yop(&mut self, res1: u8, res2: u8) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        if self.players[p].dev_cards_hand[DEV_YEAROFPLENTY] == 0 {
            return Err(EngineError::NoDevCardInHand(DEV_YEAROFPLENTY));
        }
        if self.players[p].dev_card_played_this_turn {
            return Err(EngineError::DevCardAlreadyPlayed);
        }
        if res1 as usize >= N_RESOURCES || res2 as usize >= N_RESOURCES {
            return Err(EngineError::InsufficientResources);
        }
        self.players[p].dev_cards_hand[DEV_YEAROFPLENTY] -= 1;
        self.players[p].dev_cards_played[DEV_YEAROFPLENTY] += 1;
        self.players[p].dev_card_played_this_turn = true;
        // Note: res1/res2 are in Charlesworth order at the action-vector
        // level. The Python code uses RESOURCES_CW[res1_idx] which
        // returns WOOD/BRICK/WHEAT/ORE/SHEEP. Convert to engine alpha
        // order:
        let mapped = |cw_idx: u8| -> usize {
            // Charlesworth: WOOD=0, BRICK=1, WHEAT=2, ORE=3, SHEEP=4.
            // Engine alpha: BRICK=0, ORE=1, SHEEP=2, WHEAT=3, WOOD=4.
            match cw_idx {
                0 => IDX_WOOD,
                1 => IDX_BRICK,
                2 => IDX_WHEAT,
                3 => IDX_ORE,
                4 => IDX_SHEEP,
                _ => IDX_BRICK,
            }
        };
        self.players[p].resources[mapped(res1)] += 1;
        self.players[p].resources[mapped(res2)] += 1;
        self.events.push(Event::YearOfPlenty {
            player: p as u8,
            resources: [res1, res2],
        });
        Ok(())
    }

    // -----------------------------------------------------------------
    // PlayMonopoly — take all of one resource from all other players
    // -----------------------------------------------------------------

    fn action_play_monopoly(&mut self, res1: u8) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        if self.players[p].dev_cards_hand[DEV_MONOPOLY] == 0 {
            return Err(EngineError::NoDevCardInHand(DEV_MONOPOLY));
        }
        if self.players[p].dev_card_played_this_turn {
            return Err(EngineError::DevCardAlreadyPlayed);
        }
        if res1 as usize >= N_RESOURCES {
            return Err(EngineError::InsufficientResources);
        }
        self.players[p].dev_cards_hand[DEV_MONOPOLY] -= 1;
        self.players[p].dev_cards_played[DEV_MONOPOLY] += 1;
        self.players[p].dev_card_played_this_turn = true;
        let mapped = match res1 {
            0 => IDX_WOOD,
            1 => IDX_BRICK,
            2 => IDX_WHEAT,
            3 => IDX_ORE,
            4 => IDX_SHEEP,
            _ => return Err(EngineError::InsufficientResources),
        };
        let opp = self.opponent() as usize;
        let stolen = self.players[opp].resources[mapped];
        self.players[opp].resources[mapped] = 0;
        self.players[p].resources[mapped] += stolen;
        // Python emits per-victim RESOURCE_CHANGE FIRST then the
        // structural MONOPOLY event (catan_env.py:481-499). Mirror
        // that ordering exactly so the replay recorder's classifier
        // (which expects RESOURCE_CHANGE before MONOPOLY) sees the
        // canonical stream.
        if stolen > 0 {
            let mut victim_delta = [0i8; N_RESOURCES];
            victim_delta[mapped] = -(stolen as i8);
            self.events.push(Event::ResourceChange {
                player: opp as u8,
                delta_alpha: victim_delta,
                source: ResourceChangeSource::Monopoly,
            });
            let mut taker_delta = [0i8; N_RESOURCES];
            taker_delta[mapped] = stolen as i8;
            self.events.push(Event::ResourceChange {
                player: p as u8,
                delta_alpha: taker_delta,
                source: ResourceChangeSource::Monopoly,
            });
        }
        self.events.push(Event::Monopoly {
            player: p as u8,
            resource_cw: res1,
            count: stolen,
        });
        Ok(())
    }

    // -----------------------------------------------------------------
    // PlayRoadBuilder — place 2 free roads
    // -----------------------------------------------------------------

    fn action_play_road_builder(&mut self) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let p = self.current_player as usize;
        if self.players[p].dev_cards_hand[DEV_ROADBUILDER] == 0 {
            return Err(EngineError::NoDevCardInHand(DEV_ROADBUILDER));
        }
        if self.players[p].dev_card_played_this_turn {
            return Err(EngineError::DevCardAlreadyPlayed);
        }
        self.players[p].dev_cards_hand[DEV_ROADBUILDER] -= 1;
        self.players[p].dev_cards_played[DEV_ROADBUILDER] += 1;
        self.players[p].dev_card_played_this_turn = true;
        self.road_builder_left = 2;
        self.phase = GamePhase::RoadBuilder;
        Ok(())
    }

    // -----------------------------------------------------------------
    // BankTrade — 4:1 default, 3:1 generic port, 2:1 specific port
    // -----------------------------------------------------------------

    fn action_bank_trade(&mut self, give_cw: u8, recv_cw: u8) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Main) {
            return Err(EngineError::IllegalPhase);
        }
        let mapped = |cw_idx: u8| -> Option<usize> {
            Some(match cw_idx {
                0 => IDX_WOOD,
                1 => IDX_BRICK,
                2 => IDX_WHEAT,
                3 => IDX_ORE,
                4 => IDX_SHEEP,
                _ => return None,
            })
        };
        let give = mapped(give_cw).ok_or(EngineError::InsufficientResources)?;
        let recv = mapped(recv_cw).ok_or(EngineError::InsufficientResources)?;
        if give == recv {
            return Err(EngineError::InsufficientResources);
        }
        let p = self.current_player as usize;
        // Determine trade ratio.
        // Default 4. 3:1 generic port lowers to 3. 2:1 specific port
        // for the GIVE resource lowers to 2.
        let mut ratio = 4u8;
        let mut has_3_to_1 = false;
        let mut has_2_to_1_specific_for_give = false;
        for port in &self.board.ports {
            if (self.players[p].port_mask & (1u16 << port.port_idx)) == 0 {
                continue;
            }
            if port.ratio == 3 {
                has_3_to_1 = true;
            }
            if port.ratio == 2 {
                if let Some(port_res) = port.resource {
                    if let Some(port_idx) = resource_to_idx(port_res) {
                        if port_idx == give {
                            has_2_to_1_specific_for_give = true;
                        }
                    }
                }
            }
        }
        if has_2_to_1_specific_for_give {
            ratio = 2;
        } else if has_3_to_1 {
            ratio = 3;
        }
        if self.players[p].resources[give] < ratio {
            return Err(EngineError::InsufficientResources);
        }
        self.players[p].resources[give] -= ratio;
        self.players[p].resources[recv] += 1;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Discard — discard one resource (called repeatedly until threshold met)
    // -----------------------------------------------------------------

    fn action_discard(&mut self, res_cw: u8) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Discard) {
            return Err(EngineError::NotDiscarding);
        }
        let mapped = match res_cw {
            0 => IDX_WOOD,
            1 => IDX_BRICK,
            2 => IDX_WHEAT,
            3 => IDX_ORE,
            4 => IDX_SHEEP,
            _ => return Err(EngineError::InsufficientResources),
        };
        let p = self.current_player as usize;
        if self.players[p].resources[mapped] == 0 {
            return Err(EngineError::InsufficientResources);
        }
        self.players[p].resources[mapped] -= 1;
        self.events.push(Event::Discard {
            player: p as u8,
            resources: vec![res_cw],
        });
        self.discard_cards_remaining = self.discard_cards_remaining.saturating_sub(1);
        if self.discard_cards_remaining == 0 {
            self.phase = GamePhase::Robber;
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // RollDice — main roll handler
    // -----------------------------------------------------------------

    fn action_roll_dice(&mut self) -> Result<(), EngineError> {
        if !matches!(self.phase, GamePhase::Roll) {
            return Err(EngineError::IllegalPhase);
        }
        let me = self.current_player;
        let value = self.dice.roll(me, self.last_seven_roller);
        self.last_dice_roll = value;
        self.events.push(Event::DiceRoll { player: me, value });
        if value == 7 {
            self.last_seven_roller = Some(me);
            // Discard threshold: if current player has > 9 cards, they
            // must discard floor(N/2). Note: per CLAUDE.md, discard
            // threshold is 9 cards, not 7.
            let cur_total = self.players[me as usize].resource_total();
            if cur_total > DISCARD_THRESHOLD {
                self.discard_cards_remaining = cur_total / 2;
                self.phase = GamePhase::Discard;
                return Ok(());
            }
            // No discard required; jump straight to robber placement.
            self.phase = GamePhase::Robber;
        } else {
            // Standard production roll.
            self.distribute_resources(value);
            self.phase = GamePhase::Main;
        }
        Ok(())
    }

    fn distribute_resources(&mut self, roll: u8) {
        // For each hex matching the roll AND not blocked by the robber,
        // grant resources to each adjacent owner. Emit one
        // RESOURCE_CHANGE per (player, hex) tuple — the Python engine
        // emits per resource grant.
        for hex in &self.board.hexes {
            if hex.number_token != Some(roll) {
                continue;
            }
            if hex.hex_idx == self.robber_hex {
                continue;
            }
            let res_idx = match resource_to_idx(hex.resource) {
                Some(i) => i,
                None => continue,
            };
            for v in self.board.vertices() {
                if !v.adjacent_hex_indices[..v.adjacent_count as usize].contains(&hex.hex_idx) {
                    continue;
                }
                let owner = self.vertex_owner[v.vertex_idx as usize];
                if owner == 0 {
                    continue;
                }
                // 1, 2 = settle; 3, 4 = city
                let amount = if owner <= 2 { 1 } else { 2 };
                let owner_player = if owner == 1 || owner == 3 { 0 } else { 1 };
                self.players[owner_player as usize].resources[res_idx] += amount;
                let mut delta = [0i8; N_RESOURCES];
                delta[res_idx] = amount as i8;
                self.events.push(Event::ResourceChange {
                    player: owner_player,
                    delta_alpha: delta,
                    source: ResourceChangeSource::Roll,
                });
            }
        }
    }

    // -----------------------------------------------------------------
    // Longest Road BFS
    // -----------------------------------------------------------------

    fn recompute_longest_road(&mut self) {
        let p = self.current_player as usize;
        let owner_marker = p as u8 + 1;
        // Build vertex -> edge adjacency for this player's roads.
        let mut adj: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
        for e in self.board.edges() {
            if self.edge_owner[e.edge_idx as usize] == owner_marker {
                adj[e.v1_idx as usize].push(e.edge_idx);
                adj[e.v2_idx as usize].push(e.edge_idx);
            }
        }
        // For each vertex as a DFS start, find the longest path.
        let mut max_len = 0u8;
        for start_v in 0..54u8 {
            if adj[start_v as usize].is_empty() {
                continue;
            }
            // Skip if start vertex is blocked by opponent.
            let opp_marker_s = self.opponent() + 1;
            let opp_marker_c = self.opponent() + 3;
            if self.vertex_owner[start_v as usize] == opp_marker_s
                || self.vertex_owner[start_v as usize] == opp_marker_c
            {
                continue;
            }
            let mut visited = [false; 72];
            let len = self.dfs_road(start_v, &mut visited, &adj);
            if len > max_len {
                max_len = len;
            }
        }
        self.players[p].max_road_length = max_len;
        // Transfer Longest Road if threshold reached and longer than
        // current holder.
        if max_len >= LONGEST_ROAD_THRESHOLD {
            let cur_holder = if self.players[0].has_longest_road {
                Some(0)
            } else if self.players[1].has_longest_road {
                Some(1)
            } else {
                None
            };
            let other = self.opponent() as usize;
            let other_len = self.players[other].max_road_length;
            match cur_holder {
                None => {
                    self.players[p].has_longest_road = true;
                    self.players[p].victory_points += 2;
                    self.events.push(Event::LongestRoadChange {
                        prev_owner: None,
                        new_owner: Some(p as u8),
                        length: max_len,
                    });
                }
                Some(h) if h != p as u8 && max_len > other_len => {
                    self.players[h as usize].has_longest_road = false;
                    self.players[h as usize].victory_points =
                        self.players[h as usize].victory_points.saturating_sub(2);
                    self.players[p].has_longest_road = true;
                    self.players[p].victory_points += 2;
                    self.events.push(Event::LongestRoadChange {
                        prev_owner: Some(h),
                        new_owner: Some(p as u8),
                        length: max_len,
                    });
                }
                _ => {}
            }
        }
    }

    fn dfs_road(&self, v: u8, visited: &mut [bool; 72], adj: &[Vec<u8>; 54]) -> u8 {
        let mut max_len = 0u8;
        for &e_idx in &adj[v as usize] {
            if visited[e_idx as usize] {
                continue;
            }
            visited[e_idx as usize] = true;
            let edge = self.board.edges()[e_idx as usize];
            let other_v = if edge.v1_idx == v {
                edge.v2_idx
            } else {
                edge.v1_idx
            };
            // Don't traverse through opponent's settlement (block).
            let opp_marker_s = self.opponent() + 1;
            let opp_marker_c = self.opponent() + 3;
            let blocked = self.vertex_owner[other_v as usize] == opp_marker_s
                || self.vertex_owner[other_v as usize] == opp_marker_c;
            let len = 1 + if blocked {
                0
            } else {
                self.dfs_road(other_v, visited, adj)
            };
            if len > max_len {
                max_len = len;
            }
            visited[e_idx as usize] = false;
        }
        max_len
    }

    // -----------------------------------------------------------------
    // Largest Army recompute
    // -----------------------------------------------------------------

    fn recompute_largest_army(&mut self) {
        let p0_knights = self.players[0].knights_played;
        let p1_knights = self.players[1].knights_played;
        let cur_holder = if self.players[0].has_largest_army {
            Some(0)
        } else if self.players[1].has_largest_army {
            Some(1)
        } else {
            None
        };
        let leader = if p0_knights > p1_knights {
            Some(0u8)
        } else if p1_knights > p0_knights {
            Some(1u8)
        } else {
            None
        };
        match (leader, cur_holder) {
            (Some(l), None) => {
                let l_k = self.players[l as usize].knights_played;
                if l_k >= LARGEST_ARMY_THRESHOLD {
                    self.players[l as usize].has_largest_army = true;
                    self.players[l as usize].victory_points += 2;
                    self.events.push(Event::LargestArmyChange {
                        prev_owner: None,
                        new_owner: Some(l),
                        knights: l_k,
                    });
                }
            }
            (Some(l), Some(h)) if l != h => {
                let l_k = self.players[l as usize].knights_played;
                if l_k >= LARGEST_ARMY_THRESHOLD {
                    self.players[h as usize].has_largest_army = false;
                    self.players[h as usize].victory_points =
                        self.players[h as usize].victory_points.saturating_sub(2);
                    self.players[l as usize].has_largest_army = true;
                    self.players[l as usize].victory_points += 2;
                    self.events.push(Event::LargestArmyChange {
                        prev_owner: Some(h),
                        new_owner: Some(l),
                        knights: l_k,
                    });
                }
            }
            _ => {}
        }
    }

    fn check_terminal(&mut self) {
        for p in 0..2 {
            if self.players[p].victory_points >= MAX_VP {
                self.winner = Some(p as u8);
                self.phase = GamePhase::GameOver;
                self.events.push(Event::GameEnd {
                    winner: p as u8,
                    vp_p0: self.players[0].victory_points,
                    vp_p1: self.players[1].victory_points,
                });
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal cargo tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_initial_setup_phase() {
        let g = GameState::new(42);
        assert_eq!(g.phase, GamePhase::Setup);
        assert_eq!(g.setup_step, 0);
        assert_eq!(g.current_player, 0);
        // Robber starts on desert.
        let desert_hex = g
            .board
            .hexes
            .iter()
            .find(|h| h.resource == Resource::Desert)
            .unwrap();
        assert_eq!(g.robber_hex, desert_hex.hex_idx);
    }

    #[test]
    fn invalid_action_byte_rejected() {
        let mut g = GameState::new(1);
        let r = g.apply_action([99, 0, 0, 0, 0, 0]);
        assert!(matches!(r, Err(EngineError::InvalidActionType(99))));
    }

    #[test]
    fn cannot_build_road_in_setup_without_settlement() {
        let mut g = GameState::new(1);
        // First action in setup must be BuildSettlement.
        let r = g.apply_action([ActionType::BuildRoad as u8, 0, 0, 0, 0, 0]);
        // The road should fail because it's not connected to a player's
        // settlement.
        assert!(r.is_err());
    }

    #[test]
    fn dev_deck_total_is_25() {
        let g = GameState::new(1);
        let total: u8 = g.dev_deck.iter().sum();
        assert_eq!(total, 25);
        assert_eq!(g.dev_deck[DEV_KNIGHT], 14);
        assert_eq!(g.dev_deck[DEV_VP], 5);
        assert_eq!(g.dev_deck[DEV_MONOPOLY], 2);
        assert_eq!(g.dev_deck[DEV_ROADBUILDER], 2);
        assert_eq!(g.dev_deck[DEV_YEAROFPLENTY], 2);
    }

    #[test]
    fn discard_threshold_is_9_not_7() {
        // Pin the 1v1 Colonist rule.
        assert_eq!(DISCARD_THRESHOLD, 9);
    }

    #[test]
    fn friendly_robber_threshold_is_lt_3() {
        // visible_VP < 3 → immune. VP=2 is immune; VP=3 is fair game.
        assert_eq!(FRIENDLY_ROBBER_VP_THRESHOLD, 3);
    }

    #[test]
    fn lr_threshold_is_5_la_is_3() {
        assert_eq!(LONGEST_ROAD_THRESHOLD, 5);
        assert_eq!(LARGEST_ARMY_THRESHOLD, 3);
    }

    #[test]
    fn max_vp_is_15() {
        assert_eq!(MAX_VP, 15);
    }
}
