//! Broadcast events — per-action records buffered in `GameState`
//! and drained once per env.step boundary (NOT per emit, to avoid
//! the GIL-hop cost the senior R0 review flagged at ~1µs per emit).
//!
//! The Python ``GameBroadcast`` emits one event per state change
//! (DICE_ROLL, DISCARD, BUILD, RESOURCE_CHANGE, etc.). The Rust
//! engine accumulates the same events as `Vec<Event>` then exposes
//! `drain_events()` which is called from the Python adapter at known
//! step boundaries.
//!
//! Mirrors `engine/broadcast.py:BroadcastEventType` exactly so the
//! replay recorder's event classifier doesn't need a separate
//! dispatch table.

use crate::state::N_RESOURCES;

/// Event variants emitted by the engine. Names match the Python
/// `BroadcastEventType` enum verbatim — the recorder's
/// `classify_step_events` (recorder.py:196) routes via the type
/// string; we serialize the variant name in `to_type_str`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    DiceRoll {
        player: u8,
        value: u8,
    },
    Discard {
        player: u8,
        /// One Charlesworth-order index per discarded card. Matches
        /// the Python event's ``resources: list[str]`` field shape.
        resources: Vec<u8>,
    },
    YearOfPlenty {
        player: u8,
        resources: [u8; 2], // CW indices
    },
    /// Per-event resource delta. The Python `RESOURCE_CHANGE` carries
    /// `delta: dict[str, int]` keyed by resource string; we carry a
    /// fixed `[i8; 5]` in engine-alpha order.
    ResourceChange {
        player: u8,
        delta_alpha: [i8; N_RESOURCES],
        source: ResourceChangeSource,
    },
    Monopoly {
        player: u8,
        resource_cw: u8,
        count: u8,
    },
    MoveRobber {
        player: u8,
        hex_idx: u8,
    },
    Steal {
        robber: u8,
        victim: u8,
        /// Charlesworth-order index of the stolen resource. Engine
        /// always knows; recorder presents as `"UNKNOWN"` only if a
        /// future incomplete-info mode strips it.
        resource_cw: u8,
    },
    Build {
        player: u8,
        kind: BuildKind,
        /// Edge or vertex index — interpretation per `kind`.
        location: u8,
    },
    LongestRoadChange {
        prev_owner: Option<u8>,
        new_owner: Option<u8>,
        length: u8,
    },
    LargestArmyChange {
        prev_owner: Option<u8>,
        new_owner: Option<u8>,
        knights: u8,
    },
    GameEnd {
        winner: u8,
        vp_p0: u8,
        vp_p1: u8,
    },
    /// Setup phase ended (the canonical "Phase 0.5 SETUP_COMPLETE"
    /// marker the replay recorder consumes).
    SetupComplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildKind {
    Settlement,
    City,
    Road,
}

impl BuildKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BuildKind::Settlement => "SETTLEMENT",
            BuildKind::City => "CITY",
            BuildKind::Road => "ROAD",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceChangeSource {
    Roll,
    BuildSettlement,
    BuildCity,
    BuildRoad,
    BuyDevCard,
    Yop,
    Monopoly,
    BankTrade,
    Steal,
    Discard,
    Setup,
}

impl ResourceChangeSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceChangeSource::Roll => "ROLL",
            ResourceChangeSource::BuildSettlement => "BUILD_SETTLEMENT",
            ResourceChangeSource::BuildCity => "BUILD_CITY",
            ResourceChangeSource::BuildRoad => "BUILD_ROAD",
            ResourceChangeSource::BuyDevCard => "BUY_DEV_CARD",
            ResourceChangeSource::Yop => "YOP",
            ResourceChangeSource::Monopoly => "MONOPOLY",
            ResourceChangeSource::BankTrade => "BANK_TRADE",
            ResourceChangeSource::Steal => "STEAL",
            ResourceChangeSource::Discard => "DISCARD",
            ResourceChangeSource::Setup => "SETUP",
        }
    }
}

impl Event {
    /// String tag matching the Python `BroadcastEventType` values
    /// (engine/broadcast.py:34-66). The replay recorder's
    /// `classify_step_events` routes off this.
    pub fn type_str(&self) -> &'static str {
        match self {
            Event::DiceRoll { .. } => "DICE_ROLL",
            Event::Discard { .. } => "DISCARD",
            Event::YearOfPlenty { .. } => "YOP",
            Event::ResourceChange { .. } => "RESOURCE_CHANGE",
            Event::Monopoly { .. } => "MONOPOLY",
            Event::MoveRobber { .. } => "MOVE_ROBBER",
            Event::Steal { .. } => "STEAL",
            Event::Build { .. } => "BUILD",
            Event::LongestRoadChange { .. } => "LONGEST_ROAD_CHANGE",
            Event::LargestArmyChange { .. } => "LARGEST_ARMY_CHANGE",
            Event::GameEnd { .. } => "GAME_END",
            Event::SetupComplete => "SETUP_COMPLETE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_strings_match_python_enum() {
        assert_eq!(
            Event::DiceRoll {
                player: 0,
                value: 7
            }
            .type_str(),
            "DICE_ROLL"
        );
        assert_eq!(
            Event::Discard {
                player: 0,
                resources: vec![]
            }
            .type_str(),
            "DISCARD"
        );
        assert_eq!(Event::SetupComplete.type_str(), "SETUP_COMPLETE");
        assert_eq!(BuildKind::Settlement.as_str(), "SETTLEMENT");
        assert_eq!(BuildKind::Road.as_str(), "ROAD");
        assert_eq!(
            ResourceChangeSource::BuildSettlement.as_str(),
            "BUILD_SETTLEMENT"
        );
    }
}
