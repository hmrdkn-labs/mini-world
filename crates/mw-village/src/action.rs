//! The village's digital body: the 12-tool action manifest and the codec that
//! maps tools onto the kernel's fixed `Intent` set.
//!
//! The kernel `Intent` enum is closed (Move/Interact/Speak/Idle). Scenario
//! tools therefore ride on those four carriers: movement tools become `Move`,
//! `speak` becomes `Speak`, `idle` becomes `Idle`, and every remaining tool is
//! an `Interact` whose opaque `verb` word encodes `(action, item)`. This module
//! owns that encoding so the rest of the pack speaks in `Action`, not raw bits.

use mw_core::{ActionManifest, ArgKind, ArgSchema, ToolDescriptor};

/// Items the village economy moves around. Encoded in the high byte of an
/// `Interact` verb for the tools that take an item argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Item {
    Food = 0,
    Water = 1,
}

/// Number of distinct item kinds — sizes per-entity and per-tile item arrays.
pub const ITEM_COUNT: usize = 2;

impl Item {
    pub fn from_id(id: u32) -> Option<Item> {
        match id {
            0 => Some(Item::Food),
            1 => Some(Item::Water),
            _ => None,
        }
    }
}

/// One manifest tool. Discriminants are the tool ids the observation's
/// `tool_mask` bits index (bit `i` = `Action` with id `i`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Move = 0,
    Eat = 1,
    Sleep = 2,
    Work = 3,
    Speak = 4,
    Give = 5,
    Pickup = 6,
    Drop = 7,
    Use = 8,
    Follow = 9,
    Flee = 10,
    Idle = 11,
}

/// Number of tools in the manifest.
pub const TOOL_COUNT: u32 = 12;

impl Action {
    pub fn id(self) -> u32 {
        self as u32
    }

    pub fn from_id(id: u32) -> Option<Action> {
        Some(match id {
            0 => Action::Move,
            1 => Action::Eat,
            2 => Action::Sleep,
            3 => Action::Work,
            4 => Action::Speak,
            5 => Action::Give,
            6 => Action::Pickup,
            7 => Action::Drop,
            8 => Action::Use,
            9 => Action::Follow,
            10 => Action::Flee,
            11 => Action::Idle,
            _ => return None,
        })
    }
}

/// Encode an `Interact` verb for a tool that carries an item (give/drop/use)
/// or a bare item-less tool (eat/sleep/work → `Item::Food`, unused). Low byte
/// is the action id, next byte the item id.
pub fn verb(action: Action, item: Item) -> u32 {
    action.id() | ((item as u32) << 8)
}

/// Decode an `Interact` verb back into its action (or `None` for an unknown
/// tool id) and item.
pub fn decode(v: u32) -> (Option<Action>, Option<Item>) {
    (Action::from_id(v & 0xff), Item::from_id((v >> 8) & 0xff))
}

/// The full 12-tool manifest. Ids are dense `0..12` so a bitmask over tool ids
/// is a bitmask over manifest indices.
pub fn manifest() -> ActionManifest {
    let entity = |name: &str| ArgSchema {
        name: name.to_owned(),
        kind: ArgKind::EntityRef,
    };
    let scalar = |name: &str| ArgSchema {
        name: name.to_owned(),
        kind: ArgKind::Scalar,
    };
    let item = |name: &str| ArgSchema {
        name: name.to_owned(),
        kind: ArgKind::Enum {
            variants: ITEM_COUNT as u32,
        },
    };
    let tool = |a: Action, args: Vec<ArgSchema>| ToolDescriptor {
        id: a.id(),
        name: format!("{a:?}").to_lowercase(),
        args,
    };

    ActionManifest {
        tools: vec![
            tool(Action::Move, vec![scalar("dx"), scalar("dy")]),
            tool(Action::Eat, vec![]),
            tool(Action::Sleep, vec![]),
            tool(Action::Work, vec![]),
            tool(
                Action::Speak,
                vec![
                    entity("target"),
                    ArgSchema {
                        name: "act".to_owned(),
                        kind: ArgKind::Enum { variants: 4 },
                    },
                    scalar("topic"),
                ],
            ),
            tool(Action::Give, vec![entity("target"), item("item")]),
            tool(Action::Pickup, vec![item("item")]),
            tool(Action::Drop, vec![item("item")]),
            tool(Action::Use, vec![item("item")]),
            tool(Action::Follow, vec![entity("target")]),
            tool(Action::Flee, vec![entity("threat")]),
            tool(Action::Idle, vec![]),
        ],
    }
}
