//! SOUL policies and the character state they run on.
//!
//! The brain layer of the platform: the deterministic [`persona`] a character
//! is conditioned on, the fixed-size [`obs`] encoder that turns world + memory
//! into the SOUL's input, tier-1 [`memory`], and the v0 [`soul::UtilitySoul`] —
//! a hand-written utility scorer behind the same [`mw_core::SoulPolicy`] socket
//! a distilled net drops into later (DESIGN.md §5).

pub mod dialogue;
pub mod habits;
/// Singular alias retained for the DESIGN.md building-block name.
pub use habits as habit;
pub mod memory;
pub mod obs;
pub mod persona;
pub mod soul;

pub use dialogue::{
    Conversation, ConversationLog, DialogueRenderer, FocusPoint, MockRenderer, PersonaCard,
    PersonaRegistry, RenderRequest, SliceRegistry, Vocab,
};
pub use habits::{
    HabitCache, HabitContext, HabitKey, HabitSoul, HabitStats, HABIT_CAPACITY, HABIT_TTL,
    NEED_BANDS,
};
pub use obs::{AgentObs, Goal, NeighborView, K_NEIGHBORS, NEED_ONE, N_EVENT_KINDS, N_STATS};
pub use persona::{Persona, N_FACTIONS, N_TRAITS, N_WEIGHTS, PERSONA_ONE};
pub use soul::{Body, Choice, DecisionTrace, Social, ToolSem, UtilitySoul, TOOL_SLOTS};
