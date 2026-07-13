//! SOUL policies — skeleton.
//!
//! Home of the [`SoulPolicy`] implementations: the v0 utility-AI scorer, later
//! the distilled tiny net. Both sit behind the same contract. No behavior yet.

pub mod memory;

use mw_core::{AgentRng, Intent, Observation, SoulPolicy};

/// v0 hand-written utility-AI scorer (DESIGN.md §5 training roadmap).
#[derive(Default)]
pub struct UtilityPolicy;

impl SoulPolicy for UtilityPolicy {
    fn decide(&mut self, _observation: &Observation, _rng: &mut AgentRng) -> Intent {
        todo!("utility scoring over the afforded-tool mask")
    }
}
