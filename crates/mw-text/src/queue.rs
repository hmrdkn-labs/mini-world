//! Dialogue priority queue (stub).
//!
//! TEXT is never in the tick loop and decode is the scarce resource, so render
//! requests are serviced highest-priority-first (DESIGN.md §6): a line the
//! player is watching preempts ambient chatter, which preempts AFK digests.
//! FIFO within a level keeps a single conversation's turns ordered. Wiring into
//! the scheduler comes later; this fixes the ordering contract now.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// Someone is watching this line right now — render first.
    PlayerFacing,
    /// On-screen background characters.
    Ambient,
    /// Off-screen catch-up summaries on return from AFK.
    Digest,
}

/// Three FIFO lanes drained strictly high-to-low.
#[derive(Default)]
pub struct PriorityQueue<T> {
    player_facing: VecDeque<T>,
    ambient: VecDeque<T>,
    digest: VecDeque<T>,
}

impl<T> PriorityQueue<T> {
    pub fn new() -> Self {
        Self {
            player_facing: VecDeque::new(),
            ambient: VecDeque::new(),
            digest: VecDeque::new(),
        }
    }

    pub fn push(&mut self, priority: Priority, item: T) {
        self.lane(priority).push_back(item);
    }

    /// Pop the oldest item from the highest non-empty lane.
    pub fn pop(&mut self) -> Option<T> {
        self.player_facing
            .pop_front()
            .or_else(|| self.ambient.pop_front())
            .or_else(|| self.digest.pop_front())
    }

    pub fn is_empty(&self) -> bool {
        self.player_facing.is_empty() && self.ambient.is_empty() && self.digest.is_empty()
    }

    fn lane(&mut self, priority: Priority) -> &mut VecDeque<T> {
        match priority {
            Priority::PlayerFacing => &mut self.player_facing,
            Priority::Ambient => &mut self.ambient,
            Priority::Digest => &mut self.digest,
        }
    }
}
