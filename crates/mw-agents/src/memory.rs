//! Tier-1 character memory (DESIGN.md §7): event ring buffer → decaying
//! per-entity opinion scores → salient-fact slots.
//!
//! Everything here is integer/fixed-point and driven solely by kernel [`Event`]s
//! and the kernel tick, so a memory replays bitwise-identically alongside the
//! world: no wall-clock, no hash-map iteration over state, no float math.

use mw_core::{EntityId, Event};

/// Fixed-point scale for opinion scores: `OPINION_ONE` represents 1.0.
pub const OPINION_ONE: i32 = 1 << 12;

/// Event ring capacity per character. Oldest entry is evicted at cap (FIFO),
/// which is deterministic regardless of how full the buffer is.
pub const RING_CAP: usize = 64;

/// Number of salient-fact slots surfaced (top-k by importance).
pub const SALIENT_K: usize = 4;

/// Opinion decays multiplicatively each tick: `score *= DECAY_FACTOR / 2^DECAY_SHIFT`.
/// ~0.99910/tick, so a relationship left untended fades to ~40% over 1000 ticks
/// while a fresh interaction still dominates. Multiply-then-shift keeps it exact
/// integer math (the closed form holds to within accumulated truncation).
const DECAY_SHIFT: u32 = 16;
const DECAY_FACTOR: i64 = 65_477;

/// Intrinsic opinion nudge for a `Spoke` act. Act deltas are deliberately
/// modest so conversation is a relationship signal, not a survival override.
/// `SPEAK_AFFECT` is the default greet delta and remains the dialogue baseline.
pub const SPEAK_AFFECT: i32 = OPINION_ONE / 8;

/// Return `(actor→target, target→actor)` deltas for a speak act. The two
/// directions are explicit because a taunt, warning, or gift need not be
/// received as the giver experiences it.
pub fn speak_affect(act: u32) -> (i32, i32) {
    match act {
        0 => (OPINION_ONE / 8, OPINION_ONE / 8),    // greet
        1 => (OPINION_ONE / 6, OPINION_ONE / 5),    // befriend
        2 => (-OPINION_ONE / 16, -OPINION_ONE / 8), // taunt
        3 => (OPINION_ONE / 16, OPINION_ONE / 16),  // trade
        4 => (OPINION_ONE / 8, OPINION_ONE / 6),    // console
        5 => (0, -OPINION_ONE / 16),                // warn
        _ => (SPEAK_AFFECT, SPEAK_AFFECT),
    }
}

/// One remembered event plus its precomputed magnitude (kind weight + |delta|),
/// the recency-independent half of the salience score.
#[derive(Clone, Debug)]
pub struct MemEvent {
    pub event: Event,
    /// Counterpart entity this event concerns from the owner's view, if any.
    pub other: Option<EntityId>,
    /// Non-negative importance magnitude, scaled up by recency for salience.
    pub magnitude: i32,
}

impl MemEvent {
    fn tick(&self) -> u64 {
        event_tick(&self.event)
    }
}

/// Per-character memory. Opinions live in a plain `Vec` (not a `HashMap`) so
/// iteration order — and therefore any derived state — is deterministic.
pub struct Memory {
    owner: EntityId,
    ring: Vec<MemEvent>,
    /// Oldest slot index; the ring is `[head..] ++ [..head]` in age order.
    head: usize,
    opinions: Vec<(EntityId, i32)>,
    /// Scenario-owned verb → `(actor→target, target→actor)` opinion deltas
    /// (fixed-point). Kept explicit so memory stays decoupled from any pack's
    /// verb vocabulary.
    verb_affect: Vec<(u32, i32, i32)>,
}

impl Memory {
    pub fn new(owner: EntityId, verb_affect: Vec<(u32, i32, i32)>) -> Self {
        Self {
            owner,
            ring: Vec::new(),
            head: 0,
            opinions: Vec::new(),
            verb_affect,
        }
    }

    /// Record a kernel event that involves the owner (as actor or target),
    /// pushing it into the ring and shifting the counterpart's opinion.
    /// Events not involving the owner are ignored.
    pub fn ingest(&mut self, event: &Event) {
        let (actor, target) = actor_target(event);
        let other = match (actor == self.owner, target) {
            (true, Some(t)) => Some(t),
            (_, _) if target == Some(self.owner) => Some(actor),
            _ if actor == self.owner => None,
            _ => return, // owner not involved
        };

        let delta = self.event_delta(event);
        if let (Some(o), true) = (other, delta != 0) {
            self.add_opinion(o, delta);
        }

        let magnitude = kind_weight(event) + delta.saturating_abs();
        self.push_ring(MemEvent {
            event: event.clone(),
            other,
            magnitude,
        });
    }

    /// Apply one tick of exponential decay to every opinion. Driven by the
    /// kernel tick so replay reproduces the same trajectory.
    pub fn decay(&mut self) {
        for (_, score) in self.opinions.iter_mut() {
            *score = ((*score as i64 * DECAY_FACTOR) >> DECAY_SHIFT) as i32;
        }
    }

    /// Current fixed-point opinion of `id` (0 if never interacted with).
    pub fn opinion(&self, id: EntityId) -> i32 {
        self.opinions
            .iter()
            .find(|(e, _)| *e == id)
            .map_or(0, |(_, s)| *s)
    }

    /// Events in the ring, oldest first.
    pub fn events(&self) -> impl Iterator<Item = &MemEvent> {
        let (front, back) = self.ring.split_at(self.head.min(self.ring.len()));
        back.iter().chain(front.iter())
    }

    /// Top-`SALIENT_K` events by importance = magnitude × recency, evaluated at
    /// `now`. Ordering is total and stable (importance desc, then newer tick,
    /// then lower actor index), so identical runs yield identical slots.
    pub fn salient(&self, now: u64) -> Vec<&MemEvent> {
        let mut scored: Vec<(i64, &MemEvent)> =
            self.ring.iter().map(|m| (salience(m, now), m)).collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0).then(b.1.tick().cmp(&a.1.tick())).then(
                actor_target(&a.1.event)
                    .0
                    .index()
                    .cmp(&actor_target(&b.1.event).0.index()),
            )
        });
        scored.into_iter().take(SALIENT_K).map(|(_, m)| m).collect()
    }

    fn add_opinion(&mut self, id: EntityId, delta: i32) {
        match self.opinions.iter_mut().find(|(e, _)| *e == id) {
            Some((_, score)) => *score = score.saturating_add(delta),
            None => self.opinions.push((id, delta)),
        }
    }

    fn push_ring(&mut self, m: MemEvent) {
        if self.ring.len() < RING_CAP {
            self.ring.push(m);
        } else {
            // Overwrite the oldest slot and advance the head — O(1), no shifting.
            self.ring[self.head] = m;
            self.head = (self.head + 1) % RING_CAP;
        }
    }

    fn event_delta(&self, event: &Event) -> i32 {
        let (actor, target) = actor_target(event);
        let actor_view = match *event {
            Event::Spoke { act, .. } => speak_affect(act).0,
            Event::Interacted { verb, .. } => self
                .verb_affect
                .iter()
                .find(|(v, _, _)| *v == verb)
                .map_or(0, |(_, actor_delta, _)| *actor_delta),
            _ => 0,
        };
        let target_view = match *event {
            Event::Spoke { act, .. } => speak_affect(act).1,
            Event::Interacted { verb, .. } => self
                .verb_affect
                .iter()
                .find(|(v, _, _)| *v == verb)
                .map_or(0, |(_, _, target_delta)| *target_delta),
            _ => 0,
        };
        if self.owner == actor {
            actor_view
        } else if target == Some(self.owner) {
            target_view
        } else {
            0
        }
    }
}

/// Base salience weight by event kind, before |delta| and recency.
fn kind_weight(event: &Event) -> i32 {
    match event {
        Event::Interacted { .. } | Event::Spoke { .. } => 2 * OPINION_ONE,
        Event::Moved { .. } | Event::Rejected { .. } => OPINION_ONE / 4,
    }
}

/// Recency-scaled importance. Recency weight is `SCALE / (SCALE + age)`, integer
/// throughout — recent high-magnitude events dominate, old ones fade smoothly.
fn salience(m: &MemEvent, now: u64) -> i64 {
    const SCALE: i64 = 256;
    let age = now.saturating_sub(m.tick()) as i64;
    (m.magnitude as i64 * SCALE) / (SCALE + age)
}

fn event_tick(event: &Event) -> u64 {
    match *event {
        Event::Moved { tick, .. }
        | Event::Interacted { tick, .. }
        | Event::Spoke { tick, .. }
        | Event::Rejected { tick, .. } => tick,
    }
}

fn actor_target(event: &Event) -> (EntityId, Option<EntityId>) {
    match *event {
        Event::Moved { actor, .. } | Event::Rejected { actor, .. } => (actor, None),
        Event::Interacted { actor, target, .. } | Event::Spoke { actor, target, .. } => {
            (actor, Some(target))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_core::World;

    const VERB_GIVE: u32 = 10;
    const VERB_ATTACK: u32 = 11;

    fn ids() -> (EntityId, EntityId) {
        // EntityId has no public constructor; mint real ones from a world.
        let mut w = World::new(1);
        (w.spawn((0, 0)), w.spawn((1, 0)))
    }

    fn affect() -> Vec<(u32, i32, i32)> {
        vec![
            // actor→target, target→actor
            (VERB_GIVE, OPINION_ONE / 4, OPINION_ONE),
            (VERB_ATTACK, -OPINION_ONE / 8, -2 * OPINION_ONE),
        ]
    }

    #[test]
    fn interaction_event_is_directional() {
        let (giver, receiver) = ids();
        let mut giver_memory = Memory::new(giver, affect());
        let mut receiver_memory = Memory::new(receiver, affect());
        let give = Event::Interacted {
            tick: 0,
            actor: giver,
            target: receiver,
            verb: VERB_GIVE,
        };
        giver_memory.ingest(&give);
        receiver_memory.ingest(&give);
        assert_eq!(giver_memory.opinion(receiver), OPINION_ONE / 4);
        assert_eq!(receiver_memory.opinion(giver), OPINION_ONE);

        let attack = Event::Interacted {
            tick: 1,
            actor: giver,
            target: receiver,
            verb: VERB_ATTACK,
        };
        giver_memory.ingest(&attack);
        receiver_memory.ingest(&attack);
        assert_eq!(
            giver_memory.opinion(receiver),
            OPINION_ONE / 4 - OPINION_ONE / 8
        );
        assert_eq!(
            receiver_memory.opinion(giver),
            OPINION_ONE - 2 * OPINION_ONE
        );
        assert!(
            receiver_memory.opinion(giver) < giver_memory.opinion(receiver),
            "victim should retain the stronger negative delta"
        );
    }

    #[test]
    fn interaction_event_shifts_opinion() {
        let (me, other) = ids();
        let mut mem = Memory::new(me, affect());
        assert_eq!(mem.opinion(other), 0);
        mem.ingest(&Event::Interacted {
            tick: 0,
            actor: other,
            target: me,
            verb: VERB_GIVE,
        });
        // Exact receiver delta, no decay applied yet.
        assert_eq!(mem.opinion(other), OPINION_ONE);
    }

    #[test]
    fn speak_affect_is_modest_and_act_dependent() {
        let (speaker, listener) = ids();
        let mut speaker_memory = Memory::new(speaker, Vec::new());
        let mut listener_memory = Memory::new(listener, Vec::new());
        let taunt = Event::Spoke {
            tick: 0,
            actor: speaker,
            target: listener,
            act: 2,
            topic: 0,
        };
        speaker_memory.ingest(&taunt);
        listener_memory.ingest(&taunt);
        assert_eq!(speaker_memory.opinion(listener), -OPINION_ONE / 16);
        assert_eq!(listener_memory.opinion(speaker), -OPINION_ONE / 8);
        assert!(speak_affect(1).1 > speak_affect(0).1);
    }

    #[test]
    fn decay_matches_closed_form_over_1000_ticks() {
        let (me, other) = ids();
        let mut mem = Memory::new(me, affect());
        let start = 100 * OPINION_ONE;
        mem.add_opinion(other, start);
        for _ in 0..1000 {
            mem.decay();
        }
        let factor = DECAY_FACTOR as f64 / (1u64 << DECAY_SHIFT) as f64;
        let expected = start as f64 * factor.powi(1000);
        let got = mem.opinion(other) as f64;
        // Within accumulated integer-truncation drift (< 1% here).
        assert!(
            (got - expected).abs() < expected * 0.01,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn ring_evicts_oldest_at_cap() {
        let (me, other) = ids();
        let mut mem = Memory::new(me, affect());
        for t in 0..(RING_CAP as u64 + 10) {
            mem.ingest(&Event::Moved {
                tick: t,
                actor: me,
                to: (t as i32, 0),
            });
        }
        let ticks: Vec<u64> = mem.events().map(|m| m.tick()).collect();
        assert_eq!(ticks.len(), RING_CAP);
        // Oldest 10 evicted; remaining is a contiguous ascending window.
        assert_eq!(ticks.first().copied(), Some(10));
        assert_eq!(ticks.last().copied(), Some(RING_CAP as u64 + 9));
        assert!(ticks.windows(2).all(|w| w[0] < w[1]));
        let _ = other;
    }

    fn run_salient() -> Vec<(u64, i32)> {
        let (me, other) = ids();
        let mut mem = Memory::new(me, affect());
        mem.ingest(&Event::Moved {
            tick: 0,
            actor: me,
            to: (1, 0),
        });
        mem.ingest(&Event::Moved {
            tick: 3,
            actor: me,
            to: (2, 0),
        });
        mem.ingest(&Event::Interacted {
            tick: 5,
            actor: other,
            target: me,
            verb: VERB_GIVE,
        });
        mem.ingest(&Event::Spoke {
            tick: 7,
            actor: other,
            target: me,
            act: 0,
            topic: 0,
        });
        mem.ingest(&Event::Interacted {
            tick: 9,
            actor: other,
            target: me,
            verb: VERB_ATTACK,
        });
        mem.salient(10)
            .iter()
            .map(|m| (m.tick(), m.magnitude))
            .collect()
    }

    #[test]
    fn salient_selection_is_stable_across_identical_runs() {
        assert_eq!(run_salient(), run_salient());
        // The high-magnitude give/attack interactions must outrank the move.
        let ticks: Vec<u64> = run_salient().iter().map(|(t, _)| *t).collect();
        assert!(ticks.contains(&5) && ticks.contains(&9));
        assert!(!ticks.contains(&0));
    }

    #[test]
    fn scripted_exchange_trajectory() {
        let (a, b) = ids();
        let mut mem = Memory::new(b, affect()); // b's view of a
        let mut traj = Vec::new();
        let mut record = |mem: &Memory, label: &str| {
            let v = mem.opinion(a);
            println!("{label}: opinion(a) = {v}");
            traj.push(v);
        };

        record(&mem, "start");
        mem.ingest(&Event::Interacted {
            tick: 0,
            actor: a,
            target: b,
            verb: VERB_GIVE,
        });
        record(&mem, "after give");
        mem.ingest(&Event::Spoke {
            tick: 1,
            actor: a,
            target: b,
            act: 0,
            topic: 0,
        });
        record(&mem, "after speak");
        let peak = mem.opinion(a);
        mem.ingest(&Event::Interacted {
            tick: 2,
            actor: a,
            target: b,
            verb: VERB_ATTACK,
        });
        record(&mem, "after attack");
        let after_attack = mem.opinion(a);
        for _ in 0..500 {
            mem.decay();
        }
        record(&mem, "after 500 ticks decay");
        let decayed = mem.opinion(a);

        // give+speak drive opinion up, attack drives it below zero, decay pulls
        // the (negative) score back toward zero.
        assert!(traj[1] > traj[0], "give should raise opinion");
        assert!(traj[2] > traj[1], "speak should raise opinion further");
        assert!(after_attack < peak, "attack should drop opinion");
        assert!(after_attack < 0, "attack outweighs give+speak");
        assert!(
            decayed > after_attack,
            "decay pulls negative score up toward 0"
        );
        assert!(decayed < 0, "still negative after partial decay");
    }
}
