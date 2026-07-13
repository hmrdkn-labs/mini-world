//! The v0 SOUL — a hand-written utility scorer (DESIGN.md §5 training roadmap),
//! sitting behind the same [`SoulPolicy`] socket a distilled net drops into
//! later.
//!
//! It is scenario-agnostic: it scores the currently-afforded tools of an
//! [`AgentObs`] using the character's [`Persona`] and its memory-derived
//! opinions, and defers the three pack-specific facts it cannot know — a
//! character's needs, a neighbor's faction, and how a chosen tool becomes a
//! kernel `Intent` — to a [`Body`]. Every number is integer/fixed-point and the
//! only randomness is a small tie-break draw from the agent's own kernel RNG
//! stream, so decisions replay bit-identically.

use mw_core::{AgentRng, EntityId, Event, Intent, Observation, SoulPolicy};

use crate::memory::{Memory, OPINION_ONE};
use crate::obs::{self, AgentObs, NeighborView, NEED_ONE, N_EVENT_KINDS, N_STATS};
use crate::persona::{trait_idx, Persona, PERSONA_ONE};

/// Tool-id capacity of the affordance mask (one `u32`).
pub const TOOL_SLOTS: usize = 32;

// --- scoring weights (fixed-point, PERSONA_ONE == 1.0) ---
/// Instrumental navigation weight: an agent's pure move tracks its worst unmet
/// need at full strength, so a needy character decisively relocates toward food
/// or home instead of lingering with a neighbor. A direct relief tool (eat/
/// sleep), when afforded *here*, still outscores moving because it clears the
/// whole deficit rather than one step of travel.
const MOVE_FACTOR: i32 = 1000;
/// Opinion-directed social pull (befriend high-opinion neighbors). Deliberately
/// small: companionship is a preference layered on top of survival, never a
/// reason to starve. The opinion factor is clamped to one "unit" so a long
/// friendship cannot snowball past a pressing need.
const SOCIAL_WEIGHT: i32 = 120;
/// Flee-from-threat pull (avoid neighbors held in low opinion).
const THREAT_WEIGHT: i32 = 400;
/// Affinity a biasing persona trait lends its tool (work↔industriousness, …).
const BIAS_WEIGHT: i32 = 220;
/// Gifting resistance, scaled by greed.
const GIVE_COST: i32 = 200;
/// Foresight divisor: an industrious agent works now against future hunger.
const WORK_FORESIGHT: i32 = 3;
/// Tie-break jitter drawn from the agent RNG stream (exclusive upper bound).
const NOISE: u32 = 24;

/// How a tool touches neighbor opinion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Social {
    #[default]
    None,
    /// Targets the friendliest present neighbor (speak/give/follow).
    Befriend,
    /// Targets the most-disliked present neighbor and retreats (flee).
    Flee,
}

/// Static per-tool scoring semantics, indexed by tool id — supplied by the
/// scenario so the scorer stays generic across packs.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolSem {
    /// `(need index, relief strength)` this tool satisfies, if any.
    pub relieves: Option<(usize, i16)>,
    /// Persona trait that biases this tool upward, if any.
    pub bias: Option<usize>,
    /// Social role, for both scoring and target selection.
    pub social: Social,
    /// True for the pure-movement tool: scored by overall need pressure so a
    /// stranded-but-needy agent relocates instead of idling to death.
    pub is_move: bool,
    /// True for a tool that spends an item to befriend (gifting): greed resists.
    pub gives: bool,
    /// True when the pack requires the target within interaction range: the
    /// scorer then picks the friendliest *adjacent* neighbor so the intent it
    /// emits matches the affordance mask instead of pointing at a far friend.
    pub needs_adjacent: bool,
    /// Baseline score (idle uses a small positive one; most tools zero).
    pub base: i32,
}

/// The scorer's pick: a tool plus the neighbor it points at (if social).
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub tool: u32,
    pub target: Option<EntityId>,
    pub target_pos: Option<(i32, i32)>,
    pub goal: u8,
}

/// The scenario side of the socket: everything the generic scorer needs that
/// only the installed pack knows.
pub trait Body {
    /// Self needs/stats for `entity` at `tick`, fixed-point in `[0, NEED_ONE]`.
    fn self_stats(&self, entity: EntityId, tick: u64) -> [i16; N_STATS];
    /// Faction bucket of an observed entity (a cheap label).
    fn faction(&self, entity: EntityId) -> u8;
    /// Realize a chosen tool into a kernel intent, applying whatever routing and
    /// verb encoding the pack needs.
    fn to_intent(
        &self,
        entity: EntityId,
        tick: u64,
        self_pos: (i32, i32),
        choice: &Choice,
    ) -> Intent;
}

/// The utility SOUL. Owns per-character persona + memory and a position snapshot
/// the harness refreshes each tick, so it can encode a rich observation from the
/// kernel's minimal one and score the afforded tools deterministically.
pub struct UtilitySoul<B: Body> {
    body: B,
    sem: Vec<ToolSem>,
    ids: Vec<EntityId>,
    personas: Vec<Persona>,
    positions: Vec<(i32, i32)>,
    memories: Vec<Memory>,
    hist: [u64; TOOL_SLOTS],
    // Per-tick call resolution: the kernel calls `decide` once per entity in
    // stable index order, so a counter reset on each new tick recovers which
    // entity is deciding without the kernel observation carrying an id.
    last_tick: Option<u64>,
    cursor: usize,
}

impl<B: Body> UtilitySoul<B> {
    /// `ids`/`personas`/`memories`/`positions` are parallel, indexed by entity
    /// slot (== `EntityId::index` for a world that never despawns). `sem` is
    /// indexed by tool id.
    pub fn new(
        body: B,
        sem: Vec<ToolSem>,
        ids: Vec<EntityId>,
        personas: Vec<Persona>,
        memories: Vec<Memory>,
        positions: Vec<(i32, i32)>,
    ) -> Self {
        for (slot, id) in ids.iter().enumerate() {
            debug_assert_eq!(id.index() as usize, slot, "soul state must be slot-indexed");
        }
        Self {
            body,
            sem,
            ids,
            personas,
            positions,
            memories,
            hist: [0; TOOL_SLOTS],
            last_tick: None,
            cursor: 0,
        }
    }

    /// Refresh the position snapshot from the world. Call once per tick before
    /// stepping — positions do not change mid-tick, so one snapshot is valid for
    /// every entity's decision that tick.
    pub fn snapshot(&mut self, world: &mw_core::World) {
        for (slot, &id) in self.ids.iter().enumerate() {
            if let Some(e) = world.entity(id) {
                self.positions[slot] = e.pos;
            }
        }
    }

    /// Feed the tick's new events into memory, routed to the involved parties.
    pub fn observe_events(&mut self, events: &[Event]) {
        for ev in events {
            let (actor, target) = actor_target(ev);
            self.ingest_to(actor, ev);
            if let Some(t) = target {
                if t != actor {
                    self.ingest_to(t, ev);
                }
            }
        }
    }

    /// Apply one tick of opinion decay to every character.
    pub fn decay_opinions(&mut self) {
        for m in self.memories.iter_mut() {
            m.decay();
        }
    }

    /// Decision histogram indexed by tool id.
    pub fn histogram(&self) -> &[u64; TOOL_SLOTS] {
        &self.hist
    }

    /// Opinion `owner` holds of `other` — for inspection/tests.
    pub fn opinion(&self, owner: EntityId, other: EntityId) -> i32 {
        self.memories[owner.index() as usize].opinion(other)
    }

    /// Read-only view of a character's memory (opinions, salient facts) by
    /// entity slot — for inspection UIs. The UI never mutates through it.
    pub fn memory(&self, slot: usize) -> &Memory {
        &self.memories[slot]
    }

    /// Read-only view of a character's persona by entity slot.
    pub fn persona(&self, slot: usize) -> &Persona {
        &self.personas[slot]
    }

    fn ingest_to(&mut self, e: EntityId, ev: &Event) {
        let i = e.index() as usize;
        if i < self.memories.len() {
            self.memories[i].ingest(ev);
        }
    }

    /// Score every afforded tool and pick the argmax.
    fn score(&self, obs: &AgentObs, p: &Persona, rng: &mut AgentRng) -> Choice {
        let deficits = [
            (NEED_ONE - obs.self_stats[0]).max(0) as i32,
            (NEED_ONE - obs.self_stats[1]).max(0) as i32,
            (NEED_ONE - obs.self_stats[2]).max(0) as i32,
        ];
        let pressure = |i: usize| deficits[i] * p.weights[i] as i32 / PERSONA_ONE as i32;
        let max_pressure = (0..N_STATS).map(pressure).max().unwrap_or(0);

        // Friendliest neighbor overall (for follow, which travels to reach one),
        // the friendliest *adjacent* neighbor (for in-range acts like speak/give,
        // whose affordance the pack gates on adjacency), and the most-disliked.
        // Adjacency on this grid is Chebyshev <= 1, i.e. squared distance <= 2.
        const ADJ_DIST2: i32 = 2;
        let mut friend: Option<NeighborView> = None;
        let mut friend_adj: Option<NeighborView> = None;
        let mut threat: Option<NeighborView> = None;
        for n in obs.neighbors.iter().filter(|n| n.present) {
            if friend.is_none_or(|f| n.opinion > f.opinion) {
                friend = Some(*n);
            }
            if n.dist2 <= ADJ_DIST2 && friend_adj.is_none_or(|f| n.opinion > f.opinion) {
                friend_adj = Some(*n);
            }
            if threat.is_none_or(|t| n.opinion < t.opinion) {
                threat = Some(*n);
            }
        }

        let mut best = i32::MIN;
        let mut choice = Choice {
            tool: obs.tool_mask.trailing_zeros(), // a guaranteed-afforded fallback
            target: None,
            target_pos: None,
            goal: obs.goal,
        };

        for tool in 0..TOOL_SLOTS as u32 {
            if obs.tool_mask & (1 << tool) == 0 {
                continue;
            }
            let sem = self.sem.get(tool as usize).copied().unwrap_or_default();
            let mut s = sem.base;

            if let Some((need, relief)) = sem.relieves {
                s += pressure(need) * relief as i32 / PERSONA_ONE as i32;
            }
            if let Some(bi) = sem.bias {
                s += p.traits[bi] as i32 * BIAS_WEIGHT / PERSONA_ONE as i32;
                if bi == trait_idx::INDUSTRIOUSNESS {
                    s += pressure(0) / WORK_FORESIGHT; // work now, eat later
                }
            }
            if sem.is_move {
                s += max_pressure * MOVE_FACTOR / PERSONA_ONE as i32;
            }

            let mut target = None;
            let mut target_pos = None;
            match sem.social {
                Social::Befriend => {
                    let picked = if sem.needs_adjacent {
                        friend_adj
                    } else {
                        friend
                    };
                    let Some(f) = picked else { continue };
                    // Clamp the opinion factor to one unit so companionship is a
                    // bounded nudge, never a survival-overriding obsession.
                    let op = f.opinion.clamp(0, OPINION_ONE);
                    s += p.traits[trait_idx::SOCIABILITY] as i32 * op / OPINION_ONE * SOCIAL_WEIGHT
                        / PERSONA_ONE as i32;
                    if sem.gives {
                        s -= p.traits[trait_idx::GREED] as i32 * GIVE_COST / PERSONA_ONE as i32;
                    }
                    target = f.id;
                    target_pos = Some(f.pos);
                }
                Social::Flee => {
                    let Some(t) = threat else { continue };
                    let danger = (-t.opinion).clamp(0, OPINION_ONE);
                    s += p.traits[trait_idx::CAUTION] as i32 * danger / OPINION_ONE * THREAT_WEIGHT
                        / PERSONA_ONE as i32;
                    target = t.id;
                    target_pos = Some(t.pos);
                }
                Social::None => {}
            }

            s += rng.range_u32(NOISE) as i32;
            if s > best {
                best = s;
                choice = Choice {
                    tool,
                    target,
                    target_pos,
                    goal: obs.goal,
                };
            }
        }
        choice
    }
}

impl<B: Body> SoulPolicy for UtilitySoul<B> {
    fn decide(&mut self, obs: &Observation, rng: &mut AgentRng) -> Intent {
        // Resolve which entity is deciding from the stable per-tick call order.
        if self.last_tick != Some(obs.tick) {
            self.last_tick = Some(obs.tick);
            self.cursor = 0;
        }
        let slot = self.cursor;
        self.cursor += 1;

        let entity = self.ids[slot];
        let persona = self.personas[slot];
        let self_pos = self.positions[slot];

        if obs.tool_mask == 0 {
            return Intent::Idle; // e.g. the dead afford nothing
        }

        // Encode the rich observation: self needs from the body, true K-nearest
        // neighbors from the position snapshot, opinions from this character's
        // memory, factions from each neighbor's body.
        let self_stats = self.body.self_stats(entity, obs.tick);
        let mem = &self.memories[slot];
        let mut cands = Vec::with_capacity(self.positions.len().saturating_sub(1));
        for (s, &pos) in self.positions.iter().enumerate() {
            if s == slot {
                continue;
            }
            let nid = self.ids[s];
            let dx = (pos.0 - self_pos.0) as i64;
            let dy = (pos.1 - self_pos.1) as i64;
            cands.push(NeighborView {
                present: true,
                dist2: (dx * dx + dy * dy) as i32,
                opinion: mem.opinion(nid),
                faction: self.body.faction(nid),
                kind: 0,
                id: Some(nid),
                pos,
            });
        }
        let events = event_counts(mem);
        let agent_obs = obs::encode(obs.tick, self_stats, cands, events, obs.tool_mask);

        let choice = self.score(&agent_obs, &persona, rng);
        self.hist[choice.tool as usize] += 1;
        self.body.to_intent(entity, obs.tick, self_pos, &choice)
    }
}

fn event_counts(mem: &Memory) -> [u16; N_EVENT_KINDS] {
    let mut c = [0u16; N_EVENT_KINDS];
    for m in mem.events() {
        let i = match m.event {
            Event::Moved { .. } => 0,
            Event::Interacted { .. } => 1,
            Event::Spoke { .. } => 2,
            Event::Rejected { .. } => 3,
        };
        c[i] = c[i].saturating_add(1);
    }
    c
}

fn actor_target(event: &Event) -> (EntityId, Option<EntityId>) {
    match *event {
        Event::Moved { actor, .. } | Event::Rejected { actor, .. } => (actor, None),
        Event::Interacted { actor, target, .. } | Event::Spoke { actor, target, .. } => {
            (actor, Some(target))
        }
    }
}
