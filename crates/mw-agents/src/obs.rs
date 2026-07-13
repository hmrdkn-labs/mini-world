//! The observation encoder (DESIGN.md §3): the fixed-size structured observation
//! the SOUL consumes — the versioned API between game and brain.
//!
//! Its byte size is independent of world population: a bigger world fills the
//! same `K_NEIGHBORS` slots, it never grows the struct. That is the property the
//! golden test locks down, and the reason a SOUL net can be retrained without
//! touching the sim. Everything is integer/fixed-point so it hashes and replays
//! bit-identically.

use mw_core::EntityId;

/// Nearest neighbors surfaced to the brain. Slots past the live neighbor count
/// carry `present = false`.
pub const K_NEIGHBORS: usize = 8;
/// Self stat/need slots (hunger, energy, social).
pub const N_STATS: usize = 3;
/// Recent-event summary buckets (Moved, Interacted, Spoke, Rejected).
pub const N_EVENT_KINDS: usize = 4;
/// Fixed-point ceiling of a need, mirroring the pack's `MAX_NEED`.
pub const NEED_ONE: i16 = 1000;

/// A goal is above threshold when its need's deficit exceeds this.
const GOAL_PRESSURE: i16 = 400;

/// The current-goal slot: the single most-pressing drive, or `None` when sated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Goal {
    None = 0,
    Eat = 1,
    Rest = 2,
    Socialize = 3,
}

/// One observed neighbor, enriched from the observer's memory + the neighbor's
/// persona. `dist2`/`opinion`/`faction`/`kind` are the network features; `id`
/// and `pos` are what the pointer head targets. `rel_pos` and `cell_class`
/// expose the spatial context consumed by the utility scorer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct NeighborView {
    pub present: bool,
    /// Squared grid distance from the observer — a population-independent metric.
    pub dist2: i32,
    /// Fixed-point opinion the observer holds of this neighbor.
    pub opinion: i32,
    pub faction: u8,
    pub kind: u8,
    pub id: Option<EntityId>,
    /// Absolute position used by the intent target head.
    pub pos: (i32, i32),
    /// Neighbor position relative to the observing agent.
    pub rel_pos: (i32, i32),
    /// Scenario-defined tile/cell class at the neighbor position.
    pub cell_class: u8,
}

/// The fixed-size encoded observation handed to the SOUL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentObs {
    pub tick: u64,
    /// Self needs, fixed-point in `[0, NEED_ONE]`.
    pub self_stats: [i16; N_STATS],
    /// Self position and scenario-defined tile/cell class.
    pub self_pos: (i32, i32),
    pub self_cell_class: u8,
    pub neighbors: [NeighborView; K_NEIGHBORS],
    /// Counts of remembered events by kind.
    pub events: [u16; N_EVENT_KINDS],
    /// Afforded-tool bitmask, sourced from the kernel via the pack seam.
    pub tool_mask: u32,
    /// Current-goal slot (a [`Goal`] discriminant).
    pub goal: u8,
}

/// Build the encoded observation. `cands` is every candidate neighbor; only the
/// nearest `K_NEIGHBORS` (by `dist2`, ties broken by entity index) reach the
/// slots — so adding faraway population changes neither the struct size nor the
/// chosen slots.
pub fn encode(
    tick: u64,
    self_stats: [i16; N_STATS],
    self_pos: (i32, i32),
    self_cell_class: u8,
    mut cands: Vec<NeighborView>,
    events: [u16; N_EVENT_KINDS],
    tool_mask: u32,
) -> AgentObs {
    cands.sort_by(|a, b| a.dist2.cmp(&b.dist2).then_with(|| key(a).cmp(&key(b))));

    let mut neighbors = [NeighborView::default(); K_NEIGHBORS];
    for (slot, c) in neighbors
        .iter_mut()
        .zip(cands.into_iter().take(K_NEIGHBORS))
    {
        *slot = c;
    }

    AgentObs {
        tick,
        self_stats,
        self_pos,
        self_cell_class,
        neighbors,
        events,
        tool_mask,
        goal: goal_of(&self_stats) as u8,
    }
}

/// Total, population-order-independent tiebreak key: entity index (absent last).
fn key(n: &NeighborView) -> u32 {
    n.id.map_or(u32::MAX, |e| e.index())
}

/// The active goal is the need with the largest deficit, once that deficit
/// clears [`GOAL_PRESSURE`]. Ties resolve to the lowest need index (hunger first)
/// so the goal slot is deterministic.
fn goal_of(stats: &[i16; N_STATS]) -> Goal {
    let deficits = [
        NEED_ONE - stats[0],
        NEED_ONE - stats[1],
        NEED_ONE - stats[2],
    ];
    let mut best = 0usize;
    for i in 1..N_STATS {
        if deficits[i] > deficits[best] {
            best = i;
        }
    }
    if deficits[best] < GOAL_PRESSURE {
        Goal::None
    } else {
        match best {
            0 => Goal::Eat,
            1 => Goal::Rest,
            _ => Goal::Socialize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_core::{KernelPack, World};

    fn view(idx_pos: (i32, i32), dist2: i32, id: EntityId) -> NeighborView {
        NeighborView {
            present: true,
            dist2,
            opinion: 0,
            faction: 0,
            kind: 0,
            id: Some(id),
            pos: idx_pos,
            rel_pos: idx_pos,
            cell_class: 0,
        }
    }

    // Golden snapshot: fixed inputs must produce an exactly known observation.
    #[test]
    fn golden_observation_snapshot() {
        let pack = KernelPack::new();
        let mut w = World::with_pack(1, &pack);
        let near = w.spawn((1, 0));
        let far = w.spawn((9, 9));

        let obs = encode(
            42,
            [700, 100, 550],
            (0, 0),
            0,
            vec![view((9, 9), 162, far), view((1, 0), 1, near)],
            [3, 1, 2, 0],
            0b0000_1111,
        );

        assert_eq!(obs.self_pos, (0, 0));
        assert_eq!(obs.self_cell_class, 0);
        assert_eq!(obs.neighbors[0].rel_pos, (1, 0));
        assert_eq!(obs.tick, 42);
        assert_eq!(obs.self_stats, [700, 100, 550]);
        assert_eq!(obs.events, [3, 1, 2, 0]);
        assert_eq!(obs.tool_mask, 0b0000_1111);
        // energy (100) is the deepest deficit -> Rest.
        assert_eq!(obs.goal, Goal::Rest as u8);
        // Nearest first regardless of input order.
        assert_eq!(obs.neighbors[0].id, Some(near));
        assert_eq!(obs.neighbors[0].dist2, 1);
        assert_eq!(obs.neighbors[1].id, Some(far));
        assert!(!obs.neighbors[2].present);
    }

    // The whole point: the encoded observation is a fixed-size record.
    #[test]
    fn observation_byte_size_is_population_independent() {
        // Fresh world per build so the nearest ids line up 0..8 in both cases;
        // the only difference between the runs is how much farther population
        // sits behind those eight.
        let build = |n: i32| {
            let pack = KernelPack::new();
            let mut w = World::with_pack(1, &pack);
            let mut cands = Vec::new();
            for i in 0..n {
                let e = w.spawn((i, 0));
                cands.push(view((i, 0), i * i, e));
            }
            encode(
                0,
                [500, 500, 500],
                (0, 0),
                0,
                cands,
                [0; N_EVENT_KINDS],
                u32::MAX,
            )
        };

        let small = build(10);
        let large = build(500);
        assert_eq!(
            std::mem::size_of_val(&small),
            std::mem::size_of_val(&large),
            "observation size must not depend on population"
        );
        // ...and the nearest-K content is identical: the extra 490 entities are
        // all farther, so they never displace the first eight slots.
        assert_eq!(small.neighbors, large.neighbors);
    }
}
