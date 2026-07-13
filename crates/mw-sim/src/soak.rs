//! The 50-agent village soak: the full observe → decide → validate → execute
//! loop with the village pack, the utility SOUL, and per-character memory wired
//! together at scale, plus a report the gate test checks.
//!
//! Determinism holds end to end: personas come from the seed, decisions from
//! kernel RNG streams, needs/opinions from integer closed-form updates, and the
//! only wall-clock read is the throughput timer — which never feeds sim state.

use std::rc::Rc;
use std::time::Instant;

use mw_agents::memory::{Memory, OPINION_ONE};
use mw_agents::obs::N_STATS;
use mw_agents::persona::{trait_idx, Persona};
use mw_agents::soul::{Body, Choice, Social, ToolSem, UtilitySoul, TOOL_SLOTS};
use mw_core::{EntityId, Intent, World};
use mw_village::{tile_at, verb, Action, Item, Tile, VillagePack, GRID, MAX_NEED, TOOL_COUNT};

/// Soak parameters.
#[derive(Clone, Copy, Debug)]
pub struct SoakConfig {
    pub seed: u64,
    pub agents: i32,
    pub ticks: u64,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            agents: 50,
            ticks: 10_000,
        }
    }
}

/// What the run produced. Everything here is deterministic in `(seed, agents,
/// ticks)` except `elapsed_secs` (a throughput measurement, report-only).
#[derive(Clone, Debug)]
pub struct SoakReport {
    pub cfg: SoakConfig,
    pub histogram: [u64; TOOL_SLOTS],
    pub deaths: usize,
    pub final_hash: u64,
    pub elapsed_secs: f64,
    /// Per-tick sum of every agent's projected (hunger, energy, social),
    /// accumulated across the whole run. Divided by `ticks * agents` it is the
    /// time-averaged mean need level — the reference the cold fast-forward drift
    /// test compares against. Deterministic in `(seed, agents, ticks)`.
    pub sum_needs: [u128; N_STATS],
}

impl SoakReport {
    pub fn total_actions(&self) -> u64 {
        self.histogram.iter().sum()
    }

    pub fn ticks_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.cfg.ticks as f64 / self.elapsed_secs
        } else {
            f64::INFINITY
        }
    }

    /// Time-averaged mean need level per stat over the whole run.
    pub fn mean_needs(&self) -> [f64; N_STATS] {
        let denom = (self.cfg.ticks * self.cfg.agents.max(0) as u64).max(1) as f64;
        let mut m = [0.0; N_STATS];
        for (i, s) in self.sum_needs.iter().enumerate() {
            m[i] = *s as f64 / denom;
        }
        m
    }

    /// Largest single-tool share of all decisions — the histogram-degeneracy
    /// metric the gate bounds below 80%.
    pub fn max_share(&self) -> f64 {
        let total = self.total_actions();
        if total == 0 {
            return 0.0;
        }
        let top = self.histogram.iter().copied().max().unwrap_or(0);
        top as f64 / total as f64
    }

    /// Human-readable action histogram, one non-zero tool per line.
    pub fn histogram_lines(&self) -> Vec<String> {
        let total = self.total_actions().max(1);
        (0..TOOL_COUNT)
            .filter_map(|id| {
                let n = self.histogram[id as usize];
                if n == 0 {
                    return None;
                }
                let name = Action::from_id(id).map(|a| format!("{a:?}")).unwrap();
                Some(format!(
                    "  {:<7} {:>10}  {:>5.1}%",
                    name.to_lowercase(),
                    n,
                    100.0 * n as f64 / total as f64
                ))
            })
            .collect()
    }
}

/// The scenario side of the SOUL socket for the village: needs projection,
/// precomputed factions, and turning a chosen tool into a kernel intent with the
/// village's routing + verb codec.
pub struct VillageBody {
    pack: Rc<VillagePack>,
    factions: Vec<u8>,
}

impl VillageBody {
    /// Wrap a shared pack and its precomputed faction table. The `Rc` lets the
    /// body live inside a soul that is stored alongside the pack (the TUI keeps
    /// both in one struct) without a self-referential borrow.
    pub fn new(pack: Rc<VillagePack>, factions: Vec<u8>) -> Self {
        Self { pack, factions }
    }
}

impl Body for VillageBody {
    fn self_stats(&self, entity: EntityId, tick: u64) -> [i16; N_STATS] {
        let (h, en, so) = self.pack.needs(entity).project(tick);
        [h as i16, en as i16, so as i16]
    }

    fn faction(&self, entity: EntityId) -> u8 {
        self.factions
            .get(entity.index() as usize)
            .copied()
            .unwrap_or(0)
    }

    fn to_intent(&self, entity: EntityId, tick: u64, from: (i32, i32), choice: &Choice) -> Intent {
        let Some(action) = Action::from_id(choice.tool) else {
            return Intent::Idle;
        };
        match action {
            Action::Idle => Intent::Idle,
            // Location/self acts ride an Interact whose target is self (the
            // pack ignores the target for these; it only needs to exist).
            Action::Eat => interact(entity, Action::Eat, Item::Food),
            Action::Sleep => interact(entity, Action::Sleep, Item::Food),
            Action::Work => interact(entity, Action::Work, Item::Food),
            Action::Use => interact(entity, Action::Use, Item::Water),
            Action::Drop => interact(entity, Action::Drop, self.held(entity)),
            Action::Pickup => interact(entity, Action::Pickup, self.on_ground(from)),
            // Social acts target a chosen neighbor.
            Action::Speak => match choice.target {
                Some(t) => Intent::Speak {
                    target: t,
                    act: 0,
                    topic: 0,
                },
                None => Intent::Idle,
            },
            Action::Give => match choice.target {
                Some(t) => Intent::Interact {
                    target: t,
                    verb: verb(Action::Give, self.held(entity)),
                },
                None => Intent::Idle,
            },
            // Movement tools resolve to a single kernel step.
            Action::Move => step_toward(from, self.destination(entity, tick, from), false),
            Action::Follow => match choice.target_pos {
                Some(tp) => step_toward(from, tp, false),
                None => Intent::Idle,
            },
            Action::Flee => match choice.target_pos {
                Some(tp) => step_toward(from, tp, true),
                None => Intent::Idle,
            },
        }
    }
}

impl VillageBody {
    /// An item the entity carries (food first), for give/drop.
    fn held(&self, entity: EntityId) -> Item {
        if self.pack.inventory(entity, Item::Food) > 0 {
            Item::Food
        } else {
            Item::Water
        }
    }

    /// An item lying on the entity's tile (food first), for pickup.
    fn on_ground(&self, pos: (i32, i32)) -> Item {
        let g = self.pack.ground_at(pos);
        if g[Item::Food as usize] > 0 {
            Item::Food
        } else {
            Item::Water
        }
    }

    /// Where to head for the most-pressing need: food when hungry, a home tile
    /// when tired, otherwise the bakery — the natural gathering hub where crowds
    /// (and speak/give opportunities) form.
    fn destination(&self, entity: EntityId, tick: u64, from: (i32, i32)) -> (i32, i32) {
        let (h, en, so) = self.pack.needs(entity).project(tick);
        let (dh, de, ds) = (MAX_NEED - h, MAX_NEED - en, MAX_NEED - so);
        if dh >= de && dh >= ds {
            nearest(from, |t| matches!(t, Tile::Bakery | Tile::Field))
        } else if de >= ds {
            nearest(from, |t| t == Tile::Home)
        } else {
            (8, 8)
        }
    }
}

fn interact(entity: EntityId, action: Action, item: Item) -> Intent {
    Intent::Interact {
        target: entity,
        verb: verb(action, item),
    }
}

fn in_bounds(p: (i32, i32)) -> bool {
    (0..GRID).contains(&p.0) && (0..GRID).contains(&p.1)
}

/// One unit step toward (or away from) `to`, dropping any axis that would leave
/// the map. Diagonals are legal (Chebyshev range 1). A fully blocked step idles.
fn step_toward(from: (i32, i32), to: (i32, i32), away: bool) -> Intent {
    let sign = |d: i32| d.signum();
    let (mut dx, mut dy) = (sign(to.0 - from.0), sign(to.1 - from.1));
    if away {
        dx = -dx;
        dy = -dy;
    }
    if !in_bounds((from.0 + dx, from.1)) {
        dx = 0;
    }
    if !in_bounds((from.0, from.1 + dy)) {
        dy = 0;
    }
    if dx == 0 && dy == 0 {
        Intent::Idle
    } else {
        Intent::Move { dx, dy }
    }
}

/// Nearest tile (by Chebyshev distance) satisfying `pred`; ties resolve by
/// scan order so the result is deterministic. Falls back to the bakery.
fn nearest(from: (i32, i32), pred: impl Fn(Tile) -> bool) -> (i32, i32) {
    let mut best: Option<(i32, (i32, i32))> = None;
    for y in 0..GRID {
        for x in 0..GRID {
            if pred(tile_at((x, y))) {
                let d = (x - from.0).abs().max((y - from.1).abs());
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, (x, y)));
                }
            }
        }
    }
    best.map_or((8, 8), |(_, p)| p)
}

/// Deterministic starting layout: a grid sweep so a given agent count always
/// begins the same way.
pub fn start_positions(count: i32) -> Vec<(i32, i32)> {
    (0..count).map(|i| (i % GRID, (i / GRID) % GRID)).collect()
}

/// Village tool-scoring table, indexed by [`Action`] id. Encodes which need
/// each tool relieves, its persona affinity, and its social role — the scenario
/// knowledge the generic scorer consumes.
pub fn tool_table() -> Vec<ToolSem> {
    let mut t = vec![ToolSem::default(); TOOL_COUNT as usize];
    t[Action::Move as usize] = ToolSem {
        is_move: true,
        ..Default::default()
    };
    t[Action::Eat as usize] = ToolSem {
        relieves: Some((0, 1000)),
        ..Default::default()
    };
    t[Action::Sleep as usize] = ToolSem {
        relieves: Some((1, 1000)),
        ..Default::default()
    };
    t[Action::Work as usize] = ToolSem {
        bias: Some(trait_idx::INDUSTRIOUSNESS),
        ..Default::default()
    };
    t[Action::Speak as usize] = ToolSem {
        relieves: Some((2, 1000)),
        social: Social::Befriend,
        needs_adjacent: true,
        ..Default::default()
    };
    t[Action::Give as usize] = ToolSem {
        social: Social::Befriend,
        gives: true,
        needs_adjacent: true,
        ..Default::default()
    };
    t[Action::Pickup as usize] = ToolSem {
        bias: Some(trait_idx::GREED),
        ..Default::default()
    };
    t[Action::Use as usize] = ToolSem {
        relieves: Some((1, 300)),
        ..Default::default()
    };
    t[Action::Follow as usize] = ToolSem {
        social: Social::Befriend,
        ..Default::default()
    };
    t[Action::Flee as usize] = ToolSem {
        social: Social::Flee,
        ..Default::default()
    };
    t[Action::Idle as usize] = ToolSem {
        base: 40,
        ..Default::default()
    };
    // Drop keeps the default (base 0): only picked when nothing scores better.
    t
}

/// Gifts build rapport: receiving one (or giving one) shifts opinion positively.
pub fn verb_affect() -> Vec<(u32, i32)> {
    let g = OPINION_ONE / 4;
    vec![
        (verb(Action::Give, Item::Food), g),
        (verb(Action::Give, Item::Water), g),
    ]
}

/// FNV-1a fold — a stable, architecture-independent hash mix.
fn fold(mut h: u64, x: u64) -> u64 {
    h ^= x;
    h.wrapping_mul(0x0000_0100_0000_01b3)
}

/// Canonical final hash: kernel positions plus each character's needs and alive
/// flag, so the digest reflects the social/economic state the kernel hash alone
/// (positions only) cannot see. Iterates in slot order — no map iteration.
fn hash_state(world: &World, pack: &VillagePack, ids: &[EntityId]) -> u64 {
    let tick = world.tick();
    let mut h = fold(0xcbf2_9ce4_8422_2325, world.state_hash());
    for &id in ids {
        let (hu, en, so) = pack.needs(id).project(tick);
        h = fold(h, hu as u64);
        h = fold(h, en as u64);
        h = fold(h, so as u64);
        h = fold(h, pack.is_dead(world, id) as u64);
    }
    h
}

/// Run the soak and return its report.
pub fn run(cfg: SoakConfig) -> SoakReport {
    let pack = Rc::new(VillagePack::new());
    let mut world = World::with_pack(cfg.seed, &*pack);

    let positions = start_positions(cfg.agents);
    let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
    let personas: Vec<Persona> = ids.iter().map(|&id| Persona::new(cfg.seed, id)).collect();
    let factions: Vec<u8> = personas.iter().map(|p| p.faction()).collect();
    let memories: Vec<Memory> = ids
        .iter()
        .map(|&id| Memory::new(id, verb_affect()))
        .collect();

    let body = VillageBody::new(Rc::clone(&pack), factions);
    let mut soul = UtilitySoul::new(
        body,
        tool_table(),
        ids.clone(),
        personas,
        memories,
        positions,
    );

    let mut last_events = 0usize;
    let mut sum_needs = [0u128; N_STATS];
    let t0 = Instant::now();
    for _ in 0..cfg.ticks {
        // Snapshot before stepping: positions are frozen within a tick, so one
        // snapshot serves every decision that tick.
        soul.snapshot(&world);
        world.step(&*pack, &mut soul);
        let events = world.event_log();
        soul.observe_events(&events[last_events..]);
        last_events = events.len();
        soul.decay_opinions();
        // Sample the post-tick need trajectory for the mean-need reference.
        let t = world.tick();
        for &id in &ids {
            let (h, en, so) = pack.needs(id).project(t);
            sum_needs[0] += h as u128;
            sum_needs[1] += en as u128;
            sum_needs[2] += so as u128;
        }
    }
    let elapsed_secs = t0.elapsed().as_secs_f64();

    let deaths = ids.iter().filter(|&&id| pack.is_dead(&world, id)).count();
    let final_hash = hash_state(&world, &pack, &ids);
    SoakReport {
        cfg,
        histogram: *soul.histogram(),
        deaths,
        final_hash,
        elapsed_secs,
        sum_needs,
    }
}
