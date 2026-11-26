//! Headless, single-thread-owned authority for the mini-world simulation.
//!
//! The controller deliberately contains the non-`Send` village pack and the
//! policy state together. Hosts should construct and drive it on one actor
//! thread, then copy out [`TickOutcome`] and [`RunSnapshot`] projections.
pub mod dto;

use std::rc::Rc;

use mw_agents::habits::{HabitContext, HabitSoul};
use mw_agents::memory::{Memory, OPINION_ONE};
use mw_agents::obs::N_STATS;
use mw_agents::persona::{trait_idx, Persona};
use mw_agents::soul::{Body, Choice, Social, ToolSem, UtilitySoul, TOOL_SLOTS};
use mw_core::{EntityId, Event, Intent, World};
use mw_neural::ExpertiseLevel;
use mw_village::{decode, tile_at, verb, Action, Item, Tile, VillagePack, GRID, MAX_NEED};

/// Fixed run inputs for the utility-policy runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunConfig {
    pub seed: u64,
    pub agents: usize,
    pub expertise: ExpertiseLevel,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            agents: 50,
            expertise: ExpertiseLevel::Capable,
        }
    }
}

/// Policy identity captured at construction and carried by every projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunProvenance {
    pub policy_id: String,
    pub model_hash: String,
    pub backend_id: String,
    pub expertise: ExpertiseLevel,
}

/// The minimum result emitted by one authoritative simulation tick.
#[derive(Clone, Debug)]
pub struct TickOutcome {
    pub tick: u64,
    pub events: Vec<Event>,
    pub state_hash: u64,
}

/// Owned immutable state projection for a host/DTO layer.
#[derive(Clone, Debug)]
pub struct RunSnapshot {
    pub seed: u64,
    pub tick: u64,
    pub state_hash: u64,
    pub provenance: RunProvenance,
    pub grid: Vec<Tile>,
    pub agents: Vec<(EntityId, (i32, i32))>,
}

/// The sole owner of the live kernel, village state, and utility policy.
///
/// This type intentionally has no `Send`/`Sync` bounds. [`VillagePack`] uses
/// `Rc<RefCell<_>>`, so construction and all mutation stay on its owning
/// thread.
pub struct SimulationController {
    pack: Rc<VillagePack>,
    world: World,
    ids: Vec<EntityId>,
    positions: Vec<(i32, i32)>,
    soul: HabitSoul<UtilitySoul<VillageBody>>,
    director: Director,
    provenance: RunProvenance,
    last_events: usize,
}

impl SimulationController {
    /// Construct all simulation state on the calling thread.
    pub fn new(config: RunConfig) -> Self {
        let pack = Rc::new(VillagePack::new());
        let mut world = World::with_pack(config.seed, &*pack);
        let positions = start_positions(config.agents);
        let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
        let personas: Vec<Persona> = ids
            .iter()
            .map(|&id| Persona::new(config.seed, id))
            .collect();
        let factions: Vec<u8> = personas.iter().map(Persona::faction).collect();
        let memories: Vec<Memory> = ids
            .iter()
            .map(|&id| Memory::new(id, verb_affect()))
            .collect();
        let body = VillageBody::new(Rc::clone(&pack), factions);
        let utility = UtilitySoul::new(
            body,
            tool_table(),
            ids.clone(),
            personas,
            memories,
            positions.clone(),
        );
        let soul = HabitSoul::with_hit_hook_and_tool(
            utility,
            ids.clone(),
            UtilitySoul::<VillageBody>::habit_replay_tool,
            UtilitySoul::<VillageBody>::last_tool,
        );
        let director = Director::new(RingConfig::default(), ids.len(), (8, 8));
        let provenance = RunProvenance {
            policy_id: "utility-v0".to_string(),
            model_hash: "none".to_string(),
            backend_id: "rust-utility".to_string(),
            expertise: config.expertise,
        };
        Self {
            pack,
            world,
            ids,
            positions,
            soul,
            director,
            provenance,
            last_events: 0,
        }
    }

    /// Advance exactly one tick and return its event delta and canonical hash.
    pub fn step_one(&mut self) -> TickOutcome {
        self.set_policy_context();
        self.soul.inner_mut().snapshot(&self.world);
        let director = &self.director;
        self.world
            .step_gated(&*self.pack, &mut self.soul, |id, tick| {
                director.should_run_soul(id.index() as usize, tick)
            });

        let end = self.world.event_log().len();
        let events = self.world.event_log()[self.last_events..end].to_vec();
        self.last_events = end;
        self.soul.inner_mut().observe_events(&events);
        self.soul.observe_events(&events);
        let tick = self.world.tick();
        for event in &events {
            if is_notable(event) {
                self.director
                    .note_event(actor(event).index() as usize, tick);
            }
        }
        self.soul.inner_mut().decay_opinions();
        for (slot, &id) in self.ids.iter().enumerate() {
            if let Some(entity) = self.world.entity(id) {
                self.positions[slot] = entity.pos;
            }
        }
        self.director.update(&self.positions, tick);
        TickOutcome {
            tick,
            events,
            state_hash: self.state_hash(),
        }
    }

    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    pub fn state_hash(&self) -> u64 {
        self.world.state_hash(&*self.pack)
    }

    pub fn provenance(&self) -> &RunProvenance {
        &self.provenance
    }

    /// Stable entity order used by all projections and host-side displays.
    pub fn agent_ids(&self) -> &[EntityId] {
        &self.ids
    }

    /// Current positions in the same slot order as [`Self::agent_ids`].
    pub fn positions(&self) -> &[(i32, i32)] {
        &self.positions
    }

    /// Read-only projected needs for one entity at the current tick.
    pub fn needs(&self, id: EntityId) -> (i32, i32, i32) {
        self.pack.needs(id).project(self.tick())
    }

    /// Read-only character memory for host-side inspectors.
    pub fn memory(&self, slot: usize) -> Option<&Memory> {
        (slot < self.ids.len()).then(|| self.soul.inner().memory(slot))
    }

    /// Complete event history, including the latest [`TickOutcome`] delta.
    pub fn event_log(&self) -> &[Event] {
        self.world.event_log()
    }

    /// Current LOD ring as a stable display value: 0 cold, 1 warm, 2 hot.
    pub fn ring(&self, slot: usize) -> u8 {
        self.director.ring.get(slot).copied().unwrap_or(Ring::Cold) as u8
    }

    /// Update the LOD focus without exposing the live director authority.
    pub fn set_focus(&mut self, focus: (i32, i32)) {
        self.director.set_focus(focus);
    }

    pub fn snapshot_projection(&self) -> RunSnapshot {
        let grid = (0..GRID)
            .flat_map(|y| (0..GRID).map(move |x| tile_at((x, y))))
            .collect();
        let agents = self
            .ids
            .iter()
            .copied()
            .zip(
                self.ids
                    .iter()
                    .map(|&id| self.world.entity(id).map_or((0, 0), |entity| entity.pos)),
            )
            .collect();
        RunSnapshot {
            seed: self.world.seed(),
            tick: self.tick(),
            state_hash: self.state_hash(),
            provenance: self.provenance.clone(),
            grid,
            agents,
        }
    }
    fn set_policy_context(&mut self) {
        for &id in &self.ids {
            let (hunger, energy, social) = self.pack.needs(id).project(self.world.tick());
            let pos = self.world.entity(id).map_or((0, 0), |entity| entity.pos);
            let cell_class = match tile_at(pos) {
                Tile::Empty => 0,
                Tile::Home => 1,
                Tile::Bakery => 2,
                Tile::Well => 3,
                Tile::Field => 4,
            };
            self.soul.set_context(
                id,
                HabitContext {
                    needs: [hunger as i16, energy as i16, social as i16],
                    need_max: MAX_NEED as i16,
                    cell_class,
                    goal: 0,
                },
            );
        }
    }
}

/// Deterministic row-major initial layout shared by all simulation hosts.
pub fn start_positions(count: usize) -> Vec<(i32, i32)> {
    (0..count)
        .map(|i| {
            let i = i as i32;
            (i % GRID, (i / GRID) % GRID)
        })
        .collect()
}
/// Scenario body used by the utility policy. It is kept here rather than
/// depending on `mw-sim`, so the headless runtime remains a library authority.
pub struct VillageBody {
    pack: Rc<VillagePack>,
    factions: Vec<u8>,
}

impl VillageBody {
    /// Wrap a shared pack and its precomputed faction table.
    pub fn new(pack: Rc<VillagePack>, factions: Vec<u8>) -> Self {
        Self { pack, factions }
    }
    fn held(&self, entity: EntityId) -> Item {
        if self.pack.inventory(entity, Item::Food) > 0 {
            Item::Food
        } else {
            Item::Water
        }
    }

    fn on_ground(&self, pos: (i32, i32)) -> Item {
        let ground = self.pack.ground_at(pos);
        if ground[Item::Food as usize] > 0 {
            Item::Food
        } else {
            Item::Water
        }
    }

    fn destination(&self, entity: EntityId, tick: u64, from: (i32, i32)) -> (i32, i32) {
        let (hunger, energy, social) = self.pack.needs(entity).project(tick);
        let (dh, de, ds) = (MAX_NEED - hunger, MAX_NEED - energy, MAX_NEED - social);
        if dh >= de && dh >= ds {
            nearest(from, |tile| matches!(tile, Tile::Bakery | Tile::Field))
        } else if de >= ds {
            nearest(from, |tile| tile == Tile::Home)
        } else {
            (8, 8)
        }
    }
}

impl Body for VillageBody {
    fn self_stats(&self, entity: EntityId, tick: u64) -> [i16; N_STATS] {
        let (hunger, energy, social) = self.pack.needs(entity).project(tick);
        [hunger as i16, energy as i16, social as i16]
    }

    fn cell_class(&self, pos: (i32, i32)) -> u8 {
        match tile_at(pos) {
            Tile::Empty => 0,
            Tile::Home => 1,
            Tile::Bakery => 2,
            Tile::Well => 3,
            Tile::Field => 4,
        }
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
            Action::Eat => interact(entity, Action::Eat, Item::Food),
            Action::Sleep => interact(entity, Action::Sleep, Item::Food),
            Action::Work => interact(entity, Action::Work, Item::Food),
            Action::Use => interact(entity, Action::Use, Item::Water),
            Action::Drop => interact(entity, Action::Drop, self.held(entity)),
            Action::Pickup => interact(entity, Action::Pickup, self.on_ground(from)),
            Action::Speak => choice.target.map_or(Intent::Idle, |target| Intent::Speak {
                target,
                act: 0,
                topic: 0,
            }),
            Action::Give => choice
                .target
                .map_or(Intent::Idle, |target| Intent::Interact {
                    target,
                    verb: verb(Action::Give, self.held(entity)),
                }),
            Action::Move => step_toward(from, self.destination(entity, tick, from), false),
            Action::Follow => choice
                .target_pos
                .map_or(Intent::Idle, |target| step_toward(from, target, false)),
            Action::Flee => choice
                .target_pos
                .map_or(Intent::Idle, |target| step_toward(from, target, true)),
        }
    }

    fn tool_for_intent(&self, intent: &Intent) -> Option<u32> {
        match intent {
            Intent::Move { .. } => Some(Action::Move.id()),
            Intent::Speak { .. } => Some(Action::Speak.id()),
            Intent::Interact { verb, .. } => decode(*verb).0.map(Action::id),
            Intent::Idle => Some(Action::Idle.id()),
        }
    }
}

fn interact(entity: EntityId, action: Action, item: Item) -> Intent {
    Intent::Interact {
        target: entity,
        verb: verb(action, item),
    }
}

fn step_toward(from: (i32, i32), to: (i32, i32), away: bool) -> Intent {
    let sign = |delta: i32| delta.signum();
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

fn in_bounds(pos: (i32, i32)) -> bool {
    (0..GRID).contains(&pos.0) && (0..GRID).contains(&pos.1)
}

fn nearest(from: (i32, i32), predicate: impl Fn(Tile) -> bool) -> (i32, i32) {
    let mut best: Option<(i32, (i32, i32))> = None;
    for y in 0..GRID {
        for x in 0..GRID {
            if predicate(tile_at((x, y))) {
                let distance = (x - from.0).abs().max((y - from.1).abs());
                if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                    best = Some((distance, (x, y)));
                }
            }
        }
    }
    best.map_or((8, 8), |(_, pos)| pos)
}

fn tool_table() -> Vec<ToolSem> {
    let mut table = vec![ToolSem::default(); TOOL_SLOTS];
    table[Action::Move as usize] = ToolSem {
        is_move: true,
        ..Default::default()
    };
    table[Action::Eat as usize] = ToolSem {
        relieves: Some((0, 1000)),
        ..Default::default()
    };
    table[Action::Sleep as usize] = ToolSem {
        relieves: Some((1, 1000)),
        ..Default::default()
    };
    table[Action::Work as usize] = ToolSem {
        bias: Some(trait_idx::INDUSTRIOUSNESS),
        ..Default::default()
    };
    table[Action::Speak as usize] = ToolSem {
        relieves: Some((2, 1000)),
        social: Social::Befriend,
        needs_adjacent: true,
        ..Default::default()
    };
    table[Action::Give as usize] = ToolSem {
        social: Social::Befriend,
        gives: true,
        needs_adjacent: true,
        ..Default::default()
    };
    table[Action::Pickup as usize] = ToolSem {
        bias: Some(trait_idx::GREED),
        ..Default::default()
    };
    table[Action::Use as usize] = ToolSem {
        relieves: Some((1, 300)),
        ..Default::default()
    };
    table[Action::Follow as usize] = ToolSem {
        social: Social::Befriend,
        ..Default::default()
    };
    table[Action::Flee as usize] = ToolSem {
        social: Social::Flee,
        ..Default::default()
    };
    table[Action::Idle as usize] = ToolSem {
        base: 40,
        ..Default::default()
    };
    table
}

fn verb_affect() -> Vec<(u32, i32, i32)> {
    let actor = OPINION_ONE / 4;
    let receiver = OPINION_ONE;
    vec![
        (verb(Action::Give, Item::Food), actor, receiver),
        (verb(Action::Give, Item::Water), actor, receiver),
    ]
}

fn actor(event: &Event) -> EntityId {
    match *event {
        Event::Moved { actor, .. }
        | Event::Interacted { actor, .. }
        | Event::Spoke { actor, .. }
        | Event::Rejected { actor, .. } => actor,
    }
}

fn is_notable(event: &Event) -> bool {
    matches!(event, Event::Spoke { .. } | Event::Interacted { .. })
}

/// Live LOD ring assignment. This is intentionally private to the runtime;
/// hosts only observe the resulting state and events.
struct Director {
    cfg: RingConfig,
    focus: (i32, i32),
    ring: Vec<Ring>,
    cooling_since: Vec<Option<u64>>,
    promote_until: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
struct RingConfig {
    hot_radius: i32,
    warm_radius: i32,
    warm_cadence: u64,
    hysteresis: u64,
    promote_ticks: u64,
}

impl Default for RingConfig {
    fn default() -> Self {
        Self {
            hot_radius: 4,
            warm_radius: 12,
            warm_cadence: 8,
            hysteresis: 32,
            promote_ticks: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ring {
    Cold = 0,
    Warm = 1,
    Hot = 2,
}

impl Director {
    fn new(cfg: RingConfig, agents: usize, focus: (i32, i32)) -> Self {
        Self {
            cfg,
            focus,
            ring: vec![Ring::Cold; agents],
            cooling_since: vec![None; agents],
            promote_until: vec![0; agents],
        }
    }
    fn set_focus(&mut self, focus: (i32, i32)) {
        self.focus = focus;
    }

    fn note_event(&mut self, slot: usize, tick: u64) {
        if let Some(promote_until) = self.promote_until.get_mut(slot) {
            *promote_until = tick + self.cfg.promote_ticks;
        }
        if let Some(ring) = self.ring.get_mut(slot) {
            *ring = Ring::Hot;
        }
        if let Some(cooling) = self.cooling_since.get_mut(slot) {
            *cooling = None;
        }
    }

    fn band(&self, pos: (i32, i32)) -> Ring {
        let distance = (pos.0 - self.focus.0)
            .abs()
            .max((pos.1 - self.focus.1).abs());
        if distance <= self.cfg.hot_radius {
            Ring::Hot
        } else if distance <= self.cfg.warm_radius {
            Ring::Warm
        } else {
            Ring::Cold
        }
    }

    fn update(&mut self, positions: &[(i32, i32)], tick: u64) {
        for (slot, &pos) in positions.iter().enumerate() {
            let mut target = self.band(pos);
            if tick < self.promote_until[slot] {
                target = Ring::Hot;
            }
            let current = self.ring[slot];
            match target.cmp(&current) {
                std::cmp::Ordering::Greater => {
                    self.ring[slot] = target;
                    self.cooling_since[slot] = None;
                }
                std::cmp::Ordering::Equal => self.cooling_since[slot] = None,
                std::cmp::Ordering::Less => {
                    let since = *self.cooling_since[slot].get_or_insert(tick);
                    if tick.saturating_sub(since) >= self.cfg.hysteresis {
                        self.ring[slot] = target;
                        self.cooling_since[slot] = None;
                    }
                }
            }
        }
    }

    fn should_run_soul(&self, slot: usize, tick: u64) -> bool {
        match self.ring[slot] {
            Ring::Hot => true,
            Ring::Warm => self.cfg.warm_cadence == 0 || tick % self.cfg.warm_cadence == 0,
            Ring::Cold => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_step(
        world: &mut World,
        pack: &Rc<VillagePack>,
        soul: &mut HabitSoul<UtilitySoul<VillageBody>>,
        director: &mut Director,
        ids: &[EntityId],
        positions: &mut [(i32, i32)],
        last_events: &mut usize,
    ) {
        for &id in ids {
            let (hunger, energy, social) = pack.needs(id).project(world.tick());
            let pos = world.entity(id).map_or((0, 0), |entity| entity.pos);
            let cell_class = match tile_at(pos) {
                Tile::Empty => 0,
                Tile::Home => 1,
                Tile::Bakery => 2,
                Tile::Well => 3,
                Tile::Field => 4,
            };
            soul.set_context(
                id,
                HabitContext {
                    needs: [hunger as i16, energy as i16, social as i16],
                    need_max: MAX_NEED as i16,
                    cell_class,
                    goal: 0,
                },
            );
        }
        soul.inner_mut().snapshot(world);
        world.step_gated(&**pack, soul, |id, tick| {
            director.should_run_soul(id.index() as usize, tick)
        });
        let end = world.event_log().len();
        let events = &world.event_log()[*last_events..end];
        *last_events = end;
        soul.inner_mut().observe_events(events);
        soul.observe_events(events);
        let tick = world.tick();
        for event in events {
            if is_notable(event) {
                director.note_event(actor(event).index() as usize, tick);
            }
        }
        soul.inner_mut().decay_opinions();
        for (slot, &id) in ids.iter().enumerate() {
            if let Some(entity) = world.entity(id) {
                positions[slot] = entity.pos;
            }
        }
        director.update(positions, tick);
    }
    #[test]
    fn controller_matches_direct_world_pipeline() {
        let config = RunConfig {
            seed: 0x51_4d_1a,
            agents: 12,
            expertise: ExpertiseLevel::Capable,
        };
        let mut controller = SimulationController::new(config);
        let pack = Rc::new(VillagePack::new());
        let mut world = World::with_pack(config.seed, &*pack);
        let positions = start_positions(config.agents);
        let ids: Vec<EntityId> = positions.iter().map(|&pos| world.spawn(pos)).collect();
        let personas: Vec<Persona> = ids
            .iter()
            .map(|&id| Persona::new(config.seed, id))
            .collect();
        let factions: Vec<u8> = personas.iter().map(Persona::faction).collect();
        let memories: Vec<Memory> = ids
            .iter()
            .map(|&id| Memory::new(id, verb_affect()))
            .collect();
        let utility = UtilitySoul::new(
            VillageBody::new(Rc::clone(&pack), factions),
            tool_table(),
            ids.clone(),
            personas,
            memories,
            positions.clone(),
        );
        let mut soul = HabitSoul::with_hit_hook_and_tool(
            utility,
            ids.clone(),
            UtilitySoul::<VillageBody>::habit_replay_tool,
            UtilitySoul::<VillageBody>::last_tool,
        );
        let mut director = Director::new(RingConfig::default(), ids.len(), (8, 8));
        let mut direct_positions = positions;
        let mut last_events = 0;
        for tick in 0..200 {
            let outcome = controller.step_one();
            direct_step(
                &mut world,
                &pack,
                &mut soul,
                &mut director,
                &ids,
                &mut direct_positions,
                &mut last_events,
            );
            assert_eq!(outcome.tick, tick + 1);
            assert_eq!(outcome.state_hash, world.state_hash(&*pack));
        }
    }

    #[test]
    fn controller_determinism_depends_on_seed() {
        let config = RunConfig {
            seed: 77,
            agents: 10,
            ..RunConfig::default()
        };
        let mut first = SimulationController::new(config);
        let mut second = SimulationController::new(config);
        for _ in 0..200 {
            first.step_one();
            second.step_one();
        }
        assert_eq!(first.state_hash(), second.state_hash());

        let mut different = SimulationController::new(RunConfig { seed: 78, ..config });
        for _ in 0..200 {
            different.step_one();
        }
        assert_ne!(first.state_hash(), different.state_hash());
    }
}
