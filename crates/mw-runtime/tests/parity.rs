use std::rc::Rc;

use mw_agents::memory::{Memory, OPINION_ONE};
use mw_agents::persona::{trait_idx, Persona};
use mw_agents::{HabitContext, HabitSoul, Social, ToolSem, UtilitySoul, TOOL_SLOTS};
use mw_core::{EntityId, Event, World};
use mw_neural::ExpertiseLevel;
use mw_runtime::{start_positions, RunConfig, SimulationController, VillageBody};
use mw_village::{tile_at, verb, Action, Item, Tile, VillagePack, MAX_NEED};

const HOT_RADIUS: i32 = 4;
const WARM_RADIUS: i32 = 12;
const WARM_CADENCE: u64 = 8;
const HYSTERESIS: u64 = 32;
const PROMOTE_TICKS: u64 = 64;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TestRing {
    Cold,
    Warm,
    Hot,
}

struct TestDirector {
    ring: Vec<TestRing>,
    cooling_since: Vec<Option<u64>>,
    promote_until: Vec<u64>,
}

impl TestDirector {
    fn new(agents: usize) -> Self {
        Self {
            ring: vec![TestRing::Cold; agents],
            cooling_since: vec![None; agents],
            promote_until: vec![0; agents],
        }
    }

    fn should_run_soul(&self, slot: usize, tick: u64) -> bool {
        match self.ring[slot] {
            TestRing::Hot => true,
            TestRing::Warm => tick.is_multiple_of(WARM_CADENCE),
            TestRing::Cold => false,
        }
    }

    fn note_event(&mut self, slot: usize, tick: u64) {
        self.promote_until[slot] = tick + PROMOTE_TICKS;
        self.ring[slot] = TestRing::Hot;
        self.cooling_since[slot] = None;
    }

    fn band(pos: (i32, i32)) -> TestRing {
        let distance = (pos.0 - 8).abs().max((pos.1 - 8).abs());
        if distance <= HOT_RADIUS {
            TestRing::Hot
        } else if distance <= WARM_RADIUS {
            TestRing::Warm
        } else {
            TestRing::Cold
        }
    }

    fn update(&mut self, positions: &[(i32, i32)], tick: u64) {
        for (slot, &pos) in positions.iter().enumerate() {
            let mut target = Self::band(pos);
            if tick < self.promote_until[slot] {
                target = TestRing::Hot;
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
                    if tick.saturating_sub(since) >= HYSTERESIS {
                        self.ring[slot] = target;
                        self.cooling_since[slot] = None;
                    }
                }
            }
        }
    }
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

fn direct_step(
    world: &mut World,
    pack: &Rc<VillagePack>,
    soul: &mut HabitSoul<UtilitySoul<VillageBody>>,
    director: &mut TestDirector,
    ids: &[EntityId],
    positions: &mut [(i32, i32)],
    last_events: &mut usize,
) -> u64 {
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
    world.state_hash(&**pack)
}

fn direct_pipeline(config: RunConfig) -> (Vec<u64>, u64) {
    let pack = Rc::new(VillagePack::new());
    let mut world = World::with_pack(config.seed, &*pack);
    let positions = start_positions(config.agents);
    let ids: Vec<EntityId> = positions
        .iter()
        .map(|&position| world.spawn(position))
        .collect();
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
    let mut director = TestDirector::new(ids.len());
    let mut positions = positions;
    let mut last_events = 0;
    let mut hashes = vec![world.state_hash(&*pack)];
    for _ in 0..200 {
        hashes.push(direct_step(
            &mut world,
            &pack,
            &mut soul,
            &mut director,
            &ids,
            &mut positions,
            &mut last_events,
        ));
    }
    (hashes, world.state_hash(&*pack))
}

#[test]
fn direct_and_controller_hashes_match_at_every_checkpoint() {
    let config = RunConfig {
        seed: 0x51_4d_1a,
        agents: 12,
        expertise: ExpertiseLevel::Capable,
    };
    let mut controller = SimulationController::new(config);
    let mut controller_hashes = vec![controller.state_hash()];
    for tick in 1..=200 {
        let outcome = controller.step_one();
        assert_eq!(outcome.tick, tick);
        controller_hashes.push(outcome.state_hash);
    }
    let (direct_hashes, direct_final) = direct_pipeline(config);
    assert_eq!(
        controller_hashes, direct_hashes,
        "direct/controller hash sequences diverged"
    );
    for tick in [0usize, 50, 100, 150, 200] {
        assert_eq!(
            controller_hashes[tick], direct_hashes[tick],
            "diverged at tick {tick}"
        );
    }
    assert_eq!(controller_hashes[200], direct_final);
}

#[test]
fn controllers_are_deterministic_and_seed_sensitive() {
    let config = RunConfig {
        seed: 0x51_4d_1a,
        agents: 12,
        expertise: ExpertiseLevel::Capable,
    };
    let mut first = SimulationController::new(config);
    let mut second = SimulationController::new(config);
    for _ in 0..200 {
        assert_eq!(first.step_one().state_hash, second.step_one().state_hash);
    }
    let final_hash = first.state_hash();
    let mut different = SimulationController::new(RunConfig {
        seed: config.seed + 1,
        ..config
    });
    for _ in 0..200 {
        different.step_one();
    }
    assert_ne!(final_hash, different.state_hash());
    eprintln!(
        "runtime parity seed={} final_hash=0x{final_hash:016x}",
        config.seed
    );
}
