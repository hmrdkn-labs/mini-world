//! The 50-agent village soak: the full observe → decide → validate → execute
//! loop with the village pack, the utility SOUL, and per-character memory wired
//! together at scale, plus a report the gate test checks.
//!
//! Determinism holds end to end: personas come from the seed, decisions from
//! kernel RNG streams, needs/opinions from integer closed-form updates, and the
//! only wall-clock read is the throughput timer — which never feeds sim state.

use std::rc::Rc;
use std::time::Instant;

use mw_agents::habits::{HabitContext, HabitSoul, HabitStats};
use mw_agents::memory::{Memory, OPINION_ONE};
use mw_agents::obs::N_STATS;
use mw_agents::persona::{trait_idx, Persona};
use mw_agents::soul::{Body, Choice, Social, ToolSem, UtilitySoul, TOOL_SLOTS};
use mw_core::{AgentRng, EntityId, Intent, Observation, SoulPolicy, World};
use mw_neural::{encode, NeuralRuntime};
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
    /// Per-tick sum of every agent's projected (hunger, energy, social).
    pub sum_needs: [u128; N_STATS],
    pub habits_enabled: bool,
    pub habit_stats: HabitStats,
    pub habit_cache_sizes: Vec<usize>,
}

impl SoakReport {
    /// Every chosen tool, including cache replays. The UtilitySoul histogram is
    /// updated on both scorer misses and pack-decoded habit hits.
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

    fn tool_for_intent(&self, intent: &Intent) -> Option<u32> {
        match intent {
            Intent::Move { .. } => Some(Action::Move.id()),
            Intent::Speak { .. } => Some(Action::Speak.id()),
            Intent::Interact { verb, .. } => mw_village::decode(*verb).0.map(Action::id),
            Intent::Idle => Some(Action::Idle.id()),
        }
    }
}

type VillageSoul = UtilitySoul<VillageBody>;

enum ActiveSoul {
    Plain(VillageSoul),
    Habits(HabitSoul<VillageSoul>),
    Neural(Box<NeuralSoulAdapter>),
}

/// Batched neural policy adapter. Rich observations and the village intent
/// codec remain owned by `UtilitySoul`; ONNX supplies advisory tool/target
/// argmaxes and a deterministic safety action handles an idle collapse.
struct NeuralSoulAdapter {
    observer: UtilitySoul<VillageBody>,
    runtime: NeuralRuntime,
    pending: Vec<(Intent, u32, Intent, u32)>,
    cursor: usize,
    hist: [u64; TOOL_SLOTS],
}

impl NeuralSoulAdapter {
    fn load(
        runtime: NeuralRuntime,
        ids: Vec<EntityId>,
        personas: Vec<Persona>,
        positions: Vec<(i32, i32)>,
        pack: Rc<VillagePack>,
    ) -> Self {
        let factions = personas.iter().map(Persona::faction).collect();
        let memories = ids
            .iter()
            .map(|&id| Memory::new(id, verb_affect()))
            .collect();
        Self {
            observer: UtilitySoul::new(
                VillageBody::new(pack, factions),
                tool_table(),
                ids,
                personas,
                memories,
                positions,
            ),
            runtime,
            pending: Vec::new(),
            cursor: 0,
            hist: [0; TOOL_SLOTS],
        }
    }

    fn snapshot(&mut self, world: &World) {
        self.observer.snapshot(world);
    }

    fn prepare(&mut self, observations: &[Observation]) -> Result<(), mw_neural::Error> {
        let rich = self.observer.batch_observations(observations);
        if rich.len() != observations.len() {
            return Err(mw_neural::Error::Shape(format!(
                "neural batch has {} rows for {} agents",
                rich.len(),
                observations.len()
            )));
        }
        let rows: Vec<_> = rich
            .iter()
            .enumerate()
            .map(|(slot, obs)| {
                let persona = self.observer.persona_at(slot).ok_or_else(|| {
                    mw_neural::Error::Shape(format!("missing persona for slot {slot}"))
                })?;
                encode(slot as u32, &persona, obs, self.runtime.norm())
            })
            .collect::<Result<_, _>>()?;
        let outputs = self.runtime.infer(&rows)?;
        self.pending = outputs
            .iter()
            .zip(rich.iter())
            .enumerate()
            .map(|(slot, (output, obs))| {
                let fallback_tool = safety_tool(obs, observations[slot].tool_mask, obs.tick, slot);
                let fallback = self
                    .observer
                    .intent_from_tool(slot, obs, fallback_tool, None);
                let neural =
                    self.observer
                        .intent_from_tool(slot, obs, output.tool, output.target_slot);
                (neural, output.tool, fallback, fallback_tool)
            })
            .collect();
        self.cursor = 0;
        Ok(())
    }

    fn observe_events(&mut self, events: &[mw_core::Event]) {
        self.observer.observe_events(events);
    }

    fn decay_opinions(&mut self) {
        self.observer.decay_opinions();
    }

    fn histogram(&self) -> &[u64; TOOL_SLOTS] {
        &self.hist
    }
}

impl SoulPolicy for NeuralSoulAdapter {
    fn decide(&mut self, _obs: &Observation, _rng: &mut AgentRng) -> Intent {
        let (neural_intent, neural_tool, fallback, fallback_tool) =
            self.pending.get(self.cursor).cloned().unwrap_or((
                Intent::Idle,
                Action::Idle.id(),
                Intent::Idle,
                Action::Idle.id(),
            ));
        let (intent, tool) = if neural_tool == Action::Idle.id() {
            (fallback, fallback_tool)
        } else {
            (neural_intent, neural_tool)
        };
        if let Some(count) = self.hist.get_mut(tool as usize) {
            *count += 1;
        }
        self.cursor += 1;
        intent
    }
}

fn safety_tool(obs: &mw_agents::obs::AgentObs, mask: u32, tick: u64, slot: usize) -> u32 {
    if obs.self_stats[0] < 500 && mask & (1 << Action::Eat.id()) != 0 {
        return Action::Eat.id();
    }
    if obs.self_stats[1] < 500 && mask & (1 << Action::Sleep.id()) != 0 {
        return Action::Sleep.id();
    }
    let candidates: Vec<u32> = (0..TOOL_COUNT)
        .filter(|&tool| tool != Action::Idle.id() && mask & (1 << tool) != 0)
        .collect();
    candidates
        .get((tick as usize + slot) % candidates.len().max(1))
        .copied()
        .unwrap_or(Action::Idle.id())
}

impl SoulPolicy for ActiveSoul {
    fn decide(&mut self, obs: &Observation, rng: &mut AgentRng) -> Intent {
        match self {
            Self::Plain(s) => s.decide(obs, rng),
            Self::Habits(s) => s.decide(obs, rng),
            Self::Neural(s) => s.decide(obs, rng),
        }
    }
}

impl ActiveSoul {
    fn snapshot(&mut self, world: &World) {
        match self {
            Self::Plain(s) => s.snapshot(world),
            Self::Habits(s) => s.inner_mut().snapshot(world),
            Self::Neural(s) => s.snapshot(world),
        }
    }

    fn set_context(&mut self, id: EntityId, context: HabitContext) {
        if let Self::Habits(s) = self {
            s.set_context(id, context);
        }
    }

    fn observe_events(&mut self, events: &[mw_core::Event]) {
        match self {
            Self::Plain(s) => s.observe_events(events),
            Self::Habits(s) => {
                s.inner_mut().observe_events(events);
                s.observe_events(events);
            }
            Self::Neural(s) => s.observe_events(events),
        }
    }

    fn decay_opinions(&mut self) {
        match self {
            Self::Plain(s) => s.decay_opinions(),
            Self::Habits(s) => s.inner_mut().decay_opinions(),
            Self::Neural(s) => s.decay_opinions(),
        }
    }

    fn histogram(&self) -> &[u64; TOOL_SLOTS] {
        match self {
            Self::Plain(s) => s.histogram(),
            Self::Habits(s) => s.inner().histogram(),
            Self::Neural(s) => s.histogram(),
        }
    }

    fn habit_stats(&self) -> HabitStats {
        match self {
            Self::Plain(_) | Self::Neural(_) => HabitStats::default(),
            Self::Habits(s) => s.stats(),
        }
    }

    fn cache_sizes(&self) -> Vec<usize> {
        match self {
            Self::Plain(_) | Self::Neural(_) => Vec::new(),
            Self::Habits(s) => s.cache_sizes(),
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

/// Gifts build rapport: the receiver values the giver more than the giver
/// values the act, so memories retain the interaction's direction.
pub fn verb_affect() -> Vec<(u32, i32, i32)> {
    let actor = OPINION_ONE / 4;
    let receiver = OPINION_ONE;
    vec![
        (verb(Action::Give, Item::Food), actor, receiver),
        (verb(Action::Give, Item::Water), actor, receiver),
    ]
}

// The soak's final hash is now the kernel's full canonical hash: it folds the
// village pack's own per-entity state (needs incl. the starvation clock,
// inventories, ground items) via `ScenarioPack::hash_state`, so positions and
// the social/economic state are covered by one hash that replay must reproduce.

/// Run the soak with habits enabled (the default production path).
pub fn run(cfg: SoakConfig) -> SoakReport {
    run_with_habits(cfg, true)
}

/// Run an A/B soak. Habit cache state is simulation state, but elapsed time is
/// deliberately report-only and never influences the hash.
pub fn run_with_habits(cfg: SoakConfig, habits: bool) -> SoakReport {
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
    let utility = UtilitySoul::new(
        body,
        tool_table(),
        ids.clone(),
        personas,
        memories,
        positions,
    );
    let mut soul = if habits {
        ActiveSoul::Habits(HabitSoul::with_hit_hook_and_tool(
            utility,
            ids.clone(),
            UtilitySoul::<VillageBody>::habit_replay_tool,
            UtilitySoul::<VillageBody>::last_tool,
        ))
    } else {
        ActiveSoul::Plain(utility)
    };

    let mut last_events = 0usize;
    let mut sum_needs = [0u128; N_STATS];
    let t0 = Instant::now();
    for _ in 0..cfg.ticks {
        for &id in &ids {
            let (h, en, so) = pack.needs(id).project(world.tick());
            let pos = world.entity(id).map(|e| e.pos).unwrap_or_default();
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
                    needs: [h as i16, en as i16, so as i16],
                    need_max: MAX_NEED as i16,
                    cell_class,
                    goal: 0,
                },
            );
        }
        soul.snapshot(&world);
        world.step(&*pack, &mut soul);
        let events = world.event_log();
        soul.observe_events(&events[last_events..]);
        last_events = events.len();
        soul.decay_opinions();
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
    let final_hash = world.state_hash(&*pack);
    SoakReport {
        cfg,
        histogram: *soul.histogram(),
        deaths,
        final_hash,
        elapsed_secs,
        sum_needs,
        habits_enabled: habits,
        habit_stats: soul.habit_stats(),
        habit_cache_sizes: soul.cache_sizes(),
    }
}

#[allow(dead_code)]
struct NeuralRun {
    report: SoakReport,
    world: World,
    pack: Rc<VillagePack>,
    positions: Vec<(i32, i32)>,
}

pub fn run_neural(
    cfg: SoakConfig,
    model_path: impl AsRef<std::path::Path>,
) -> Result<SoakReport, mw_neural::Error> {
    Ok(run_neural_logged(cfg, model_path)?.report)
}

fn run_neural_logged(
    cfg: SoakConfig,
    model_path: impl AsRef<std::path::Path>,
) -> Result<NeuralRun, mw_neural::Error> {
    let model_path = model_path.as_ref();
    let norm_path = model_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("training/artifacts"))
        .join("norm_stats.json");
    let runtime = NeuralRuntime::load(model_path, norm_path)?;
    let pack = Rc::new(VillagePack::new());
    let mut world = World::with_pack(cfg.seed, &*pack);
    let positions = start_positions(cfg.agents);
    let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
    let personas: Vec<Persona> = ids.iter().map(|&id| Persona::new(cfg.seed, id)).collect();
    let mut soul = ActiveSoul::Neural(Box::new(NeuralSoulAdapter::load(
        runtime,
        ids.clone(),
        personas,
        positions.clone(),
        Rc::clone(&pack),
    )));
    let mut last_events = 0usize;
    let mut sum_needs = [0u128; N_STATS];
    let t0 = Instant::now();
    for _ in 0..cfg.ticks {
        soul.snapshot(&world);
        let observations: Vec<Observation> = ids
            .iter()
            .map(|&id| {
                let mut obs = world.observe(id);
                obs.tool_mask = mw_core::ScenarioPack::afforded_tools(&*pack, &world, id, &obs);
                obs
            })
            .collect();
        if let ActiveSoul::Neural(neural) = &mut soul {
            neural.prepare(&observations)?;
        }
        world.step(&*pack, &mut soul);
        let events = world.event_log();
        soul.observe_events(&events[last_events..]);
        last_events = events.len();
        soul.decay_opinions();
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
    let final_hash = world.state_hash(&*pack);
    Ok(NeuralRun {
        report: SoakReport {
            cfg,
            histogram: *soul.histogram(),
            deaths,
            final_hash,
            elapsed_secs,
            sum_needs,
            habits_enabled: false,
            habit_stats: HabitStats::default(),
            habit_cache_sizes: Vec::new(),
        },
        world,
        pack,
        positions,
    })
}

#[cfg(test)]
mod neural_tests {
    use super::*;

    const MODEL: &str = "../../training/artifacts/model.onnx";

    #[test]
    fn neural_same_seed_same_final_hash() {
        let cfg = SoakConfig {
            seed: 7,
            agents: 8,
            ticks: 40,
        };
        assert_eq!(
            run_neural(cfg, MODEL).unwrap().final_hash,
            run_neural(cfg, MODEL).unwrap().final_hash
        );
    }

    #[test]
    fn neural_intent_log_replay_reproduces_hash() {
        let cfg = SoakConfig {
            seed: 17,
            agents: 8,
            ticks: 40,
        };
        let run = run_neural_logged(cfg, MODEL).unwrap();
        let log = run.world.intent_log().to_vec();
        let replay_pack = VillagePack::new();
        let replay = World::replay(cfg.seed, &run.positions, cfg.ticks, &log, &replay_pack);
        assert_eq!(
            run.world.state_hash(&*run.pack),
            replay.state_hash(&replay_pack)
        );
    }
}
