use mw_agents::habits::{HabitContext, HabitSoul};
use mw_agents::obs::{AgentObs, K_NEIGHBORS};
use mw_agents::persona::Persona;
use mw_agents::soul::{DecisionTrace, UtilitySoul};
use mw_core::{EntityId, Event, Intent, LogEntry, SoulPolicy, World};
use mw_village::{tile_at, Action, Tile, VillagePack, MAX_NEED};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportProfile {
    Healthy,
    Scarcity,
    Hostile,
    Exhausted,
}

impl ExportProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Scarcity => "scarcity",
            Self::Hostile => "hostile",
            Self::Exhausted => "exhausted",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrajectoryExportConfig {
    pub habits: bool,
    pub include_replays: bool,
    pub profile: ExportProfile,
    pub fraction: u8,
}

impl Default for TrajectoryExportConfig {
    fn default() -> Self {
        Self {
            habits: true,
            include_replays: false,
            profile: ExportProfile::Healthy,
            fraction: 25,
        }
    }
}
pub const OUTCOME_WINDOW: u64 = 8;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonaRecord {
    pub traits: [i16; 5],
    pub need_weights: [i16; 3],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeighborRecord {
    pub present: bool,
    pub dist2: i32,
    pub opinion: i32,
    pub faction: u8,
    pub kind: u8,
    pub id_slot: Option<u32>,
    /// Absolute position retained for pointer-target reconstruction.
    pub pos: [i32; 2],
    /// Relative position consumed by the scorer.
    pub rel_pos: [i32; 2],
    pub cell_class: u8,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsRecord {
    pub tick: u64,
    pub self_stats: [i16; 3],
    pub self_pos: [i32; 2],
    pub self_cell_class: u8,
    pub neighbors: [NeighborRecord; K_NEIGHBORS],
    pub events: [u16; 4],
    pub tool_mask: u32,
    pub goal: u8,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub tool: String,
    pub target_slot: Option<u32>,
    pub params: serde_json::Value,
    /// Difference between the best and second-best afforded-tool scores.
    /// `i32::MAX` marks records without two scorer candidates (e.g. replays).
    pub score_margin: i32,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub tick: u64,
    pub kind: String,
    pub actor_slot: u32,
    pub target_slot: Option<u32>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub events: Vec<EventRecord>,
    pub need_deltas: [i32; 3],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayIntentRecord {
    pub tick: u64,
    pub actor_slot: u32,
    pub intent: serde_json::Value,
}

/// Information sufficient to identify and audit the exact pre-action state.
///
/// The validated intent prefix is retained rather than re-running a policy:
/// this is the kernel's replay contract.  `replay_hash` is the canonical
/// `World::state_hash` immediately before the exported decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayProvenance {
    pub version: u32,
    pub profile: String,
    pub fraction: u8,
    pub agent_count: u32,
    pub initial_positions: Vec<[i32; 2]>,
    pub replay_log: Vec<ReplayIntentRecord>,
    pub replay_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Canonical entity positions and pack state immediately before the action.
    pub positions: Vec<[i32; 2]>,
    pub needs: Vec<[i32; 3]>,
    pub inventory: Vec<[u8; 2]>,
    pub ground: Vec<[u8; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub schema_version: u32,
    pub seed: u64,
    pub tick: u64,
    pub agent_slot: u32,
    pub persona: PersonaRecord,
    pub obs: ObsRecord,
    pub afforded_mask: u32,
    pub decision: DecisionRecord,
    pub outcome: OutcomeRecord,
    pub replay: bool,
    pub replay_provenance: ReplayProvenance,
    pub state_snapshot: StateSnapshot,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportStats {
    pub records: u64,
    pub per_tool: [u64; mw_agents::soul::TOOL_SLOTS],
    pub bytes: u64,
    pub hash: u64,
    pub final_hash: u64,
}
/// Conservative preflight ceiling for JSONL trajectory exports.
///
/// Replay provenance and state snapshots are repeated on every selected record,
/// so an otherwise modest `(agents, ticks)` request can become quadratic in
/// serialized bytes. Reject it before creating the output file.
pub const MAX_ESTIMATED_EXPORT_BYTES: u64 = 512 * 1024 * 1024;

fn estimated_export_bytes(agents: u64, ticks: u64) -> Option<u64> {
    let records = agents.checked_mul(ticks)?;
    let per_record = 16_384u64
        .checked_add(agents.checked_mul(512)?)?
        .checked_add(records.checked_mul(64)?)?;
    records.checked_mul(per_record)
}
struct Pending {
    record: TrajectoryRecord,
    before_needs: [i32; 3],
    events: Vec<EventRecord>,
}
struct HashWriter<W> {
    inner: W,
    hash: u64,
    bytes: u64,
}
impl<W: Write> HashWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: 0xcbf29ce484222325,
            bytes: 0,
        }
    }
}
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(b)?;
        for &x in &b[..n] {
            self.hash ^= x as u64;
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
fn obs_record(o: AgentObs) -> ObsRecord {
    ObsRecord {
        tick: o.tick,
        self_stats: o.self_stats,
        self_pos: [o.self_pos.0, o.self_pos.1],
        self_cell_class: o.self_cell_class,
        neighbors: o.neighbors.map(|n| NeighborRecord {
            present: n.present,
            dist2: n.dist2,
            opinion: n.opinion,
            faction: n.faction,
            kind: n.kind,
            id_slot: n.id.map(|x| x.index()),
            pos: [n.pos.0, n.pos.1],
            rel_pos: [n.rel_pos.0, n.rel_pos.1],
            cell_class: n.cell_class,
        }),
        events: o.events,
        tool_mask: o.tool_mask,
        goal: o.goal,
    }
}
fn decision_record(t: &DecisionTrace) -> DecisionRecord {
    let target_slot = t.choice.target.map(|x| x.index());
    let params = match &t.intent {
        Intent::Move { dx, dy } => serde_json::json!({"dx":dx,"dy":dy}),
        Intent::Interact { verb, .. } => serde_json::json!({"verb":verb}),
        Intent::Speak { act, topic, .. } => serde_json::json!({"act":act,"topic":topic}),
        Intent::Idle => serde_json::json!({}),
    };
    let tool = Action::from_id(t.choice.tool)
        .map(|a| format!("{a:?}").to_lowercase())
        .unwrap_or_else(|| format!("tool_{}", t.choice.tool));
    DecisionRecord {
        tool,
        target_slot,
        params,
        score_margin: t.score_margin,
    }
}
fn event_record(e: &Event) -> EventRecord {
    match *e {
        Event::Moved { tick, actor, .. } => EventRecord {
            tick,
            kind: "moved".into(),
            actor_slot: actor.index(),
            target_slot: None,
        },
        Event::Interacted {
            tick,
            actor,
            target,
            ..
        } => EventRecord {
            tick,
            kind: "interacted".into(),
            actor_slot: actor.index(),
            target_slot: Some(target.index()),
        },
        Event::Spoke {
            tick,
            actor,
            target,
            ..
        } => EventRecord {
            tick,
            kind: "spoke".into(),
            actor_slot: actor.index(),
            target_slot: Some(target.index()),
        },
        Event::Rejected { tick, actor, .. } => EventRecord {
            tick,
            kind: "rejected".into(),
            actor_slot: actor.index(),
            target_slot: None,
        },
    }
}
fn replay_intent_record(entry: &LogEntry) -> Option<ReplayIntentRecord> {
    let LogEntry::Intent(logged) = entry else {
        return None;
    };
    let intent = match &logged.intent {
        Intent::Move { dx, dy } => serde_json::json!({"kind":"move","dx":dx,"dy":dy}),
        Intent::Interact { target, verb } => {
            serde_json::json!({"kind":"interact","target_slot":target.index(),"verb":verb})
        }
        Intent::Speak { target, act, topic } => serde_json::json!({
            "kind":"speak","target_slot":target.index(),"act":act,"topic":topic
        }),
        Intent::Idle => serde_json::json!({"kind":"idle"}),
    };
    Some(ReplayIntentRecord {
        tick: logged.tick,
        actor_slot: logged.actor.index(),
        intent,
    })
}

fn replay_provenance(
    world: &World,
    t: u64,
    positions: &[(i32, i32)],
    profile: ExportProfile,
    fraction: u8,
    replay_hash: u64,
) -> ReplayProvenance {
    ReplayProvenance {
        version: 1,
        profile: profile.as_str().into(),
        fraction,
        agent_count: positions.len() as u32,
        initial_positions: positions.iter().map(|&(x, y)| [x, y]).collect(),
        replay_log: world
            .intent_log()
            .iter()
            .filter(|entry| matches!(entry, LogEntry::Intent(l) if l.tick < t))
            .filter_map(replay_intent_record)
            .collect(),
        replay_hash,
    }
}
fn state_snapshot(pack: &VillagePack, world: &World, ids: &[EntityId], tick: u64) -> StateSnapshot {
    StateSnapshot {
        positions: ids
            .iter()
            .map(|&id| {
                let pos = world.entity(id).map(|e| e.pos).unwrap_or_default();
                [pos.0, pos.1]
            })
            .collect(),
        needs: ids.iter().map(|&id| needs(pack, id, tick)).collect(),
        inventory: ids
            .iter()
            .map(|&id| {
                [
                    pack.inventory(id, mw_village::Item::Food),
                    pack.inventory(id, mw_village::Item::Water),
                ]
            })
            .collect(),
        ground: ids
            .iter()
            .map(|&id| {
                let pos = world.entity(id).map(|e| e.pos).unwrap_or_default();
                let items = pack.ground_at(pos);
                [
                    items[mw_village::Item::Food as usize],
                    items[mw_village::Item::Water as usize],
                ]
            })
            .collect(),
    }
}

/// Reconstruct and hash the exact pre-action state represented by one export.
pub fn replay_provenance_hash(seed: u64, record: &TrajectoryRecord) -> Result<u64, String> {
    let profile = match record.replay_provenance.profile.as_str() {
        "healthy" => ExportProfile::Healthy,
        "scarcity" => ExportProfile::Scarcity,
        "hostile" => ExportProfile::Hostile,
        "exhausted" => ExportProfile::Exhausted,
        other => return Err(format!("unknown profile {other}")),
    };
    let pack = VillagePack::new();
    let positions: Vec<(i32, i32)> = record
        .replay_provenance
        .initial_positions
        .iter()
        .map(|p| (p[0], p[1]))
        .collect();
    let mut world = World::with_pack(seed, &pack);
    let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
    seed_profile(&pack, &ids, seed, profile, record.replay_provenance.fraction);
    let mut log = Vec::with_capacity(record.replay_provenance.replay_log.len());
    for item in &record.replay_provenance.replay_log {
        let actor = ids
            .get(item.actor_slot as usize)
            .copied()
            .ok_or_else(|| format!("actor slot {} out of range", item.actor_slot))?;
        let obj = item.intent.as_object().ok_or_else(|| "replay intent is not an object".to_string())?;
        let kind = obj.get("kind").and_then(serde_json::Value::as_str)
            .ok_or_else(|| "replay intent has no kind".to_string())?;
        let intent = match kind {
            "move" => Intent::Move {
                dx: obj.get("dx").and_then(serde_json::Value::as_i64).unwrap_or(0) as i32,
                dy: obj.get("dy").and_then(serde_json::Value::as_i64).unwrap_or(0) as i32,
            },
            "interact" => Intent::Interact {
                target: EntityId::from_parts(obj.get("target_slot")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| "interact intent has no target".to_string())? as u32, 0),
                verb: obj.get("verb").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
            },
            "speak" => Intent::Speak {
                target: EntityId::from_parts(obj.get("target_slot")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| "speak intent has no target".to_string())? as u32, 0),
                act: obj.get("act").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
                topic: obj.get("topic").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
            },
            "idle" => Intent::Idle,
            other => return Err(format!("unknown replay intent {other}")),
        };
        log.push(LogEntry::Intent(mw_core::LoggedIntent { tick: item.tick, actor, intent }));
    }
    let replayed = World::replay(seed, &positions, record.tick, &log, &pack);
    Ok(replayed.state_hash(&pack))
}
fn needs(p: &VillagePack, id: EntityId, t: u64) -> [i32; 3] {
    let n = p.needs(id).project(t);
    [n.0, n.1, n.2]
}
fn affects(e: &EventRecord, s: u32) -> bool {
    e.actor_slot == s || e.target_slot == Some(s)
}
fn persona_record(p: Persona) -> PersonaRecord {
    PersonaRecord {
        traits: p.traits,
        need_weights: p.weights,
    }
}

enum ExportSoul {
    Plain(UtilitySoul<crate::soak::VillageBody>),
    Habits(HabitSoul<UtilitySoul<crate::soak::VillageBody>>),
}
impl ExportSoul {
    fn snapshot(&mut self, w: &World) {
        match self {
            Self::Plain(s) => s.snapshot(w),
            Self::Habits(s) => s.inner_mut().snapshot(w),
        }
    }
    fn set_context(&mut self, id: EntityId, c: HabitContext) {
        if let Self::Habits(s) = self {
            s.set_context(id, c)
        }
    }
    fn observe_events(&mut self, e: &[Event]) {
        match self {
            Self::Plain(s) => s.observe_events(e),
            Self::Habits(s) => {
                s.inner_mut().observe_events(e);
                s.observe_events(e)
            }
        }
    }
    fn decay(&mut self) {
        match self {
            Self::Plain(s) => s.decay_opinions(),
            Self::Habits(s) => s.inner_mut().decay_opinions(),
        }
    }
    fn traces(&mut self) -> Vec<DecisionTrace> {
        match self {
            Self::Plain(s) => s.drain_traces(),
            Self::Habits(s) => s.inner_mut().drain_traces(),
        }
    }
}
impl SoulPolicy for ExportSoul {
    fn decide(&mut self, o: &mw_core::Observation, r: &mut mw_core::AgentRng) -> Intent {
        match self {
            Self::Plain(s) => s.decide(o, r),
            Self::Habits(s) => s.decide(o, r),
        }
    }
}
fn make_policy(
    pack: Rc<VillagePack>,
    ids: &[EntityId],
    ps: &[Persona],
    positions: &[(i32, i32)],
    habits: bool,
) -> ExportSoul {
    let factions = ps.iter().map(Persona::faction).collect();
    let memories = ids
        .iter()
        .map(|&id| mw_agents::memory::Memory::new(id, crate::soak::verb_affect()))
        .collect();
    let mut u = UtilitySoul::new(
        crate::soak::VillageBody::new(pack, factions),
        crate::soak::tool_table(),
        ids.to_vec(),
        ps.to_vec(),
        memories,
        positions.to_vec(),
    );
    u.enable_telemetry();
    if habits {
        ExportSoul::Habits(HabitSoul::with_hit_hook_and_tool_and_trace(
            u,
            ids.to_vec(),
            UtilitySoul::<crate::soak::VillageBody>::habit_replay_tool,
            UtilitySoul::<crate::soak::VillageBody>::last_tool,
            UtilitySoul::<crate::soak::VillageBody>::record_replay,
        ))
    } else {
        ExportSoul::Plain(u)
    }
}
fn tool_id(r: &TrajectoryRecord) -> usize {
    match r.decision.tool.as_str() {
        "move" => 0,
        "eat" => 1,
        "sleep" => 2,
        "work" => 3,
        "speak" => 4,
        "give" => 5,
        "pickup" => 6,
        "use" => 7,
        "follow" => 8,
        "flee" => 9,
        "idle" => 10,
        "drop" => 11,
        _ => 0,
    }
}
fn seed_profile(
    pack: &VillagePack,
    ids: &[EntityId],
    seed: u64,
    profile: ExportProfile,
    fraction: u8,
) {
    let fraction = fraction.min(100) as u64;
    match profile {
        ExportProfile::Healthy | ExportProfile::Hostile => {}
        ExportProfile::Scarcity => {
            for (slot, &id) in ids.iter().enumerate() {
                let jitter = seed
                    .wrapping_add(slot as u64 * 0x9e37_79b9)
                    .rotate_left((slot % 31) as u32)
                    % 380;
                pack.seed_needs(id, [120 + jitter as i32, 700, 650]);
            }
        }
        ExportProfile::Exhausted => {
            let cohort = if fraction == 0 {
                0
            } else {
                ((ids.len() as u64 * fraction).saturating_add(99) / 100) as usize
            };
            for (slot, &id) in ids.iter().take(cohort).enumerate() {
                let jitter = seed.wrapping_add(slot as u64 * 0x517c_c1b7) % 260;
                pack.seed_needs(id, [700, 80 + jitter as i32, 650]);
            }
        }
    }
}

fn seed_hostility(soul: &mut ExportSoul, ids: &[EntityId], fraction: u8) {
    let fraction = fraction.min(100) as usize;
    let pair_count = if fraction == 0 {
        0
    } else {
        ((ids.len() / 2) * fraction).div_ceil(100)
    };
    let mut events = Vec::with_capacity(pair_count * 2);
    for pair in 0..pair_count.min(ids.len() / 2) {
        let a = ids[pair * 2];
        let b = ids[pair * 2 + 1];
        events.push(Event::Spoke {
            tick: 0,
            actor: a,
            target: b,
            act: 5,
            topic: 0,
        });
        events.push(Event::Spoke {
            tick: 0,
            actor: b,
            target: a,
            act: 5,
            topic: 0,
        });
    }
    soul.observe_events(&events);
}

pub fn export_trajectories(
    seed: u64,
    agents: i32,
    ticks: u64,
    out: &str,
    habits: bool,
    include_replays: bool,
) -> std::io::Result<ExportStats> {
    export_trajectories_profile(
        seed,
        agents,
        ticks,
        out,
        TrajectoryExportConfig {
            habits,
            include_replays,
            profile: ExportProfile::Healthy,
            fraction: 25,
        },
    )
}

pub fn export_trajectories_profile(
    seed: u64,
    agents: i32,
    ticks: u64,
    out: &str,
    config: TrajectoryExportConfig,
) -> std::io::Result<ExportStats> {
    let TrajectoryExportConfig {
        habits,
        include_replays,
        profile,
        fraction,
    } = config;
    let agents = agents.max(1);
    let estimated = estimated_export_bytes(agents as u64, ticks).unwrap_or(u64::MAX);
    if estimated > MAX_ESTIMATED_EXPORT_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "trajectory export plan exceeds {} MiB (agents={agents}, ticks={ticks}, estimated_bytes={estimated})",
                MAX_ESTIMATED_EXPORT_BYTES / (1024 * 1024)
            ),
        ));
    }
    let pack = Rc::new(VillagePack::new());
    let mut world = World::with_pack(seed, &*pack);
    let positions = crate::soak::start_positions(agents);
    let ids: Vec<_> = positions.iter().map(|&p| world.spawn(p)).collect();
    seed_profile(&pack, &ids, seed, profile, fraction);
    let personas: Vec<_> = ids.iter().map(|&id| Persona::new(seed, id)).collect();
    let mut soul = make_policy(Rc::clone(&pack), &ids, &personas, &positions, habits);
    if profile == ExportProfile::Hostile {
        seed_hostility(&mut soul, &ids, fraction);
    }
    let file = File::create(out)?;
    let mut w = HashWriter::new(BufWriter::new(file));
    let mut pending = VecDeque::new();
    let mut st = ExportStats {
        records: 0,
        per_tool: [0; mw_agents::soul::TOOL_SLOTS],
        bytes: 0,
        hash: 0,
        final_hash: 0,
    };
    let mut last_events = 0usize;
    for _ in 0..ticks {
        let t = world.tick();
        let mut before = Vec::with_capacity(ids.len());
        let pre_state_hash = world.state_hash(&*pack);
        let pre_snapshot = state_snapshot(&pack, &world, &ids, t);
        for &id in &ids {
            let n = needs(&pack, id, t);
            before.push(n);
            let pos = world.entity(id).map(|e| e.pos).unwrap_or_default();
            let cell = match tile_at(pos) {
                Tile::Empty => 0,
                Tile::Home => 1,
                Tile::Bakery => 2,
                Tile::Well => 3,
                Tile::Field => 4,
            };
            soul.set_context(
                id,
                HabitContext {
                    needs: n.map(|v| v as i16),
                    need_max: MAX_NEED as i16,
                    cell_class: cell,
                    goal: 0,
                },
            );
        }
        soul.snapshot(&world);
        world.step(&*pack, &mut soul);
        let new_events = &world.event_log()[last_events..];
        soul.observe_events(new_events);
        last_events = world.event_log().len();
        for (slot, tr) in soul.traces().into_iter().enumerate() {
            if include_replays || !tr.replay {
                pending.push_back(Pending {
                    record: TrajectoryRecord {
                        schema_version: SCHEMA_VERSION,
                        seed,
                        tick: t,
                        agent_slot: slot as u32,
                        persona: persona_record(personas[slot]),
                        obs: obs_record(tr.obs),
                        afforded_mask: tr.obs.tool_mask,
                        decision: decision_record(&tr),
                        outcome: OutcomeRecord {
                            events: Vec::new(),
                            need_deltas: [0; 3],
                        },
                        replay: tr.replay,
                        replay_provenance: replay_provenance(
                            &world,
                            t,
                            &positions,
                            profile,
                            fraction,
                            pre_state_hash,
                        ),
                        state_snapshot: pre_snapshot.clone(),
                    },
                    before_needs: before[slot],
                    events: Vec::new(),
                });
            }
        }
        for e in new_events.iter().map(event_record) {
            for p in pending.iter_mut() {
                if e.tick > p.record.tick
                    && e.tick <= p.record.tick + OUTCOME_WINDOW
                    && affects(&e, p.record.agent_slot)
                {
                    p.events.push(e.clone())
                }
            }
        }
        let now = world.tick();
        while pending
            .front()
            .is_some_and(|p| now > p.record.tick + OUTCOME_WINDOW)
        {
            let mut p = pending.pop_front().unwrap();
            let a = needs(
                &pack,
                ids[p.record.agent_slot as usize],
                p.record.tick + OUTCOME_WINDOW,
            );
            p.record.outcome = OutcomeRecord {
                events: p.events,
                need_deltas: [
                    a[0] - p.before_needs[0],
                    a[1] - p.before_needs[1],
                    a[2] - p.before_needs[2],
                ],
            };
            serde_json::to_writer(&mut w, &p.record).expect("JSONL serialization");
            w.write_all(b"\n")?;
            st.records += 1;
            st.per_tool[tool_id(&p.record)] += 1;
        }
        soul.decay();
    }
    while let Some(mut p) = pending.pop_front() {
        let end = world.tick().min(p.record.tick + OUTCOME_WINDOW);
        let a = needs(&pack, ids[p.record.agent_slot as usize], end);
        p.record.outcome = OutcomeRecord {
            events: p.events,
            need_deltas: [
                a[0] - p.before_needs[0],
                a[1] - p.before_needs[1],
                a[2] - p.before_needs[2],
            ],
        };
        serde_json::to_writer(&mut w, &p.record).expect("JSONL serialization");
        w.write_all(b"\n")?;
        st.records += 1;
        st.per_tool[tool_id(&p.record)] += 1;
    }
    w.flush()?;
    st.bytes = w.bytes;
    st.hash = w.hash;
    st.final_hash = world.state_hash(&*pack);
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn schema_json_round_trip() {
        let r = TrajectoryRecord {
            schema_version: SCHEMA_VERSION,
            seed: 7,
            tick: 3,
            agent_slot: 1,
            persona: PersonaRecord {
                traits: [1, 2, 3, 4, 5],
                need_weights: [6, 7, 8],
            },
            state_snapshot: StateSnapshot {
                positions: vec![[0, 0], [1, 0], [2, 0]],
                needs: vec![[900, 800, 700]; 3],
                inventory: vec![[0, 0]; 3],
                ground: vec![[0, 0]; 3],
            },
            obs: ObsRecord {
                tick: 3,
                self_stats: [900, 800, 700],
                self_pos: [2, 3],
                self_cell_class: 1,
                neighbors: std::array::from_fn(|_| NeighborRecord {
                    present: false,
                    dist2: 0,
                    opinion: 0,
                    faction: 0,
                    kind: 0,
                    id_slot: None,
                    pos: [0, 0],
                    rel_pos: [0, 0],
                    cell_class: 0,
                }),
                events: [0; 4],
                tool_mask: 3,
                goal: 1,
            },
            afforded_mask: 3,
            decision: DecisionRecord {
                tool: "move".into(),
                target_slot: None,
                params: serde_json::json!({"dx":1,"dy":0}),
                score_margin: 42,
            },
            outcome: OutcomeRecord {
                events: Vec::new(),
                need_deltas: [-2, -1, -3],
            },
            replay: false,
            replay_provenance: ReplayProvenance {
                version: 1,
                profile: "healthy".into(),
                fraction: 25,
                agent_count: 3,
                initial_positions: vec![[0, 0], [1, 0], [2, 0]],
                replay_log: Vec::new(),
                replay_hash: 0,
            },
        };
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(r, serde_json::from_str(&s).unwrap())
    }
    #[test]
    fn same_seed_exports_byte_identical() {
        let a = std::env::temp_dir().join("mw-ta.jsonl");
        let b = std::env::temp_dir().join("mw-tb.jsonl");
        let x = export_trajectories(19, 3, 20, a.to_str().unwrap(), false, false).unwrap();
        let y = export_trajectories(19, 3, 20, b.to_str().unwrap(), false, false).unwrap();
        assert_eq!(x.hash, y.hash);
        assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
        let _ = fs::remove_file(a);
        let _ = fs::remove_file(b);
    }
    #[test]
    fn exported_pre_action_state_replays_to_recorded_hash() {
        let path = std::env::temp_dir().join("mw-replay-provenance.jsonl");
        export_trajectories_profile(
            23,
            3,
            4,
            path.to_str().unwrap(),
            TrajectoryExportConfig {
                habits: false,
                include_replays: false,
                profile: ExportProfile::Scarcity,
                fraction: 25,
            },
        )
        .unwrap();
        for line in fs::read_to_string(&path).unwrap().lines() {
            let record: TrajectoryRecord = serde_json::from_str(line).unwrap();
            assert_eq!(
                replay_provenance_hash(23, &record).unwrap(),
                record.replay_provenance.replay_hash,
                "tick {} slot {} failed replay parity",
                record.tick,
                record.agent_slot
            );
        }
        let _ = fs::remove_file(path);
    }
    #[test]
    fn outcome_window_uses_eight_ticks() {
        let p = std::env::temp_dir().join("mw-tw.jsonl");
        export_trajectories(1, 1, 9, p.to_str().unwrap(), false, false).unwrap();
        let first: TrajectoryRecord =
            serde_json::from_str(fs::read_to_string(&p).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!(first.outcome.need_deltas, [-16, -8, -24]);
        let _ = fs::remove_file(p);
    }
    #[test]
    fn stress_profiles_seed_expected_observations() {
        let scarcity = std::env::temp_dir().join("mw-scarcity.jsonl");
        export_trajectories_profile(
            41,
            8,
            2,
            scarcity.to_str().unwrap(),
            TrajectoryExportConfig {
                habits: false,
                include_replays: false,
                profile: ExportProfile::Scarcity,
                fraction: 25,
            },
        )
        .unwrap();
        let first: TrajectoryRecord = serde_json::from_str(
            fs::read_to_string(&scarcity)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert!(first.obs.self_stats.iter().any(|&v| v < 500));
        let _ = fs::remove_file(scarcity);

        let hostile = std::env::temp_dir().join("mw-hostile.jsonl");
        export_trajectories_profile(
            41,
            8,
            2,
            hostile.to_str().unwrap(),
            TrajectoryExportConfig {
                habits: false,
                include_replays: false,
                profile: ExportProfile::Hostile,
                fraction: 100,
            },
        )
        .unwrap();
        let hostile_first: TrajectoryRecord = serde_json::from_str(
            fs::read_to_string(&hostile)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert!(hostile_first
            .obs
            .neighbors
            .iter()
            .any(|n| n.present && n.opinion < 0));
        let _ = fs::remove_file(hostile);

        let exhausted = std::env::temp_dir().join("mw-exhausted.jsonl");
        export_trajectories_profile(
            41,
            8,
            2,
            exhausted.to_str().unwrap(),
            TrajectoryExportConfig {
                habits: false,
                include_replays: false,
                profile: ExportProfile::Exhausted,
                fraction: 50,
            },
        )
        .unwrap();
        let exhausted_first: TrajectoryRecord = serde_json::from_str(
            fs::read_to_string(&exhausted)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert!(exhausted_first.obs.self_stats[1] < 500);
        let _ = fs::remove_file(exhausted);
    }
    #[test]
    fn rejects_quadratic_export_before_creating_output() {
        let path = std::env::temp_dir().join("mw-pathological-export.jsonl");
        let error = export_trajectories(17, 50, 300, path.to_str().unwrap(), false, false)
            .expect_err("50x300 replay export must fail preflight");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!path.exists());
    }
}

#[cfg(test)]
mod different_seed_test {
    use super::export_trajectories;
    use std::fs;

    #[test]
    fn different_seed_changes_export() {
        let a = std::env::temp_dir().join("mw-ts-a.jsonl");
        let b = std::env::temp_dir().join("mw-ts-b.jsonl");
        let x = export_trajectories(31, 2, 12, a.to_str().unwrap(), false, false).unwrap();
        let y = export_trajectories(32, 2, 12, b.to_str().unwrap(), false, false).unwrap();
        assert_ne!(x.hash, y.hash);
        let _ = fs::remove_file(a);
        let _ = fs::remove_file(b);
    }
}
