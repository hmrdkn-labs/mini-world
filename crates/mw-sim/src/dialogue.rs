//! Latent-dialogue wiring: bridges the kernel + village pack + agent memory to
//! the dialogue ledger, and adapts the real TEXT backend to the renderer seam.
//!
//! The [`Scene`] harness scripts exact speak acts through the deterministic
//! replay path, feeds the resulting kernel events into both per-character memory
//! and the [`ConversationLog`], and gates rendering on a [`FocusPoint`]. It is
//! the shared body of the two gate tests (a live one against `llama-server`, a
//! deterministic one against [`MockRenderer`]).

use mw_agents::dialogue::{
    ConversationLog, DialogueRenderer, FocusPoint, RenderRequest, SliceRegistry, Vocab,
};
use mw_agents::memory::Memory;
use mw_agents::persona::Persona;
use mw_core::{EntityId, Intent, LoggedIntent, World};
use mw_text::{LlamaServerBackend, PromptSpec};

/// Adapts the real [`LlamaServerBackend`] to the [`DialogueRenderer`] seam:
/// resolves the request's real name/persona/act/topic into a [`PromptSpec`] and
/// keys the KV slot by conversation. A failed render yields an empty line — TEXT
/// is advisory and never on the tick path.
pub struct LlamaDialogue<'a> {
    pub backend: &'a LlamaServerBackend,
}

impl DialogueRenderer for LlamaDialogue<'_> {
    fn render(&self, req: &RenderRequest<'_>) -> String {
        // Fold the listener's real name into the scene so the line addresses a
        // named counterpart, not a numeric code.
        let context = format!("{} You are speaking to {}.", req.context, req.listener.name);
        let spec = PromptSpec {
            persona: &req.speaker.summary,
            act: req.act,
            topic: req.topic,
            context: &context,
        };
        self.backend
            .render_line(&spec, req.conversation)
            .map(|r| r.text)
            .unwrap_or_default()
    }
}

/// One scripted speak act: `speaker` says `act` about `topic` to `listener` on
/// `tick`. Indices refer to the scene's entity list.
#[derive(Clone, Copy, Debug)]
pub struct Script {
    pub tick: u64,
    pub speaker: usize,
    pub listener: usize,
    pub act: u32,
    pub topic: u32,
}

/// A scripted latent-dialogue scene: a fully wired sim slice plus its dialogue
/// ledger. Deterministic in `(seed, positions, scripts)`.
pub struct Scene {
    pub world: World,
    pub pack: mw_village::VillagePack,
    pub ids: Vec<EntityId>,
    pub personas: Vec<Persona>,
    pub memories: Vec<Memory>,
    pub registry: SliceRegistry,
    pub vocab: Vocab,
    pub log: ConversationLog,
}

impl Scene {
    /// Build a scene, run the scripted speak acts through the deterministic
    /// replay path, then feed the kernel events into memory (both parties) and
    /// the conversation log. No text is rendered here — that is the latent step.
    pub fn script(seed: u64, positions: &[(i32, i32)], scripts: &[Script]) -> Self {
        // Mint stable entity ids from a bare world (no pack, so the real pack's
        // state stays pristine); replay re-spawns in the same order and mints the
        // identical ids.
        let mut scratch = World::new(seed);
        let ids: Vec<EntityId> = positions.iter().map(|&p| scratch.spawn(p)).collect();

        let last_tick = scripts.iter().map(|s| s.tick).max().unwrap_or(0);
        let intent_log: Vec<LoggedIntent> = scripts
            .iter()
            .map(|s| LoggedIntent {
                tick: s.tick,
                actor: ids[s.speaker],
                intent: Intent::Speak {
                    target: ids[s.listener],
                    act: s.act,
                    topic: s.topic,
                },
            })
            .collect();

        let pack = mw_village::VillagePack::new();
        let world = World::replay(seed, positions, last_tick + 1, &intent_log, &pack);

        let personas: Vec<Persona> = ids.iter().map(|&id| Persona::new(seed, id)).collect();
        let mut memories: Vec<Memory> = ids.iter().map(|&id| Memory::new(id, Vec::new())).collect();
        let mut log = ConversationLog::new();

        // Both memory and the conversation log are downstream event-log consumers
        // — the mechanical outcome (opinion delta on both parties) always applies.
        for event in world.event_log() {
            for mem in memories.iter_mut() {
                mem.ingest(event);
            }
        }
        log.ingest(world.event_log());

        let registry = SliceRegistry::village(&personas);
        Scene {
            world,
            pack,
            ids,
            personas,
            memories,
            registry,
            vocab: Vocab::village(),
            log,
        }
    }

    /// Current position of an entity.
    pub fn pos(&self, id: EntityId) -> (i32, i32) {
        self.world.entity(id).map(|e| e.pos).unwrap_or((0, 0))
    }

    /// Whether conversation row `i` is within the focus point (either party).
    pub fn is_observed(&self, i: usize, focus: &FocusPoint) -> bool {
        let row = &self.log.rows()[i];
        focus.is_observed(self.pos(row.speaker), self.pos(row.listener))
    }

    /// Opinion the memory of entity `a` holds toward entity `b`.
    pub fn opinion(&self, a: EntityId, b: EntityId) -> i32 {
        self.memories[a.index() as usize].opinion(b)
    }

    /// Render every currently-observed conversation, leaving the rest latent.
    /// Returns how many rows were rendered this pass.
    pub fn render_observed<T: DialogueRenderer>(&mut self, focus: &FocusPoint, tb: &T) -> usize {
        let observed: Vec<usize> = (0..self.log.len())
            .filter(|&i| self.is_observed(i, focus) && self.log.rows()[i].text.is_none())
            .collect();
        for &i in &observed {
            self.log.render(i, &self.registry, tb, &self.vocab);
        }
        observed.len()
    }

    /// Backfill (or fetch cached) the line for a latent conversation.
    pub fn backfill<T: DialogueRenderer>(&mut self, i: usize, tb: &T) -> String {
        self.log
            .render(i, &self.registry, tb, &self.vocab)
            .to_string()
    }

    /// FNV-1a fold of the sim state dialogue can touch — positions plus every
    /// pairwise opinion. Deliberately excludes rendered text, so comparing this
    /// before and after rendering proves the one-way street (sim → text).
    pub fn state_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325;
        for &id in &self.ids {
            let p = self.pos(id);
            h = fold(h, p.0 as u64);
            h = fold(h, p.1 as u64);
        }
        for mem in &self.memories {
            for &other in &self.ids {
                h = fold(h, mem.opinion(other) as u64);
            }
        }
        h
    }
}

/// FNV-1a mix, matching the kernel's stable hash style.
fn fold(mut h: u64, x: u64) -> u64 {
    for b in x.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The canonical gate scenario: three adjacent pairs — one inside the focus,
/// two outside — so exactly one of the three conversations is observed. Shared
/// by both gate tests so the live and deterministic runs exercise the same
/// scene.
pub fn demo() -> (Vec<(i32, i32)>, Vec<Script>, FocusPoint) {
    let positions = vec![(8, 8), (8, 9), (0, 0), (1, 0), (15, 15), (14, 15)];
    let scripts = vec![
        // observed pair (both inside the focus radius): befriend about the harvest.
        Script {
            tick: 0,
            speaker: 0,
            listener: 1,
            act: 1,
            topic: 0,
        },
        // latent pair (far corner): taunt about the newcomer.
        Script {
            tick: 0,
            speaker: 2,
            listener: 3,
            act: 2,
            topic: 1,
        },
        // latent pair (far corner): greet about the weather.
        Script {
            tick: 0,
            speaker: 4,
            listener: 5,
            act: 0,
            topic: 2,
        },
    ];
    (positions, scripts, FocusPoint::new((8, 8), 2))
}
