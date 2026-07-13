//! Latent dialogue (DESIGN.md §4): a speak act commits its mechanical outcome
//! immediately, but the *words* are rendered only when someone is watching —
//! otherwise the conversation stays latent and is backfilled on demand.
//!
//! This module owns the sim→text seam. It never touches world state: like
//! [`crate::memory`], the [`ConversationLog`] is a downstream consumer of the
//! kernel event log, so the street runs one way (sim → text) and replay holds.
//!
//! It also fixes the prior seam gap where a render request carried only numeric
//! codes: a scenario-owned [`PersonaRegistry`] resolves an entity to a real name
//! and a one-line persona summary, and a [`Vocab`] resolves act/topic codes to
//! words, so the TEXT model always prompts on real identities (DESIGN.md §8).

use std::cell::Cell;

use mw_core::{EntityId, Event};

use crate::memory::SPEAK_AFFECT;
use crate::persona::{trait_idx, Persona, N_TRAITS};

/// A character's prompt identity: what the TEXT model is told it is voicing.
/// Replaces the raw numeric persona code the old `SpeakRequest` carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonaCard {
    pub name: String,
    /// One-line persona summary (name + salient traits) for the system prompt.
    pub summary: String,
}

/// Scenario-owned map from an entity to its prompt identity. Packs build this
/// however they like; the village derives it from persona traits so replay
/// regrows identical identities (DESIGN.md load-bearing decision 2).
pub trait PersonaRegistry {
    fn card(&self, entity: EntityId) -> &PersonaCard;
}

/// A registry backed by a per-entity-slot vector — the common case, since
/// entity slot indices are dense and stable.
pub struct SliceRegistry {
    cards: Vec<PersonaCard>,
}

impl SliceRegistry {
    pub fn new(cards: Vec<PersonaCard>) -> Self {
        Self { cards }
    }

    /// Build a village registry from a persona per entity slot (index order).
    pub fn village(personas: &[Persona]) -> Self {
        Self::new(personas.iter().map(village_card).collect())
    }
}

impl PersonaRegistry for SliceRegistry {
    fn card(&self, entity: EntityId) -> &PersonaCard {
        &self.cards[entity.index() as usize]
    }
}

// --- deterministic village name + persona generation ---

/// Name syllable tables. A name is one prefix + one suffix chosen by folding the
/// trait vector, so two personas rarely collide yet the map is a pure function
/// of the persona (no stored state, replay-stable).
const NAME_PREFIX: [&str; 12] = [
    "Bram", "Elda", "Tor", "Mira", "Fen", "Osric", "Wren", "Ada", "Corvin", "Sable", "Hollis",
    "Nyx",
];
const NAME_SUFFIX: [&str; 8] = ["", "wyn", "ric", "a", "ford", "is", "beth", "or"];

/// Adjective per persona trait, ordered by [`trait_idx`].
const TRAIT_ADJ: [&str; N_TRAITS] = [
    "fiery",       // AGGRESSION
    "gregarious",  // SOCIABILITY
    "hardworking", // INDUSTRIOUSNESS
    "acquisitive", // GREED
    "wary",        // CAUTION
];

/// Deterministic name + one-line persona for a village character, derived purely
/// from its trait vector.
pub fn village_card(persona: &Persona) -> PersonaCard {
    // Distinct small primes per slot spread the traits across the name space so
    // similar personas still get different-sounding names.
    const PRIMES: [u32; N_TRAITS] = [7, 13, 31, 61, 127];
    let key = persona.traits.iter().zip(PRIMES).fold(0u32, |a, (&t, p)| {
        a.wrapping_add((t as u32).wrapping_mul(p))
    });
    let name = format!(
        "{}{}",
        NAME_PREFIX[key as usize % NAME_PREFIX.len()],
        NAME_SUFFIX[(key >> 8) as usize % NAME_SUFFIX.len()],
    );

    // Two most-pronounced traits become the persona's descriptors.
    let mut order = [
        trait_idx::AGGRESSION,
        trait_idx::SOCIABILITY,
        trait_idx::INDUSTRIOUSNESS,
        trait_idx::GREED,
        trait_idx::CAUTION,
    ];
    order.sort_by_key(|&i| std::cmp::Reverse(persona.traits[i]));
    let summary = format!(
        "{name}, a {} and {} villager who speaks plainly.",
        TRAIT_ADJ[order[0]], TRAIT_ADJ[order[1]],
    );
    PersonaCard { name, summary }
}

/// Scenario-owned act/topic vocabulary: resolves the numeric `act`/`topic` codes
/// a [`mw_core::Intent::Speak`] carries into words for the prompt.
pub struct Vocab {
    pub acts: Vec<String>,
    pub topics: Vec<String>,
}

impl Vocab {
    pub fn new(acts: Vec<String>, topics: Vec<String>) -> Self {
        Self { acts, topics }
    }

    /// The village's default vocabulary. Codes are indices into these lists.
    pub fn village() -> Self {
        Self::new(
            [
                "greet",
                "befriend",
                "taunt",
                "trade with",
                "console",
                "warn",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            [
                "the harvest",
                "the newcomer",
                "the weather",
                "a debt owed",
                "the well running low",
                "old grievances",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        )
    }

    fn act(&self, code: u32) -> &str {
        self.acts.get(code as usize).map_or("speak with", |s| s)
    }

    fn topic(&self, code: u32) -> &str {
        self.topics.get(code as usize).map_or("things", |s| s)
    }
}

/// A resolved render request: real names, persona summaries, and act/topic
/// words. TEXT verbalizes the already-committed act; it never decides
/// (DESIGN.md §5).
#[derive(Clone, Copy, Debug)]
pub struct RenderRequest<'a> {
    pub speaker: &'a PersonaCard,
    pub listener: &'a PersonaCard,
    pub act: &'a str,
    pub topic: &'a str,
    /// Scene / relationship context (includes the committed opinion outcome).
    pub context: &'a str,
    /// Stable per-conversation key so the speaker's persona prefix stays warm in
    /// the TEXT backend's KV cache across turns.
    pub conversation: u64,
}

/// Turns a committed speak act into a line of dialogue. The real TEXT model and
/// the deterministic [`MockRenderer`] both implement this; the sim only ever
/// holds one behind this trait.
pub trait DialogueRenderer {
    fn render(&self, request: &RenderRequest<'_>) -> String;
}

/// A deterministic, offline stand-in for the TEXT model: formats the request
/// into a fixed line and counts calls. Lets the whole latent-dialogue pipeline
/// run without a live model and proves the one-way street — its output is never
/// read back into sim state.
#[derive(Default)]
pub struct MockRenderer {
    calls: Cell<u32>,
}

impl MockRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times [`DialogueRenderer::render`] has run — the attention-gate
    /// test seam.
    pub fn calls(&self) -> u32 {
        self.calls.get()
    }
}

impl DialogueRenderer for MockRenderer {
    fn render(&self, request: &RenderRequest<'_>) -> String {
        self.calls.set(self.calls.get() + 1);
        // Deterministic and in-persona enough to assert on: names + act + topic.
        format!(
            "{} to {}: I {} you about {}.",
            request.speaker.name, request.listener.name, request.act, request.topic,
        )
    }
}

/// One committed conversation. The mechanical `outcome` (opinion delta) is
/// applied to both parties' memory elsewhere; here it is only recorded. `text`
/// is `None` while latent, `Some` once observed or backfilled (then cached).
#[derive(Clone, Debug)]
pub struct Conversation {
    pub tick: u64,
    pub speaker: EntityId,
    pub listener: EntityId,
    pub act: u32,
    pub topic: u32,
    /// Fixed-point opinion delta each party applied to the other.
    pub outcome: i32,
    pub text: Option<String>,
}

/// The conversation ledger: a latent row per committed speak act, rendered on
/// demand. A pure event-log consumer, so it holds no world state.
#[derive(Default)]
pub struct ConversationLog {
    rows: Vec<Conversation>,
}

impl ConversationLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a latent row for every `Spoke` in `events`. The mechanical outcome
    /// is the intrinsic speak affect both parties apply through their memory
    /// (recorded here for coherent backfill).
    pub fn ingest(&mut self, events: &[Event]) {
        for event in events {
            if let Event::Spoke {
                tick,
                actor,
                target,
                act,
                topic,
            } = *event
            {
                self.rows.push(Conversation {
                    tick,
                    speaker: actor,
                    listener: target,
                    act,
                    topic,
                    outcome: SPEAK_AFFECT,
                    text: None,
                });
            }
        }
    }

    pub fn rows(&self) -> &[Conversation] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Render row `i`'s line (or return the cached one), resolving personas via
    /// `reg` and act/topic via `vocab`. First call invokes `tb`; a re-inspect is
    /// free — that is the backfill cache the gate checks.
    pub fn render<R: PersonaRegistry, T: DialogueRenderer>(
        &mut self,
        i: usize,
        reg: &R,
        tb: &T,
        vocab: &Vocab,
    ) -> &str {
        if self.rows[i].text.is_none() {
            let (speaker, listener, act, topic, outcome) = {
                let r = &self.rows[i];
                (r.speaker, r.listener, r.act, r.topic, r.outcome)
            };
            let context = context_line(outcome);
            let req = RenderRequest {
                speaker: reg.card(speaker),
                listener: reg.card(listener),
                act: vocab.act(act),
                topic: vocab.topic(topic),
                context: &context,
                // Speaker keys the KV slot so its persona prefix stays warm.
                conversation: speaker.index() as u64,
            };
            self.rows[i].text = Some(tb.render(&req));
        }
        self.rows[i].text.as_deref().unwrap()
    }
}

/// Scene/relationship context string from the committed opinion outcome, so the
/// rendered line is coherent with the mechanical result of the act.
fn context_line(outcome: i32) -> String {
    let mood = match outcome.signum() {
        1 => "the exchange warmed relations",
        -1 => "the exchange soured relations",
        _ => "relations were unchanged",
    };
    format!("A chance meeting in the village; {mood}.")
}

/// A focus point the player/camera is centered on. This is the local half of the
/// same anchor concept the Director/LOD ring uses (DESIGN.md §10); defined here
/// so latent dialogue never depends on that code.
#[derive(Clone, Copy, Debug)]
pub struct FocusPoint {
    pub center: (i32, i32),
    pub radius: i32,
}

impl FocusPoint {
    pub fn new(center: (i32, i32), radius: i32) -> Self {
        Self { center, radius }
    }

    /// A conversation is observed when either participant is within the focus
    /// radius (Chebyshev) — close enough for the player to hear it.
    pub fn is_observed(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        self.within(a) || self.within(b)
    }

    fn within(&self, p: (i32, i32)) -> bool {
        (p.0 - self.center.0).abs() <= self.radius && (p.1 - self.center.1).abs() <= self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_core::World;

    fn ids(n: usize) -> Vec<EntityId> {
        let mut w = World::new(1);
        (0..n).map(|i| w.spawn((i as i32, 0))).collect()
    }

    #[test]
    fn village_card_is_deterministic_and_distinct() {
        let people = ids(2);
        let p0 = Persona::new(9, people[0]);
        let p1 = Persona::new(9, people[1]);
        // Pure function of the persona: identical input → identical card.
        assert_eq!(village_card(&p0), village_card(&p0));
        // Different personas get their own identity (name or summary differs).
        assert_ne!(village_card(&p0), village_card(&p1));
        // The summary leads with the name, so the prompt is grounded in a person.
        let c0 = village_card(&p0);
        assert!(c0.summary.starts_with(&c0.name));
    }

    #[test]
    fn ingest_records_latent_row_per_spoke() {
        let people = ids(2);
        let mut log = ConversationLog::new();
        log.ingest(&[
            Event::Spoke {
                tick: 0,
                actor: people[0],
                target: people[1],
                act: 2,
                topic: 1,
            },
            Event::Moved {
                tick: 0,
                actor: people[0],
                to: (1, 0),
            }, // ignored
        ]);
        assert_eq!(log.len(), 1);
        let row = &log.rows()[0];
        assert!(row.text.is_none(), "row starts latent");
        assert_eq!((row.act, row.topic), (2, 1));
        assert_eq!(row.outcome, SPEAK_AFFECT);
    }

    #[test]
    fn render_invokes_once_then_caches() {
        let people = ids(2);
        let personas: Vec<Persona> = people.iter().map(|&id| Persona::new(9, id)).collect();
        let reg = SliceRegistry::village(&personas);
        let vocab = Vocab::village();
        let mut log = ConversationLog::new();
        log.ingest(&[Event::Spoke {
            tick: 0,
            actor: people[0],
            target: people[1],
            act: 2,
            topic: 1,
        }]);

        let tb = MockRenderer::new();
        let first = log.render(0, &reg, &tb, &vocab).to_string();
        assert_eq!(tb.calls(), 1);
        assert!(first.contains("taunt") && first.contains("the newcomer"));
        // Second inspect is cached — no additional render.
        let second = log.render(0, &reg, &tb, &vocab).to_string();
        assert_eq!(tb.calls(), 1);
        assert_eq!(first, second);
    }

    #[test]
    fn focus_observes_only_within_radius() {
        let f = FocusPoint::new((8, 8), 2);
        assert!(f.is_observed((8, 8), (0, 0)), "one party inside is enough");
        assert!(f.is_observed((10, 10), (0, 0)), "edge of radius counts");
        assert!(!f.is_observed((0, 0), (15, 15)), "both far → unobserved");
    }
}
