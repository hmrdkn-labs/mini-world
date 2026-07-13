//! Director / LOD rings and the analytic AFK fast-forward (DESIGN.md §10).
//!
//! Two cooperating pieces:
//!
//! * [`Director`] — the live level-of-detail brain. Around a focus point (the
//!   camera/player anchor) it sorts agents into three rings — **hot** (SOUL every
//!   tick), **warm** (SOUL every N ticks, idle-extrapolated between), **cold**
//!   (no SOUL) — with hysteresis so an agent drifting along a band edge does not
//!   flap, and promote-on-notable-event so drama pulls a character back to full
//!   fidelity. It decides *tick rates*; it never touches world state.
//!
//! * [`fast_forward`] — the AFK enabler. When the player is away there is nobody
//!   to watch, so the whole population runs **cold**: no neural net, no kernel
//!   observe/validate pipeline, just closed-form need integration plus coarse
//!   persona-driven event sampling. It advances an in-game week over 50 agents in
//!   well under a second and hands back a returning-player digest.
//!
//! Determinism holds exactly as in the hot path: the only inputs are the seed,
//! the entity ids, and the tick, every draw comes from a kernel RNG stream keyed
//! by `(seed, entity, tag, quantum)`, all state is integer, and nothing iterates
//! a hash map. Two fast-forwards of the same `(seed, agents, ticks)` produce a
//! bitwise-identical digest and final hash.

use std::time::Instant;

use mw_agents::persona::{trait_idx, Persona, PERSONA_ONE};
use mw_core::{agent_rng, EntityId, StreamTag, World};
use mw_village::{VillagePack, MAX_NEED, STARVE_TICKS};

use crate::soak::start_positions;

/// In-game-time unit, defined in exactly one place: one tick is one in-game
/// second, so a day is 86 400 ticks. Everything time-shaped (a "week" of
/// fast-forward, digest tick labels) derives from this.
pub const TICKS_PER_DAY: u64 = 86_400;
/// One in-game week — the fast-forward gate horizon.
pub const TICKS_PER_WEEK: u64 = 7 * TICKS_PER_DAY;

/// The three level-of-detail rings (DESIGN.md §10), ordered by fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ring {
    /// Off screen, out of mind: resolved analytically, never runs SOUL.
    Cold = 0,
    /// Nearby / plot-relevant: SOUL every `warm_cadence` ticks.
    Warm = 1,
    /// On screen: SOUL every tick, dialogue eligible.
    Hot = 2,
}

/// Ring-assignment tuning. Radii are Chebyshev distance from the focus point,
/// matching the grid's movement metric.
#[derive(Clone, Copy, Debug)]
pub struct RingConfig {
    /// Inside this radius of the focus → hot.
    pub hot_radius: i32,
    /// Inside this radius (but outside `hot_radius`) → warm; beyond → cold.
    pub warm_radius: i32,
    /// Warm agents run SOUL once every this many ticks (the `N` of DESIGN §10).
    pub warm_cadence: u64,
    /// A demotion (to a colder ring) only takes effect after its condition has
    /// held continuously for this many ticks — the anti-flap band. Promotions
    /// are always immediate (drama should never wait).
    pub hysteresis: u64,
    /// A notable event pins its agent hot for this many ticks.
    pub promote_ticks: u64,
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

/// Live LOD assignment with hysteresis and event-driven promotion. Slot-indexed
/// by `EntityId::index`, like the rest of the agent state.
pub struct Director {
    cfg: RingConfig,
    focus: (i32, i32),
    ring: Vec<Ring>,
    /// Tick at which the agent first wanted a *colder* ring than it currently
    /// holds; `None` once it is at (or hotter than) its target. Demotion fires
    /// once the gap has persisted `hysteresis` ticks.
    cooling_since: Vec<Option<u64>>,
    /// Tick until which a notable event keeps the agent pinned hot.
    promote_until: Vec<u64>,
}

impl Director {
    pub fn new(cfg: RingConfig, agents: usize, focus: (i32, i32)) -> Self {
        Self {
            cfg,
            focus,
            ring: vec![Ring::Cold; agents],
            cooling_since: vec![None; agents],
            promote_until: vec![0; agents],
        }
    }

    pub fn set_focus(&mut self, focus: (i32, i32)) {
        self.focus = focus;
    }

    pub fn ring(&self, slot: usize) -> Ring {
        self.ring[slot]
    }

    /// The band an agent falls in purely from its distance to the focus.
    fn band(&self, pos: (i32, i32)) -> Ring {
        let d = (pos.0 - self.focus.0)
            .abs()
            .max((pos.1 - self.focus.1).abs());
        if d <= self.cfg.hot_radius {
            Ring::Hot
        } else if d <= self.cfg.warm_radius {
            Ring::Warm
        } else {
            Ring::Cold
        }
    }

    /// Pin `slot` hot for `promote_ticks` — call on a notable event (a death or
    /// attack witnessed, a relationship swing). Takes effect immediately.
    pub fn note_event(&mut self, slot: usize, tick: u64) {
        self.promote_until[slot] = tick + self.cfg.promote_ticks;
        self.ring[slot] = Ring::Hot;
        self.cooling_since[slot] = None;
    }

    /// Recompute every agent's ring from `positions` at `tick`. Promotions (a
    /// hotter target, or an active event pin) apply at once; demotions wait out
    /// the hysteresis window so a jittering distance cannot cause ring flapping.
    pub fn update(&mut self, positions: &[(i32, i32)], tick: u64) {
        for (slot, &pos) in positions.iter().enumerate() {
            let mut target = self.band(pos);
            if tick < self.promote_until[slot] {
                target = Ring::Hot;
            }
            let current = self.ring[slot];
            match target.cmp(&current) {
                std::cmp::Ordering::Greater => {
                    // Hotter: promote now.
                    self.ring[slot] = target;
                    self.cooling_since[slot] = None;
                }
                std::cmp::Ordering::Equal => self.cooling_since[slot] = None,
                std::cmp::Ordering::Less => {
                    // Colder: hold until the gap has persisted long enough.
                    let since = *self.cooling_since[slot].get_or_insert(tick);
                    if tick.saturating_sub(since) >= self.cfg.hysteresis {
                        self.ring[slot] = target;
                        self.cooling_since[slot] = None;
                    }
                }
            }
        }
    }

    /// Whether `slot` runs SOUL this tick: hot always, warm on its cadence, cold
    /// never. This is the tick-rate the harness gates the policy with.
    pub fn should_run_soul(&self, slot: usize, tick: u64) -> bool {
        match self.ring[slot] {
            Ring::Hot => true,
            Ring::Warm => self.cfg.warm_cadence == 0 || tick.is_multiple_of(self.cfg.warm_cadence),
            Ring::Cold => false,
        }
    }
}

// --- cold analytic need model --------------------------------------------
//
// The cold ring never runs SOUL, so it models "the agent maintains each need on
// a routine cycle" in closed form. Rates and gains mirror the village's needs
// system (mw_village::needs); the cycle for each need is `gain / decay`, so a
// full restore exactly offsets one cycle of decay and the need holds a bounded
// sawtooth around a steady mean instead of drifting to starvation. This is the
// `state += rate·min(Δt, cap)`, `cycles = floor(Δt / cycle)` resolution of
// DESIGN §10, calibrated so cold trajectories track the hot ones (drift test).

/// Per-tick decay, indexed [hunger, energy, social].
const DECAY: [i64; 3] = [2, 1, 3];
/// Restore delivered by one maintenance cycle (eat / sleep / socialize).
const GAIN: [i64; 3] = [600, 500, 300];
/// Maintenance period per need = how often the agent tops it up. Each is
/// sustainable (`GAIN >= DECAY * CYCLE`, so restore offsets a cycle of decay)
/// and tuned so the resulting sawtooth's mean tracks the hot policy's steady
/// need levels (the drift gate). Hunger's period is persona-scaled below.
const CYCLE: [u64; 3] = [160, 360, 100];
/// A need dipping below this at a trough is a "crisis" worth logging.
const CRISIS: i64 = 150;
/// Largest slice of ticks resolved in one analytic step — the `cap` above. It
/// bounds a single need's swing per step for numeric stability and gives the
/// coarse event sampler its cadence.
const COLD_QUANTUM: u64 = 64;

/// Fixed-point rapport swing that counts as a notable relationship event.
const SWING_UNIT: i64 = 4_000;
/// One coarse social encounter's rapport delta (before the sign).
const ENCOUNTER: i64 = 900;
/// Ceiling on |rapport|: relationships saturate rather than running away, so
/// importance stays bounded and each pair contributes a finite set of swings.
const RAPPORT_CAP: i64 = 3 * SWING_UNIT;

// Distinct RNG stream tags for cold sampling ("COLD\0EVT" / "COLD\0ENC").
const EVENT_TAG: StreamTag = 0x434f_4c44_0045_5654;

/// Kinds of notable event the ledger accumulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// A relationship warmed past a threshold.
    Bond,
    /// A relationship soured past a threshold.
    Rift,
    /// A need bottomed out dangerously.
    NeedCrisis,
    /// An agent starved to death.
    Death,
}

/// One notable event, with a precomputed importance for digest ranking.
#[derive(Clone, Copy, Debug)]
pub struct LedgerEntry {
    pub tick: u64,
    pub kind: EventKind,
    pub who: EntityId,
    pub other: Option<EntityId>,
    pub importance: i32,
}

/// The returning-player digest: the run's biggest moments and a per-agent line.
#[derive(Clone, Debug)]
pub struct Digest {
    /// Top notable events, most important first.
    pub top: Vec<String>,
    /// One line per agent: its most-changed relationship and neediest stat.
    pub per_agent: Vec<String>,
}

/// Fast-forward parameters.
#[derive(Clone, Copy, Debug)]
pub struct FfConfig {
    pub seed: u64,
    pub agents: i32,
    pub ticks: u64,
    /// How many top events the digest surfaces.
    pub top_k: usize,
}

impl Default for FfConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            agents: 50,
            ticks: TICKS_PER_WEEK,
            top_k: 10,
        }
    }
}

/// What a fast-forward produced. Deterministic in `(seed, agents, ticks)` except
/// `elapsed_secs` (a wall-clock throughput measurement, report-only).
#[derive(Clone, Debug)]
pub struct FfReport {
    pub cfg: FfConfig,
    pub elapsed_secs: f64,
    pub final_hash: u64,
    pub ledger_len: usize,
    pub deaths: usize,
    /// Time-averaged mean need level per stat, comparable to
    /// [`crate::soak::SoakReport::mean_needs`].
    pub mean_needs: [f64; 3],
    pub digest: Digest,
}

impl FfReport {
    pub fn ticks_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.cfg.ticks as f64 / self.elapsed_secs
        } else {
            f64::INFINITY
        }
    }
}

/// Reconstruct a plausible instantaneous hot need vector from an agent's cold
/// aggregate when it is promoted back to hot (boundary consistency, DESIGN §10
/// point 4). The cold model carries a faithful per-need level, so promotion
/// adopts it directly, clamped into the valid band — a pure, deterministic
/// function of the cold state, so both replays reconstruct identically.
pub fn reconstruct_hot(cold_needs: [i64; 3]) -> [i16; 3] {
    let m = MAX_NEED as i64;
    [
        cold_needs[0].clamp(0, m) as i16,
        cold_needs[1].clamp(0, m) as i16,
        cold_needs[2].clamp(0, m) as i16,
    ]
}

/// The hunger maintenance cycle for one agent: a less industrious character
/// tops up its larder a little less often (a longer period, a lower mean
/// hunger), but every period stays sustainable (`<= GAIN/DECAY = 300`) so the
/// calibrated population holds steady rather than starving. The Death path
/// still guards genuinely depleted inputs (e.g. a fast-forward begun on an
/// already-starving world).
fn hunger_cycle(p: &Persona) -> u64 {
    let ind = p.traits[trait_idx::INDUSTRIOUSNESS].clamp(0, PERSONA_ONE) as u64;
    // slack in [0, 90]: least-industrious → +90 (period 250, mean ~750),
    // most-industrious → +0 (period 160, mean ~840).
    let slack = 90 * (PERSONA_ONE as u64 - ind) / PERSONA_ONE as u64;
    CYCLE[0] + slack
}

/// FNV-1a fold — stable, architecture-independent hash mix (mirrors the soak's).
fn fold(mut h: u64, x: u64) -> u64 {
    h ^= x;
    h.wrapping_mul(0x0000_0100_0000_01b3)
}

/// Run the analytic fast-forward and return its digest + report.
pub fn fast_forward(cfg: FfConfig) -> FfReport {
    let pack = VillagePack::new();
    let mut world = World::with_pack(cfg.seed, &pack);

    // Same deterministic setup as the hot soak, so a fast-forward and a fully
    // hot run of the same seed start from an identical population.
    let positions = start_positions(cfg.agents);
    let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
    let personas: Vec<Persona> = ids.iter().map(|&id| Persona::new(cfg.seed, id)).collect();
    let n = ids.len();

    // Initial need levels read straight from the pack (full at t=0).
    let t0_tick = world.tick();
    let mut needs: Vec<[i64; 3]> = ids
        .iter()
        .map(|&id| {
            let (h, e, s) = pack.needs(id).project(t0_tick);
            [h as i64, e as i64, s as i64]
        })
        .collect();
    let mut alive = vec![true; n];
    let mut starve = vec![0u64; n]; // consecutive ticks at zero hunger

    // Cold agents never move, so each agent's nearest neighbour (its only social
    // partner in the cold model) is fixed for the whole fast-forward.
    let nn: Vec<Option<usize>> = (0..n).map(|a| nearest_neighbor(&positions, a)).collect();
    let mut rapport = vec![0i64; n]; // cumulative feeling toward `nn`
    let mut swing_marks = vec![0i64; n]; // last rapport level logged as an event
    let hcycle: Vec<u64> = personas.iter().map(hunger_cycle).collect();

    let mut ledger: Vec<LedgerEntry> = Vec::new();
    let mut sum_needs = [0u128; 3];
    let max = MAX_NEED as i64;

    let t_start = Instant::now();
    let mut t = 0u64;
    while t < cfg.ticks {
        let dt = COLD_QUANTUM.min(cfg.ticks - t);
        let t_end = t + dt;
        let q = t / COLD_QUANTUM; // quantum index — the RNG time key

        for a in 0..n {
            if !alive[a] {
                continue; // stays at hunger 0, contributes nothing further
            }
            let cyc = [hcycle[a], CYCLE[1], CYCLE[2]];
            let mut crisis = false;
            for i in 0..3 {
                // Decay then discrete restores over this quantum; clamp once at
                // the end (net change per step is bounded, so the sawtooth is
                // faithful without per-tick clamping).
                let restores = (t_end / cyc[i]) as i64 - (t / cyc[i]) as i64;
                let v = needs[a][i] - DECAY[i] * dt as i64 + GAIN[i] * restores;
                let clamped = v.clamp(0, max);
                if clamped < CRISIS {
                    crisis = true;
                }
                needs[a][i] = clamped;
            }

            // Starvation: zero hunger sustained for STARVE_TICKS is death.
            if needs[a][0] == 0 {
                starve[a] += dt;
                if starve[a] >= STARVE_TICKS {
                    alive[a] = false;
                    ledger.push(LedgerEntry {
                        tick: t_end,
                        kind: EventKind::Death,
                        who: ids[a],
                        other: None,
                        importance: 10_000,
                    });
                    continue;
                }
            } else {
                starve[a] = 0;
            }

            if crisis {
                ledger.push(LedgerEntry {
                    tick: t_end,
                    kind: EventKind::NeedCrisis,
                    who: ids[a],
                    other: None,
                    importance: 2_000,
                });
            }

            // Coarse social sampling: a sociable agent occasionally has an
            // encounter with its neighbour, warming or (across factions) souring
            // the relationship. Deterministic from the agent's own RNG stream.
            if let Some(other) = nn[a] {
                let mut rng = agent_rng(cfg.seed, ids[a], EVENT_TAG, q);
                let soci = personas[a].traits[trait_idx::SOCIABILITY] as u32;
                // Encounter chance rises with sociability (0..~1 per quantum).
                if rng.range_u32(PERSONA_ONE as u32) < soci {
                    let friendly = personas[a].faction() == personas[other].faction();
                    let delta = if friendly { ENCOUNTER } else { -ENCOUNTER };
                    rapport[a] = (rapport[a] + delta).clamp(-RAPPORT_CAP, RAPPORT_CAP);
                    // Log only when rapport crosses another SWING_UNIT band, so
                    // one drifting relationship yields events, not spam.
                    let band = rapport[a].div_euclid(SWING_UNIT);
                    let last = swing_marks[a].div_euclid(SWING_UNIT);
                    if band != last {
                        swing_marks[a] = rapport[a];
                        ledger.push(LedgerEntry {
                            tick: t_end,
                            kind: if delta >= 0 {
                                EventKind::Bond
                            } else {
                                EventKind::Rift
                            },
                            who: ids[a],
                            other: Some(ids[other]),
                            importance: rapport[a].unsigned_abs().min(i32::MAX as u64) as i32,
                        });
                    }
                }
            }
        }

        for row in &needs {
            for (i, &v) in row.iter().enumerate() {
                sum_needs[i] += v as u128 * dt as u128;
            }
        }
        t = t_end;
    }
    let elapsed_secs = t_start.elapsed().as_secs_f64();

    let denom = (cfg.ticks * n as u64).max(1) as f64;
    let mean_needs = [
        sum_needs[0] as f64 / denom,
        sum_needs[1] as f64 / denom,
        sum_needs[2] as f64 / denom,
    ];
    let deaths = alive.iter().filter(|&&a| !a).count();
    let final_hash = hash_ff(&ids, &needs, &alive, &rapport, &ledger);
    let digest = build_digest(&cfg, &ids, &needs, &rapport, &nn, &ledger);

    FfReport {
        cfg,
        elapsed_secs,
        final_hash,
        ledger_len: ledger.len(),
        deaths,
        mean_needs,
        digest,
    }
}

/// Nearest other agent by Chebyshev distance; ties resolve to the lower slot so
/// the pairing is deterministic. `None` only for a lone agent.
fn nearest_neighbor(positions: &[(i32, i32)], me: usize) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    for (i, &p) in positions.iter().enumerate() {
        if i == me {
            continue;
        }
        let d = (p.0 - positions[me].0)
            .abs()
            .max((p.1 - positions[me].1).abs());
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Canonical hash over the final cold state and the ledger shape. Iterates in
/// slot order — no map iteration, integer only.
fn hash_ff(
    ids: &[EntityId],
    needs: &[[i64; 3]],
    alive: &[bool],
    rapport: &[i64],
    ledger: &[LedgerEntry],
) -> u64 {
    let mut h = fold(0xcbf2_9ce4_8422_2325, ids.len() as u64);
    for a in 0..ids.len() {
        h = fold(h, ids[a].index() as u64);
        for &v in &needs[a] {
            h = fold(h, v as u64);
        }
        h = fold(h, alive[a] as u64);
        h = fold(h, rapport[a] as u64);
    }
    h = fold(h, ledger.len() as u64);
    for e in ledger {
        h = fold(h, e.tick);
        h = fold(h, e.kind as u64);
        h = fold(h, e.who.index() as u64);
        h = fold(h, e.other.map_or(u64::MAX, |o| o.index() as u64));
        h = fold(h, e.importance as u64);
    }
    h
}

/// Rank the ledger and compose the per-agent lines.
fn build_digest(
    cfg: &FfConfig,
    ids: &[EntityId],
    needs: &[[i64; 3]],
    rapport: &[i64],
    nn: &[Option<usize>],
    ledger: &[LedgerEntry],
) -> Digest {
    // A drifting relationship logs a swing each time it crosses a band; for the
    // digest we want the population's distinct big moments, so collapse each
    // (unordered pair, or solo agent) to its single strongest entry first.
    let key = |e: &LedgerEntry| match e.other {
        Some(o) => {
            let (x, y) = (e.who.index(), o.index());
            (x.min(y), x.max(y))
        }
        None => (e.who.index(), u32::MAX),
    };
    let mut ranked: Vec<&LedgerEntry> = ledger.iter().collect();
    // Group by key (importance desc within a group), then keep the first — the
    // strongest — of each group.
    ranked.sort_by(|a, b| {
        key(a)
            .cmp(&key(b))
            .then(b.importance.cmp(&a.importance))
            .then(a.tick.cmp(&b.tick))
    });
    ranked.dedup_by(|a, b| key(a) == key(b));
    // Now order the distinct moments: importance desc, earlier tick, lower slot.
    ranked.sort_by(|a, b| {
        b.importance
            .cmp(&a.importance)
            .then(a.tick.cmp(&b.tick))
            .then(a.who.index().cmp(&b.who.index()))
    });
    let top = ranked
        .into_iter()
        .take(cfg.top_k)
        .map(|e| {
            let who = e.who.index();
            match e.other {
                Some(o) => format!(
                    "  t{:>8} {:?} agent{} <-> agent{} (imp {})",
                    e.tick,
                    e.kind,
                    who,
                    o.index(),
                    e.importance
                ),
                None => format!(
                    "  t{:>8} {:?} agent{} (imp {})",
                    e.tick, e.kind, who, e.importance
                ),
            }
        })
        .collect();

    let names = ["hunger", "energy", "social"];
    let per_agent = (0..ids.len())
        .map(|a| {
            let (need_i, need_v) = needs[a]
                .iter()
                .enumerate()
                .min_by_key(|(_, &v)| v)
                .map(|(i, &v)| (i, v))
                .unwrap();
            let rel = match nn[a] {
                Some(o) => format!("closest agent{} (rapport {})", o, rapport[a]),
                None => "no neighbours".to_string(),
            };
            format!(
                "  agent{:>3}: neediest {} {}, {}",
                a, names[need_i], need_v, rel
            )
        })
        .collect();

    Digest { top, per_agent }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn director() -> Director {
        Director::new(RingConfig::default(), 3, (0, 0))
    }

    #[test]
    fn rings_assign_by_distance_to_focus() {
        let mut d = director();
        // slot0 on focus (hot), slot1 mid (warm), slot2 far (cold).
        let pos = [(0, 0), (8, 0), (30, 0)];
        d.update(&pos, 0);
        assert_eq!(d.ring(0), Ring::Hot);
        assert_eq!(d.ring(1), Ring::Warm);
        assert_eq!(d.ring(2), Ring::Cold);
    }

    #[test]
    fn promotion_is_immediate_demotion_waits_hysteresis() {
        let mut d = director();
        let near = [(0, 0), (0, 0), (0, 0)];
        d.update(&near, 0);
        assert_eq!(d.ring(0), Ring::Hot);

        // Agent walks away: it should stay hot until hysteresis elapses, then
        // drop to cold — never flapping in between.
        let far = [(30, 0), (0, 0), (0, 0)];
        d.update(&far, 1);
        assert_eq!(d.ring(0), Ring::Hot, "demotion must not be instant");
        d.update(&far, RingConfig::default().hysteresis); // < since+hysteresis
        assert_eq!(d.ring(0), Ring::Hot);
        d.update(&far, 1 + RingConfig::default().hysteresis);
        assert_eq!(d.ring(0), Ring::Cold, "demotion after hysteresis");

        // Coming back is instant.
        d.update(&near, 200);
        assert_eq!(d.ring(0), Ring::Hot);
    }

    #[test]
    fn a_flapping_edge_does_not_thrash() {
        let mut d = director();
        // Sit exactly on the hot/warm boundary and jitter one cell each side;
        // the ring must not oscillate every tick.
        let mut ring_changes = 0;
        let mut prev = d.ring(0);
        for tick in 0..20 {
            let x = 4 + (tick % 2) as i32; // 4 (hot) / 5 (warm) alternating
            d.update(&[(x, 0), (0, 0), (0, 0)], tick);
            if d.ring(0) != prev {
                ring_changes += 1;
                prev = d.ring(0);
            }
        }
        assert!(
            ring_changes <= 1,
            "edge jitter flapped {ring_changes} times"
        );
    }

    #[test]
    fn notable_event_pins_hot_then_releases() {
        let mut d = director();
        let far = [(30, 0), (0, 0), (0, 0)];
        d.update(&far, 0);
        assert_eq!(d.ring(0), Ring::Cold);
        d.note_event(0, 100);
        assert_eq!(d.ring(0), Ring::Hot, "event promotes immediately");
        // Still pinned inside the window.
        d.update(&far, 100 + RingConfig::default().promote_ticks - 1);
        assert_eq!(d.ring(0), Ring::Hot);
        // After the window it decays back (past hysteresis too).
        let release = 100 + RingConfig::default().promote_ticks;
        d.update(&far, release);
        d.update(&far, release + RingConfig::default().hysteresis);
        assert_eq!(d.ring(0), Ring::Cold);
    }

    #[test]
    fn warm_cadence_gates_soul() {
        let mut d = Director::new(
            RingConfig {
                warm_cadence: 8,
                ..RingConfig::default()
            },
            1,
            (0, 0),
        );
        d.update(&[(8, 0)], 0); // warm band
        assert_eq!(d.ring(0), Ring::Warm);
        assert!(d.should_run_soul(0, 0));
        assert!(!d.should_run_soul(0, 1));
        assert!(d.should_run_soul(0, 8));
    }

    #[test]
    fn reconstruct_clamps_into_the_need_band() {
        assert_eq!(reconstruct_hot([2000, -5, 500]), [MAX_NEED as i16, 0, 500]);
    }
}
