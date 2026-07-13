//! The needs system: hunger, energy, social, and death by starvation.
//!
//! Needs decay linearly per tick and are resolved *closed-form* — a need is
//! stored with the tick it was last touched, and its current value is projected
//! on demand (`stored - rate * elapsed`, clamped at zero). This is the same
//! analytic resolution the cold LOD ring uses (DESIGN.md, section 10): no
//! per-tick sweep, no wall-clock, order-independent because each entity's
//! projection depends only on its own stored state and the tick.

use mw_core::FnvHasher;

/// Fixed-point ceiling for every need (0 = fully depleted, `MAX_NEED` = sated).
pub const MAX_NEED: i32 = 1000;

/// Per-tick decay rates. Distinct primes so a projection bug in one need cannot
/// masquerade as another.
pub const HUNGER_DECAY: i32 = 2;
pub const ENERGY_DECAY: i32 = 1;
pub const SOCIAL_DECAY: i32 = 3;

/// Ticks of continuous zero hunger before an entity dies. Death is the only
/// terminal need outcome (DESIGN: "death only by prolonged starvation").
pub const STARVE_TICKS: u64 = 100;

/// Restore amounts for the acts that satisfy each need.
/// Eating restores two hundred ticks of hunger; the shorter sustainable cycle
/// makes the calibrated urgency policy exercise feeding instead of coasting on
/// a single large reserve.
pub const EAT_GAIN: i32 = 400;
pub const SLEEP_GAIN: i32 = 500;
pub const SPEAK_GAIN: i32 = 300;
/// Drinking water is a small energy top-up.
pub const USE_ENERGY_GAIN: i32 = 150;
/// Working burns energy; `work` is gated on having energy to spend.
pub const WORK_ENERGY_COST: i32 = 60;

#[derive(Clone, Copy, Debug)]
pub struct Needs {
    hunger: i32,
    energy: i32,
    social: i32,
    /// Tick the stored values are current as of.
    settled: u64,
    /// First tick hunger reached zero, latched persistently. Once set it is NOT
    /// cleared by continued settling at zero — only by eating back above zero —
    /// so the death clock keeps running for an agent that idles/moves at zero
    /// hunger instead of being reset on every act.
    starving_since: Option<u64>,
}

fn decay(stored: i32, rate: i32, elapsed: u64) -> i32 {
    // i64 keeps the product exact for long AFK fast-forwards.
    let dropped = rate as i64 * elapsed as i64;
    (stored as i64 - dropped).max(0) as i32
}

fn div_ceil(n: i32, d: i32) -> u64 {
    debug_assert!(n >= 0 && d > 0);
    ((n + d - 1) / d) as u64
}

impl Needs {
    pub fn full() -> Self {
        Self {
            hunger: MAX_NEED,
            energy: MAX_NEED,
            social: MAX_NEED,
            settled: 0,
            starving_since: None,
        }
    }

    /// Current values at `tick`, without mutating.
    pub fn project(&self, tick: u64) -> (i32, i32, i32) {
        let e = tick.saturating_sub(self.settled);
        (
            decay(self.hunger, HUNGER_DECAY, e),
            decay(self.energy, ENERGY_DECAY, e),
            decay(self.social, SOCIAL_DECAY, e),
        )
    }

    pub fn hunger(&self, tick: u64) -> i32 {
        self.project(tick).0
    }

    pub fn energy(&self, tick: u64) -> i32 {
        self.project(tick).1
    }

    /// Dead once hunger has sat at zero for `STARVE_TICKS`. The zero-crossing is
    /// latched in `starving_since` on the first settle at zero, so an agent that
    /// keeps idling/moving at zero hunger still dies on schedule (the latch is
    /// not reset by continued action). An entity that never acts again is caught
    /// by projecting its stored slope to the first-zero tick.
    pub fn dead(&self, tick: u64) -> bool {
        let zero_tick = self
            .starving_since
            .unwrap_or_else(|| self.settled + div_ceil(self.hunger, HUNGER_DECAY));
        tick >= zero_tick + STARVE_TICKS
    }

    /// Realize decay up to `tick` into the stored values. Call before applying
    /// a restore so the gain lands on the current (decayed) level. Latches
    /// `starving_since` the first time hunger reaches zero and never clears it
    /// here — that is what makes the death clock survive continued activity.
    pub fn settle(&mut self, tick: u64) {
        let (h, en, so) = self.project(tick);
        if h == 0 && self.starving_since.is_none() {
            // The true first-zero tick from the pre-settle slope, so the latched
            // value matches what `dead` would have projected before settling.
            self.starving_since = Some(self.settled + div_ceil(self.hunger, HUNGER_DECAY));
        }
        self.hunger = h;
        self.energy = en;
        self.social = so;
        self.settled = tick;
    }

    /// Add to a need after settling, clamped to `[0, MAX_NEED]`. `delta` may be
    /// negative (e.g. the energy cost of `work`).
    pub fn adjust(&mut self, tick: u64, hunger: i32, energy: i32, social: i32) {
        self.settle(tick);
        self.hunger = (self.hunger + hunger).clamp(0, MAX_NEED);
        self.energy = (self.energy + energy).clamp(0, MAX_NEED);
        self.social = (self.social + social).clamp(0, MAX_NEED);
        // Eating back above zero clears the starvation latch.
        if self.hunger > 0 {
            self.starving_since = None;
        }
    }

    /// Fold the full stored state into the canonical hash — including
    /// `starving_since`, so a replayed run reproduces the death clock exactly.
    pub fn hash_into(&self, h: &mut FnvHasher) {
        h.write_i32(self.hunger);
        h.write_i32(self.energy);
        h.write_i32(self.social);
        h.write_u64(self.settled);
        // Encode Option<u64> as a present-flag plus value (no float, fixed width).
        h.write_u64(self.starving_since.is_some() as u64);
        h.write_u64(self.starving_since.unwrap_or(0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_closed_form_linear_decay() {
        let n = Needs::full();
        // Distinct rates must project independently.
        assert_eq!(n.project(0), (1000, 1000, 1000));
        assert_eq!(n.project(100), (1000 - 200, 1000 - 100, 1000 - 300));
        // Each need clamps at zero on its own schedule; hunger (rate 2) empties
        // at 500, social (rate 3) at ~334, energy (rate 1) at 1000.
        assert_eq!(n.project(500), (0, 500, 0));
        assert_eq!(n.project(1000), (0, 0, 0));
    }

    #[test]
    fn settle_then_restore_lands_on_decayed_level() {
        let mut n = Needs::full();
        n.adjust(100, EAT_GAIN, 0, 0); // hunger 800 -> clamp(800+400)=1000
        assert_eq!(n.hunger(100), MAX_NEED);
        // Decay resumes from the settled tick, not from tick 0.
        assert_eq!(n.hunger(200), MAX_NEED - 2 * 100);
    }

    #[test]
    fn death_only_after_prolonged_starvation() {
        let n = Needs::full(); // hunger hits 0 at tick 500
        assert!(!n.dead(500)); // just reached zero
        assert!(!n.dead(500 + STARVE_TICKS - 1));
        assert!(n.dead(500 + STARVE_TICKS));
        // A well-fed entity is never dead.
        let mut fed = Needs::full();
        fed.adjust(400, EAT_GAIN, 0, 0);
        assert!(!fed.dead(400));
    }

    #[test]
    fn acting_at_zero_hunger_still_starves() {
        // The bug: settle() clamped hunger to 0 and reset the death clock, so an
        // agent that kept idling/moving never crossed the starvation threshold.
        // Now the zero-crossing is latched, so activity cannot cheat death.
        let mut n = Needs::full(); // hunger reaches 0 at tick 500
        for t in 500..(500 + STARVE_TICKS) {
            // Alternate idle (settle) and a "move" (also settle) every tick.
            n.settle(t);
            assert!(!n.dead(t), "must not die before the starve window elapses");
        }
        n.settle(500 + STARVE_TICKS);
        assert!(
            n.dead(500 + STARVE_TICKS),
            "dies exactly STARVE_TICKS after hunger first hit zero, despite acting"
        );
    }

    #[test]
    fn eating_above_zero_clears_the_starvation_latch() {
        let mut n = Needs::full();
        n.settle(500); // latch starving_since = 500
        assert!(n.dead(600));
        n.adjust(560, EAT_GAIN, 0, 0); // eat back above zero, clearing the latch
        assert!(!n.dead(600), "eating resets the death clock");
    }
}
