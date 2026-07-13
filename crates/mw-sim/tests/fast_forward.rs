//! Fast-forward gate: the analytic cold LOD ring must fast-forward an in-game
//! week over 50 agents near-instantly, track a fully-hot run's mean need levels,
//! and be bit-for-bit deterministic.

use mw_sim::director::{self, FfConfig, TICKS_PER_WEEK};
use mw_sim::soak::{self, SoakConfig};

const SEED: u64 = 1;
const AGENTS: i32 = 50;

/// FF one in-game week at 50 agents must finish well under the 10s budget.
#[test]
fn fast_forward_a_week_runs_under_ten_seconds() {
    let r = director::fast_forward(FfConfig {
        seed: SEED,
        agents: AGENTS,
        ticks: TICKS_PER_WEEK,
        ..FfConfig::default()
    });
    println!(
        "FF week: {} ticks / {} agents in {:.3}s ({:.0} ticks/s), {} events, {} deaths",
        TICKS_PER_WEEK,
        AGENTS,
        r.elapsed_secs,
        r.ticks_per_sec(),
        r.ledger_len,
        r.deaths,
    );
    assert!(
        r.elapsed_secs < 10.0,
        "week fast-forward took {:.3}s, budget is 10s",
        r.elapsed_secs
    );
}

/// The cold ring is a calibrated approximation of the hot policy: over the same
/// seed and period, its mean need levels must stay within ±15% of a fully-hot
/// run. Guards against the cold model silently drifting off the real dynamics.
#[test]
fn cold_fast_forward_tracks_hot_within_fifteen_percent() {
    // A period long enough to leave the from-full transient behind, short enough
    // that the fully-hot reference stays quick.
    const PERIOD: u64 = 10_000;

    let hot = soak::run(SoakConfig {
        seed: SEED,
        agents: AGENTS,
        ticks: PERIOD,
    })
    .mean_needs();
    let cold = director::fast_forward(FfConfig {
        seed: SEED,
        agents: AGENTS,
        ticks: PERIOD,
        ..FfConfig::default()
    })
    .mean_needs;

    let names = ["hunger", "energy", "social"];
    for i in 0..3 {
        let drift = (cold[i] - hot[i]).abs() / hot[i];
        println!(
            "{:>6}: hot={:.1} cold={:.1} drift={:+.1}%",
            names[i],
            hot[i],
            cold[i],
            100.0 * (cold[i] - hot[i]) / hot[i],
        );
        assert!(
            drift <= 0.15,
            "{} drifted {:.1}% (hot={:.1} cold={:.1})",
            names[i],
            100.0 * drift,
            hot[i],
            cold[i],
        );
    }
}

/// Same seed → identical digest and final hash. Fast-forward is a pure function
/// of `(seed, agents, ticks)`, which is what makes it a replay-safe AFK skip.
#[test]
fn fast_forward_is_deterministic() {
    let cfg = FfConfig {
        seed: SEED,
        agents: AGENTS,
        ticks: TICKS_PER_WEEK,
        ..FfConfig::default()
    };
    let a = director::fast_forward(cfg);
    let b = director::fast_forward(cfg);
    println!("FF hash a={:#018x} b={:#018x}", a.final_hash, b.final_hash);
    assert_eq!(a.final_hash, b.final_hash, "final hash must reproduce");
    assert_eq!(a.digest.top, b.digest.top, "top events must reproduce");
    assert_eq!(
        a.digest.per_agent, b.digest.per_agent,
        "per-agent digest must reproduce"
    );
}

/// A different seed grows a different village, so the digest must diverge —
/// proves the determinism above is real reproduction, not a constant.
#[test]
fn fast_forward_is_seed_sensitive() {
    let base = FfConfig {
        seed: SEED,
        agents: AGENTS,
        ticks: TICKS_PER_WEEK,
        ..FfConfig::default()
    };
    let a = director::fast_forward(base);
    let b = director::fast_forward(FfConfig { seed: 2, ..base });
    assert_ne!(a.final_hash, b.final_hash, "a different seed must diverge");
}
