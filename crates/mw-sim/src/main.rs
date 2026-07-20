//! mini-world headless runner.
//!
//! Two subcommands: `run` drives a trivial random-walk policy through the bare
//! kernel (a determinism smoke test), and `soak` runs the full village +
//! utility-SOUL + memory loop and reports throughput, an action histogram,
//! deaths, and a final state hash.

use clap::{Parser, Subcommand, ValueEnum};
use mw_agents::dialogue::{DialogueRenderer, FocusPoint, MockRenderer};
use mw_core::{AgentRng, Intent, KernelPack, Observation, SoulPolicy, World};
use mw_sim::dialogue::{demo, LlamaDialogue, Scene};
use mw_sim::director::{self, FfConfig, TICKS_PER_DAY};
use mw_sim::soak::{self, SoakConfig};
use mw_neural::ExpertiseLevel;
use mw_text::{Config, LlamaServerBackend};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HabitsFlag {
    On,
    Off,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TrajectoryProfile {
    Healthy,
    Scarcity,
    Hostile,
    Exhausted,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExpertiseFlag {
    Novice,
    Capable,
    Expert,
}

impl From<ExpertiseFlag> for ExpertiseLevel {
    fn from(level: ExpertiseFlag) -> Self {
        match level {
            ExpertiseFlag::Novice => Self::Novice,
            ExpertiseFlag::Capable => Self::Capable,
            ExpertiseFlag::Expert => Self::Expert,
        }
    }
}

impl From<TrajectoryProfile> for mw_sim::trajectory::ExportProfile {
    fn from(profile: TrajectoryProfile) -> Self {
        match profile {
            TrajectoryProfile::Healthy => Self::Healthy,
            TrajectoryProfile::Scarcity => Self::Scarcity,
            TrajectoryProfile::Hostile => Self::Hostile,
            TrajectoryProfile::Exhausted => Self::Exhausted,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PolicyFlag {
    Utility,
    Neural,
    Both,
    Omni,
    OmniBoth,
}

#[derive(Parser)]
#[command(about = "mini-world headless kernel runner")]
struct Args {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Bare-kernel random-walk run; prints the final canonical state hash.
    Run {
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        #[arg(long, default_value_t = 32)]
        entities: i32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    Trajectories {
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 50)]
        agents: i32,
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        #[arg(long)]
        out: String,
        #[arg(long, value_enum, default_value_t = HabitsFlag::On)]
        habits: HabitsFlag,
        #[arg(long)]
        include_replays: bool,
        #[arg(long, value_enum, default_value_t = TrajectoryProfile::Healthy)]
        profile: TrajectoryProfile,
        /// Percentage of agent pairs/cohort members to stress for hostile/exhausted profiles.
        #[arg(long, default_value_t = 25, value_parser = clap::value_parser!(u8).range(0..=100))]
        fraction: u8,
        /// Shortcut for `--profile scarcity`.
        #[arg(long, conflicts_with = "profile")]
        scarcity: bool,
        /// Shortcut for `--profile hostile`.
        #[arg(long, conflicts_with = "profile")]
        hostile: bool,
        /// Shortcut for `--profile exhausted`.
        #[arg(long, conflicts_with = "profile")]
        exhausted: bool,
    },
    /// Pre-registered OmniSoul promotion gate over fixed held-out worlds.
    Promotion {
        /// Release OmniSoul model to evaluate (tier-0 by default).
        #[arg(long, default_value = "training/artifacts/ladder/tier-0/model.onnx")]
        model_path: String,
    },
    /// Village social-sim soak with the utility or neural SOUL.
    Soak {
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        #[arg(long, default_value_t = 50)]
        agents: i32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Utility, neural, or both for an A/B comparison on the same seed.
        #[arg(long, value_enum, default_value_t = PolicyFlag::Utility)]
        policy: PolicyFlag,
        /// ONNX model path for `--policy neural`.
        #[arg(long, default_value = "training/artifacts/model.onnx")]
        onnx_path: String,
        /// Replay routine decisions from the per-agent habit cache.
        #[arg(long, value_enum, default_value_t = HabitsFlag::On)]
        habits: HabitsFlag,
        /// Explicit OMNI expertise conditioning; defaults to capable.
        #[arg(long, value_enum, default_value_t = ExpertiseFlag::Capable)]
        expertise: ExpertiseFlag,
    },
    /// Analytic AFK fast-forward: advance the village by an in-game span using
    /// the cold LOD ring, then print the returning-player digest.
    Ff {
        /// In-game days to skip (default one week).
        #[arg(long, default_value_t = 7)]
        days: u64,
        #[arg(long, default_value_t = 50)]
        agents: i32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Ratatui debug viewer over the live village sim. Without `--smoke` it
    /// opens the interactive TUI; `--smoke` renders one headless frame (CI-safe,
    /// no TTY) and exits.
    View {
        /// Render one frame to a test backend and exit 0 (no TTY needed).
        #[arg(long)]
        smoke: bool,
        #[arg(long, default_value_t = 50)]
        agents: i32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Latent-dialogue demo: script the canonical scene, render the observed
    /// conversation, then backfill one latent conversation on demand. Uses the
    /// offline mock unless `MW_TEXT_LIVE=1` selects the real TEXT backend.
    Dialogue {
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

/// Picks one of four unit steps from the entity's own RNG stream. It ignores the
/// observation entirely — enough to exercise the per-entity RNG and the intent
/// pipeline.
struct RandomWalk;

impl SoulPolicy for RandomWalk {
    fn decide(&mut self, _observation: &Observation, rng: &mut AgentRng) -> Intent {
        match rng.range_u32(4) {
            0 => Intent::Move { dx: 1, dy: 0 },
            1 => Intent::Move { dx: -1, dy: 0 },
            2 => Intent::Move { dx: 0, dy: 1 },
            _ => Intent::Move { dx: 0, dy: -1 },
        }
    }
}

fn start_positions(count: i32) -> Vec<(i32, i32)> {
    (0..count).map(|i| (i % 16, i / 16)).collect()
}

fn run_kernel(ticks: u64, entities: i32, seed: u64) {
    let pack = KernelPack::new();
    let mut world = World::with_pack(seed, &pack);
    for pos in start_positions(entities) {
        world.spawn(pos);
    }
    let mut policy = RandomWalk;
    for _ in 0..ticks {
        world.step(&pack, &mut policy);
    }
    println!(
        "seed={} entities={} ticks={} hash={:#018x}",
        seed,
        world.entity_count(),
        world.tick(),
        world.state_hash(&pack),
    );
}

fn run_soak(
    ticks: u64,
    agents: i32,
    seed: u64,
    policy: PolicyFlag,
    onnx_path: &str,
    habits: bool,
    expertise: ExpertiseLevel,
) {
    if matches!(policy, PolicyFlag::Both | PolicyFlag::OmniBoth) {
        let omni = policy == PolicyFlag::OmniBoth;
        let comparison = match if omni {
            soak::run_ab_omni_with_expertise(
                SoakConfig { seed, agents, ticks },
                onnx_path,
                expertise,
            )
        } else {
            soak::run_ab(
                SoakConfig { seed, agents, ticks },
                onnx_path,
            )
        } {
            Ok(comparison) => comparison,
            Err(e) => {
                eprintln!("neural soak failed: {e}");
                std::process::exit(1);
            }
        };
        print_ab(&comparison, if omni { "OmniSoul" } else { "NeuralSoul" });
        return;
    }
    let report = match policy {
        PolicyFlag::Utility => Ok(soak::run_with_habits(
            SoakConfig { seed, agents, ticks },
            habits,
        )),
        PolicyFlag::Neural => soak::run_neural(
            SoakConfig { seed, agents, ticks },
            onnx_path,
        ),
        PolicyFlag::Omni => soak::run_omni_with_expertise(
            SoakConfig { seed, agents, ticks },
            onnx_path,
            expertise,
        ),
        PolicyFlag::Both | PolicyFlag::OmniBoth => unreachable!("handled above"),
    };
    let report = match report {
        Ok(report) => report,
        Err(e) => {
            eprintln!("neural soak failed: {e}");
            std::process::exit(1);
        }
    };
    print_report(&report, policy, habits);
}

fn run_promotion(model_path: &str) {
    let report = match mw_sim::promotion::run_expertise_promotion(
        mw_sim::promotion::PromotionConfig::default(),
        model_path,
    ) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("promotion evaluation failed: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "promotion evaluator={} seeds={} agents={} ticks={} passed={}",
        report.evaluator,
        report.seeds.len(),
        report.agents,
        report.ticks,
        report.decision.passed
    );
    println!("machine_metrics_json:");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("promotion report serializes")
    );
}

fn print_report(report: &soak::SoakReport, policy: PolicyFlag, habits: bool) {
    println!(
        "soak seed={} agents={} ticks={} policy={:?} habits={}",
        report.cfg.seed, report.cfg.agents, report.cfg.ticks, policy, habits
    );
    println!(
        "ticks/sec={:.0} actions={} deaths={} (starvation)",
        report.ticks_per_sec(),
        report.total_actions(),
        report.deaths,
    );
    let hs = report.habit_stats;
    println!(
        "habits hits={} misses={} invalidations={} skipped={} hit_rate={:.1}% cache_entries={}",
        hs.hits,
        hs.misses,
        hs.invalidations,
        hs.scoring_calls_skipped,
        100.0 * hs.hit_rate(),
        report.habit_cache_sizes.iter().sum::<usize>(),
    );

    println!("final_hash={:#018x}", report.final_hash);
    let m = report.mean_needs();
    println!(
        "mean_needs hunger={:.0} energy={:.0} social={:.0}",
        m[0], m[1], m[2]
    );
    println!(
        "action histogram (max share {:.1}%):",
        100.0 * report.max_share()
    );
    for line in report.histogram_lines() {
        println!("{line}");
    }
}
fn print_ab(comparison: &soak::SoakComparison, policy_name: &str) {
    let utility = &comparison.utility;
    let neural = &comparison.neural;
    println!(
        "soak A/B seed={} agents={} ticks={} (same initial state)",
        utility.cfg.seed, utility.cfg.agents, utility.cfg.ticks
    );
    for (name, report) in [("UtilitySoul", utility), (policy_name, neural)] {
        let mean = report.mean_needs();
        println!(
            "{name}: deaths={} mean_needs hunger={:.0} energy={:.0} social={:.0} hash={:#018x}",
            report.deaths, mean[0], mean[1], mean[2], report.final_hash
        );
        println!("{name} action histogram:");
        for line in report.histogram_lines() {
            println!("{line}");
        }
    }
    println!(
        "deltas neural-utility: deaths={:+} mean_needs hunger={:+.1} energy={:+.1} social={:+.1}",
        comparison.deaths_delta,
        comparison.mean_needs_delta[0],
        comparison.mean_needs_delta[1],
        comparison.mean_needs_delta[2],
    );
    println!("action histogram deltas neural-utility:");
    for (id, delta) in comparison.histogram_delta.iter().enumerate() {
        if *delta == 0 {
            continue;
        }
        let name = mw_village::Action::from_id(id as u32).map_or_else(
            || format!("tool_{id}"),
            |action| format!("{action:?}").to_lowercase(),
        );
        println!("  {name:<7} {delta:+}");
    }
}

fn run_trajectories(
    seed: u64,
    agents: i32,
    ticks: u64,
    out: &str,
    config: mw_sim::trajectory::TrajectoryExportConfig,
) {
    match mw_sim::trajectory::export_trajectories_profile(seed, agents, ticks, out, config) {
        Ok(stats) => {
            println!(
                "trajectories seed={seed} agents={agents} ticks={ticks} habits={} include_replays={} profile={}",
                config.habits,
                config.include_replays,
                config.profile.as_str()
            );
            println!(
                "records={} bytes={} hash={:#018x} final_hash={:#018x}",
                stats.records, stats.bytes, stats.hash, stats.final_hash
            );
            println!("decision distribution:");
            for (id, count) in stats.per_tool.iter().enumerate() {
                if *count > 0 {
                    let name = mw_village::Action::from_id(id as u32)
                        .map(|a| format!("{a:?}").to_lowercase())
                        .unwrap_or_else(|| format!("tool_{id}"));
                    println!("  {name}={count}");
                }
            }
        }
        Err(e) => {
            eprintln!("trajectory export failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_ff(days: u64, agents: i32, seed: u64) {
    let ticks = days * TICKS_PER_DAY;
    let report = director::fast_forward(FfConfig {
        seed,
        agents,
        ticks,
        ..FfConfig::default()
    });
    println!("fast-forward seed={seed} agents={agents} days={days} ({ticks} ticks)");
    println!(
        "wall={:.3}s ticks/sec={:.0} events={} deaths={}",
        report.elapsed_secs,
        report.ticks_per_sec(),
        report.ledger_len,
        report.deaths,
    );
    let m = report.mean_needs;
    println!(
        "mean_needs hunger={:.0} energy={:.0} social={:.0}",
        m[0], m[1], m[2]
    );
    println!("final_hash={:#018x}", report.final_hash);
    println!("top events:");
    for line in &report.digest.top {
        println!("{line}");
    }
    println!("per-agent digest (first 8):");
    for line in report.digest.per_agent.iter().take(8) {
        println!("{line}");
    }
}

/// Open the debug viewer. `MW_TEXT_LIVE=1` selects the real TEXT backend for
/// dialogue backfill; otherwise the offline mock renders lines.
fn run_view(smoke: bool, agents: i32, seed: u64) {
    let live = std::env::var("MW_TEXT_LIVE").as_deref() == Ok("1");
    let cfg = mw_sim::view::ViewConfig { seed, agents, live };
    if smoke {
        // Prove the frame is non-empty and shaped, then exit 0 with no TTY.
        let buf = mw_sim::view::smoke_buffer(cfg);
        assert!(buf.contains("Map 16x16"), "map pane missing");
        println!("view --smoke ok: {} rows rendered", buf.lines().count());
    } else if let Err(e) = mw_sim::view::run(cfg) {
        eprintln!("viewer error: {e}");
        std::process::exit(1);
    }
}

/// Latent-dialogue demo: render the observed conversation, then backfill one
/// latent conversation on demand (DESIGN §4). Mock backend unless MW_TEXT_LIVE=1.
fn run_dialogue(seed: u64) {
    let (positions, scripts, focus) = demo();
    let mut scene = Scene::script(seed, &positions, &scripts);
    println!(
        "latent-dialogue demo: {} committed conversations",
        scene.log.len()
    );
    let live = std::env::var("MW_TEXT_LIVE").as_deref() == Ok("1");
    if live {
        match LlamaServerBackend::spawn(Config::default()) {
            Ok(b) => return play(&mut scene, &focus, &LlamaDialogue { backend: &b }),
            Err(e) => eprintln!("live TEXT backend unavailable ({e}); using mock"),
        }
    }
    play(&mut scene, &focus, &MockRenderer::new());
}

/// Render observed then backfill one latent conversation, printing both.
fn play<R: DialogueRenderer>(scene: &mut Scene, focus: &FocusPoint, r: &R) {
    let rendered = scene.render_observed(focus, r);
    println!("rendered {rendered} observed conversation(s) at focus");
    match (0..scene.log.len()).find(|&i| !scene.is_observed(i, focus)) {
        Some(i) => println!(
            "backfilled latent conversation #{i}: {}",
            scene.backfill(i, r)
        ),
        None => println!("no latent conversation to backfill"),
    }
}

fn main() {
    match Args::parse().cmd {
        Command::Run {
            ticks,
            entities,
            seed,
        } => run_kernel(ticks, entities, seed),
        Command::Promotion { model_path } => run_promotion(
            &std::env::var("MW_ONNX_PATH").unwrap_or(model_path),
        ),
        Command::Soak {
            ticks,
            agents,
            seed,
            policy,
            onnx_path,
            habits,
            expertise,
        } => run_soak(
            ticks,
            agents,
            seed,
            policy,
            &std::env::var("MW_ONNX_PATH").unwrap_or(onnx_path),
            matches!(habits, HabitsFlag::On),
            expertise.into(),
        ),
        Command::Trajectories {
            seed,
            agents,
            ticks,
            out,
            habits,
            include_replays,
            profile,
            fraction,
            scarcity,
            hostile,
            exhausted,
        } => {
            let profile = if scarcity {
                TrajectoryProfile::Scarcity
            } else if hostile {
                TrajectoryProfile::Hostile
            } else if exhausted {
                TrajectoryProfile::Exhausted
            } else {
                profile
            };
            run_trajectories(
                seed,
                agents,
                ticks,
                &out,
                mw_sim::trajectory::TrajectoryExportConfig {
                    habits: matches!(habits, HabitsFlag::On),
                    include_replays,
                    profile: profile.into(),
                    fraction,
                },
            )
        }
        Command::Ff { days, agents, seed } => run_ff(days, agents, seed),
        Command::View {
            smoke,
            agents,
            seed,
        } => run_view(smoke, agents, seed),
        Command::Dialogue { seed } => run_dialogue(seed),
    }
}
