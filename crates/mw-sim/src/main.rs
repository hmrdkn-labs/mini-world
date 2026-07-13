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
use mw_text::{Config, LlamaServerBackend};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HabitsFlag {
    On,
    Off,
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
    /// Export one JSONL trajectory record per scored SOUL decision.
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
    },
    /// Village social-sim soak with the utility SOUL.
    Soak {
        #[arg(long, default_value_t = 10_000)]
        ticks: u64,
        #[arg(long, default_value_t = 50)]
        agents: i32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Replay routine decisions from the per-agent habit cache.
        #[arg(long, value_enum, default_value_t = HabitsFlag::On)]
        habits: HabitsFlag,
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

fn run_soak(ticks: u64, agents: i32, seed: u64, habits: bool) {
    let report = soak::run_with_habits(
        SoakConfig {
            seed,
            agents,
            ticks,
        },
        habits,
    );
    println!(
        "soak seed={} agents={} ticks={} habits={}",
        report.cfg.seed, report.cfg.agents, report.cfg.ticks, habits
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

fn run_trajectories(
    seed: u64,
    agents: i32,
    ticks: u64,
    out: &str,
    habits: bool,
    include_replays: bool,
) {
    match mw_sim::trajectory::export_trajectories(seed, agents, ticks, out, habits, include_replays)
    {
        Ok(stats) => {
            println!("trajectories seed={seed} agents={agents} ticks={ticks} habits={habits} include_replays={include_replays}");
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
        Command::Soak {
            ticks,
            agents,
            seed,
            habits,
        } => run_soak(ticks, agents, seed, matches!(habits, HabitsFlag::On)),
        Command::Trajectories {
            seed,
            agents,
            ticks,
            out,
            habits,
            include_replays,
        } => run_trajectories(
            seed,
            agents,
            ticks,
            &out,
            matches!(habits, HabitsFlag::On),
            include_replays,
        ),
        Command::Ff { days, agents, seed } => run_ff(days, agents, seed),
        Command::View {
            smoke,
            agents,
            seed,
        } => run_view(smoke, agents, seed),
        Command::Dialogue { seed } => run_dialogue(seed),
    }
}
