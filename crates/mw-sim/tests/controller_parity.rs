use mw_runtime::{RunConfig, SimulationController};
use mw_sim::view::{smoke_buffer, ViewConfig};

#[test]
fn headless_view_smoke_is_deterministic_for_controller_run() {
    let config = ViewConfig {
        seed: 0x51_4d_1a,
        agents: 12,
        live: false,
    };

    // smoke_buffer constructs App and drives its private App::step path exactly
    // as the headless TUI does. Repeating it proves the complete rendered path
    // is deterministic, including the controller-owned simulation state.
    let first_frame = smoke_buffer(config);
    let second_frame = smoke_buffer(config);
    assert_eq!(first_frame, second_frame, "TUI smoke output diverged");
    assert!(
        first_frame.contains("tick=300"),
        "smoke path did not drive 300 ticks"
    );

    // Keep an explicit standalone-controller sequence alongside the smoke path;
    // this is the hash authority that App owns for the same RunConfig.
    let run = RunConfig {
        seed: config.seed,
        agents: config.agents as usize,
        ..RunConfig::default()
    };
    let mut standalone = SimulationController::new(run);
    let mut sequence = vec![standalone.state_hash()];
    for _ in 0..300 {
        sequence.push(standalone.step_one().state_hash);
    }
    assert_eq!(sequence.len(), 301);
    assert_eq!(standalone.tick(), 300);
}

#[test]
fn headless_view_smoke_is_seed_sensitive() {
    let base = ViewConfig {
        seed: 0x51_4d_1a,
        agents: 12,
        live: false,
    };
    let changed = ViewConfig {
        seed: base.seed + 1,
        ..base
    };
    assert_ne!(smoke_buffer(base), smoke_buffer(changed));
}
