//! Live gate tests. Ignored by default (spawn a real llama-server + model);
//! run with `cargo test -p mw-text -- --ignored`.

use std::process::Command;
use std::time::Instant;

use mw_text::{Config, LlamaServerBackend, PromptSpec};

fn spec() -> PromptSpec<'static> {
    PromptSpec {
        persona: "Bramble, a gruff but warm-hearted village blacksmith who speaks plainly.",
        act: "befriend",
        topic: "the newcomer who just arrived at the tavern",
        context: "Evening at the tavern. You noticed the stranger sitting alone, looking lost.",
    }
}

/// (1) One in-character line renders quickly once warm, with no leaked thinking.
#[test]
#[ignore]
fn renders_in_character_line_fast() {
    let backend = LlamaServerBackend::spawn(Config::default()).expect("spawn llama-server");
    // Warm the model/slot once so we measure decode, not cold load.
    let _ = backend.render_line(&spec(), 1).expect("warmup render");

    let start = Instant::now();
    let out = backend.render_line(&spec(), 1).expect("render");
    let elapsed = start.elapsed();

    println!("warm render {:?}: {:?}", elapsed, out.text);
    assert!(!out.text.trim().is_empty(), "expected a non-empty line");
    assert!(
        !out.text.contains("<think>") && !out.text.contains("</think>"),
        "thinking leaked into output: {:?}",
        out.text
    );
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "warm render took {elapsed:?}, expected <2s"
    );
}

/// (2) A second turn on the same conversation reuses the cached prefix, so it
/// evaluates far fewer prompt tokens (measurably faster prefill).
#[test]
#[ignore]
fn second_turn_reuses_prefill() {
    let backend = LlamaServerBackend::spawn(Config::default()).expect("spawn llama-server");
    let conv = 7;

    let first = backend.render_line(&spec(), conv).expect("first render");
    let second = backend.render_line(&spec(), conv).expect("second render");

    println!(
        "prefill turn1: prompt_n={} prompt_ms={:.1} (of {} tokens)",
        first.prompt_n, first.prompt_ms, first.prompt_tokens
    );
    println!(
        "prefill turn2: prompt_n={} prompt_ms={:.1} (of {} tokens)",
        second.prompt_n, second.prompt_ms, second.prompt_tokens
    );
    assert!(
        second.prompt_n < first.prompt_n,
        "second turn should evaluate fewer prompt tokens (cache reuse): {} !< {}",
        second.prompt_n,
        first.prompt_n
    );
}

/// (3) Dropping the backend leaves no orphaned llama-server process.
#[test]
#[ignore]
fn drop_kills_server() {
    let backend = LlamaServerBackend::spawn(Config::default()).expect("spawn llama-server");
    let pid = backend.pid();
    assert!(pid_alive(pid), "server should be running at pid {pid}");

    drop(backend);
    // Give the OS a beat to reap.
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(!pid_alive(pid), "server pid {pid} still alive after drop");
}

/// True if a process with `pid` currently exists (via `kill -0`).
fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
