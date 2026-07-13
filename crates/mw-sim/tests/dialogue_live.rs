//! Live latent-dialogue gate. Ignored by default (spawns a real llama-server +
//! model); run with `cargo test -p mw-sim --test dialogue_live -- --ignored`.
//!
//! End-to-end: three committed conversations (two out of observer range, one in)
//! render exactly one line at render time (counter == 1); the two latent rows
//! keep their opinion deltas with no text; backfilling one latent conversation
//! renders on demand (counter == 2) and a re-inspect is served from cache
//! (counter stays 2).

use mw_sim::dialogue::{demo, LlamaDialogue, Scene};
use mw_text::{Config, LlamaServerBackend};

const SEED: u64 = 7;

#[test]
#[ignore]
fn latent_dialogue_end_to_end() {
    let backend = LlamaServerBackend::spawn(Config::default()).expect("spawn llama-server");
    let renderer = LlamaDialogue { backend: &backend };

    let (positions, scripts, focus) = demo();
    let mut scene = Scene::script(SEED, &positions, &scripts);
    assert_eq!(scene.log.len(), 3, "three committed conversations");

    // Render at "render time": only the in-focus conversation costs a TEXT call.
    let rendered = scene.render_observed(&focus, &renderer);
    assert_eq!(rendered, 1, "exactly one conversation observed");
    assert_eq!(
        backend.render_count(),
        1,
        "attention-gated: one render only"
    );

    // The two out-of-range conversations stay latent, but their mechanical
    // outcome (opinion delta, both parties) is already applied.
    let latent: Vec<usize> = (0..scene.log.len())
        .filter(|&i| !scene.is_observed(i, &focus))
        .collect();
    assert_eq!(latent.len(), 2);
    for &i in &latent {
        let row = scene.log.rows()[i].clone();
        assert!(
            row.text.is_none(),
            "unobserved conversation must stay latent"
        );
        assert_ne!(scene.opinion(row.speaker, row.listener), 0);
        assert_ne!(scene.opinion(row.listener, row.speaker), 0);
    }

    // Retroactive backfill: render one latent conversation on demand.
    let i = latent[0];
    let row = scene.log.rows()[i].clone();
    let line = scene.backfill(i, &renderer);
    println!(
        "backfilled conv (act={}, topic={}): {line:?}",
        row.act, row.topic
    );
    assert_eq!(backend.render_count(), 2, "backfill costs one more render");
    assert!(!line.trim().is_empty(), "backfilled line must be non-empty");
    assert!(
        !line.contains("<think>") && !line.contains("</think>"),
        "thinking leaked into the line: {line:?}"
    );

    // Second inspect is served from cache — no new TEXT call.
    let again = scene.backfill(i, &renderer);
    assert_eq!(backend.render_count(), 2, "cached inspect adds no render");
    assert_eq!(line, again, "cached line is identical");
}
