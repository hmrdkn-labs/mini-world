//! Deterministic latent-dialogue gate.
//!
//! Runs the whole pipeline against a [`MockRenderer`] — no live model — and
//! asserts: (a) it is deterministic in the seed, (b) the mechanical outcome
//! (opinion delta on both parties) always applies while text stays latent,
//! (c) attention-gating renders exactly the observed conversation, and (d) the
//! one-way street holds — rendering text never changes sim state.

use mw_agents::dialogue::MockRenderer;
use mw_sim::dialogue::{demo, Scene};

const SEED: u64 = 7;

#[test]
fn pipeline_is_deterministic() {
    let (positions, scripts, _) = demo();
    let a = Scene::script(SEED, &positions, &scripts);
    let b = Scene::script(SEED, &positions, &scripts);
    assert_eq!(
        a.state_hash(),
        b.state_hash(),
        "same seed must reproduce the same sim state"
    );
}

#[test]
fn mechanical_outcome_applies_while_text_latent() {
    let (positions, scripts, _) = demo();
    let scene = Scene::script(SEED, &positions, &scripts);

    assert_eq!(scene.log.len(), 3, "three committed conversations");
    for row in scene.log.rows() {
        assert!(row.text.is_none(), "every conversation starts latent");
        assert_ne!(row.outcome, 0, "the mechanical outcome is recorded");
        // Both parties applied the opinion delta through their own memory.
        assert_ne!(
            scene.opinion(row.speaker, row.listener),
            0,
            "speaker's opinion of listener must shift"
        );
        assert_ne!(
            scene.opinion(row.listener, row.speaker),
            0,
            "listener's opinion of speaker must shift"
        );
    }
}

#[test]
fn text_is_one_way_and_attention_gated() {
    let (positions, scripts, focus) = demo();
    let mut scene = Scene::script(SEED, &positions, &scripts);

    let before = scene.state_hash();
    let mock = MockRenderer::new();

    let rendered = scene.render_observed(&focus, &mock);
    assert_eq!(rendered, 1, "exactly one conversation is in focus");
    assert_eq!(
        mock.calls(),
        1,
        "only the observed conversation costs a render"
    );
    assert_eq!(
        scene.state_hash(),
        before,
        "rendering text must not influence sim state (one-way street)"
    );

    let with_text = scene.log.rows().iter().filter(|r| r.text.is_some()).count();
    assert_eq!(with_text, 1, "the two unobserved rows stay latent");
}

#[test]
fn backfill_renders_then_caches() {
    let (positions, scripts, focus) = demo();
    let mut scene = Scene::script(SEED, &positions, &scripts);
    let mock = MockRenderer::new();
    scene.render_observed(&focus, &mock);
    assert_eq!(mock.calls(), 1);

    // A still-latent conversation renders on demand and is coherent with its act.
    let latent = (0..scene.log.len())
        .find(|&i| !scene.is_observed(i, &focus))
        .expect("a latent conversation");
    let line = scene.backfill(latent, &mock);
    assert_eq!(mock.calls(), 2, "backfill costs one render");
    assert!(!line.is_empty());
    // The taunt/newcomer pair — the mock voices the committed act and topic.
    assert!(
        line.contains("taunt"),
        "line reflects the committed act: {line:?}"
    );
    assert!(
        line.contains("the newcomer"),
        "line reflects the topic: {line:?}"
    );

    // Re-inspecting is free: the cached line is returned verbatim.
    let again = scene.backfill(latent, &mock);
    assert_eq!(mock.calls(), 2, "cached inspect adds no render");
    assert_eq!(line, again);
}
