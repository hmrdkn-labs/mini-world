//! Headless smoke gate for the ratatui debug viewer: build the app, render one
//! frame to a `TestBackend`, and assert the frame is non-empty and carries the
//! map glyphs plus at least one agent row — all without a TTY, so CI is safe.

use mw_sim::view::{smoke_buffer, ViewConfig};

#[test]
fn smoke_renders_map_and_agent_row() {
    let buf = smoke_buffer(ViewConfig {
        seed: 1,
        agents: 50,
        live: false,
    });

    assert!(!buf.trim().is_empty(), "rendered buffer is empty");
    // Map pane: its title plus fixture location glyphs (bakery/home) prove the
    // 16x16 grid actually drew.
    assert!(buf.contains("Map 16x16"), "map pane title missing");
    assert!(
        buf.contains('B') && buf.contains('H'),
        "map location glyphs missing"
    );
    // Agent inspector: the selected-agent header is the "agent row".
    assert!(buf.contains("Agent #"), "no agent row rendered");
    // The other panes are present too.
    assert!(buf.contains("Event feed"), "event feed pane missing");
    assert!(buf.contains("Conversations"), "dialogue pane missing");
}
