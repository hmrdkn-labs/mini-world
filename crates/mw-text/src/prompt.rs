//! Persona prompt construction for the shared TEXT model.
//!
//! TEXT only ever *verbalizes* a decision SOUL already made (DESIGN.md §5), so
//! the prompt hard-constrains the model to a single in-character line for the
//! committed act. Persona lives in the system message and per-turn detail in
//! the user message: the system prefix is identical across a character's turns,
//! which is exactly the stable prefix llama-server's prompt cache reuses.

use serde_json::{json, Value};

/// A committed speak act ready to be rendered into dialogue. String-shaped so
/// callers that hold the scenario manifest resolve codes to names themselves;
/// the [`crate::LlamaServerBackend`] trait impl maps raw [`mw_core::SpeakRequest`]
/// codes through best-effort defaults.
pub struct PromptSpec<'a> {
    /// One- or two-line persona summary (name, traits, current mood).
    pub persona: &'a str,
    /// The committed act, e.g. `befriend`, `taunt`, `trade`.
    pub act: &'a str,
    /// What the line is about.
    pub topic: &'a str,
    /// Recent events / relationship / scene.
    pub context: &'a str,
}

/// Build the OpenAI-shaped `messages` array. Qwen3 thinking is disabled inline
/// via the `/no_think` control token so we never pay for (or have to strip) a
/// reasoning trace on a one-line social utterance.
pub fn messages(spec: &PromptSpec<'_>) -> Value {
    let system = format!(
        "You are {}. Stay fully in character. Reply with exactly one short \
         line of spoken dialogue — no narration, no quotation marks, no stage \
         directions, at most one sentence.",
        spec.persona.trim()
    );
    let user = format!(
        "Scene: {}\nYou have decided to {} regarding {}. Say your line now. /no_think",
        spec.context.trim(),
        spec.act.trim(),
        spec.topic.trim(),
    );
    json!([
        { "role": "system", "content": system },
        { "role": "user", "content": user },
    ])
}

/// Strip any Qwen3 `<think>…</think>` block the model emits despite `/no_think`,
/// plus surrounding whitespace and wrapping quotes, leaving the bare line.
pub fn clean(raw: &str) -> String {
    let mut s = raw;
    if let Some(end) = s.find("</think>") {
        s = &s[end + "</think>".len()..];
    }
    let s = s.trim();
    let s = s.strip_prefix('"').unwrap_or(s);
    let s = s.strip_suffix('"').unwrap_or(s);
    s.trim().to_string()
}
