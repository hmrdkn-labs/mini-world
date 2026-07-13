# mini-world

A sandbox, semi-AFK simulation platform where **every character is a small on-device AI model**. Characters act through a deterministic world, build memories and relationships, and keep moving while the player is away.

**Status: v0.5 review complete.** The shipped slice is a deterministic village with a utility-AI SOUL stub, shared TEXT rendering, latent dialogue, live LOD, analytic fast-forward, replay, a Ratatui viewer, and a deterministic per-character habit cache. The architecture and ratified contracts are documented in [DESIGN.md](DESIGN.md).

## Architecture at a glance

SOUL and TEXT have deliberately different jobs. SOUL is the per-character decision policy: it reads a fixed observation and emits an **intent**, never a world mutation. The kernel validates intents, applies them in canonical entity-id order, records the validated log, and owns all simulation truth. The shared TEXT model only verbalizes a committed `speak` act; it is not in the tick loop and cannot change simulation state.

Dialogue is **latent**: an off-screen conversation still applies its mechanical relationship outcome, but costs no TEXT render. When the player focuses the scene, the viewer renders and caches the line; an unobserved row can also be backfilled on demand. This makes dialogue cost track attention rather than population.

The workspace is split into small seams:

| Crate | Responsibility |
| --- | --- |
| `mw-core` | Deterministic integer tick kernel, intent validation/execution/logging, canonical hash, and the minimal `Observation`/`ScenarioPack`/`SoulPolicy`/`TextBackend` contracts. |
| `mw-village` | Village scenario pack: needs, inventory and ground items, action affordances, validation, outcomes, and analytic pack fast-forward/hash state. |
| `mw-agents` | Persona generation, structured memory/opinions, the ratified rich `AgentObs` schema, and the utility-AI SOUL implementation. |
| `mw-text` | Managed llama.cpp `llama-server` bridge for the shared Qwen3-0.6B Q4_0 TEXT backend, with prompt/KV-slot reuse. |
| `mw-sim` | Village wiring, live Director/LOD gates, analytic AFK fast-forward and digest, latent-dialogue demo, soak runner, and Ratatui TUI. |

## Verified results

These are measured results from the v0 and POST-REVIEW v0.5 gates, not projections:

| Area | Verified result |
| --- | --- |
| Determinism and replay | Same seed for 10,000 ticks produces an identical hash. Replaying `(seed, intent log)` — including `FfSegment` records — reproduces the full state hash, including pack state. v0.5 habit runs are deterministic; habit policy state is not kernel hash state. |
| Live simulation | 50 agents at **12,893 ticks/s** in release on an M4 Pro; the largest action-histogram share is **37.9%** in the v0 measurement. |
| POST-REVIEW health | **0 starvation deaths across 50 seeds × 10,000 ticks** (the earlier v0 emergent result was 8/50); eat share is **2.4–2.9%**. |
| Habit cache | Implemented in DESIGN block 4. Honest hit rate is **51.7%** in the 50×10k soak and **50.7%** in the 86,400-tick demo. At 50×10k, habits-on runs at **2,151 ticks/s** versus **1,088 ticks/s** with habits off (about 2×); deaths remain 0 and hashes are deterministic. |
| Habit-cache semantics | Speak and Give are always re-scored through the social passthrough; urgency invalidation and bounded TTLs prevent stale routine replay. Sociality stays within the reviewed ~2× no-cache bound. The earlier **82.7%** hit rate was pre-fix telemetry and partly reflected social lockout; it was not an honest measure of useful reuse. |
| Analytic fast-forward | One in-game week (**604,800 ticks**) in **0.014 s** (about **43M ticks/s analytic**); drift against the hot reference is **≤4%** with a **15%** bound, and the digest is deterministic. Fast-forward analytic gains are derived from the village pack constants (single source of truth); the suite remains within the **15%** drift bound. |
| Opinions, viewer, and model lifecycle | Opinion deltas are asymmetric and directional. TUI dialogue rendering is asynchronous, so the UI remains responsive during a live render. Stale `llama-server` instances are reaped on the next startup; PID reuse is guarded but the remaining TOCTOU is not atomic. |
| TEXT latency and cache | Qwen3-0.6B Q4_0 (**359 MiB**, via llama.cpp): warm render **79 ms**; prompt-token work falls from **104 → 1** with KV-slot reuse. |
| TEXT throughput | M4 Pro Metal: **pp512 2691 t/s**, **tg128 193 t/s**; CPU-only: **pp512 388 t/s**, **tg128 76 t/s**. |
| Latent dialogue | Unobserved conversations make **0 `TextBackend` calls** while relationship deltas still apply. Retroactive backfill renders act-coherent lines, caches them, and TEXT never mutates sim state. |
| Gates | **58 tests** green after the v0.5 review, plus clean `clippy -D warnings`, formatting, and `scripts/demo.sh`. Ratatui TUI was verified in a real PTY, and `view --smoke` exits 0 headless. |

The v0.5 review also fixed the feeding calibration, asynchronous render, stale-server cleanup, asymmetric opinions, and pack-constant drift items that were previously listed as backlog.

## Quickstart

### Prerequisites

- Rust and Cargo.
- Optional live TEXT: install llama.cpp and download the documented Qwen3-0.6B Q4_0 GGUF. The default path is `~/.cache/mini-world/models/`, and `MW_MODEL_PATH` overrides it.

```sh
brew install llama.cpp
mkdir -p "$HOME/.cache/mini-world/models"
curl -L --fail --retry 3 \
  -o "$HOME/.cache/mini-world/models/Qwen3-0.6B-Q4_0.gguf" \
  "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true"
```

The default demos use the offline dialogue renderer, so the model is not needed for the kernel, soak, fast-forward, or non-live viewer checks.

### Run the slice

```sh
cargo test --workspace
bash scripts/demo.sh
cargo run -p mw-sim -- soak
cargo run -p mw-sim -- view
```

`soak` runs the village loop. `scripts/demo.sh` builds release, runs a live day, fast-forwards a week, and exercises observed plus backfilled latent dialogue. `view` opens the interactive Ratatui viewer; use `cargo run -p mw-sim -- view --smoke` for one headless CI-safe frame.

Viewer keys: **arrows** move focus, **Tab** selects an agent, **j/k** move through conversations, **ENTER** backfills the selected latent row, **Space** pauses/resumes, **1** selects 1× speed, **8** selects 8× speed, **F** fast-forwards one day, and **q** quits. Set `MW_TEXT_LIVE=1` when opening `dialogue` or `view` to use the live TEXT backend.

Live model gates are opt-in because they spawn `llama-server`:

```sh
cargo test -- --ignored
```

## Roadmap

**Next: SOUL v1 distillation.** Generate LLM roleplay trajectories and distill them into a **1–5M parameter** policy that plugs into the existing `SoulPolicy` socket.

Remaining backlog:

- PID reuse TOCTOU in stale-server cleanup (narrowed and guarded, but not atomic).
- SOUL v1 distillation and its evaluation.
