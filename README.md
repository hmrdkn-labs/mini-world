# mini-world

[![ci](https://github.com/hmrdkn-labs/mini-world/actions/workflows/ci.yml/badge.svg)](https://github.com/hmrdkn-labs/mini-world/actions/workflows/ci.yml)

A deterministic village simulation for local-AI hobbyists: a 50-character village that runs on a laptop with no API key, fast-forwards a week in well under a second, and replays to the same state hash from the same seed. Characters act through a deterministic world, build memories and relationships, and keep going while you are away.

**What the default run is.** Each character is driven by a hand-written **utility-AI policy**, not a language model. Dialogue text comes from an **offline mock renderer** unless you opt in to a live model (`MW_TEXT_LIVE=1`, which needs `llama.cpp` and the Qwen3-0.6B GGUF, see [Quickstart](#quickstart)). A single small text model is shared by all characters and only writes a line when someone looks at a conversation; it never decides what a character does. A trained per-character policy (SOUL v1) is not shipped.

**Status: v0.5 plus a browser observatory.** Shipped today:

- a deterministic village kernel with the utility-AI SOUL policy, latent dialogue, live level-of-detail, analytic fast-forward with a returning-player digest, replay, and a per-character habit cache;
- a Ratatui terminal viewer (`mw-sim view`);
- a **loopback-only browser observatory**: the `mw-host` server (JSON HTTP + SSE on `127.0.0.1`) and the React/Vite frontend in `web/`. It is minimal: a 16×16 map, status, and a single "Step one tick" control (1 tick = 1 in-game second, so a day is 86,400 steps). No names, inspector, or dialogue in the browser yet. You start it as **two processes in two terminals** (see [Browser observatory](#browser-observatory)); there is no one-command launcher yet.

The architecture and ratified contracts are in [DESIGN.md](DESIGN.md); the browser layer's design is in the [browser simulation layer plan](docs/web-ui-simulation-plan.md).

## Architecture at a glance

SOUL and TEXT have deliberately different jobs. SOUL is the per-character decision policy: it reads a fixed observation and emits an **intent**, never a world mutation. The kernel validates intents, applies them in canonical entity-id order, records the validated log, and owns all simulation truth. The shared TEXT model only verbalizes a committed `speak` act; it is not in the tick loop and cannot change simulation state.

Dialogue is **latent**: an off-screen conversation still applies its mechanical relationship outcome, but costs no TEXT render. When the player focuses the scene, the viewer renders and caches the line; an unobserved row can also be backfilled on demand. This makes dialogue cost track attention rather than population.

The workspace is eight crates (`ls crates`):

| Crate | What it is today |
| --- | --- |
| `mw-core` | Deterministic integer tick kernel: intent validation, execution and logging, canonical state hash, and the minimal `Observation` / `ScenarioPack` / `SoulPolicy` / `TextBackend` contracts. |
| `mw-village` | The one shipped scenario pack: needs, inventory and ground items, action affordances, validation, outcomes, and analytic fast-forward/hash state. |
| `mw-agents` | Persona generation, structured memory and opinions, the rich `AgentObs` schema, the habit cache, and the utility-AI SOUL policy. |
| `mw-text` | Managed `llama-server` (llama.cpp) bridge for the shared Qwen3-0.6B Q4_0 TEXT backend, with prompt/KV-slot reuse. Used only when `MW_TEXT_LIVE=1`. |
| `mw-sim` | The `mw-sim` CLI (`run`, `soak`, `ff`, `dialogue`, `view`, `promotion`, `trajectories`), Director/LOD gates, fast-forward digest, latent-dialogue demo, and the Ratatui viewer. |
| `mw-runtime` | The headless `SimulationController` (single-thread owner of the world, pack and policy state) and the versioned snapshot/command DTOs shared by the TUI and the host. |
| `mw-host` | The `mw-host` binary: an actor thread around the controller plus a loopback-only axum server (`/v1/capabilities`, `/v1/snapshot`, `/v1/commands`, `/v1/events`). |
| `mw-neural` | Optional ONNX (tract) inference for a neural SOUL experiment, used by `mw-sim soak --policy neural`. It needs a model file that is not in the repository, so its tests skip on a fresh clone. |

## Measured results

Every number below was re-measured on **2026-11-26** on an **Apple M5** (`sysctl -n machdep.cpu.brand_string`), a shared machine with a load average of about 3–4, from branch `phase-a` (`6dac5eb` plus the documentation changes), release build unless noted. They replace the older figures from an M4 Pro, which this README no longer repeats. Commands are run from the repository root; `B=target/release/mw-sim` after `cargo build --release -p mw-sim`.

| Area | Result | Command |
| --- | --- | --- |
| Determinism | Seed 1, 50 agents, 10,000 ticks: final hash `0xe3122e71457daa14` on two separate runs; seed 2 gives `0xc2b39c37df7f5277`. Replay of `(seed, intent log)`, including fast-forward segments, reproduces the state hash (covered by `cargo test`). Habit-cache state is not kernel hash state. | `$B soak` (twice), `$B soak --seed 2` |
| Live simulation speed | 50 agents, 86,400 ticks (one day): **17,640 ticks/s**. 50 agents × 10,000 ticks, seed 1: **19,447 ticks/s** with habits on, **15,569 ticks/s** with habits off (about 1.25×). Means over 50 seeds: 16,578 (on) and 12,059 (off). | `bash scripts/demo.sh`; `$B soak --habits on\|off` |
| Habit cache | Hit rate **44.7%** (50×10k, seed 1) and **44.3%** (86,400-tick demo day). Across 50 seeds the hit rate ranges 14.0%–54.9%, mean 47.3%. Speak and Give are always re-scored. | `$B soak`; `bash scripts/demo.sh` |
| Starvation, 50 seeds × 10,000 ticks | **Habits off: 0 deaths in all 50 seeds**, eat share 2.3–2.8% of actions. **Habits on (the default): 0 deaths in 49 seeds, but seed 22 loses 44 of 50 agents to starvation** (deterministic, repeated three times). The earlier "0 deaths, habits on" claim does not hold. | `for s in $(seq 1 50); do $B soak --seed $s --habits on; done` (and `off`) |
| Action mix | Largest action share: 32.7% (`speak`, seed 1, habits on); 22.2%–39.0% across 49 seeds excluding seed 22. With habits off `speak` dominates at 60.0% (seed 1), up to 64.1%. | same soak runs |
| Analytic fast-forward | One in-game week (604,800 ticks): **0.014–0.024 s** (25.7M–44.4M ticks/s) in three runs, 0.015 s in the demo. Drift against the hot reference is **5.3%–7.4%** (hunger −6.2%, energy −5.3%, social −7.4%; debug-build test), inside the 15% bound. | `$B ff --days 7` (×3); `cargo test -p mw-sim --test fast_forward -- --nocapture` |
| Latent dialogue | An unobserved conversation makes no TEXT call; backfill renders a line and caches it (demo step 3 backfilled one line with the offline mock). | `bash scripts/demo.sh` |
| Tests | **102 passed, 0 failed, 5 ignored** by `cargo test --workspace` on `phase-a` (debug, no model files present). The 5 ignored are opt-in live-model tests (4 in `mw-text`, 1 in `mw-sim`). | `cargo test --workspace` |
| Tests that skip on a fresh clone | **10** of the 102 print `skipped: <path> not present` and pass without exercising the model: 7 need `training/artifacts/model.onnx` and 3 need `training/artifacts/ladder/tier-0/model.onnx`. `training/` ignores those files, so they are absent on every fresh clone. They ran and passed when the files were present (A1 run). | `cargo test --workspace -- --nocapture \| grep skipped:` |

Live TEXT numbers (Qwen3-0.6B Q4_0, llama.cpp 0.9.0 build 12110 `cd959bae9`, same M5, Metal build; 3 repetitions of `llama-bench`):

| Measure | Result | Command |
| --- | --- | --- |
| Model file | `Qwen3-0.6B-Q4_0.gguf` is **382,156,480 bytes = 364.5 MiB (382 MB)**. `llama-bench` reports 358.78 MiB because it counts tensor data only; the old "359 MiB" was that figure. | `ls -l ~/.cache/mini-world/models/`; Hugging Face tree API |
| Prompt / decode throughput | pp512 **8,242 t/s**, tg128 **213 t/s** (default backends, GPU offload). With `-ngl 0` the same build still reports a BLAS/Metal backend; pp512 **318 t/s**, tg128 **125 t/s**. These are not comparable to the earlier M4 Pro table, and the `-ngl 0` row is not a clean CPU-only measurement. | `llama-bench -m <gguf> -p 512 -n 128 -r 3` (and `-ngl 0`) |
| Warm render and cache reuse | Warm render **88 ms**; prompt tokens processed fall from **104 to 1** on the second turn with KV-slot reuse (prefill 30.7 ms → 6.0 ms). | `cargo test --workspace -- --ignored --nocapture` |

The TEXT model comparison in [DESIGN.md](DESIGN.md) (Qwen3-0.6B vs Qwen3.5-0.8B vs Gemma3-270M, M4 Pro, 2026-07-15) is a dated historical result; only the Qwen3-0.6B row was re-run here.

## Quickstart

### Prerequisites

- Rust and Cargo.
- Node.js and npm, only for the browser observatory.
- Optional live TEXT: `llama.cpp` and the Qwen3-0.6B Q4_0 GGUF. The default model path is `~/.cache/mini-world/models/`; `MW_MODEL_PATH` overrides it.

```sh
brew install llama.cpp
mkdir -p "$HOME/.cache/mini-world/models"
curl -L --fail --retry 3 \
  -o "$HOME/.cache/mini-world/models/Qwen3-0.6B-Q4_0.gguf" \
  "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true"
```

The default demo uses the utility-AI policy and the offline mock renderer, so no model, key, or network is needed.

### Run the headless demo

```sh
cargo test --workspace
bash scripts/demo.sh
cargo run -p mw-sim -- soak
```

`scripts/demo.sh` builds release, runs a live in-game day, fast-forwards a week and prints the digest, and renders one observed plus one backfilled latent conversation. Set `MW_TEXT_LIVE=1` to render dialogue with the real model.

### Terminal viewer

```sh
cargo run -p mw-sim -- view          # interactive; `--smoke` renders one headless frame
```

Keys: **arrows** move focus, **Tab** selects an agent, **j/k** move through conversations, **ENTER** backfills the selected latent row, **Space** pauses/resumes, **1** / **8** select 1× / 8× speed, **F** opens a one-day fast-forward report (it does not change the live world), **q** quits. `MW_TEXT_LIVE=1` uses the live TEXT backend.

### Browser observatory

Two terminals (verified 2026-11-26):

```sh
# terminal 1: simulation host, loopback only, prints "mw-host ready: http://127.0.0.1:7878"
cargo run --release -p mw-host            # optional port: -- --port 7878

# terminal 2: frontend dev server, proxies /v1 to the host
cd web && npm ci && npm run dev           # open the printed http://localhost:5173
```

`mw-host` binds `127.0.0.1` only and refuses non-loopback addresses; there is no authentication and no remote control surface. Frontend tests: `npm test`, `npm run build`, and `npm run e2e` (Playwright; needs `npx playwright install` first).

Live model gates are opt-in because they spawn `llama-server`:

```sh
cargo test --workspace -- --ignored
```

## Roadmap

**Next:** a one-command observatory launcher, readable names and an agent inspector in the browser, and a prose returning-player digest.

**SOUL v1 distillation:** generate LLM roleplay trajectories and distill them into a small (1–5M parameter) policy that plugs into the existing `SoulPolicy` socket. Not shipped; the default run does not use it.

**Not planned now:** a semi-AFK game and extra scenario packs (for example MOBA or football).

Known backlog: the PID-reuse race in stale-server cleanup is narrowed and guarded but not atomic; habits-on seed 22 starves most of the village (see Measured results).

## Licence

Apache-2.0. See [LICENSE](LICENSE). The Qwen3-0.6B model and the unsloth GGUF are separately licensed (Apache-2.0 upstream) and are not included in this repository.
