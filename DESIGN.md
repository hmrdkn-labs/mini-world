# mini-world — Architecture Design v1

> Sandbox semi-AFK simulation platform where every character is driven by small on-device AI models.
> Design date: 2026-07-12. Research-grounded; sources linked inline. Claims marked `[INFERENCE]` are engineering estimates pending our own benchmarks.

## Concept

Each character has:
- a **SOUL model** — a tiny policy network (~1–5M params) that decides *what to do* every simulation tick: move, attack, interact, speak. It has a "digital body": a tool-calling interface over the scenario's action manifest.
- a shared **TEXT model** — a small LLM (~0.6B) that renders dialogue *only when someone is watching*.

The platform ("base OS") hosts scenario packs on one kernel: village social sims, advanced NPCs, AFK autobattlers/MOBA management, football-manager-style stat games.

Targets consumer laptops and phones (GPU or CPU). No cloud dependency for core play.

## Load-bearing decisions

### 1. Models never mutate the world — they emit intents

The world kernel validates and executes every intent. Consequences:

- **Determinism & replay.** Cross-device bitwise NN equality is unachievable (thread scheduling, backend kernels, quantization differences). Therefore the **validated-intent log is the ground truth**: record sim seed + model hash + backend ID + chosen intents and analytic fast-forward segments. Replay and multiplayer verification replay intents; AFK fast-forward consumes each `FfSegment` analytically. NN output is always advisory.
- **Canonical application order.** `World::apply_intents` sorts each tick's batch by entity id (`index`, then `generation`) before validation, execution, and logging. Per-entity RNG draws are keyed by `(seed, entity, stream, tick)`, so a character's draws do not depend on iteration order; conflicting effects are resolved by the canonical entity-id apply order rather than by submission order.
- **Swappable brains.** Utility-AI stub today, distilled net tomorrow, cloud LLM for hero characters — same socket.
- **Scenario rules are uncheatable.** The kernel rejects invalid calls (range, cooldowns, resources) regardless of what the brain proposes.

### 2. Shared weights + per-character state, never per-character weights

1,000 characters × unique 10M-param models = gigabytes of cold-cache weights — dead on a phone. Instead: one hot SOUL network shared by all characters, each conditioned on its own persona/memory/experience state. Character identity lives in **data** (~10–100KB each), not weights. A thousand divergent individuals ≈ one MP3.

### 3. SOUL is a tool-caller, the body is the toolset

Each scenario pack registers an **action manifest** — typed action schemas, MCP-tool-shaped:

```
move(direction | target)        attack(target, style?)
interact(target, verb)          speak(target, act, topic)
craft(recipe)                   trade(target, offer)
follow(target)                  flee(threat)                idle()
```

- Every SOUL tick sees the observation **plus which tools the body currently affords** (masked action space: no `attack` if disarmed, no `craft` away from a bench).
- SOUL output = discrete head (pick tool) + pointer head (pick target among observed entities) + param head (scalars).
- The kernel is the tool executor: validate → apply → emit result events into the character's memory. Call → result → context, at simulation speed.
- 10–30 tools per scenario is a *classification* problem, not a language problem — that's why SOUL stays tiny.
- The manifest is the scenario-pack API surface: football pack registers `pass/shoot/press/train`; MOBA pack registers `farm/gank/push/recall`. Same SOUL architecture, different manifest + retrained head.

### 4. TEXT is attention-gated ("latent dialogue")

SOUL emits `speak(target, act, topic)`; the mechanical outcome (relationship delta) always applies, but words are rendered **only when observed** — player nearby, log open, replay inspection. Off-screen conversations stay latent, generated on demand.

Measured phone decode makes this mandatory, not optional: a 0.6B-class model yields single-digit concurrent dialogues per device. Dialogue cost must scale with *attention*, not population.

**Research status: no shipped prior art found** for render-only-when-observed NPC dialogue (closest: AGA's ~100-token social summaries). This is a novel design bet — v0 must prove it early.

### 5. Say/do coherence: TEXT renders decisions, never makes them

Project Sid's PIANO identified say/do divergence as the core coherence failure (chat says pickaxe, body does otherwise). Enforcement: TEXT is constrained to verbalize the act SOUL already committed. One decision-maker per character.

## Building blocks

```mermaid
flowchart LR
    WK[1 World Kernel<br/>deterministic tick ECS] -->|observation| OE[3 Observation Encoder]
    OE --> HC[4 Habit Cache]
    HC -->|novel context| SOUL[5 SOUL<br/>1-5M policy]
    HC -->|routine: cached intent| WK
    PG[8 Persona/Genome] --> SOUL
    MEM[7 Memory] --> OE
    SOUL -->|intent| WK
    WK -->|events| MEM
    SOUL -->|speak intent + steering| TEXT[6 TEXT<br/>shared 0.6B LM]
    PG --> TEXT
    TEXT -->|dialogue when observed| WK
    DIR[10 Director / LOD] -.tick rates.-> SOUL
    SC[11 Scenario Pack] -.manifest + schemas.-> WK
    RT[9 Inference Runtime] -.executes.-> SOUL
    RT -.executes.-> TEXT
```

### 1. World Kernel
Deterministic, fixed-timestep, tick-based, ECS-style. Seeded RNG, no wall-clock. Owns all truth: entities, positions, stats, relationships, inventory. Validates and executes intents. The canonical hash also folds pack-owned state through `ScenarioPack::hash_state`; this is what makes AFK progression (= fast-forwarding the sim) and replay (= debugging + training-data generation) verifiable.

### 2. Action Manifest (the digital body)
Per-scenario typed tool schemas with affordance masking. See load-bearing decision 3.

### 3. Observation Encoder
The ratified rich observation schema is `mw_agents::obs::AgentObs`: fixed-size self needs, eight nearest enriched neighbors, event buckets, goal, and afforded-tool mask. The kernel's `mw_core::Observation` is the minimal seam — tick, position, nearest slots, event count, and tool mask — used for one kernel scan and affordance masking. `World::observe_for_policy` constructs that minimal observation once, then passes it to `ScenarioPack::afforded_tools`; the pack fills the mask without a second neighbor scan. The SOUL-facing `AgentObs` can therefore evolve without changing the kernel seam.

### 4. Habit Cache *(added from research)*
Sits between observation and SOUL. Caches decisions as **plan→actions + explicit validity predicates**; replays while predicates hold, invokes SOUL only on novel/invalidated contexts. Evidence: Affordable Generative Agents cut token cost to 31–43% of baseline with *higher* human-rated quality using exactly this pattern (cosine plan-match 0.97 + state conditions) — [arXiv:2402.02053](https://arxiv.org/abs/2402.02053).

Dual role: it is also the **habit tier of character plasticity** — each cache is unique, grown from that character's history. Characters get cheaper as they settle into routines, like the real world.

### 5. SOUL model
Tiny discrete policy, ~1–5M params, **not an LLM**. Input: encoded observation + persona vector + experience embedding. Output: tool + target + params (see decision 3). Runs every N ticks per character (LOD-dependent), batched.

Feasibility evidence:
- Decision Transformer: structured state→action at ~1.1M params (~1–2MB int8) — [repo](https://github.com/kzl/decision-transformer)
- TD-MPC2: 1M/5M checkpoints across 104 control tasks — [repo](https://github.com/nicklashansen/tdmpc2)
- Throughput `[INFERENCE]`: 1–10M INT8 policy, batched — ~1k–10k agents/s laptop, ~200–2k/s phone; memory-bandwidth-bound; benchmark required.

Training roadmap:
- **v0**: hand-written utility-AI scorer behind the same interface (zero ML; ships the game loop).
- **Original v1 roadmap (historical design target):** distill an LLM playing characters. The corrected v1 evidence below uses Spark grounding and an OMNI manifest-conditioned policy; the original recipe remains a future research option, not a reported result.
### Pilot evidence: 1K Spark-labeled OMNI ladder (pipeline validation only)

The first trained OMNI run is evidence that the Spark-label → dataset → training → ONNX → release-runtime path can execute. It is **not** a capacity result and must not be used to rank the smallest tier or claim a scaling curve. The run used 1,000 records (800 train / 200 heldout by `index % 5` record stride) and seven widths; the corresponding seeds were `20260715, 20260716, 20260717, 20260718, 20260719, 20260720, 20260721`:

| Tier | Parameters | Train match | Heldout match |
|---|---:|---:|---:|
| tier-0 | 311,997 | 0.9225 | 0.725 |
| tier-1 | 676,493 | 0.90625 | 0.695 |
| tier-2 | 1,335,053 | 0.9075 | 0.690 |
| tier-3 | 2,557,197 | 0.9125 | 0.705 |
| tier-4 | 5,127,693 | 0.82125 | 0.685 |
| tier-5 | 9,931,277 | 0.86875 | 0.690 |
| tier-6 | 20,085,773 | 0.8900 | 0.680 |

The tail probe used 8,448 rows × 33 dimensions. For tiers 0–6 respectively, tail heldout match was `0.625, 0.59375, 0.6171875, 0.625, 0.5859375, 0.5859375, 0.59375`; after clamping observations it was `0.640625, 0.59375, 0.625, 0.640625, 0.609375, 0.5859375, 0.6171875`. Clamp-changed fractions were `0.046875, 0.0078125, 0.015625, 0.0234375, 0.0390625, 0.015625, 0.0390625`; raw tail-feature fraction was `0.00429990328848362` for every tier. The +4/−4 tail match pairs were, in order: `(0.6053503751754761, 0.6048768758773804)`, `(0.5707859992980957, 0.5705492496490479)`, `(0.6060606241226196, 0.5984848737716675)`, `(0.6053503751754761, 0.6032196879386902)`, `(0.5901988744735718, 0.578125)`, `(0.5828598737716675, 0.5854640007019043)`, `(0.5802556872367859, 0.5686553120613098)`.

These numbers are diagnostic action-match measurements only. The heldout set has n=200, p≈0.7, and an approximate 95% binomial interval of ±6.4 percentage points; the observed 4.5-point tier spread is therefore within this noise. Width is confounded with seed, the record-stride split can leak autocorrelated states from the same trajectories, normalization was fit over train and heldout, the reported heldout checkpoint was selected on that same heldout set, and there is no independent test set. The labels are also imbalanced: Speak is 564/1,000 records. The defensible conclusion is data-limited/proxy-saturated pipeline validation, not capacity limitation or a smallest-tier ranking.

The release A/B smoke was run with `target/release/mw-sim soak --policy omni-both --ticks 20 --agents 4 --seed 7 --onnx-path training/artifacts/ladder/tier-0/model.onnx --habits off` (8.75 s). UtilitySoul produced deaths=0, hunger=979, energy=990, social=968, hash `0x674bce3b4ca910f9`, idle=80; OmniSoul produced deaths=0, hunger=979, energy=997, social=984, hash `0x94505170865734c8`, sleep=52/speak=28. Deltas (OmniSoul−UtilitySoul) were deaths=0, hunger=+0.0, energy=+7.7, social=+15.1. This establishes release loading/execution and same-initial-state comparison, not policy quality or capacity. Release determinism passed for 8 agents × 40 ticks in 76 s; the corresponding debug run exceeded 3,600 s, so debug performance and any throughput claim remain unresolved.

That historical next sequence was subsequently executed as the corrected v1 protocol below. The pilot remains invalid evidence: it must not be used to rank widths, claim scaling, or claim capacity.
### Corrected v1 evidence and promotion contract

The corrected 1K reference contains 1,000 Spark-grounded records partitioned by whole `(world_seed, trajectory)` groups: 500 train rows over 2 groups, 250 validation rows over 1 group, and 250 untouched test rows over 1 group. Normalization is fit on train only; validation selects the checkpoint; test is evaluated once after selection. The smallest corrected reference width is 296 hidden units (311,997 parameters), trained with independent model seeds `20260715`, `20260716`, and `20260717`. Test action match is `0.884` for all three seeds. Canonical evidence is `training/artifacts/corrected-reference-1k/aggregate.json`, with corrected validation summary hash `d5f3d6a85002e2ef6b6b8a92a93077eba2a71382556f38cdbaa63ff7d57562dc`.

Scaling from 5K through 100K was **SKIPPED** at the 1K promotion stop gate; it was not executed. The full seven-tier capacity ladder is also **SKIPPED**: there is no valid cross-width evidence and therefore no capacity claim. The historical pilot's 4.5 percentage-point spread lies within its approximate ±6.4-point uncertainty and is confounded by width/seed and leakage; it is retained only as explicitly invalid pipeline history.

OMNI expertise is an explicit three-way one-hot contract (`novice`, `capable`, `expert`) spanning Python encoding/training, ONNX inputs, and Rust release inference. Legacy three-input ONNX graphs deliberately use `capable` as their default. Matched expertise data contain 3,000 records from 1,000 matched states (one triplet per state), but only 3/1,000 states are strictly triplet-separable; this degeneracy is a prominent limitation, not a post-result correction.

Despite that label limitation, three independently seeded expertise checkpoints passed the preregistered paired simulator separation across 72 release runs. The expert-minus-novice balanced means are `1.484375`, `1.5`, and `1.4375`; the death-rate confidence intervals are computed in rate units (not raw counts). Preregistration hash: `97f056f6113149ece76a391da2bd1e66d8f83f811082ff5b916119905937bdc5`.

Promotion is simulator behavior, not action match alone: the deterministic 8-tick rollout selects legal capable targets, while Spark is a persona-aware candidate generator. These results support the stated v1 contract only; they do not support a capacity, scaling, or generalization claim.

### 6. TEXT model
One shared instruct model, Q4, never in the tick loop, priority-queued (player-facing > ambient > AFK digests).

- **Pick: Qwen3-0.6B, Q4_0 = 364MB file** ([HF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)); alternatives: Gemma3-270M Q4 ≈ 230MB, Gemma3-1B / Llama-3.2-1B Q4 ≈ 690–770MB. Runtime RSS > file size (KV cache etc.).
- Measured phone decode: TinyLlama 1.1B Q4_0 on llama.cpp/Metal: A14 39 → A17 Pro 57 → A19 87 tok/s ([official bench, 2023 baseline](https://github.com/ggml-org/llama.cpp/discussions/4508)); Qwen2.5-0.5B int8 ~30 tok/s on S24 Ultra CPU ([LiteRT card](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct)). Vendor "100+ tok/s NPU" claims: unverified marketing.
- Persona/act conditioning: system prompt from genome (baseline); **prefix/control-vector steering** as the upgrade path — Dialogue Action Tokens showed a tiny MLP steering a frozen LM via 2 prefix tokens beats GPT-4 on social benchmarks ([arXiv:2406.11978](https://arxiv.org/abs/2406.11978)); llama.cpp ships `--control-vector` natively.

### 7. Memory
Structured, not embedding-soup: event ring buffer (fast) → decaying per-entity relationship/opinion scores (the 90% case for social gameplay) → periodic compression into salient-fact slots. AGA-style ~100-token social summaries instead of transcript retrieval. A-MEM's linked atomic notes ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110)) is the richer upgrade path if needed.

### 8. Persona/Genome + lifetime plasticity
Character sheet: trait vector, stats, goals, backstory. Conditions SOUL (input vector) and TEXT (prompt/steering).

Characters **diverge over their lifetime** via plastic layers on the frozen shared backbone — all updates deterministic, sim-driven (part of the tick, so replay holds):

| Tier | Timescale | Mechanism | Size |
|---|---|---|---|
| 1 Memory/relationships | seconds | event buffer, opinion scores | KBs |
| 2 Habits + experience | in-game days | habit cache + experience embedding + contextual action-bias table (advantage-style updates) | few KB |
| 3 Trait drift | lifetime | bounded persona-vector drift under repeated experience | bytes |
| 4 (optional, heroes) | AFK "sleep" | per-character LoRA delta trained from own trajectory log | tens KB–MB |

Scope honesty: tiers 1–3 give divergent *character* (preferences, habits, relationships, personality) — not novel skills outside the action manifest.

### 9. Inference Runtime layer
One interface, backend per platform/model:

| Runtime | License | Fit |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | **TEXT baseline everywhere.** GGUF, iOS XCFramework, Android/Kotlin binding, `--control-vector`, continuous batching, grammar-constrained output. Lowest engine-embed friction. |
| [ONNX Runtime (+GenAI)](https://github.com/microsoft/onnxruntime-genai) | MIT | SOUL policy on desktop/Android (QNN). iOS GenAI still "under development" — do not depend on it. |
| [ExecuTorch](https://github.com/pytorch/executorch) | BSD-3 | SOUL policy if trained in PyTorch; 50KB base runtime, Swift/Kotlin APIs, CoreML/QNN backends. AOT `.pte` export. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Apache-2.0 | Accelerated TEXT tier (Metal/OpenCL/WebGPU); heavier compile pipeline. |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | Apache-2.0 | Google/NPU Android route; Swift/JS early preview. |
| [WebLLM](https://github.com/mlc-ai/web-llm) | Apache-2.0 | Browser build. |

Owns the **batch scheduler**: SOUL ticks batched per frame on desktop; mobile NPUs want fixed shapes/batch=1 → sequential small calls there. TEXT priority queue. Mobile caveats: NNAPI no dynamic shapes; pin threads for reproducibility; thermals dominate sustained decode.

### 10. Director / LOD
Three rings — the AFK enabler:
- **hot** (on screen): SOUL every tick, TEXT eligible
- **warm** (nearby/plot-relevant): SOUL on its configured cadence (every N ticks)
- **cold** (everyone else): no NN — analytic resolution from persona stats: `state += rate·min(Δt, cap)`, discrete cycles = `floor(Δt/cycle)`, event ledger for catch-up digest
The v0 Director computes the live ring for each entity, and `World::step_gated` enforces it: hot entities run every tick, warm entities run on cadence, and cold entities receive a zero-mask idle observation. Analytic cold fast-forward is recorded separately as an `FfSegment`.

Prior art: Dwarf Fortress worldgen/history-as-event-log, RimWorld world-pawns (persist as data, no ticking), X4 out-of-sector numeric combat, Football Manager Instant Result. AFK/offline = run mostly cold at high speed; promote to hot around notable events; digest on return.

### 11. Scenario Packs
A pack = entity/component schemas + action manifest + intent-validation rules + observation-schema extensions + win/stat definitions (+ optional scenario-tuned SOUL checkpoint). Village sim, AFK arena, football manager = packs on the same kernel.

## v0 milestone (vertical slice)

Deterministic kernel + one village scenario, ~50 agents, utility-AI SOUL stub, one shared TEXT model with latent dialogue, hot/cold LOD, fast-forward. **No trained models in v0** — it proves the intent/observation contracts and the latent-dialogue bet; the tiny net then drops into a working socket.

## Implementation status (v0 and v0.5)

**Verified 2026-07-13.** The vertical slice shipped with a deterministic integer kernel, one village scenario and ~50-agent utility-AI SOUL loop, persona and memory state, shared Qwen3-0.6B Q4_0 TEXT rendering, latent dialogue, live hot/warm/cold LOD, analytic AFK fast-forward with a returning-player digest, replay, and a Ratatui viewer with a headless smoke path. POST-REVIEW v0.5 adds feeding calibration, directional opinions, asynchronous TUI rendering, stale-server reaping, pack-derived fast-forward constants, and a deterministic per-character habit cache. No trained SOUL model is included; the utility policy occupies the production `SoulPolicy` socket.

Measured gates:

| Area | Result |
| --- | --- |
| Determinism/replay | Same seed, 10,000 ticks: identical hash. Replay from `(seed, intent log)`, including `FfSegment` entries, reproduces the full state hash including pack state. Habit-enabled runs are deterministic; habit policy state is intentionally outside kernel hash state. |
| Live village | 50 agents at 12,893 ticks/s in release on an M4 Pro; maximum action-histogram share 37.9% in the v0 measurement. |
| POST-REVIEW health | 0 starvation deaths across 50 seeds × 10,000 ticks (earlier v0 emergent result: 8/50); eat share 2.4–2.9%. |
| Habit cache | Honest hit rate 51.7% in the 50×10k soak and 50.7% in the 86,400-tick demo; 2,151 ticks/s habits-on versus 1,088 ticks/s off at 50×10k; deaths 0, deterministic hashes, and per-character divergence gate green. Speak/Give passthrough, urgency invalidation, and bounded TTLs preserve social behavior. |
| Analytic FF | 604,800 ticks in 0.014 s (~43M ticks/s); drift versus the hot reference ≤4% under the 15% bound; digest deterministic. Analytic gains are read from village pack constants, with no duplicate calibration constants. |
| Opinions, viewer, and server lifecycle | Opinion deltas are asymmetric and directional; live TEXT rendering is asynchronous; stale `llama-server` processes are reaped on startup. PID-reuse handling is narrowed but its TOCTOU is not atomic. |
| TEXT | Qwen3-0.6B Q4_0, 364.5 MiB file via llama.cpp; warm render 79 ms; KV-slot reuse reduces prompt tokens 104 → 1. Fresh M4 Pro bake-off (llama.cpp b9960-a935fbffe, llama-bench, 3 reps, pp512/tg128): Metal 3,136.8 / 180.2 t/s; CPU-only 373.7 / 70.2 t/s. |
| Latent dialogue | Unobserved conversations make 0 `TextBackend` calls while relationship deltas apply; retroactive backfill is act-coherent and cached; text is one-way and never mutates sim state. |
| Gates/viewer | 58 tests green after the v0.5 review; `clippy -D warnings`, formatting, and `scripts/demo.sh` are clean; Ratatui TUI was verified in a real PTY and `view --smoke` exits 0 headless. |

### TEXT bake-off (M4 Pro, 2026-07-15)

The repeatable command is `REPS=3 scripts/bench_text.sh`; it downloads missing GGUFs into `~/.cache/mini-world/models`, runs `llama-bench` with Metal and `-ngl 0` CPU paths, and records `/usr/bin/time -l` maximum RSS. Prompt/decode columns are pp512/tg128 averages over three repetitions; RSS is the real process maximum, not the quantized file size.

| Candidate (GGUF) | Metal prompt / decode tok/s | CPU prompt / decode tok/s | File MiB | Max RSS Metal / CPU MiB | llama.cpp load |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B Q4_0 (incumbent) | 3,136.8 / 180.2 | 373.7 / 70.2 | 364.5 | 514.2 / 901.2 | loaded |
| Qwen3.5-0.8B Q4_0 (hybrid Gated-DeltaNet + MoE) | 2,754.8 / 115.1 | 284.8 / 39.1 | 483.7 | 637.4 / 1,156.2 | loaded |
| Gemma3-270M it Q4_K_M (floor) | 15,186.5 / 159.8 | 726.9 / 90.8 | 241.4 | 336.6 / 551.6 | loaded |

All three candidates load and run on this llama.cpp build. Qwen3.5's hybrid architecture is therefore not a load blocker here, but runtime maturity is explicitly provisional: its decode is 36% slower than Qwen3 on Metal and 44% slower on CPU, while file size is 33% larger and CPU RSS is 28% higher. Its chat template also needed `chat_template_kwargs.enable_thinking=false`; otherwise the identical smoke request spent its 48-token budget in visible reasoning rather than dialogue.

Identical OpenAI-message dialogue smoke (Metal, seed 1, temperature 0.7, `/no_think`, `enable_thinking=false`) produced:

| Candidate | Output |
| --- | --- |
| Qwen3-0.6B Q4_0 | “It’s late for you, but I’m here to help. Let me know if there’s anything else I can assist with.” |
| Qwen3.5-0.8B Q4_0 | “Give it three more seconds.” |
| Gemma3-270M it Q4_K_M | “Thank you for your concern. I'll get it to you as soon as possible.” |

**Recommendation:** keep **Qwen3-0.6B Q4_0 as the default TEXT tier**: it is materially faster than Qwen3.5 at decode, smaller, and lower-RSS while producing a usable smoke line. Use **Gemma3-270M Q4_K_M as the floor tier** for constrained devices: its 241.4 MiB file and 336.6 MiB Metal RSS are the lowest by a wide margin, and its 15,186.5 tok/s prompt path is useful for short latent-dialogue prompts, accepting weaker rendering quality and slightly slower decode than Qwen3 on this run. Do not promote Qwen3.5 to default until hybrid-architecture runtime behavior is re-benchmarked on the target llama.cpp release.

The earlier 82.7% habit hit rate was pre-fix telemetry: cache accounting counted behavior that could suppress social scoring, and the first implementation allowed social lockout. The review made the telemetry truthful and always re-scored Speak/Give; 51.7% (soak) and 50.7% (demo) are the honest measurements.

### Ratified v0 contract changes

- **Canonical apply order replaces submission-order assumptions.** The kernel sorts each intent batch by entity id before validating and applying it. Per-entity RNG remains stateless and keyed by seed, entity, stream, and tick, which makes draws iteration-order independent; effect order is the canonical sort, including shared-cell conflicts.
- **Pack state is hash state.** `ScenarioPack::hash_state` folds needs, inventories, ground items, and other pack-owned state into `World::state_hash`.
- **Fast-forward is logged.** Cold analytic spans are `FfSegment` entries in the intent log; replay consumes those entries and applies the pack's analytic advance, so the log covers AFK time.
- **Live LOD is a kernel gate.** The Director's hot/warm/cold decisions are enforced through `World::step_gated`, not by mutating world state outside the normal tick pipeline.
- **Observation has two ratified layers.** `AgentObs` in `mw-agents` is the rich, SOUL-facing schema. Kernel `Observation` is the minimal seam and the single scan used to feed `ScenarioPack::afforded_tools`.

### Ratified v0.5 contract changes (2026-07-13)

- **Habit cache semantics are predicate-gated and per-character.** A cache key combines quantized needs, location, tool mask, and goal; validity predicates are checked every tick, with urgency/event invalidation and bounded TTLs. Cache state is deterministic and supports character-level divergence.
- **Social acts always pass through the scorer.** Speak and Give are never allowed to become cached social lockout; their mechanical outcomes remain applied by the kernel/event path.
- **Habits are policy plasticity, not kernel truth.** Habit-cache state is intentionally outside `world.state_hash`. Replay and canonical hashes cover world/pack state and validated intents; cache contents can evolve as policy state without changing the kernel truth contract.
- **Single source of truth for fast-forward constants.** Analytic gains are derived from the installed village pack constants; the drift gate remains ≤15% (measured ≤4%).
- **Post-review health and lifecycle fixes are ratified.** Directional opinion deltas, asynchronous TUI rendering, stale-server reaping, and the narrowed (non-atomic) PID-reuse TOCTOU behavior are part of the v0.5 implementation status.

## Open questions

1. **Stack**: Rust sim core + Godot front (portable, WASM, FFI to llama.cpp) vs TypeScript/web-first (velocity, WebLLM). Leaning Rust core.
2. **First scenario**: village social sim (de-risks SOUL breadth + latent dialogue, the novel bet) vs AFK battler (simpler manifest, stresses LOD). Leaning village.
3. **Multiplayer**: in scope? Escalates determinism from "nice" to mandatory lockstep.

## Research provenance

Compiled 2026-07-12 from three parallel research passes (small-model landscape; agent-brain prior art; scale/LOD/determinism). Key sources beyond those inline: Generative Agents ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)), Lyfe Agents cost analysis ([arXiv:2310.02172](https://arxiv.org/abs/2310.02172)), Project Sid/PIANO ([arXiv:2411.00114](https://arxiv.org/abs/2411.00114)), LLM-OBTEA hybrid planning ([IJCAI 2024](https://www.ijcai.org/proceedings/2024/755)), MapCoder-Lite multi-role distillation ([arXiv:2509.17489](https://arxiv.org/abs/2509.17489)), lm-Meter phone profiling ([arXiv:2510.06126](https://arxiv.org/html/2510.06126)), S-LoRA multi-adapter serving ([arXiv:2311.03285](https://arxiv.org/abs/2311.03285)).
