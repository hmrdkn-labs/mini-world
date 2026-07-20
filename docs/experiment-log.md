# mini-world experiment log

Methods backbone for the v0/v0.5 vertical slice. This is a chronological export of the orchestration record in `local://mw-provenance.json`, checked against the repository history at HEAD `490e656`. It records gates and measured outcomes as reported; it does not add estimates. Dates below use the timestamps in the provenance record unless a git timestamp is explicitly labelled.

## Record index

- **Actions:** `mw-arch-v1`, `mw-v0`, and `mw-v05`.
- **Steps:** 20 total: 4 research/design steps, 10 v0 implementation/review steps, and 6 v0.5 steps.
- **Final implementation HEAD:** `490e656` (`fix(agents,sim): truthful habit telemetry, social passthrough, urgency invalidation, pack-derived FF constants (v0.5 review pass)`).
- **Final v0.5 gate reported in provenance:** 58 tests green, `clippy -D warnings`, formatting, and `scripts/demo.sh` green.

The research action used sonnet-class scout agents for the research passes. The v0 implementation action was explicitly assigned to Opus workers at low/medium effort (no Luna on that action). The v0.5 action used `gpt-5.6-luna` at high effort, including its review step.

## 2026-07-12 — `mw-arch-v1`: architecture research and synthesis

**Goal.** Design the mini-world architecture: a deterministic semi-AFK village platform in which small per-character SOUL policies emit intents and a shared TEXT model renders dialogue. The criteria required coverage of the world kernel, observation/action contract, SOUL and TEXT approaches, memory/persona, inference runtime, scenario layer, and LOD/AFK director, with empirical claims grounded in fresh research.

**Action record.** Created `2026-07-12 16:14:42`; updated `2026-07-12 16:31:31`. Deliverable commit: `8f30071` (`docs: architecture design v1 — SOUL/TEXT split, intent kernel, habit cache, LOD director`; git timestamp `2026-07-12T23:59:52+07:00`).

### Steps and gates

1. **Research on-device small-LM landscape (sonnet-class scout).** Gate: name concrete sub-2B models, quantized memory footprints, measured phone CPU/GPU tok/s, and runtime options with sources. Outcome: runtime matrix covering llama.cpp, MLC, ORT-GenAI, ExecuTorch, and LiteRT-LM; phone decode benchmarks; GGUF sizes; 1–5M policy evidence from Decision Transformer and TD-MPC2.
2. **Research agent-brain prior art (sonnet-class scout).** Gate: cover generative-agent architectures, shipped LLM-NPC games, utility-AI/BT hybrids, and LLM-to-small-policy distillation, including failures and sources. Outcome: AGA decision caching (31–43% of baseline token use), a 2,000-trajectory/LoRA distillation recipe, DAT-style tiny-policy steering, PIANO coherence concerns, A-MEM structured memory, and the finding that latent dialogue had no shipped prior art in the review; AGA was the closest comparison.
3. **Research many-agent inference and LOD (sonnet-class scout).** Gate: cover batched inference, shared weights plus per-agent conditioning, and simulation LOD/offline fast-forward patterns with sources. Outcome: batching and LoRA/prefix/control-vector patterns; DF/RimWorld/Football Manager LOD references; a determinism contract based on logging validated intents rather than logits; a scout throughput estimate for 1–10M policies (marked as an estimate in the design, not a project measurement).
4. **Synthesize architecture (research synthesis).** Gate: reconcile the conductor draft and scout findings, and name open questions and the v0 milestone. Outcome: v1 architecture with a habit cache, SOUL 1–5M policy socket, Qwen3-0.6B Q4 via llama.cpp TEXT, latent dialogue, and intent-log determinism. Open questions were stack, first scenario, and multiplayer scope.

## 2026-07-12/13 — `mw-v0`: deterministic vertical slice

**Goal.** Implement the Rust core plus village scenario described by DESIGN.md at commit target `8f30071`. The action criteria required all 9 implementation/review gates to pass locally: TEXT measurements, kernel determinism/replay, village schema and affordances, observation encoding, a 50-agent soak, memory decay, LOD/fast-forward drift, latent-dialogue rendering/backfill, and a usable viewer. The operator instruction assigned v0 waves to Opus low/medium workers.

**Action record.** Created `2026-07-12 23:30:25`; updated `2026-07-13 02:50:51`. The implementation history is the following sequence: `5b2b72a`, `ce43e04`, `99f8ac0`, `590c8b7`, `c9519be`, `65bc955`, and the v0 documentation anchor `f47fab9`.

### Steps and gates

0. **Runtime bench (Opus medium assignment in the v0 wave).** Gate: measure Qwen3-0.6B Q4_0 decode/prompt throughput and memory on this Mac using Metal and CPU. Outcome: M4 Pro, llama.cpp b9960, 359 MiB model; Metal pp512 **2691±705 t/s**, tg128 **193±44 t/s**; CPU-only pp512 **388±8 t/s**, tg128 **76±9 t/s**. A dialogue smoke produced 165 t/s generation and 2.2 s wall time including load; the provenance notes that a first TTY wait timed out. A roughly 30-token line was reported as ~0.16 s GPU/~0.4 s CPU, with prompt processing dominant and KV reuse important. This was a measurement, not an MLX result; llama.cpp Metal was the shipping runtime and CPU was the portability floor. Gate passed. Commit context: `8f30071` design baseline; measurements precede the implementation commits below.
1. **Cargo workspace and `mw-core` kernel (Opus, medium).** Gate: package tests for same-seed 10,000-tick determinism, replay from intent log, seed divergence, order independence, and a headless 10,000-tick run. Outcome: stateless tick-keyed per-entity RNG, FNV-1a canonical hash, replay reproducing the hash, and all named tests green; `clippy -D warnings` and formatting clean. Commit `5b2b72a`.
2. **`mw-village` scenario pack (Opus, medium).** Gate: manifest and affordance tests plus a 1,000-tick scripted run exercising valid and invalid intents with typed rejection reasons. Outcome: 12-tool manifest, needs decay, 16×16 map, affordance masking, rejection reasons, deterministic/replayable run, 9 tests plus the scripted functional run. The hardcoded kernel tool-mask seam was recorded and carried into the later specification. Commit `ce43e04`.
3. **Memory and relationships (Opus, low).** Gate: expected interaction delta, closed-form decay, stable salient facts under replay, and ring-buffer eviction at capacity. Outcome: ring buffer cap 64, integer opinion decay within 1% of the closed form, salient top-k facts, and 5/5 tests plus a trajectory functional test. Symmetric deltas were explicitly a v0 simplification. Commit `ce43e04`.
4. **`mw-text` llama.cpp bridge (Opus, medium).** Gate: ignored live test renders an in-character line in under 2 s warm, reuses the conversation slot, enforces `/no_think`, and leaves no orphan on shutdown. Outcome: managed `LlamaServerBackend`, 79 ms warm render, KV reuse reducing prompt work 104→1, clean Drop lifecycle, and 3/3 live gates. A numeric-persona prompt issue was recorded and fixed by the later persona registry. Commit `ce43e04`.
5. **Observation encoder, utility SOUL, and 50-agent soak (Opus, medium).** Gate: golden observation snapshot, population-independent observation size, 50×10k soak without panic/deadlock, non-degenerate histogram, deterministic hash, and throughput report. Outcome: fixed-size `AgentObs`, `UtilitySoul`, fixed tool-mask seam, 50×10k at 9,600 ticks/s, deterministic, maximum histogram share 32.5%, rejection rate 0.26%, deaths 0. The record flagged observation-schema drift and need-floor tuning for later review. Commit `99f8ac0`.
6. **Director/LOD and analytic fast-forward (Opus, medium).** Gate: one in-game week under 10 s at 50 agents, cold-vs-hot mean need drift within ±15%, deterministic digest. Outcome: 604,800 ticks in 0.02–0.2 s, drift ≤4% against the hot reference, deterministic digest. The record flagged that live per-tick SOUL skipping still needed a seam fix. Commit `590c8b7`.
7. **Latent-dialogue integration (Opus, medium).** Gate: unobserved conversation has zero TEXT calls while deltas apply; observed output is act-consistent; backfill is coherent and cached. Outcome: persona registry, always-applied relationship deltas, observed-only render, backfill/cache, live counter 1→2→2, and proof that TEXT cannot change the simulation hash. A 0.6B backfill for `taunt` was semantically soft and recorded as a prompt/steering quality note. Commit `590c8b7`.
8. **Ratatui debug viewer (Opus, low).** Gate: viewer map/inspector/event/dialogue panes, speed and fast-forward controls, and headless `--smoke` exit 0; manual live-TTY check. Outcome: viewer shipped with backfill controls and smoke mode; worker smoke/unit checks passed; the conductor verified real-PTY input initialization and clean `q` exit. Commit `c9519be`.
9. **Adversarial review and hardening (Opus reviewer, medium).** Gate: address or explicitly waive determinism/contract findings, run workspace tests and `clippy -D warnings`, and complete the end-to-end demo (live day, one-week FF, digest, backfill). Outcome: 9 findings and 6 adjudicated fixes; `scripts/demo.sh` exited 0; the v0 gate reported 49 tests plus 4 live-model tests green, clean clippy/formatting, live throughput about 12.9k ticks/s, week FF 0.014 s, digest, and backfill. Commit `65bc955`; v0 status docs followed in `f47fab9`.

### What the first green suite hid

The adversarial review found that the pre-review green suite did not prove the starvation path was reachable under the intended death semantics. Once the death path was made reachable and tested, the 50-agent run exposed a real emergent starvation result (reported as 7–8/50 in the v0 review record, summarized elsewhere as 8/50), rather than a harmless unreachable branch. Feeding calibration was therefore carried into v0.5 rather than waived.

## 2026-07-13 — `mw-v05`: sim health and habit cache

**Goal.** Complete the v0 backlog fixes (feeding calibration, asymmetric opinions, asynchronous viewer rendering, stale `llama-server` cleanup), implement DESIGN block 4's per-character predicate-gated habit cache, measure its cost/speed effect, and refresh the research provenance. All v0.5 worker and reviewer steps were assigned to `gpt-5.6-luna` at high effort.

**Action record.** Created and updated `2026-07-13 07:00:30`; latest implementation commit is `490e656`. Wave commits: `8865893` (health, opinions, async render, stale-server cleanup), `35b1921` (initial habit cache), and `490e656` (review hardening and truthful telemetry).

### Steps and gates

1. **Sim health and asymmetric opinions (gpt-5.6-luna, high).** Gate: 50×10k soak with zero starvation deaths, sane histogram, deterministic hash; asymmetric delta tests; workspace tests. Outcome: convex hunger pressure (`HUNGER_CURVE=8`) and `EAT_GAIN=400`; 0/0 starvation deaths across 50 seeds × 10,000 ticks (previously 7–8 at seed 1); eat share 1.9→2.4–2.9%; deterministic; directional `verb_affect` and asymmetric Give/Attack/Speak tests. The step also identified duplicate analytic `GAIN` constants for later unification. Commit `8865893`.
2. **Non-blocking dialogue render (gpt-5.6-luna, high).** Gate: a deliberately slow 500 ms mock backend must allow the UI frame counter/draw loop to advance during rendering; smoke and PTY paths remain green. Outcome: render-worker channels, placeholder and swap-in, and a SlowMock gate proving frames advance mid-render; smoke reported 44 rows and exited 0. Commit `8865893`.
3. **Stale-server cleanup (gpt-5.6-luna, high).** Gate: stale pidfile plus live orphan is detected, verified, killed, and replaced; missing/dead/reused PID cases and existing live gates remain green; no orphan remains. Outcome: `MW_LLAMA_PIDFILE`, `ps` command verification, zombie-aware reaping, missing/dead/reused PID tests, live orphan-reap gate, and clean `pgrep` result. The final review narrowed PID-reuse risk but accepted that the TOCTOU is not atomic. Commit `8865893`.
4. **Habit cache, DESIGN block 4 (gpt-5.6-luna, high).** Gate: cache-enabled soak preserves zero deaths and sane histogram, same seed gives same hash, throughput is compared with cache off, and two agents with different histories diverge in cache contents. Outcome before review: predicate-gated `HabitSoul`, quantized keys, event/urgency invalidation, bounded deterministic eviction, 82.7% reported hit rate, 1,088→2,151 ticks/s, deaths 0, deterministic hash, and divergence test green. The histogram was flagged for adjudication: 38.7% max share off versus 7.1% on. Commit `35b1921`.
5. **Review, fix, and full verify (gpt-5.6-luna reviewer, high).** Gate: adjudicate review findings, make docs and measurements agree, run workspace tests, clippy/formatting, and demo. Outcome: histogram accounting was a falsehood and the first cache behavior also caused real social lockout; both were fixed with truthful attribution, Speak/Give passthrough, an Idle guard, urgency-edge invalidation, Move TTL=8, and pack-derived FF gains. Honest hit rate became 51.7% in the soak; the 86,400-tick demo reports 50.7%; sociality remained within the reviewed ~2× no-cache bound; maximum share was 22.7%; deaths stayed 0; 4.32M demo decisions were reported at 50.7% hits; all gates were green (58 tests, clippy, formatting, demo). Commit `490e656`.
6. **Research-provenance export (gpt-5.6-luna, high).** Gate: export all action steps, gates, model assignments, timestamps, measured outcomes, and git anchors to this log; provide a defensible paper outline with evidence mapping; add no fabricated results. Outcome: this document and `docs/paper-notes.md` are the requested documentation deliverables. No new implementation commit is claimed: the implementation anchor remains `490e656`.

### What the second green suite hid

The initial habit-cache gate was green while reporting 82.7% hits. Review showed that this was not a clean compute-saving measure: cache accounting and action attribution could hide scorer work, while cached routine behavior could suppress social decisions. The histogram therefore combined an accounting falsehood with a real social lockout. The final implementation makes Speak and Give passthrough, preserves urgency invalidation and bounded TTLs, and reports the lower honest rates (51.7% soak; 50.7% demo). This is why the pre-review 82.7% number is retained only as a documented pre-fix result, not as the v0.5 claim.

## Reproducibility notes

The deterministic contract is the validated-intent log plus seed and model/backend identifiers; neural outputs are advisory. Kernel and pack state are hash state, while habit-cache state is deliberately policy state outside `world.state_hash`. Throughput and wall-clock measurements are report-only and do not feed simulation state. Re-running the named gates should preserve hashes and behavioral assertions, but hardware-dependent throughput and live-model latency should be treated as measurements for the recorded machine, not universal constants.

## 2026-07-15 — 1K Spark-labeled OMNI ladder (pipeline-validation evidence only)

**Purpose and status.** This run validated the data/label/training/export/release path; it did not establish a capacity curve, a smallest-tier ranking, or a promotion result. The dataset contained 1,000 records, split 800 train / 200 heldout by `index % 5` record stride. All seven tiers used the same records and shared normalization fit over the full 1,000-record set; width was changed together with the seed (`20260715, 20260716, 20260717, 20260718, 20260719, 20260720, 20260721`). There was no independent test set.

### Action-match measurements

| Tier | Parameters | Train match | Heldout match |
|---|---:|---:|---:|
| tier-0 | 311,997 | 0.9225 | 0.725 |
| tier-1 | 676,493 | 0.90625 | 0.695 |
| tier-2 | 1,335,053 | 0.9075 | 0.690 |
| tier-3 | 2,557,197 | 0.9125 | 0.705 |
| tier-4 | 5,127,693 | 0.82125 | 0.685 |
| tier-5 | 9,931,277 | 0.86875 | 0.690 |
| tier-6 | 20,085,773 | 0.8900 | 0.680 |

The tail probe covered 8,448 rows × 33 dimensions. In tier order 0–6, tail heldout match was `0.625, 0.59375, 0.6171875, 0.625, 0.5859375, 0.5859375, 0.59375`; tail-clamp match was `0.640625, 0.59375, 0.625, 0.640625, 0.609375, 0.5859375, 0.6171875`; and clamp-changed fraction was `0.046875, 0.0078125, 0.015625, 0.0234375, 0.0390625, 0.015625, 0.0390625`. The raw tail-feature fraction was `0.00429990328848362` for every tier. The +4/−4 tail match pairs were `(0.6053503751754761, 0.6048768758773804)`, `(0.5707859992980957, 0.5705492496490479)`, `(0.6060606241226196, 0.5984848737716675)`, `(0.6053503751754761, 0.6032196879386902)`, `(0.5901988744735718, 0.578125)`, `(0.5828598737716675, 0.5854640007019043)`, and `(0.5802556872367859, 0.5686553120613098)`.
The corresponding final train/heldout losses were, in tier order: `(0.21769002199172974, 1.5785443258285523)`, `(0.30663896024227144, 1.4133897161483764)`, `(0.2811539214849472, 1.6312757015228272)`, `(0.2250473192334175, 1.8670896434783935)`, `(0.5746346044540406, 1.1518525385856628)`, `(0.4109943062067032, 1.2729276251792907)`, `(0.29342915177345275, 1.555791847705841)`. The +4/−4 prediction-change fractions were `(0.1276041716337204, 0.11055871099233627)`, `(0.09232954680919647, 0.09398674219846725)`, `(0.09280303120613098, 0.08262310922145844)`, `(0.08001893758773804, 0.07954545319080353)`, `(0.059659089893102646, 0.0703125)`, `(0.09209280461072922, 0.09801136702299118)`, and `(0.07528409361839294, 0.07007575780153275)`.

### Why this is not a capacity result

Only 200 records were held out, with p≈0.7 and an approximate 95% binomial interval of ±6.4 percentage points. The observed 4.5-point heldout spread is flat within that uncertainty. Width is confounded with seed, and the record-stride split leaks correlated states from the same trajectories. Normalization leakage comes from fitting over train plus heldout; the selected checkpoint is also chosen by that heldout score, so the score is optimistically biased. There is no independent test set. Speak is 564/1,000 labels, so class imbalance further limits action-match interpretation. The result is data-limited/proxy-saturated pipeline validation, not evidence that the smallest tier wins or that capacity is the bottleneck.

### Release A/B and performance caveat

The real release A/B smoke command was `target/release/mw-sim soak --policy omni-both --ticks 20 --agents 4 --seed 7 --onnx-path training/artifacts/ladder/tier-0/model.onnx --habits off` (8.75 s). UtilitySoul: deaths=0, hunger=979, energy=990, social=968, hash `0x674bce3b4ca910f9`, idle=80. OmniSoul: deaths=0, hunger=979, energy=997, social=984, hash `0x94505170865734c8`, sleep=52/speak=28. Deltas (OmniSoul−UtilitySoul): deaths=0, hunger=+0.0, energy=+7.7, social=+15.1. Release determinism passed at 8 agents × 40 ticks in 76 s. The corresponding debug run exceeded 3,600 s; debug performance is unresolved and these runs support no throughput claim. This A/B is a release wiring/execution check and simulator diagnostic, not a promotion gate.

### Corrected v1 evidence, expertise validation, and explicit non-claims

The corrected 1K reference replaced the pilot methodology before promotion: 1,000 Spark-grounded records are partitioned by whole `(world_seed, trajectory)` groups into 500 train rows over 2 groups, 250 validation rows over 1 group, and 250 untouched test rows over 1 group. Normalization is train-only; validation selects the checkpoint; test is evaluated once after selection. The 296-hidden-unit (311,997-parameter) reference was trained with seeds `20260715`, `20260716`, and `20260717`; test action match was `0.884` for every seed. See `training/artifacts/corrected-reference-1k/aggregate.json`; corrected validation summary SHA-256 is `d5f3d6a85002e2ef6b6b8a92a93077eba2a71382556f38cdbaa63ff7d57562dc`.

The 1K promotion stop gate **SKIPPED** the planned 5K→100K scaling run; those tiers were not executed. The seven-tier capacity ladder is likewise **SKIPPED** because no valid cross-width evidence exists. The historical pilot above remains explicitly invalid evidence: its 4.5-point spread was within approximately ±6.4 points of uncertainty and was confounded by width/seed, record-stride correlation, full-data normalization, and heldout checkpoint reuse. It supports no capacity, scaling, or smallest-tier claim.

Expertise is a deliberate `novice`/`capable`/`expert` one-hot input across Python, ONNX, and Rust release inference; legacy 3-input graphs default to `capable`. Matched expertise has 3,000 records from 1,000 matched states, yet only 3/1,000 states are strictly triplet-separable. This degeneracy remains a prominent limitation. Nevertheless, three independently seeded checkpoints passed preregistered paired simulator separation over 72 release runs. Expert-minus-novice balanced means were `1.484375`, `1.5`, and `1.4375`. The review corrected death-rate CI calculations to use rates rather than raw death counts. Preregistration SHA-256 is `97f056f6113149ece76a391da2bd1e66d8f83f811082ff5b916119905937bdc5`.

Spark is a persona-aware candidate generator; deterministic 8-tick rollout selects legal capable targets. Promotion is based on simulator behavior, not action match alone. These results are bounded v1 evidence and do not establish capacity, scaling, or broad generalization.
