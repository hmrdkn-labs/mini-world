# Paper notes: mini-world resource paper

## Positioning and venue class

Target a computational-social-science / agent-based-modeling **resource** paper, not a benchmarks or ML-methods paper. Plausible venue classes are JASSS, CSCW (system/resource or methods-facing work), the AAMAS demo/resource track, and ICWSM datasets or infrastructure tracks. The paper should present a reproducible simulation platform and its measurement protocol; it should not claim that the utility policy is a learned-model advance.

Working title: **mini-world: deterministic replay, latent dialogue, and habit-cache plasticity for on-device agent simulation**.

## Abstract claims to support

1. mini-world is an open, deterministic, replayable agent-simulation platform. NN policies are advisory: they emit intents, while the kernel validates and applies them in canonical order. The validated intent log and pack state are ground truth.
2. Dialogue is latent: mechanical relationship outcomes are always applied, while TEXT rendering is attention-gated and can be backfilled. The closest reviewed prior art is Affordable Generative Agents (AGA), arXiv:2402.02053; no shipped prior art for this exact render-only-when-observed NPC dialogue design was found in the project review. This is a bounded literature-review finding, not a claim that no related work exists.
3. The habit cache is a deterministic, per-character plasticity layer. Predicate checks, urgency invalidation, bounded TTLs, and social passthrough preserve behavior while reducing repeated SOUL scoring. The paper must use the honest telemetry: 51.7% hit rate in the 50-seed × 10,000-tick soak and 50.7% in the 86,400-tick demo; 2,151 versus 1,088 ticks/s in the 50×10k habits-on/off comparison.
4. The implementation records measured on-device budgets for Qwen3-0.6B Q4_0 through llama.cpp (M4 Pro Metal/CPU), rather than presenting hardware-independent performance claims.

## Proposed outline

### 1. Introduction and scope

- Define the problem: many agents, deterministic world truth, offline/AFK progression, and limited on-device TEXT compute.
- State that this is a resource/platform contribution for agent-based modeling and social-simulation research.
- Separate the claims supported by the current vertical slice and corrected OMNI-v1 evidence from future general SOUL work.
- State the paper's negative scope: no benchmark leaderboard, no claim of human-level social behavior, no broad capacity/scaling claim, and no claim that generated words determine outcomes.

### 2. Related work and design rationale

- Generative Agents and AGA: event/memory/social architecture and decision caching; cite AGA arXiv:2402.02053 as the closest compute/cache comparison.
- PIANO/Project Sid: say/do divergence motivates making TEXT verbalize a committed act rather than decide one.
- Dwarf Fortress world history, RimWorld world-pawns, X4 out-of-sector simulation, and Football Manager instant result as LOD/offline progression precedents.
- Small-model/on-device runtime evidence from the architecture research record; distinguish external measurements from mini-world's own M4 Pro measurements.
- Explain the literature-review boundary around latent dialogue: the project review found no shipped system with the exact attention-gated render-only semantics; frame this as a gap motivating an experiment, not a universal novelty proof.

### 3. System and contracts

- Kernel: fixed timestep, integer/fixed-point state, seeded per-entity RNG, canonical entity-id application order, validation and event/intent logs, and pack-owned state in the canonical hash.
- Policy seam: `SoulPolicy` emits intents; `ScenarioPack` defines typed tools, affordances, validation, and analytic state; policy outputs are advisory.
- TEXT seam: shared Qwen3-0.6B Q4_0 backend, committed `speak` act and persona conditioning, observed render and lazy backfill, one-way TEXT→no-world-state rule.
- Character state: structured events, decaying directional opinions, persona vector, and per-character habit cache.
- LOD: hot/warm/cold rings and replayable `FfSegment` analytic fast-forward.
- Viewer and asynchronous render worker as a usability and inspection surface, not as a simulation authority.

### 4. Reproducibility protocol

- Publish seed, scenario-pack version/constants, model/backend identifiers, validated intent log, event log, and canonical state hash.
- Define deterministic gates: same-seed 10,000-tick hash, replay hash including `FfSegment`, seed divergence, fixed-seed soak hash, deterministic fast-forward digest, and deterministic latent-dialogue pipeline.
- Keep report-only wall-clock values (throughput and latency) separate from state-changing inputs.
- Explicitly document that habit-cache state is deterministic policy state but intentionally outside `world.state_hash`: it is plasticity, not kernel truth.
- Include the full chronology and gates in [`experiment-log.md`](experiment-log.md), with implementation commits from `8f30071` through `490e656`.

### 5. Experiments and results

#### 5.1 Kernel and replay

- Evidence: `mw-core/tests/determinism.rs` (`determinism_same_seed` and replay/seed-divergence tests), `crates/mw-core/src/world.rs` (`submission_order_does_not_affect_state_kernel_sorts`), commit `5b2b72a`, and experiment-log sections `mw-v0` steps 1 and 9.
- Report the observed 10,000-tick same-seed equality and full-state replay, including pack state and `FfSegment` records.

#### 5.2 Village health and social behavior

- Evidence: `crates/mw-sim/tests/soak.rs` (`soak_is_deterministic_for_a_fixed_seed`, `soak_has_no_paralysis_or_starvation_deadlock`, `soak_histogram_is_non_degenerate`), `mw-agents/src/memory.rs` (`interaction_event_is_directional`, `speak_affect_is_modest_and_act_dependent`), commit `8865893`, and experiment-log `mw-v05` steps 1 and 5.
- Report 0 starvation deaths across 50 seeds × 10,000 ticks and 2.4–2.9% eat share. Identify the earlier 8/50 emergent v0 result as superseded by the calibrated run, not erase it from the methods history.

#### 5.3 Latent dialogue

- Evidence: `crates/mw-sim/tests/dialogue_det.rs` (`pipeline_is_deterministic`, `backfill_renders_then_caches`), `crates/mw-sim/tests/dialogue_live.rs` (observed/backfill render counter), commits `590c8b7` and `490e656`, and experiment-log `mw-v0` step 7.
- Report zero TEXT calls for unobserved rows, always-applied mechanical deltas, act-coherent observed/backfilled lines, cache reuse on re-inspection, and unchanged simulation hash after rendering.
- Gap: the current evidence is a functional gate, not a human evaluation of dialogue quality or social believability.

#### 5.4 Habit-cache plasticity and telemetry honesty

- Evidence: `crates/mw-agents/src/habits.rs` (`different_histories_diverge_cache_contents`), `crates/mw-sim/src/view.rs` (`slow_render_does_not_block_frames_and_caches_response` for the adjacent viewer gate), commits `35b1921` and `490e656`, and experiment-log `mw-v05` steps 4–5.
- Report the final honest 51.7% soak and 50.7% demo hit rates, 2,151 versus 1,088 ticks/s, zero deaths, deterministic hashes, social passthrough, urgency invalidation, and bounded TTLs.
- Explain the adversarial-review correction: 82.7% was pre-fix telemetry and partly reflected social lockout; the histogram was both an accounting falsehood and evidence of a real cached-social behavior regression. Do not present 82.7% as the result.
- Gap: the current A/B is one scenario and one recorded hardware setup; it does not establish generalization across populations, packs, hardware, or long-run social distributions.

#### 5.5 LOD and measured device budget

- Evidence: `crates/mw-sim/tests/fast_forward.rs` (`fast_forward_is_deterministic`), commit `590c8b7` plus pack-constant unification in `490e656`, and experiment-log `mw-v0` step 6 / `mw-v05` step 5.
- Report 604,800 ticks in 0.014 s, drift ≤4% against the hot reference under a 15% bound, deterministic digest, and analytic gains sourced from village pack constants.
- Evidence for TEXT budget: provenance `mw-v0` step 0 and commit context `ce43e04`; report 359 MiB Qwen3-0.6B Q4_0, 79 ms warm render, KV prompt work 104→1, and the measured M4 Pro Metal/CPU pp/tg values.
- Gap: no phone-hardware measurement is in the current record. The architecture document's phone numbers are external research context, not mini-world measurements.

#### 5.6 1K Spark-labeled OMNI ladder: pipeline validation only

The seven-tier run used 1,000 records (800 train / 200 heldout) and widths from 311,997 to 20,085,773 parameters. Train match rates were `0.9225, 0.90625, 0.9075, 0.9125, 0.82125, 0.86875, 0.8900`; heldout match rates were `0.725, 0.695, 0.690, 0.705, 0.685, 0.690, 0.680` in tier order. The complete tail measurements and provenance are recorded in [`experiment-log.md`](experiment-log.md#2026-07-15--1k-spark-labeled-omni-ladder-pipeline-validation-evidence-only). These are action-match diagnostics for pipeline validation, not a capacity curve or a ranking of the smallest tier.

Interpretation must state every limitation: n=200 heldout records, p≈0.7, approximate 95% binomial interval ±6.4 percentage points, observed tier spread 4.5 points, width confounded with seed, record-stride leakage across correlated trajectory states, normalization fit over train plus heldout, heldout-selected checkpoint optimism, no independent test set, and Speak imbalance (564/1,000). The defensible conclusion is data-limited/proxy-saturated, not capacity-limited.

The real release A/B wiring smoke used the tier-0 artifact and 20 ticks × 4 agents: UtilitySoul had deaths=0, mean needs hunger=979, energy=990, social=968, hash `0x674bce3b4ca910f9`, idle=80; OmniSoul had deaths=0, hunger=979, energy=997, social=984, hash `0x94505170865734c8`, sleep=52/speak=28; deltas were deaths=0, hunger=+0.0, energy=+7.7, social=+15.1. It took 8.75 s in release. Release determinism passed at 8 agents × 40 ticks in 76 s, while the debug run exceeded 3,600 s; therefore debug performance and throughput remain unresolved. This is a release execution check, not a policy-quality or promotion result.

The corrected v1 sequence is now complete: group-partitioned 1K evaluation, train-only normalization, validation-only checkpoint selection, untouched test evaluation, explicit expertise, and paired simulator promotion. The planned 5K→100K scaling run stopped at the 1K promotion gate and was not executed. The capacity ladder is **SKIP**: no valid cross-width evidence, hence no capacity claim.

#### 5.7 Corrected v1 reference and expertise axis

The corrected 1K reference has 500 train rows over 2 whole world/trajectory groups, 250 validation rows over 1 group, and 250 untouched test rows over 1 group. The 296-hidden-unit (311,997-parameter) checkpoint uses train-only normalization and validation selection; three model seeds (`20260715`, `20260716`, `20260717`) each reach `0.884` test action match. The corrected validation-summary hash is `d5f3d6a85002e2ef6b6b8a92a93077eba2a71382556f38cdbaa63ff7d57562dc`.

Expertise is an explicit Python/ONNX/Rust one-hot contract (`novice`, `capable`, `expert`), with deliberate `capable` default for legacy 3-input graphs. The matched set has 3,000 records and 1,000 matched states, but only 3/1,000 strict triplet-separable states; this label degeneracy must remain prominent. Three expertise checkpoints passed preregistered paired simulator separation across 72 release runs. Expert-minus-novice balanced means were `1.484375`, `1.5`, and `1.4375`; death-rate CI calculations use rates, not raw counts. The preregistration hash is `97f056f6113149ece76a391da2bd1e66d8f83f811082ff5b916119905937bdc5`.

Spark is a persona-aware candidate generator and deterministic 8-tick rollout selects legal capable targets. Promotion is simulator behavior, not action match alone. The historical pilot remains invalid pipeline evidence (4.5-point spread within approximately ±6.4-point uncertainty with confounds); neither it nor the skipped scaling/ladder supports a capacity or scaling claim.

### 6. Methodology transparency: AI-agent team under human direction

- State plainly that an AI-agent team built the system under human direction. The operator is the author and made the scope, contract, model-assignment, and adjudication decisions.
- Identify the orchestration record as the provenance source: the historical record covers three actions and 20 v0/v0.5 steps; the corrected v1 action adds the data, expertise, promotion, and documentation gates recorded below. Research used sonnet-class scouts; v0 implementation waves used Opus low/medium; v0.5 and v1 implementation/review used `gpt-5.6-luna` high effort.
- Describe agents as implementation/research instruments, not independent authors or scientific validators. The human operator selected claims, reviewed diffs, required adversarial episodes, and rejected the pre-fix habit telemetry as a final metric.
- Include prompts/agent assignments, commit history, test commands, model/backend versions, and the unedited provenance export as supplementary material where venue policy permits.
- Report limitations: agent-generated code and prose can encode omissions; green tests can hide unreachable or mis-accounted behavior; review and targeted adversarial tests are part of the method, not evidence that all behavior is correct.

### 7. Limitations and next experiments

The following are **gaps requiring new experiments before submission** rather than claims supported by this record:

- Multi-seed sociality distributions, not only aggregate histogram bounds or one A/B summary; include confidence intervals and sensitivity to cache TTLs/invalidation.
- Long-horizon habit-cache runs and multiple scenario packs to test whether plasticity remains useful without social lockout.
- Human evaluation of observed and backfilled dialogue for act coherence, persona consistency, and perceived appropriateness; the current gate checks non-empty/act-constrained output, not human judgment.
- Phone-hardware measurements (latency, sustained throughput, thermals, memory) for the target device classes; the current device budget is an M4 Pro measurement.
- Independent replication on another machine/backend and a published replay-log fixture.
- A broader trained SOUL-v1 policy beyond the corrected OMNI reference, plus evaluation across packs/populations; v0/v0.5 contain no trained SOUL model.
- Fair compute accounting for scorer calls, cache hits, social passthrough, and render work, with telemetry independently cross-checked against intent/event logs.
- A stronger PID-reuse mitigation if process lifecycle is treated as a production claim; current handling narrows but does not make the TOCTOU atomic.

### 8. Availability and artifact plan

- Repository source and Rust workspace at the implementation history ending in `490e656`.
- `DESIGN.md` for contracts and architecture; `README.md` for current measured gates; [`experiment-log.md`](experiment-log.md) for chronological provenance.
- Automated tests and `scripts/demo.sh` as the executable artifact. Live-model tests remain opt-in because they spawn `llama-server`.
- Archive the model/backend identifiers and machine details alongside any future performance table; do not turn external phone estimates into project measurements.

## Claim-to-evidence ledger

| Claim | Current evidence | Commit / log pointer | Status before submission |
| --- | --- | --- | --- |
| Deterministic replayable platform; intents are ground truth | `determinism_same_seed`, replay and seed-divergence tests; canonical sort test | `5b2b72a`; `mw-v0` steps 1, 9 | Supported for the shipped kernel/pack; replicate externally |
| Latent dialogue with always-applied mechanics and attention-gated TEXT | `pipeline_is_deterministic`, `backfill_renders_then_caches`, live render-counter gate | `590c8b7`; `mw-v0` step 7 | Supported as a functional systems claim; human-quality evaluation needed |
| Habit cache saves repeated scoring while preserving social behavior | `different_histories_diverge_cache_contents`; 50×10k and demo telemetry; review fixes | `35b1921`, `490e656`; `mw-v05` steps 4–5 | Supported only for this scenario/hardware; multi-seed distributions and pack replication needed |
| Measured on-device TEXT budget | llama.cpp runtime bench in provenance and live gates | `mw-v0` step 0; `ce43e04`; `mw-v0` step 4 | Supported for M4 Pro; phone measurements needed |
| Async viewer and stale-server cleanup | SlowMock frame-progress test; ignored stale-pidfile live gate | `8865893`; `mw-v05` steps 2–3 | Supported as lifecycle/UI behavior; PID TOCTOU remains a limitation |
| AGA is the closest reviewed cache/dialogue comparison | Research scouts and architecture synthesis | `8f30071`; `mw-arch-v1` steps 1–4 | Literature claim must be updated by a fresh systematic search before publication |
