//! Deterministic ONNX SOUL inference.
//!
//! The encoder in this crate is deliberately boring: its field order and
//! divisors are a byte-for-byte mirror of `training/mw_training/dataset.py`.
//! `NeuralRuntime` owns one immutable tract plan and exposes both single-record
//! and batched inference.  The kernel's validated intent log remains the replay
//! authority; floating point output is advisory.

use mw_agents::obs::{AgentObs, K_NEIGHBORS, N_EVENT_KINDS, N_STATS};
use mw_agents::persona::{Persona, N_TRAITS, N_WEIGHTS};
use mw_core::{AgentRng, Intent, Observation, SoulPolicy};
use serde::Deserialize;
use std::fmt;
use std::fs;
use std::path::Path;
use tract_onnx::prelude::*;

pub const FEATURE_DIM: usize = 129;
pub const N_TOOLS: usize = 12;
pub const MODEL_INPUT_DIM: usize = FEATURE_DIM;
pub const N_CELL_CLASSES: usize = 5;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Shape(String),
    Inference(tract_onnx::prelude::TractError),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O: {e}"),
            Self::Json(e) => write!(f, "normalization JSON: {e}"),
            Self::Shape(e) => write!(f, "shape: {e}"),
            Self::Inference(e) => write!(f, "ONNX inference: {e}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct NormStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}
impl NormStats {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let value: Self = serde_json::from_slice(&fs::read(path)?)?;
        value.validate()?;
        Ok(value)
    }
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let value: Self = serde_json::from_str(json)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), Error> {
        if self.mean.len() != FEATURE_DIM || self.std.len() != FEATURE_DIM {
            return Err(Error::Shape(format!(
                "norm stats are {} / {}, expected {FEATURE_DIM}",
                self.mean.len(),
                self.std.len()
            )));
        }
        if self.std.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(Error::Shape("norm std must be finite and positive".into()));
        }
        Ok(())
    }
}

/// Inputs to one policy row. `features` are already normalized exactly as in
/// the Python dataset; `mask` is kept separate because it is a model input and
/// the hard output mask, not part of the 129 observation features.
pub struct EncodedInput {
    pub features: [f32; FEATURE_DIM],
    pub mask: u32,
    pub present: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyLogits {
    pub tool: [f32; N_TOOLS],
    pub target: [f32; K_NEIGHBORS],
}

/// Result after hard affordance masking and deterministic argmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyOutput {
    pub tool: u32,
    pub target_slot: Option<usize>,
}

/// Encode one rich observation. This is kept public for parity fixtures and
/// for callers that schedule their own batched inference.
pub fn encode(
    agent_slot: u32,
    persona: &Persona,
    obs: &AgentObs,
    norm: &NormStats,
) -> Result<EncodedInput, Error> {
    norm.validate()?;
    let mut raw = [0.0f32; FEATURE_DIM];
    let mut i = 0;
    for x in persona.traits {
        raw[i] = x as f32 / 1000.0;
        i += 1;
    }
    for x in persona.weights {
        raw[i] = x as f32 / 1000.0;
        i += 1;
    }
    raw[i] = agent_slot as f32 / 50.0;
    i += 1;
    raw[i] = obs.tick as f32 / 10_000.0;
    i += 1;
    for x in obs.self_stats {
        raw[i] = x as f32 / 1000.0;
        i += 1;
    }
    raw[i] = obs.self_pos.0 as f32 / 16.0;
    i += 1;
    raw[i] = obs.self_pos.1 as f32 / 16.0;
    i += 1;
    one_hot(&mut raw, &mut i, obs.self_cell_class, "self_cell_class")?;
    let mut present = 0u8;
    for (slot, n) in obs.neighbors.into_iter().enumerate() {
        present |= (n.present as u8) << slot;
        raw[i] = if n.present { 1.0 } else { 0.0 };
        i += 1;
        raw[i] = n.dist2 as f32 / 512.0;
        i += 1;
        raw[i] = n.opinion as f32 / 1000.0;
        i += 1;
        raw[i] = n.faction as f32 / 4.0;
        i += 1;
        raw[i] = n.kind as f32 / 4.0;
        i += 1;
        raw[i] = n.id.map_or(0, |id| id.index()) as f32 / 50.0;
        i += 1;
        raw[i] = n.rel_pos.0 as f32 / 16.0;
        i += 1;
        raw[i] = n.rel_pos.1 as f32 / 16.0;
        i += 1;
        one_hot(&mut raw, &mut i, n.cell_class, "neighbor.cell_class")?;
    }
    for x in obs.events {
        raw[i] = x as f32 / 100.0;
        i += 1;
    }
    raw[i] = obs.goal as f32 / 8.0;
    i += 1;
    debug_assert_eq!(i, FEATURE_DIM);
    for ((x, mean), std) in raw.iter_mut().zip(&norm.mean).zip(&norm.std) {
        *x = (*x - mean) / std;
    }
    Ok(EncodedInput {
        features: raw,
        mask: obs.tool_mask & ((1 << N_TOOLS) - 1),
        present,
    })
}

fn one_hot(
    raw: &mut [f32; FEATURE_DIM],
    i: &mut usize,
    value: u8,
    field: &str,
) -> Result<(), Error> {
    if value as usize >= N_CELL_CLASSES {
        return Err(Error::Shape(format!(
            "{field} outside one-hot range: {value}"
        )));
    }
    for k in 0..N_CELL_CLASSES {
        raw[*i + k] = if k == value as usize { 1.0 } else { 0.0 };
    }
    *i += N_CELL_CLASSES;
    Ok(())
}

type Plan = TypedRunnableModel<TypedModel>;

/// A single loaded tract graph. It is immutable after construction, so callers
/// can share it across policy contexts while each tick performs one batch run.
pub struct NeuralRuntime {
    plan: Plan,
    norm: NormStats,
}
impl NeuralRuntime {
    pub fn load(model_path: impl AsRef<Path>, norm_path: impl AsRef<Path>) -> Result<Self, Error> {
        let norm = NormStats::from_path(norm_path)?;
        let plan = onnx()
            .model_for_path(model_path)
            .map_err(Error::Inference)?
            .into_optimized()
            .map_err(Error::Inference)?
            .into_runnable()
            .map_err(Error::Inference)?;
        Ok(Self { plan, norm })
    }
    pub fn norm(&self) -> &NormStats {
        &self.norm
    }
    pub fn infer_logits(&self, rows: &[EncodedInput]) -> Result<Vec<PolicyLogits>, Error> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let batch = rows.len();
        let mut input = vec![0.0f32; batch * MODEL_INPUT_DIM];
        let mut masks = vec![0i64; batch];
        for (r, row) in rows.iter().enumerate() {
            input[r * MODEL_INPUT_DIM..(r + 1) * MODEL_INPUT_DIM].copy_from_slice(&row.features);
            masks[r] = row.mask as i64;
        }
        let obs = Tensor::from_shape(&[batch, MODEL_INPUT_DIM], &input)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let mask = Tensor::from_shape(&[batch], &masks).map_err(|e| Error::Shape(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(obs.into(), mask.into()))
            .map_err(Error::Inference)?;
        if outputs.len() < 2 {
            return Err(Error::Shape(format!(
                "expected two outputs, got {}",
                outputs.len()
            )));
        }
        let tools = outputs[0]
            .to_array_view::<f32>()
            .map_err(Error::Inference)?;
        let targets = outputs[1]
            .to_array_view::<f32>()
            .map_err(Error::Inference)?;
        let mut result = Vec::with_capacity(batch);
        for r in 0..batch {
            let mut tool = [0.0; N_TOOLS];
            let mut target = [0.0; K_NEIGHBORS];
            for t in 0..N_TOOLS {
                tool[t] = tools[[r, t]];
            }
            for n in 0..K_NEIGHBORS {
                target[n] = targets[[r, n]];
            }
            result.push(PolicyLogits { tool, target });
        }
        Ok(result)
    }
    pub fn infer(&self, rows: &[EncodedInput]) -> Result<Vec<PolicyOutput>, Error> {
        let logits = self.infer_logits(rows)?;
        Ok(logits
            .into_iter()
            .zip(rows)
            .map(|(l, row)| PolicyOutput {
                tool: argmax_tool(&l.tool, row.mask),
                target_slot: argmax_target(&l.target, row),
            })
            .collect())
    }
}
fn argmax_tool(logits: &[f32; N_TOOLS], mask: u32) -> u32 {
    if mask == 0 {
        return 11;
    }
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (t, &value) in logits.iter().enumerate() {
        if mask & (1 << t) == 0 {
            continue;
        }
        if value > best_value {
            best = t;
            best_value = value;
        }
    }
    best as u32
}
fn argmax_target(logits: &[f32; K_NEIGHBORS], input: &EncodedInput) -> Option<usize> {
    let mut best = None;
    let mut best_value = f32::NEG_INFINITY;
    for (n, &value) in logits.iter().enumerate() {
        if input.present & (1 << n) == 0 {
            continue;
        }
        if value > best_value {
            best = Some(n);
            best_value = value;
        }
    }
    best
}

/// Convenience policy. For full rich observations use `infer_agent`; the
/// `SoulPolicy` adapter supplies the kernel's minimal observation with neutral
/// persona/stats, preserving the trait contract for generic callers.
pub struct NeuralSoul {
    runtime: NeuralRuntime,
    persona: Persona,
    agent_slot: u32,
}
impl NeuralSoul {
    pub fn load(model_path: impl AsRef<Path>, norm_path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            runtime: NeuralRuntime::load(model_path, norm_path)?,
            persona: Persona {
                traits: [500; N_TRAITS],
                weights: [500; N_WEIGHTS],
            },
            agent_slot: 0,
        })
    }
    pub fn with_context(runtime: NeuralRuntime, agent_slot: u32, persona: Persona) -> Self {
        Self {
            runtime,
            agent_slot,
            persona,
        }
    }
    pub fn runtime(&self) -> &NeuralRuntime {
        &self.runtime
    }
    pub fn infer_agent(
        &self,
        agent_slot: u32,
        persona: &Persona,
        obs: &AgentObs,
    ) -> Result<PolicyOutput, Error> {
        let row = encode(agent_slot, persona, obs, self.runtime.norm())?;
        self.runtime
            .infer(std::slice::from_ref(&row))
            .map(|mut x| x.remove(0))
    }
    pub fn infer_batch(&self, rows: &[EncodedInput]) -> Result<Vec<PolicyOutput>, Error> {
        self.runtime.infer(rows)
    }
}
impl SoulPolicy for NeuralSoul {
    fn decide(&mut self, observation: &Observation, _rng: &mut AgentRng) -> Intent {
        let mut neighbors = [mw_agents::obs::NeighborView::default(); K_NEIGHBORS];
        for (slot, n) in observation.neighbors.iter().enumerate() {
            if slot >= K_NEIGHBORS {
                break;
            }
            neighbors[slot].present = n.present;
            neighbors[slot].id = Some(n.id);
            neighbors[slot].rel_pos = (n.dx, n.dy);
            neighbors[slot].pos = (observation.self_pos.0 + n.dx, observation.self_pos.1 + n.dy);
            neighbors[slot].dist2 = n.dx * n.dx + n.dy * n.dy;
        }
        let obs = AgentObs {
            tick: observation.tick,
            self_stats: [500; N_STATS],
            self_pos: observation.self_pos,
            self_cell_class: 0,
            neighbors,
            events: [0; N_EVENT_KINDS],
            tool_mask: observation.tool_mask,
            goal: 0,
        };
        let out = self
            .infer_agent(self.agent_slot, &self.persona, &obs)
            .unwrap_or(PolicyOutput {
                tool: 11,
                target_slot: None,
            });
        match out.tool {
            0 => {
                let (dx, dy) = out
                    .target_slot
                    .and_then(|i| obs.neighbors[i].present.then_some(obs.neighbors[i].rel_pos))
                    .map(|(x, y)| (x.signum(), y.signum()))
                    .unwrap_or((1, 0));
                Intent::Move { dx, dy }
            }
            4 => out
                .target_slot
                .and_then(|i| obs.neighbors[i].id)
                .map_or(Intent::Idle, |target| Intent::Speak {
                    target,
                    act: 0,
                    topic: 0,
                }),
            11 => Intent::Idle,
            tool => {
                out.target_slot
                    .and_then(|i| obs.neighbors[i].id)
                    .map_or(Intent::Idle, |target| Intent::Interact {
                        target,
                        verb: tool,
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_agents::obs::NeighborView;
    #[test]
    fn encoder_layout_and_normalization() {
        let norm = NormStats {
            mean: vec![0.0; FEATURE_DIM],
            std: vec![1.0; FEATURE_DIM],
        };
        let p = Persona {
            traits: [1000; 5],
            weights: [1000; 3],
        };
        let o = AgentObs {
            tick: 10000,
            self_stats: [1000; 3],
            self_pos: (16, -16),
            self_cell_class: 2,
            neighbors: [NeighborView::default(); 8],
            events: [100; 4],
            tool_mask: 0xfff,
            goal: 8,
        };
        let x = encode(50, &p, &o, &norm).unwrap();
        assert_eq!(x.features.len(), FEATURE_DIM);
        assert_eq!(x.features[0], 1.0);
        assert_eq!(x.features[13], 1.0);
        assert_eq!(x.features[20], 0.0);
        assert_eq!(x.features[17], 1.0);
        assert_eq!(x.mask, 0xfff);
    }
    #[test]
    fn bundled_model_loads_and_runs() {
        let rt = NeuralRuntime::load(
            "../../training/artifacts/model.onnx",
            "../../training/artifacts/norm_stats.json",
        )
        .unwrap();
        let p = Persona {
            traits: [500; 5],
            weights: [500; 3],
        };
        let o = AgentObs {
            tick: 0,
            self_stats: [500; 3],
            self_pos: (0, 0),
            self_cell_class: 0,
            neighbors: [NeighborView::default(); 8],
            events: [0; 4],
            tool_mask: 1 << 11,
            goal: 0,
        };
        let row = encode(0, &p, &o, rt.norm()).unwrap();
        let out = rt.infer(&[row]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tool, 11);
        assert_eq!(out[0].target_slot, None);
    }
    #[derive(Debug, Deserialize)]
    struct FixtureFile {
        schema_version: u32,
        records: Vec<FixtureRecord>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureRecord {
        raw_features: Vec<f32>,
        features: Vec<f32>,
        mask: u32,
        present: u8,
        tool_logits: Vec<f32>,
        target_logits: Vec<f32>,
    }

    #[test]
    fn python_onnxruntime_fixture_matches_tract() {
        let fixtures: FixtureFile =
            serde_json::from_slice(&fs::read("../../training/artifacts/fixtures.json").unwrap())
                .unwrap();
        assert_eq!(fixtures.schema_version, 1);
        assert_eq!(fixtures.records.len(), 32);
        let rt = NeuralRuntime::load(
            "../../training/artifacts/model.onnx",
            "../../training/artifacts/norm_stats.json",
        )
        .unwrap();
        for fixture in &fixtures.records {
            assert_eq!(fixture.raw_features.len(), FEATURE_DIM);
            for ((raw, expected), (mean, std)) in fixture
                .raw_features
                .iter()
                .zip(&fixture.features)
                .zip(rt.norm().mean.iter().zip(&rt.norm().std))
            {
                let encoded = (*raw - mean) / std;
                assert!(
                    (encoded - expected).abs() <= 1e-5,
                    "feature mismatch: expected {expected}, got {encoded}"
                );
            }
        }
        let rows: Vec<_> = fixtures
            .records
            .iter()
            .map(|f| {
                let features: [f32; FEATURE_DIM] = f.features.clone().try_into().unwrap();
                EncodedInput {
                    features,
                    mask: f.mask,
                    present: f.present,
                }
            })
            .collect();
        let logits = rt.infer_logits(&rows).unwrap();
        for (fixture, actual) in fixtures.records.iter().zip(logits) {
            assert_eq!(fixture.features.len(), FEATURE_DIM);
            assert_eq!(fixture.tool_logits.len(), N_TOOLS);
            assert_eq!(fixture.target_logits.len(), K_NEIGHBORS);
            assert!(fixture.features.iter().all(|x| x.is_finite()));
            for (expected, got) in fixture.tool_logits.iter().zip(actual.tool) {
                let matches_masked = *expected <= -1.0e8 && !got.is_finite();
                assert!(
                    matches_masked || (expected - got).abs() <= 1e-3,
                    "tool logit mismatch: expected {expected}, got {got}"
                );
            }
            for (expected, got) in fixture.target_logits.iter().zip(actual.target) {
                assert!(
                    (expected - got).abs() <= 1e-3,
                    "target logit mismatch: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn bundled_model_supports_one_tick_batch() {
        let rt = NeuralRuntime::load(
            "../../training/artifacts/model.onnx",
            "../../training/artifacts/norm_stats.json",
        )
        .unwrap();
        let p = Persona {
            traits: [500; 5],
            weights: [500; 3],
        };
        let o = AgentObs {
            tick: 0,
            self_stats: [500; 3],
            self_pos: (0, 0),
            self_cell_class: 0,
            neighbors: [NeighborView::default(); 8],
            events: [0; 4],
            tool_mask: (1 << 0) | (1 << 11),
            goal: 0,
        };
        let a = encode(0, &p, &o, rt.norm()).unwrap();
        let b = encode(1, &p, &o, rt.norm()).unwrap();
        let out = rt.infer(&[a, b]).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|x| x.tool == 0 || x.tool == 11));
    }
    #[test]
    fn repeated_cpu_inference_is_bit_identical() {
        let rt = NeuralRuntime::load(
            "../../training/artifacts/model.onnx",
            "../../training/artifacts/norm_stats.json",
        )
        .unwrap();
        let p = Persona {
            traits: [500; 5],
            weights: [500; 3],
        };
        let o = AgentObs {
            tick: 9,
            self_stats: [500; 3],
            self_pos: (0, 0),
            self_cell_class: 0,
            neighbors: [NeighborView::default(); 8],
            events: [0; 4],
            tool_mask: 0xfff,
            goal: 0,
        };
        let row = encode(0, &p, &o, rt.norm()).unwrap();
        let a = rt.infer_logits(std::slice::from_ref(&row)).unwrap();
        let b = rt.infer_logits(std::slice::from_ref(&row)).unwrap();
        assert_eq!(a, b);
    }
}
