//! Deterministic ONNX SOUL inference.
//!
//! The encoder in this crate is deliberately boring: its field order and
//! divisors are a byte-for-byte mirror of `training/mw_training/dataset.py`.
//! `NeuralRuntime` owns one immutable tract plan and exposes both single-record
//! and batched inference.  The kernel's validated intent log remains the replay
//! authority; floating point output is advisory.

use blake2::{Blake2s256, Digest};
use mw_agents::obs::{AgentObs, K_NEIGHBORS, N_EVENT_KINDS, N_STATS};
use mw_agents::persona::{Persona, N_TRAITS, N_WEIGHTS};
use mw_core::{AgentRng, Intent, Observation, SoulPolicy};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

pub const FEATURE_DIM: usize = 129;
pub const N_TOOLS: usize = 12;
pub const MODEL_INPUT_DIM: usize = FEATURE_DIM;
pub const N_CELL_CLASSES: usize = 5;
pub const EXPERTISE_DIM: usize = 3;

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
/// Descriptor width used by the training exporter.
pub const OMNI_DESCRIPTOR_DIM: usize = 16;
pub const OMNI_PARAM_DIM: usize = 4;
/// The seven trained OMNI distillation artifacts, in deterministic assignment order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OmniTier {
    Tier0 = 0,
    Tier1 = 1,
    Tier2 = 2,
    Tier3 = 3,
    Tier4 = 4,
    Tier5 = 5,
    Tier6 = 6,
}

impl OmniTier {
    pub const ALL: [Self; 7] = [
        Self::Tier0, Self::Tier1, Self::Tier2, Self::Tier3, Self::Tier4, Self::Tier5, Self::Tier6,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn hidden_dim(self) -> usize {
        match self {
            Self::Tier0 => 296,
            Self::Tier1 => 448,
            Self::Tier2 => 640,
            Self::Tier3 => 896,
            Self::Tier4 => 1280,
            Self::Tier5 => 1792,
            Self::Tier6 => 2560,
        }
    }

    pub const fn artifact_dir(self) -> &'static str {
        match self {
            Self::Tier0 => "tier-0",
            Self::Tier1 => "tier-1",
            Self::Tier2 => "tier-2",
            Self::Tier3 => "tier-3",
            Self::Tier4 => "tier-4",
            Self::Tier5 => "tier-5",
            Self::Tier6 => "tier-6",
        }
    }

    pub fn model_path(self, artifact_root: impl AsRef<Path>) -> PathBuf {
        artifact_root.as_ref().join(self.artifact_dir()).join("model.onnx")
    }

    pub fn norm_path(self, artifact_root: impl AsRef<Path>) -> PathBuf {
        artifact_root
            .as_ref()
            .join(self.artifact_dir())
            .join("norm_stats.json")
    }
}

/// Assign a character to one of the trained artifacts without ambient randomness.
pub fn assign_omni_tier(seed: u64, agent_slot: u32) -> OmniTier {
    let mut value = seed ^ (agent_slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    OmniTier::ALL[(value % OmniTier::ALL.len() as u64) as usize]
}

/// Stable explicit expertise conditioning levels, encoded as novice/capable/expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExpertiseLevel {
    Novice = 0,
    Capable = 1,
    Expert = 2,
}

impl ExpertiseLevel {
    pub const fn one_hot(self) -> [f32; EXPERTISE_DIM] {
        match self {
            Self::Novice => [1.0, 0.0, 0.0],
            Self::Capable => [0.0, 1.0, 0.0],
            Self::Expert => [0.0, 0.0, 1.0],
        }
    }
}

pub const DEFAULT_EXPERTISE: ExpertiseLevel = ExpertiseLevel::Capable;


/// Inputs to one OMNI policy row. Descriptor rows are flattened row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct OmniEncodedInput {
    pub features: [f32; FEATURE_DIM],
    pub tool_descriptors: Vec<f32>,
    pub tool_ids: Vec<u32>,
    pub afforded: Vec<bool>,
    pub expertise: [f32; EXPERTISE_DIM],
    pub present: u8,
}

impl OmniEncodedInput {
    fn validate(&self) -> Result<usize, Error> {
        if self.afforded.is_empty() {
            return Err(Error::Shape(
                "OMNI manifest must contain at least one tool".into(),
            ));
        }
        if self.tool_ids.len() != self.afforded.len() {
            return Err(Error::Shape(format!(
                "OMNI tool ids have {}, expected {}",
                self.tool_ids.len(),
                self.afforded.len()
            )));
        }
        let expected = self.afforded.len() * OMNI_DESCRIPTOR_DIM;
        if self.tool_descriptors.len() != expected {
            return Err(Error::Shape(format!(
                "OMNI descriptors have {}, expected {expected}",
                self.tool_descriptors.len()
            )));
        }
        Ok(self.afforded.len())
    }
}

/// Deterministic BLAKE2s feature matching `mw_training.omni._stable_unit`.
fn stable_unit(text: &str) -> f32 {
    let digest = Blake2s256::digest(text.as_bytes());
    let value = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    value as f32 / 4_294_967_296.0
}

/// Encode manifest descriptors exactly as the Python OMNI exporter does.
pub fn descriptor_rows(manifest: &mw_core::ActionManifest) -> Result<Vec<f32>, Error> {
    use mw_core::ArgKind;
    if manifest.tools.is_empty() {
        return Err(Error::Shape(
            "OMNI manifest must contain at least one tool".into(),
        ));
    }
    let last = manifest.tools.len().saturating_sub(1).max(1) as f32;
    let mut rows = Vec::with_capacity(manifest.tools.len() * OMNI_DESCRIPTOR_DIM);
    for (index, tool) in manifest.tools.iter().enumerate() {
        let kinds: Vec<&str> = tool
            .args
            .iter()
            .map(|arg| match &arg.kind {
                ArgKind::EntityRef => "entity",
                ArgKind::Scalar => "scalar",
                ArgKind::Enum { .. } => "enum",
            })
            .collect();
        let lower = tool.name.to_lowercase();
        let has_entity = kinds
            .iter()
            .any(|kind| matches!(*kind, "entity" | "entity_ref" | "pointer"));
        let has_scalar = kinds
            .iter()
            .any(|kind| matches!(*kind, "scalar" | "number" | "float" | "int"));
        let has_enum = kinds
            .iter()
            .any(|kind| matches!(*kind, "enum" | "string" | "item"));
        let kind_text = kinds.join(",");
        rows.extend([
            index as f32 / last,
            kinds.len() as f32 / 8.0,
            has_entity as u8 as f32 / 8.0,
            has_scalar as u8 as f32 / 8.0,
            has_enum as u8 as f32 / 8.0,
            stable_unit(&tool.name),
            stable_unit(&lower),
            lower.contains("move") as u8 as f32,
            (lower.contains("speak") || lower.contains("talk")) as u8 as f32,
            (lower.contains("idle") || lower.contains("wait")) as u8 as f32,
            (lower.contains("target") || has_entity) as u8 as f32,
            (lower.contains("param") || !kinds.is_empty()) as u8 as f32,
            (index % 7) as f32 / 7.0,
            (index % 11) as f32 / 11.0,
            stable_unit(&format!("args:{kind_text}")),
            1.0,
        ]);
    }
    Ok(rows)
}

/// Encode a rich observation with the legacy-capable default expertise level.
pub fn encode_omni(
    agent_slot: u32,
    persona: &Persona,
    obs: &AgentObs,
    norm: &NormStats,
    manifest: &mw_core::ActionManifest,
) -> Result<OmniEncodedInput, Error> {
    encode_omni_with_expertise(agent_slot, persona, obs, norm, manifest, DEFAULT_EXPERTISE)
}

/// Encode a rich observation with explicit deterministic expertise conditioning.
pub fn encode_omni_with_expertise(
    agent_slot: u32,
    persona: &Persona,
    obs: &AgentObs,
    norm: &NormStats,
    manifest: &mw_core::ActionManifest,
    expertise: ExpertiseLevel,
) -> Result<OmniEncodedInput, Error> {
    let base = encode(agent_slot, persona, obs, norm)?;
    Ok(OmniEncodedInput {
        features: base.features,
        tool_descriptors: descriptor_rows(manifest)?,
        tool_ids: manifest.tools.iter().map(|tool| tool.id).collect(),
        afforded: manifest
            .tools
            .iter()
            .map(|tool| tool.id < 32 && obs.tool_mask & (1u32 << tool.id) != 0)
            .collect(),
        expertise: expertise.one_hot(),
        present: base.present,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyLogits {
    pub tool: [f32; N_TOOLS],
    pub target: [f32; K_NEIGHBORS],
}
#[derive(Clone, Debug, PartialEq)]
pub struct OmniPolicyLogits {
    pub tool_scores: Vec<f32>,
    pub target: [f32; K_NEIGHBORS],
    pub params: [f32; OMNI_PARAM_DIM],
}

#[derive(Clone, Debug, PartialEq)]
pub struct OmniPolicyOutput {
    pub tool: u32,
    pub target_slot: Option<usize>,
    pub params: [f32; OMNI_PARAM_DIM],
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
type OmniModel = tract_hir::internal::InferenceModel;
type OmniPlan = TypedRunnableModel<TypedModel>;

// Tract cannot optimize the exported symbolic Expand for a dynamic tool axis;
// specialize concrete `(batch, tools)` facts once and reuse each plan.
fn compile_omni_plan(
    model: &OmniModel,
    batch: usize,
    tools: usize,
    has_expertise: bool,
) -> Result<OmniPlan, Error> {
    let mut model = model.clone();
    model
        .set_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), [batch, FEATURE_DIM]),
        )
        .map_err(Error::Inference)?;
    model
        .set_input_fact(
            1,
            InferenceFact::dt_shape(f32::datum_type(), [batch, tools, OMNI_DESCRIPTOR_DIM]),
        )
        .map_err(Error::Inference)?;
    model
        .set_input_fact(
            2,
            InferenceFact::dt_shape(f32::datum_type(), [batch, tools]),
        )
        .map_err(Error::Inference)?;
    if has_expertise {
        model
            .set_input_fact(
                3,
                InferenceFact::dt_shape(f32::datum_type(), [batch, EXPERTISE_DIM]),
            )
            .map_err(Error::Inference)?;
    }
    model
        .into_typed()
        .map_err(Error::Inference)?
        .into_runnable()
        .map_err(Error::Inference)
}

/// Dynamic-manifest ONNX runtime. Tool rows remain data, so a manifest can
/// grow without changing the output classifier.
pub struct OmniRuntime {
    model: OmniModel,
    norm: NormStats,
    has_expertise: bool,
    plans: RefCell<HashMap<(usize, usize), OmniPlan>>,
}

impl OmniRuntime {
    pub fn load(model_path: impl AsRef<Path>, norm_path: impl AsRef<Path>) -> Result<Self, Error> {
        let norm = NormStats::from_path(norm_path)?;
        let model = onnx()
            .model_for_path(model_path)
            .map_err(Error::Inference)?;
        let has_expertise = model
            .input_outlets()
            .map_err(Error::Inference)?
            .len()
            >= 4;
        Ok(Self {
            model,
            norm,
            has_expertise,
            plans: RefCell::new(HashMap::new()),
        })
    }

    pub fn norm(&self) -> &NormStats {
        &self.norm
    }

    pub fn infer_logits(&self, rows: &[OmniEncodedInput]) -> Result<Vec<OmniPolicyLogits>, Error> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let tools = rows[0].validate()?;
        if rows.iter().any(|row| row.validate().ok() != Some(tools)) {
            return Err(Error::Shape(
                "OMNI batch rows must have the same tool count".into(),
            ));
        }
        let batch = rows.len();
        let plan_key = (batch, tools);
        if !self.plans.borrow().contains_key(&plan_key) {
            let plan = compile_omni_plan(&self.model, batch, tools, self.has_expertise)?;
            self.plans.borrow_mut().insert(plan_key, plan);
        }
        let mut obs_data = Vec::with_capacity(batch * FEATURE_DIM);
        let mut descriptor_data = Vec::with_capacity(batch * tools * OMNI_DESCRIPTOR_DIM);
        let mut afforded_data = Vec::with_capacity(batch * tools);
        let mut expertise_data = Vec::with_capacity(batch * EXPERTISE_DIM);
        for row in rows {
            obs_data.extend_from_slice(&row.features);
            descriptor_data.extend_from_slice(&row.tool_descriptors);
            afforded_data.extend(
                row.afforded
                    .iter()
                    .map(|&value| if value { 1.0f32 } else { 0.0f32 }),
            );
            expertise_data.extend_from_slice(&row.expertise);
        }
        let obs = Tensor::from_shape(&[batch, FEATURE_DIM], &obs_data)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let descriptors =
            Tensor::from_shape(&[batch, tools, OMNI_DESCRIPTOR_DIM], &descriptor_data)
                .map_err(|e| Error::Shape(e.to_string()))?;
        let afforded = Tensor::from_shape(&[batch, tools], &afforded_data)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let expertise = Tensor::from_shape(&[batch, EXPERTISE_DIM], &expertise_data)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let plans = self.plans.borrow();
        let plan = plans.get(&plan_key).expect("OMNI plan inserted above");
        let outputs = if self.has_expertise {
            plan.run(tvec![obs.into(), descriptors.into(), afforded.into(), expertise.into()])
        } else {
            plan.run(tvec![obs.into(), descriptors.into(), afforded.into()])
        }
        .map_err(Error::Inference)?;
        if outputs.len() < 3 {
            return Err(Error::Shape(format!(
                "expected three OMNI outputs, got {}",
                outputs.len()
            )));
        }
        let scores = outputs[0]
            .to_array_view::<f32>()
            .map_err(Error::Inference)?;
        let targets = outputs[1]
            .to_array_view::<f32>()
            .map_err(Error::Inference)?;
        let params = outputs[2]
            .to_array_view::<f32>()
            .map_err(Error::Inference)?;
        let mut result = Vec::with_capacity(batch);
        for row in 0..batch {
            let mut tool_scores = Vec::with_capacity(tools);
            for tool in 0..tools {
                tool_scores.push(scores[[row, tool]]);
            }
            let mut target = [0.0; K_NEIGHBORS];
            for slot in 0..K_NEIGHBORS {
                target[slot] = targets[[row, slot]];
            }
            let mut parameter = [0.0; OMNI_PARAM_DIM];
            for parameter_index in 0..OMNI_PARAM_DIM {
                parameter[parameter_index] = params[[row, parameter_index]];
            }
            result.push(OmniPolicyLogits {
                tool_scores,
                target,
                params: parameter,
            });
        }
        Ok(result)
    }

    pub fn infer(&self, rows: &[OmniEncodedInput]) -> Result<Vec<OmniPolicyOutput>, Error> {
        let logits = self.infer_logits(rows)?;
        Ok(logits
            .into_iter()
            .zip(rows)
            .map(|(logit, row)| {
                let tool = argmax_omni_tool(&logit.tool_scores, row);
                OmniPolicyOutput {
                    tool,
                    target_slot: argmax_target_omni(&logit.target, row.present),
                    params: logit.params,
                }
            })
            .collect())
    }
}

fn argmax_omni_tool(scores: &[f32], row: &OmniEncodedInput) -> u32 {
    let mut best = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &score) in scores.iter().enumerate() {
        if !row.afforded[index] {
            continue;
        }
        if score > best_value {
            best = Some(row.tool_ids[index]);
            best_value = score;
        }
    }
    best.or_else(|| row.tool_ids.iter().copied().find(|&id| id == 11))
        .unwrap_or(row.tool_ids[0])
}

fn argmax_target_omni(logits: &[f32; K_NEIGHBORS], present: u8) -> Option<usize> {
    let mut best = None;
    let mut best_value = f32::NEG_INFINITY;
    for (slot, &value) in logits.iter().enumerate() {
        if present & (1 << slot) == 0 {
            continue;
        }
        if value > best_value {
            best = Some(slot);
            best_value = value;
        }
    }
    best
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

fn minimal_agent_obs(observation: &Observation) -> AgentObs {
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
    AgentObs {
        tick: observation.tick,
        self_stats: [500; N_STATS],
        self_pos: observation.self_pos,
        self_cell_class: 0,
        neighbors,
        events: [0; N_EVENT_KINDS],
        tool_mask: observation.tool_mask,
        goal: 0,
    }
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
        let obs = minimal_agent_obs(observation);
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
/// Manifest-conditioned SOUL adapter for the dynamic OMNI graph.
pub struct OmniSoul {
    runtime: OmniRuntime,
    persona: Persona,
    agent_slot: u32,
    manifest: mw_core::ActionManifest,
}

impl OmniSoul {
    pub fn load(
        model_path: impl AsRef<Path>,
        norm_path: impl AsRef<Path>,
        manifest: mw_core::ActionManifest,
    ) -> Result<Self, Error> {
        Ok(Self {
            runtime: OmniRuntime::load(model_path, norm_path)?,
            persona: Persona {
                traits: [500; N_TRAITS],
                weights: [500; N_WEIGHTS],
            },
            agent_slot: 0,
            manifest,
        })
    }
    /// Load one of the seven ladder artifacts under `artifact_root`.
    pub fn load_tier(
        artifact_root: impl AsRef<Path>,
        tier: OmniTier,
        manifest: mw_core::ActionManifest,
    ) -> Result<Self, Error> {
        Self::load(tier.model_path(&artifact_root), tier.norm_path(artifact_root), manifest)
    }


    pub fn with_context(
        runtime: OmniRuntime,
        agent_slot: u32,
        persona: Persona,
        manifest: mw_core::ActionManifest,
    ) -> Self {
        Self {
            runtime,
            agent_slot,
            persona,
            manifest,
        }
    }

    /// Consume the policy and return its runtime for another deterministic run.
    pub fn into_runtime(self) -> OmniRuntime {
        self.runtime
    }

    pub fn runtime(&self) -> &OmniRuntime {
        &self.runtime
    }
    pub fn manifest(&self) -> &mw_core::ActionManifest {
        &self.manifest
    }


    pub fn infer_agent(
        &self,
        agent_slot: u32,
        persona: &Persona,
        obs: &AgentObs,
    ) -> Result<OmniPolicyOutput, Error> {
        self.infer_agent_with_expertise(agent_slot, persona, obs, DEFAULT_EXPERTISE)
    }

    pub fn infer_agent_with_expertise(
        &self,
        agent_slot: u32,
        persona: &Persona,
        obs: &AgentObs,
        expertise: ExpertiseLevel,
    ) -> Result<OmniPolicyOutput, Error> {
        let row = encode_omni_with_expertise(
            agent_slot,
            persona,
            obs,
            self.runtime.norm(),
            &self.manifest,
            expertise,
        )?;
        self.runtime
            .infer(std::slice::from_ref(&row))
            .map(|mut outputs| outputs.remove(0))
    }

    pub fn infer_batch(&self, rows: &[OmniEncodedInput]) -> Result<Vec<OmniPolicyOutput>, Error> {
        self.runtime.infer(rows)
    }
}

impl SoulPolicy for OmniSoul {
    fn decide(&mut self, observation: &Observation, _rng: &mut AgentRng) -> Intent {
        let obs = minimal_agent_obs(observation);
        let out = self
            .infer_agent(self.agent_slot, &self.persona, &obs)
            .unwrap_or(OmniPolicyOutput {
                tool: 11,
                target_slot: None,
                params: [0.0; OMNI_PARAM_DIM],
            });
        match out.tool {
            0 => {
                let (dx, dy) = out
                    .target_slot
                    .and_then(|slot| {
                        obs.neighbors[slot]
                            .present
                            .then_some(obs.neighbors[slot].rel_pos)
                    })
                    .map(|(x, y)| (x.signum(), y.signum()))
                    .unwrap_or((1, 0));
                Intent::Move { dx, dy }
            }
            4 => out
                .target_slot
                .and_then(|slot| obs.neighbors[slot].id)
                .map_or(Intent::Idle, |target| Intent::Speak {
                    target,
                    act: 0,
                    topic: 0,
                }),
            11 => Intent::Idle,
            tool => out
                .target_slot
                .and_then(|slot| obs.neighbors[slot].id)
                .map_or(Intent::Idle, |target| Intent::Interact {
                    target,
                    verb: tool,
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_core::{agent_rng, stream, Intent, KernelPack, ScenarioPack, SoulPolicy, World};
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
    fn expertise_encoding_is_stable_and_manifest_conditioned() {
        let norm = NormStats {
            mean: vec![0.0; FEATURE_DIM],
            std: vec![1.0; FEATURE_DIM],
        };
        let persona = Persona {
            traits: [500; 5],
            weights: [500; 3],
        };
        let obs = AgentObs {
            tick: 0,
            self_stats: [500; 3],
            self_pos: (0, 0),
            self_cell_class: 0,
            neighbors: [NeighborView::default(); 8],
            events: [0; 4],
            tool_mask: 0xfff,
            goal: 0,
        };
        let manifest = KernelPack::new().manifest().clone();
        let novice = encode_omni_with_expertise(
            0,
            &persona,
            &obs,
            &norm,
            &manifest,
            ExpertiseLevel::Novice,
        )
        .unwrap();
        let capable = encode_omni_with_expertise(
            0,
            &persona,
            &obs,
            &norm,
            &manifest,
            ExpertiseLevel::Capable,
        )
        .unwrap();
        let expert = encode_omni_with_expertise(
            0,
            &persona,
            &obs,
            &norm,
            &manifest,
            ExpertiseLevel::Expert,
        )
        .unwrap();
        assert_eq!(novice.expertise, [1.0, 0.0, 0.0]);
        assert_eq!(capable.expertise, [0.0, 1.0, 0.0]);
        assert_eq!(expert.expertise, [0.0, 0.0, 1.0]);
        assert_eq!(encode_omni(0, &persona, &obs, &norm, &manifest).unwrap().expertise, capable.expertise);
        assert_eq!(novice.tool_descriptors, capable.tool_descriptors);
        assert_eq!(novice.afforded, capable.afforded);
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
    #[test]
    fn neural_soul_implements_policy_socket() {
        let mut soul = NeuralSoul::load(
            "../../training/artifacts/model.onnx",
            "../../training/artifacts/norm_stats.json",
        )
        .unwrap();
        let mut world = World::new(7);
        let id = world.spawn((0, 0));
        let mut observation = world.observe(id);
        observation.tool_mask = 1 << 11;
        let mut rng = agent_rng(7, id, stream::SOUL, observation.tick);
        assert_eq!(soul.decide(&observation, &mut rng), Intent::Idle);
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
    #[test]
    fn best_ladder_tier_loads_through_omni_soul() {
        let pack = KernelPack::new();
        let mut soul = OmniSoul::load_tier(
            "../../training/artifacts/ladder",
            OmniTier::Tier0,
            pack.manifest().clone(),
        )
        .unwrap();
        let persona = Persona {
            traits: [500; 5],
            weights: [500; 3],
        };
        let observation = AgentObs {
            tick: 1,
            self_stats: [500; 3],
            self_pos: (0, 0),
            self_cell_class: 0,
            neighbors: [NeighborView::default(); 8],
            events: [0; 4],
            tool_mask: (1 << 0) | (1 << 3),
            goal: 0,
        };
        let output = soul.infer_agent(0, &persona, &observation).unwrap();
        assert!(matches!(output.tool, 0 | 3));
    }

    #[test]
    fn omni_tier_assignment_and_artifact_mapping_are_deterministic() {
        let widths: Vec<_> = OmniTier::ALL.iter().map(|tier| tier.hidden_dim()).collect();
        assert_eq!(widths, vec![296, 448, 640, 896, 1280, 1792, 2560]);
        for seed in [0, 1, 7, u64::MAX] {
            for slot in 0..64 {
                assert_eq!(assign_omni_tier(seed, slot), assign_omni_tier(seed, slot));
            }
        }
        let tier = OmniTier::Tier4;
        assert_eq!(tier.model_path("training/artifacts"), PathBuf::from("training/artifacts/tier-4/model.onnx"));
        assert_eq!(tier.norm_path("training/artifacts"), PathBuf::from("training/artifacts/tier-4/norm_stats.json"));
    }

}
