//! Versioned JSON DTOs for the loopback web boundary.
//!
//! This module is deliberately data-only: it has no actor, server, or async
//! concerns.  The wire contract keeps simulation-sized integers as decimal
//! strings so JavaScript clients cannot lose precision.

use mw_core::{EntityId, Event, RejectReason};
use mw_neural::ExpertiseLevel;
use mw_village::{Tile, GRID};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

/// Joint compatibility version for the web DTO family.
pub const WEB_DTO_VERSION: u32 = 1;

/// Serde adapter for every simulation `u64` crossing the JSON boundary.
mod u64_str {
    use super::*;

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse::<u64>().map_err(de::Error::custom)
    }
}

/// Render a state/model hash as the canonical lowercase hexadecimal form.
pub fn hash_to_hex(value: u64) -> String {
    format!("0x{value:016x}")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunProvenanceDto {
    pub policy_id: String,
    pub model_hash: String,
    pub backend_id: String,
    pub expertise: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioDto {
    pub id: String,
    pub version: u32,
}

impl ScenarioDto {
    pub fn village() -> Self {
        Self {
            id: "village".to_string(),
            version: 1,
        }
    }
}

impl Default for ScenarioDto {
    fn default() -> Self {
        Self::village()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridDto {
    pub width: u32,
    pub height: u32,
    pub tiles: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdDto {
    pub index: u32,
    pub generation: u32,
}

impl From<EntityId> for AgentIdDto {
    fn from(id: EntityId) -> Self {
        Self {
            index: id.index(),
            generation: id.generation(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDto {
    pub id: AgentIdDto,
    pub position: [i32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotV1 {
    pub schema_version: u32,
    pub run_id: String,
    #[serde(with = "u64_str")]
    pub seed: u64,
    pub scenario: ScenarioDto,
    #[serde(with = "u64_str")]
    pub tick: u64,
    pub state_hash: String,
    pub run_provenance: RunProvenanceDto,
    pub grid: GridDto,
    pub agents: Vec<AgentDto>,
    #[serde(with = "u64_str")]
    pub event_seq: u64,
}

/// All command payloads accepted by the first web slice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandKind {
    Step { ticks: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepLimits {
    pub min: u32,
    pub max: u32,
}

/// A command advertised by capabilities (as opposed to a submitted command).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandCapability {
    #[serde(rename = "type")]
    pub command_type: String,
    pub ticks: StepLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitsDto {
    pub command_queue: String,
    pub event_ring: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtoVersions {
    pub capabilities: u32,
    pub snapshot: u32,
    pub event: u32,
    pub command: u32,
    pub command_result: u32,
    pub error: u32,
    pub control_log: u32,
}

impl DtoVersions {
    pub const V1: Self = Self {
        capabilities: WEB_DTO_VERSION,
        snapshot: WEB_DTO_VERSION,
        event: WEB_DTO_VERSION,
        command: WEB_DTO_VERSION,
        command_result: WEB_DTO_VERSION,
        error: WEB_DTO_VERSION,
        control_log: WEB_DTO_VERSION,
    };
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitiesV1 {
    pub schema_version: u32,
    pub run_id: String,
    pub run_provenance: RunProvenanceDto,
    pub dto_versions: DtoVersions,
    pub scenario: ScenarioDto,
    pub commands: Vec<CommandCapability>,
    pub limits: LimitsDto,
}

impl CapabilitiesV1 {
    pub fn for_run(run_id: &str, provenance: &crate::RunProvenance, scenario: ScenarioDto) -> Self {
        Self {
            schema_version: WEB_DTO_VERSION,
            run_id: run_id.to_string(),
            run_provenance: provenance.into(),
            dto_versions: DtoVersions::V1,
            scenario,
            commands: vec![CommandCapability {
                command_type: "step".to_string(),
                ticks: StepLimits { min: 1, max: 1 },
            }],
            limits: LimitsDto {
                command_queue: "256".to_string(),
                event_ring: "4096".to_string(),
            },
        }
    }
}

/// Typed event payload.  `event_type` on [`EventV1`] remains a stable, easy to
/// route discriminator while this enum preserves the payload's shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventPayload {
    Moved {
        actor: AgentIdDto,
        to: [i32; 2],
    },
    Interacted {
        actor: AgentIdDto,
        target: AgentIdDto,
        verb: u32,
    },
    Spoke {
        actor: AgentIdDto,
        target: AgentIdDto,
        act: u32,
        topic: u32,
    },
    Rejected {
        actor: AgentIdDto,
        reason: String,
    },
    StepApplied {
        command_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventV1 {
    pub schema_version: u32,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<String>,
    #[serde(with = "u64_str")]
    pub event_seq: u64,
    #[serde(with = "u64_str")]
    pub tick: u64,
    pub state_hash: String,
    pub event_type: String,
    pub payload: EventPayload,
}

impl EventV1 {
    pub fn from_event(run_id: &str, event_seq: u64, state_hash: u64, event: &Event) -> Self {
        let (event_type, payload, tick) = match event {
            Event::Moved { tick, actor, to } => (
                "moved",
                EventPayload::Moved {
                    actor: (*actor).into(),
                    to: [to.0, to.1],
                },
                *tick,
            ),
            Event::Interacted {
                tick,
                actor,
                target,
                verb,
            } => (
                "interacted",
                EventPayload::Interacted {
                    actor: (*actor).into(),
                    target: (*target).into(),
                    verb: *verb,
                },
                *tick,
            ),
            Event::Spoke {
                tick,
                actor,
                target,
                act,
                topic,
            } => (
                "spoke",
                EventPayload::Spoke {
                    actor: (*actor).into(),
                    target: (*target).into(),
                    act: *act,
                    topic: *topic,
                },
                *tick,
            ),
            Event::Rejected {
                tick,
                actor,
                reason,
            } => (
                "rejected",
                EventPayload::Rejected {
                    actor: (*actor).into(),
                    reason: reject_reason_name(*reason).to_string(),
                },
                *tick,
            ),
        };
        Self {
            schema_version: WEB_DTO_VERSION,
            run_id: run_id.to_string(),
            command_id: None,
            event_seq,
            tick,
            state_hash: hash_to_hex(state_hash),
            event_type: event_type.to_string(),
            payload,
        }
    }

    pub fn step_applied(
        run_id: &str,
        command_id: impl Into<String>,
        event_seq: u64,
        tick: u64,
        state_hash: u64,
    ) -> Self {
        let command_id = command_id.into();
        Self {
            schema_version: WEB_DTO_VERSION,
            run_id: run_id.to_string(),
            command_id: Some(command_id.clone()),
            event_seq,
            tick,
            state_hash: hash_to_hex(state_hash),
            event_type: "step_applied".to_string(),
            payload: EventPayload::StepApplied { command_id },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandV1 {
    pub schema_version: u32,
    pub run_id: String,
    pub command_id: String,
    #[serde(with = "u64_str")]
    pub expected_tick: u64,
    pub command: CommandKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandResultV1 {
    pub schema_version: u32,
    pub run_id: String,
    pub command_id: String,
    #[serde(with = "u64_str")]
    pub applied_tick: u64,
    pub state_hash: String,
    #[serde(with = "u64_str")]
    pub event_seq: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    IdempotencyConflict,
    TickConflict,
    Malformed,
    Schema,
    RunConflict,
    SnapshotRequired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorV1 {
    pub schema_version: u32,
    pub code: ErrorCode,
    pub message: String,
}

pub fn accept_version(version: u32) -> Result<(), ErrorV1> {
    if version == WEB_DTO_VERSION {
        Ok(())
    } else {
        Err(ErrorV1 {
            schema_version: WEB_DTO_VERSION,
            code: ErrorCode::Schema,
            message: format!("unsupported web DTO schema version {version}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlLogV1 {
    pub schema_version: u32,
    pub run_id: String,
    #[serde(with = "u64_str")]
    pub control_seq: u64,
    pub command_id: String,
    #[serde(with = "u64_str")]
    pub applied_tick: u64,
    pub control_type: String,
    pub payload: CommandKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resulting_state_hash: Option<String>,
}

impl From<&crate::RunProvenance> for RunProvenanceDto {
    fn from(provenance: &crate::RunProvenance) -> Self {
        Self {
            policy_id: provenance.policy_id.clone(),
            model_hash: provenance.model_hash.clone(),
            backend_id: provenance.backend_id.clone(),
            expertise: expertise_name(provenance.expertise).to_string(),
        }
    }
}

impl SnapshotV1 {
    pub fn from_projection(run_id: &str, snapshot: &crate::RunSnapshot, event_seq: u64) -> Self {
        let agents = snapshot
            .agents
            .iter()
            .map(|(id, (x, y))| AgentDto {
                id: (*id).into(),
                position: [*x, *y],
            })
            .collect();
        Self {
            schema_version: WEB_DTO_VERSION,
            run_id: run_id.to_string(),
            seed: snapshot.seed,
            scenario: ScenarioDto::default(),
            tick: snapshot.tick,
            state_hash: hash_to_hex(snapshot.state_hash),
            run_provenance: (&snapshot.provenance).into(),
            grid: GridDto {
                width: GRID as u32,
                height: GRID as u32,
                tiles: snapshot.grid.iter().copied().map(tile_code).collect(),
            },
            agents,
            event_seq,
        }
    }
}

fn expertise_name(level: ExpertiseLevel) -> &'static str {
    match level {
        ExpertiseLevel::Novice => "novice",
        ExpertiseLevel::Capable => "capable",
        ExpertiseLevel::Expert => "expert",
    }
}

fn tile_code(tile: Tile) -> u8 {
    match tile {
        Tile::Empty => 0,
        Tile::Home => 1,
        Tile::Bakery => 2,
        Tile::Well => 3,
        Tile::Field => 4,
    }
}

fn reject_reason_name(reason: RejectReason) -> &'static str {
    match reason {
        RejectReason::UnknownTool => "unknown_tool",
        RejectReason::OutOfRange => "out_of_range",
        RejectReason::InvalidTarget => "invalid_target",
        RejectReason::NotAfforded => "not_afforded",
        RejectReason::Depleted => "depleted",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_snapshot() -> SnapshotV1 {
        SnapshotV1 {
            schema_version: WEB_DTO_VERSION,
            run_id: "local-01".to_string(),
            seed: 1,
            scenario: ScenarioDto::village(),
            tick: 120,
            state_hash: hash_to_hex(0x0123),
            run_provenance: RunProvenanceDto {
                policy_id: "utility-v0".to_string(),
                model_hash: "none".to_string(),
                backend_id: "rust-utility".to_string(),
                expertise: "capable".to_string(),
            },
            grid: GridDto {
                width: 16,
                height: 16,
                tiles: vec![0; 256],
            },
            agents: vec![AgentDto {
                id: AgentIdDto {
                    index: 0,
                    generation: 0,
                },
                position: [8, 8],
            }],
            event_seq: 240,
        }
    }

    #[test]
    fn snapshot_u64_fields_are_decimal_strings_and_hash_is_hex() {
        let value = serde_json::to_value(sample_snapshot()).unwrap();
        assert_eq!(value["seed"], json!("1"));
        assert_eq!(value["tick"], json!("120"));
        assert_eq!(value["event_seq"], json!("240"));
        assert_eq!(value["state_hash"], json!("0x0000000000000123"));
    }

    #[test]
    fn snapshot_round_trips() {
        let snapshot = sample_snapshot();
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: SnapshotV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn additive_unknown_fields_are_accepted() {
        let mut value = serde_json::to_value(sample_snapshot()).unwrap();
        value["future_field"] = json!({"safe": true});
        assert!(serde_json::from_value::<SnapshotV1>(value).is_ok());
    }

    #[test]
    fn versions_are_rejected_when_breaking() {
        assert!(accept_version(2).is_err());
        assert!(accept_version(1).is_ok());
    }

    #[test]
    fn capabilities_advertise_only_fixed_one_tick_step() {
        let provenance = crate::RunProvenance {
            policy_id: "utility-v0".into(),
            model_hash: "none".into(),
            backend_id: "rust-utility".into(),
            expertise: ExpertiseLevel::Capable,
        };
        let capabilities = CapabilitiesV1::for_run("local-01", &provenance, ScenarioDto::village());
        assert_eq!(capabilities.commands.len(), 1);
        assert_eq!(capabilities.commands[0].command_type, "step");
        assert_eq!(
            capabilities.commands[0].ticks,
            StepLimits { min: 1, max: 1 }
        );
        assert_eq!(capabilities.dto_versions, DtoVersions::V1);
        assert_eq!(capabilities.schema_version, WEB_DTO_VERSION);
    }
}
