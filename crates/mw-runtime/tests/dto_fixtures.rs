use mw_neural::ExpertiseLevel;
use mw_runtime::dto::{
    accept_version, CapabilitiesV1, CommandKind, CommandV1, ScenarioDto, SnapshotV1,
    WEB_DTO_VERSION,
};
use mw_runtime::{RunConfig, SimulationController};
use serde_json::{json, Value};

#[test]
fn snapshot_projection_uses_wire_safe_numeric_strings() {
    let config = RunConfig {
        seed: 0xfeed_face_cafe_beef,
        agents: 3,
        expertise: ExpertiseLevel::Capable,
    };
    let mut controller = SimulationController::new(config);
    controller.step_one();
    let snapshot =
        SnapshotV1::from_projection("fixture-run", &controller.snapshot_projection(), 17);
    let value = serde_json::to_value(&snapshot).expect("snapshot serializes");

    assert_eq!(value["schema_version"], json!(WEB_DTO_VERSION));
    assert_eq!(value["seed"], json!(config.seed.to_string()));
    assert_eq!(value["tick"], json!("1"));
    assert_eq!(value["event_seq"], json!("17"));
    assert_eq!(value["state_hash"].as_str().unwrap().len(), 18);
    assert!(value["state_hash"].as_str().unwrap().starts_with("0x"));
    assert!(value["state_hash"].as_str().unwrap()[2..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

    let mut additive = value.clone();
    additive["unknown_future_field"] = json!({"safe": true});
    let decoded: SnapshotV1 =
        serde_json::from_value(additive).expect("unknown fields are additive-compatible");
    assert_eq!(decoded, snapshot);
}

#[test]
fn capabilities_and_version_negotiation_are_exact() {
    let config = RunConfig {
        seed: 9,
        agents: 2,
        expertise: ExpertiseLevel::Expert,
    };
    let controller = SimulationController::new(config);
    let capabilities =
        CapabilitiesV1::for_run("cap-run", controller.provenance(), ScenarioDto::village());

    assert_eq!(capabilities.schema_version, 1);
    assert_eq!(capabilities.dto_versions.capabilities, 1);
    assert_eq!(capabilities.dto_versions.snapshot, 1);
    assert_eq!(capabilities.dto_versions.event, 1);
    assert_eq!(capabilities.dto_versions.command, 1);
    assert_eq!(capabilities.dto_versions.command_result, 1);
    assert_eq!(capabilities.dto_versions.error, 1);
    assert_eq!(capabilities.dto_versions.control_log, 1);
    assert_eq!(capabilities.commands.len(), 1);
    assert_eq!(capabilities.commands[0].command_type, "step");
    assert_eq!(capabilities.commands[0].ticks.min, 1);
    assert_eq!(capabilities.commands[0].ticks.max, 1);

    assert!(accept_version(1).is_ok());
    assert!(accept_version(2).is_err());
}

#[test]
fn command_expected_tick_round_trips_as_decimal_string() {
    let command = CommandV1 {
        schema_version: 1,
        run_id: "command-run".to_string(),
        command_id: "command-1".to_string(),
        expected_tick: 9_007_199_254_740_993,
        command: CommandKind::Step { ticks: 1 },
    };
    let encoded = serde_json::to_value(&command).expect("command serializes");
    assert_eq!(encoded["expected_tick"], json!("9007199254740993"));
    let decoded: CommandV1 = serde_json::from_value(encoded).expect("command deserializes");
    assert_eq!(decoded, command);
}

#[test]
fn snapshot_state_hash_is_canonical_lowercase_hex() {
    let config = RunConfig {
        seed: 1,
        agents: 1,
        expertise: ExpertiseLevel::Novice,
    };
    let controller = SimulationController::new(config);
    let snapshot = SnapshotV1::from_projection("hash-run", &controller.snapshot_projection(), 0);
    let value: Value = serde_json::to_value(snapshot).expect("snapshot serializes");
    let hash = value["state_hash"].as_str().expect("hash is a string");
    assert!(hash.starts_with("0x"));
    assert_eq!(hash.len(), 18);
    assert!(hash[2..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
}
