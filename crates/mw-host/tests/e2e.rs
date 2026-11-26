use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use mw_host::http::router;
use mw_host::{bind_loopback, Host};
use mw_runtime::RunConfig;
use reqwest::{Client, Response, StatusCode};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::timeout;

struct TestServer {
    base_url: String,
    host: Host,
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

async fn start_server(run_id: &str) -> TestServer {
    start_server_with_capacity(run_id, 4096).await
}

async fn start_server_with_capacity(run_id: &str, event_ring_capacity: usize) -> TestServer {
    let host =
        Host::new_with_event_ring_capacity(RunConfig::default(), run_id, event_ring_capacity);
    let std_listener = bind_loopback("127.0.0.1:0".parse::<SocketAddr>().unwrap()).unwrap();
    std_listener.set_nonblocking(true).unwrap();
    let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stop_signal) = oneshot::channel();
    let app: Router = router(host.clone());
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = stop_signal.await;
            })
            .await;
    });
    TestServer {
        base_url: format!("http://{address}"),
        host,
        stop,
        task,
    }
}

impl TestServer {
    async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.task.await;
        self.host.shutdown();
    }
}

struct SseReader {
    response: Response,
    buffer: String,
}

#[derive(Debug)]
struct SseFrame {
    id: Option<String>,
    event: Option<String>,
    data: String,
}

impl SseReader {
    async fn connect(client: &Client, url: &str) -> Self {
        let response = client.get(url).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        Self {
            response,
            buffer: String::new(),
        }
    }

    async fn next(&mut self) -> SseFrame {
        loop {
            if let Some(end) = self.buffer.find("\n\n") {
                let frame_text: String = self.buffer.drain(..end + 2).collect();
                let mut id = None;
                let mut event = None;
                let mut data_lines = Vec::new();
                for line in frame_text.lines().map(|line| line.trim_end_matches('\r')) {
                    if let Some(value) = line.strip_prefix("id:") {
                        id = Some(value.trim().to_string());
                    } else if let Some(value) = line.strip_prefix("event:") {
                        event = Some(value.trim().to_string());
                    } else if let Some(value) = line.strip_prefix("data:") {
                        data_lines.push(value.trim_start().to_string());
                    }
                }
                if event.is_none() && data_lines.is_empty() {
                    continue;
                }
                return SseFrame {
                    id,
                    event,
                    data: data_lines.join("\n"),
                };
            }
            let chunk = timeout(Duration::from_secs(3), self.response.chunk())
                .await
                .expect("SSE frame should arrive promptly")
                .unwrap()
                .expect("SSE stream ended unexpectedly");
            self.buffer.push_str(std::str::from_utf8(&chunk).unwrap());
        }
    }
}

async fn get_json(client: &Client, url: &str) -> (StatusCode, String, Value) {
    let response = client.get(url).send().await.unwrap();
    let status = response.status();
    let raw = response.text().await.unwrap();
    let value = serde_json::from_str(&raw).unwrap();
    (status, raw, value)
}

async fn post_json(client: &Client, url: &str, body: &str) -> (StatusCode, String, Value) {
    let response = client
        .post(url)
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let status = response.status();
    let raw = response.text().await.unwrap();
    let value = serde_json::from_str(&raw).unwrap();
    (status, raw, value)
}

fn command(run_id: &str, command_id: &str, expected_tick: u64) -> String {
    json!({
        "schema_version": 1,
        "run_id": run_id,
        "command_id": command_id,
        "expected_tick": expected_tick.to_string(),
        "command": { "type": "step", "ticks": 1 }
    })
    .to_string()
}

fn assert_hash(value: &Value) -> &str {
    let hash = value.as_str().expect("state hash is a string");
    assert_eq!(hash.len(), 18);
    assert!(hash.starts_with("0x"));
    assert_eq!(hash, hash.to_ascii_lowercase());
    assert!(hash[2..]
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
    hash
}

fn assert_u64_string(value: &Value, field: &str) {
    assert!(
        value.get(field).and_then(Value::as_str).is_some(),
        "{field} must be a JSON string"
    );
}

fn assert_snapshot_unchanged(snapshot: &Value, expected_tick: &str, expected_hash: &str) {
    assert_eq!(snapshot["tick"], expected_tick);
    assert_eq!(snapshot["state_hash"], expected_hash);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_sse_end_to_end_contract() {
    let server = start_server("e2e-run").await;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let capabilities_url = format!("{}/v1/capabilities", server.base_url);
    let snapshot_url = format!("{}/v1/snapshot", server.base_url);
    let commands_url = format!("{}/v1/commands", server.base_url);
    let events_url = format!("{}/v1/events", server.base_url);

    let (status, capabilities_raw, capabilities) = get_json(&client, &capabilities_url).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(capabilities["schema_version"], 1);
    assert_eq!(capabilities["commands"].as_array().unwrap().len(), 1);
    assert_eq!(
        capabilities["commands"][0],
        json!({"type":"step","ticks":{"min":1,"max":1}})
    );
    assert_eq!(
        capabilities["dto_versions"],
        json!({
            "capabilities": 1, "snapshot": 1, "event": 1, "command": 1,
            "command_result": 1, "error": 1, "control_log": 1,
        })
    );
    assert_eq!(capabilities["run_provenance"]["policy_id"], "utility-v0");
    assert_eq!(capabilities["run_provenance"]["backend_id"], "rust-utility");
    assert_eq!(capabilities["run_provenance"]["expertise"], "capable");
    assert!(capabilities_raw.contains("\"schema_version\":1"));

    let (status, snapshot_raw, snapshot) = get_json(&client, &snapshot_url).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(snapshot["tick"], "0");
    for field in ["seed", "tick", "event_seq"] {
        assert_u64_string(&snapshot, field);
    }
    assert_hash(&snapshot["state_hash"]);
    assert_eq!(snapshot["grid"]["width"], 16);
    assert_eq!(snapshot["grid"]["height"], 16);
    assert_eq!(snapshot["grid"]["tiles"].as_array().unwrap().len(), 256);
    assert!(!snapshot["agents"].as_array().unwrap().is_empty());
    for agent in snapshot["agents"].as_array().unwrap() {
        assert!(agent["id"]["index"].is_number());
        assert!(agent["id"]["generation"].is_number());
        assert_eq!(agent["position"].as_array().unwrap().len(), 2);
    }
    for field in ["seed", "tick", "event_seq"] {
        assert!(
            snapshot_raw.contains(&format!("\"{field}\":\"")),
            "raw {field} is not encoded as a string"
        );
    }
    let initial_hash = snapshot["state_hash"].as_str().unwrap().to_string();

    let mut first_events = SseReader::connect(&client, &events_url).await;
    let first_body = command("e2e-run", "one", 0);
    let (status, first_raw, first_result) = post_json(&client, &commands_url, &first_body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first_result["applied_tick"], "1");
    assert_eq!(first_result["command_id"], "one");
    assert_u64_string(&first_result, "applied_tick");
    assert_u64_string(&first_result, "event_seq");
    assert_hash(&first_result["state_hash"]);
    assert_ne!(first_result["state_hash"], initial_hash);
    assert!(first_raw.contains("\"applied_tick\":\"1\""));
    let first_frame = first_events.next().await;
    assert_eq!(first_frame.id.as_deref(), Some("1"));
    assert_eq!(first_frame.event.as_deref(), Some("step_applied"));
    let first_event: Value = serde_json::from_str(&first_frame.data).unwrap();
    assert_eq!(first_event["event_seq"], "1");
    assert_eq!(first_event["tick"], "1");
    assert_eq!(first_event["command_id"], "one");
    assert_u64_string(&first_event, "event_seq");
    assert_u64_string(&first_event, "tick");
    assert_hash(&first_event["state_hash"]);
    assert_eq!(first_event["state_hash"], first_result["state_hash"]);
    assert!(
        timeout(Duration::from_millis(100), first_events.next())
            .await
            .is_err(),
        "one step emits exactly one SSE event"
    );
    let first_seq = first_result["event_seq"]
        .as_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    let first_hash = first_result["state_hash"].as_str().unwrap().to_string();
    drop(first_events);

    let (status, duplicate_raw, duplicate_result) =
        post_json(&client, &commands_url, &first_body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(duplicate_result, first_result);
    assert_eq!(duplicate_raw, first_raw);
    let (_, _, duplicate_snapshot) = get_json(&client, &snapshot_url).await;
    assert_snapshot_unchanged(&duplicate_snapshot, "1", &first_hash);

    let (status, _, conflict) =
        post_json(&client, &commands_url, &command("e2e-run", "one", 1)).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(conflict["code"], "idempotency_conflict");
    let (status, _, stale) = post_json(&client, &commands_url, &command("e2e-run", "two", 0)).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(stale["code"], "tick_conflict");
    let (status, _, malformed) = post_json(&client, &commands_url, "{not-json").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(malformed["code"], "malformed");
    let (_, _, rejected_snapshot) = get_json(&client, &snapshot_url).await;
    assert_snapshot_unchanged(&rejected_snapshot, "1", &first_hash);

    let mut reconnect =
        SseReader::connect(&client, &format!("{events_url}?after_seq={first_seq}")).await;
    let (_, second_raw, second_result) =
        post_json(&client, &commands_url, &command("e2e-run", "two", 1)).await;
    assert_eq!(second_result["applied_tick"], "2");
    let second_frame = reconnect.next().await;
    assert_eq!(second_frame.id.as_deref(), Some("2"));
    let second_event: Value = serde_json::from_str(&second_frame.data).unwrap();
    assert_eq!(second_event["event_seq"], "2");
    assert_eq!(second_event["tick"], "2");
    assert_eq!(second_event["state_hash"], second_result["state_hash"]);
    assert_u64_string(&second_result, "applied_tick");
    assert_u64_string(&second_result, "event_seq");
    assert!(second_raw.contains("\"applied_tick\":\"2\""));
    assert!(
        timeout(Duration::from_millis(100), reconnect.next())
            .await
            .is_err(),
        "only the missing event is replayed"
    );
    drop(reconnect);

    // Keep a live SSE consumer attached while several commands are accepted.
    let mut live = SseReader::connect(&client, &format!("{events_url}?after_seq=2")).await;
    let mut prior_seq = 2;
    let mut final_result = second_result;
    for tick in 2..5 {
        let (status, _, result) = timeout(
            Duration::from_secs(2),
            post_json(
                &client,
                &commands_url,
                &command("e2e-run", &format!("live-{tick}"), tick),
            ),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result["applied_tick"], (tick + 1).to_string());
        let frame = timeout(Duration::from_secs(2), live.next()).await.unwrap();
        let seq = frame.id.unwrap().parse::<u64>().unwrap();
        assert!(seq > prior_seq);
        prior_seq = seq;
        let event: Value = serde_json::from_str(&frame.data).unwrap();
        assert_eq!(event["event_seq"], seq.to_string());
        assert_eq!(event["tick"], (tick + 1).to_string());
        assert_u64_string(&event, "event_seq");
        assert_u64_string(&event, "tick");
        final_result = result;
    }
    assert_eq!(prior_seq, 5);
    drop(live);
    assert_hash(&final_result["state_hash"]);
    let (_, _, final_snapshot) = get_json(&client, &snapshot_url).await;
    assert_eq!(final_snapshot["tick"], "5");
    assert_eq!(final_snapshot["event_seq"], "5");
    assert_eq!(final_snapshot["state_hash"], final_result["state_hash"]);
    eprintln!(
        "http_sse final tick={} hash={}",
        final_snapshot["tick"], final_snapshot["state_hash"]
    );
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_sse_ring_miss_requires_snapshot() {
    let server = start_server_with_capacity("ring-run", 2).await;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let events_url = format!("{}/v1/events?after_seq=0", server.base_url);
    for tick in 0..=2u64 {
        let result = server
            .host
            .submit_command(command("ring-run", &format!("ring-{tick}"), tick).into_bytes())
            .await
            .unwrap();
        assert_eq!(result.applied_tick, tick + 1);
    }
    let mut replay = SseReader::connect(&client, &events_url).await;
    let frame = replay.next().await;
    assert_eq!(frame.event.as_deref(), Some("snapshot_required"));
    assert!(frame.id.is_none());
    let body: Value = serde_json::from_str(&frame.data).unwrap();
    assert_eq!(body["code"], "snapshot_required");
    drop(replay);
    server.stop().await;
}
