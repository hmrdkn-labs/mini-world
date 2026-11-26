//! Local actor host for the single-threaded simulation controller.
//!
//! The controller is deliberately constructed inside the actor thread. Only
//! versioned, owned DTOs cross the thread boundary, leaving HTTP and SSE
//! adapters free to be added without weakening the `!Send` runtime seam.

pub mod http;

use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use mw_runtime::dto::{
    accept_version, CapabilitiesV1, CommandKind, CommandResultV1, CommandV1, ControlLogV1,
    ErrorCode, ErrorV1, EventV1, SnapshotV1, WEB_DTO_VERSION,
};
use mw_runtime::{RunConfig, SimulationController};
use tokio::sync::{broadcast, oneshot};

const COMMAND_QUEUE_CAPACITY: usize = 256;
const EVENT_RING_CAPACITY: usize = 4096;

/// Messages accepted by the simulation actor.
///
/// `TextResult` is reserved for a later render-only result channel. When it is
/// enabled, fairness must bound one result batch between command/shutdown
/// opportunities so slow text work cannot starve control messages.
pub enum Envelope {
    Command {
        req_bytes: Vec<u8>,
        parsed: CommandV1,
        reply: oneshot::Sender<Result<CommandResultV1, ErrorV1>>,
    },
    Snapshot {
        reply: oneshot::Sender<SnapshotV1>,
    },
    Capabilities {
        reply: oneshot::Sender<CapabilitiesV1>,
    },
    Shutdown,
    TextResult {
        text: String,
    },
}

/// Events retained by the bounded replay ring, or a signal to recover via a
/// fresh snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventsAfter {
    Events(Vec<(u64, EventV1)>),
    SnapshotRequired,
}

/// The async-facing handle to one local simulation actor.
pub struct Host {
    tx: mpsc::SyncSender<Envelope>,
    event_ring: Arc<Mutex<VecDeque<(u64, EventV1)>>>,
    events: broadcast::Sender<EventV1>,
}

impl Clone for Host {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            event_ring: Arc::clone(&self.event_ring),
            events: self.events.clone(),
        }
    }
}

impl Host {
    /// Start an actor with a fixed run id and deterministic runtime config.
    pub fn new(config: RunConfig, run_id: impl Into<String>) -> Self {
        Self::new_with_event_ring_capacity(config, run_id, EVENT_RING_CAPACITY)
    }

    /// Start an actor with an explicit event replay ring capacity.
    pub fn new_with_event_ring_capacity(
        config: RunConfig,
        run_id: impl Into<String>,
        event_ring_capacity: usize,
    ) -> Self {
        assert!(
            event_ring_capacity > 0,
            "event ring capacity must be positive"
        );
        let run_id = run_id.into();
        let (tx, rx) = mpsc::sync_channel(COMMAND_QUEUE_CAPACITY);
        let event_ring = Arc::new(Mutex::new(VecDeque::with_capacity(event_ring_capacity)));
        let (events, _) = broadcast::channel(event_ring_capacity);

        let actor_ring = Arc::clone(&event_ring);
        let actor_events = events.clone();
        thread::Builder::new()
            .name("mw-sim-actor".to_string())
            .spawn(move || {
                actor_loop(
                    config,
                    run_id,
                    rx,
                    actor_ring,
                    actor_events,
                    event_ring_capacity,
                )
            })
            .expect("simulation actor thread must start");

        Self {
            tx,
            event_ring,
            events,
        }
    }

    /// Alias for [`Host::new`] that makes actor startup explicit at call sites.
    pub fn spawn(config: RunConfig, run_id: impl Into<String>) -> Self {
        Self::new(config, run_id)
    }

    /// Alias retained as a conventional host startup name for the next HTTP
    /// adapter layer.
    pub fn start(config: RunConfig, run_id: impl Into<String>) -> Self {
        Self::new(config, run_id)
    }
    /// Convenience constructor for the default local run.
    pub fn with_default_config(run_id: impl Into<String>) -> Self {
        Self::new(RunConfig::default(), run_id)
    }

    /// Submit the exact JSON request bytes to the actor.
    pub async fn submit_command(
        &self,
        req_bytes: impl Into<Vec<u8>>,
    ) -> Result<CommandResultV1, ErrorV1> {
        let req_bytes = req_bytes.into();
        let parsed = match serde_json::from_slice::<CommandV1>(&req_bytes) {
            Ok(parsed) => parsed,
            Err(parse_error) => {
                return Err(error(
                    ErrorCode::Malformed,
                    format!("malformed command JSON: {parse_error}"),
                ))
            }
        };
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Envelope::Command {
                req_bytes,
                parsed,
                reply,
            })
            .map_err(|_| error(ErrorCode::Malformed, "simulation actor is unavailable"))?;
        response.await.map_err(|_| {
            error(
                ErrorCode::Malformed,
                "simulation actor dropped command reply",
            )
        })?
    }

    /// Read the actor-owned immutable projection.
    pub async fn snapshot(&self) -> SnapshotV1 {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Envelope::Snapshot { reply })
            .expect("simulation actor must be running for snapshot");
        response
            .await
            .expect("simulation actor must reply to snapshot")
    }

    /// Read authoritative capabilities from the actor's run provenance.
    pub async fn capabilities(&self) -> CapabilitiesV1 {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Envelope::Capabilities { reply })
            .expect("simulation actor must be running for capabilities");
        response
            .await
            .expect("simulation actor must reply to capabilities")
    }

    /// Subscribe to accepted simulation events.
    pub fn subscribe(&self) -> broadcast::Receiver<EventV1> {
        self.events.subscribe()
    }

    /// Return retained events strictly newer than `after`, or a gap marker.
    pub fn events_after(&self, after: u64) -> EventsAfter {
        let ring = self
            .event_ring
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match ring.front() {
            None => EventsAfter::Events(Vec::new()),
            Some((oldest, _)) if after < oldest.saturating_sub(1) => EventsAfter::SnapshotRequired,
            Some(_) => EventsAfter::Events(
                ring.iter()
                    .filter(|(seq, _)| *seq > after)
                    .cloned()
                    .collect(),
            ),
        }
    }

    /// Ask the actor to stop after all earlier FIFO messages.
    pub fn shutdown(&self) {
        let _ = self.tx.send(Envelope::Shutdown);
    }
}

fn actor_loop(
    config: RunConfig,
    run_id: String,
    rx: mpsc::Receiver<Envelope>,
    event_ring: Arc<Mutex<VecDeque<(u64, EventV1)>>>,
    events: broadcast::Sender<EventV1>,
    event_ring_capacity: usize,
) {
    // This construction MUST remain inside the owning thread: the controller
    // contains Rc<RefCell<_>> state and is intentionally !Send/!Sync.
    let mut controller = SimulationController::new(config);
    let capabilities = CapabilitiesV1::for_run(
        &run_id,
        controller.provenance(),
        mw_runtime::dto::ScenarioDto::default(),
    );
    let mut event_seq = 0_u64;
    let mut control_seq = 0_u64;
    let mut idempotency: HashMap<String, (Vec<u8>, CommandResultV1)> = HashMap::new();
    let mut control_log: Vec<ControlLogV1> = Vec::new();

    while let Ok(envelope) = rx.recv() {
        match envelope {
            Envelope::Command {
                req_bytes,
                parsed,
                reply,
            } => {
                let result = process_command(
                    &mut controller,
                    &run_id,
                    req_bytes,
                    parsed,
                    &mut event_seq,
                    &mut control_seq,
                    &mut idempotency,
                    &mut control_log,
                    &event_ring,
                    &events,
                    event_ring_capacity,
                );
                let _ = reply.send(result);
            }
            Envelope::Snapshot { reply } => {
                let snapshot = SnapshotV1::from_projection(
                    &run_id,
                    &controller.snapshot_projection(),
                    event_seq,
                );
                let _ = reply.send(snapshot);
            }
            Envelope::Capabilities { reply } => {
                let _ = reply.send(capabilities.clone());
            }
            Envelope::Shutdown => break,
            Envelope::TextResult { .. } => {
                // Reserved until the text result protocol is specified.
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_command(
    controller: &mut SimulationController,
    run_id: &str,
    req_bytes: Vec<u8>,
    request: CommandV1,
    event_seq: &mut u64,
    control_seq: &mut u64,
    idempotency: &mut HashMap<String, (Vec<u8>, CommandResultV1)>,
    control_log: &mut Vec<ControlLogV1>,
    event_ring: &Arc<Mutex<VecDeque<(u64, EventV1)>>>,
    events: &broadcast::Sender<EventV1>,
    event_ring_capacity: usize,
) -> Result<CommandResultV1, ErrorV1> {
    // Validation is deliberately before idempotency and expected_tick checks.
    accept_version(request.schema_version)?;
    if request.run_id != run_id {
        return Err(error(
            ErrorCode::RunConflict,
            "command run_id does not match the active run",
        ));
    }
    if request.command_id.is_empty() {
        return Err(error(ErrorCode::Malformed, "command_id must not be empty"));
    }
    if !matches!(request.command, CommandKind::Step { ticks: 1 }) {
        return Err(error(
            ErrorCode::Malformed,
            "only step with exactly one tick is supported",
        ));
    }

    if let Some((stored_bytes, stored_result)) = idempotency.get(&request.command_id) {
        if stored_bytes == &req_bytes {
            return Ok(stored_result.clone());
        }
        return Err(error(
            ErrorCode::IdempotencyConflict,
            "command_id was already used with different request bytes",
        ));
    }

    if request.expected_tick != controller.tick() {
        return Err(error(
            ErrorCode::TickConflict,
            format!(
                "expected_tick {} does not match current tick {}",
                request.expected_tick,
                controller.tick()
            ),
        ));
    }

    let outcome = controller.step_one();
    *event_seq += 1;
    let result = CommandResultV1 {
        schema_version: WEB_DTO_VERSION,
        run_id: run_id.to_string(),
        command_id: request.command_id.clone(),
        applied_tick: outcome.tick,
        state_hash: mw_runtime::dto::hash_to_hex(outcome.state_hash),
        event_seq: *event_seq,
    };

    let event = EventV1::step_applied(
        run_id,
        request.command_id.clone(),
        *event_seq,
        outcome.tick,
        outcome.state_hash,
    );
    {
        let mut ring = event_ring
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ring.push_back((*event_seq, event.clone()));
        if ring.len() > event_ring_capacity {
            ring.pop_front();
        }
    }
    let _ = events.send(event);

    *control_seq += 1;
    control_log.push(ControlLogV1 {
        schema_version: WEB_DTO_VERSION,
        run_id: run_id.to_string(),
        control_seq: *control_seq,
        command_id: request.command_id.clone(),
        applied_tick: outcome.tick,
        control_type: "step".to_string(),
        payload: request.command.clone(),
        resulting_state_hash: Some(mw_runtime::dto::hash_to_hex(outcome.state_hash)),
    });
    idempotency.insert(request.command_id, (req_bytes, result.clone()));
    Ok(result)
}

fn error(code: ErrorCode, message: impl Into<String>) -> ErrorV1 {
    ErrorV1 {
        schema_version: WEB_DTO_VERSION,
        code,
        message: message.into(),
    }
}

/// Bind a TCP listener only on the explicitly permitted loopback addresses.
pub fn bind_loopback(addr: SocketAddr) -> std::io::Result<TcpListener> {
    let permitted = match addr.ip() {
        IpAddr::V4(ip) => ip.octets() == [127, 0, 0, 1],
        IpAddr::V6(ip) => ip.is_loopback() && ip == std::net::Ipv6Addr::LOCALHOST,
    };
    if !permitted {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "mw-host refuses non-loopback binds",
        ));
    }
    TcpListener::bind(addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_runtime::dto::{CommandKind, ErrorCode};

    fn command(id: &str, tick: u64) -> Vec<u8> {
        serde_json::to_vec(&CommandV1 {
            schema_version: WEB_DTO_VERSION,
            run_id: "test-run".to_string(),
            command_id: id.to_string(),
            expected_tick: tick,
            command: CommandKind::Step { ticks: 1 },
        })
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_idempotency_and_loopback_contract() {
        let host = Host::new(RunConfig::default(), "test-run");
        let capabilities = host.capabilities().await;
        assert_eq!(capabilities.commands[0].ticks.min, 1);
        assert_eq!(capabilities.commands[0].ticks.max, 1);
        assert_eq!(host.snapshot().await.tick, 0);

        let first_bytes = command("one", 0);
        let first = host.submit_command(first_bytes.clone()).await.unwrap();
        assert_eq!(first.applied_tick, 1);
        let event = match host.events_after(0) {
            EventsAfter::Events(events) => events.into_iter().next().expect("accepted event").1,
            EventsAfter::SnapshotRequired => panic!("fresh event ring cannot require snapshot"),
        };
        assert_eq!(event.command_id.as_deref(), Some("one"));
        assert_eq!(event.event_type, "step_applied");
        assert_eq!(event.tick, first.applied_tick);
        assert_eq!(event.state_hash, first.state_hash);
        assert_eq!(host.snapshot().await.tick, 1);
        assert_eq!(
            host.events_after(0),
            EventsAfter::Events(vec![(
                1,
                host.events_after(0).events().unwrap()[0].1.clone()
            )])
        );

        assert_eq!(host.submit_command(first_bytes).await.unwrap(), first);
        assert_eq!(host.snapshot().await.tick, 1);

        let conflict = host.submit_command(command("one", 1)).await.unwrap_err();
        assert_eq!(conflict.code, ErrorCode::IdempotencyConflict);
        let stale = host.submit_command(command("two", 0)).await.unwrap_err();
        assert_eq!(stale.code, ErrorCode::TickConflict);
        assert_eq!(host.snapshot().await.tick, 1);

        assert!(bind_loopback("0.0.0.0:0".parse().unwrap()).is_err());
        assert!(bind_loopback("127.0.0.1:0".parse().unwrap()).is_ok());
        host.shutdown();
    }

    impl EventsAfter {
        fn events(&self) -> Option<&Vec<(u64, EventV1)>> {
            match self {
                Self::Events(events) => Some(events),
                Self::SnapshotRequired => None,
            }
        }
    }
}
