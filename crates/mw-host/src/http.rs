use std::convert::Infallible;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mw_runtime::dto::{ErrorCode, ErrorV1, EventV1, WEB_DTO_VERSION};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::{EventsAfter, Host};

/// Build the loopback-only HTTP API router for a simulation host.
pub fn router(host: Host) -> Router {
    Router::new()
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/snapshot", get(snapshot))
        .route("/v1/commands", post(commands))
        .route("/v1/events", get(events))
        .with_state(host)
}

async fn capabilities(State(host): State<Host>) -> Json<mw_runtime::dto::CapabilitiesV1> {
    Json(host.capabilities().await)
}

async fn snapshot(State(host): State<Host>) -> Json<mw_runtime::dto::SnapshotV1> {
    Json(host.snapshot().await)
}

async fn commands(State(host): State<Host>, body: Bytes) -> Response {
    match host.submit_command(body.to_vec()).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(error) => (status_for_error(error.code), Json(error)).into_response(),
    }
}

fn status_for_error(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::RunConflict | ErrorCode::TickConflict | ErrorCode::IdempotencyConflict => {
            StatusCode::CONFLICT
        }
        ErrorCode::Malformed | ErrorCode::Schema | ErrorCode::SnapshotRequired => {
            StatusCode::BAD_REQUEST
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct EventsQuery {
    after_seq: Option<String>,
}

async fn events(
    State(host): State<Host>,
    Query(query): Query<EventsQuery>,
    headers: HeaderMap,
) -> Sse<impl tokio_stream::Stream<Item = Result<SseEvent, Infallible>> + Send + 'static> {
    // Subscribe before taking the replay snapshot so no accepted event can fall
    // between replay and live delivery.
    let receiver = host.subscribe();
    let cursor = match headers.get("last-event-id") {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0),
        None => query
            .after_seq
            .as_deref()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0),
    };

    let replay = match host.events_after(cursor) {
        EventsAfter::Events(events) => events
            .into_iter()
            .map(|(_, event)| Ok(event_to_sse(event)))
            .collect::<Vec<_>>(),
        EventsAfter::SnapshotRequired => vec![Ok(snapshot_required_event())],
    };

    let live = BroadcastStream::new(receiver).filter_map(|received| match received {
        Ok(event) => Some(Ok(event_to_sse(event))),
        // A slow consumer can outlive the bounded broadcast buffer. A typed
        // marker tells it to refetch the authoritative snapshot before resume.
        Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(_)) => {
            Some(Ok(snapshot_required_event()))
        }
    });
    // Flush response headers immediately so browser EventSource clients enter
    // the open state even when no simulation event has been accepted yet.
    let connected = tokio_stream::once(Ok(SseEvent::default().comment("connected")));
    let stream = connected.chain(tokio_stream::iter(replay)).chain(live);

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
}

fn event_to_sse(event: EventV1) -> SseEvent {
    let event_type = event.event_type.clone();
    let event_seq = event.event_seq.to_string();
    let data = serde_json::to_string(&event).expect("EventV1 must serialize");
    SseEvent::default()
        .id(event_seq)
        .event(event_type)
        .data(data)
}

fn snapshot_required_event() -> SseEvent {
    let error = ErrorV1 {
        schema_version: WEB_DTO_VERSION,
        code: ErrorCode::SnapshotRequired,
        message: "event replay cursor is older than the retained event horizon".to_string(),
    };
    SseEvent::default()
        .event("snapshot_required")
        .data(serde_json::to_string(&error).expect("ErrorV1 must serialize"))
}
