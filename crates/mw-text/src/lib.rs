//! TEXT backend — a managed `llama-server` subprocess.
//!
//! Implements [`TextBackend`]: renders dialogue for an already-committed speak
//! act (latent dialogue, DESIGN.md §4/§6) with the shared Qwen3-0.6B Q4 model.
//! We own a `llama-server` child: spawn it on a free loopback port, health-poll
//! until ready, talk OpenAI `/v1/chat/completions`, and kill it on Drop. Per
//! character we pin a server slot so its persona prefix stays warm in the KV
//! cache and repeat turns skip re-prefill.

mod prompt;
mod queue;

pub use prompt::PromptSpec;
pub use queue::{Priority, PriorityQueue};

use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use mw_core::{SpeakRequest, TextBackend};
use serde_json::{json, Value};

/// Errors carry no secrets and cross no async boundary that needs a bespoke
/// type, so a boxed std error keeps call sites honest without a new dependency.
pub type Result<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

const PIDFILE_ENV: &str = "MW_LLAMA_PIDFILE";
const PIDFILE_RELATIVE: &str = ".cache/mini-world/llama-server.pid";

/// Location of the single managed server's pidfile. Tests can isolate this
/// process-wide resource with `MW_LLAMA_PIDFILE`.
pub fn pidfile_path() -> PathBuf {
    if let Ok(path) = std::env::var(PIDFILE_ENV) {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(PIDFILE_RELATIVE)
}

fn parse_pid(contents: &str) -> Option<u32> {
    let first = contents.lines().next()?.trim();
    let value = first.strip_prefix("pid=").unwrap_or(first);
    value.parse().ok().filter(|pid: &u32| *pid != 0)
}

fn process_alive(pid: u32) -> bool {
    if !Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
    {
        return false;
    }
    // A SIGKILLed child can remain as an unreaped zombie when its parent was
    // leaked; it is no longer a live server and must not block reaping.
    Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "stat="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|status| status.split_whitespace().next().map(str::to_owned))
        .map(|status| !status.starts_with('Z'))
        .unwrap_or(false)
}

fn process_is_llama_server(pid: u32) -> bool {
    let pid = pid.to_string();
    let comm = Command::new("ps")
        .args(["-p", &pid, "-o", "comm="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|name| {
            Path::new(name.trim())
                .file_name()
                .and_then(|name| name.to_str())
                == Some("llama-server")
        });
    let command = Command::new("ps")
        .args(["-p", &pid, "-o", "command="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|line| line.split_whitespace().next().map(str::to_owned))
        .map(|executable| {
            Path::new(&executable)
                .file_name()
                .and_then(|name| name.to_str())
                == Some("llama-server")
        });
    comm == Some(true) && command == Some(true)
}

fn kill_stale(pid: u32) -> Result<()> {
    let status = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()?;
    if !status.success() && process_alive(pid) {
        return Err(format!("failed to kill stale llama-server pid {pid}").into());
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while process_alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    if process_alive(pid) {
        return Err(format!("stale llama-server pid {pid} did not exit").into());
    }
    Ok(())
}

fn reap_stale_server(path: &Path) -> Result<()> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if let Some(pid) = parse_pid(&contents) {
        if process_alive(pid) && process_is_llama_server(pid) {
            kill_stale(pid)?;
        }
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn write_pidfile(path: &Path, pid: u32, port: u16) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("pid={pid}\nport={port}\n"))?;
    Ok(())
}

fn remove_pidfile_if_owned(path: &Path, pid: u32) {
    let owned = fs::read_to_string(path)
        .ok()
        .and_then(|contents| parse_pid(&contents))
        == Some(pid);
    if owned {
        let _ = fs::remove_file(path);
    }
}
const DEFAULT_MODEL: &str = ".cache/mini-world/models/Qwen3-0.6B-Q4_0.gguf";
const LLAMA_SERVER: &str = "/opt/homebrew/bin/llama-server";

/// One rendered line plus the prefill timings that prove KV reuse.
#[derive(Clone, Debug)]
pub struct Rendered {
    pub text: String,
    /// Milliseconds spent processing prompt tokens (non-cached).
    pub prompt_ms: f64,
    /// Prompt tokens actually evaluated this call — drops to near-zero once the
    /// shared prefix is cached.
    pub prompt_n: u64,
    /// Total prompt tokens in the request (cached + evaluated).
    pub prompt_tokens: u64,
}

/// Configuration for the managed server.
pub struct Config {
    pub model_path: String,
    /// Server slots (`-np`); also the number of characters that can keep a warm
    /// KV prefix concurrently.
    pub slots: u32,
    /// How long to wait for the model to load and the server to report healthy.
    pub startup_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: model_path_from_env(),
            slots: 4,
            startup_timeout: Duration::from_secs(120),
        }
    }
}

/// Model path from `MW_MODEL_PATH`, else the documented default under `$HOME`.
fn model_path_from_env() -> String {
    if let Ok(p) = std::env::var("MW_MODEL_PATH") {
        return p;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/{DEFAULT_MODEL}")
}

/// Grab a currently-free loopback TCP port by binding to `:0` and reading back
/// the assigned port. Small TOCTOU window before the server re-binds it, which
/// is acceptable for a locally managed child.
fn free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

pub struct LlamaServerBackend {
    child: Child,
    base_url: String,
    slots: u32,
    agent: ureq::Agent,
    pidfile: PathBuf,
    /// TEXT render calls served — the attention-gate test seam. Atomic so the
    /// counter survives the `&self` render path without extra locking.
    renders: AtomicU64,
}

impl LlamaServerBackend {
    /// Spawn `llama-server` and block until it reports healthy.
    pub fn spawn(config: Config) -> Result<Self> {
        let pidfile = pidfile_path();
        reap_stale_server(&pidfile)?;
        let port = free_port()?;
        let child = Command::new(LLAMA_SERVER)
            .args([
                "-m",
                &config.model_path,
                "--port",
                &port.to_string(),
                "-np",
                &config.slots.to_string(),
                // Prompt caching is what makes slot affinity pay off; the web UI
                // is dead weight for a headless embed.
                "--cache-prompt",
                "--no-webui",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build();

        let backend = Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
            slots: config.slots,
            agent,
            pidfile,
            renders: AtomicU64::new(0),
        };
        write_pidfile(&backend.pidfile, backend.pid(), port)?;
        backend.await_health(config.startup_timeout)?;
        Ok(backend)
    }

    fn await_health(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let url = format!("{}/health", self.base_url);
        loop {
            if let Ok(resp) = self.agent.get(&url).call() {
                if resp.status() == 200 {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err("llama-server did not become healthy in time".into());
            }
            std::thread::sleep(Duration::from_millis(150));
        }
    }

    /// Render one line for a committed act. `conversation_id` maps to a stable
    /// server slot so a conversation's persona prefix stays warm across turns.
    pub fn render_line(&self, spec: &PromptSpec<'_>, conversation_id: u64) -> Result<Rendered> {
        self.renders.fetch_add(1, Ordering::Relaxed);
        let slot = (conversation_id % self.slots as u64) as i64;
        let body = json!({
            "messages": prompt::messages(spec),
            "max_tokens": 60,
            "temperature": 0.7,
            "cache_prompt": true,
            "id_slot": slot,
        });
        let resp: Value = self
            .agent
            .post(&format!("{}/v1/chat/completions", self.base_url))
            .send_json(body)?
            .into_json()?;

        let text = prompt::clean(
            resp["choices"][0]["message"]["content"]
                .as_str()
                .ok_or("chat completion missing message content")?,
        );
        let timings = &resp["timings"];
        Ok(Rendered {
            text,
            prompt_ms: timings["prompt_ms"].as_f64().unwrap_or(0.0),
            prompt_n: timings["prompt_n"].as_u64().unwrap_or(0),
            prompt_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        })
    }

    /// OS process id of the managed server child.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Total TEXT render calls served so far — proves attention-gating (only
    /// observed/backfilled conversations cost a render).
    pub fn render_count(&self) -> u64 {
        self.renders.load(Ordering::Relaxed)
    }
}

impl Drop for LlamaServerBackend {
    fn drop(&mut self) {
        // Kill then reap so no orphaned server lingers holding the model in RAM.
        let pid = self.child.id();
        let _ = self.child.kill();
        let _ = self.child.wait();
        remove_pidfile_if_owned(&self.pidfile, pid);
    }
}

impl TextBackend for LlamaServerBackend {
    /// Adapts the raw [`SpeakRequest`] codes to a prompt. Act/topic names and a
    /// real persona summary flow from the scenario manifest later; until that
    /// seam exists we map through best-effort defaults and lean on `context`.
    /// The speaker's persona keys its slot, keeping its prefix warm turn to turn.
    fn render(&self, request: &SpeakRequest<'_>) -> String {
        let persona = format!("Character #{}", request.persona);
        let topic = format!("topic #{}", request.topic);
        let spec = PromptSpec {
            persona: &persona,
            act: act_label(request.act),
            topic: &topic,
            context: request.context,
        };
        match self.render_line(&spec, request.persona) {
            Ok(r) => r.text,
            // TEXT is advisory and off the tick path; a failed render must not
            // crash the sim — drop the line and surface the cause on stderr.
            Err(e) => {
                eprintln!("mw-text: render failed: {e}");
                String::new()
            }
        }
    }
}

/// Placeholder act-code → verb mapping. Real names arrive with the scenario
/// manifest; these cover the acts named in DESIGN/task and fall back safely.
fn act_label(code: u32) -> &'static str {
    match code {
        0 => "greet",
        1 => "befriend",
        2 => "taunt",
        3 => "trade with",
        4 => "threaten",
        5 => "comfort",
        _ => "speak with",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pidfile(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mini-world-{name}-{}.pid", std::process::id()))
    }

    #[test]
    fn missing_pidfile_is_ignored() {
        let path = test_pidfile("missing");
        let _ = fs::remove_file(&path);
        reap_stale_server(&path).expect("missing pidfile should be harmless");
    }

    #[test]
    fn dead_pidfile_is_cleaned_without_killing() {
        let path = test_pidfile("dead");
        fs::write(&path, "pid=4294967295\nport=1234\n").expect("write test pidfile");
        reap_stale_server(&path).expect("dead pidfile should be harmless");
        assert!(!path.exists());
    }

    #[test]
    fn reused_pidfile_is_not_killed() {
        let path = test_pidfile("reused");
        let pid = std::process::id();
        fs::write(&path, format!("pid={pid}\nport=1234\n")).expect("write test pidfile");
        assert!(process_alive(pid));
        assert!(!process_is_llama_server(pid));
        reap_stale_server(&path).expect("reused pidfile should be harmless");
        assert!(process_alive(pid));
        assert!(!path.exists());
    }
}
