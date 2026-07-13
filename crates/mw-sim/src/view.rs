//! Ratatui debug viewer over the live village sim (DESIGN.md §10 focus/LOD, §4
//! latent dialogue, §7 memory). It is a *read-only* window on the sim: it only
//! ever advances the world through the same `World::step` intent pipeline the
//! soak uses and the analytic `director::fast_forward` API — it never writes
//! world state directly. Frame pacing reads the wall-clock (allowed); nothing
//! wall-clock-derived ever feeds sim state.
//!
//! Panes: a 16x16 map with ring-colored agent glyphs and a movable focus point,
//! an agent inspector (needs, persona, opinions, salient facts, last action), a
//! scrolling event feed (with dialogue lines as they are observed), and a
//! conversation list where a latent row can be backfilled on demand. `--smoke`
//! renders one frame to a `TestBackend` and exits, so CI needs no TTY.

use std::io;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CtEvent, KeyCode, KeyEventKind,
    MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::{Backend, CrosstermBackend, TestBackend};
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use mw_agents::dialogue::{
    ConversationLog, DialogueRenderer, FocusPoint, MockRenderer, PersonaCard, PersonaRegistry,
    SliceRegistry, Vocab,
};
use mw_agents::memory::{Memory, OPINION_ONE};
use mw_agents::persona::Persona;
use mw_agents::soul::UtilitySoul;
use mw_core::{EntityId, Event, World};
use mw_text::{Config, LlamaServerBackend};
use mw_village::{decode, tile_at, Tile, VillagePack, GRID, MAX_NEED};

use crate::dialogue::LlamaDialogue;
use crate::director::{self, Director, FfConfig, Ring, RingConfig, TICKS_PER_DAY};
use crate::soak::{self, VillageBody};

/// How the viewer is opened.
#[derive(Clone, Copy, Debug)]
pub struct ViewConfig {
    pub seed: u64,
    pub agents: i32,
    /// Use the real TEXT backend for dialogue backfill (else the offline mock).
    pub live: bool,
}

/// Sim advance rate, in ticks stepped per rendered frame. `Normal`/`Fast` keep
/// the 1x/8x ratio the controls advertise; `Paused` freezes the sim.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Speed {
    Paused,
    Normal,
    Fast,
}

impl Speed {
    fn ticks_per_frame(self) -> u64 {
        match self {
            Speed::Paused => 0,
            Speed::Normal => 3,
            Speed::Fast => 24,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Speed::Paused => "PAUSED",
            Speed::Normal => "1x",
            Speed::Fast => "8x",
        }
    }
}

/// The dialogue renderer behind the `DialogueRenderer` seam: an offline mock, or
/// a managed llama-server when the viewer is opened live (MW_TEXT_LIVE=1).
enum Text {
    Mock(MockRenderer),
    Live(LlamaServerBackend),
}
#[derive(Clone, Copy, Debug)]
enum RenderSource {
    Heard,
    Backfill,
}

struct RenderJob {
    index: usize,
    speaker: PersonaCard,
    listener: PersonaCard,
    act: String,
    topic: String,
    context: String,
    conversation: u64,
}

struct RenderResponse {
    index: usize,
    text: String,
}

trait RenderBackend: Send + 'static {
    fn render(&mut self, job: &RenderJob) -> String;
}

impl RenderBackend for Text {
    fn render(&mut self, job: &RenderJob) -> String {
        let req = mw_agents::dialogue::RenderRequest {
            speaker: &job.speaker,
            listener: &job.listener,
            act: &job.act,
            topic: &job.topic,
            context: &job.context,
            conversation: job.conversation,
        };
        match self {
            Text::Mock(renderer) => renderer.render(&req),
            Text::Live(backend) => LlamaDialogue { backend }.render(&req),
        }
    }
}

struct RenderWorker {
    requests: Sender<RenderJob>,
    responses: Receiver<RenderResponse>,
    thread: Option<JoinHandle<()>>,
}

impl RenderWorker {
    fn new<B: RenderBackend>(mut backend: B) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<RenderJob>();
        let (response_tx, response_rx) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("mw-text-render".to_string())
            .spawn(move || {
                while let Ok(job) = request_rx.recv() {
                    let response = RenderResponse {
                        index: job.index,
                        text: backend.render(&job),
                    };
                    if response_tx.send(response).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn dialogue render worker");
        Self {
            requests: request_tx,
            responses: response_rx,
            thread: Some(thread),
        }
    }

    fn request(&self, job: RenderJob) -> bool {
        self.requests.send(job).is_ok()
    }

    fn try_response(&self) -> Result<RenderResponse, TryRecvError> {
        self.responses.try_recv()
    }
}

impl Drop for RenderWorker {
    fn drop(&mut self) {
        // Closing the request channel lets the worker exit after its final line.
        // Joining keeps the managed llama-server child owned and reaped.
        let (closed_tx, _) = mpsc::channel();
        let old = std::mem::replace(&mut self.requests, closed_tx);
        drop(old);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct PendingRender {
    index: usize,
    source: RenderSource,
}

/// Everything the viewer owns: the live sim (`pack`/`world`/`soul`), the LOD
/// director, the conversation ledger, and the UI cursor state.
pub struct App {
    cfg: ViewConfig,
    pack: Rc<VillagePack>,
    world: World,
    ids: Vec<EntityId>,
    positions: Vec<(i32, i32)>,
    soul: UtilitySoul<VillageBody>,
    director: Director,
    registry: SliceRegistry,
    vocab: Vocab,
    log: ConversationLog,
    renderer: RenderWorker,
    pending: Vec<PendingRender>,
    focus: (i32, i32),
    selected: usize,
    dlg_sel: usize,
    speed: Speed,
    last_events: usize,
    feed: Vec<String>,
    popup: Option<String>,
    map_area: Rect,
}

impl App {
    pub fn new(cfg: ViewConfig) -> Self {
        // Mirror the soak's wiring so the viewer runs the exact same sim, then
        // hold the pieces together via an `Rc` pack (the soul's body shares it)
        // so nothing is a self-referential borrow.
        let pack = Rc::new(VillagePack::new());
        let mut world = World::with_pack(cfg.seed, &*pack);
        let positions = soak::start_positions(cfg.agents);
        let ids: Vec<EntityId> = positions.iter().map(|&p| world.spawn(p)).collect();
        let personas: Vec<Persona> = ids.iter().map(|&id| Persona::new(cfg.seed, id)).collect();
        let factions: Vec<u8> = personas.iter().map(|p| p.faction()).collect();
        let memories: Vec<Memory> = ids
            .iter()
            .map(|&id| Memory::new(id, soak::verb_affect()))
            .collect();
        let registry = SliceRegistry::village(&personas);
        let body = VillageBody::new(Rc::clone(&pack), factions);
        let soul = UtilitySoul::new(
            body,
            soak::tool_table(),
            ids.clone(),
            personas,
            memories,
            positions.clone(),
        );
        let director = Director::new(RingConfig::default(), ids.len(), (8, 8));
        // The live backend is expensive (spawns llama-server + model); only pay
        // for it when explicitly asked, and fall back to the mock on failure.
        let text = if cfg.live {
            match LlamaServerBackend::spawn(Config::default()) {
                Ok(b) => Text::Live(b),
                Err(_) => Text::Mock(MockRenderer::new()),
            }
        } else {
            Text::Mock(MockRenderer::new())
        };
        let renderer = RenderWorker::new(text);
        Self {
            cfg,
            pack,
            world,
            ids,
            positions,
            soul,
            director,
            registry,
            vocab: Vocab::village(),
            log: ConversationLog::new(),
            renderer,
            pending: Vec::new(),
            focus: (8, 8),
            selected: 0,
            dlg_sel: 0,
            speed: Speed::Normal,
            last_events: 0,
            feed: Vec::new(),
            popup: None,
            map_area: Rect::default(),
        }
    }

    fn pos(&self, id: EntityId) -> (i32, i32) {
        self.world.entity(id).map(|e| e.pos).unwrap_or((0, 0))
    }

    /// The focus observation predicate: the same one latent dialogue uses, sized
    /// to the director's hot radius so what the camera renders matches what the
    /// LOD promotes.
    fn focus_point(&self) -> FocusPoint {
        FocusPoint::new(self.focus, RingConfig::default().hot_radius)
    }

    /// Advance the live sim one tick: step the kernel, route events to memory,
    /// the conversation ledger, and the feed, refresh the director rings, then
    /// render any conversation the focus now observes (attention-gating).
    fn step(&mut self) {
        self.soul.snapshot(&self.world);
        // LOD gate: only hot (every tick) and warm (on cadence) entities run
        // SOUL; cold entities idle-extrapolate. The director's rings were set by
        // the previous tick's update, so they gate this tick's decisions.
        let director = &self.director;
        self.world
            .step_gated(&*self.pack, &mut self.soul, |id, tick| {
                director.should_run_soul(id.index() as usize, tick)
            });
        let end = self.world.event_log().len();
        // Clone the tick's new events out so the borrow of the world ends before
        // we touch other fields.
        let new: Vec<Event> = self.world.event_log()[self.last_events..end].to_vec();
        self.last_events = end;
        self.soul.observe_events(&new);
        let tick = self.world.tick();
        for ev in &new {
            if let Some(line) = self.notable(ev) {
                self.feed.push(line);
            }
            // Drama pulls a character back to full fidelity (DESIGN §10).
            if is_notable(ev) {
                self.director.note_event(actor(ev).index() as usize, tick);
            }
        }
        self.soul.decay_opinions();
        for (slot, &id) in self.ids.iter().enumerate() {
            if let Some(e) = self.world.entity(id) {
                self.positions[slot] = e.pos;
            }
        }
        self.director.update(&self.positions, tick);
        self.render_observed();
        // Keep the feed bounded; it is a scrolling tail, not a full ledger.
        if self.feed.len() > 500 {
            let cut = self.feed.len() - 500;
            self.feed.drain(0..cut);
        }
    }

    /// Queue every latent conversation the focus now observes. TEXT runs off
    /// the UI thread; the response is applied by [`Self::poll_renders`].
    fn render_observed(&mut self) {
        let focus = self.focus_point();
        let to_render: Vec<usize> = (0..self.log.len())
            .filter(|&i| {
                let r = &self.log.rows()[i];
                r.text.is_none() && focus.is_observed(self.pos(r.speaker), self.pos(r.listener))
            })
            .collect();
        for i in to_render {
            self.request_render(i, RenderSource::Heard);
        }
    }

    fn request_render(&mut self, i: usize, source: RenderSource) -> bool {
        if i >= self.log.len()
            || self.log.rows()[i].text.is_some()
            || self.pending.iter().any(|p| p.index == i)
        {
            return false;
        }
        let row = self.log.rows()[i].clone();
        let job = RenderJob {
            index: i,
            speaker: self.registry.card(row.speaker).clone(),
            listener: self.registry.card(row.listener).clone(),
            act: self
                .vocab
                .acts
                .get(row.act as usize)
                .map_or("speak with", String::as_str)
                .to_string(),
            topic: self
                .vocab
                .topics
                .get(row.topic as usize)
                .map_or("things", String::as_str)
                .to_string(),
            context: format!(
                "A chance meeting in the village; {}.",
                match row.outcome.signum() {
                    1 => "the exchange warmed relations",
                    -1 => "the exchange soured relations",
                    _ => "relations were unchanged",
                }
            ),
            conversation: row.speaker.index() as u64,
        };
        if self.renderer.request(job) {
            self.pending.push(PendingRender { index: i, source });
            true
        } else {
            false
        }
    }

    /// Apply completed lines without ever letting TEXT write into sim state.
    fn poll_renders(&mut self) {
        while let Ok(response) = self.renderer.try_response() {
            let Some(pending) = self
                .pending
                .iter()
                .position(|p| p.index == response.index)
                .map(|p| self.pending.remove(p))
            else {
                continue;
            };
            if response.index >= self.log.len() {
                continue;
            }
            if self.log.rows()[response.index].text.is_some() {
                continue;
            }
            let name = self
                .registry
                .card(self.log.rows()[response.index].speaker)
                .name
                .clone();
            let prefix = match pending.source {
                RenderSource::Heard => "[heard]",
                RenderSource::Backfill => "[backfill]",
            };
            let text = response.text;
            self.log.cache_rendered(response.index, text);
            let line = self.log.rows()[response.index]
                .text
                .as_deref()
                .unwrap_or_default();
            self.feed.push(format!("{prefix} {name}: {line}"));
        }
    }

    /// A one-line feed entry for a notable event, or `None` to skip it (plain
    /// moves are too noisy for the feed).
    fn notable(&self, ev: &Event) -> Option<String> {
        match ev {
            Event::Moved { .. } => None,
            Event::Spoke { .. } | Event::Interacted { .. } => Some(format!(
                "{} {}",
                actor_name(ev, &self.registry),
                describe(ev, &self.registry, &self.vocab)
            )),
            Event::Rejected { .. } => None,
        }
    }

    /// Analytic fast-forward of one in-game day and a digest popup — the same FF
    /// API the CLI uses. Read-only w.r.t. the live world.
    fn fast_forward(&mut self) {
        let report = director::fast_forward(FfConfig {
            seed: self.cfg.seed,
            agents: self.cfg.agents,
            ticks: TICKS_PER_DAY,
            ..FfConfig::default()
        });
        let mut s = format!(
            "Fast-forward 1 day ({} ticks)\nevents={} deaths={}\n\nTop moments:\n",
            TICKS_PER_DAY, report.ledger_len, report.deaths
        );
        for line in report.digest.top.iter().take(6) {
            s.push_str(line);
            s.push('\n');
        }
        s.push_str("\n(press any key to dismiss)");
        self.popup = Some(s);
    }

    fn move_focus(&mut self, dx: i32, dy: i32) {
        self.focus.0 = (self.focus.0 + dx).clamp(0, GRID - 1);
        self.focus.1 = (self.focus.1 + dy).clamp(0, GRID - 1);
        self.director.set_focus(self.focus);
    }

    /// Handle one key. Returns `true` when the viewer should quit.
    fn on_key(&mut self, code: KeyCode) -> bool {
        // A popup swallows the next key (except quit).
        if self.popup.is_some() {
            self.popup = None;
            return matches!(code, KeyCode::Char('q') | KeyCode::Esc);
        }
        let n = self.ids.len().max(1);
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Left => self.move_focus(-1, 0),
            KeyCode::Right => self.move_focus(1, 0),
            KeyCode::Up => self.move_focus(0, -1),
            KeyCode::Down => self.move_focus(0, 1),
            KeyCode::Tab => self.selected = (self.selected + 1) % n,
            KeyCode::BackTab => self.selected = (self.selected + n - 1) % n,
            KeyCode::Char('k') => self.dlg_sel = self.dlg_sel.saturating_sub(1),
            KeyCode::Char('j') => {
                if self.dlg_sel + 1 < self.log.len() {
                    self.dlg_sel += 1;
                }
            }
            KeyCode::Enter => {
                if self.dlg_sel < self.log.len() {
                    self.request_render(self.dlg_sel, RenderSource::Backfill);
                }
            }
            KeyCode::Char(' ') => {
                self.speed = if self.speed == Speed::Paused {
                    Speed::Normal
                } else {
                    Speed::Paused
                };
            }
            KeyCode::Char('1') => self.speed = Speed::Normal,
            KeyCode::Char('8') => self.speed = Speed::Fast,
            KeyCode::Char('f') | KeyCode::Char('F') => self.fast_forward(),
            _ => {}
        }
        false
    }

    /// A map click moves the focus to the clicked cell and selects an agent
    /// standing there — the "click-through" half of agent selection.
    fn on_mouse(&mut self, col: u16, row: u16, kind: MouseEventKind) {
        if kind != MouseEventKind::Down(MouseButton::Left) {
            return;
        }
        let a = self.map_area;
        // Inside the bordered map body: one-cell border inset on each side.
        let x = col as i32 - (a.x as i32 + 1);
        let y = row as i32 - (a.y as i32 + 1);
        if (0..GRID).contains(&x) && (0..GRID).contains(&y) {
            self.focus = (x, y);
            self.director.set_focus(self.focus);
            if let Some(slot) = self.positions.iter().position(|&p| p == (x, y)) {
                self.selected = slot;
            }
        }
    }
}

/// The actor of an event.
fn actor(ev: &Event) -> EntityId {
    match *ev {
        Event::Moved { actor, .. }
        | Event::Interacted { actor, .. }
        | Event::Spoke { actor, .. }
        | Event::Rejected { actor, .. } => actor,
    }
}

/// Whether an event is dramatic enough to pin its actor hot.
fn is_notable(ev: &Event) -> bool {
    matches!(ev, Event::Spoke { .. } | Event::Interacted { .. })
}

fn actor_name(ev: &Event, reg: &SliceRegistry) -> String {
    reg.card(actor(ev)).name.clone()
}

/// A short human description of one kernel event.
fn describe(ev: &Event, reg: &SliceRegistry, vocab: &Vocab) -> String {
    let word = |list: &[String], code: u32, fallback: &str| {
        list.get(code as usize)
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    };
    match *ev {
        Event::Moved { to, .. } => format!("moved to {to:?}"),
        Event::Interacted { target, verb, .. } => {
            let (a, it) = decode(verb);
            let act = a
                .map(|a| format!("{a:?}").to_lowercase())
                .unwrap_or_default();
            let item = it
                .map(|i| format!(" {i:?}").to_lowercase())
                .unwrap_or_default();
            format!("{act}{item} with {}", reg.card(target).name)
        }
        Event::Spoke {
            target, act, topic, ..
        } => format!(
            "said \"{} about {}\" to {}",
            word(&vocab.acts, act, "speak with"),
            word(&vocab.topics, topic, "things"),
            reg.card(target).name,
        ),
        Event::Rejected { reason, .. } => format!("rejected ({reason:?})"),
    }
}

fn ring_color(r: Ring) -> Color {
    match r {
        Ring::Hot => Color::LightRed,
        Ring::Warm => Color::Yellow,
        Ring::Cold => Color::Blue,
    }
}

// --- rendering -----------------------------------------------------------

/// Draw one full frame. Takes `&mut App` only to stash the map rect for mouse
/// hit-testing; it never mutates sim state.
pub fn render(f: &mut Frame, app: &mut App) {
    let root = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(f.area());
    let cols =
        Layout::horizontal([Constraint::Percentage(52), Constraint::Percentage(48)]).split(root[0]);
    let left =
        Layout::vertical([Constraint::Length(GRID as u16 + 2), Constraint::Min(3)]).split(cols[0]);
    let right = Layout::vertical([Constraint::Length(14), Constraint::Min(3)]).split(cols[1]);

    app.map_area = left[0];
    render_map(f, left[0], app);
    render_feed(f, left[1], app);
    render_inspector(f, right[0], app);
    render_dialogue(f, right[1], app);
    render_status(f, root[1], app);

    if let Some(text) = app.popup.clone() {
        let area = centered(f.area(), 60, 14);
        f.render_widget(Clear, area);
        let p = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" fast-forward "),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(p, area);
    }
}

fn render_map(f: &mut Frame, area: Rect, app: &App) {
    // Best (hottest) occupant per cell, so an overlapping cell shows the most
    // relevant agent.
    let mut occ: Vec<Option<usize>> = vec![None; (GRID * GRID) as usize];
    for (slot, &(x, y)) in app.positions.iter().enumerate() {
        if (0..GRID).contains(&x) && (0..GRID).contains(&y) {
            let idx = (y * GRID + x) as usize;
            let better = occ[idx].is_none_or(|o| app.director.ring(slot) > app.director.ring(o));
            if better {
                occ[idx] = Some(slot);
            }
        }
    }

    let mut lines = Vec::with_capacity(GRID as usize);
    for y in 0..GRID {
        let mut spans = Vec::with_capacity(GRID as usize);
        for x in 0..GRID {
            let is_focus = (x, y) == app.focus;
            let span = if let Some(slot) = occ[(y * GRID + x) as usize] {
                let ring = app.director.ring(slot);
                let name = &app.registry.card(app.ids[slot]).name;
                let glyph = name.chars().next().unwrap_or('@');
                let mut style = Style::default()
                    .fg(ring_color(ring))
                    .add_modifier(Modifier::BOLD);
                if slot == app.selected {
                    style = style.add_modifier(Modifier::REVERSED);
                }
                Span::styled(glyph.to_string(), style)
            } else {
                let (ch, color) = tile_glyph(tile_at((x, y)));
                let mut style = Style::default().fg(color);
                if is_focus {
                    style = style.bg(Color::DarkGray);
                }
                Span::styled(ch.to_string(), style)
            };
            spans.push(span);
        }
        lines.push(Line::from(spans));
    }
    let title = format!(" Map 16x16  focus=({},{}) ", app.focus.0, app.focus.1);
    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(p, area);
}

fn tile_glyph(t: Tile) -> (char, Color) {
    match t {
        Tile::Empty => ('·', Color::DarkGray),
        Tile::Home => ('H', Color::Green),
        Tile::Bakery => ('B', Color::LightYellow),
        Tile::Well => ('W', Color::Cyan),
        Tile::Field => ('F', Color::LightGreen),
    }
}

fn render_inspector(f: &mut Frame, area: Rect, app: &App) {
    let s = app.selected;
    let id = app.ids[s];
    let tick = app.world.tick();
    let card = app.registry.card(id);
    let mem = app.soul.memory(s);
    let (h, en, so) = app.pack.needs(id).project(tick);

    let mut lines = vec![
        Line::from(Span::raw(card.summary.clone())),
        Line::from(bar("hunger", h)),
        Line::from(bar("energy", en)),
        Line::from(bar("social", so)),
    ];

    // Top opinions (largest magnitude), name + fixed-point value in units.
    let mut ops: Vec<(EntityId, i32)> = app
        .ids
        .iter()
        .map(|&o| (o, mem.opinion(o)))
        .filter(|&(_, v)| v != 0)
        .collect();
    ops.sort_by_key(|&(_, v)| std::cmp::Reverse(v.abs()));
    lines.push(Line::from(Span::styled(
        "opinions:",
        Style::default().add_modifier(Modifier::UNDERLINED),
    )));
    if ops.is_empty() {
        lines.push(Line::from(Span::raw("  (none yet)")));
    }
    for (o, v) in ops.into_iter().take(3) {
        let f = v as f64 / OPINION_ONE as f64;
        lines.push(Line::from(format!(
            "  {:<10} {:+.2}",
            app.registry.card(o).name,
            f
        )));
    }

    lines.push(Line::from(Span::styled(
        "salient:",
        Style::default().add_modifier(Modifier::UNDERLINED),
    )));
    let salient = mem.salient(tick);
    if salient.is_empty() {
        lines.push(Line::from(Span::raw("  (nothing memorable)")));
    }
    for m in salient.into_iter().take(3) {
        lines.push(Line::from(format!(
            "  {}",
            describe(&m.event, &app.registry, &app.vocab)
        )));
    }

    let last = app
        .world
        .event_log()
        .iter()
        .rev()
        .find(|e| actor(e) == id)
        .map(|e| describe(e, &app.registry, &app.vocab))
        .unwrap_or_else(|| "idle".to_string());
    lines.push(Line::from(format!("last action: {last}")));

    let title = format!(" Agent #{s} {} [{:?}] ", card.name, app.director.ring(s));
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: true });
    f.render_widget(p, area);
}

/// A 10-cell fixed-point need bar.
fn bar(label: &str, val: i32) -> String {
    const W: i32 = 10;
    let filled = (val.max(0) * W / MAX_NEED).clamp(0, W) as usize;
    format!(
        "{label:<6} [{}{}] {val:>4}",
        "#".repeat(filled),
        "-".repeat(W as usize - filled)
    )
}

fn render_feed(f: &mut Frame, area: Rect, app: &App) {
    let cap = area.height.saturating_sub(2) as usize;
    let start = app.feed.len().saturating_sub(cap);
    let items: Vec<ListItem> = app.feed[start..]
        .iter()
        .map(|l| ListItem::new(l.as_str()))
        .collect();
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(" Event feed "));
    f.render_widget(list, area);
}

fn render_dialogue(f: &mut Frame, area: Rect, app: &App) {
    let rows = app.log.rows();
    let items: Vec<ListItem> = rows
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let act = app.vocab.acts.get(c.act as usize).map_or("speak", |s| s);
            let topic = app
                .vocab
                .topics
                .get(c.topic as usize)
                .map_or("things", |s| s);
            let who = format!(
                "{}→{}",
                app.registry.card(c.speaker).name,
                app.registry.card(c.listener).name
            );
            let mut style = Style::default();
            let cached = c.text.as_ref();
            if cached.is_none() {
                style = style.fg(Color::DarkGray);
            }
            if i == app.dlg_sel {
                style = style.add_modifier(Modifier::REVERSED);
            }
            let body = match cached {
                Some(t) => t.clone(),
                None if app.pending.iter().any(|p| p.index == i) => "[rendering...]".to_string(),
                None => format!("[latent] {act} about {topic}"),
            };
            ListItem::new(format!("{who}: {body}")).style(style)
        })
        .collect();
    let title = format!(" Conversations ({}) — ENTER backfills latent ", rows.len());
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(list, area);
}

fn render_status(f: &mut Frame, area: Rect, app: &App) {
    let text = format!(
        " tick={} [{}]  q:quit  arrows:focus  Tab:agent  j/k:dialogue  ENTER:render  Space:pause  1:1x  8:8x  F:fast-forward ",
        app.world.tick(),
        app.speed.label(),
    );
    let p = Paragraph::new(text)
        .style(Style::default().bg(Color::Blue).fg(Color::White))
        .alignment(Alignment::Left);
    f.render_widget(p, area);
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

// --- entry points --------------------------------------------------------

/// Run the interactive TUI against a real terminal.
pub fn run(cfg: ViewConfig) -> io::Result<()> {
    let mut app = App::new(cfg);
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    let res = event_loop(&mut term, &mut app);

    disable_raw_mode()?;
    execute!(
        term.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    term.show_cursor()?;
    res
}

fn event_loop<B: Backend>(term: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    let frame = Duration::from_millis(80);
    loop {
        app.poll_renders();
        term.draw(|f| render(f, app))?;
        // Poll for input up to one frame — the wall-clock pacing seam. It never
        // reaches sim state; it only decides how many ticks to advance.
        let start = Instant::now();
        if event::poll(frame)? {
            match event::read()? {
                CtEvent::Key(k) if k.kind == KeyEventKind::Press => {
                    if app.on_key(k.code) {
                        return Ok(());
                    }
                }
                CtEvent::Mouse(m) => app.on_mouse(m.column, m.row, m.kind),
                _ => {}
            }
        }
        app.poll_renders();
        // Advance the sim by the current speed once per frame budget.
        if start.elapsed() >= frame || !event::poll(Duration::ZERO)? {
            for _ in 0..app.speed.ticks_per_frame() {
                app.step();
            }
        }
    }
}

/// Build the app, render one frame to an in-memory backend, and return the
/// buffer as text — the CI-safe headless path (`--smoke`) and the gate test.
pub fn smoke_buffer(cfg: ViewConfig) -> String {
    let mut app = App::new(cfg);
    // Advance enough for agents to cluster, act, and start conversations so the
    // frame exercises every pane.
    for _ in 0..300 {
        app.step();
    }
    let backend = TestBackend::new(110, 44);
    let mut term = Terminal::new(backend).expect("test backend");
    term.draw(|f| render(f, &mut app)).expect("draw");
    buffer_to_string(term.backend().buffer())
}

fn buffer_to_string(buf: &Buffer) -> String {
    let area = *buf.area();
    let mut s = String::with_capacity((area.width as usize + 1) * area.height as usize);
    for y in 0..area.height {
        for x in 0..area.width {
            s.push_str(buf[(x, y)].symbol());
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowMockTextBackend {
        calls: std::sync::Arc<AtomicUsize>,
    }

    impl RenderBackend for SlowMockTextBackend {
        fn render(&mut self, job: &RenderJob) -> String {
            std::thread::sleep(Duration::from_millis(500));
            self.calls.fetch_add(1, Ordering::Relaxed);
            format!("{} to {}: rendered", job.speaker.name, job.listener.name)
        }
    }

    #[test]
    fn slow_render_does_not_block_frames_and_caches_response() {
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut app = App::new(ViewConfig {
            seed: 7,
            agents: 2,
            live: false,
        });
        let old = std::mem::replace(
            &mut app.renderer,
            RenderWorker::new(SlowMockTextBackend {
                calls: std::sync::Arc::clone(&calls),
            }),
        );
        drop(old);
        let ids = app.ids.clone();
        app.log.ingest(&[Event::Spoke {
            tick: 0,
            actor: ids[0],
            target: ids[1],
            act: 0,
            topic: 0,
        }]);
        assert!(app.request_render(0, RenderSource::Backfill));

        let start = Instant::now();
        let mut frames = 0;
        while start.elapsed() < Duration::from_millis(250) {
            app.poll_renders();
            let backend = TestBackend::new(110, 44);
            let mut term = Terminal::new(backend).expect("test backend");
            term.draw(|f| render(f, &mut app)).expect("draw");
            app.step();
            frames += 1;
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(frames >= 4, "frames={frames}");
        assert!(app.log.rows()[0].text.is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let deadline = Instant::now() + Duration::from_secs(2);
        while app.log.rows()[0].text.is_none() && Instant::now() < deadline {
            app.poll_renders();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(app.log.rows()[0].text.is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(!app.request_render(0, RenderSource::Backfill));
        app.poll_renders();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
