//! Village scenario pack.
//!
//! A social-sim [`ScenarioPack`] on the mini-world kernel: a 16x16 map with
//! homes/bakery/well/field, a hunger/energy/social needs system with
//! closed-form decay and starvation death, an inventory + ground-item economy,
//! and affordance-masked tools whose legality is enforced in [`validate`].
//!
//! State ownership follows the kernel's split: **positions live in the World**
//! (only `Move` mutates them), and everything else — needs, inventories,
//! dropped items — lives in this pack behind a `RefCell`, keyed by entity slot
//! index. All of it is integer-only and driven solely by the tick pipeline, so
//! determinism and replay hold (see DESIGN.md, load-bearing decision 1).
//!
//! [`validate`]: ScenarioPack::validate

mod action;
mod map;
mod needs;

use std::cell::RefCell;

use mw_core::{
    ActionManifest, EntityId, FnvHasher, Intent, Observation, RejectReason, ScenarioPack,
    StatRegistry, World,
};

pub use action::{decode, verb, Action, Item, ITEM_COUNT, TOOL_COUNT};
pub use map::{adjacent, tile_at, Tile, GRID};
pub use needs::{
    Needs, EAT_GAIN, ENERGY_DECAY, HUNGER_DECAY, MAINTENANCE_CYCLE, MAINTENANCE_GAIN, MAX_NEED,
    NEED_DECAY, SLEEP_GAIN, SOCIAL_DECAY, SPEAK_GAIN, STARVE_TICKS,
};

/// Per-pack mutable state. Entity-indexed vectors grow lazily; the world never
/// tells the pack about spawns, but slot indices are stable so first touch of
/// an entity is enough to allocate its state.
struct VillageState {
    needs: Vec<Needs>,
    inv: Vec<[u8; ITEM_COUNT]>,
    /// Dropped items per tile, row-major (`map::index`).
    ground: Vec<[u8; ITEM_COUNT]>,
}

impl VillageState {
    fn new() -> Self {
        Self {
            needs: Vec::new(),
            inv: Vec::new(),
            ground: vec![[0; ITEM_COUNT]; map::TILES],
        }
    }

    fn ensure(&mut self, idx: usize) {
        while self.needs.len() <= idx {
            self.needs.push(Needs::full());
            self.inv.push([0; ITEM_COUNT]);
        }
    }
}

pub struct VillagePack {
    manifest: ActionManifest,
    state: RefCell<VillageState>,
}

impl VillagePack {
    pub fn new() -> Self {
        Self {
            manifest: action::manifest(),
            state: RefCell::new(VillageState::new()),
        }
    }

    /// Stored needs for an entity; project it onto a tick for current values.
    /// For tests, UI, and the observation encoder.
    pub fn needs(&self, entity: EntityId) -> Needs {
        let mut st = self.state.borrow_mut();
        st.ensure(entity.index() as usize);
        st.needs[entity.index() as usize]
    }
    /// Seed deterministic initial needs for scenario stress-start collection.
    pub fn seed_needs(&self, entity: EntityId, values: [i32; 3]) {
        let mut st = self.state.borrow_mut();
        st.ensure(entity.index() as usize);
        let n = &mut st.needs[entity.index() as usize];
        *n = Needs::full();
        n.adjust(
            0,
            values[0] - MAX_NEED,
            values[1] - MAX_NEED,
            values[2] - MAX_NEED,
        );
    }

    /// Inventory count of `item` for an entity.
    pub fn inventory(&self, entity: EntityId, item: Item) -> u8 {
        let mut st = self.state.borrow_mut();
        st.ensure(entity.index() as usize);
        st.inv[entity.index() as usize][item as usize]
    }

    /// Dropped-item counts on the tile at `pos` (`[0; _]` off-map) — lets a
    /// brain pick a valid `pickup` item instead of guessing.
    pub fn ground_at(&self, pos: (i32, i32)) -> [u8; ITEM_COUNT] {
        if !map::in_bounds(pos) {
            return [0; ITEM_COUNT];
        }
        self.state.borrow().ground[map::index(pos)]
    }

    /// Whether an entity has starved to death by the world's tick.
    pub fn is_dead(&self, world: &World, entity: EntityId) -> bool {
        self.needs(entity).dead(world.tick())
    }

    /// Bitmask of tools the body currently affords for `entity` (bit `i` = the
    /// [`Action`] with id `i`). This is the mask that belongs in
    /// [`mw_core::Observation::tool_mask`]; the kernel's placeholder mask is
    /// overridden here once a pack is installed. `validate` rejects exactly the
    /// intents this mask omits, so the two stay in lockstep. Neighbor proximity
    /// is read from the caller's `obs`, so no second K-nearest scan is done.
    pub fn afforded_tools(&self, world: &World, entity: EntityId, obs: &Observation) -> u32 {
        let Some(e) = world.entity(entity) else {
            return 0;
        };
        let tick = world.tick();
        let pos = e.pos;

        let mut st = self.state.borrow_mut();
        st.ensure(entity.index() as usize);
        let needs = st.needs[entity.index() as usize];
        if needs.dead(tick) {
            return 0; // the dead afford nothing.
        }
        let inv = st.inv[entity.index() as usize];
        let ground = st.ground[map::index(pos)];
        drop(st);

        let tile = tile_at(pos);
        let has_item = inv.iter().any(|&c| c > 0);
        let ground_item = ground.iter().any(|&c| c > 0);

        // Neighbor proximity from the single observation the kernel already built.
        let mut any_neighbor = false;
        let mut adjacent_neighbor = false;
        for slot in obs.neighbors.iter().filter(|s| s.present) {
            any_neighbor = true;
            if slot.dx.abs() <= 1 && slot.dy.abs() <= 1 {
                adjacent_neighbor = true;
            }
        }

        let mut mask = 0u32;
        let mut set = |a: Action| mask |= 1 << a.id();

        set(Action::Move);
        set(Action::Idle);
        if inv[Item::Food as usize] > 0 || tile == Tile::Bakery {
            set(Action::Eat);
        }
        if tile == Tile::Home {
            set(Action::Sleep);
        }
        if map::is_workplace(tile) && needs.energy(tick) > 0 {
            set(Action::Work);
        }
        if adjacent_neighbor {
            set(Action::Speak);
            if has_item {
                set(Action::Give);
            }
        }
        if ground_item {
            set(Action::Pickup);
        }
        if has_item {
            set(Action::Drop);
        }
        if inv[Item::Water as usize] > 0 {
            set(Action::Use);
        }
        if any_neighbor {
            set(Action::Follow);
            set(Action::Flee);
        }
        mask
    }

    /// The workplace product for a tile (what `work`/`pickup` yields there).
    fn product(tile: Tile) -> Option<Item> {
        match tile {
            Tile::Bakery | Tile::Field => Some(Item::Food),
            Tile::Well => Some(Item::Water),
            _ => None,
        }
    }
}

impl Default for VillagePack {
    fn default() -> Self {
        Self::new()
    }
}

impl ScenarioPack for VillagePack {
    fn manifest(&self) -> &ActionManifest {
        &self.manifest
    }

    fn validate(
        &self,
        world: &World,
        actor: EntityId,
        intent: &Intent,
    ) -> Result<(), RejectReason> {
        let pos = world.entity(actor).ok_or(RejectReason::InvalidTarget)?.pos;
        let tick = world.tick();

        let mut st = self.state.borrow_mut();
        st.ensure(actor.index() as usize);
        let needs = st.needs[actor.index() as usize];
        if needs.dead(tick) {
            return Err(RejectReason::NotAfforded); // the dead cannot act.
        }
        let inv = st.inv[actor.index() as usize];
        let ground = st.ground[map::index(pos)];
        drop(st);

        match *intent {
            // Single-step magnitude is a kernel base rule; the scenario adds
            // map bounds.
            Intent::Move { dx, dy } => {
                if map::in_bounds((pos.0 + dx, pos.1 + dy)) {
                    Ok(())
                } else {
                    Err(RejectReason::OutOfRange)
                }
            }
            // Target existence is a kernel base rule; the scenario adds range.
            Intent::Speak { target, .. } => self.require_adjacent(world, actor, pos, target),
            Intent::Interact { target, verb } => {
                self.validate_interact(world, actor, pos, inv, ground, target, verb)
            }
            Intent::Idle => Ok(()),
        }
    }

    fn apply(&self, world: &mut World, actor: EntityId, intent: &Intent) {
        // Only reached for validated intents; effects trust that contract.
        let tick = world.tick();
        let pos = match world.entity(actor) {
            Some(e) => e.pos,
            None => return,
        };
        let ai = actor.index() as usize;

        let mut st = self.state.borrow_mut();
        st.ensure(ai);

        match *intent {
            // Movement: needs simply decay to the current tick.
            Intent::Move { .. } => st.needs[ai].settle(tick),
            Intent::Speak { .. } => {
                st.needs[ai].adjust(tick, 0, 0, needs::SPEAK_GAIN);
            }
            Intent::Idle => st.needs[ai].settle(tick),
            Intent::Interact { target, verb } => {
                let (action, item) = decode(verb);
                let item = item.unwrap_or(Item::Food);
                match action {
                    Some(Action::Eat) => {
                        if tile_at(pos) != Tile::Bakery {
                            st.inv[ai][Item::Food as usize] -= 1; // free at the bakery.
                        }
                        st.needs[ai].adjust(tick, needs::EAT_GAIN, 0, 0);
                    }
                    Some(Action::Sleep) => st.needs[ai].adjust(tick, 0, needs::SLEEP_GAIN, 0),
                    Some(Action::Work) => {
                        st.needs[ai].adjust(tick, 0, -needs::WORK_ENERGY_COST, 0);
                        if let Some(p) = Self::product(tile_at(pos)) {
                            let c = &mut st.inv[ai][p as usize];
                            *c = c.saturating_add(1);
                        }
                    }
                    Some(Action::Give) => {
                        let ti = target.index() as usize;
                        st.needs[ai].settle(tick);
                        st.inv[ai][item as usize] -= 1;
                        st.ensure(ti);
                        let c = &mut st.inv[ti][item as usize];
                        *c = c.saturating_add(1);
                    }
                    Some(Action::Pickup) => {
                        let gi = map::index(pos);
                        st.ground[gi][item as usize] -= 1;
                        let c = &mut st.inv[ai][item as usize];
                        *c = c.saturating_add(1);
                        st.needs[ai].settle(tick);
                    }
                    Some(Action::Drop) => {
                        let gi = map::index(pos);
                        st.inv[ai][item as usize] -= 1;
                        let c = &mut st.ground[gi][item as usize];
                        *c = c.saturating_add(1);
                        st.needs[ai].settle(tick);
                    }
                    Some(Action::Use) => {
                        st.inv[ai][Item::Water as usize] -= 1;
                        st.needs[ai].adjust(tick, 0, needs::USE_ENERGY_GAIN, 0);
                    }
                    // Move/Speak/Follow/Flee/Idle/None never validate as an
                    // Interact; settle and move on if one slips through.
                    _ => st.needs[ai].settle(tick),
                }
            }
        }
    }

    fn register(&self, registry: &mut StatRegistry) {
        registry.register("hunger");
        registry.register("energy");
        registry.register("social");
    }

    // The seam: route the kernel's affordance query to this pack's body-state
    // mask. Delegates to the inherent method so direct callers (tests, UI) keep
    // working unchanged.
    fn afforded_tools(&self, world: &World, entity: EntityId, obs: &Observation) -> u32 {
        VillagePack::afforded_tools(self, world, entity, obs)
    }

    /// Fold every character's needs, inventory, and each cell's ground items into
    /// the canonical hash, in entity-id / cell order (integer only). Including
    /// `Needs::starving_since` means the death clock is part of the hash, so
    /// replay must reproduce it too.
    fn hash_state(&self, h: &mut FnvHasher) {
        let st = self.state.borrow();
        for n in st.needs.iter() {
            n.hash_into(h);
        }
        for inv in st.inv.iter() {
            for &c in inv.iter() {
                h.write_u32(c as u32);
            }
        }
        for cell in st.ground.iter() {
            for &c in cell.iter() {
                h.write_u32(c as u32);
            }
        }
    }

    /// Cold analytic advance: cold agents do not move, so a fast-forward is just
    /// closed-form need decay realized to `to_tick` for every character. Pure in
    /// `(from, to)` and idempotent under replay.
    fn fast_forward(&self, _world: &mut World, _from_tick: u64, to_tick: u64) {
        let mut st = self.state.borrow_mut();
        for n in st.needs.iter_mut() {
            n.settle(to_tick);
        }
    }
}

impl VillagePack {
    fn require_adjacent(
        &self,
        world: &World,
        actor: EntityId,
        pos: (i32, i32),
        target: EntityId,
    ) -> Result<(), RejectReason> {
        if target == actor {
            return Err(RejectReason::InvalidTarget);
        }
        let tpos = world.entity(target).ok_or(RejectReason::InvalidTarget)?.pos;
        if adjacent(pos, tpos) {
            Ok(())
        } else {
            Err(RejectReason::OutOfRange)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_interact(
        &self,
        world: &World,
        actor: EntityId,
        pos: (i32, i32),
        inv: [u8; ITEM_COUNT],
        ground: [u8; ITEM_COUNT],
        target: EntityId,
        verb: u32,
    ) -> Result<(), RejectReason> {
        let (action, item) = decode(verb);
        let Some(action) = action else {
            return Err(RejectReason::UnknownTool);
        };
        let tile = tile_at(pos);
        match action {
            Action::Eat => {
                if inv[Item::Food as usize] > 0 || tile == Tile::Bakery {
                    Ok(())
                } else {
                    Err(RejectReason::Depleted)
                }
            }
            Action::Sleep => {
                if tile == Tile::Home {
                    Ok(())
                } else {
                    Err(RejectReason::NotAfforded)
                }
            }
            Action::Work => {
                if !map::is_workplace(tile) {
                    Err(RejectReason::NotAfforded)
                } else if self.needs(actor).energy(world.tick()) <= 0 {
                    Err(RejectReason::Depleted)
                } else {
                    Ok(())
                }
            }
            Action::Give => {
                // Target validity/range first, then the resource check.
                self.require_adjacent(world, actor, pos, target)?;
                let item = item.ok_or(RejectReason::UnknownTool)?;
                if inv[item as usize] > 0 {
                    Ok(())
                } else {
                    Err(RejectReason::Depleted)
                }
            }
            Action::Pickup => {
                let item = item.ok_or(RejectReason::UnknownTool)?;
                if ground[item as usize] > 0 {
                    Ok(())
                } else {
                    Err(RejectReason::Depleted)
                }
            }
            Action::Drop => {
                let item = item.ok_or(RejectReason::UnknownTool)?;
                if inv[item as usize] > 0 {
                    Ok(())
                } else {
                    Err(RejectReason::Depleted)
                }
            }
            Action::Use => {
                // Only water is a usable consumable in v0.
                if item == Some(Item::Water) && inv[Item::Water as usize] > 0 {
                    Ok(())
                } else {
                    Err(RejectReason::Depleted)
                }
            }
            // These tools are carried by other kernel intents, never Interact.
            Action::Move | Action::Speak | Action::Follow | Action::Flee | Action::Idle => {
                Err(RejectReason::UnknownTool)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mw_core::{ArgKind, World};

    fn afforded(mask: u32, a: Action) -> bool {
        mask & (1 << a.id()) != 0
    }

    /// The affordance mask, built the way the kernel does: one observation, then
    /// the mask off it.
    fn mask_for(pack: &VillagePack, world: &World, e: EntityId) -> u32 {
        pack.afforded_tools(world, e, &world.observe(e))
    }

    #[test]
    fn manifest_lists_twelve_dense_tools() {
        let m = action::manifest();
        assert_eq!(m.tools.len(), TOOL_COUNT as usize);
        for (i, t) in m.tools.iter().enumerate() {
            assert_eq!(t.id, i as u32, "tool ids must be dense 0..12");
        }
        let names: Vec<&str> = m.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "move", "eat", "sleep", "work", "speak", "give", "pickup", "drop", "use", "follow",
                "flee", "idle"
            ]
        );
        // move carries two scalar params; give carries a 2-variant item enum.
        let mv = &m.tools[Action::Move.id() as usize];
        assert!(matches!(mv.args[0].kind, ArgKind::Scalar));
        assert!(matches!(mv.args[1].kind, ArgKind::Scalar));
        let give = &m.tools[Action::Give.id() as usize];
        assert!(matches!(give.args[1].kind, ArgKind::Enum { variants } if variants == 2));
    }

    #[test]
    fn isolated_entity_affords_only_move_and_idle() {
        // Empty tile, no items, no neighbor: every gated tool is masked off.
        let pack = VillagePack::new();
        let mut world = World::with_pack(1, &pack);
        let e = world.spawn((3, 6));
        let mask = mask_for(&pack, &world, e);
        assert_eq!(mask, (1 << Action::Move.id()) | (1 << Action::Idle.id()));
        for a in [
            Action::Eat,
            Action::Sleep,
            Action::Work,
            Action::Speak,
            Action::Give,
            Action::Pickup,
            Action::Use,
            Action::Follow,
            Action::Flee,
        ] {
            assert!(
                !afforded(mask, a),
                "{a:?} must not be afforded when isolated"
            );
        }
    }

    #[test]
    fn location_gates_eat_sleep_and_work() {
        let pack = VillagePack::new();
        let mut world = World::with_pack(1, &pack);
        let baker = world.spawn((8, 8)); // Bakery
        let sleeper = world.spawn((0, 0)); // Home
        let farmer = world.spawn((13, 13)); // Field

        let bm = mask_for(&pack, &world, baker);
        assert!(afforded(bm, Action::Eat)); // free food at the bakery
        assert!(afforded(bm, Action::Work));
        assert!(!afforded(bm, Action::Sleep));

        let sm = mask_for(&pack, &world, sleeper);
        assert!(afforded(sm, Action::Sleep));
        assert!(!afforded(sm, Action::Work));
        assert!(!afforded(sm, Action::Eat));

        let fm = mask_for(&pack, &world, farmer);
        assert!(afforded(fm, Action::Work));
        assert!(!afforded(fm, Action::Eat));
    }

    #[test]
    fn social_tools_gate_on_proximity() {
        let pack = VillagePack::new();
        let mut world = World::with_pack(1, &pack);
        let a = world.spawn((5, 5));
        let _b = world.spawn((6, 5)); // adjacent to a
        let mask = mask_for(&pack, &world, a);
        assert!(afforded(mask, Action::Speak));
        assert!(afforded(mask, Action::Follow));
        assert!(afforded(mask, Action::Flee));
        // No inventory, so give stays masked even with a neighbor in range.
        assert!(!afforded(mask, Action::Give));
    }
}
