//! Entity store with generational ids.
//!
//! A freed slot's generation is bumped so a stale `EntityId` can never silently
//! address a recycled entity. Slots keep their index for the process lifetime,
//! which also gives us a canonical iteration order for hashing.

/// All simulation state currently lives in integers/fixed-point. If a float
/// stat is ever added it MUST be quantized before it reaches the hash path
/// (see [`crate::world::World::state_hash`]).
#[derive(Clone, Debug)]
pub struct Entity {
    /// Grid position — the v0 self-state.
    pub pos: (i32, i32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EntityId {
    index: u32,
    generation: u32,
}

impl EntityId {
    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }
    /// Reconstruct an id from the canonical slot/generation pair in a replay log.
    pub fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

struct Slot {
    generation: u32,
    value: Option<Entity>,
}

pub(crate) struct EntityStore {
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl EntityStore {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub(crate) fn spawn(&mut self, value: Entity) -> EntityId {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            EntityId {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            EntityId {
                index,
                generation: 0,
            }
        }
    }

    pub(crate) fn get(&self, id: EntityId) -> Option<&Entity> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation == id.generation {
            slot.value.as_ref()
        } else {
            None
        }
    }

    pub(crate) fn get_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation == id.generation {
            slot.value.as_mut()
        } else {
            None
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Present entities in slot-index order — the canonical order for hashing.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (EntityId, &Entity)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            slot.value.as_ref().map(|e| {
                (
                    EntityId {
                        index: i as u32,
                        generation: slot.generation,
                    },
                    e,
                )
            })
        })
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.value.is_some())
            .map(|(i, slot)| EntityId {
                index: i as u32,
                generation: slot.generation,
            })
    }
}

/// Needs/stats a scenario pack tracks. The kernel stores the registration so a
/// pack can declare its state surface at world init; v0 does not yet read it.
#[derive(Default, Debug)]
pub struct StatRegistry {
    names: Vec<String>,
}

impl StatRegistry {
    pub fn register(&mut self, name: &str) {
        self.names.push(name.to_owned());
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }
}
