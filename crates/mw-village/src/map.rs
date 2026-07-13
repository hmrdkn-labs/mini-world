//! The 16x16 village map: a static fixture layout plus grid geometry helpers.
//!
//! The layout is pattern-defined rather than stored, so it needs no allocation
//! and is trivially identical across machines. Per-tile *mutable* state (dropped
//! items) lives in the pack, not here.

/// Grid side length. The world is `GRID x GRID` cells, `[0, GRID)` on each axis.
pub const GRID: i32 = 16;

/// Total tile count — sizes the ground-item array.
pub const TILES: usize = (GRID * GRID) as usize;

/// A map fixture. Empty tiles afford only movement; the rest gate location
/// actions (sleep at home, work/pickup at a workplace).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tile {
    Empty,
    Home,
    Bakery,
    Well,
    Field,
}

/// Fixture at `pos`. Fixed layout: a row of homes along the north edge, a
/// bakery and a well as single tiles, and a field block in the south-east.
pub fn tile_at(pos: (i32, i32)) -> Tile {
    let (x, y) = pos;
    if y == 0 && (0..5).contains(&x) {
        Tile::Home
    } else if pos == (8, 8) {
        Tile::Bakery
    } else if pos == (4, 12) {
        Tile::Well
    } else if x >= 11 && y >= 11 {
        Tile::Field
    } else {
        Tile::Empty
    }
}

/// A workplace is anywhere `work` (and its `pickup`/product) is afforded.
pub fn is_workplace(t: Tile) -> bool {
    matches!(t, Tile::Bakery | Tile::Well | Tile::Field)
}

pub fn in_bounds(pos: (i32, i32)) -> bool {
    (0..GRID).contains(&pos.0) && (0..GRID).contains(&pos.1)
}

/// Row-major tile index. Caller guarantees `in_bounds`.
pub fn index(pos: (i32, i32)) -> usize {
    (pos.1 * GRID + pos.0) as usize
}

/// Chebyshev distance — the interaction range metric, so the eight cells
/// touching an actor (including diagonals) count as adjacent.
pub fn chebyshev(a: (i32, i32), b: (i32, i32)) -> i32 {
    (a.0 - b.0).abs().max((a.1 - b.1).abs())
}

pub fn adjacent(a: (i32, i32), b: (i32, i32)) -> bool {
    chebyshev(a, b) <= 1
}
