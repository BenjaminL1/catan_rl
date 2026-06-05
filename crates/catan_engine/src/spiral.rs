//! Spiral chip placement — the official ABC chip sequence Colonist.io
//! uses for 1v1 ranked boards (empirically verified 2026-06-02
//! against 5 ranked boards). The 18-element sequence guarantees no
//! 6/8 adjacency, no identical-number adjacency, no 2/12 adjacency
//! without needing rejection sampling.

/// The 18-chip ABC sequence (board.py:36-54).
pub const SPIRAL_CHIP_SEQUENCE: [u8; 18] =
    [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];

/// Outer ring in CW pixel order starting from top.
pub const OUTER_RING_CW: [u8; 12] = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

/// Inner ring in CW pixel order starting top-left.
pub const INNER_RING_CW: [u8; 6] = [1, 2, 3, 4, 5, 6];

/// The 6 outer hexes that sit at hexagonal-board corners.
pub const OUTER_CORNERS: [u8; 6] = [7, 9, 11, 13, 15, 17];

/// Build a 19-hex spiral traversal starting at `start_corner` and
/// going CW or CCW. Returns hex indices in order: 12 outer ring →
/// 6 inner ring → 0 (centre).
///
/// Mirrors `board.py:_build_spiral_path`.
pub fn build_spiral_path(start_corner: u8, clockwise: bool) -> [u8; 19] {
    let target_idx = OUTER_RING_CW
        .iter()
        .position(|&h| h == start_corner)
        .expect("start_corner must be in OUTER_RING_CW");
    let rotation = target_idx / 2;

    let mut outer: Vec<u8> = OUTER_RING_CW.to_vec();
    outer.rotate_left(target_idx);
    let mut inner: Vec<u8> = INNER_RING_CW.to_vec();
    inner.rotate_left(rotation);

    if !clockwise {
        // Mirror: keep the start hex first, reverse the rest of the
        // ring. Python: `[outer[0]] + outer[:0:-1]`.
        let head = outer[0];
        let tail: Vec<u8> = outer[1..].iter().rev().copied().collect();
        outer = vec![head];
        outer.extend(tail);
        let head = inner[0];
        let tail: Vec<u8> = inner[1..].iter().rev().copied().collect();
        inner = vec![head];
        inner.extend(tail);
    }

    let mut out = [0u8; 19];
    for (i, &h) in outer.iter().enumerate() {
        out[i] = h;
    }
    for (i, &h) in inner.iter().enumerate() {
        out[12 + i] = h;
    }
    out[18] = 0;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_has_19_unique_hexes() {
        let path = build_spiral_path(7, true);
        let mut sorted = path;
        sorted.sort();
        for i in 0..19 {
            assert_eq!(sorted[i], i as u8);
        }
    }

    #[test]
    fn spiral_centre_is_last() {
        let path = build_spiral_path(11, false);
        assert_eq!(path[18], 0);
    }

    #[test]
    fn spiral_first_is_start_corner() {
        for &start in &OUTER_CORNERS {
            let path = build_spiral_path(start, true);
            assert_eq!(path[0], start);
            let path = build_spiral_path(start, false);
            assert_eq!(path[0], start);
        }
    }
}
