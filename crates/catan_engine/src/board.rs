//! `BoardStatic` — the 19-tile / 54-vertex / 72-edge 1v1 Colonist
//! board with random resource placement + spiral chip sequence.
//!
//! Static topology (hex axial coords, vertex graph, edge graph, port
//! locations) is precomputed at compile time — every game shares the
//! same graph; only resource types, number tokens, and port-type
//! assignments are randomized.
//!
//! Per the Q2 decision in `docs/plans/rust_engine_migration.md`,
//! Rust edge ordering does NOT replicate Python's
//! `dict.items()` insertion order. A one-shot
//! `scripts/migrate_checkpoint.py` permutation patch is responsible
//! for remapping old checkpoints' `edge` action-head weights.

// PyO3 0.22's `#[pyfunction]` / `#[pymethods]` macros emit `.into()`
// on `PyResult<T>` returns — clippy flags it as a useless conversion
// against the same `PyErr` type. The warning is on macro expansion,
// not the source.
#![allow(clippy::useless_conversion)]

use crate::rng::EngineRng;
use crate::spiral::{build_spiral_path, OUTER_CORNERS, SPIRAL_CHIP_SEQUENCE};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Resource enum + constants
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resource {
    Desert = 0,
    Wood = 1,
    Brick = 2,
    Wheat = 3,
    Ore = 4,
    Sheep = 5,
}

impl Resource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Resource::Desert => "DESERT",
            Resource::Wood => "WOOD",
            Resource::Brick => "BRICK",
            Resource::Wheat => "WHEAT",
            Resource::Ore => "ORE",
            Resource::Sheep => "SHEEP",
        }
    }
}

/// The 19-tile resource distribution for a standard 1v1 Colonist board.
/// Order is fixed; the caller shuffles before placement.
pub fn resource_pool() -> [Resource; 19] {
    [
        Resource::Desert,
        Resource::Ore,
        Resource::Ore,
        Resource::Ore,
        Resource::Brick,
        Resource::Brick,
        Resource::Brick,
        Resource::Wheat,
        Resource::Wheat,
        Resource::Wheat,
        Resource::Wheat,
        Resource::Wood,
        Resource::Wood,
        Resource::Wood,
        Resource::Wood,
        Resource::Sheep,
        Resource::Sheep,
        Resource::Sheep,
        Resource::Sheep,
    ]
}

// ---------------------------------------------------------------------------
// Static hex axial coords (lifted verbatim from board.py:202-225)
// ---------------------------------------------------------------------------

/// Hex axial coordinates indexed by engine `hex_idx ∈ [0, 19)`.
/// `(q, r)`. Hex 0 is the centre.
pub const HEX_AXIAL_COORDS: [(i8, i8); 19] = [
    (0, 0),   // 0 center
    (0, -1),  // 1 inner ring
    (1, -1),  // 2
    (1, 0),   // 3
    (0, 1),   // 4
    (-1, 1),  // 5
    (-1, 0),  // 6
    (0, -2),  // 7 outer ring
    (1, -2),  // 8
    (2, -2),  // 9
    (2, -1),  // 10
    (2, 0),   // 11
    (1, 1),   // 12
    (0, 2),   // 13
    (-1, 2),  // 14
    (-2, 2),  // 15
    (-2, 1),  // 16
    (-2, 0),  // 17
    (-1, -1), // 18
];

// ---------------------------------------------------------------------------
// Static vertex + edge tables
// ---------------------------------------------------------------------------
//
// The 19-hex board has exactly 54 unique vertices and 72 unique edges.
// We compute these once at module load (no Python dependency) by
// walking the hex corners using pointy-top axial geometry:
//
//   For axial (q, r), the 6 corner offsets in axial-fractional units are:
//     corner 0: ( 2/3,    -1/3)  // E
//     corner 1: ( 1/3,    -2/3)  // NE  (wait — let's just use pixel-comparison)
//
// Rather than encoding fractional axial corner positions, we use the
// same pixel-corner-distance technique the Python uses: compute each
// hex's 6 corner pixel positions under a notional flat layout, then
// deduplicate corners that lie within a small epsilon.
//
// This is a one-shot ``OnceLock`` computation; the resulting
// VERTEX_TABLE and EDGE_TABLE are static for the rest of the process.

use std::sync::OnceLock;

/// A single hex's 6 corner pixels in a notional unit-edge-length
/// pointy-top layout.
///
/// Corner index 0 is at +30° from the +x axis; corners proceed
/// **clockwise** (decreasing angle: 30°, -30°, -90°, …), matching
/// Python's ``HexLayout.get_corners`` (geometry.py:94-103). This
/// CW convention is load-bearing: ``PORT_HEX_CORNERS`` is authored
/// against Python's indexing; flipping to CCW would put 6/9 ports
/// on inner-ring–facing edges instead of coast edges.
fn hex_corners_pixels(q: i8, r: i8) -> [(f64, f64); 6] {
    // Pointy-top axial → pixel. Unit edge length.
    let q = q as f64;
    let r = r as f64;
    let cx = 3f64.sqrt() * (q + r / 2.0);
    let cy = 1.5 * r;
    let mut out = [(0.0f64, 0.0f64); 6];
    for (i, item) in out.iter_mut().enumerate() {
        // CW enumeration: 30°, -30°, -90°, -150°, -210°, -270°.
        let angle_deg = 30.0 - 60.0 * i as f64;
        let angle = angle_deg.to_radians();
        *item = (cx + angle.cos(), cy + angle.sin());
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct VertexStatic {
    pub vertex_idx: u8,
    /// 1, 2, or 3 adjacent hexes — padded with sentinel `u8::MAX`.
    pub adjacent_hex_indices: [u8; 3],
    pub adjacent_count: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeStatic {
    pub edge_idx: u8,
    pub v1_idx: u8,
    pub v2_idx: u8,
}

struct GraphTables {
    vertices: Vec<VertexStatic>,
    edges: Vec<EdgeStatic>,
    /// For each (hex_idx, corner_idx), the resolved vertex_idx.
    /// Used by port assignment.
    hex_corner_to_vertex: [[u8; 6]; 19],
}

fn graph_tables() -> &'static GraphTables {
    static TABLES: OnceLock<GraphTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        // Collect every (hex_idx, corner_idx, pixel) tuple, then merge
        // pixels that are within EPS of each other into a single vertex.
        const EPS: f64 = 1e-4;
        struct Candidate {
            pos: (f64, f64),
            members: Vec<(u8, u8)>, // (hex_idx, corner_idx)
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for (hex_idx, &(q, r)) in HEX_AXIAL_COORDS.iter().enumerate() {
            let corners = hex_corners_pixels(q, r);
            for (corner_idx, &pos) in corners.iter().enumerate() {
                let mut merged = false;
                for cand in &mut candidates {
                    if (cand.pos.0 - pos.0).abs() < EPS && (cand.pos.1 - pos.1).abs() < EPS {
                        cand.members.push((hex_idx as u8, corner_idx as u8));
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    candidates.push(Candidate {
                        pos,
                        members: vec![(hex_idx as u8, corner_idx as u8)],
                    });
                }
            }
        }
        // Sort candidates for deterministic vertex_idx assignment.
        // Order: by (hex_idx, corner_idx) of the first occurrence —
        // matches the natural construction order so vertex_idx 0 is
        // the first corner of hex 0, etc.
        candidates.sort_by_key(|c| c.members[0]);

        assert_eq!(candidates.len(), 54, "expected 54 unique vertices");

        let mut vertices = Vec::with_capacity(54);
        let mut hex_corner_to_vertex = [[u8::MAX; 6]; 19];
        for (vertex_idx, cand) in candidates.iter().enumerate() {
            let vidx = vertex_idx as u8;
            let mut adj = [u8::MAX; 3];
            let mut count = 0u8;
            // Deduplicate hex_idx (a corner is shared across multiple
            // hex corner-indices but each hex is unique).
            let mut seen_hexes: Vec<u8> = Vec::new();
            for &(h, c) in &cand.members {
                hex_corner_to_vertex[h as usize][c as usize] = vidx;
                if !seen_hexes.contains(&h) {
                    seen_hexes.push(h);
                }
            }
            seen_hexes.sort();
            for (i, &h) in seen_hexes.iter().enumerate().take(3) {
                adj[i] = h;
                count = (i + 1) as u8;
            }
            vertices.push(VertexStatic {
                vertex_idx: vidx,
                adjacent_hex_indices: adj,
                adjacent_count: count,
            });
        }

        // Build edges: walk hex edges (adjacent corners on each hex)
        // and dedupe by sorted (min_v, max_v) tuple.
        let mut edge_pairs: Vec<(u8, u8)> = Vec::new();
        for hex_corners in hex_corner_to_vertex.iter() {
            for c in 0..6u8 {
                let v1 = hex_corners[c as usize];
                let v2 = hex_corners[((c + 1) % 6) as usize];
                let pair = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                if !edge_pairs.contains(&pair) {
                    edge_pairs.push(pair);
                }
            }
        }
        edge_pairs.sort();
        assert_eq!(edge_pairs.len(), 72, "expected 72 unique edges");

        let edges: Vec<EdgeStatic> = edge_pairs
            .into_iter()
            .enumerate()
            .map(|(edge_idx, (v1, v2))| EdgeStatic {
                edge_idx: edge_idx as u8,
                v1_idx: v1,
                v2_idx: v2,
            })
            .collect();

        GraphTables {
            vertices,
            edges,
            hex_corner_to_vertex,
        }
    })
}

// ---------------------------------------------------------------------------
// Port definitions (board.py:314-324)
// ---------------------------------------------------------------------------

/// Fixed port locations — (hex_idx, corner1, corner2). 9 ports total.
pub const PORT_HEX_CORNERS: [(u8, u8, u8); 9] = [
    (7, 2, 3),
    (8, 1, 2),
    (10, 1, 2),
    (11, 0, 1),
    (12, 5, 0),
    (14, 0, 5),
    (15, 4, 5),
    (16, 3, 4),
    (18, 3, 4),
];

/// Port type pool: 5 specific 2:1 (one per resource) + 4 generic 3:1.
pub const PORT_TYPE_POOL: [(u8, Option<Resource>); 9] = [
    (2, Some(Resource::Wood)),
    (2, Some(Resource::Brick)),
    (2, Some(Resource::Wheat)),
    (2, Some(Resource::Ore)),
    (2, Some(Resource::Sheep)),
    (3, None),
    (3, None),
    (3, None),
    (3, None),
];

#[derive(Debug, Clone, Copy)]
pub struct PortStatic {
    pub port_idx: u8,
    pub vertex_idx_pair: [u8; 2],
    pub ratio: u8, // 2 or 3
    pub resource: Option<Resource>,
}

// ---------------------------------------------------------------------------
// HexStatic — the per-hex randomized state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct HexStatic {
    pub hex_idx: u8,
    pub q: i8,
    pub r: i8,
    pub resource: Resource,
    pub number_token: Option<u8>, // 2..=12; None on desert
    pub has_robber_initial: bool,
}

// ---------------------------------------------------------------------------
// BoardStatic — the full board layout
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct BoardStatic {
    pub hexes: [HexStatic; 19],
    pub ports: [PortStatic; 9],
}

impl BoardStatic {
    /// Generate a random 1v1 Colonist board using the provided RNG
    /// stream. Consumes RNG for:
    /// * Fisher-Yates shuffle of the 19-resource pool.
    /// * Uniform [0, 6) pick for the spiral start corner.
    /// * Uniform {0, 1} for CW/CCW.
    /// * Fisher-Yates shuffle of the 9-port-type pool.
    pub fn new_random(rng: &mut EngineRng) -> Self {
        // 1. Resource shuffle.
        let mut resources = resource_pool();
        rng.shuffle(&mut resources);

        // 2. Spiral orientation.
        let corner_idx = rng.gen_range_u32(OUTER_CORNERS.len() as u32) as usize;
        let start_corner = OUTER_CORNERS[corner_idx];
        let clockwise = rng.gen_range_u32(2) != 0;
        let spiral = build_spiral_path(start_corner, clockwise);

        // 3. Walk spiral; assign chips to non-desert hexes.
        let mut number_tokens: [Option<u8>; 19] = [None; 19];
        let mut chip_idx = 0;
        for &hex_idx in &spiral {
            if resources[hex_idx as usize] == Resource::Desert {
                continue;
            }
            number_tokens[hex_idx as usize] = Some(SPIRAL_CHIP_SEQUENCE[chip_idx]);
            chip_idx += 1;
        }
        debug_assert_eq!(chip_idx, 18, "should assign all 18 chips");

        // 4. Construct hexes.
        let mut hexes = [HexStatic {
            hex_idx: 0,
            q: 0,
            r: 0,
            resource: Resource::Desert,
            number_token: None,
            has_robber_initial: false,
        }; 19];
        for (hex_idx, h) in hexes.iter_mut().enumerate() {
            let (q, r) = HEX_AXIAL_COORDS[hex_idx];
            let resource = resources[hex_idx];
            *h = HexStatic {
                hex_idx: hex_idx as u8,
                q,
                r,
                resource,
                number_token: number_tokens[hex_idx],
                has_robber_initial: resource == Resource::Desert,
            };
        }

        // 5. Ports: shuffle the type pool, assign to fixed locations.
        let mut port_types = PORT_TYPE_POOL;
        rng.shuffle(&mut port_types);
        let tables = graph_tables();
        let mut ports = [PortStatic {
            port_idx: 0,
            vertex_idx_pair: [0, 0],
            ratio: 3,
            resource: None,
        }; 9];
        for (port_idx, ((hex_idx, c1, c2), (ratio, resource))) in
            PORT_HEX_CORNERS.iter().zip(port_types.iter()).enumerate()
        {
            let v1 = tables.hex_corner_to_vertex[*hex_idx as usize][*c1 as usize];
            let v2 = tables.hex_corner_to_vertex[*hex_idx as usize][*c2 as usize];
            ports[port_idx] = PortStatic {
                port_idx: port_idx as u8,
                vertex_idx_pair: [v1, v2],
                ratio: *ratio,
                resource: *resource,
            };
        }

        BoardStatic { hexes, ports }
    }

    /// Borrow the static vertex table — same instance per process.
    pub fn vertices(&self) -> &'static [VertexStatic] {
        &graph_tables().vertices
    }

    /// Borrow the static edge table.
    pub fn edges(&self) -> &'static [EdgeStatic] {
        &graph_tables().edges
    }

    /// (hex_idx, corner_idx) → vertex_idx lookup. Static table.
    pub fn hex_corner_to_vertex(&self, hex_idx: u8, corner_idx: u8) -> u8 {
        graph_tables().hex_corner_to_vertex[hex_idx as usize][corner_idx as usize]
    }
}

// ---------------------------------------------------------------------------
// PyO3 exposure: PyBoardStatic.board_static() returns a JSON-safe dict
// matching the shape produced by Python's catanBoard.board_static().
// ---------------------------------------------------------------------------

#[pyclass(name = "BoardStatic", module = "catan_engine")]
pub(crate) struct PyBoardStatic {
    inner: BoardStatic,
}

#[pymethods]
impl PyBoardStatic {
    /// Construct a random board from a u64 seed.
    #[new]
    fn py_new(seed: u64) -> Self {
        let mut rng = EngineRng::from_u64(seed);
        Self {
            inner: BoardStatic::new_random(&mut rng),
        }
    }

    /// Return the board_static dict matching Python's
    /// `catanBoard.board_static()` shape (board.py:590-704):
    ///
    /// ```python
    /// {
    ///   "hexes":    [{"hex_idx", "q", "r", "resource",
    ///                 "number_token", "has_robber_initial"}],
    ///   "vertices": [{"vertex_idx", "adjacent_hex_indices"}],
    ///   "edges":    [{"edge_idx", "v1_idx", "v2_idx"}],
    ///   "ports":    [{"port_idx", "vertex_idx_pair",
    ///                 "ratio", "resource"}],
    /// }
    /// ```
    fn board_static<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);

        // Hexes.
        let hexes_list = pyo3::types::PyList::empty_bound(py);
        for h in &self.inner.hexes {
            let entry = PyDict::new_bound(py);
            entry.set_item("hex_idx", h.hex_idx)?;
            entry.set_item("q", h.q)?;
            entry.set_item("r", h.r)?;
            entry.set_item("resource", h.resource.as_str())?;
            match h.number_token {
                Some(n) => entry.set_item("number_token", n)?,
                None => entry.set_item("number_token", py.None())?,
            }
            entry.set_item("has_robber_initial", h.has_robber_initial)?;
            hexes_list.append(entry)?;
        }
        out.set_item("hexes", hexes_list)?;

        // Vertices.
        let vertices_list = pyo3::types::PyList::empty_bound(py);
        for v in self.inner.vertices() {
            let entry = PyDict::new_bound(py);
            entry.set_item("vertex_idx", v.vertex_idx)?;
            let adj: Vec<u8> = v.adjacent_hex_indices[..v.adjacent_count as usize].to_vec();
            entry.set_item("adjacent_hex_indices", adj)?;
            vertices_list.append(entry)?;
        }
        out.set_item("vertices", vertices_list)?;

        // Edges.
        let edges_list = pyo3::types::PyList::empty_bound(py);
        for e in self.inner.edges() {
            let entry = PyDict::new_bound(py);
            entry.set_item("edge_idx", e.edge_idx)?;
            entry.set_item("v1_idx", e.v1_idx)?;
            entry.set_item("v2_idx", e.v2_idx)?;
            edges_list.append(entry)?;
        }
        out.set_item("edges", edges_list)?;

        // Ports.
        let ports_list = pyo3::types::PyList::empty_bound(py);
        for p in &self.inner.ports {
            let entry = PyDict::new_bound(py);
            entry.set_item("port_idx", p.port_idx)?;
            entry.set_item(
                "vertex_idx_pair",
                [p.vertex_idx_pair[0], p.vertex_idx_pair[1]],
            )?;
            entry.set_item("ratio", if p.ratio == 2 { "2:1" } else { "3:1" })?;
            match p.resource {
                Some(r) => entry.set_item("resource", r.as_str())?,
                None => entry.set_item("resource", py.None())?,
            }
            ports_list.append(entry)?;
        }
        out.set_item("ports", ports_list)?;

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Internal cargo tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_topology_is_54_vertices_72_edges() {
        let tables = graph_tables();
        assert_eq!(tables.vertices.len(), 54);
        assert_eq!(tables.edges.len(), 72);
    }

    #[test]
    fn every_hex_corner_resolves_to_a_vertex() {
        let tables = graph_tables();
        for h in 0..19 {
            for c in 0..6 {
                assert!(
                    tables.hex_corner_to_vertex[h][c] != u8::MAX,
                    "hex {h} corner {c} did not resolve"
                );
            }
        }
    }

    #[test]
    fn vertex_adjacency_counts_are_in_valid_range() {
        let tables = graph_tables();
        let mut counts = [0usize; 4];
        for v in &tables.vertices {
            assert!(
                v.adjacent_count >= 1 && v.adjacent_count <= 3,
                "vertex {} has adjacent_count={} out of [1, 3]",
                v.vertex_idx,
                v.adjacent_count
            );
            counts[v.adjacent_count as usize] += 1;
        }
        // Total must be 54 (the 19-hex Catan board has exactly 54
        // vertices). Per-count distribution depends on the exact
        // geometry but is invariant across boards. We pin the
        // distribution as a regression check; if R3+ changes the
        // hex layout, update this.
        assert_eq!(counts[1] + counts[2] + counts[3], 54);
        // Empirical breakdown for the standard 19-hex pointy-top
        // layout under our axial-coord mapping:
        // 18 1-hex tip corners, 12 2-hex edge corners, 24 3-hex
        // interior corners. Pinned to catch geometry regressions.
        assert_eq!(counts[1], 18, "1-hex tip vertices");
        assert_eq!(counts[2], 12, "2-hex edge vertices");
        assert_eq!(counts[3], 24, "3-hex interior vertices");
    }

    #[test]
    fn random_board_has_correct_resource_counts() {
        let mut rng = EngineRng::from_u64(42);
        let board = BoardStatic::new_random(&mut rng);
        let mut counts: [u8; 6] = [0; 6];
        for h in &board.hexes {
            counts[h.resource as usize] += 1;
        }
        assert_eq!(counts[Resource::Desert as usize], 1);
        assert_eq!(counts[Resource::Ore as usize], 3);
        assert_eq!(counts[Resource::Brick as usize], 3);
        assert_eq!(counts[Resource::Wheat as usize], 4);
        assert_eq!(counts[Resource::Wood as usize], 4);
        assert_eq!(counts[Resource::Sheep as usize], 4);
    }

    #[test]
    fn random_board_has_correct_chip_distribution() {
        let mut rng = EngineRng::from_u64(7);
        let board = BoardStatic::new_random(&mut rng);
        // 18 non-desert hexes have number tokens; desert has None.
        let with_token = board
            .hexes
            .iter()
            .filter(|h| h.number_token.is_some())
            .count();
        let desert = board
            .hexes
            .iter()
            .filter(|h| h.resource == Resource::Desert)
            .count();
        assert_eq!(desert, 1);
        assert_eq!(with_token, 18);

        // Token multiset must match SPIRAL_CHIP_SEQUENCE.
        let mut bag: Vec<u8> = board.hexes.iter().filter_map(|h| h.number_token).collect();
        bag.sort();
        let mut expected: Vec<u8> = SPIRAL_CHIP_SEQUENCE.to_vec();
        expected.sort();
        assert_eq!(bag, expected);
    }

    #[test]
    fn random_board_has_correct_port_distribution() {
        let mut rng = EngineRng::from_u64(99);
        let board = BoardStatic::new_random(&mut rng);
        let mut ratios: [u8; 4] = [0; 4];
        for p in &board.ports {
            ratios[p.ratio as usize] += 1;
        }
        assert_eq!(ratios[2], 5, "expected 5 2:1 specific ports");
        assert_eq!(ratios[3], 4, "expected 4 3:1 generic ports");
    }

    #[test]
    fn board_deterministic_under_same_seed() {
        let mut rng_a = EngineRng::from_u64(123);
        let mut rng_b = EngineRng::from_u64(123);
        let a = BoardStatic::new_random(&mut rng_a);
        let b = BoardStatic::new_random(&mut rng_b);
        for i in 0..19 {
            assert_eq!(a.hexes[i].resource as u8, b.hexes[i].resource as u8);
            assert_eq!(a.hexes[i].number_token, b.hexes[i].number_token);
        }
        for i in 0..9 {
            assert_eq!(a.ports[i].vertex_idx_pair, b.ports[i].vertex_idx_pair);
            assert_eq!(a.ports[i].ratio, b.ports[i].ratio);
            assert_eq!(
                a.ports[i].resource.map(|r| r as u8),
                b.ports[i].resource.map(|r| r as u8)
            );
        }
    }

    #[test]
    fn ports_sit_on_coast_edges_of_their_named_rim_hex() {
        // R2 reviewer BLOCK fix: ensure every port's vertex pair
        // touches ONLY the rim hex named in PORT_HEX_CORNERS. If a
        // port landed on an inner-ring–facing edge, one of the
        // vertices would have multiple adjacent hexes including a
        // non-rim hex.
        let mut rng = EngineRng::from_u64(42);
        let board = BoardStatic::new_random(&mut rng);
        let tables = graph_tables();
        for (port_idx, (expected_rim_hex, _c1, _c2)) in PORT_HEX_CORNERS.iter().enumerate() {
            let port = &board.ports[port_idx];
            let v1 = &tables.vertices[port.vertex_idx_pair[0] as usize];
            let v2 = &tables.vertices[port.vertex_idx_pair[1] as usize];
            // Both vertices must list the rim hex among their
            // adjacent hexes.
            let v1_hexes = &v1.adjacent_hex_indices[..v1.adjacent_count as usize];
            let v2_hexes = &v2.adjacent_hex_indices[..v2.adjacent_count as usize];
            assert!(
                v1_hexes.contains(expected_rim_hex),
                "port {port_idx} v1={} touches {:?}, expected to include rim hex {}",
                v1.vertex_idx,
                v1_hexes,
                expected_rim_hex
            );
            assert!(
                v2_hexes.contains(expected_rim_hex),
                "port {port_idx} v2={} touches {:?}, expected to include rim hex {}",
                v2.vertex_idx,
                v2_hexes,
                expected_rim_hex
            );
            // The shared hex between v1 and v2 must be ONLY the rim hex
            // (anything else means the port is on an inner-facing edge).
            let shared: Vec<u8> = v1_hexes
                .iter()
                .filter(|h| v2_hexes.contains(*h))
                .copied()
                .collect();
            assert_eq!(
                shared,
                vec![*expected_rim_hex],
                "port {port_idx} should sit on a coast edge of hex {} only; \
                 shared hexes between its two vertices = {:?}",
                expected_rim_hex,
                shared
            );
        }
    }

    #[test]
    fn topology_index_pin_for_centre_and_rim() {
        // Pin the corner→vertex mapping for the centre hex (0) and
        // one rim hex (7) so a future angle-convention change can't
        // silently rotate the table without flagging this test.
        let tables = graph_tables();
        let centre = tables.hex_corner_to_vertex[0];
        let rim7 = tables.hex_corner_to_vertex[7];
        // Centre hex's 6 corners must all resolve to 3-hex interior
        // vertices (every centre corner is shared with two ring hexes).
        for &v_idx in &centre {
            assert!(v_idx != u8::MAX);
            assert_eq!(
                tables.vertices[v_idx as usize].adjacent_count, 3,
                "centre corner {v_idx} should be 3-hex interior"
            );
        }
        // Rim hex 7 (axial (0, -2), top of outer ring) has 2 corners
        // that are 1-hex tips, 2 that are 2-hex coast edges, 2 that
        // are 3-hex interior touching adjacent rim+inner hexes.
        let mut counts = [0u8; 4];
        for &v_idx in &rim7 {
            counts[tables.vertices[v_idx as usize].adjacent_count as usize] += 1;
        }
        // Standard 19-hex Catan rim-corner-hex topology.
        assert_eq!(counts[1], 2, "rim hex 7 should have 2 tip corners");
        assert_eq!(counts[2], 2, "rim hex 7 should have 2 coast corners");
        assert_eq!(counts[3], 2, "rim hex 7 should have 2 interior corners");
    }

    #[test]
    fn robber_starts_on_desert() {
        let mut rng = EngineRng::from_u64(42);
        let board = BoardStatic::new_random(&mut rng);
        let robber_hexes: Vec<u8> = board
            .hexes
            .iter()
            .filter(|h| h.has_robber_initial)
            .map(|h| h.hex_idx)
            .collect();
        assert_eq!(robber_hexes.len(), 1);
        let robber_hex = &board.hexes[robber_hexes[0] as usize];
        assert_eq!(robber_hex.resource, Resource::Desert);
    }
}
