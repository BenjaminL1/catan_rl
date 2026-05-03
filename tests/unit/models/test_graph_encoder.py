"""Tests for Phase 2.3 hex/vertex/edge GNN encoder."""

from __future__ import annotations

import torch

from catan_rl.models.graph_encoder import (
    N_EDGES,
    N_HEXES,
    N_VERTICES,
    GraphEncoder,
    _board_adjacency_tables,
)


def test_adjacency_tables_have_correct_shapes() -> None:
    """Adjacency builder produces the canonical 19/54/72 sizes."""
    adj = _board_adjacency_tables()
    assert adj["hex_to_vertex"].shape == (N_HEXES, 6)
    assert adj["vertex_to_hex"].shape == (N_VERTICES, 3)
    assert adj["edge_to_vertex"].shape == (N_EDGES, 2)
    assert adj["vertex_to_edge"].shape == (N_VERTICES, 3)


def test_each_edge_has_two_real_endpoints() -> None:
    """No -1 sentinel in the edge→vertex table; every edge has 2 endpoints."""
    adj = _board_adjacency_tables()
    e2v = adj["edge_to_vertex"]
    assert (e2v >= 0).all()
    # Endpoints are distinct.
    assert (e2v[:, 0] != e2v[:, 1]).all()


def test_every_hex_has_six_distinct_corners() -> None:
    """All 19 hexes have 6 valid distinct vertex corners."""
    adj = _board_adjacency_tables()
    h2v = adj["hex_to_vertex"]
    assert (h2v >= 0).all()
    for h in range(N_HEXES):
        assert len(set(h2v[h].tolist())) == 6


def test_vertex_hex_consistency() -> None:
    """If hex h has corner v, then v's adjacent_hex_indices contains h."""
    adj = _board_adjacency_tables()
    h2v = adj["hex_to_vertex"]
    v2h = adj["vertex_to_hex"]
    for h in range(N_HEXES):
        for v in h2v[h]:
            adj_hexes = set(v2h[v].tolist()) - {-1}
            assert h in adj_hexes, f"hex {h} corner {v} missing reverse edge"


def test_graph_encoder_output_shape() -> None:
    """Pooled output shape: (B, out_dim)."""
    enc = GraphEncoder(tile_in_dim=79, hidden_dim=32, n_rounds=1, out_dim=24)
    out = enc(torch.randn(4, 19, 79))
    assert out.shape == (4, 24)


def test_graph_encoder_runs_with_two_rounds() -> None:
    """Default n_rounds=2 runs end-to-end on a realistic input."""
    enc = GraphEncoder(tile_in_dim=79, hidden_dim=64, n_rounds=2, out_dim=64)
    out = enc(torch.randn(2, 19, 79))
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_graph_encoder_responds_to_input() -> None:
    """Different tile inputs produce different graph outputs."""
    torch.manual_seed(0)
    enc = GraphEncoder(tile_in_dim=79, hidden_dim=32, n_rounds=2, out_dim=24)
    enc.eval()
    a = enc(torch.randn(1, 19, 79))
    b = enc(torch.randn(1, 19, 79))
    assert (a - b).abs().max().item() > 1e-6


def test_observation_module_with_graph_encoder() -> None:
    """ObservationModule integrates graph encoder when ``use_graph_encoder=True``."""
    from catan_rl.models.observation_module import ObservationModule

    om = ObservationModule(
        use_graph_encoder=True,
        graph_hidden_dim=32,
        graph_n_rounds=1,
        graph_out_dim=24,
    )
    assert om.use_graph_encoder is True
    assert om.graph_out_dim == 24
    # First fusion linear must include 24 extra input features.
    assert om.final_layer.in_features == 19 * 25 + 2 * 128 + 24


def test_observation_module_default_no_graph() -> None:
    """Off by default — back-compat."""
    from catan_rl.models.observation_module import ObservationModule

    om = ObservationModule()
    assert om.use_graph_encoder is False
    assert om.graph_out_dim == 0
    assert not hasattr(om, "graph_encoder") or om.graph_encoder is None
