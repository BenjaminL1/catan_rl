"""Tests for Phase 2.2 transformer recipe (pre-norm + GELU + dropout)."""

from __future__ import annotations

import torch
import torch.nn as nn

from catan_rl.models.observation_module import ObservationModule
from catan_rl.models.tile_encoder import TileEncoder


def test_transformer_default_activation_is_relu() -> None:
    """Legacy default — preserves Phase 0/1 checkpoint compatibility."""
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128)
    layer = enc.encoder.layers[0]
    # PyTorch's TransformerEncoderLayer stores activation as a callable on
    # ``self.activation``. ReLU vs GELU is the simplest signal.
    assert isinstance(layer.activation, type(torch.nn.functional.relu)) or (
        layer.activation is torch.nn.functional.relu
    )


def test_transformer_gelu_activation() -> None:
    """``activation='gelu'`` swaps the FFN nonlinearity to GELU."""
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128, activation="gelu")
    layer = enc.encoder.layers[0]
    assert layer.activation is torch.nn.functional.gelu


def test_transformer_pre_norm() -> None:
    """All Phase 2 paths use pre-norm (norm before attn/FFN). Already on by default."""
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128)
    layer = enc.encoder.layers[0]
    assert layer.norm_first is True


def _dropout_p(layer: nn.Module) -> float:
    """Pull the dropout probability off a TransformerEncoderLayer.

    PyTorch exposes the dropout submodule as ``layer.dropout`` (and
    ``dropout1``/``dropout2`` for the residual paths). Each is an
    ``nn.Dropout`` whose ``.p`` is the probability we plumbed through.
    """
    return float(layer.dropout1.p)


def test_transformer_dropout_propagates() -> None:
    """``dropout=0.05`` must reach the encoder layer's dropout modules."""
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128, dropout=0.05)
    layer = enc.encoder.layers[0]
    assert abs(_dropout_p(layer) - 0.05) < 1e-9


def test_observation_module_transformer_dropout_override() -> None:
    """When ``transformer_dropout`` is set explicitly, it overrides ``dropout``."""
    om = ObservationModule(
        tile_in_dim=79,
        tile_model_dim=128,
        dropout=0.1,
        transformer_dropout=0.0,
    )
    layer = om.tile_encoder.encoder.layers[0]
    assert abs(_dropout_p(layer) - 0.0) < 1e-9


def test_observation_module_transformer_dropout_inherits() -> None:
    """When ``transformer_dropout=None``, the encoder inherits the global ``dropout``."""
    om = ObservationModule(
        tile_in_dim=79,
        tile_model_dim=128,
        dropout=0.05,
        transformer_dropout=None,
    )
    layer = om.tile_encoder.encoder.layers[0]
    assert abs(_dropout_p(layer) - 0.05) < 1e-9


def test_dropout_makes_train_eval_differ() -> None:
    """Sanity: with dropout > 0, training and eval modes produce different outputs."""
    torch.manual_seed(0)
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128, dropout=0.5)
    x = torch.randn(2, 19, 79)
    enc.train()
    out_train = enc(x)
    enc.eval()
    out_eval = enc(x)
    assert not torch.allclose(out_train, out_eval), (
        "With dropout=0.5, train and eval outputs should differ"
    )
