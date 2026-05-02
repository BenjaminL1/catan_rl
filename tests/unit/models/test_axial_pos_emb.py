"""Tests for Phase 2.1 axial positional embedding in TileEncoder."""

from __future__ import annotations

import torch

from catan_rl.models.tile_encoder import TileEncoder, _hex_axial_coords


def test_axial_coords_lookup_in_valid_range() -> None:
    """All 19 tiles must land in the [0..4] square after the +2 shift."""
    coords = _hex_axial_coords()
    assert coords.shape == (19, 2)
    assert coords.min() >= 0
    assert coords.max() <= 4


def test_axial_coords_unique_per_tile() -> None:
    """Every tile has a distinct ``(q, r)`` — the embedding can disambiguate."""
    coords = _hex_axial_coords()
    seen = {tuple(row) for row in coords.tolist()}
    assert len(seen) == 19


def test_axial_pos_emb_shape() -> None:
    """Output shape unchanged when pos-emb is enabled."""
    enc = TileEncoder(
        tile_in_dim=79,
        tile_model_dim=128,
        proj_tile_dim=25,
        use_axial_pos_emb=True,
        axial_pos_dim=24,
    )
    x = torch.randn(4, 19, 79)
    out = enc(x)
    assert out.shape == (4, 19 * 25)


def test_axial_pos_emb_initial_no_op() -> None:
    """At init, pos-emb std=0.02 — the output stays close to the no-pos-emb run.

    This isn't bit-exact (random init differs across runs), but the encoder's
    output magnitude shouldn't explode when we add a small pos-emb.
    """
    torch.manual_seed(0)
    enc_off = TileEncoder(tile_in_dim=79, tile_model_dim=128, proj_tile_dim=25)
    torch.manual_seed(0)
    enc_on = TileEncoder(
        tile_in_dim=79,
        tile_model_dim=128,
        proj_tile_dim=25,
        use_axial_pos_emb=True,
        axial_pos_dim=24,
    )
    x = torch.randn(2, 19, 79)
    out_off = enc_off(x)
    out_on = enc_on(x)
    # Outputs differ (pos-emb is real), but both stay in a sane range.
    assert out_off.abs().max() < 50.0
    assert out_on.abs().max() < 50.0


def test_axial_pos_emb_breaks_permutation_equivariance() -> None:
    """With pos-emb, swapping two tiles' features must change their outputs.

    Without pos-emb, the transformer is permutation-equivariant: swapping
    tiles 0 and 1 in the input would produce the same outputs (just swapped).
    With pos-emb, tile 0's output is now anchored to (q_0, r_0), so swapping
    inputs no longer simply swaps outputs.
    """
    torch.manual_seed(0)
    enc = TileEncoder(
        tile_in_dim=79,
        tile_model_dim=128,
        tile_encoder_num_layers=1,
        proj_tile_dim=25,
        use_axial_pos_emb=True,
        axial_pos_dim=24,
    )
    enc.eval()  # disable dropout if any

    x = torch.randn(1, 19, 79)
    x_swapped = x.clone()
    x_swapped[:, [0, 1]] = x_swapped[:, [1, 0]]

    out = enc(x).reshape(1, 19, 25)
    out_swapped = enc(x_swapped).reshape(1, 19, 25)

    # If equivariant, out[:, [0,1]] would equal out_swapped[:, [1,0]]. We
    # require *some* divergence here, because pos-emb breaks equivariance.
    expected_if_equivariant = out.clone()
    expected_if_equivariant[:, [0, 1]] = expected_if_equivariant[:, [1, 0]]
    diff = (out_swapped - expected_if_equivariant).abs().max().item()
    assert diff > 1e-3, (
        f"Pos-emb didn't break equivariance: max diff between swapped and"
        f" equivariant prediction was only {diff:.4e}"
    )


def test_axial_pos_dim_must_be_even() -> None:
    """``axial_pos_dim`` is split into halves — odd values are rejected."""
    try:
        TileEncoder(tile_in_dim=79, tile_model_dim=128, use_axial_pos_emb=True, axial_pos_dim=25)
    except ValueError as e:
        assert "axial_pos_dim must be even" in str(e)
    else:
        raise AssertionError("Expected ValueError for odd axial_pos_dim")


def test_axial_pos_emb_off_omits_params() -> None:
    """The default (off) build must NOT register q_emb/r_emb buffers."""
    enc = TileEncoder(tile_in_dim=79, tile_model_dim=128)
    assert not hasattr(enc, "q_emb") or enc.q_emb is None
    # When use_axial_pos_emb=False the attributes simply aren't registered;
    # use_axial_pos_emb is the source of truth.
    assert enc.use_axial_pos_emb is False
