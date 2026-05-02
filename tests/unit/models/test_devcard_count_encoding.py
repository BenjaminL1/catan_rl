"""Tests for Phase 1.4 dev-card count encoding."""

from __future__ import annotations

import torch

from catan_rl.models.player_modules import DevCardCountEncoder


def test_count_encoder_output_shape() -> None:
    """Output shape matches ``(batch_size, output_dim)`` for any padded input."""
    enc = DevCardCountEncoder(vocab_excl_pad=5, output_dim=25, hidden_dim=32)
    ids = torch.zeros(8, 16, dtype=torch.long)  # all-pad batch
    out = enc(ids)
    assert out.shape == (8, 25)


def test_count_encoder_invariant_to_order() -> None:
    """The multiset is order-independent: shuffled input → identical output."""
    enc = DevCardCountEncoder()
    ids_a = torch.tensor([[0, 1, 1, 2, 3, 0, 0, 0]], dtype=torch.long)
    ids_b = torch.tensor([[3, 0, 1, 0, 2, 1, 0, 0]], dtype=torch.long)
    out_a = enc(ids_a)
    out_b = enc(ids_b)
    torch.testing.assert_close(out_a, out_b)


def test_count_encoder_drops_padding_column() -> None:
    """Padding (id=0) does not contribute — adding pads to a batch must not change output."""
    enc = DevCardCountEncoder()
    ids_short = torch.tensor([[1, 2, 3, 0, 0, 0]], dtype=torch.long)
    ids_long = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    torch.testing.assert_close(enc(ids_short), enc(ids_long))


def test_count_encoder_distinguishes_distinct_multisets() -> None:
    """Two inputs with different per-type counts produce different outputs."""
    enc = DevCardCountEncoder()
    ids_a = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.long)  # three KNIGHTs
    ids_b = torch.tensor([[2, 2, 2, 0, 0, 0]], dtype=torch.long)  # three VPs
    out_a = enc(ids_a)
    out_b = enc(ids_b)
    assert not torch.allclose(out_a, out_b)


def test_observation_module_count_path_drops_mha_params() -> None:
    """When ``use_devcard_mha=False``, the dev-card embedding + 2 MHAs are not built."""
    from catan_rl.models.observation_module import ObservationModule

    om_legacy = ObservationModule(use_devcard_mha=True)
    om_count = ObservationModule(use_devcard_mha=False)

    # Legacy path holds the embedding + MHAs.
    assert om_legacy.dev_card_embedding is not None
    assert om_legacy.hidden_card_mha is not None
    assert om_legacy.played_card_mha is not None

    # Count path zeroes them out.
    assert om_count.dev_card_embedding is None
    assert om_count.hidden_card_mha is None
    assert om_count.played_card_mha is None

    # And has fewer parameters overall.
    n_legacy = sum(p.numel() for p in om_legacy.parameters() if p.requires_grad)
    n_count = sum(p.numel() for p in om_count.parameters() if p.requires_grad)
    assert n_count < n_legacy, f"count={n_count} should be < legacy={n_legacy}"
