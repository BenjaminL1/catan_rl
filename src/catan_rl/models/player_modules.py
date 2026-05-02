import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class DevCardCountEncoder(nn.Module):
    """Phase 1.4: count-based encoder for dev-card multisets.

    Replaces the embedding+MHA+sum-pool pipeline with a tiny MLP over per-type
    counts. Dev cards are a multiset (order does not matter), so the MHA was
    spending capacity learning to be permutation-invariant — count encoding
    is permutation-invariant by construction and ~30k params lighter.

    Input  : (B, max_seq) int64 with 0 = pad, 1..vocab_excl_pad = card type.
    Output : (B, output_dim) float32, normalized via final LayerNorm + ReLU.

    The hidden width and output dim are the same as the legacy MHA pipeline
    so this is a drop-in replacement at the embedding-output level.
    """

    def __init__(
        self,
        vocab_excl_pad: int = 5,
        output_dim: int = 25,
        hidden_dim: int = 32,
        max_count: float = 16.0,
    ):
        super().__init__()
        self.vocab_excl_pad = vocab_excl_pad
        self.max_count = float(max_count)
        # Two-layer MLP with the same final shape (LN + ReLU) as the MHA path's
        # projection. Init with orthogonal sqrt(2) for ReLU compensation.
        self.fc1 = nn.Linear(vocab_excl_pad, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.act = nn.ReLU()
        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, padded_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-type counts (excluding the pad column), normalize, MLP."""
        if padded_ids.dim() == 1:  # defensive — caller is supposed to batch
            padded_ids = padded_ids.unsqueeze(0)
        ids = padded_ids.long().clamp(min=0, max=self.vocab_excl_pad)
        # Histogram across vocab+1 columns (column 0 is pad).
        counts = torch.zeros(
            padded_ids.shape[0],
            self.vocab_excl_pad + 1,
            dtype=torch.float32,
            device=padded_ids.device,
        )
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        # Drop the pad column, normalize.
        counts = counts[:, 1:] / self.max_count
        h = self.act(self.fc1(counts))
        h = self.fc2(h)
        return self.act(self.norm_out(h))


class CurrentPlayerModule(nn.Module):
    """Charlesworth-style current player encoder.

    Takes:
      - main_input: (B, 152) current_player_main
      - hidden_dev_cards: list[Tensor[L_i]] or (B, Lmax) ints (0 pad, 1–5 cards)
      - played_dev_cards: same, for played cards

    Uses external dev_card_embedding and MHA modules.
    """

    def __init__(
        self,
        main_input_dim,
        dev_card_embed_dim,
        dev_card_model_dim,
        proj_dev_card_dim,
        use_count_encoding: bool = False,
        dev_card_vocab_excl_pad: int = 5,
        max_dev_seq: int = 16,
    ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.main_input_dim = main_input_dim
        self.dev_card_embed_dim = dev_card_embed_dim
        self.dev_card_model_dim = dev_card_model_dim
        self.proj_dev_card_dim = proj_dev_card_dim
        self.use_count_encoding = use_count_encoding

        self.main_input_layer_1 = init_(nn.Linear(main_input_dim, 256))
        self.relu = nn.ReLU()

        # Phase 1.4: when count-encoding, dev cards go through DevCardCountEncoder
        # (one for hidden, one for played). The MHA-related buffers are still
        # constructed for parameter-count parity in the legacy path; in the
        # count path they are simply unused.
        if use_count_encoding:
            self.hidden_count_encoder = DevCardCountEncoder(
                vocab_excl_pad=dev_card_vocab_excl_pad,
                output_dim=proj_dev_card_dim,
                max_count=float(max_dev_seq),
            )
            self.played_count_encoder = DevCardCountEncoder(
                vocab_excl_pad=dev_card_vocab_excl_pad,
                output_dim=proj_dev_card_dim,
                max_count=float(max_dev_seq),
            )
        else:
            self.norm = nn.LayerNorm(dev_card_model_dim)
            self.proj_hidden_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))
            self.proj_played_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))
            self.norm_2 = nn.LayerNorm(proj_dev_card_dim)
            self.norm_3 = nn.LayerNorm(proj_dev_card_dim)

        self.norm_1 = nn.LayerNorm(256)
        self.norm_4 = nn.LayerNorm(128)
        self.final_linear_layer = init_(nn.Linear(2 * proj_dev_card_dim + 256, 128))

    def forward(
        self,
        main_input,
        hidden_dev_cards,
        played_dev_cards,
        dev_card_embedding,
        hidden_card_mha,
        played_card_mha,
    ):
        # Phase 1.4: count-encoding short-circuit.
        if self.use_count_encoding:
            return self._forward_count(main_input, hidden_dev_cards, played_dev_cards)
        # Hidden dev cards
        if isinstance(hidden_dev_cards, list):
            hidden_dev_cards_lengths = [len(hidden_cards) for hidden_cards in hidden_dev_cards]
            padded_hidden_dev_cards = pad_sequence(hidden_dev_cards, batch_first=True).long()
        else:
            hidden_dev_cards_lengths = (
                (hidden_dev_cards.shape[-1] - (hidden_dev_cards == 0).sum(dim=-1)).cpu().numpy()
            )
            hidden_dev_cards_lengths[hidden_dev_cards_lengths == 0] = 1
            hidden_dev_cards_lengths = list(hidden_dev_cards_lengths)
            padded_hidden_dev_cards = hidden_dev_cards

        hidden_dev_card_embeddings = dev_card_embedding(padded_hidden_dev_cards)
        hidden_dev_card_masks = torch.zeros(
            main_input.shape[0],
            1,
            1,
            padded_hidden_dev_cards.shape[1],
            device=self.dummy_param.device,
        )
        for b in range(main_input.shape[0]):
            hidden_dev_card_masks[b, ..., : hidden_dev_cards_lengths[b]] = 1.0

        hidden_dev_representations = self.norm(
            hidden_card_mha(
                hidden_dev_card_embeddings,
                hidden_dev_card_embeddings,
                hidden_dev_card_embeddings,
                hidden_dev_card_masks,
            )
        )
        hidden_dev_card_masks_2 = hidden_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        hidden_dev_representations[
            hidden_dev_card_masks_2.repeat(1, 1, hidden_dev_representations.shape[-1]) == 0
        ] = 0.0
        hidden_dev_out = hidden_dev_representations.sum(dim=1)
        hidden_dev_out = self.relu(self.norm_2(self.proj_hidden_dev_card(hidden_dev_out)))

        # Played dev cards
        if isinstance(played_dev_cards, list):
            played_dev_card_lengths = [len(played_cards) for played_cards in played_dev_cards]
            padded_played_dev_cards = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_dev_card_lengths = (
                (played_dev_cards.shape[-1] - (played_dev_cards == 0).sum(dim=-1)).cpu().numpy()
            )
            played_dev_card_lengths[played_dev_card_lengths == 0] = 1
            played_dev_card_lengths = list(played_dev_card_lengths)
            padded_played_dev_cards = played_dev_cards

        played_dev_card_embeddings = dev_card_embedding(padded_played_dev_cards)
        played_dev_card_masks = torch.zeros(
            main_input.shape[0],
            1,
            1,
            padded_played_dev_cards.shape[1],
            device=self.dummy_param.device,
        )
        for b in range(main_input.shape[0]):
            played_dev_card_masks[b, ..., : played_dev_card_lengths[b]] = 1.0

        played_dev_representations = self.norm(
            played_card_mha(
                played_dev_card_embeddings,
                played_dev_card_embeddings,
                played_dev_card_embeddings,
                played_dev_card_masks,
            )
        )
        played_dev_card_masks_2 = played_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        played_dev_representations[
            played_dev_card_masks_2.repeat(1, 1, played_dev_representations.shape[-1]) == 0
        ] = 0.0
        played_dev_out = played_dev_representations.sum(dim=1)
        played_dev_out = self.relu(self.norm_3(self.proj_played_dev_card(played_dev_out)))

        # Main input
        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))

        final_input = torch.cat((main_input, played_dev_out, hidden_dev_out), dim=-1)
        return self.relu(self.norm_4(self.final_linear_layer(final_input)))

    def _forward_count(self, main_input, hidden_dev_cards, played_dev_cards):
        """Phase 1.4 count-encoding path. Same output shape as MHA path."""
        # Coerce list-of-tensor inputs (legacy variable-length) to a padded
        # int tensor so the count encoder sees a consistent shape.
        if isinstance(hidden_dev_cards, list):
            hidden_padded = pad_sequence(hidden_dev_cards, batch_first=True).long()
        else:
            hidden_padded = hidden_dev_cards.long()
        if isinstance(played_dev_cards, list):
            played_padded = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_padded = played_dev_cards.long()

        hidden_dev_out = self.hidden_count_encoder(hidden_padded)
        played_dev_out = self.played_count_encoder(played_padded)

        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))
        final_input = torch.cat((main_input, played_dev_out, hidden_dev_out), dim=-1)
        return self.relu(self.norm_4(self.final_linear_layer(final_input)))


class OtherPlayersModule(nn.Module):
    """Charlesworth-style other-player encoder.

    Takes:
      - main_input: (B, 159) next_player_main (or others)
      - played_dev_cards: list/Tensor of ints (0 pad, 1–5 types)
    """

    def __init__(
        self,
        main_input_dim,
        dev_card_embed_dim,
        dev_card_model_dim,
        proj_dev_card_dim,
        use_count_encoding: bool = False,
        dev_card_vocab_excl_pad: int = 5,
        max_dev_seq: int = 16,
    ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.main_input_dim = main_input_dim
        self.dev_card_embed_dim = dev_card_embed_dim
        self.dev_card_model_dim = dev_card_model_dim
        self.proj_dev_card_dim = proj_dev_card_dim
        self.use_count_encoding = use_count_encoding

        self.main_input_layer_1 = init_(nn.Linear(main_input_dim, 256))
        self.relu = nn.ReLU()

        if use_count_encoding:
            # Phase 1.4: per-type count encoder for the opponent's played cards
            # (the only sequence the opponent module ever sees).
            self.played_count_encoder = DevCardCountEncoder(
                vocab_excl_pad=dev_card_vocab_excl_pad,
                output_dim=proj_dev_card_dim,
                max_count=float(max_dev_seq),
            )
        else:
            self.proj_played_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))
            self.norm = nn.LayerNorm(dev_card_model_dim)
            self.norm_2 = nn.LayerNorm(proj_dev_card_dim)

        self.final_linear_layer = init_(nn.Linear(proj_dev_card_dim + 256, 128))
        self.norm_1 = nn.LayerNorm(256)
        self.norm_3 = nn.LayerNorm(128)

    def forward(self, main_input, played_dev_cards, dev_card_embedding, played_card_mha):
        if self.use_count_encoding:
            return self._forward_count(main_input, played_dev_cards)
        if isinstance(played_dev_cards, list):
            played_dev_card_lengths = [len(played_cards) for played_cards in played_dev_cards]
            padded_played_dev_cards = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_dev_card_lengths = (
                (played_dev_cards.shape[-1] - (played_dev_cards == 0).sum(dim=-1)).cpu().numpy()
            )
            played_dev_card_lengths[played_dev_card_lengths == 0] = 1
            played_dev_card_lengths = list(played_dev_card_lengths)
            padded_played_dev_cards = played_dev_cards

        played_dev_card_embeddings = dev_card_embedding(padded_played_dev_cards)
        played_dev_card_masks = torch.zeros(
            main_input.shape[0],
            1,
            1,
            padded_played_dev_cards.shape[1],
            device=self.dummy_param.device,
        )
        for b in range(main_input.shape[0]):
            played_dev_card_masks[b, ..., : played_dev_card_lengths[b]] = 1.0

        played_dev_representations = self.norm(
            played_card_mha(
                played_dev_card_embeddings,
                played_dev_card_embeddings,
                played_dev_card_embeddings,
                played_dev_card_masks,
            )
        )
        played_dev_card_masks_2 = played_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        played_dev_representations[
            played_dev_card_masks_2.repeat(1, 1, played_dev_representations.shape[-1]) == 0
        ] = 0.0
        played_dev_out = played_dev_representations.sum(dim=1)
        played_dev_out = self.relu(self.norm_2(self.proj_played_dev_card(played_dev_out)))

        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))

        final_input = torch.cat((main_input, played_dev_out), dim=-1)
        return self.relu(self.norm_3(self.final_linear_layer(final_input)))

    def _forward_count(self, main_input, played_dev_cards):
        """Phase 1.4 count-encoding path for OtherPlayersModule."""
        if isinstance(played_dev_cards, list):
            played_padded = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_padded = played_dev_cards.long()
        played_dev_out = self.played_count_encoder(played_padded)
        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))
        final_input = torch.cat((main_input, played_dev_out), dim=-1)
        return self.relu(self.norm_3(self.final_linear_layer(final_input)))
