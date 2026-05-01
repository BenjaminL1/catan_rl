"""
Multi-headed attention for dev-card sequences (Charlesworth-style).

Inputs: (batch, seq_len, model_dim). Mask: True = attend, False = mask out (-inf).
"""

import math

import torch
import torch.nn as nn

from catan_rl.models.utils import get_clones


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension: int, num_heads: int) -> None:
        super().__init__()
        assert model_dimension % num_heads == 0
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.head_dimension = model_dimension // num_heads
        self.num_heads = num_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_proj_net = nn.Linear(model_dimension, model_dimension)
        self.softmax = nn.Softmax(dim=-1)

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        if mask is not None:
            # mask: True = valid, False = invalid; we fill invalid with -inf
            scores = scores.masked_fill(mask == False, float("-inf"))
        attention_weights = self.softmax(scores)
        return torch.matmul(attention_weights, value)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        query, key, value = [
            net(x).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
            for net, x in zip(self.qkv_nets, (query, key, value))
        ]
        intermediate = self.attention(query, key, value, mask)
        intermediate = intermediate.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dimension
        )
        return self.out_proj_net(intermediate)
