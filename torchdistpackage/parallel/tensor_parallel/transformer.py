from typing import Optional, Tuple, Union

import torch
from torch import nn as nn

from .attn import TpAttention,Attention
from .mlp import TpMlp,Mlp

from .tp_utils import set_sequence_parallel_attr, gather_from_sequence_parallel_region, maybe_split_into_sequence_parallel

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4,num_heads=8, **not_used):
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim*mlp_ratio))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class ParallelBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, num_heads=8, sequence_parallel=False):
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = TpAttention(dim, num_heads=num_heads, sequence_parallel=sequence_parallel)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = TpMlp(dim, hidden_features=int(dim*mlp_ratio), sequence_parallel=sequence_parallel)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states):
        # if self.sequence_parallel:
        #     hidden_states = maybe_split_into_sequence_parallel(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        # tp block: gather from seq-p and output seq-p
        attn_output = self.attn(
            hidden_states
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        # tp block: gather from seq-p and output seq-p
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # if self.sequence_parallel:
        #     set_sequence_parallel_attr(hidden_states)
        return hidden_states

    @torch.no_grad()
    def init_from_full(self, blk):
        self.mlp.fc2.init_weight_from_full(blk.mlp.fc2.weight)
        self.mlp.fc1.init_weight_from_full(blk.mlp.fc1.weight)
        self.attn.qkv.init_weight_from_full_attn(blk.attn.qkv.weight)
        self.attn.proj.init_weight_from_full(blk.attn.proj.weight)

        # lns
        self.ln_1.weight.copy_(blk.ln_1.weight)
        self.ln_1.bias.copy_(blk.ln_1.bias)
        self.ln_2.weight.copy_(blk.ln_2.weight)
        self.ln_2.bias.copy_(blk.ln_2.bias)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_ratio=4, num_heads=8, depth=12, tensor_parallel=True, sequence_parallel=True):
        super().__init__()
        blk_type = Block if not tensor_parallel else ParallelBlock
        self.blocks = nn.ModuleList([blk_type(dim, mlp_ratio=mlp_ratio, num_heads=num_heads, sequence_parallel=sequence_parallel) for _ in range(depth)])
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        if self.sequence_parallel:
            x = maybe_split_into_sequence_parallel(x)
        for blk in self.blocks:
            x = blk(x)
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x)
        return x