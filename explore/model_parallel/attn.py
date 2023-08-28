import torch
from torch import nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange


from tp_utils import get_tp_group, set_tp_group, mylinear, RowParallelLinear, ColParallelLinear, _ReduceFromModelParallelRegion

def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = mylinear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = mylinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def _naive_attn(self, x):
        B, N, DIM = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # make torchscript happy (cannot use tensor as tuple)
        q= _split_heads(q, self.num_heads, self.head_dim)
        k= _split_heads(k, self.num_heads, self.head_dim)
        v= _split_heads(v, self.num_heads, self.head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, DIM)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        x = self._naive_attn(x)
        return x


class TpAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 tp_group = None, sequence_parallel=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        set_tp_group(tp_group)
        self.tp_size = torch.distributed.get_world_size(get_tp_group())
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.head_num_per_partition = self.num_heads//self.tp_size

        self.qkv = ColParallelLinear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = RowParallelLinear(dim, dim, sequence_parallel=sequence_parallel)
        self.proj_drop = nn.Dropout(proj_drop)



    def _naive_attn(self, x):
        B, N, DIM = x.shape     # DIM=self.head_dim*self.num_heads
        qkv_out = self.qkv(x)       # B,N,3*DIM

        q, k, v = qkv_out.chunk(3, dim=-1) # B.N.DIM
        q= _split_heads(q, self.head_num_per_partition, self.head_dim)
        k= _split_heads(k, self.head_num_per_partition, self.head_dim)
        v= _split_heads(v, self.head_num_per_partition, self.head_dim)


        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, DIM//self.tp_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        x = self._naive_attn(x)
        return x